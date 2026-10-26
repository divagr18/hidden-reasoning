"""P3: tracking circuits compiled into extra Gated DeltaNet chip layers with the q35 compiler.

The circuits read ordinary Qwen3.5 text tokens of the task prompts (no chip tokens, nothing
written to the transcript) and keep the tracked state in DeltaNet state. Two layers each:

  layer 1  "prev" head (conv taps copy the token features of t-1, t-2, t-3 into the residual)
           + MLP ANDs that recognise each step's last token and its arguments
  layer 2  tracker heads

T1 (last write wins) uses stock heads: beta = 1, one-hot keys, an exact row overwrite.
T2-T4 need reflections, which a stock layer cannot make (beta = sigmoid <= 1):

  T2  key e0, v = 0, beta = 2:          row0 <- -row0           (a sign flip per toggle)
  T3  key (e_i + e_j)/sqrt2, v = 0, b=2: row_i <- -row_j, row_j <- -row_i (signed swap)
  T4  the same on 5 rows (boxes) holding a one-hot item each, one head per box queried

beta = 2 comes from one flag (``BETA2`` below): the chip layers compute beta = 2 sigmoid(b).
Qwen's own layers and the stock class are otherwise untouched; with the flag the checkpoint no
longer loads as a stock Qwen3_5 model, which is fine for a demonstration.

Fresh rows: with beta = 2 the first write into an empty row stores 2v, so initial-state writes
carry v/2 (powers of 2, exact).

    python -m hcot.p3 --task T2 --fmt sym --tokens 100000
"""

import argparse
import random
import time

import numpy as np
import torch

from hcot.compiler.graph import Expr
from hcot.compiler import ir
from hcot.compiler.emit import embedding_rows, emit_layers, check_bf16
from hcot.compiler.ir import A_BIT, A_REF, HD, K, REF_COLS, Chan, Target
from hcot.compiler.prims import Circ
from hcot.compiler.tiny import BASE, pick_layout

from hcot import tasks

TARGET = Target(key_heads=16, value_heads=32, intermediate=1024)  # Qwen3.5-4B head shapes

# ------------------------------------------------------------------------------------------
# beta range flag


class BETA2:
    """Layers listed in ``model.config.chip_beta2_layers`` compute beta = 2 * sigmoid(b)."""

    active = False
    installed = False

    @classmethod
    def install(cls, model):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

        if not cls.installed:
            for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
                orig = getattr(mq, name)

                def wrap(*a, _orig=orig, **kw):
                    if cls.active:
                        if "beta" in kw:
                            kw["beta"] = kw["beta"] * 2.0
                        else:
                            a = list(a)
                            a[4] = a[4] * 2.0
                    return _orig(*a, **kw)

                setattr(mq, name, wrap)
            cls.installed = True
        for L in model.config.chip_beta2_layers:
            m = model.model.layers[L].linear_attn
            m.register_forward_pre_hook(lambda *_: setattr(cls, "active", True))
            m.register_forward_hook(lambda *_: setattr(cls, "active", False))


# ------------------------------------------------------------------------------------------
# IR: reflection heads (evaluated by our own exact simulator; the stock IR models only
# overwrites)

REFL = set()  # names of heads that run with beta = 2
_orig_eval_head = ir.eval_head


def eval_head(node, m, chip, check):
    if node.name not in REFL:
        return _orig_eval_head(node, m, chip, check)
    n = m.shape[0]
    q = {r: ir.conv(c, m, check) for r, c in node.q.items()}
    k = {r: ir.conv(c, m, check) for r, c in node.k.items()}
    v = np.zeros((n, node.n_cols))
    for col, c in node.v.items():
        v[:, col] = ir.conv(c, m, check)
    rows = sorted(set(q) | set(k))
    S = np.zeros((len(rows), node.n_cols))
    ri = {r: i for i, r in enumerate(rows)}
    out = np.zeros((n, node.n_cols))
    for t in range(n):
        kv = np.zeros(len(rows))
        for r, x in k.items():
            kv[ri[r]] = x[t]
        if kv.any():
            kv /= np.linalg.norm(kv)
            S += 2.0 * np.outer(kv, v[t] - kv @ S)  # beta = 2
        qv = np.zeros(len(rows))
        for r, x in q.items():
            qv[ri[r]] = x[t]
        if qv.any():
            o = (qv > 0) @ S
            ref = (o[list(REF_COLS)] ** 2).sum()
            if ref > 0:
                out[t] = o / np.sqrt(ref / node.n_cols)
    for col, d, w in node.out:
        zz = ir.ev(node.z.get(col, Expr()), m)
        # the hardware adds bf16 head outputs to a bf16 residual
        m[:, d] = ir.bf16(out[:, col] * np.maximum(zz, 0.0) * w)


ir.eval_head = eval_head


# ------------------------------------------------------------------------------------------
# circuits


class Tracker(Circ):
    def __init__(self, tok, strings):
        super().__init__(TARGET)
        self.tok = tok
        self.s = strings  # token id -> decoded string
        self.feat = {}

    def f(self, name, pred):
        if name not in self.feat:
            self.feat[name] = self.token_feature(name, lambda t: int(pred(self.s.get(t, ""))))
        return self.feat[name]

    def is_(self, *texts):
        return self.f("is" + "|".join(texts), lambda s: s in texts)

    def prev(self, feats: dict):
        """{(name, lag): Expr feature} -> {(name, lag): Expr holding feature(t - lag)}."""
        cols, keys = {}, list(feats)
        for i, key in enumerate(keys):
            lag = key[1]
            taps = tuple(1.0 if j == lag else 0.0 for j in range(4))
            cols[2 + i] = Chan(feats[key], taps)
        outs = self.slot_head("prev", {0: self.one}, {0: self.one}, cols, self.one)
        return dict(zip(keys, outs))

    def refl_head(self, name, keys, query_rows, init, data, outs):
        """A beta = 2 head. keys: row -> 0/1 Expr (two rows set = a reflection, one row = a
        write or a sign flip); init: 0/1 Expr, the tokens that write a fresh row (ballast plus
        ``data`` bits, stored as v/2); every other keyed token has v = 0. query_rows: rows read
        (summed) every token; outs: [(col, weight)]."""
        sel = lambda c: Chan(c * (2 * K) - self.one * K)  # noqa: E731
        kk = {r: sel(c) for r, c in keys.items()}
        q = {r: Chan(self.one * K) for r in query_rows}
        v = {col: Chan(init * (A_REF // 2)) for col in REF_COLS}
        for col, bit in data.items():  # bits must be 0 wherever init is 0 and a key fires
            v[col] = Chan(bit * (A_BIT // 2))
        z = {col: self.one * K for col, _ in outs}
        REFL.add(name)
        return self.head(name, q, kk, v, Expr(), z, outs)


def build_circuit(task, fmt, tok, strings):
    c = Tracker(tok, strings)
    one, digit = c.one, [c.is_(str(d)) for d in range(10)]
    isdig = sum(digit, Expr())
    readout = {}  # name -> list of Exprs (for the verifier)
    if task == "T1":
        var = {v: c.is_(" " + v) for v in tasks.VARS}
        eq, sp, eqs, to = c.is_("="), c.is_(" "), c.is_(" ="), c.is_(" to")
        p = c.prev({**{(f"v{v}", L): var[v] for v in tasks.VARS for L in (2, 3)},
                    ("eq", 1): eq, ("sp", 1): sp, ("eqs", 2): eqs, ("to", 2): to})
        for v in tasks.VARS:
            if fmt == "sym":
                w = c.and_([isdig, p[("eq", 1)], p[(f"v{v}", 2)]], f"w.{v}")
            else:
                w = c.node(c.t_and([isdig, p[("sp", 1)], p[("eqs", 2)], p[(f"v{v}", 3)]])
                           + c.t_and([isdig, p[("sp", 1)], p[("to", 2)], p[(f"v{v}", 3)]]), f"w.{v}")
            # stock head: beta = 1, one-hot key, the row is overwritten with the digit's one-hot
            readout[v] = c.slot_head(f"var.{v}", {0: w}, {0: one},
                                     {2 + d: digit[d] for d in range(10)}, w)
    elif task == "T2":
        if fmt == "sym":
            start, tog = c.is_("Bit"), c.is_(" T")
            p = c.prev({("sp", 1): c.is_(" "), ("at", 2): c.is_(" at"), ("starts", 3): c.is_(" starts")})
            init1 = c.and_([digit[1], p[("sp", 1)], p[("at", 2)], p[("starts", 3)]], "init1")
            flip = c.node(c.t_and([tog]) + c.t_and([init1]), "flip")
        else:
            p = c.prev({("A", 1): c.is_("A"), ("is", 1): c.is_(" is"), ("light", 2): c.is_(" light"),
                        ("either", 1): c.is_(" either")})
            start = c.and_([c.is_(" light"), p[("A", 1)]], "start")
            init1 = c.and_([c.is_(" on"), p[("is", 1)], p[("light", 2)]], "init1")
            fl = c.and_([c.is_(" flips"), one - p[("either", 1)]], "toggle")
            flip = c.node(c.t_and([fl]) + c.t_and([init1]), "flip")
        readout["bit"] = c.refl_head("refl.switch", {0: start + flip}, [0], start, {},
                                     [(REF_COLS[0], 2.0**-10)])
    elif task == "T3":
        # Every row needs a ballast: a reflection leaves ~1e-16 of a swapped-out row behind,
        # and the head's output RMSNorm would rescale that to full size in an empty row. So the
        # rows hold the cups' *labels* (a permutation of A, B, C, all ballasted, as in T4), a
        # stock head remembers which label the ball started under, and the MLP combines them.
        cup = {x: c.is_(" " + x) for x in tasks.CUPS}
        anycup = sum(cup.values(), Expr())
        if fmt == "sym":
            pair = {x: c.is_(*[" " + a + b for a in tasks.CUPS for b in tasks.CUPS if a != b and x in a + b])
                    for x in tasks.CUPS}
            p = c.prev({("under", 1): c.is_(" under"), ("Ball", 1): c.is_("Ball"), ("Ball", 2): c.is_("Ball")})
            ball0 = c.and_([anycup, p[("under", 1)], p[("Ball", 2)]], "ball0")
            lab = {"A": c.is_("Ball"), "B": c.and_([c.is_(" under"), p[("Ball", 1)]], "lab.B"), "C": ball0}
        else:
            p = c.prev({**{(x, 2): cup[x] for x in tasks.CUPS}, ("anycup", 2): anycup,
                        ("and", 1): c.is_(" and"), ("cups", 3): c.is_(" cups"), ("cups", 2): c.is_(" cups"),
                        ("comma", 1): c.is_(","), ("comma", 3): c.is_(","), ("cup", 1): c.is_(" cup"), ("under", 2): c.is_(" under")})
            pair = {x: c.node(c.t_and([cup[x], p[("and", 1)], p[("anycup", 2)], p[("cups", 3)]])
                              + c.t_and([anycup, p[("and", 1)], p[(x, 2)], p[("cups", 3)]]), f"swap.{x}")
                    for x in tasks.CUPS}
            ball0 = c.and_([anycup, p[("cup", 1)], p[("under", 2)]], "ball0")
            # "Three cups, A, B and C," labels each row once
            lab = {"A": c.and_([cup["A"], p[("comma", 1)], p[("cups", 2)]], "lab.A"),
                   "B": c.and_([cup["B"], p[("comma", 1)], p[("A", 2)]], "lab.B"),
                   "C": c.and_([cup["C"], p[("and", 1)], p[("B", 2)], p[("comma", 3)]], "lab.C")}
        keys = {i: pair[x] + lab[x] for i, x in enumerate(tasks.CUPS)}
        anylab = sum(lab.values(), Expr())
        labels = [c.refl_head(f"refl.cup{x}", keys, [i], anylab,
                              {2 + j: lab[y] for j, y in enumerate(tasks.CUPS)},
                              [(2 + j, 2.0**7) for j in range(3)]) for i, x in enumerate(tasks.CUPS)]
        start = c.slot_head("ball0", {0: ball0}, {0: one}, {2 + j: cup[y] for j, y in enumerate(tasks.CUPS)},
                            ball0)
        # labels are +-1 (reflections flip signs) up to a ~1e-14 leak in swapped-out entries:
        # [|x| = 1 and ball0 = 1] as two threshold ANDs (gate 4(+-x + s) - 6), exact for leaks < 1/4
        readout["cup"] = [c.node(sum((c.t_and([x, start[j]]) + c.t_and([-x, start[j]])
                                      for j, x in enumerate(labels[i])), []), f"ball.{X}")
                          for i, X in enumerate(tasks.CUPS)]
    elif task == "T4":
        items = tasks.SYM_ITEMS if fmt == "sym" else tasks.ITEMS
        item = [c.is_(x if fmt == "sym" else " " + x) for x in items]
        if fmt == "sym":
            p = c.prev({**{(f"d{d}", 1): digit[d] for d in range(1, 6)}, ("sp", 2): c.is_(" "),
                        ("dig", 1): isdig})
            init = {b: c.and_([sum(item, Expr()), p[(f"d{b + 1}", 1)], p[("sp", 2)]], f"init.{b}")
                    for b in range(5)}
            # a swap ends at its second digit: digit(t), digit(t-1), " "(t-2)
            sw = {b: c.node(c.t_and([digit[b + 1], p[("dig", 1)], p[("sp", 2)]])
                            + c.t_and([isdig, p[(f"d{b + 1}", 1)], p[("sp", 2)]]), f"swap.{b}")
                  for b in range(5)}
        else:
            p = c.prev({**{(f"d{d}", L): digit[d] for d in range(1, 6) for L in (2, 3)},
                        ("dig", 3): isdig, ("the", 1): c.is_(" the"), ("sp", 1): c.is_(" "),
                        ("and", 2): c.is_(" and"), ("box", 3): c.is_(" box")})
            init = {b: c.and_([sum(item, Expr()), p[("the", 1)], p[(f"d{b + 1}", 2)]], f"init.{b}")
                    for b in range(5)}
            sw = {b: c.node(c.t_and([digit[b + 1], p[("sp", 1)], p[("and", 2)], p[("dig", 3)]])
                            + c.t_and([isdig, p[("sp", 1)], p[("and", 2)], p[(f"d{b + 1}", 3)]]),
                            f"swap.{b}") for b in range(5)}
        keys = {b: sw[b] + init[b] for b in range(5)}
        anyinit = sum(init.values(), Expr())
        readout["box"] = [c.refl_head(f"refl.box{b}", keys, [b], anyinit,
                                      {2 + i: item[i] for i in range(5)},
                                      [(2 + i, 2.0**7) for i in range(5)]) for b in range(5)]
    return c, readout


# ------------------------------------------------------------------------------------------
# streams and reference


def stream(task, fmt, tok, n_tokens, seed=0):
    """(token ids, [(position of each step's last token, reference state)])."""
    rng = random.Random(f"p3-{task}-{fmt}-{seed}")
    per = {"T1": 6, "T2": 6, "T3": 6, "T4": 8} if fmt == "nl" else {"T1": 3, "T2": 1, "T3": 1, "T4": 3}
    n = n_tokens // per[task] + 8
    lat = tasks.SAMPLE[task](rng, n)
    r = tasks.render(task, lat, fmt)
    ids = tok(r["prefix"], add_special_tokens=False)["input_ids"]
    marks = []
    step_text = r["steps"]
    lines = step_text.splitlines(keepends=True) if fmt == "nl" else [" " + x for x in step_text.split()]
    assert len(lines) == len(lat["steps"])
    import copy

    state = copy.deepcopy(lat["init"])
    for x, line in zip(lat["steps"], lines):
        ids += tok(line, add_special_tokens=False)["input_ids"]
        state = step_state(task, state, x)
        marks.append((len(ids) - 1, copy.deepcopy(state)))
        if len(ids) >= n_tokens:
            break
    return ids[:n_tokens], [m for m in marks if m[0] < n_tokens], r["prefix"], lat


def step_state(task, s, x):
    if task == "T1":
        s = dict(s)
        s[x[0]] = x[1]
    elif task == "T2":
        s ^= int(x)
    elif task == "T3":
        i, j = x
        s = j if s == i else i if s == j else s
    else:
        s = list(s)
        i, j = x
        s[i], s[j] = s[j], s[i]
    return s


def decode(task, fmt, h, readout, lay):
    """Tracked state read from the residual h (positions x dims) -> list per position."""
    col = lambda e: h[:, lay.dim(next(iter(e.terms)))]  # noqa: E731
    if task == "T1":
        vals = {v: np.stack([col(e) for e in readout[v]], 1) for v in tasks.VARS}
        return vals
    if task == "T2":
        return col(readout["bit"][0])
    if task == "T3":
        return np.stack([col(e) for e in readout["cup"]], 1)
    return np.stack([np.stack([col(e) for e in readout["box"][b]], 1) for b in range(5)], 1)


def check_marks(task, dec, marks, fmt, init_bit=None):
    """Count positions where the decoded state differs from the reference."""
    bad, worst = 0, 0.0
    for pos, ref in marks:
        if task == "T1":
            for vi, v in enumerate(tasks.VARS):
                row = dec[v][pos]
                ok = np.argmax(row) == ref[v] and abs(row[ref[v]] - 1) < 0.25 and \
                    np.abs(np.delete(row, ref[v])).max() < 0.25
                worst = max(worst, abs(row[ref[v]] - 1))
                bad += not ok
        elif task == "T2":
            x = dec[pos]  # +1: off/0, -1: on/1
            ok = (x < 0) == bool(ref) and abs(abs(x) - 1) < 0.25
            worst = max(worst, abs(abs(x) - 1))
            bad += not ok
        elif task == "T3":
            row = np.abs(dec[pos])
            ok = np.argmax(row) == ref and abs(row[ref] - 1) < 0.25 and np.delete(row, ref).max() < 0.25
            worst = max(worst, abs(row[ref] - 1), np.delete(row, ref).max())
            bad += not ok
        else:
            m = np.abs(dec[pos])  # boxes x items
            for b in range(5):
                ok = np.argmax(m[b]) == ref[b] and abs(m[b, ref[b]] - 1) < 0.25 and \
                    np.delete(m[b], ref[b]).max() < 0.25
                worst = max(worst, abs(m[b, ref[b]] - 1), np.delete(m[b], ref[b]).max())
                bad += not ok
    return bad, worst


# ------------------------------------------------------------------------------------------
# model


def build_model(c, vocab, dtype=torch.bfloat16, beta2=True):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    lay = pick_layout(len(c.dims))
    n_chip = c.n_layers()
    n_vocab = -(-(len(vocab)) // 64) * 64
    tc = dict(BASE, hidden_size=lay.hidden, vocab_size=n_vocab, num_hidden_layers=n_chip + 1,
              layer_types=["linear_attention"] * n_chip + ["full_attention"],
              linear_num_key_heads=TARGET.key_heads, linear_num_value_heads=TARGET.value_heads,
              intermediate_size=TARGET.intermediate)
    sd = emit_layers(c, lay, tc, 0, "model.")
    emb = torch.zeros(n_vocab, lay.hidden)
    emb[: len(vocab)] = embedding_rows(c, lay, vocab, lay.hidden)
    sd["model.embed_tokens.weight"] = emb
    sd["model.norm.weight"] = torch.full((lay.hidden,), -1.0)
    cfg = Qwen3_5TextConfig(**tc)
    model = Qwen3_5ForCausalLM(cfg)
    for p in model.parameters():
        p.data.zero_()
    model.load_state_dict(sd, strict=False)
    model.tie_weights()
    # the flag: chip layers holding reflection heads run with beta in (0, 2]
    refl_layers = sorted({(n.level - 1) // 2 for n in c.nodes if getattr(n, "name", "") in REFL})
    model.config.chip_beta2_layers = refl_layers if beta2 else []
    model = model.to(dtype).eval()
    if refl_layers and beta2:
        BETA2.install(model)
    return model, lay


def run(task, fmt, n_tokens, decode_tokens=0, beta_override=None, seed=0, chunk=4096, device="cpu"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    ids, marks, prefix, lat = stream(task, fmt, tok, n_tokens, seed)
    # the stream is ordinary text: check our per-step tokenization equals the tokenizer's own
    head = tok.decode(ids[:400])
    assert tok(head, add_special_tokens=False)["input_ids"][: 300] == ids[:300]
    vocab = sorted(set(ids))
    strings = {t: tok.decode([t]) for t in vocab}
    REFL.clear()
    c, readout = build_circuit(task, fmt, tok, strings)
    remap = {t: i for i, t in enumerate(vocab)}
    x = [remap[t] for t in ids]
    # exactness of the circuit in the IR (bf16 rounding, silu ranges) on a prefix
    m = ir.evaluate(c, vocab and [vocab[i] for i in x[:3000]], check=True)
    # ablation (beta_override): the same weights with stock beta = sigmoid(b) <= 1 everywhere
    model, lay = build_model(c, vocab, beta2=beta_override is None)
    model = model.to(device)
    t0 = time.time()
    hs, cache = [], None
    with torch.no_grad():  # chunked prefill: each chunk starts from the previous chunk's states
        for i in range(0, len(x), chunk):
            out = model(input_ids=torch.tensor([x[i:i + chunk]], device=device), past_key_values=cache, use_cache=True,
                        output_hidden_states=True)
            cache = out.past_key_values
            hs.append(out.hidden_states[c.n_layers()][0].float().cpu().numpy())
    h = np.concatenate(hs)
    dt = time.time() - t0
    dec = decode(task, fmt, h, readout, lay)
    ir_dec = decode(task, fmt, m, readout, type("L", (), {"dim": staticmethod(lambda d: d)})())
    ir_bad, _ = check_marks(task, ir_dec, [mk for mk in marks if mk[0] < 3000], fmt)
    bad, worst = check_marks(task, dec, marks, fmt)
    res = dict(task=task, fmt=fmt, tokens=len(ids), steps=len(marks), layers=c.n_layers(),
               hidden=lay.hidden, heads=sum(isinstance(n, ir.Head) for n in c.nodes),
               refl_heads=len(REFL), ir_bad=ir_bad, prefill_bad=bad, prefill_worst_dev=worst,
               prefill_s=round(dt, 1), beta2_layers=model.config.chip_beta2_layers)
    if decode_tokens:
        # token-by-token (the recurrent kernel), on the first decode_tokens tokens
        n = min(decode_tokens, len(x))
        hs = []
        with torch.no_grad():
            o = model(input_ids=torch.tensor([x[:1]], device=device), use_cache=True, output_hidden_states=True)
            cache = o.past_key_values
            hs.append(o.hidden_states[c.n_layers()][0])
            for t in range(1, n):
                o = model(input_ids=torch.tensor([x[t:t + 1]], device=device), past_key_values=cache, use_cache=True,
                          output_hidden_states=True)
                hs.append(o.hidden_states[c.n_layers()][0])
        hd = torch.cat(hs).float().cpu().numpy()
        decd = decode(task, fmt, hd, readout, lay)
        b2, w2 = check_marks(task, decd, [mk for mk in marks if mk[0] < n], fmt)
        res.update(decode_tokens=n, decode_bad=b2, decode_worst_dev=w2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="*", default=tasks.TASKS)
    ap.add_argument("--fmt", nargs="*", default=tasks.FORMATS)
    ap.add_argument("--tokens", type=int, default=100000)
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--stock-beta", action="store_true", help="ablation: beta <= 1 on all layers")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    for t in a.task:
        for f in a.fmt:
            print(run(t, f, a.tokens, a.decode, beta_override=True if a.stock_beta else None, device=a.device), flush=True)


if __name__ == "__main__":
    main()
