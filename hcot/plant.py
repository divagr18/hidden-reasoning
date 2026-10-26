"""Planted positive control: P3's compiled beta = 2 parity tracker (T2 sym) inside a real
Qwen3.5-4B, built with the q35 merge machinery (the chip-first layout of q35/merge.py and
rv/merge.py), then the P2 detector run blind over every DeltaNet head.

Layout of the planted model (a text-only Qwen3_5ForCausalLM):

- hidden widened from 2560 to ``pick_hidden``; Qwen's weights are zero-padded (q35.merge rules:
  Qwen never reads or writes chip dims; the full-residual norms are rescaled by sqrt(D / D')),
- 3 chip layers first (the T2 circuit of hcot/p3.py plus an answer gate), then Qwen's 32 layers
  unchanged (24 DeltaNet + 8 full attention),
- every Qwen vocabulary row carries the ballast and its token features in the chip dims; the last
  chip sublayer clears all of it except one kept dim ``ans`` = kappa * (+1 bit 0 / -1 bit 1) at the
  answer positions (":" after "Answer", and " " after "Answer:") and 0 everywhere else,
- the lm_head (untied, otherwise Qwen's embedding) reads ``ans`` with +lam on "0" and -lam on
  "1": the tracker's readout reaches the answer logits through the residual stream,
- beta = 2 sigmoid(b) only in the chip layer that holds the reflection head (p3.BETA2's wrapper).

    python -m hcot.plant selftest            # CPU: IR check + a random tiny Qwen as the host
    python -m hcot.plant all --out results/plant   # GPU box: real Qwen3.5-4B
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

from hcot import p3, tasks

SRC = "Qwen/Qwen3.5-4B"
KAPPA = 2.0**-8  # residual value of the readout at an answer position (Qwen's norms barely see it)
ANS_TOKENS = ("0", "1")  # bit 0 -> "0", bit 1 -> "1"


# ------------------------------------------------------------------------------------------
# circuit


def circuit(tok, strings):
    """P3's T2 sym tracker + the answer gate and the exit (the only dim kept for Qwen's layers)."""
    p3.REFL.clear()
    c, readout = p3.build_circuit("T2", "sym", tok, strings)
    bit = readout["bit"][0]  # +1: bit 0, -1: bit 1, on every token after "Bit"; 0 before
    p = c.prev({("colon", 1): c.is_(":"), ("Answer", 1): c.is_("Answer"), ("Answer", 2): c.is_("Answer")})
    gate = c.node(c.t_and([c.is_(":"), p[("Answer", 1)]])
                  + c.t_and([c.is_(" "), p[("colon", 1)], p[("Answer", 2)]]), "gate")
    ans = c.ffn(c.t_and([gate], coef=KAPPA, a=bit), "exit.ans")
    c.keep = [next(iter(ans.terms))]
    return c, dict(bit=bit, gate=gate, ans=ans)


def all_strings(tok, n):
    out = {}
    for i in range(n):
        try:
            out[i] = tok.decode([i])
        except Exception:  # noqa: BLE001 - ids past the tokenizer
            out[i] = ""
    return out


# ------------------------------------------------------------------------------------------
# weights


def load_text(src=SRC):
    """(text config dict, state dict with "model.*" keys) of a Qwen3.5 checkpoint's text model."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = Path(snapshot_download(src, allow_patterns=["*.json", "*.safetensors"]))
    cfg = json.loads((path / "config.json").read_text())
    sd = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                if k.startswith("model.language_model."):
                    sd["model." + k[len("model.language_model."):]] = fh.get_tensor(k)
    return cfg["text_config"], sd


def instantiate(tc, sd, dtype=torch.bfloat16):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    tc = {k: v for k, v in tc.items() if k not in ("architectures", "model_type", "full_attention_interval")}
    config = Qwen3_5TextConfig(**tc)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if config.tie_word_embeddings:
        missing = [k for k in missing if k != "lm_head.weight"]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    if config.tie_word_embeddings:
        model.tie_weights()
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=config)
    return model.to(dtype).eval()


def stock(tc0, sd0, device, dtype=torch.bfloat16):
    model = instantiate(dict(tc0, tie_word_embeddings=True), dict(sd0), dtype).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build(tc0, sd0, tok, lam, device="cpu", beta2=True, fp32_norms=True):
    """The planted model. sd0 is not modified."""
    from hcot.compiler.emit import Layout, check_bf16, emit_layers
    from hcot.compiler.merge import IN_COLS, OUT_ROWS, pick_hidden

    d0, v0, n0 = tc0["hidden_size"], tc0["vocab_size"], tc0["num_hidden_layers"]
    strings = all_strings(tok, v0)
    c, nd = circuit(tok, strings)
    keep = set(c.keep)
    hidden, m = pick_hidden(d0, len(c.dims))
    lay = Layout(hidden=hidden, off=d0, m=m)
    assert d0 + m + len(c.dims) <= hidden
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4
    n_chip = c.n_layers()
    ans_dim = lay.dim(next(iter(nd["ans"].terms)))

    scale = (d0 / hidden) ** 0.5
    sd, fp32 = {}, {}
    for k, w in sd0.items():
        if k == "model.embed_tokens.weight":
            continue
        if IN_COLS.search(k):
            w = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d0)], -1)
        elif OUT_ROWS.search(k):
            w = torch.cat([w, w.new_zeros(hidden - d0, *w.shape[1:])], 0)
        elif re.search(r"(input_layernorm|post_attention_layernorm)\.weight$|^model\.norm\.weight$", k):
            g = (1.0 + w.float()) * scale - 1.0
            w = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d0)])
            g32 = torch.cat([g, g.new_zeros(hidden - d0)])
        k = re.sub(r"^(model\.layers\.)(\d+)\.", lambda mt: f"{mt[1]}{int(mt[2]) + n_chip}.", k)
        sd[k] = w
        if fp32_norms and w.ndim == 1 and re.search(r"layernorm\.weight$|^model\.norm\.weight$", k):
            fp32[k] = g32
    tc = {k: v for k, v in tc0.items() if k != "full_attention_interval"}
    tc.update(hidden_size=hidden, num_hidden_layers=n_chip + n0, mtp_num_hidden_layers=0,
              layer_types=["linear_attention"] * n_chip + list(tc0["layer_types"]),
              tie_word_embeddings=False)
    for k, w in emit_layers(c, lay, tc, 0, "model.", None, keep).items():
        sd[k] = w.to(torch.bfloat16)

    E = sd0["model.embed_tokens.weight"]
    V = E.shape[0]
    emb = torch.zeros(V, hidden, dtype=torch.bfloat16)
    emb[:, :d0] = E
    emb[:, lay.off: lay.off + lay.m] = lay.ballast  # every token is a "chip" token for the circuit
    for d, fn in c.token_fns.items():
        col = torch.tensor([float(fn(t)) for t in range(V)])
        emb[:, lay.dim(d)] = col.to(torch.bfloat16)
    check_bf16("embedding", emb[:, d0:].float())
    sd["model.embed_tokens.weight"] = emb
    head = torch.zeros(V, hidden, dtype=torch.bfloat16)
    head[:, :d0] = E
    ids = [tok.convert_tokens_to_ids(t) for t in ANS_TOKENS]
    assert [tok.decode([i]) for i in ids] == list(ANS_TOKENS)
    head[ids[0], ans_dim] = lam
    head[ids[1], ans_dim] = -lam
    sd["lm_head.weight"] = head
    norm = sd["model.norm.weight"].float()
    norm[d0:] = -1.0  # chip dims are invisible to the lm_head ...
    norm[ans_dim] = 0.0  # ... except the readout (gain 1)
    sd["model.norm.weight"] = norm.to(torch.bfloat16)
    if fp32_norms:
        fp32["model.norm.weight"][d0:] = norm[d0:]

    model = instantiate(tc, sd).to(device)
    # The rescaled gains (1 + w) * sqrt(D / D') are not bf16 values: kept in fp32 (HF's RMSNorm
    # computes in fp32 anyway), Qwen's normalized inputs match the stock model to fp32 rounding.
    own = dict(model.named_parameters())
    for k, g in fp32.items():
        own[k].data = g.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    refl = sorted({(n.level - 1) // 2 for n in c.nodes if getattr(n, "name", "") in p3.REFL})
    model.config.chip_beta2_layers = []
    p3.BETA2.install(model)  # installs the kernel wrappers only (no layers listed)
    model.config.chip_beta2_layers = refl
    model.chip = dict(n_chip=n_chip, hidden=hidden, ballast_dims=m, ans_dim=ans_dim, refl_layers=refl,
                      planted=[(L, 0) for L in refl], circuit=c.stats(), lam=lam, kappa=KAPPA,
                      fp32_norm_gains=len(fp32),
                      heads={n.name: (n.level - 1) // 2 for n in c.nodes if hasattr(n, "q")})
    model._beta_hooks = []
    set_beta2(model, beta2)
    return model, c, nd, lay


def set_beta2(model, on: bool):
    for h in model._beta_hooks:
        h.remove()
    model._beta_hooks = []
    if not on:
        return
    for L in model.config.chip_beta2_layers:
        m = model.model.layers[L].linear_attn
        model._beta_hooks += [m.register_forward_pre_hook(lambda *_: setattr(p3.BETA2, "active", True)),
                              m.register_forward_hook(lambda *_: setattr(p3.BETA2, "active", False))]


# ------------------------------------------------------------------------------------------
# checks


def ir_check(tok, steps=(16, 64), n=10):
    """The circuit in the q35 IR (bf16 exactness asserted), on the templated T2 prompts: the kept
    dim is kappa * (+-1) at the answer positions with the right sign, 0 elsewhere."""
    from hcot.compiler import ir
    from hcot.models import chat

    ids_all = []
    for k in steps:
        for it in tasks.make("T2", k, "sym", n, seed=5):
            ids = tok(chat(tok, "q35", tasks.user_text(it), think=False) + " ", add_special_tokens=False)["input_ids"]
            ids_all.append((ids, int(it["answer"])))
    strings = all_strings(tok, max(max(i) for i, _ in ids_all) + 1)
    c, nd = circuit(tok, strings)
    ans = next(iter(nd["ans"].terms))
    bad = 0
    for ids, want in ids_all:
        mm = ir.evaluate(c, ids, check=True)
        a = mm[:, ans]
        nz = list((a != 0).nonzero()[0])
        ok = nz == [len(ids) - 2, len(ids) - 1] and all((a[t] < 0) == bool(want) for t in nz) \
            and all(abs(a[t]) == KAPPA for t in nz)
        bad += not ok
    return dict(prompts=len(ids_all), bad=bad, layers=c.n_layers(), stats=c.stats())


def items_for(k, n, seed=0):
    return tasks.make("T2", k, "sym", n, seed=seed)


@torch.no_grad()
def answers(model, tok, lengths, n, max_new=8, seed=0, log=print):
    """Greedy answer-only accuracy (the P1 / fine-tuning eval: thinking off, reply prefilled with
    "Answer:", greedy up to 8 tokens, parsed with behave.parse_answer), batch 1."""
    from hcot.behave import parse_answer
    from hcot.models import chat

    dev = next(model.parameters()).device
    ids01 = [tok.convert_tokens_to_ids(t) for t in ANS_TOKENS]
    res = {}
    for k in lengths:
        rows = []
        t0 = time.time()
        for it in items_for(k, n, seed):
            text = chat(tok, "q35", tasks.user_text(it), think=False)
            x = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=dev)
            g = model.generate(input_ids=x, max_new_tokens=max_new, do_sample=False, output_scores=True,
                               return_dict_in_generate=True, temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.eos_token_id)
            gen = g.sequences[0, x.shape[1]:].tolist()
            eos = [j for j, t in enumerate(gen) if t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))]
            out = tok.decode(gen[: eos[0] if eos else len(gen)])
            pred, leaked = parse_answer(out, "answer")
            s0 = g.scores[0][0].float()
            want = int(it["answer"])
            margin = float(s0[ids01[want]] - s0[ids01[1 - want]])
            rows.append(dict(correct=pred == it["answer"], leaked=leaked, out=out, margin=margin,
                             first=tok.decode(gen[:1])))
        acc = sum(r["correct"] for r in rows) / len(rows)
        res[k] = dict(acc=acc, acc_first_token=sum(r["first"] == it_ans for r, it_ans in
                                                  zip(rows, [x["answer"] for x in items_for(k, n, seed)])) / len(rows),
                      n=len(rows), leaks=sum(r["leaked"] for r in rows),
                      outputs=sorted({r["out"] for r in rows}),
                      margin_min=min(r["margin"] for r in rows),
                      margin_mean=sum(r["margin"] for r in rows) / len(rows))
        log(f"answers {k} steps: acc {acc:.3f} n {len(rows)} leaks {res[k]['leaks']} first-token acc {res[k]['acc_first_token']:.3f} outs {ascii(res[k]['outputs'][:6])} "
            f"margin min {res[k]['margin_min']:.1f} mean {res[k]['margin_mean']:.1f} {time.time() - t0:.0f}s")
    return res


TEXT_FILES = ["README.md", "docs/q35.md", "research/hybrid-cot/PLAN.md"]
CHAT_PROMPTS = [
    "Explain photosynthesis in two sentences.",
    "Write a haiku about autumn leaves.",
    "What is the capital of Australia, and why isn't it Sydney?",
    "Give me three tips for writing clear emails.",
    "Summarize the plot of Hamlet in three sentences.",
    "Translate 'Where is the train station?' into French and German.",
]


def text_inputs(tok, texts, chunk=1024):
    out = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        out += [ids[i:i + chunk] for i in range(0, len(ids), chunk) if len(ids[i:i + chunk]) > 16]
    return out


@torch.no_grad()
def logprobs(model, chunks):
    dev = next(model.parameters()).device
    return [torch.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0].float(), -1).to(torch.bfloat16).cpu()
            for c in chunks]


@torch.no_grad()
def greedy(model, tok, prompts, max_new=48):
    from hcot.models import chat

    dev = next(model.parameters()).device
    outs = []
    for p in prompts:
        text = chat(tok, "q35", p, think=False)[: -len("Answer:")]  # a normal reply, no prefill
        x = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=dev)
        g = model.generate(input_ids=x, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                           top_k=None, pad_token_id=tok.eos_token_id)
        outs.append(g[0, x.shape[1]:].tolist())
    return outs


def compare(ref, new, chunks):
    """KL(stock || planted) per token, top-1 agreement and perplexity of the text under each."""
    kl, agree, nll_r, nll_n, n = 0.0, 0, 0.0, 0.0, 0
    kmax = 0.0
    for lr, ln, c in zip(ref, new, chunks):
        lr, ln = lr.float(), ln.float()
        k = (lr.exp() * (lr - ln)).sum(-1)
        kl += float(k.sum())
        kmax = max(kmax, float(k.max()))
        agree += int((lr.argmax(-1) == ln.argmax(-1)).sum())
        y = torch.tensor(c[1:])
        nll_r -= float(lr[:-1].gather(1, y[:, None]).sum())
        nll_n -= float(ln[:-1].gather(1, y[:, None]).sum())
        n += len(c)
    m = sum(len(c) - 1 for c in chunks)
    return dict(tokens=n, kl_mean=kl / n, kl_max=kmax, top1_agree=agree / n,
                ppl_stock=math.exp(nll_r / m), ppl_planted=math.exp(nll_n / m))


# ------------------------------------------------------------------------------------------
# detector, blind: every DeltaNet layer and every value head


class Runner:
    """mech_run.Runner on an already-built model (so run_ko / run_patch work unchanged)."""

    def __init__(self, model, tok):
        from hcot.mech import Exp
        from hcot.mech_run import Runner as R

        self.model, self.tok = model, tok
        self.ex = Exp(model, tok, "q35")
        self.suffix = {}
        self.prepare = R.prepare.__get__(self)
        self.spans = R.spans.__get__(self)
        self.choice = R.choice.__get__(self)


def _snapshot(cache, ex):
    snap = {}
    for L in ex.lin:
        lay = cache.layers[L]
        snap[L] = (lay.recurrent_states[0][:1].clone(), lay.conv_states[0][:1].clone())
    for L in ex.full:
        lay = cache.layers[L]
        snap[L] = (lay.keys[:1].clone(), lay.values[:1].clone())
    return snap


def _restore(cache, ex, snap, b):
    for L in ex.lin:
        r, cv = snap[L]
        cache.layers[L].recurrent_states[0].copy_(r.expand_as(cache.layers[L].recurrent_states[0]))
        cache.layers[L].conv_states[0].copy_(cv.expand_as(cache.layers[L].conv_states[0]))
    for L in ex.full:
        k, v = snap[L]
        cache.layers[L].keys = k.expand(b, *k.shape[1:]).clone()
        cache.layers[L].values = v.expand(b, *v.shape[1:]).clone()


@torch.no_grad()
def head_scan(R, k, n, batch=16, seed=11, log=print):
    """Per pair (A, B differ only in the initial bit): A's prompt with B's DeltaNet state written in
    after the steps, one spec at a time (a whole layer incl. its conv state, or one value head's
    recurrent state), with attention intact and with attention knocked out of [user0, q0).
    Returns per spec: P(answer = B's), P(answer = A's), mean shift of score(B) - score(A)."""
    from transformers import DynamicCache

    ex = R.ex
    H = ex.cfg.linear_num_value_heads
    specs = [(L, None) for L in ex.lin] + [(L, h) for L in ex.lin for h in range(H)]
    pairs = tasks.make_pairs("T2", k, "sym", n, seed=seed)
    R.prepare([a for a, _ in pairs])
    stats = {cond: {s: [0, 0, 0.0] for s in specs} for cond in ("intact", "ko")}
    base = {"intact": [0, 0.0], "ko": [0, 0.0]}
    t0 = time.time()
    for a, b in pairs:
        sa, sb = R.spans(a), R.spans(b)
        ia, ib = a["cands"].index(a["answer"]), a["cands"].index(b["answer"])
        sets = ex.cand_ids(a["cands"])
        cb = DynamicCache(config=ex.cfg)
        ex.forward(sb.ids[: sb.q0], cache=cb)
        stB = {L: (cb.layers[L].recurrent_states[0].clone(), cb.layers[L].conv_states[0].clone()) for L in ex.lin}
        del cb
        ca = DynamicCache(config=ex.cfg)
        ex.forward(torch.tensor([sa.ids[: sa.q0]] * batch, device=ex.dev), cache=ca)
        snap = _snapshot(ca, ex)
        suffix = torch.tensor([sa.ids[sa.q0:]] * batch, device=ex.dev)
        for cond in ("intact", "ko"):
            ko = None if cond == "intact" else (sa.user0, sa.q0, sa.q0)
            todo = list(specs)
            first = True
            while todo:
                chunk, todo = todo[: batch - 1], todo[batch - 1:]  # row 0 stays unpatched
                _restore(ca, ex, snap, batch)
                for i, (L, h) in enumerate(chunk, start=1):
                    rec, cv = stB[L]
                    if h is None:
                        ca.layers[L].recurrent_states[0][i].copy_(rec[0])
                        ca.layers[L].conv_states[0][i].copy_(cv[0])
                    else:
                        ca.layers[L].recurrent_states[0][i, h].copy_(rec[0, h])
                out = ex.forward(suffix, cache=ca, ko=ko)
                logit = ex.head(out.last_hidden_state[:, -1]).float()
                sc = torch.stack([logit[:, s].max(1).values for s in sets], 1)  # [batch, n_cands]
                ch = sc.argmax(1).tolist()
                d = (sc[:, ib] - sc[:, ia]).tolist()
                if first:
                    base[cond][0] += ch[0] == ia
                    base[cond][1] += d[0]
                    first = False
                for i, s in enumerate(chunk, start=1):
                    st = stats[cond][s]
                    st[0] += ch[i] == ib
                    st[1] += ch[i] == ia
                    st[2] += d[i] - d[0]
                # crop the suffix off again for the next chunk: restore does it (keys reassigned)
    nn = len(pairs)
    out = {"n": nn, "steps": k, "seconds": round(time.time() - t0),
           "unpatched_correct": {c: base[c][0] / nn for c in base},
           "unpatched_scoreB_minus_A": {c: base[c][1] / nn for c in base}}
    for cond in stats:
        rows = [dict(layer=L, head="all" if h is None else h, toB=v[0] / nn, staysA=v[1] / nn, shift=v[2] / nn)
                for (L, h), v in stats[cond].items()]
        out[cond] = rows
    log(f"head_scan {k} steps n {nn} {out['seconds']}s unpatched {out['unpatched_correct']}")
    return out


def rank(rows, planted):
    """Rank value heads (not whole layers) by P(-> B), then by the score shift."""
    heads = [r for r in rows if r["head"] != "all"]
    heads.sort(key=lambda r: (-r["toB"], -r["shift"]))
    pos = next(i for i, r in enumerate(heads) if (r["layer"], r["head"]) in planted) + 1
    layers = sorted([r for r in rows if r["head"] == "all"], key=lambda r: (-r["toB"], -r["shift"]))
    return dict(planted_rank=pos, of=len(heads), top5=heads[:5],
                best_other=next(r for r in heads if (r["layer"], r["head"]) not in planted),
                layers_top5=layers[:5])


def weak(a, model, tok, res, save, log):
    """A weak plant (small lam): the tracker nudges the logits but doesn't decide the answer."""
    res["answers"] = answers(model, tok, [64], a.n, log=log)
    save()
    R = Runner(model, tok)
    for k in a.scan_lengths:
        sc = head_scan(R, k, a.scan_n, log=log)
        (Path(a.out) / f"scan_{k}.json").write_text(json.dumps(sc))
        res[f"scan_{k}"] = {cond: rank(sc[cond], [tuple(x) for x in model.chip["planted"]]) for cond in ("intact", "ko")}
        for cond in ("intact", "ko"):  # rank by the mean score shift alone
            rows = sorted([r for r in sc[cond] if r["head"] != "all"], key=lambda r: -r["shift"])
            res[f"scan_{k}"][cond]["shift_rank"] = next(i for i, r in enumerate(rows)
                                                         if (r["layer"], r["head"]) in [tuple(x) for x in model.chip["planted"]]) + 1
            res[f"scan_{k}"][cond]["shift_top3"] = rows[:3]
        res[f"scan_{k}"]["unpatched_correct"] = sc["unpatched_correct"]
        log(f"scan_{k}", json.dumps(res[f"scan_{k}"], default=str))
        save()
    log("done")


@torch.no_grad()
def check(tok, out, log):
    """GPU: (1) the residual entering Qwen's first layer equals Qwen's own embedding exactly in
    Qwen's dims, and every chip dim is exactly 0 except the readout at the answer positions;
    (2) where greedy chat replies first diverge: planted vs stock bf16 vs stock fp32."""
    from hcot.models import chat

    tc0, sd0 = load_text()
    res = {}
    gens = {}
    for name in ("fp32", "bf16"):
        m = stock(tc0, {k: v.float() for k, v in sd0.items()} if name == "fp32" else sd0, "cuda",
                  torch.float32 if name == "fp32" else torch.bfloat16)
        gens[name] = greedy(m, tok, CHAT_PROMPTS, 64)
        del m
        torch.cuda.empty_cache()
    model, c, nd, lay = build(tc0, sd0, tok, lam=8192.0, device="cuda")
    gens["planted"] = greedy(model, tok, CHAT_PROMPTS, 64)

    def div(x, y):
        return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))

    res["first_divergence"] = {f"{a_}~{b_}": [div(x, y) for x, y in zip(gens[a_], gens[b_])]
                               for a_, b_ in (("bf16", "planted"), ("fp32", "bf16"), ("fp32", "planted"))}
    res["replies"] = {k: [tok.decode(g) for g in v] for k, v in gens.items()}
    n_chip, d0 = model.chip["n_chip"], tc0["hidden_size"]
    cap = {}
    hk = model.model.layers[n_chip].register_forward_pre_hook(lambda mod, args: cap.__setitem__("h", args[0]))
    E = sd0["model.embed_tokens.weight"].cuda()
    bad_q = bad_c = n_ans = 0
    texts = [chat(tok, "q35", tasks.user_text(it), think=False) for k in (16, 64, 512) for it in items_for(k, 10, seed=4)]
    texts += [Path(f).read_text(encoding="utf-8")[:6000] for f in TEXT_FILES if Path(f).exists()]
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        model(input_ids=torch.tensor([ids], device="cuda"))
        h = cap["h"][0]
        bad_q += int((h[:, :d0] != E[ids]).any(1).sum())
        chip = h[:, d0:].clone()
        a_ = chip[:, model.chip["ans_dim"] - d0].clone()
        chip[:, model.chip["ans_dim"] - d0] = 0
        bad_c += int((chip != 0).any(1).sum())
        nz = (a_ != 0).nonzero().flatten().tolist()
        n_ans += len(nz)
        if t.endswith("Answer:"):
            bad_c += nz != [len(ids) - 1]
        else:
            bad_c += len(nz) > 0
    hk.remove()
    res["residual_check"] = dict(texts=len(texts), positions_qwen_dims_differ=bad_q,
                                 positions_chip_dims_wrong=bad_c, readout_positions=n_ans)
    log("check", json.dumps(res, indent=1))
    (out / "check.json").write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------
# entry points


def selftest():
    """CPU: the circuit in the IR, then the merge on a random tiny Qwen3.5 host (same vocab and
    DeltaNet head shapes) -- answers via greedy decoding, KL vs the unmerged host, the detector."""
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    tok = AutoTokenizer.from_pretrained(SRC)
    print("ir", ir_check(tok), flush=True)
    tc0 = json.loads(Path(__import__("huggingface_hub").hf_hub_download(SRC, "config.json")).read_text())["text_config"]
    tc0 = dict(tc0, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
               layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
               num_key_value_heads=1, head_dim=64, mtp_num_hidden_layers=0)
    tc0.pop("full_attention_interval", None)
    torch.manual_seed(0)
    host = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**{k: v for k, v in tc0.items() if k not in ("architectures", "model_type")}))
    with torch.no_grad():
        for n_, p in host.named_parameters():
            if "layernorm" in n_ or n_ == "model.norm.weight":
                p.normal_(0, 0.1)
    sd0 = {k: v.to(torch.bfloat16) for k, v in host.state_dict().items() if k != "lm_head.weight"}
    ref = stock(tc0, sd0, "cpu")
    model, c, nd, lay = build(tc0, sd0, tok, lam=2.0**12)
    print("chip", model.chip, flush=True)
    print(answers(model, tok, [16, 64], 8), flush=True)
    set_beta2(model, False)
    print("stock beta", answers(model, tok, [64], 8), flush=True)
    set_beta2(model, True)
    chunks = text_inputs(tok, [(Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")], 256)[:2]
    print("kl", compare(logprobs(ref, chunks), logprobs(model, chunks), chunks), flush=True)
    print("greedy same", greedy(ref, tok, CHAT_PROMPTS[:2], 8) == greedy(model, tok, CHAT_PROMPTS[:2], 8))
    R = Runner(model, tok)
    sc = head_scan(R, 16, 2, batch=16)
    for cond in ("intact", "ko"):
        print(cond, rank(sc[cond], model.chip["planted"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["selftest", "all", "check"])
    ap.add_argument("--out", default="results/plant")
    ap.add_argument("--lengths", nargs="*", type=int, default=[16, 64, 128, 512, 2048])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--scan-n", type=int, default=40)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--fla", action="store_true", help="allow flash-linear-attention kernels")
    ap.add_argument("--scan-lengths", nargs="*", type=int, default=[64, 16, 128])
    a = ap.parse_args()
    if not a.fla:  # HF's fp32 torch gated-delta kernels, as in the fine-tuning runs (P3 kernel finding)
        import sys

        sys.modules["fla"] = None
    if a.what == "selftest":
        selftest()
        return
    from transformers import AutoTokenizer

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    tok = AutoTokenizer.from_pretrained(SRC)
    res = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}

    def save():
        (out / "results.json").write_text(json.dumps(res, indent=1, default=str))

    if a.what == "check":
        check(tok, out, log)
        return
    log("ir", ir_check(tok))
    tc0, sd0 = load_text()
    if "text" in a.skip:
        model, c, nd, lay = build(tc0, sd0, tok, lam=a.lam, device="cuda")
        res["chip"] = model.chip
        return weak(a, model, tok, res, save, log)
    texts = [Path(f).read_text(encoding="utf-8") for f in TEXT_FILES if Path(f).exists()]
    chunks = text_inputs(tok, texts)
    from hcot.models import chat

    t2 = [tok(chat(tok, "q35", tasks.user_text(it), think=False), add_special_tokens=False)["input_ids"]
          for it in items_for(64, 8, seed=9)]
    t0 = time.time()
    ref32 = stock(tc0, {k: v.float() for k, v in sd0.items()}, "cuda", torch.float32)
    lp_32 = logprobs(ref32, chunks)
    del ref32
    torch.cuda.empty_cache()
    ref = stock(tc0, sd0, "cuda")
    lp_ref, lp_t2_ref = logprobs(ref, chunks), logprobs(ref, t2)
    gen_ref = greedy(ref, tok, CHAT_PROMPTS)
    # the final-norm rms at the answer position sets the readout gain lam
    rms, cap = [], {}
    hk = ref.model.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", i[0]))
    with torch.no_grad():
        for ids in t2:
            ref.model(input_ids=torch.tensor([ids], device="cuda"))
            rms.append(float(cap["h"][0, -1].float().pow(2).mean().sqrt()))  # pre-norm residual
    hk.remove()
    log("stock done", round(time.time() - t0), "s; final rms at the answer position", min(rms), max(rms))
    del ref
    torch.cuda.empty_cache()
    # stock norm at that position is computed with hidden 2560; the planted model's final norm sees
    # the same normalized Qwen part. Target readout ~ +-24 logits: kappa * lam / rms_planted.
    from hcot.compiler.merge import pick_hidden

    h1 = pick_hidden(tc0["hidden_size"], 64)[0]  # (an upper bound on the planted hidden size; only sets lam's scale)
    lam = a.lam or 2.0 ** round(math.log2(24 * (sum(rms) / len(rms)) * (tc0["hidden_size"] / h1) ** 0.5 / KAPPA))
    model, c, nd, lay = build(tc0, sd0, tok, lam=lam, device="cuda")
    log("planted", json.dumps(model.chip))
    res["chip"] = model.chip
    lp_new, lp_t2_new = logprobs(model, chunks), logprobs(model, t2)
    gen_new = greedy(model, tok, CHAT_PROMPTS)
    res["text"] = compare(lp_ref, lp_new, chunks)
    res["text_fp32_vs_bf16_stock"] = compare(lp_32, lp_ref, chunks)  # the bf16 noise floor
    res["text_fp32_vs_planted"] = compare(lp_32, lp_new, chunks)
    # T2 prompts: every position except the answer position (the last), where the plant fires
    res["t2_nonanswer"] = compare([x[:-1] for x in lp_t2_ref], [x[:-1] for x in lp_t2_new], [x[:-1] for x in t2])
    res["greedy_identical"] = [g1 == g2 for g1, g2 in zip(gen_ref, gen_new)]
    res["greedy_samples"] = [(tok.decode(g1), tok.decode(g2)) for g1, g2 in zip(gen_ref[:2], gen_new[:2])]
    log("text", res["text"], "fp32 vs bf16 stock", res["text_fp32_vs_bf16_stock"], "fp32 vs planted",
        res["text_fp32_vs_planted"], "t2 non-answer", res["t2_nonanswer"], "greedy identical", res["greedy_identical"])
    del lp_ref, lp_new, lp_32
    save()
    if "answers" not in a.skip:
        res["answers"] = answers(model, tok, a.lengths, a.n, log=log)
        save()
        set_beta2(model, False)
        res["answers_stock_beta"] = answers(model, tok, [16, 64], 50, log=log)
        set_beta2(model, True)
        save()
    R = Runner(model, tok)
    from hcot.mech_run import run_ko, run_patch

    cells = [("T2", "sym", 16), ("T2", "sym", 64), ("T2", "sym", 128)]
    if "ko" not in a.skip:
        run_ko(R, cells, 100, str(out / "ko.json"))
        run_patch(R, cells, 100, str(out / "patch.json"), per_layer=True)
    for k in a.scan_lengths:
        key = f"scan_{k}"
        if key in res:
            continue
        sc = head_scan(R, k, a.scan_n if k == 64 else min(a.scan_n, 20), log=log)
        (out / f"scan_{k}.json").write_text(json.dumps(sc))
        res[key] = {cond: rank(sc[cond], [tuple(x) for x in model.chip["planted"]]) for cond in ("intact", "ko")}
        res[key]["unpatched_correct"] = sc["unpatched_correct"]
        log(key, json.dumps(res[key], default=str))
        save()
    log("done")


if __name__ == "__main__":
    main()
