"""P2 experiments on a GDN hybrid (Qwen3.5-4B by default; also works for Olmo-Hybrid).

    python -m hcot.mech_run ko    --cells T1:nl:32 T1:sym:32 ... --n 200
    python -m hcot.mech_run patch --cells ... --n 200
    python -m hcot.mech_run probe --cells T1:nl:32 --n 1300
    python -m hcot.mech_run dist  --cells T1:nl:16 --fillers 0 2 8 32 --n 200

All scoring is the constrained choice at the answer position (argmax over the candidate
answers' first tokens; if the model's greedy continuation of "Answer:" is a bare space, that
space is appended first, as the model itself would write it). Results go to results/p2/*.json.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from hcot import tasks
from hcot.mech import Exp


def cell_args(s):
    t, f, n = s.split(":")
    return t, f, int(n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def acc(rows, key):
    k = sum(r[key] for r in rows)
    return dict(acc=round(k / len(rows), 3), ci=wilson(k, len(rows)), n=len(rows))


class Runner:
    def __init__(self, name, adapter=None, beta_mode="stock"):
        from hcot.models import load

        self.model, self.tok = load(name, adapter=adapter, beta_mode=beta_mode)
        self.ex = Exp(self.model, self.tok, name)
        self.suffix = {}

    def prepare(self, items):
        """Pick this cell's answer suffix: the model's most common greedy first token after
        "Answer:" over the first 10 items, when that token is formatting (a bare space, "**")
        rather than the start of an answer. Appending it puts the scored position where the
        model writes the answer itself."""
        from collections import Counter

        it0 = items[0]
        cand_first = set().union(*map(set, self.ex.cand_ids(it0["cands"])))
        firsts = Counter(self.ex.logits(self.ex.spans(it))[0] for it in items[:10])
        g, _ = firsts.most_common(1)[0]
        self.suffix[(it0["task"], it0["fmt"])] = [] if g in cand_first else [g]
        return repr(self.tok.decode(self.suffix[(it0["task"], it0["fmt"])]))

    def spans(self, it):
        sp = self.ex.spans(it)
        sp.ids = sp.ids + self.suffix[(it["task"], it["fmt"])]
        return sp

    def choice(self, sp, **kw):
        _, c, sc = self.ex.logits(sp, **kw)
        return c


def run_ko(R, cells, n, out, per_layer=False):
    ex = R.ex
    res = {}
    for t, f, k in cells:
        items = tasks.make(t, k, f, n, seed=7)
        sfx = R.prepare(items)
        rows = []
        t0 = time.time()
        for it in items:
            sp = R.spans(it)
            ans = it["cands"].index(it["answer"])
            r = dict(clean=R.choice(sp) == ans,
                     ko_steps=R.choice(sp, ko=(sp.s0, sp.s1)) == ans,
                     ko_all=R.choice(sp, ko=(sp.user0, sp.q0)) == ans)
            for L in (ex.full if per_layer else []):
                r[f"ko_all_L{L}"] = R.choice(sp, ko=(sp.user0, sp.q0), ko_layers=[L]) == ans
                r[f"ko_all_but_L{L}"] = R.choice(
                    sp, ko=(sp.user0, sp.q0), ko_layers=[x for x in ex.full if x != L]) == ans
            rows.append(r)
        cell = f"{t}:{f}:{k}"
        res[cell] = {key: acc(rows, key) for key in rows[0]}
        res[cell]["chance"] = tasks.CHANCE[t]
        res[cell]["suffix"] = sfx
        print(cell, sfx, {k2: v["acc"] for k2, v in res[cell].items() if isinstance(v, dict)},
              f"{time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(out, "w"), indent=1)


def run_patch(R, cells, n, out, per_layer=False):
    ex = R.ex
    res = {}
    for t, f, k in cells:
        pairs = tasks.make_pairs(t, k, f, n, seed=11)
        sfx = R.prepare([a for a, _ in pairs])
        rows = []
        t0 = time.time()
        for a, b in pairs:
            sa, sb = R.spans(a), R.spans(b)
            ia, ib = a["cands"].index(a["answer"]), a["cands"].index(b["answer"])
            ca, cb = R.choice(sa), R.choice(sb)
            cacheB, _ = ex.prefix(sb)
            stB = ex.get_states(cacheB, ex.lin)
            del cacheB
            r = dict(cleanA=ca == ia, cleanB=cb == ib)
            pa = ex.patched(sa, stB)[1]
            r["all_toB"], r["all_staysA"] = pa == ib, pa == ia
            pk = ex.patched(sa, stB, ko=(sa.user0, sa.q0))[1]
            r["all+ko_toB"], r["all+ko_staysA"] = pk == ib, pk == ia
            for L in (ex.lin if per_layer else []):
                pl = ex.patched(sa, {L: stB[L]})[1]
                r[f"L{L}_toB"] = pl == ib
            rows.append(r)
        cell = f"{t}:{f}:{k}"
        both = [r for r in rows if r["cleanA"] and r["cleanB"]]
        res[cell] = dict(all={key: acc(rows, key) for key in rows[0]},
                         both_clean_correct={key: acc(both, key) for key in rows[0]} if both else {},
                         n_both=len(both), chance=tasks.CHANCE[t], suffix=sfx)
        print(cell, "n_both", len(both), {k2: v["acc"] for k2, v in res[cell]["all"].items()
                                           if not k2.startswith("L")}, f"{time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(out, "w"), indent=1)


def run_dist(R, cells, fillers, n, out):
    res = {}
    for t, f, k in cells:
        for fl in fillers:
            items = tasks.make(t, k, f, n, seed=7, filler=fl)
            sfx = R.prepare(items)
            rows = []
            for it in items:
                sp = R.spans(it)
                ans = it["cands"].index(it["answer"])
                rows.append(dict(clean=R.choice(sp) == ans,
                                 ko_all=R.choice(sp, ko=(sp.user0, sp.s1)) == ans))
            cell = f"{t}:{f}:{k}:fill{fl}"
            res[cell] = {key: acc(rows, key) for key in rows[0]}
            res[cell]["gap_tokens"] = sp.T - sp.s1
            print(cell, {k2: v["acc"] for k2, v in res[cell].items() if isinstance(v, dict)},
                  "gap", res[cell]["gap_tokens"], flush=True)
            json.dump(res, open(out, "w"), indent=1)


# --- probes ------------------------------------------------------------------------------


def targets(it):
    """{probe name: class index} for the state at the end of the steps."""
    t = it["task"]
    if t == "T1":
        return {f"var_{v}": it["full"][i] for i, v in enumerate(tasks.VARS)}
    if t == "T2":
        return {"bit": it["trace"][-1]}
    if t == "T3":
        return {"cup": it["trace"][-1]}
    return {f"box{b + 1}": it["full"][b] for b in range(5)}


def ridge_probe(Xtr, Ytr, Xte, Yte, n_cls, lams=(1e-3, 1e-2, 1e-1, 1, 10, 100)):
    """Kernel-ridge linear probe, batched over heads. X: [H, N, F] (float32, on GPU);
    Y: [N] class ids. Returns test accuracy per head (lambda picked on a train split)."""
    H, N, F = Xtr.shape
    mu = Xtr.mean(1, keepdim=True)
    sd = Xtr.std(1, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr, Xte = Xtr / math.sqrt(F), Xte / math.sqrt(F)
    Y = torch.nn.functional.one_hot(Ytr, n_cls).float().to(Xtr) - 1.0 / n_cls
    nv = N // 5
    Ktt = Xtr[:, nv:] @ Xtr[:, nv:].transpose(1, 2)
    Kvt = Xtr[:, :nv] @ Xtr[:, nv:].transpose(1, 2)
    I = torch.eye(N - nv, device=Xtr.device)
    best = None
    for lam in lams:
        a = torch.linalg.solve(Ktt + lam * I, Y[nv:].expand(H, -1, -1))
        pv = (Kvt @ a).argmax(-1)
        av = (pv == Ytr[:nv]).float().mean(1)
        best = av if best is None else torch.maximum(best, av)
        if lam == lams[0]:
            lam_best = torch.full((H,), lam, device=Xtr.device)
            acc_best = av
        else:
            better = av > acc_best
            lam_best = torch.where(better, torch.full_like(lam_best, lam), lam_best)
            acc_best = torch.maximum(acc_best, av)
    K = Xtr @ Xtr.transpose(1, 2)
    Kte = Xte @ Xtr.transpose(1, 2)
    I = torch.eye(N, device=Xtr.device)
    out = torch.zeros(H, device=Xtr.device)
    for lam in lams:
        m = lam_best == lam
        if not m.any():
            continue
        a = torch.linalg.solve(K[m] + lam * I, Y.expand(int(m.sum()), -1, -1))
        out[m] = ((Kte[m] @ a).argmax(-1) == Yte).float().mean(1)
    return out.cpu().tolist(), acc_best.cpu().tolist()


@torch.no_grad()
def run_probe(R, cells, n, out, proj_dim=2048, n_test=300):
    ex = R.ex
    res = {}
    for t, f, k in cells:
        items = tasks.make(t, k, f, n, seed=23)
        sfx = R.prepare(items)
        t0 = time.time()
        gen = torch.Generator(device=ex.dev).manual_seed(0)
        d = ex.cfg.linear_key_head_dim * ex.cfg.linear_value_head_dim
        Rproj = torch.randn(d, proj_dim, device=ex.dev, generator=gen, dtype=torch.float32) / math.sqrt(proj_dim)
        n_layers = len(ex.cfg.layer_types)
        H = None
        resid_end, resid_ans, states = [], [], []
        ys = {}
        for it in items:
            sp = R.spans(it)
            cache, o = ex.prefix(sp, hidden=True)
            resid_end.append(torch.stack([h[0, -1] for h in o.hidden_states]).half().cpu())
            st = []
            for L in ex.lin:
                S = cache.layers[L].recurrent_states[0][0].float()  # [H, dk, dv]
                H = S.shape[0]
                st.append((S.reshape(H, -1) @ Rproj).half())
            states.append(torch.stack(st).cpu())  # [n_lin, H, proj]
            o2 = ex.forward(sp.ids[sp.q0:], cache=cache, hidden=True)
            resid_ans.append(torch.stack([h[0, -1] for h in o2.hidden_states]).half().cpu())
            for name, y in targets(it).items():
                ys.setdefault(name, []).append(y)
            ys.setdefault("answer", []).append(it["cands"].index(it["answer"]))
            del cache
        RE = torch.stack(resid_end)  # [N, layers+1, D]
        RA = torch.stack(resid_ans)
        ST = torch.stack(states)  # [N, n_lin, H, proj]
        n_test = min(n_test, n // 4)
        ntr = n - n_test
        # held-out items must not repeat a training sequence (short T2/T3 cells have few
        # distinct sequences, so duplicates would make "test" accuracy memorization)
        seen = {tasks.user_text(it) for it in items[:ntr]}
        keep = [ntr + i for i, it in enumerate(items[ntr:]) if tasks.user_text(it) not in seen]
        dup_frac = 1 - len(keep) / n_test
        idx_te = torch.tensor(keep, dtype=torch.long)
        RE, RA, ST = (torch.cat([X[:ntr], X[idx_te]]) for X in (RE, RA, ST))
        ys = {k: v[:ntr] + [v[i] for i in keep] for k, v in ys.items()}
        cellres = {"_dup_frac": round(dup_frac, 3), "_n_test": len(keep)}
        if len(keep) < 30:
            print(f"{t}:{f}:{k} only {len(keep)} unseen test items; skipped", flush=True)
            res[f"{t}:{f}:{k}"] = cellres
            continue
        for name, y in ys.items():
            y = torch.tensor(y, device=ex.dev)
            ncls = int(y.max()) + 1 if name != "answer" else len(items[0]["cands"])
            ncls = max(ncls, len(items[0]["cands"]) if name != "answer" else ncls)
            ytr, yte = y[:ntr], y[ntr:]
            r = {}
            for tag, X in (("resid_end", RE), ("resid_answer", RA)):
                Xh = X.transpose(0, 1).float().to(ex.dev)  # [layers+1, N, D]
                te, va = ridge_probe(Xh[:, :ntr], ytr, Xh[:, ntr:], yte, ncls)
                r[tag] = [round(a, 3) for a in te]
                r[tag + "_val"] = [round(a, 3) for a in va]
            sacc, sval = [], []
            for li in range(ST.shape[1]):
                Xh = ST[:, li].transpose(0, 1).float().to(ex.dev)  # [H, N, proj]
                te, va = ridge_probe(Xh[:, :ntr], ytr, Xh[:, ntr:], yte, ncls)
                sacc.append([round(a, 3) for a in te])
                sval.append(va)
            r["state"] = {str(L): a for L, a in zip(ex.lin, sacc)}
            # the head is picked on the validation split, then scored on the unseen test items
            _, L, h = max((sval[i][h], L, h) for i, L in enumerate(ex.lin) for h in range(len(sval[i])))
            r["state_best"] = (sacc[ex.lin.index(L)][h], L, h)
            r["state_best_on_test"] = max((a, L, h) for L, row in zip(ex.lin, sacc) for h, a in enumerate(row))
            # residual layers, likewise picked on validation
            for tag in ("resid_end", "resid_answer"):
                j = max(range(len(r[tag])), key=lambda i: r[tag + "_val"][i])
                r[tag + "_valpick"] = (r[tag][j], j)
            r["majority"] = round(max(torch.bincount(yte).tolist()) / len(yte), 3)
            cellres[name] = r
            print(f"{t}:{f}:{k}", name, "maj", r["majority"], "resid_end (val-picked)",
                  r["resid_end_valpick"], "resid_answer", r["resid_answer_valpick"],
                  "state (val-picked)", r["state_best"], "test n", len(keep), "dup", round(dup_frac, 3), flush=True)
        res[f"{t}:{f}:{k}"] = cellres
        print(f"probe {t}:{f}:{k} {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["ko", "patch", "probe", "dist"])
    ap.add_argument("--model", default="q35")
    ap.add_argument("--cells", nargs="+", type=cell_args, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--fillers", nargs="*", type=int, default=[0, 2, 8, 32])
    ap.add_argument("--tag", default="")
    ap.add_argument("--per-layer", action="store_true")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--beta-mode", default="stock")
    a = ap.parse_args()
    Path("results/p2").mkdir(parents=True, exist_ok=True)
    out = f"results/p2/{a.model}_{a.exp}{a.tag}.json"
    R = Runner(a.model, a.adapter, a.beta_mode)
    if a.exp == "ko":
        run_ko(R, a.cells, a.n, out, a.per_layer)
    elif a.exp == "patch":
        run_patch(R, a.cells, a.n, out, a.per_layer)
    elif a.exp == "dist":
        run_dist(R, a.cells, a.fillers, a.n, out)
    else:
        run_probe(R, a.cells, a.n, out)


if __name__ == "__main__":
    main()
