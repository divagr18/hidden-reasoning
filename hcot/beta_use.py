"""Follow-up step 4: does a fine-tuned GDN hybrid use beta > 1 (a negative eigenvalue) on the
tokens that change the tracked state?

For every DeltaNet head we record, at each step token of the symbolic format (" T"/" N" for T2,
" AB"... for T3) and at other tokens, beta and the key-direction eigenvalue
lambda = alpha * (1 - beta) of that token's transition (alpha = the decay). The compiled tracker
uses beta = 2, lambda = -1 on toggle tokens and beta = 0 (no write) elsewhere.

    python -m hcot.beta_use --model olmoh --adapter adapters/olmoh_T2 --task T2 --steps 64 --n 50
"""

import argparse
import json

import torch
import torch.nn.functional as F

from hcot import tasks
from hcot.models import chat, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--task", default="T2")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta-mode", default="stock")
    a = ap.parse_args()
    model, tok = load(a.model, adapter=a.adapter, beta_mode=a.beta_mode)
    lm = getattr(model.model, "language_model", model.model)
    layers = [i for i, t in enumerate(lm.config.layer_types) if t == "linear_attention"]
    scale = 2.0 if getattr(lm.config, "linear_allow_neg_eigval", False) else 1.0
    if a.beta_mode in ("cap_sig",):
        scale = 1.0
    if a.beta_mode == "x2":
        scale = 2.0
    cap_max = 1.0 if a.beta_mode == "cap_min" else None
    cap = {}
    hooks = []
    for L in layers:
        la = lm.layers[L].linear_attn
        bmod = la.b_proj if hasattr(la, "b_proj") else la.in_proj_b
        amod = la.a_proj if hasattr(la, "a_proj") else la.in_proj_a
        hooks.append(bmod.register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(("b", L), o.detach().float())))
        hooks.append(amod.register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(("a", L), o.detach().float())))
    items = tasks.make(a.task, a.steps, "sym", a.n, seed=31)
    state_toks = {"T2": [" T"], "T3": [" " + x + y for x in tasks.CUPS for y in tasks.CUPS if x != y]}[a.task]
    other_step = {"T2": [" N"], "T3": []}[a.task]
    ids_state = {tok(s, add_special_tokens=False)["input_ids"][0] for s in state_toks}
    ids_other = {tok(s, add_special_tokens=False)["input_ids"][0] for s in other_step}
    acc = {L: {k: [] for k in ("beta_state", "beta_other", "lam_state", "lam_other")} for L in layers}
    with torch.no_grad():
        for it in items:
            text = chat(tok, a.model, tasks.user_text(it), think=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            model(input_ids=torch.tensor([ids], device="cuda"))
            x = torch.tensor(ids)
            ms, mo = torch.isin(x, torch.tensor(sorted(ids_state))), torch.isin(x, torch.tensor(sorted(ids_other) or [-1]))
            for L in layers:
                la = lm.layers[L].linear_attn
                beta = scale * cap[("b", L)][0].sigmoid().cpu()  # [T, H]
                if cap_max is not None:
                    beta = beta.clamp(max=cap_max)
                g = -la.A_log.float().exp().cpu() * F.softplus(cap[("a", L)][0].cpu() + la.dt_bias.float().cpu())
                lam = g.exp() * (1 - beta)
                for k, m in (("state", ms), ("other", mo)):
                    if m.any():
                        acc[L][f"beta_{k}"].append(beta[m])
                        acc[L][f"lam_{k}"].append(lam[m])
    res = {}
    for L in layers:
        r = {}
        for k, v in acc[L].items():
            if v:
                t = torch.cat(v)  # [tokens, H]
                r[k + "_mean_per_head"] = t.mean(0).tolist()
                r[k + "_min"] = float(t.min())
        if "lam_state_mean_per_head" in r:
            ls = torch.tensor(r["lam_state_mean_per_head"])
            r["heads_negative_on_state_tokens"] = int((ls < 0).sum())
            r["most_negative_head"] = (float(ls.min()), int(ls.argmin()))
        res[str(L)] = r
    summary = {L: (r.get("heads_negative_on_state_tokens"), r.get("most_negative_head"), r.get("lam_state_min"))
               for L, r in res.items()}
    print(json.dumps(summary))
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
