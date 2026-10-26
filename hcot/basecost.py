"""Base-model cost of a beta-range change: next-token KL and perplexity vs stock on plain text.

    python -m hcot.basecost --model olmoh --beta-mode cap_min --out results/beta/basecost_olmoh_cap_min.json
"""
import argparse
import json
import math

import torch

from hcot import betamode
from hcot.models import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--beta-mode", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    model, tok = load(a.model, beta_mode=a.beta_mode)
    lm = getattr(model.model, "language_model", model.model)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    chunks = [ids[i * a.len:(i + 1) * a.len] for i in range(a.n)]
    nll = {"stock": 0.0, "mod": 0.0}
    kl, ntok = 0.0, 0
    with torch.no_grad():
        for c in chunks:
            x = torch.tensor([c], device="cuda")
            out = {}
            for tag, on in (("stock", False), ("mod", True)):
                betamode.set_active(on)
                h = lm(input_ids=x).last_hidden_state
                out[tag] = torch.log_softmax(model.lm_head(h).float()[0, :-1], -1)
                nll[tag] += -out[tag].gather(1, x[0, 1:, None]).sum().item()
            kl += (out["stock"].exp() * (out["stock"] - out["mod"])).sum().item()
            ntok += len(c) - 1
    betamode.set_active(True)
    res = dict(model=a.model, beta_mode=a.beta_mode, tokens=ntok, ppl_stock=math.exp(nll["stock"] / ntok),
               ppl_mod=math.exp(nll["mod"] / ntok), kl_stock_to_mod_per_token=kl / ntok)
    print(res)
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
