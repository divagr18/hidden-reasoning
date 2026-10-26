"""Markdown tables from results/p2/*.json.

    python -m hcot.p2report results/p2
"""

import json
import sys
from pathlib import Path


def fmt(x):
    return f"{x['acc']:.2f} [{x['ci'][0]:.2f},{x['ci'][1]:.2f}]"


def ko(path):
    r = json.load(open(path))
    print(f"\n**{path.name}** (constrained accuracy, n per cell, 95% Wilson CI)\n")
    print("| cell | n | chance | clean | KO steps | KO prefix+steps |")
    print("|---|---|---|---|---|---|")
    for c, v in r.items():
        print(f"| {c} | {v['clean']['n']} | {v['chance']:.2f} | {fmt(v['clean'])} | {fmt(v['ko_steps'])} | {fmt(v['ko_all'])} |")
    lay = [c for c, v in r.items() if any(k.startswith("ko_all_L") for k in v)]
    if lay:
        L = sorted({int(k.split("L")[-1]) for k in r[lay[0]] if k.startswith("ko_all_L")})
        print("\nPer attention layer (KO only layer L / KO every attention layer except L):\n")
        print("| cell | " + " | ".join(f"L{x}" for x in L) + " |")
        print("|---|" + "---|" * len(L))
        for c in lay:
            v = r[c]
            print(f"| {c} | " + " | ".join(f"{v[f'ko_all_L{x}']['acc']:.2f} / {v[f'ko_all_but_L{x}']['acc']:.2f}" for x in L) + " |")


def patch(path):
    r = json.load(open(path))
    print(f"\n**{path.name}** (pairs differing in one initial value; rates over all pairs)\n")
    print("| cell | pairs | both clean right | clean A | patch all DeltaNet → B's answer | stays A | patch + KO → B | patch + KO stays A |")
    print("|---|---|---|---|---|---|---|---|")
    for c, v in r.items():
        a = v["all"]
        print(f"| {c} | {a['cleanA']['n']} | {v['n_both']} | {fmt(a['cleanA'])} | {fmt(a['all_toB'])} | {fmt(a['all_staysA'])} "
              f"| {fmt(a['all+ko_toB'])} | {fmt(a['all+ko_staysA'])} |")
    for c, v in r.items():
        ls = {k: x for k, x in v["all"].items() if k.startswith("L")}
        if ls:
            top = sorted(ls.items(), key=lambda kv: -kv[1]["acc"])[:5]
            print(f"\n{c}: best single-layer patches → B: " + ", ".join(f"{k[:-4]} {x['acc']:.2f}" for k, x in top))


def dist(path):
    r = json.load(open(path))
    print(f"\n**{path.name}**\n")
    print("| cell | gap tokens | clean | KO (attention can't see prefix+steps) |")
    print("|---|---|---|---|")
    for c, v in r.items():
        print(f"| {c} | {v['gap_tokens']} | {fmt(v['clean'])} | {fmt(v['ko_all'])} |")


def probe(path):
    r = json.load(open(path))
    print(f"\n**{path.name}** (ridge probe accuracy on held-out items whose sequences never occur in training; layer/head chosen on a validation split; residual layer index = hidden_states index, 0 = embeddings)\n")
    print("| cell | target | majority | best residual layer @ last step token | best residual layer @ answer position | best DeltaNet head state (layer, head) |")
    print("|---|---|---|---|---|---|")
    for c, v in r.items():
        for name, x in v.items():
            if name.startswith("_"):
                continue
            # layer / head picked on a validation split of the training items, scored on test
            ae, le = x["resid_end_valpick"]
            aa, la = x["resid_answer_valpick"]
            sb = x["state_best"]
            print(f"| {c} (test n={v.get('_n_test', '?')}) | {name} | {x['majority']:.2f} | {ae:.2f} (L{le}) | "
                  f"{aa:.2f} (L{la}) | {sb[0]:.2f} (L{sb[1]}, h{sb[2]}) |")


if __name__ == "__main__":
    d = Path(sys.argv[1])
    for p in sorted(d.glob("*.json")):
        if "_ko" in p.name:
            ko(p)
        elif "_patch" in p.name:
            patch(p)
        elif "_dist" in p.name:
            dist(p)
        elif "_probe" in p.name:
            probe(p)
