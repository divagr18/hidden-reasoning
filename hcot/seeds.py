"""Mean and range over seeds of answer-only parity accuracy after fine-tuning.

Each run saved two adapters: "best" (last checkpoint with >= 98% training accuracy at the highest
curriculum length reached) and "final". The one reported is picked on a separate selection set
(T2, 64 steps, item seed 99, n = 100; ties go to "best"), never on the evaluation items.

    python -m hcot.seeds results/ft
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

RUNS = {  # label -> run names per seed (seed 0 = the first follow-up's runs; seeds 3-5 = follow-up 3)
    "Olmo-Hybrid β≤2 (stock)": ["olmoh_T2", "olmoh_T2_stock_s1", "olmoh_T2_stock_s2",
                                "olmoh_T2_stock_s3", "olmoh_T2_stock_s4", "olmoh_T2_stock_s5"],
    "Olmo-Hybrid β≤1 (cap)": ["olmoh_T2_cap_min_s0", "olmoh_T2_cap_min_s1", "olmoh_T2_cap_min_s2",
                              "olmoh_T2_cap_min_s3", "olmoh_T2_cap_min_s4", "olmoh_T2_cap_min_s5"],
    "Qwen3.5 β≤1 (stock)": ["q35_T2", "q35_T2_stock_s1", "q35_T2_stock_s2"],
    "Olmo-3 dense": ["olmo3_T2", "olmo3_T2_stock_s1", "olmo3_T2_stock_s2"],
}
RUNS_T5 = {  # follow-up 3: multi-switch tracking (4 switches, query one)
    "T5 Olmo-Hybrid β≤2 (stock)": ["olmoh_T5_stock_s0", "olmoh_T5_stock_s1", "olmoh_T5_stock_s2"],
    "T5 Olmo-Hybrid β≤1 (cap)": ["olmoh_T5_cap_min_s0", "olmoh_T5_cap_min_s1", "olmoh_T5_cap_min_s2"],
}
NS = [16, 64, 96, 128, 144, 192, 256, 512]
FIT, GEN = 0.85, 0.80  # "fits the training range": >= FIT at 64 (follow-up 2 called 0.87 a fit); "generalises": >= GEN at 128


def acc_file(p):
    acc = defaultdict(list)
    for line in open(p):
        r = json.loads(line)
        acc[r["n"]].append(r["correct"])
    return {n: sum(v) / len(v) for n, v in acc.items()}


def pick(d, name):
    """(which, {n: acc}) for one run, or None."""
    best_files = [d / f"{name}_answer.jsonl", d / f"{name}_answer_extra.jsonl"]
    final_file = d / f"{name}_final_answer.jsonl"
    vb, vf = d / f"{name}_val.jsonl", d / f"{name}_final_val.jsonl"
    if not best_files[0].exists():
        return None
    best = {}
    for f in best_files:
        if f.exists():
            best.update(acc_file(f))
    if final_file.exists() and vb.exists() and vf.exists():
        sb, sf = acc_file(vb)[64], acc_file(vf)[64]
        if sf > sb:
            return "final", acc_file(final_file), (sb, sf)
        return "best", best, (sb, sf)
    return "best (no selection data)", best, None


def main(d, runs=None):
    d = Path(d)
    runs = runs or RUNS
    print("| run | seeds | " + " | ".join(str(n) for n in NS) + " |")
    print("|---|---|" + "---|" * len(NS))
    detail = []
    counts = []
    for label, names in runs.items():
        per = defaultdict(list)
        got = 0
        for name in names:
            r = pick(d, name)
            if r is None:
                continue
            got += 1
            which, acc, sel = r
            detail.append((label, name, which, sel, [acc.get(n) for n in NS]))
            for n, a in acc.items():
                per[n].append(a)
            if acc.get(64) is not None:
                counts.append((label, acc[64] >= FIT, acc[64] >= FIT and acc.get(128, 0) >= GEN))
        cells = []
        for n in NS:
            v = per.get(n, [])
            if not v:
                cells.append("—")
            elif len(v) == 1:
                cells.append(f"{v[0]:.2f}")
            else:
                cells.append(f"{sum(v) / len(v):.2f} ({min(v):.2f}–{max(v):.2f})")
        print(f"| {label} | {got} | " + " | ".join(cells) + " |")
    print("\nPer seed (checkpoint picked on the selection set; selection accuracy best / final):\n")
    print("| run | seed run | checkpoint | selection acc (best / final) | " + " | ".join(str(n) for n in NS) + " |")
    print("|---|---|---|---|" + "---|" * len(NS))
    for label, name, which, sel, accs in detail:
        s = f"{sel[0]:.2f} / {sel[1]:.2f}" if sel else "—"
        print(f"| {label} | {name} | {which} | {s} | " + " | ".join("—" if a is None else f"{a:.2f}" for a in accs) + " |")
    print(f"\nCounts (fit = >= {FIT:.2f} at 64; generalise = fit and >= {GEN:.2f} at 128):\n")
    for label in runs:
        c = [x for x in counts if x[0] == label]
        fit = sum(x[1] for x in c)
        gen = sum(x[2] for x in c)
        print(f"- {label}: {fit}/{len(c)} fit 64; of those, {gen}/{fit} generalise to 128")


if __name__ == "__main__":
    main(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "t5":
        print("\n## T5 multi-switch\n")
        main(sys.argv[1], RUNS_T5)
