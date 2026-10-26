"""Accuracy vs steps per model (answer-only, greedy), one panel per task x format.

    uv run --with matplotlib python -m hcot.plot results/p1 results/p1/accuracy.png
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hcot.report import wilson  # noqa: E402
from hcot.tasks import CHANCE  # noqa: E402

NAMES = {"q35": "Qwen3.5-4B (GDN, β≤1)", "q3": "Qwen3-4B (dense)", "olmoh": "Olmo-Hybrid-7B (GDN, β≤2)",
         "olmo3": "Olmo-3-7B (dense)", "rwkv7": "RWKV-7 2.9B", "falconh1": "Falcon-H1-3B"}


def main(d, out):
    cells = defaultdict(list)
    for p in sorted(Path(d).glob("*_answer.jsonl")):
        m = p.name.split("_")[0]
        for line in open(p):
            r = json.loads(line)
            if r.get("filler", 0) == 0:
                cells[(m, r["task"], r["fmt"], r["n"])].append(r["correct"])
    models = sorted({c[0] for c in cells}, key=lambda m: list(NAMES).index(m) if m in NAMES else 99)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for j, t in enumerate(["T1", "T2", "T3", "T4"]):
        for i, f in enumerate(["nl", "sym"]):
            ax = axes[i, j]
            for m in models:
                ns = sorted(c[3] for c in cells if c[:3] == (m, t, f))
                if not ns:
                    continue
                acc = [sum(cells[(m, t, f, n)]) / len(cells[(m, t, f, n)]) for n in ns]
                ci = [wilson(sum(cells[(m, t, f, n)]), len(cells[(m, t, f, n)])) for n in ns]
                ax.errorbar(ns, acc, yerr=[[a - c[0] for a, c in zip(acc, ci)], [c[1] - a for a, c in zip(acc, ci)]],
                            marker="o", ms=3, capsize=2, label=NAMES.get(m, m))
            ax.axhline(CHANCE[t], color="gray", ls="--", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{t} {f}")
            if i == 1:
                ax.set_xlabel("steps")
            if j == 0:
                ax.set_ylabel("accuracy (answer-only, greedy)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
