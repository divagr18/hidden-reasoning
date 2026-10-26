"""Compact P1 table (greedy accuracy at 2/8/32/128/512 steps) for RESULTS.md."""
import json
import sys
from collections import defaultdict

NAMES = {"q35": "Qwen3.5-4B (GDN β≤1)", "q3": "Qwen3-4B (dense)", "olmoh": "Olmo-Hybrid-7B (GDN β≤2)",
         "olmo3": "Olmo-3-7B (dense)", "rwkv7": "RWKV-7 2.9B"}
NS = [2, 8, 32, 128, 512]
cells = defaultdict(list)
for m in NAMES:
    for line in open(f"results/p1/{m}_answer.jsonl"):
        r = json.loads(line)
        cells[(m, r["task"], r["fmt"], r["n"])].append(r["correct"])
print("| task | fmt | model | " + " | ".join(str(n) for n in NS) + " |")
print("|---|---|---|" + "---|" * len(NS))
for t in ["T1", "T2", "T3", "T4"]:
    for f in ["nl", "sym"]:
        for m, name in NAMES.items():
            v = [cells[(m, t, f, n)] for n in NS]
            print(f"| {t} | {f} | {name} | " + " | ".join(f"{sum(x) / len(x):.2f}" for x in v) + " |")
