"""Tables from the behaviour jsonl files: accuracy per (model, task, format, steps) with Wilson CIs.

    python -m hcot.report results/p1/*.jsonl > results/p1/table.md
"""

import json
import math
import sys
from collections import Counter, defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load(paths):
    rows = []
    for p in paths:
        name = p.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
        for line in open(p):
            r = json.loads(line)
            r["run"] = name
            rows.append(r)
    return rows


def table(rows, key="correct"):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["run"], r["task"], r["fmt"], r.get("filler", 0), r["n"])].append(r)
    runs = sorted({c[0] for c in cells})
    out = []
    for run in runs:
        for task in ["T1", "T2", "T3", "T4"]:
            for fmt in ["nl", "sym"]:
                for fill in sorted({c[3] for c in cells if c[0] == run}):
                    ns = sorted(c[4] for c in cells if c[:4] == (run, task, fmt, fill))
                    if not ns:
                        continue
                    parts = []
                    for n in ns:
                        rs = cells[(run, task, fmt, fill, n)]
                        k = sum(bool(r.get(key)) for r in rs)
                        lo, hi = wilson(k, len(rs))
                        pk = "cpred" if key == "ccorrect" else "pred"
                        top, cnt = Counter(str(r.get(pk)).lower() for r in rs).most_common(1)[0]
                        flag = "*" if cnt / len(rs) >= 0.8 else ""  # one answer for >= 80% of items
                        parts.append(f"{n}: {k / len(rs):.2f} [{lo:.2f},{hi:.2f}]{flag}")
                    leak = sum(r.get("leaked", False) for c in cells if c[:4] == (run, task, fmt, fill)
                               for r in cells[c])
                    out.append(f"| {run} | {task} | {fmt} | {fill} | " + " | ".join(parts) + f" | {leak} |")
    return "\n".join(out)


if __name__ == "__main__":
    rows = load(sys.argv[1:])
    print("Greedy exact match:\n")
    print(table(rows, "correct"))
    if any("ccorrect" in r for r in rows):
        print("\nConstrained (argmax over answer candidates):\n")
        print(table([r for r in rows if "ccorrect" in r], "ccorrect"))
