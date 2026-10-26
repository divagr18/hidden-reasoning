"""P1 behaviour runs: answer-only (thinking off, greedy) and a CoT-on ceiling.

    python -m hcot.behave --model q35 --mode answer --n 200 --out results/p1/q35_answer.jsonl
    python -m hcot.behave --model q35 --mode cot --n 30 --lengths 4 16 64 --out results/p1/q35_cot.jsonl

One jsonl row per item. Answer mode greedily decodes up to 8 tokens and also records the
constrained choice (argmax over the candidate answers' first tokens at the first reply position).
Cells already in the output file are skipped, so a run can resume.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from hcot import tasks
from hcot.models import chat, load


def norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(the )?(final )?(answer|value|bit)( is)?[:\s]*", "", s)
    s = s.strip(" \t\n.,;:!'\"*`()[]")
    s = re.sub(r"^the ", "", s)
    return s


def parse_answer(text: str, mode: str) -> tuple[str, bool]:
    """(normalized answer, leaked) — leaked means the reply holds more than an answer."""
    if mode == "cot":
        tail = text.split("</think>")[-1] if "</think>" in text else text
        m = re.findall(r"answer\s*[:：]\s*\**\s*([A-Za-z0-9]+)", tail, flags=re.I)
        if m:
            return norm(m[-1]), False
        lines = [x for x in tail.strip().splitlines() if x.strip()]
        return (norm(lines[-1].split()[-1]) if lines and lines[-1].split() else ""), False
    t = text.replace("<think>", "").replace("</think>", "").strip()
    first = t.splitlines()[0] if t else ""
    leaked = len(first.split()) > 3 or len(t.splitlines()) > 1
    return norm(first), leaked


def cand_tokens(tok, cands: list[str]) -> list[set[int]]:
    sets = []
    for c in cands:
        s = set()
        for v in (c, " " + c, "\n" + c, c.capitalize(), " " + c.capitalize(), c.upper(), " " + c.upper()):
            ids = tok(v, add_special_tokens=False)["input_ids"]
            if ids:
                s.add(ids[0])
        sets.append(s)
    shared = set()
    for i in range(len(sets)):
        for j in range(i):
            shared |= sets[i] & sets[j]
    return [s - shared for s in sets]  # a first token shared by two answers decides nothing


def run_cell(model, tok, name, items, mode, max_new, tok_budget, max_batch):
    texts = [chat(tok, name, tasks.user_text(it), think=(mode == "cot")) for it in items]
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    csets = cand_tokens(tok, items[0]["cands"])
    order = sorted(range(len(items)), key=lambda i: -len(enc[i]))
    out = [None] * len(items)
    dev = next(model.parameters()).device
    i = 0
    while i < len(order):
        L = len(enc[order[i]])
        per = L + max_new
        bs = max(1, min(max_batch, tok_budget // per))
        idx = order[i : i + bs]
        i += bs
        ids = [enc[j] for j in idx]
        m = max(len(x) for x in ids)
        pad = tok.pad_token_id
        inp = torch.tensor([[pad] * (m - len(x)) + x for x in ids], device=dev)
        att = torch.tensor([[0] * (m - len(x)) + [1] * len(x) for x in ids], device=dev)
        with torch.no_grad():
            g = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new,
                               do_sample=False, output_scores=(mode == "answer"),
                               return_dict_in_generate=True, pad_token_id=pad,
                               temperature=None, top_p=None, top_k=None)
        seqs = g.sequences[:, m:]
        for b, j in enumerate(idx):
            gen = seqs[b].tolist()
            eos = [k for k, t in enumerate(gen) if t in (tok.eos_token_id, pad)]
            n_gen = eos[0] if eos else len(gen)
            text = tok.decode(gen[:n_gen], skip_special_tokens=False)
            ans, leaked = parse_answer(text, mode)
            it = items[j]
            row = dict(task=it["task"], fmt=it["fmt"], n=it["n"], idx=it["idx"], filler=it["filler"],
                       answer=it["answer"], output=text[:4000] if mode == "cot" else text,
                       pred=ans, correct=ans == it["answer"].lower(), leaked=leaked,
                       n_gen=n_gen, n_prompt=len(enc[j]), truncated=(not eos))
            if mode == "answer":
                # score the candidates where the greedy path emits its first non-blank token
                # (Qwen writes " 5" as " " + "5", so step 0 can be a bare space)
                step = next((k for k in range(min(3, len(g.scores)))
                             if tok.decode([gen[k]]).strip()), 0)
                logit = g.scores[step][b].float()
                row["cstep"] = step
                sc = [max(logit[t].item() for t in s) if s else -1e9 for s in csets]
                k = max(range(len(sc)), key=lambda q: sc[q])
                row["cpred"] = items[0]["cands"][k]
                row["ccorrect"] = row["cpred"] == it["answer"]
                row["top1_in_cands"] = int(logit.argmax()) in set().union(*csets)
            out[j] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="answer", choices=["answer", "cot"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tasks", nargs="*", default=tasks.TASKS)
    ap.add_argument("--formats", nargs="*", default=tasks.FORMATS)
    ap.add_argument("--lengths", nargs="*", type=int, default=tasks.LENGTHS)
    ap.add_argument("--filler", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=None)
    ap.add_argument("--tok-budget", type=int, default=60000)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--beta-mode", default="stock")
    ap.add_argument("--seed", type=int, default=0, help="item seed (99 = the checkpoint-selection set)")
    a = ap.parse_args()
    max_new = a.max_new or (8 if a.mode == "answer" else 4096)
    model, tok = load(a.model, adapter=a.adapter, beta_mode=a.beta_mode)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            r = json.loads(line)
            done.add((r["task"], r["fmt"], r["n"], r["filler"]))
    for t in a.tasks:
        for f in a.formats:
            for n in a.lengths:
                if (t, f, n, a.filler) in done:
                    continue
                items = tasks.make(t, n, f, a.n, seed=a.seed, filler=a.filler)
                t0 = time.time()
                rows = run_cell(model, tok, a.model, items, a.mode, max_new, a.tok_budget, a.max_batch)
                with out.open("a") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                acc = sum(r["correct"] for r in rows) / len(rows)
                extra = ""
                if a.mode == "answer":
                    extra = (f" constrained={sum(r['ccorrect'] for r in rows) / len(rows):.3f}"
                             f" leak={sum(r['leaked'] for r in rows)}")
                else:
                    extra = f" trunc={sum(r['truncated'] for r in rows)} gen={sum(r['n_gen'] for r in rows) / len(rows):.0f}"
                print(f"{a.model} {a.mode} {t} {f} n={n} acc={acc:.3f}{extra} "
                      f"len={rows[0]['n_prompt']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
