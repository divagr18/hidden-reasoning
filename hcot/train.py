"""Follow-up: answer-only LoRA fine-tuning on a state-tracking task (no CoT anywhere).

    python -m hcot.train --model olmoh --task T2 --minutes 40 --out adapters/olmoh_T2

The prompt is the same chat prompt as P1 (thinking off, reply prefilled with "Answer:"); the loss
is only on the answer tokens. A time-based curriculum raises the maximum length 4 -> 64 steps.
LoRA (r=16) on every attention / DeltaNet / MLP projection of the language model; the DeltaNet
beta and gate-decay projections (tiny: hidden -> n_heads) and the per-head decay parameters
(A_log, dt_bias) are trained in full, so a beta <= 2 model can push beta above 1.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from hcot import tasks
from hcot.models import MODELS, chat

TARGETS = {
    "olmoh": dict(lora=r".*\.(q_proj|k_proj|v_proj|o_proj|g_proj|gate_proj|up_proj|down_proj)$",
                  full=("b_proj", "a_proj", "A_log", "dt_bias")),
    "olmo3": dict(lora=r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$", full=()),
    "q35": dict(lora=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|"
                     r"gate_proj|up_proj|down_proj)$",
                full=("in_proj_b", "in_proj_a", "A_log", "dt_bias")),
}


def max_len_at(frac):
    for f, L in ((0.10, 4), (0.20, 8), (0.35, 16), (0.55, 32)):
        if frac < f:
            return L
    return 64


def sample(task, fmt, rng, frac):
    hi = max_len_at(frac)
    n = rng.randint(2, hi) if rng.random() < 0.5 else hi  # half at the current max
    lat = tasks.SAMPLE[task](rng, n)
    it = tasks._item(task, lat, fmt, n, 0, 0)
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="T2")
    ap.add_argument("--fmt", default="sym")
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--steps", type=int, default=0, help="fixed step budget (overrides the time budget)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta-mode", default="stock")
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    spec = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(spec["id"])
    cls = AutoModelForImageTextToText if spec["kind"] == "vl" else AutoModelForCausalLM
    torch.manual_seed(a.seed)
    model = cls.from_pretrained(spec["id"], dtype=torch.bfloat16).cuda()
    from hcot import betamode

    betamode.apply(model, a.model, a.beta_mode)
    tg = TARGETS[a.model]
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=tg["lora"],
                     modules_to_save=[x for x in ("b_proj", "a_proj", "in_proj_b", "in_proj_a") if x in tg["full"]] or None)
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():
        if n.split(".")[-1] in ("A_log", "dt_bias") and "visual" not in n:
            p.requires_grad_(True)
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("trainable params", sum(p.numel() for p in trainable), flush=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    rng = random.Random(12345 + 1000 * a.seed)  # seed 0 = the original runs
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0 = time.time()
    step, log, hist, best_len = 0, [], [], 0
    model.train()
    while (el := time.time() - t0) < a.minutes * 60 and (not a.steps or step < a.steps):
        # curriculum and LR schedule follow the step budget when one is given (so runs sharing a
        # GPU get the same recipe as the pilot), otherwise wall-clock time
        frac = step / a.steps if a.steps else el / (a.minutes * 60)
        lr = a.lr * min(1.0, (step + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = [sample(a.task, a.fmt, rng, frac) for _ in range(a.bs)]
        seqs, labs = [], []
        for it in batch:
            p = tok(chat(tok, a.model, tasks.user_text(it), think=False), add_special_tokens=False)["input_ids"]
            t = tok(" " + it["answer"], add_special_tokens=False)["input_ids"]
            seqs.append(p + t)
            labs.append([-100] * len(p) + t)
        m = max(map(len, seqs))
        ids = torch.tensor([s + [pad] * (m - len(s)) for s in seqs]).cuda()
        att = torch.tensor([[1] * len(s) + [0] * (m - len(s)) for s in seqs]).cuda()
        lab = torch.tensor([l + [-100] * (m - len(l)) for l in labs]).cuda()
        out = model(input_ids=ids, attention_mask=att)
        logits = out.logits[:, :-1].float()
        tgt = lab[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                                 ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        with torch.no_grad():
            pred = logits.argmax(-1)
            ok = ((pred == tgt) | (tgt == -100)).all(1).float().mean().item()
        hist.append((loss.item(), ok))
        step += 1
        if step % 25 == 0:
            l = sum(h[0] for h in hist) / len(hist)
            acc = sum(h[1] for h in hist) / len(hist)
            hist = []
            log.append(dict(step=step, minutes=round(el / 60, 1), max_len=max_len_at(frac), loss=round(l, 4),
                            train_acc=round(acc, 3), lr=lr))
            print(log[-1], flush=True)
            # keep the last checkpoint that solved the training distribution at the highest length
            # reached so far (the pilot learned 32 steps perfectly, then collapsed)
            if acc >= 0.98 and max_len_at(frac) >= best_len:
                best_len = max_len_at(frac)
                save(model, a.out, dict(vars(a), steps=step, best_max_len=best_len, log=log))
                print("checkpoint at max_len", best_len, flush=True)
    save(model, a.out + "_final", dict(vars(a), steps=step, log=log))
    if best_len == 0:
        save(model, a.out, dict(vars(a), steps=step, best_max_len=0, log=log))
    print("done", a.out, "steps", step, "best max_len", best_len, flush=True)


def save(model, out, meta):
    Path(out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    # per-head decay parameters are not in the LoRA state dict: save them separately
    extra = {n: p.detach().cpu() for n, p in model.named_parameters()
             if n.split(".")[-1] in ("A_log", "dt_bias") and p.requires_grad}
    torch.save(extra, Path(out) / "decay_params.pt")
    json.dump(meta, open(Path(out) / "train_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
