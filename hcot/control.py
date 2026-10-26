"""Follow-up step 0: positive control for the P2 detector on a known hidden-state tracker.

The compiled beta=2 T2 tracker (hcot/p3.py) carries the parity bit only in DeltaNet state.
Wired as a model whose answer is read from the final residual (after every layer, including the
full-attention layer), the P2 detector should say:
  knockout (full attention can't see initial state + steps from the question on) -> intact
  state swap (B's DeltaNet states written into A after the steps) -> answer follows B

    python -m hcot.control --n 50 --steps 64
"""

import argparse

import torch
from transformers import AutoTokenizer, DynamicCache

from hcot import p3, tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--steps", type=int, default=64)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    pairs = tasks.make_pairs("T2", a.steps, "sym", a.n, seed=3)

    def enc(it):
        pre = tok(it["prefix"], add_special_tokens=False)["input_ids"]
        steps = tok(it["prefix"] + it["steps"], add_special_tokens=False)["input_ids"]
        full = tok(tasks.user_text(it), add_special_tokens=False)["input_ids"]
        assert full[: len(steps)] == steps
        return full, len(pre), len(steps)  # ids, first step token, first question token

    data = [(enc(x), enc(y), x, y) for x, y in pairs]
    vocab = sorted({t for (ea, eb, *_) in data for t in ea[0] + eb[0]})
    strings = {t: tok.decode([t]) for t in vocab}
    p3.REFL.clear()
    c, readout = p3.build_circuit("T2", "sym", tok, strings)
    model, lay = p3.build_model(c, vocab)
    model.eval()
    remap = {t: i for i, t in enumerate(vocab)}
    col = lay.dim(next(iter(readout["bit"][0].terms)))
    full_layers = [i for i, t in enumerate(model.config.layer_types) if t == "full_attention"]
    lm = model.model
    final = {}
    lm.norm.register_forward_hook(lambda m, inp, out: final.__setitem__("h", inp[0]))
    ko = {}

    def pre_hook(module, args, kwargs):
        if "mask" in ko:
            kwargs["attention_mask"] = ko["mask"]
        return args, kwargs

    for L in full_layers:
        lm.layers[L].self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def mask(T, past, rows_from=None, cols=None):
        q = torch.arange(past, past + T)[:, None]
        k = torch.arange(past + T)[None, :]
        m = k <= q
        if rows_from is not None:
            m &= ~((q >= rows_from) & (k >= cols[0]) & (k < cols[1]))
        return m[None, None]

    def run(ids, cache=None, knock=None):
        x = torch.tensor([[remap[t] for t in ids]])
        past = cache.get_seq_length() if cache is not None else 0
        if knock is not None:
            ko["mask"] = mask(len(ids), past, *knock)
        try:
            with torch.no_grad():
                model(input_ids=x, past_key_values=cache, use_cache=True,
                      attention_mask={"full_attention": mask(len(ids), past), "linear_attention": None})
        finally:
            ko.pop("mask", None)
        return float(final["h"][0, -1, col])

    bit = lambda r: int(r < 0)  # noqa: E731  (+1 = 0/off, -1 = 1/on)
    res = dict(clean=0, ko=0, swap_toB=0, swap_staysA=0, n=0)
    for (ea, eb, x, y) in data:
        ids_a, s0, q0 = ea
        ids_b, _, q0b = eb
        assert q0 == q0b
        want_a, want_b = int(x["answer"]), int(y["answer"])
        res["clean"] += bit(run(ids_a)) == want_a
        res["ko"] += bit(run(ids_a, knock=(q0, (0, q0)))) == want_a
        ca = DynamicCache(config=model.config)
        run(ids_a[:q0], ca)
        cb = DynamicCache(config=model.config)
        run(ids_b[:q0], cb)
        for L in range(c.n_layers()):
            ca.layers[L].recurrent_states[0].copy_(cb.layers[L].recurrent_states[0])
            ca.layers[L].conv_states[0].copy_(cb.layers[L].conv_states[0])
        r = bit(run(ids_a[q0:], ca))
        res["swap_toB"] += r == want_b
        res["swap_staysA"] += r == want_a
        res["n"] += 1
    print({k: (v / res["n"] if k != "n" else v) for k, v in res.items()}, "steps", a.steps)


if __name__ == "__main__":
    main()
