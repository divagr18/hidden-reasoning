"""Re-check: is T2 parity really linearly decodable from Qwen3.5's layer-0 DeltaNet state?

Targets on the state at the last step token (unseen test sequences):
  answer   init XOR parity(#T)            parity  parity(#T) alone
  init     the initial bit alone          mod3    #T mod 3
  shuffled the answer with labels permuted (null)
Also: the same probe after re-ordering each sequence's steps at random (same #T, same init), to
see whether the decodable thing depends on the count only.
"""
import math
import random
import sys

import torch

from hcot import tasks
from hcot.mech import Exp
from hcot.mech_run import ridge_probe
from hcot.models import load

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 16
fmt = sys.argv[2] if len(sys.argv) > 2 else "sym"
N = 1300
model, tok = load("q35")
ex = Exp(model, tok)
rng = random.Random(5)
lats = [tasks.SAMPLE["T2"](rng, n_steps) for _ in range(N)]
gen = torch.Generator(device=ex.dev).manual_seed(0)
R = torch.randn(128 * 128, 2048, device=ex.dev, generator=gen) / math.sqrt(2048)


@torch.no_grad()
def feats(lats, layers=(0, 1, 2)):
    out = {L: [] for L in layers}
    for lat in lats:
        it = tasks._item("T2", lat, fmt, n_steps, 0, 0)
        sp = ex.spans(it)
        cache, _ = ex.prefix(sp)
        for L in layers:
            S = cache.layers[L].recurrent_states[0][0].float()
            out[L].append((S.reshape(S.shape[0], -1) @ R).half())
    return {L: torch.stack(v).transpose(0, 1).float() for L, v in out.items()}  # [H, N, 2048]


X = feats(lats)
cnt = torch.tensor([sum(l["steps"]) for l in lats], device=ex.dev)
init = torch.tensor([l["init"] for l in lats], device=ex.dev)
tg = {"answer": init ^ (cnt % 2), "parity": cnt % 2, "init": init, "mod3": cnt % 3,
      "shuffled": (init ^ (cnt % 2))[torch.randperm(N, generator=torch.Generator().manual_seed(1)).to(ex.dev)]}
ntr = 1000
for name, y in tg.items():
    for L in X:
        te, va = ridge_probe(X[L][:, :ntr], y[:ntr], X[L][:, ntr:], y[ntr:], int(y.max()) + 1)
        h = max(range(len(va)), key=lambda i: va[i])
        print(f"n={n_steps} {fmt} target={name:8s} layer {L}: val-picked head {h} test {te[h]:.3f} (best test {max(te):.3f})", flush=True)
# order control: same multiset of steps, shuffled order -> features of the shuffled sequences,
# probe trained on the originals
lats2 = []
for l in lats[ntr:]:
    s = list(l["steps"]); rng.shuffle(s); lats2.append(dict(l, steps=s))
X2 = feats(lats2, layers=(0,))
y = tg["answer"]
te, va = ridge_probe(X[0][:, :ntr], y[:ntr], X2[0], y[ntr:], 2)
h = max(range(len(va)), key=lambda i: va[i])
print(f"order-shuffled test sequences, layer 0 head {h}: {te[h]:.3f}")
