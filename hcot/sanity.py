"""Sanity checks for the P2 tooling: cache continuation == full forward; patching changes things."""
import torch

from hcot import tasks
from hcot.mech import Exp
from hcot.models import load

model, tok = load("q35")
ex = Exp(model, tok)
for it, jt in zip(tasks.make("T1", 8, "nl", 3, seed=99), tasks.make("T1", 8, "nl", 3, seed=98)):
    sp, sq = ex.spans(it), ex.spans(jt)
    full = ex.forward(sp.ids).last_hidden_state[0, -1]
    cache, _ = ex.prefix(sp)
    cont = ex.forward(sp.ids[sp.q0:], cache=cache).last_hidden_state[0, -1]
    print("cache continuation vs full: max |dh| =", (full - cont).abs().max().item(), "| |h| =", full.abs().max().item())
    own = ex.get_states(ex.prefix(sp)[0], ex.lin)
    _, c_own, s_own = ex.patched(sp, own)
    _, c_cl, s_cl = ex.logits(sp)
    zero = {L: (torch.zeros_like(r), torch.zeros_like(c)) for L, (r, c) in own.items()}
    _, c_z, s_z = ex.patched(sp, zero)
    other = ex.get_states(ex.prefix(sq)[0], ex.lin)
    _, c_o, s_o = ex.patched(sp, other)
    print("scores clean", s_cl.tolist()[:4], "\n  own-patch", s_own.tolist()[:4], "\n  zero-patch", s_z.tolist()[:4],
          "\n  other-prompt patch", s_o.tolist()[:4], "| choices", c_cl, c_own, c_z, c_o, "answer", it["answer"])
