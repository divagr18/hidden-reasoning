"""P2 mechanism tools for Qwen3.5-4B (weights read-only): spans, attention knockout, DeltaNet
state capture and patching. Everything runs at batch 1 on the text model directly.

    ex = Exp(model, tok)
    sp = ex.spans(item)                       # token ids and the steps / question positions
    ex.logits(sp)                             # clean answer-position logits
    ex.logits(sp, ko=(sp.user0, sp.q0))       # full-attention rows >= q0 cannot see [user0, q0)
    st = ex.states(sp)                        # DeltaNet states after the last step token
    ex.patched(spA, stB, layers, heads)       # A's run with B's states in chosen heads

Positions: ``user0`` first token of the user content, ``s0``/``s1`` the steps span, ``q0`` first
token after the steps (filler or question), ``T`` the prompt length (the answer is predicted at
T - 1).
"""

from dataclasses import dataclass

import torch

from hcot import tasks
from hcot.models import chat


@dataclass
class Spans:
    ids: list[int]
    user0: int
    s0: int
    s1: int
    q0: int
    item: dict

    @property
    def T(self) -> int:
        return len(self.ids)


class Exp:
    def __init__(self, model, tok, name="q35"):
        self.model, self.tok, self.name = model, tok, name
        self.lm = getattr(model.model, "language_model", model.model)  # Qwen3.5 VL or a CausalLM
        self.head = model.lm_head
        self.cfg = self.lm.config
        self.full = [i for i, t in enumerate(self.cfg.layer_types) if t == "full_attention"]
        self.lin = [i for i, t in enumerate(self.cfg.layer_types) if t == "linear_attention"]
        self.dev = next(model.parameters()).device
        self._ko_layers = None
        self._ko_mask = None
        self._hooks = [self.lm.layers[i].self_attn.register_forward_pre_hook(self._hook, with_kwargs=True)
                       for i in self.full]
        self._cands = {}

    # --- spans --------------------------------------------------------------------------------

    def spans(self, it: dict) -> Spans:
        user = tasks.user_text(it)
        text = chat(self.tok, self.name, user, think=False)
        u = text.index(user)
        c0, c1 = u + len(it["prefix"]), u + len(it["prefix"]) + len(it["steps"])
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        off = enc["offset_mapping"]
        user0 = next(i for i, (a, b) in enumerate(off) if b > u)
        s0 = next(i for i, (a, b) in enumerate(off) if b > c0)
        q0 = next(i for i, (a, b) in enumerate(off) if a >= c1)
        return Spans(enc["input_ids"], user0, s0, q0, q0, it)

    def cand_ids(self, cands):
        key = tuple(cands)
        if key not in self._cands:
            from hcot.behave import cand_tokens

            self._cands[key] = [sorted(s) for s in cand_tokens(self.tok, cands)]
        return self._cands[key]

    # --- knockout ---------------------------------------------------------------------------

    def _hook(self, module, args, kwargs):
        if self._ko_mask is not None and module.layer_idx in self._ko_layers:
            kwargs["attention_mask"] = self._ko_mask
        return args, kwargs

    def _mask(self, T, past, ko):
        """Bool [1, 1, T, past + T]: causal, minus rows >= ko[2] seeing cols in [ko[0], ko[1])."""
        q = torch.arange(past, past + T, device=self.dev)[:, None]
        k = torch.arange(past + T, device=self.dev)[None, :]
        m = k <= q
        if ko is not None:
            a, b, r = ko
            m &= ~((q >= r) & (k >= a) & (k < b))
        return m[None, None]

    def forward(self, ids, cache=None, ko=None, ko_layers=None, hidden=False):
        """ids: list or tensor of new tokens; ko = (col0, col1, row0) in absolute positions."""
        x = torch.as_tensor(ids, device=self.dev)[None] if not torch.is_tensor(ids) else ids
        past = cache.get_seq_length() if cache is not None else 0
        T = x.shape[1]
        mask = self._mask(T, past, None)
        self._ko_layers = set(self.full if ko_layers is None else ko_layers)
        self._ko_mask = self._mask(T, past, ko) if ko is not None else None
        try:
            out = self.lm(input_ids=x, attention_mask={"full_attention": mask, "linear_attention": None},
                          past_key_values=cache, use_cache=cache is not None or True,
                          output_hidden_states=hidden)
        finally:
            self._ko_mask = None
        return out

    def answer(self, h_last, cands):
        """(full-vocab argmax id, constrained choice index, candidate scores) from the last hidden."""
        logit = self.head(h_last).float()
        sets = self.cand_ids(cands)
        sc = torch.stack([logit[s].max() for s in sets])
        return int(logit.argmax()), int(sc.argmax()), sc

    @torch.no_grad()
    def logits(self, sp: Spans, ko=None, ko_layers=None):
        """Constrained scores at the answer position. ko = (col0, col1): masked for rows >= q0."""
        k = None if ko is None else (ko[0], ko[1], sp.q0)
        out = self.forward(sp.ids, ko=k, ko_layers=ko_layers)
        return self.answer(out.last_hidden_state[0, -1], sp.item["cands"])

    # --- states -------------------------------------------------------------------------------

    @torch.no_grad()
    def prefix(self, sp: Spans, hidden=False):
        """Run the prompt up to (excluding) q0; returns (cache, output)."""
        from transformers import DynamicCache

        cache = DynamicCache(config=self.cfg)
        out = self.forward(sp.ids[: sp.q0], cache=cache, hidden=hidden)
        return out.past_key_values, out

    @staticmethod
    def get_states(cache, layers):
        return {L: (cache.layers[L].recurrent_states[0].clone(), cache.layers[L].conv_states[0].clone())
                for L in layers}

    @torch.no_grad()
    def patched(self, spA: Spans, statesB: dict, heads=None, conv=True, ko=None):
        """A's prompt with DeltaNet states from B written in after the steps.

        statesB: {layer: (recurrent [1, H, 128, 128], conv)}; heads: {layer: list of value heads}
        (None = every head of the layers in statesB). ko masks full attention as in ``logits``."""
        cache, _ = self.prefix(spA)
        for L, (rec, cv) in statesB.items():
            R = cache.layers[L].recurrent_states[0]
            hs = None if heads is None else heads.get(L)
            if hs is None:
                R.copy_(rec)
                if conv:
                    cache.layers[L].conv_states[0].copy_(cv)
            else:
                R[:, hs] = rec[:, hs]
        k = None if ko is None else (ko[0], ko[1], spA.q0)
        out = self.forward(spA.ids[spA.q0 :], cache=cache, ko=k)
        return self.answer(out.last_hidden_state[0, -1], spA.item["cands"])
