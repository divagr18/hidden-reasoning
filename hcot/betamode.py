"""Change only the DeltaNet beta range of a loaded model (follow-up beta ablation).

  olmoh  cap_min  beta = min(2*sigmoid(b), 1): eigenvalues capped to [0, 1]; heads/tokens whose
                  stock beta is <= 1 are untouched (the minimal intervention)
  olmoh  cap_sig  beta = sigmoid(b) (the model's own linear_allow_neg_eigval=False switch): every
                  beta halved
  q35    x2       beta = 2*sigmoid(b): the reverse, Qwen3.5 given Olmo-Hybrid's range

The wrappers sit on the HF modules' gated-delta-rule functions (looked up at call time by the
DeltaNet forward), so they cover prefill and decode, training and eval.
"""

MODES = {"olmoh": ("stock", "cap_min", "cap_sig"), "q35": ("stock", "x2"), "olmo3": ("stock",)}
_installed = {}
ACTIVE = {"on": True}  # basecost.py flips this to compare against stock in one process
_capsig_modules = []


def set_active(on: bool):
    ACTIVE["on"] = on
    for mod in _capsig_modules:
        mod.allow_neg_eigval = not on


def _wrap(mod, fn):
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        orig = getattr(mod, name)

        def w(*a, _orig=orig, **kw):
            if ACTIVE["on"]:
                kw["beta"] = fn(kw["beta"])
            return _orig(*a, **kw)

        setattr(mod, name, w)


def apply(model, name, mode):
    if mode in (None, "stock"):
        return
    assert mode in MODES[name], (name, mode)
    if (name, mode) in _installed:
        return
    if name == "olmoh" and mode == "cap_min":
        from transformers.models.olmo_hybrid import modeling_olmo_hybrid as m

        _wrap(m, lambda b: b.clamp(max=1.0))
    elif name == "olmoh" and mode == "cap_sig":
        n = 0
        for mod in model.modules():
            if hasattr(mod, "allow_neg_eigval"):
                mod.allow_neg_eigval = False
                _capsig_modules.append(mod)
                n += 1
        assert n > 0
    elif name == "q35" and mode == "x2":
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m

        _wrap(m, lambda b: b * 2.0)
    _installed[(name, mode)] = True
