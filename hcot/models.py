"""Model registry and chat formatting for the behaviour runs (all weights read-only)."""

import torch

MODELS = {
    "q35": dict(id="Qwen/Qwen3.5-4B", kind="vl"),
    "q3": dict(id="Qwen/Qwen3-4B", kind="causal"),
    "rwkv7": dict(id="fla-hub/rwkv7-2.9B-g1", kind="remote"),
    "falconh1": dict(id="tiiuae/Falcon-H1-3B-Instruct", kind="causal"),
    # matched pair from one lab, same post-training stage: GDN hybrid with beta in (0, 2) vs dense
    "olmoh": dict(id="allenai/Olmo-Hybrid-Instruct-DPO-7B", kind="causal"),
    "olmo3": dict(id="allenai/Olmo-3-7B-Instruct-DPO", kind="causal"),
}


def load(name: str, device: str = "cuda", attn: str | None = None, adapter: str | None = None,
         beta_mode: str = "stock"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODELS[name]
    remote = spec["kind"] == "remote"
    tok = AutoTokenizer.from_pretrained(spec["id"], trust_remote_code=remote)
    kw = dict(dtype=torch.bfloat16, trust_remote_code=remote)
    if attn:
        kw["attn_implementation"] = attn
    if spec["kind"] == "vl":
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(spec["id"], **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(spec["id"], **kw)
    if adapter:  # follow-up fine-tuning: LoRA + fully trained beta/decay projections, merged
        from pathlib import Path

        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        extra = Path(adapter) / "decay_params.pt"
        if extra.exists():
            own = dict(model.named_parameters())
            for n, t in torch.load(extra).items():
                key = n.replace("base_model.model.", "", 1)
                own[key].data.copy_(t.to(own[key].dtype))
    from hcot import betamode

    betamode.apply(model, name, beta_mode)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


COT_SUFFIX = ("\nThink it through step by step, then write the final answer on the last line as "
              "'Answer: <answer>'.")


ANSWER_PREFIX = "Answer:"  # forced start of the reply in answer-only mode: no room for reasoning


def chat(tok, name: str, user: str, think: bool) -> str:
    """The templated prompt, ending where the assistant's reply starts (answer-only mode also
    prefills ANSWER_PREFIX, so the next token has to be the answer)."""
    msgs = [{"role": "user", "content": user}]
    if name in ("q35", "q3", "rwkv7"):
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=think)
    else:
        if think:
            msgs = [{"role": "user", "content": user + COT_SUFFIX}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if not think:
        s += (" " if s.endswith("</think>") else "") + ANSWER_PREFIX
    return s
