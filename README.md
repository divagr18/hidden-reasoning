# Hybrid models can hide reasoning in their RNN state

Code, raw results and lab notes for the post
[Hybrid models can hide reasoning in their RNN state](https://divagr.com/blog/hybrid-models-can-hide-reasoning-in-their-rnn-state).

It builds on [Qwenputer](https://divagr.com/blog/qwenputer), where I compiled a working RISC-V CPU
into Qwen 3.5's weights, using its GDN layers as RAM.
- The same compiler builds the exact trackers used here as an answer key, and the tracker planted
  in a real Qwen 3.5.
- The companion post on
  [compiled, in-weight monitoring tools](https://divagr.com/blog/building-oversight-tools-into-a-models-weights)
  uses the same setup.

## The question

CoT monitoring rests on a bet: a model can only do so much serial work per token, so hard
reasoning has to show up in its chain of thought.

Hybrid models weaken that bet. Qwen 3.5 and OLMo-Hybrid mix attention with Gated DeltaNet (GDN)
layers, a linear RNN that carries state from token to token
([interactive explainer of GDN](https://divagr.com/attention-lab/delta)). Can they do serial
reasoning in that state instead of writing it down, where a CoT monitor can't see it?

## Findings

The task is parity: track a switch through many flips, then answer on or off with no reasoning
allowed. It's the simplest task that needs state tracking.

- **Out of the box, no model tracks it** past 4–8 flips. With CoT on, Qwen 3.5 solves it.
- **Under answer-only fine-tuning, OLMo-Hybrid learns to track it silently** in its GDN state, and
  still gets it right at twice the training length. Qwen 3.5 and the dense OLMo-3 only fit the
  lengths they were trained on.
- **What decides it is β, the GDN's write strength.**
  - OLMo-Hybrid allows β up to 2, which gives the state update negative eigenvalues, so it can flip
    a sign every token. Qwen 3.5 caps β at 1.
  - Capping OLMo-Hybrid's β at 1, with nothing else changed, stops the silent tracking from
    generalising.
  - This matches the theory of Grazzi, Siems et al. (ICLR 2025).

## Results

**Base models.** I tested five models: Qwen3.5-4B, Qwen3-4B, Olmo-Hybrid-7B, Olmo-3-7B and
RWKV-7 2.9B.

**Fine-tuning.**
- Each run trains a LoRA, with the β and decay projections also unfrozen.
- Targets are answer-only (no CoT anywhere), on a curriculum up to 64 flips, so longer lengths
  test generalisation.
- The table gives accuracy as the mean over seeds, with the range in brackets. Each cell has
  n = 100, and chance is 0.50.

| model | seeds | 16 flips | 64 | **128** | 192 | 256 |
|---|---|---|---|---|---|---|
| Olmo-Hybrid, β ≤ 2 (stock) | 6 | 1.00 | 0.96 | **0.89** (0.53–1.00) | 0.71 | 0.54 |
| Olmo-Hybrid, β capped at 1 | 6 | 0.91 | 0.73 | **0.52** (0.49–0.56) | 0.50 | 0.55 |
| Qwen3.5-4B, β ≤ 1 (stock) | 3 | 0.99 | 0.80 | **0.55** | 0.48 | 0.58 |
| Olmo-3-7B, dense | 3 | 0.96 | 0.60 | **0.54** | 0.52 | 0.57 |

- **Uncapped Olmo-Hybrid.** 5 of the 6 runs fit the training range (≥ 0.85 at 64 steps), and all 5
  still get 128 steps right (0.89–1.00).
- **Capped Olmo-Hybrid.** No capped run generalises. Over all runs that's 5/6 against 0/6 (Fisher
  exact, one-sided, p ≈ 0.008).
- **What the cap costs.** On plain text it raises perplexity by about 3.5%.
- **Where the bit lives.**
  - Swapping the DeltaNet state between counterfactual sequences moves the answer 0.00 of the
    time, because attention keeps a backup copy of the bit.
  - Swapping the state *and* knocking out attention moves it 1.00 of the time.
  - The bit lives in layer 16 head 2 and layer 17 head 4. A monitor that only reads the recurrent
    state would miss it.
- **The constructive bound.**
  - Compiled β = 2 trackers in stock GDN layers track parity, 3-cup and 5-box permutations exactly
    over 100k tokens. With stock β the same weights fail.
  - A compiled tracker planted in a real Qwen3.5-4B was ranked 1st of 864 heads by the blind
    detector.

**Limits.**
- The task is one bit of state tracking (parity).
- A four-switch version was learned by neither β arm at 64 steps, so it can't separate them.
- Answer-only fine-tuning stands in for training pressure that pushes models to skip their working.
  It isn't a real RL setup.

`docs/RESULTS.md` is the full lab notebook: every table, the detector controls, the planted model
and per-box compute. `docs/PLAN.md` is the plan as written before the runs.

## Layout

```
hcot/
  tasks.py        T1 last-write-wins, T2 parity, T3 3-cup S3, T4 5-box S5, T5 multi-switch; counterfactual pairs
  models.py       model registry and chat formatting
  behave.py       answer-only and CoT behaviour evals            (P1)
  mech.py, mech_run.py, sanity.py, diag_parity.py
                  attention knockout, DeltaNet state swap, probes (P2)
  p3.py           compiled β = 2 trackers and the 100k-token verifier (P3)
  control.py      positive control for the detector on the compiled tracker
  train.py        answer-only LoRA fine-tuning with a length curriculum
  betamode.py     change only the DeltaNet β range (stock, cap_min, cap_sig)
  beta_use.py     does a fine-tuned head use β > 1 on state-changing tokens
  basecost.py     perplexity and KL cost of a β-range change
  plant.py        compiled tracker planted in a real Qwen3.5-4B, then the blind head scan
  seeds.py, report.py, p2report.py, compact.py, plot.py
                  tables and the figure
  compiler/       the Gated DeltaNet compiler, vendored from chipformer
scripts/          the shell scripts the GPU boxes ran
results/          every raw output: jsonl per eval, json per experiment, tables, remote logs
docs/             PLAN.md and RESULTS.md
```

## Running it

Python 3.12, from the repo root:

```bash
uv venv -p 3.12 && uv pip install -e .
python -m hcot.tasks                                   # CPU: task generators re-derive every answer
python -m hcot.p3 --task T2 --fmt sym --tokens 100000  # CPU: exact compiled tracker over 100k tokens
python -m hcot.p3 --task T2 --fmt sym --stock-beta     # the same weights with stock beta: fails
python -m hcot.plant selftest                          # CPU: planted tracker in a random tiny Qwen3.5
```

The behaviour, fine-tuning and mechanism runs need a GPU. An L4 is enough; the scripts ran on
L4, A10G, L40S and RTX PRO 6000 boxes. For example, one seed of the main result:

```bash
python -m hcot.train --model olmoh --task T2 --steps 2002 --bs 16 --seed 3 --beta-mode stock --out adapters/olmoh_T2_stock_s3
python -m hcot.behave --model olmoh --adapter adapters/olmoh_T2_stock_s3 --tasks T2 --formats sym \
    --lengths 16 64 96 128 144 192 256 512 --n 100 --out results/ft/olmoh_T2_stock_s3_answer.jsonl
python -m hcot.seeds results/ft                        # the seed table above
```

Use `--beta-mode cap_min` for the capped arm. `scripts/ft_run4.sh` and `scripts/ft_eval4.sh` are
the exact train and eval recipe, including checkpoint selection on a separate item seed.
`scripts/mech_f3.sh` runs the knockout and state swap on a trained adapter.

## Notes

- The models are pulled from Hugging Face and loaded read-only. RWKV-7 needs `trust_remote_code`.
- Fine-tuning and evals use the fp32 torch gated-delta kernels. flash-linear-attention's chunked
  Triton kernel isn't accurate enough for the compiled 2-hot reflections.
- `hcot/compiler/` is a subset of [chipformer](https://github.com/divagr18/chipformer), the
  compiler behind [Qwenputer](https://divagr.com/blog/qwenputer). Only its imports were changed.

## License

MIT
