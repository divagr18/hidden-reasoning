# Do hybrid LLMs compute where the chain of thought can't see?

## Correction first

Earlier I said Qwenputer shows that DeltaNet state can carry millions of hidden serial steps. That overstated it, in two ways:

- **The CPU writes its work down.** Each step of the CPU is a trace token that the model generates. RvChat hides those tokens from the user, but they're in the sequence, so the computation is externalized, like a chain of thought written in machine code.
- **The recurrence is linear.** A Gated DeltaNet layer updates its state as S_t = S_{t-1}·A_t + v_t·k_tᵀ, with A_t = decay·(I − β·k·kᵀ). The only serial computation a layer does across tokens is multiplying these matrices together. The nonlinear depth is still capped by the number of layers, as in a transformer ("The Illusion of State", Merrill, Petty & Sabharwal, ICML 2024).

## The question

Monitoring chain of thought assumes that long serial reasoning has to show up in the tokens, because a transformer's hidden serial depth is capped by its layers. Hybrid models replace most attention layers with linear-RNN layers. Can those layers do serial state tracking that never shows up in the tokens?

The theory says it depends on the eigenvalues of A_t (Grazzi, Siems et al., "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues", ICLR 2025):

- **Eigenvalues in [0, 1] cannot solve parity.** That rules out tracking cyclic or permutation state, such as an odd/even count or a shuffled set of items.
- **Allowing negative eigenvalues makes that kind of tracking possible.** In DeltaNet this means β ∈ [0, 2]. With products of these matrices (DeltaProduct) and several layers, it can track any regular language.

**Qwen3.5 sits on the restricted side.** β = sigmoid(b), which lies in (0, 1], and the decay is in (0, 1]. Our compiler relies on exactly this: β = sigmoid(128) = 1 in `chipformer/q35/emit.py:223`. Some newer recurrent models allow negative eigenvalues; RWKV-7, for example, claims to recognise all regular languages.

The question becomes three testable ones:

1. Does Qwen3.5 do hidden state tracking that a dense model of similar size can't? Theory predicts no for parity and permutation tasks, and maybe a gain on "last write wins" tasks at long range.
2. Where does Qwen3.5 keep tracked state: in the DeltaNet state, or by re-reading the raw tokens with its attention layers?
3. How much does one constant matter? The plan is to show by construction that capping β at 1 makes the architecture monitorable for these tasks, and that β up to 2 lets it track state with no tokens.

Every outcome is reportable:
- **Qwen3.5 shows no hidden-tracking advantage.** The CoT argument holds for today's hybrids, and we can say exactly which design change would break it. That's useful to safety people.
- **It shows an advantage on long-range "last write" tasks.** That's a real, if limited, hidden channel.
- **It shows an advantage on parity or permutation tasks.** That's surprising and would need checking against the theory.

## Models

- **Qwen3.5-4B:** a hybrid of Gated DeltaNet layers and full attention every few layers.
- **Qwen3-4B:** dense, the closest same-family baseline. Training data differs, so this comparison is suggestive only. The causal tests in P2 are the controlled part.
- **RWKV-7, about 3B:** recurrent, and claims state tracking beyond the eigenvalue limit. This tests the positive side of the theory in a released model.
- **Optional:** a Mamba-2 hybrid of similar size (Falcon-H1 or Nemotron-H). Mamba-2 has a diagonal recurrence with positive eigenvalues.
- **Excluded:** 0.8B models (per the standing rule), and any weight training.

## Phases

### P0. Literature and setup (about 1 day, CPU)
- **Read:**
  - Grazzi et al. 2025; Merrill et al. 2024
  - the RWKV-7 paper
  - Li, Guo & Andreas 2025, "(How) Do Language Models Track State?" (LMs learn associative-scan algorithms for permutations)
  - Korbak et al. 2025, CoT monitorability
  - entity-tracking work, for example Kim & Schuster 2023
- **Check whether anyone has already tested a production hybrid on hidden state tracking.** If someone has, sharpen the plan around what they left out.
- **Confirm the β and decay ranges** for Qwen3.5 and each comparison model from their HF modeling code.

### P1. Behaviour: state tracking without CoT (about 4 GPU-h, 1–2 A10G/L4 boxes)

Tasks, each generated with exact answers at 2, 4, 8, … 512 steps:

| Task | Type | Theory for eigenvalues in [0, 1] |
|---|---|---|
| T1 Variable reassignment ("x=3 … y=x … x=7 … what is y?") | last write wins, counter-free | possible |
| T2 Light switch ("toggle, toggle, …: is it on?") | parity | impossible |
| T3 Shell game with 3 cups (S3 swaps) | solvable group | impossible without negative eigenvalues |
| T4 Five items shuffled (S5) | non-solvable group, NC1-complete | impossible, and also hard for constant-depth transformers |

Write each task both as natural language and as compact symbols.

Protocol:
- Turn thinking off, require an answer-only format, decode greedily, n = 200 per cell.
- Also run each task with CoT on as a ceiling.

Report accuracy against the number of steps, per model, with confidence intervals.

**Gate:** is there any task and length range where Qwen3.5 clearly beats Qwen3 without CoT? Whatever the answer, P2 runs on Qwen3.5 for the tasks it can do at all.

### P2. Mechanism: where Qwen3.5 keeps the state (about 4 GPU-h)

This is the controlled, within-model part. Qwenputer's tooling reads and writes per-head DeltaNet state exactly (`hf.head_slots`, executor state access, init injection), so state patching is easy.

1. **Probes.**
   - Train linear probes for the running value (current y, switch state, ball location) on the residual stream at each layer.
   - Train probes directly on each DeltaNet head's state matrix S at the last step token.
   - Find out where the value first becomes decodable, and whether it lives in S.
2. **Attention knockout.**
   - At the answer position, mask the full-attention layers so they can't see the step tokens. Only the DeltaNet state can then carry the information.
   - If accuracy holds, the state carries it. If it drops to chance, Qwen re-reads the steps through attention, the way a transformer does.
   - This test is only possible in hybrids, and it's the cleanest single result in the plan.
3. **State patching.**
   - Swap the DeltaNet state of chosen heads between two sequences that differ only in the tracked value, then check whether the answer follows the swap.
   - This shows causally which heads carry the state.
4. **Distance.** Put irrelevant text between the steps and the question. DeltaNet state decays and attention doesn't, so this shows which channel holds up.

**Gate:** we have causal evidence for which channel carries each task.

### P3. Constructive bound (mostly CPU, plus a short GPU check)

Use the compiler to build tracking circuits as extra chip layers, leaving Qwen's weights untouched. They read ordinary text tokens and emit nothing.

- **β ≤ 1 (stock Qwen3.5):**
  - Compile T1 exactly for 100k tokens. It's an overwrite, the same as the store's write path.
  - Write up why T2 to T4 can't be compiled by a single linear pass with these eigenvalues. That's the theorem; we restate it for this exact parameterization, decay and head sizes.
- **β ≤ 2 (a one-line change to the chip layers' beta):**
  - Compile T2 with a Householder reflection: β = 2 turns I − 2kkᵀ into a sign flip.
  - Compile T3 and T4 with products of reflections spread across layers, following Grazzi's construction.
  - Show they're exact over 100k tokens, with nothing in the transcript.

This gives a crisp figure: the same layers with β capped at 1 can't track the switch, and with β up to 2 they track it exactly forever with no tokens.

It needs a custom flag in the modeling code for the chip layers' β range. The checkpoint then no longer loads as a stock class. That's fine for a demonstration; say so plainly.

### P4. Write-up and outreach
- **The write-up:** a short paper or post, "Hybrid LLMs and chain-of-thought monitorability". It covers the P1 curves, the P2 knockout and patching results, the P3 construction, and what it means for design choices (eigenvalue range as a monitorability knob).
- **Neel Nanda:** one line on Qwenputer and a GIF, one line on the P2 result, and one question: does the eigenvalue range belong in monitorability evaluations of new architectures?
- **Julien Siems:** first, because P3 builds on his paper. Ask whether the construction matches his bound and whether he knows of production models with negative eigenvalues.
- **Also relevant:** Songlin Yang (DeltaNet) and Imanol Schlag.

## Costs and constraints
- **Compute:** about 8–10 GPU-h total, on 1–2 boxes (up to 4 if it saves time). Stop each box when idle.
- **Qwen's weights:** P1 and P2 only read them. P3 adds chip layers and leaves Qwen's weights untouched.
- **Size and identity:** no 0.8B models; commit with the noreply identity only.
- **Before any public post:** check with the user before posting and before sending any results to anyone.

## Risks
- **Models fail every task even at short lengths, so there's no signal.**
  - Use the natural-language versions and start at 2 steps.
  - T1 should work for both models.
- **Qwen3 and Qwen3.5 differ in more than architecture.**
  - P2 gives within-model causal evidence.
  - P1 is reported as suggestive only.
- **Thinking mode leaks reasoning into visible text.**
  - Force answer-only output and check that no reasoning tokens appear.
- **Someone has already tested production hybrids this way.** P0 checks this. If so, P2 and P3 are still likely to be new.
