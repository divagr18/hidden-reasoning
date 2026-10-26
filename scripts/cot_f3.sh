#!/bin/bash
# sanity check: is T5 solvable with visible reasoning? base Olmo-Hybrid, CoT on, n=20
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
.venv_nofla/bin/python -m hcot.behave --model olmoh --mode cot --tasks T5 --formats sym nl --lengths 8 32 --n 20 --max-new 2048 --tok-budget 60000 --max-batch 20 --out results/ft/olmoh_base_T5_cot.jsonl > logs/cot_T5.log 2>&1
echo done > logs/cot_T5.done
