#!/bin/bash
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
.venv/bin/python -m hcot.behave --model q3 --mode cot --n 30 --lengths 4 16 --max-new 4096 --tok-budget 60000 --max-batch 16 --out results/p1/q3_cot.jsonl > logs/q3_cot.log 2>&1
echo done > logs/q3cot.done
