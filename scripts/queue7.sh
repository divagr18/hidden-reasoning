#!/bin/bash
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
while [ ! -f logs/queue6.done ]; do sleep 30; done
.venv/bin/python -m hcot.behave --model q3 --mode cot --n 30 --lengths 4 16 64 --max-new 4096 --tok-budget 160000 --max-batch 48 --out results/p1/q3_cot.jsonl > logs/q3_cot.log 2>&1
echo done > logs/queue7.done
