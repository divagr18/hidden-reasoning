#!/bin/bash
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
while [ ! -f logs/queue1.done ]; do sleep 30; done
.venv/bin/python -m hcot.behave --model olmo3 --n 200 --tok-budget 4000 --max-batch 16 --out results/p1/olmo3_answer.jsonl > logs/olmo3_answer2.log 2>&1
echo done > logs/queue6.done
