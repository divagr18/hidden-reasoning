#!/bin/bash
# usage: ft_run2.sh MODEL TASK -> train with the pilot's step budget (2002), then answer-only eval
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
M=$1; T=$2
if [ ! -f adapters/${M}_${T}/train_log.json ]; then
  .venv_nofla/bin/python -m hcot.train --model $M --task $T --steps 2002 --minutes 240 --bs 16 --out adapters/${M}_${T} > logs/train_${M}_${T}.log 2>&1
fi
.venv_nofla/bin/python -m hcot.behave --model $M --adapter adapters/${M}_${T} --tasks $T --formats sym --lengths 16 64 128 256 512 --n 100 --tok-budget 40000 --max-batch 32 --out results/ft/${M}_${T}_answer.jsonl > logs/ft_eval_${M}_${T}.log 2>&1
echo done > logs/ft_${M}_${T}.done
