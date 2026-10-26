#!/bin/bash
# usage: eval_final.sh MODEL NAME BETA_MODE -> eval NAME_final at all lengths, and a selection set
# (T2, 64 steps, item seed 99, n=100) for both NAME and NAME_final
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
M=$1; N=$2; B=${3:-stock}
P=".venv_nofla/bin/python -m hcot.behave --model $M --beta-mode $B --tasks T2 --formats sym --tok-budget 20000 --max-batch 32"
[ -f results/ft/${N}_final_answer.jsonl ] || $P --adapter adapters/${N}_final --lengths 16 64 96 128 144 192 256 512 --n 100 --out results/ft/${N}_final_answer.jsonl > logs/evalf_$N.log 2>&1
for X in $N ${N}_final; do [ -f results/ft/${X}_val.jsonl ] || $P --adapter adapters/$X --lengths 64 --n 100 --seed 99 --out results/ft/${X}_val.jsonl >> logs/evalf_$N.log 2>&1; done
