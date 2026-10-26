#!/bin/bash
# usage: ft_eval4.sh MODEL TASK SEED BETA_MODE -> eval "best" and "final" adapters at all lengths
# (n=100, symbolic) and the checkpoint-selection set (TASK, 64 steps, item seed 99, n=100) for both.
# Resumable: behave skips cells already in the output file.
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
M=$1; T=$2; S=$3; B=${4:-stock}; N=${M}_${T}_${B}_s${S}
P=".venv_nofla/bin/python -m hcot.behave --model $M --beta-mode $B --tasks $T --formats sym --tok-budget ${TOKB:-20000} --max-batch 32"
$P --adapter adapters/$N --lengths 16 64 96 128 144 192 256 512 --n 100 --out results/ft/${N}_answer.jsonl >> logs/ft_eval_$N.log 2>&1
$P --adapter adapters/${N}_final --lengths 16 64 96 128 144 192 256 512 --n 100 --out results/ft/${N}_final_answer.jsonl >> logs/evalf_$N.log 2>&1
for X in $N ${N}_final; do $P --adapter adapters/$X --lengths 64 --n 100 --seed 99 --out results/ft/${X}_val.jsonl >> logs/evalf_$N.log 2>&1; done
n=$(cat results/ft/${N}_answer.jsonl results/ft/${N}_final_answer.jsonl results/ft/${N}_val.jsonl results/ft/${N}_final_val.jsonl 2>/dev/null | wc -l)
[ "$n" = 1800 ] && echo done > logs/ft_$N.done || echo "incomplete $n rows" > logs/ft_$N.incomplete
