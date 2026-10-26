#!/bin/bash
# usage: ft_run3.sh MODEL SEED BETA_MODE -> parity run with the pilot recipe, then eval (same beta mode)
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
M=$1; S=$2; B=${3:-stock}; N=${M}_T2_${B}_s${S}
if [ ! -f adapters/$N/train_log.json ]; then
  .venv_nofla/bin/python -m hcot.train --model $M --task T2 --steps 2002 --minutes 400 --bs 16 --seed $S --beta-mode $B --out adapters/$N > logs/train_$N.log 2>&1
fi
.venv_nofla/bin/python -m hcot.behave --model $M --adapter adapters/$N --beta-mode $B --tasks T2 --formats sym --lengths 16 64 96 128 144 192 256 512 --n 100 --tok-budget 20000 --max-batch 32 --out results/ft/${N}_answer.jsonl > logs/ft_eval_$N.log 2>&1
echo done > logs/ft_$N.done
