#!/bin/bash
# usage: ft_run4.sh MODEL TASK SEED BETA_MODE -> follow-up 3: the follow-up 2 recipe on any task.
# Train (2002 steps, curriculum to 64), then (unless logs/defer_eval exists) ft_eval4.sh: eval
# "best" and "final" at all lengths and the checkpoint-selection set for both.
# NB: update this file only by writing a new file and mv-ing it over (bash reads scripts lazily).
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
M=$1; T=$2; S=$3; B=${4:-stock}; N=${M}_${T}_${B}_s${S}
PY=.venv_nofla/bin/python
[ -f logs/skip_$N ] && { echo "skip $N"; exit 0; }  # run reassigned to another box
if [ ! -f adapters/${N}_final/train_log.json ]; then
  $PY -m hcot.train --model $M --task $T --steps 2002 --minutes 400 --bs 16 --seed $S --beta-mode $B --out adapters/$N > logs/train_$N.log 2>&1
fi
[ -f logs/defer_eval ] && { echo trained > logs/ft_$N.trained; exit 0; }
bash ~/hcot/ft_eval4.sh $M $T $S $B
