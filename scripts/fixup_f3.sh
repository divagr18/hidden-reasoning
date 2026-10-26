#!/bin/bash
# usage: fixup_f3.sh MODEL TASK SEED BETA_MODE -> wait until that run's training process is gone and
# its final adapter exists, then run ft_run4.sh (which skips training and only evaluates).
# (Needed because ft_run4.sh was overwritten in place while three runs were executing it.)
M=$1; T=$2; S=$3; B=$4; N=${M}_${T}_${B}_s${S}
cd ~/hcot
while pgrep -f "hcot.train .*--task $T .*--seed $S --beta-mode $B " >/dev/null || [ ! -f adapters/${N}_final/train_log.json ]; do sleep 30; done
bash ~/hcot/ft_run4.sh $M $T $S $B
