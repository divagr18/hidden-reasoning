#!/bin/bash
# usage: eval_sweep.sh MAXTRAIN "TASK SEED BETA" ... -> evaluate each listed run once its training has
# finished, starting an eval only while at most MAXTRAIN trainings are running (GPU memory).
# Several sweeps can run at once: a run is claimed with mkdir.
cd ~/hcot
MAX=$1; shift
left=("$@")
while [ ${#left[@]} -gt 0 ]; do
  next=()
  for r in "${left[@]}"; do
    set -- $r; N=olmoh_$1_$3_s$2
    [ -f logs/ft_$N.done ] && continue
    if pgrep -f "hcot.train .*--task $1 .*--seed $2 --beta-mode $3 " >/dev/null || [ ! -f adapters/${N}_final/train_log.json ] \
       || [ $(pgrep -fc "hcot.train") -gt $MAX ] || ! mkdir logs/claim_$N 2>/dev/null; then
      next+=("$r"); continue
    fi
    TOKB=10000 bash ~/hcot/ft_eval4.sh olmoh $1 $2 $3
    [ -f logs/ft_$N.done ] || { rmdir logs/claim_$N; next+=("$r"); sleep 60; }
  done
  left=("${next[@]}")
  [ ${#left[@]} -gt 0 ] && sleep 60
done
echo done > logs/eval_sweep_$$.done
