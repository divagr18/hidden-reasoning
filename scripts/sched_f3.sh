#!/bin/bash
# usage: sched_f3.sh MAXJOBS "TASK SEED BETA" ... -> start each run (ft_run4.sh) once fewer than
# MAXJOBS hcot.train processes are running on this box
MAX=$1; shift
for r in "$@"; do
  set -- $r
  while [ $(pgrep -fc "hcot.train") -ge $MAX ]; do sleep 60; done
  setsid nohup bash ~/hcot/ft_run4.sh olmoh $1 $2 $3 > ~/hcot/logs/sched_$1_$2_$3.out 2>&1 < /dev/null &
  sleep 120
done
