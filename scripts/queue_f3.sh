#!/bin/bash
# usage: queue_f3.sh MODEL TASK BETA_MODE SEED... -> ft_run4.sh for each seed in turn
M=$1; T=$2; B=$3; shift 3
for S in "$@"; do bash ~/hcot/ft_run4.sh $M $T $S $B; done
echo done > ~/hcot/logs/queue_f3_${T}_${B}.done
