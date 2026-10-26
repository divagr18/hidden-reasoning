#!/bin/bash
cd ~/hcot
./ft_run3.sh q35 1 stock & sleep 600; ./ft_run3.sh q35 2 stock & wait
./ft_run3.sh q35 0 x2
echo done > logs/plan_h2.done
