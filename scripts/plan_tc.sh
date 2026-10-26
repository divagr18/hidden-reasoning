#!/bin/bash
cd ~/hcot
./ft_run3.sh olmoh 0 cap_min & ./ft_run3.sh olmoh 1 cap_min & wait
./ft_run3.sh olmoh 1 stock & ./ft_run3.sh olmoh 2 stock & wait
./ft_run3.sh olmoh 2 cap_min
echo done > logs/plan_tc.done
