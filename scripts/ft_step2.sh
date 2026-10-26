#!/bin/bash
cd ~/hcot
./ft_run2.sh olmo3 T2 & ./ft_run2.sh q35 T2 & wait
./ft_run2.sh olmoh T3 & ./ft_run2.sh olmo3 T3 & wait
echo done > logs/ft_step2.done
