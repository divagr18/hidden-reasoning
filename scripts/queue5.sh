#!/bin/bash
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
while [ ! -f logs/queue3.done ]; do sleep 30; done
while pgrep -f "tag _v2" > /dev/null; do sleep 30; done
M=".venv/bin/python -m hcot.mech_run"
$M probe --model olmoh --cells T2:sym:16 T2:sym:32 T2:sym:128 T2:sym:512 T2:nl:32 T3:sym:16 T3:sym:128 T1:nl:32 --n 1300 > logs/olmoh_probe.log 2>&1
$M ko --model olmoh --cells T1:nl:8 T1:nl:32 T1:sym:8 T2:sym:2 T2:sym:4 --n 200 > logs/olmoh_ko.log 2>&1
$M patch --model olmoh --cells T1:nl:8 T1:sym:8 T2:sym:2 --n 200 > logs/olmoh_patch.log 2>&1
.venv/bin/python -m hcot.behave --model q35 --mode cot --n 30 --lengths 4 16 64 --max-new 4096 --tok-budget 400000 --max-batch 96 --out results/p1/q35_cot.jsonl > logs/q35_cot.log 2>&1
echo done > logs/queue5.done
