#!/bin/bash
# usage: mech_f3.sh RUN BETA_MODE TAG CELLS... -> knockout + state-swap (with and without knockout)
# on a fine-tuned adapter, n=100; writes results/p2/olmoh_{ko,patch}TAG.json
cd ~/hcot
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
R=$1; B=$2; TAG=$3; shift 3
for E in ko patch; do
  .venv_nofla/bin/python -m hcot.mech_run $E --model olmoh --adapter adapters/$R --beta-mode $B --cells "$@" --n 100 --tag $TAG > logs/mech_${E}$TAG.log 2>&1
done
echo done > logs/mech$TAG.done
