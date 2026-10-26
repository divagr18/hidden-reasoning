#!/bin/bash
# setup on a fresh DLAMI box; HF token arrives on stdin (first line) and stays in memory only
set -e
read -r HF_TOKEN || true
export HF_TOKEN
cd ~
command -v ~/.local/bin/uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh) >/dev/null 2>&1
cd ~/hcot
~/.local/bin/uv venv -q -p 3.12 .venv_nofla
~/.local/bin/uv pip install -q -p .venv_nofla/bin/python torch==2.14.1 transformers==5.19.0 safetensors numpy tokenizers huggingface_hub accelerate peft datasets
for m in "$@"; do .venv_nofla/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$m'))"; done
.venv_nofla/bin/python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo SETUP_OK
