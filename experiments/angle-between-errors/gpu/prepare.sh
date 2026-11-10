#!/usr/bin/env bash
# Prepares the machine with the GPU. Everything lives inside $RND: at the end a single folder is deleted.
# Usage, on the machine with the GPU:  bash gpu/prepare.sh
set -euo pipefail
RND="${RND:-$HOME/rnd}"
mkdir -p "$RND"
. "$(dirname "$0")/environment.sh"
python3 -m venv "$RND/venv"
. "$RND/venv/bin/activate"
pip install -q --no-cache-dir --upgrade pip
pip install -q --no-cache-dir vllm timm pillow pandas pyarrow scipy scikit-learn qwen-vl-utils
python -c "import vllm, torch, timm; print('vllm', vllm.__version__, 'torch', torch.__version__, 'timm', timm.__version__, 'cuda', torch.cuda.is_available())"
