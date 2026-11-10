#!/usr/bin/env bash
# After the language models: vision and the silicon people.
set -uo pipefail
. gpu/environment.sh
. "$RND/venv/bin/activate"
export HF_HUB_DOWNLOAD_TIMEOUT=120
pip install -q --no-cache-dir torchvision 2>&1 | tail -1
until [ -f "$RND/run_llm.done" ]; do sleep 30; done
python3 gpu/machines_vision.py "$RND/geirhos" "$RND/gpu-results" 2>&1 | grep -vE 'Warning|^$' | tail -40
python3 gpu/silicon.py "$RND/silicon/outcomes" 2>&1 | grep -vE 'Warning|^$' | tail -100
echo END-QUEUE
