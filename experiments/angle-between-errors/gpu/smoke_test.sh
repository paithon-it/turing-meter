#!/usr/bin/env bash
set -uo pipefail
. gpu/environment.sh
. "$RND/venv/bin/activate"
hf download Qwen/Qwen2.5-7B-Instruct-AWQ > /dev/null
python3 gpu/machines_llm.py Qwen/Qwen2.5-7B-Instruct-AWQ qwen2.5-7b chatbench letter
