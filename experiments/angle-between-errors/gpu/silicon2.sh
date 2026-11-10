#!/usr/bin/env bash
# The silicon people, second run: defect that depends on the image. Starts after the LLMs.
set -uo pipefail
. gpu/environment.sh
. "$RND/venv/bin/activate"
until [ -f "$RND/run_rest.done" ]; do sleep 30; done
python3 gpu/silicon.py "$RND/silicon/outcomes2" 2>&1 | grep -vE 'Warning|^$' | tail -100
touch "$RND/silicon2.done"
