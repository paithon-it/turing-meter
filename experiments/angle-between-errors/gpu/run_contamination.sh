#!/usr/bin/env bash
# A11 on the L4: the open language models on αNLI with the two hypotheses swapped, letter only, one model at a time
# (download, answer, delete the weights). The same models and the same code as run_llm.sh. Results in $RND/gpu-results.
set -uo pipefail
. gpu/environment.sh
. "$RND/venv/bin/activate"
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1   # the xet backend fails here ("Background writer channel closed")
LIST=(
 "HuggingFaceTB/SmolLM3-3B|smollm3-3b"
 "Qwen/Qwen2.5-3B-Instruct-AWQ|qwen2.5-3b"
 "microsoft/Phi-3.5-mini-instruct|phi-3.5-mini"
 "Qwen/Qwen2.5-7B-Instruct-AWQ|qwen2.5-7b"
 "allenai/Olmo-3-7B-Instruct|olmo-3-7b"
 "Qwen/Qwen3-8B-AWQ|qwen3-8b"
 "ibm-granite/granite-3.3-8b-instruct|granite-3.3-8b"
 "Qwen/Qwen3.5-9B|qwen3.5-9b"
 "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit|gemma-4-26b"
 "cyankiwi/Qwen3.8-27B-AWQ-INT4|qwen3.8-27b"
)
for entry in "${LIST[@]}"; do
  IFS='|' read -r id name <<<"$entry"
  t0=$(date +%s); echo "=== $name $(date -u +%T)"
  if ! timeout 2400 hf download "$id" > /dev/null 2>"$RND/hf-$name.err"; then
    echo "ERROR download $name"; tail -2 "$RND/hf-$name.err"
    rm -rf "$HF_HOME/hub/models--$(echo "$id" | sed 's#/#--#g')" "$HF_HOME/xet"; continue
  fi
  timeout 5400 python3 gpu/machines_llm.py "$id" "$name" alphanli-swapped letter 2>&1 | grep -E '^(loaded|alphanli|Traceback|[A-Za-z]*Error)' || echo "ERROR run $name"
  rm -rf "$HF_HOME/hub/models--$(echo "$id" | sed 's#/#--#g')"
  echo "--- done $name in $(( $(date +%s) - t0 )) s"
done
echo "=== END $(date -u +%T)"
