#!/usr/bin/env bash
# All the language models, one at a time: download, make it answer, delete the weights.
set -uo pipefail
. gpu/environment.sh
. "$RND/venv/bin/activate"
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ENABLE_HF_TRANSFER=0
TEXT=chatbench,alphanli,haiid_sarcasm,haiid_census
IMAGES=haiid_art,haiid_cities,haiid_dermatology
round() { # id name tasks
  local id="$1" name="$2" tasks="$3" t0=$(date +%s)
  echo "=== $name $tasks $(date -u +%T)"
  if ! timeout 2400 hf download "$id" > /dev/null 2>"$RND/hf-$name.err"; then echo "ERROR download $name"; tail -2 "$RND/hf-$name.err"; return; fi
  timeout 5400 python3 gpu/machines_llm.py "$id" "$name" "$tasks" letter,reason 2>&1 | grep -E '^(loaded|chatbench|alphanli|haiid|Traceback|[A-Za-z]*Error|ValueError)' || echo "ERROR run $name"
  echo "--- done $name in $(( $(date +%s) - t0 )) s"
}
clean() { rm -rf "$HF_HOME/hub/models--$(echo "$1" | sed 's#/#--#g')"; }
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
for entry in "${LIST[@]}"; do IFS='|' read -r id name <<<"$entry"; round "$id" "$name" "$TEXT"; clean "$id"; done
for entry in "Qwen/Qwen2.5-VL-3B-Instruct-AWQ|qwen2.5-vl-3b" "Qwen/Qwen2.5-VL-7B-Instruct-AWQ|qwen2.5-vl-7b" "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit|gemma-4-26b" "cyankiwi/Qwen3.8-27B-AWQ-INT4|qwen3.8-27b"; do
  IFS='|' read -r id name <<<"$entry"; round "$id" "$name" "$IMAGES"; clean "$id"
done
echo END $(date -u +%T)
