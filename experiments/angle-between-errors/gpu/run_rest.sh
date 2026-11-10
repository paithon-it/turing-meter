#!/usr/bin/env bash
# The models that dropped out in the first round because of network errors. Waits until the GPU is free,
# retries the download and deletes the weights after each model.
set -uo pipefail
. gpu/environment.sh
. "$RND/venv/bin/activate"
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ENABLE_HF_TRANSFER=0
while pgrep -f "bash gpu/[q]ueue.sh" > /dev/null; do sleep 30; done
download() { # id name: up to eight attempts, without xet from the fourth
  local id="$1" name="$2" i
  for i in 1 2 3 4 5 6 7 8; do
    if [ "$i" -ge 4 ]; then export HF_HUB_DISABLE_XET=1; fi
    timeout 2400 hf download "$id" > /dev/null 2>"$RND/hf-$name.err" && { unset HF_HUB_DISABLE_XET; return 0; }
    echo "attempt $i failed: $(tail -2 "$RND/hf-$name.err" | head -1 | cut -c1-120)"
    sleep 60
  done
  unset HF_HUB_DISABLE_XET
  return 1
}
round() { # id name tasks
  local id="$1" name="$2" tasks="$3" t0=$(date +%s)
  echo "=== $name $tasks $(date -u +%T)"
  if ! download "$id" "$name"; then echo "ERROR download $name"; return; fi
  timeout 5400 python3 gpu/machines_llm.py "$id" "$name" "$tasks" letter,reason > "$RND/log-$name-$(echo "$tasks" | cut -c1-9).txt" 2>&1
  grep -E '^(loaded|chatbench|alphanli|haiid)' "$RND/log-$name-$(echo "$tasks" | cut -c1-9).txt"
  tail -1 "$RND/log-$name-$(echo "$tasks" | cut -c1-9).txt" | cut -c1-200
  echo "--- done $name in $(( $(date +%s) - t0 )) s"
  rm -rf "$HF_HOME/hub"
}
TEXT=chatbench,alphanli,haiid_sarcasm,haiid_census
IMAGES=haiid_art,haiid_cities,haiid_dermatology
round "Qwen/Qwen2.5-3B-Instruct-AWQ" qwen2.5-3b "$TEXT"
round "Qwen/Qwen2.5-7B-Instruct-AWQ" qwen2.5-7b "$TEXT"
round "Qwen/Qwen3.5-9B" qwen3.5-9b "$TEXT"
round "cyankiwi/Qwen3.8-27B-AWQ-INT4" qwen3.8-27b "$TEXT"
round "cyankiwi/Qwen3.8-27B-AWQ-INT4" qwen3.8-27b "$IMAGES"
round "Qwen/Qwen2.5-VL-3B-Instruct-AWQ" qwen2.5-vl-3b haiid_dermatology
round "Qwen/Qwen2.5-VL-7B-Instruct-AWQ" qwen2.5-vl-7b haiid_dermatology
echo END $(date -u +%T); touch "$RND/run_rest.done"
