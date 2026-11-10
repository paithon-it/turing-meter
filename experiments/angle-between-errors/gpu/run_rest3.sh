#!/usr/bin/env bash
# Qwen3.8 27B: the rest of the text and the images, with a context that fits in memory.
set -uo pipefail
. gpu/environment.sh
. "$RND/venv/bin/activate"
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ENABLE_HF_TRANSFER=0
download() {
  local id="$1" name="$2" i
  for i in 1 2 3 4 5 6 7 8; do
    if [ "$i" -ge 4 ]; then export HF_HUB_DISABLE_XET=1; fi
    timeout 2400 hf download "$id" > /dev/null 2>"$RND/hf-$name.err" && { unset HF_HUB_DISABLE_XET; return 0; }
    echo "attempt $i failed"; sleep 60
  done
  unset HF_HUB_DISABLE_XET; return 1
}
round() {
  local id="$1" name="$2" tasks="$3" t0=$(date +%s) log
  log="$RND/log-$name-$(echo "$tasks" | cut -c1-9).txt"
  echo "=== $name $tasks $(date -u +%T)"
  if ! download "$id" "$name"; then echo "ERROR download $name"; return; fi
  timeout 7200 python3 gpu/machines_llm.py "$id" "$name" "$tasks" letter,reason > "$log" 2>&1
  grep -E '^(loaded|chatbench|alphanli|haiid)' "$log"
  grep -E 'Error' "$log" | tail -2 | cut -c1-240
  echo "--- done $name in $(( $(date +%s) - t0 )) s"
}
TEXT=alphanli,haiid_sarcasm,haiid_census
IMAGES=haiid_art,haiid_cities,haiid_dermatology
MEM_GPU=0.98 MAX_LEN=2048 round "cyankiwi/Qwen3.8-27B-AWQ-INT4" qwen3.8-27b "$TEXT"
MEM_GPU=0.98 MAX_LEN=2300 SIDE_MAX=1024 round "cyankiwi/Qwen3.8-27B-AWQ-INT4" qwen3.8-27b "$IMAGES"
rm -rf "$HF_HOME/hub"
echo END $(date -u +%T); touch "$RND/run_rest3.done"
