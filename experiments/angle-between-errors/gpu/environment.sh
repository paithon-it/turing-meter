# Common variables: every cache inside $RND, vLLM usage statistics off.
RND="${RND:-$HOME/rnd}"
export RND HF_HOME="$RND/hf" XDG_CACHE_HOME="$RND/cache" XDG_CONFIG_HOME="$RND/config" XDG_DATA_HOME="$RND/data" \
       TRITON_CACHE_DIR="$RND/triton" TORCH_HOME="$RND/torch" PIP_CACHE_DIR="$RND/pip" MPLCONFIGDIR="$RND/mpl" \
       VLLM_CACHE_ROOT="$RND/vllm" VLLM_CONFIG_ROOT="$RND/vllm-config" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
       HF_HUB_DISABLE_TELEMETRY=1 TMPDIR="$RND/tmp"
mkdir -p "$TMPDIR"
