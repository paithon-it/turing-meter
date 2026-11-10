#!/usr/bin/env bash
# The Hugging Face cache keeps files in hub/blobs: deleting a model's folder does not free the disk.
# Removes the blobs no model uses any more, if untouched for at least MIN minutes (so it does not touch a download in progress).
set -uo pipefail
HUB="${HF_HOME:-$HOME/rnd/hf}/hub"
MIN="${1:-20}"
cd "$HUB" || exit 0
used=$(mktemp -p "${TMPDIR:-$HOME/rnd/tmp}")
find . -path '*/snapshots/*' -type l -exec readlink -f {} \; | sort -u > "$used"
find "$PWD/blobs" -type f -mmin +"$MIN" | sort | comm -23 - "$used" | xargs -r rm -f
rm -f "$used"
