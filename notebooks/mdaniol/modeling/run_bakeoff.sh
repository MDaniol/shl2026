#!/bin/bash
# Sequential frozen-FM bake-off: extract embeddings + probe/fuse, one model at a
# time (avoids MPS contention). Each model: cache embeddings, then evaluate
# logreg / lgbm / lgbm-fusion vs the handcrafted baseline. Safe to re-run
# (extraction skips cached files).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=SHL-local/bin/python
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(mantisv2 utica mantis8m moment-small)

for m in "${MODELS[@]}"; do
  echo "=================== $m  ($(date +%H:%M)) ==================="
  $PY notebooks/mdaniol/modeling/extract_embeddings.py --model "$m" --variant V0 \
      || { echo "EXTRACT FAILED: $m"; continue; }
  $PY notebooks/mdaniol/modeling/probe_fusion.py --emb "${m}_V0" \
      || echo "PROBE FAILED: $m"
done
echo "=================== bake-off done ($(date +%H:%M)) ==================="
cat notebooks/mdaniol/BAKEOFF_RESULTS.md 2>/dev/null