#!/bin/bash
# Continue the local bake-off for the remaining models (memory-fixed extractor).
# Waits for any in-flight probe to finish first (avoid CPU/RAM contention).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=SHL-local/bin/python
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1

while pgrep -f "probe_fusion.py" >/dev/null; do sleep 30; done

for m in "$@"; do
  echo "===== $m EXTRACT ($(date +%H:%M)) ====="
  $PY notebooks/mdaniol/modeling/extract_embeddings.py --model "$m" --variant V0 \
      || { echo "EXTRACT FAIL $m"; continue; }
  echo "===== $m PROBE ($(date +%H:%M)) ====="
  $PY notebooks/mdaniol/modeling/probe_fusion.py --emb "${m}_V0" || echo "PROBE FAIL $m"
done
echo "===== remaining models done ($(date +%H:%M)) ====="
cat notebooks/mdaniol/BAKEOFF_RESULTS.md