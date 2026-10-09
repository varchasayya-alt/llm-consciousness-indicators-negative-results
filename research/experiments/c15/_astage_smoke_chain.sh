#!/usr/bin/env bash
# Engineering only: SMOKE material end-to-end runs (non-evidential), sequential.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export HF_HOME="$ROOT/hf_cache"
LOG="$ROOT/research/results/raw/c15a/engineering"
mkdir -p "$LOG"
for k in "$@"; do
  echo "$(date -Iseconds) SMOKE $k start" >> "$LOG/smoke_chain.log"
  "$ROOT/.venv/Scripts/python.exe" "$ROOT/research/experiments/c15/run_astage.py" SMOKE --model "$k" > "$LOG/smoke_$k.log" 2>&1 \
    && echo "$(date -Iseconds) SMOKE $k done" >> "$LOG/smoke_chain.log" \
    || echo "$(date -Iseconds) SMOKE $k FAILED" >> "$LOG/smoke_chain.log"
done
echo "$(date -Iseconds) SMOKE CHAIN DONE" >> "$LOG/smoke_chain.log"
