#!/usr/bin/env bash
# Engineering only (SMOKE material, non-evidential): validation + smoke runs, sequential.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export HF_HOME="$ROOT/hf_cache"
PY="$ROOT/.venv/Scripts/python.exe"
LOG="$ROOT/research/results/raw/c15a/engineering"
mkdir -p "$LOG"
for k in qwen3-1.7b qwen3.5-2b qwen3-4b; do
  "$PY" "$ROOT/research/experiments/c15/tools/validate_engineering.py" "$k" > "$LOG/validate_$k.log" 2>&1 \
    && echo "$(date -Iseconds) VALIDATE $k done" >> "$LOG/eng_chain.log" || echo "$(date -Iseconds) VALIDATE $k FAILED" >> "$LOG/eng_chain.log"
done
for k in qwen3-1.7b qwen3-4b; do
  "$PY" "$ROOT/research/experiments/c15/run_astage.py" SMOKE --model "$k" > "$LOG/smoke_$k.log" 2>&1 \
    && echo "$(date -Iseconds) SMOKE $k done" >> "$LOG/eng_chain.log" || echo "$(date -Iseconds) SMOKE $k FAILED" >> "$LOG/eng_chain.log"
done
echo "$(date -Iseconds) ENG CHAIN DONE" >> "$LOG/eng_chain.log"
