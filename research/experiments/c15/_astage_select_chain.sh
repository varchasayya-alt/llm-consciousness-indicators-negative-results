#!/usr/bin/env bash
# C15-R A-stage: SELECT on G_select for all three candidates (sequential), then CHOOSE and REPORT.
# Run detached (background tool tasks are capped at 2 h). FREEZE and CONFIRM are run separately, after the
# FREEZE record has been committed (guards enforce this).
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export HF_HOME="$ROOT/hf_cache"
PY="$ROOT/.venv/Scripts/python.exe"
RUN="$ROOT/research/experiments/c15/run_astage.py"
LOG="$ROOT/research/results/raw/c15a/logs"
mkdir -p "$LOG"
for k in qwen3-1.7b qwen3.5-2b qwen3-4b; do
  if [ -f "$ROOT/research/results/raw/c15a/select_$k.json" ]; then echo "skip $k (done)" >> "$LOG/chain.log"; continue; fi
  echo "$(date -Iseconds) SELECT $k start" >> "$LOG/chain.log"
  if "$PY" "$RUN" SELECT --model "$k" > "$LOG/select_$k.log" 2>&1; then
    echo "$(date -Iseconds) SELECT $k done" >> "$LOG/chain.log"
  else
    echo "$(date -Iseconds) SELECT $k FAILED" >> "$LOG/chain.log"
  fi
done
"$PY" "$RUN" CHOOSE > "$LOG/choose.log" 2>&1 && echo "$(date -Iseconds) CHOOSE done" >> "$LOG/chain.log"
"$PY" "$RUN" REPORT >> "$LOG/choose.log" 2>&1
echo "$(date -Iseconds) CHAIN DONE" >> "$LOG/chain.log"
