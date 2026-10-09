#!/usr/bin/env bash
# Stage-1 D2 STORE-ONLY kill test (d2_protocol.md; D62-D65). No monitor. Development seeds 9101-9103 only.
# Per-seed work runs in parallel processes; S1 and SELECT are single-process gates. Exit 3 = pre-declared stop.
L="../../results/raw/calibration"
PY="../../../.venv/Scripts/python.exe"
R="run_d2.py"
per_seed() {   # args: phase threads
  local ph=$1 th=$2 pids=() s pid
  for s in 9101 9102 9103; do
    S1_THREADS=$th $PY -u -W ignore $R --phase $ph --seed $s > "$L/log_d2_${ph}_${s}.txt" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid || { echo "$ph CRASH"; tail -20 "$L"/log_d2_${ph}_*.txt; exit 1; }; done
}
gate() {   # arg: phase
  echo "=== start $1 $(date +%H:%M:%S)"
  $PY -u -W ignore $R --phase $1 > "$L/log_d2_$1.txt" 2>&1
  local rc=$?
  tail -4 "$L/log_d2_$1.txt" | cut -c1-500
  if [ $rc -eq 3 ]; then echo "PHASE $1: STOP CONDITION -> STOP (D2 kill report)"; exit 3; fi
  if [ $rc -ne 0 ]; then echo "PHASE $1: CRASH (rc=$rc)"; exit $rc; fi
  echo "=== done $1 $(date +%H:%M:%S)"
}
for st in ${STAGES:-TRAIN S1 S2 S3 SELECT}; do
  case $st in
    TRAIN|S2|S3) echo "=== parallel $st $(date +%H:%M:%S)"; per_seed $st 4; echo "=== done $st $(date +%H:%M:%S)" ;;
    S1|SELECT) gate $st ;;
  esac
done
echo "D2 STORE-ONLY KILL TEST COMPLETE. STOP before any monitor."
