#!/usr/bin/env bash
# v4.2 (final B1 revision) feasibility ladder: calibration_plan.md Revision v4.2; D57-D60. Methods only; no monitors.
# Stops at the first pre-declared stop (exit 3) -- any v4.2 stop ends the B1 line. Stages are trained in parallel
# processes by the TRAINA / TRAINB utilities (no evaluation); phases then evaluate the cached stages.
# VAL is never run by this script: FREEZE + commit first.
L="../../results/raw/calibration"
PY="../../../.venv/Scripts/python.exe"
R="run_calibration_v42.py"
run_phase() {
  echo "=== start $1 $(date +%H:%M:%S)"
  $PY -u -W ignore $R --phase $1 > "$L/log_v42_$1.txt" 2>&1
  rc=$?
  tail -3 "$L/log_v42_$1.txt" | cut -c1-400
  if [ $rc -eq 3 ]; then echo "PHASE $1: STOP CONDITION -> STOP (B1 line ends)"; exit 3; fi
  if [ $rc -ne 0 ]; then echo "PHASE $1: CRASH (rc=$rc)"; tail -20 "$L/log_v42_$1.txt"; exit $rc; fi
  echo "=== done $1 $(date +%H:%M:%S)"
}
train_parallel() {   # args: phase threads seed:prd ...
  local tp=$1 th=$2; shift 2
  local pids=() sp s p pid
  for sp in "$@"; do
    s=${sp%%:*}; p=${sp##*:}
    S1_THREADS=$th $PY -u -W ignore $R --phase $tp --seed $s --prd $p > "$L/log_v42_${tp}_${s}_${p}.txt" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid || { echo "$tp CRASH"; exit 1; }; done
}
SEL_of() { $PY -c "import json;print(json.load(open('$L/calibration_results_v42.json'))['F0A']['selected_p_rd'])"; }

for ph in ${PHASES:-F0A F0B F1C F234 P2}; do
  case $ph in
    F0A) echo "=== parallel TRAINA 9071 grid $(date +%H:%M:%S)"
         train_parallel TRAINA 2 9071:0.05 9071:0.1 9071:0.2 9071:0.35 9071:0.5 ;;
    F1C) P=$(SEL_of); echo "=== parallel TRAINA + TRAINB 9072/9073 at p_rd=$P $(date +%H:%M:%S)"
         train_parallel TRAINA 6 9072:$P 9073:$P
         train_parallel TRAINB 6 9072:$P 9073:$P ;;
  esac
  run_phase $ph
done
echo "CHAIN COMPLETE (development). Next: FREEZE, commit, then VAL once."
