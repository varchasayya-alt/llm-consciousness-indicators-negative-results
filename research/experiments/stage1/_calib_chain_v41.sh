#!/usr/bin/env bash
# v4.1 feasibility ladder (calibration_plan.md Revision v4.1; D52-D55). Methods only; no monitors. Stops at the first
# pre-declared stop/kill (exit 3). Grid stores (and the two confirmation stores) are trained in parallel by the TRAIN
# utility (no evaluation); phases then evaluate the cached stores. VAL is never run by this script: FREEZE + commit first.
L="../../results/raw/calibration"
PY="../../../.venv/Scripts/python.exe"
R="run_calibration_v4.py"
run_phase() {
  echo "=== start $1 $(date +%H:%M:%S)"
  $PY -u -W ignore $R --phase $1 > "$L/log_v41_$1.txt" 2>&1
  rc=$?
  tail -3 "$L/log_v41_$1.txt"
  if [ $rc -eq 3 ]; then echo "PHASE $1: STOP CONDITION -> STOP"; exit 3; fi
  if [ $rc -ne 0 ]; then echo "PHASE $1: CRASH (rc=$rc)"; tail -20 "$L/log_v41_$1.txt"; exit $rc; fi
  echo "=== done $1 $(date +%H:%M:%S)"
}
train_parallel() {   # args: threads seed:prd ...
  th=$1; shift
  pids=()
  for sp in "$@"; do
    s=${sp%%:*}; p=${sp##*:}
    S1_THREADS=$th $PY -u -W ignore $R --phase TRAIN --seed $s --prd $p --epochs $E > "$L/log_v41_TRAIN_${s}_${p}.txt" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid || { echo "TRAIN CRASH"; exit 1; }; done
}
E_of() { $PY -c "import json;print(json.load(open('$L/calibration_results_v41.json'))['F0']['E_store'])"; }
SEL_of() { $PY -c "import json;print(json.load(open('$L/calibration_results_v41.json'))['F1']['selected_p_rd'])"; }

for ph in ${PHASES:-F0 F1G F1C F234 P2}; do
  case $ph in
    F1G) E=$(E_of); echo "=== parallel TRAIN 9061 grid (E=$E) $(date +%H:%M:%S)"; train_parallel 3 9061:0.05 9061:0.1 9061:0.2 9061:0.5 ;;
    F1C) E=$(E_of); P=$(SEL_of); echo "=== parallel TRAIN confirm p_rd=$P (E=$E) $(date +%H:%M:%S)"; train_parallel 6 9062:$P 9063:$P ;;
  esac
  run_phase $ph
done
echo "CHAIN COMPLETE (development). Next: FREEZE, commit, then VAL once."
