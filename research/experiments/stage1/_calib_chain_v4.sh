#!/usr/bin/env bash
# v4 (B1) feasibility ladder on development seeds 9041-9043 (methods only; no monitors). Stops at the first
# pre-declared stop/kill (exit 3). VAL is NOT in the default list: the recipe is frozen and committed first.
L="../../results/raw/calibration"
PY="../../../.venv/Scripts/python.exe"
for ph in ${PHASES:-F0 F1 F234 P2}; do
  echo "=== start $ph $(date +%H:%M:%S)"
  $PY -u -W ignore run_calibration_v4.py --phase $ph > "$L/log_v4_$ph.txt" 2>&1
  rc=$?
  tail -3 "$L/log_v4_$ph.txt"
  if [ $rc -eq 3 ]; then echo "PHASE $ph: STOP CONDITION -> STOP"; exit 3; fi
  if [ $rc -ne 0 ]; then echo "PHASE $ph: CRASH (rc=$rc)"; tail -20 "$L/log_v4_$ph.txt"; exit $rc; fi
  echo "=== done $ph $(date +%H:%M:%S)"
done
echo "CHAIN COMPLETE"
