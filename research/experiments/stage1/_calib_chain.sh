#!/usr/bin/env bash
# Calibration chain, revision v3 (methods development only). Stops at the first failing phase:
# exit code 3 = a pre-declared acceptance criterion failed (report; never repair silently).
L="../../results/raw/calibration"
PY="../../../.venv/Scripts/python.exe"
for ph in ${PHASES:-C2dev C4dev C2val C4val C7 C8 C9 C10 VAL}; do
  echo "=== start $ph $(date +%H:%M:%S)"
  $PY -u -W ignore run_calibration.py --phase $ph > "$L/log_v3_$ph.txt" 2>&1
  rc=$?
  tail -3 "$L/log_v3_$ph.txt"
  if [ $rc -eq 3 ]; then echo "PHASE $ph: ACCEPTANCE CRITERION FAILED -> STOP"; exit 3; fi
  if [ $rc -ne 0 ]; then echo "PHASE $ph: CRASH (rc=$rc)"; tail -20 "$L/log_v3_$ph.txt"; exit $rc; fi
  echo "=== done $ph $(date +%H:%M:%S)"
done
echo "CHAIN COMPLETE"
