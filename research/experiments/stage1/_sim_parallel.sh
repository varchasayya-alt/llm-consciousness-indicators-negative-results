#!/usr/bin/env bash
# Parallel simulation-only comparison of v3 primary statistics (methods validation; no model information).
PY="../../../.venv/Scripts/python.exe"
L="../../results/statistics/v3_statistic_simulation"
mkdir -p "$L"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for w in 0 1 2 3 4 5; do
  $PY -W ignore sim_statistic_comparison.py --reps 100 --worker $w --n-workers 6 > "$L/worker_$w.log" 2>&1 &
done
wait
$PY -W ignore sim_statistic_comparison.py --reps 100 --merge > "$L/merge.log" 2>&1
echo "SIM DONE"
