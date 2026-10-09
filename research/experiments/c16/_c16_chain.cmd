@echo off
setlocal EnableDelayedExpansion
rem C16 Stage 0/1 chain: S0 -> S1B -> S1C -> S1A -> REPORT (prereg c16_s0s1_preregistration_FROZEN.md). Launched via WMI.
for %%I in ("%~dp0..\..\..") do set "ROOT=%%~fI"
set "HF_HOME=%ROOT%\hf_cache"
set HF_HUB_OFFLINE=1
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
cd /d "%ROOT%"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "RUN=%ROOT%\research\experiments\c16\run_c16.py"
set "LOG=%ROOT%\research\results\raw\c16\logs"
if not exist "%LOG%" mkdir "%LOG%"
for %%P in (S0 S1B S1C S1A REPORT) do (
  echo !date! !time! %%P start>> "%LOG%\chain.log"
  "%PY%" "%RUN%" %%P > "%LOG%\phase_%%P.out" 2>&1
  set RC=!errorlevel!
  if !RC! neq 0 (
    echo !date! !time! %%P FAILED rc=!RC!>> "%LOG%\chain.log"
    exit /b !RC!
  )
  echo !date! !time! %%P done>> "%LOG%\chain.log"
)
echo !date! !time! CHAIN COMPLETE>> "%LOG%\chain.log"
exit /b 0
