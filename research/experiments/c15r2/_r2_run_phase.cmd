@echo off
setlocal EnableDelayedExpansion
rem C15-R2: run one runner phase with the standard environment and logging. Usage: _r2_run_phase.cmd PHASE [MODEL] [LAYER]
rem Exit code is the runner's, so callers can chain on success. Launched through WMI.
for %%I in ("%~dp0..\..\..") do set "ROOT=%%~fI"
set "HF_HOME=%ROOT%\hf_cache"
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
cd /d "%ROOT%"
set "PY=%ROOT%\.venv\Scripts\python.exe"
set "RUN=%ROOT%\research\experiments\c15r2\run_r2.py"
set "LOG=%ROOT%\research\results\raw\c15r2\logs"
if not exist "%LOG%" mkdir "%LOG%"
set "ARGS=%1"
if not "%2"=="" set "ARGS=!ARGS! --model %2"
if not "%3"=="" set "ARGS=!ARGS! --layer %3"
set "TAG=%1_%2_%3"
echo !date! !time! %1 %2 %3 start>> "%LOG%\chain.log"
"%PY%" "%RUN%" !ARGS! > "%LOG%\phase_!TAG!.log" 2>&1
set RC=!errorlevel!
if !RC! neq 0 (echo !date! !time! %1 %2 %3 FAILED rc=!RC!>> "%LOG%\chain.log") else (echo !date! !time! %1 %2 %3 done>> "%LOG%\chain.log")
exit /b !RC!
