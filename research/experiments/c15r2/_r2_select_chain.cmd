@echo off
setlocal EnableDelayedExpansion
rem C15-R2: SELECT on G_select2 for all three candidates (sequential), then CLASSIFY and REPORT.
rem Launched through WMI so that it does not belong to the app's process tree. The runner refuses SELECT unless
rem the tree is clean, the A-stage pins hold, R2-Z passed and SMOKE2 passed on every non-excluded model.
rem FREEZE and CONFIRM are run separately, after the FREEZE2 record has been committed (guards enforce this).
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
echo !date! !time! SELECT CHAIN start>> "%LOG%\chain.log"
for %%K in (qwen3-1.7b qwen3.5-2b qwen3-4b) do (
  if exist "%ROOT%\research\results\raw\c15r2\select2_%%K.json" (
    echo !date! !time! skip %%K done>> "%LOG%\chain.log"
  ) else (
    echo !date! !time! SELECT %%K start>> "%LOG%\chain.log"
    "%PY%" "%RUN%" SELECT --model %%K > "%LOG%\select2_%%K.log" 2>&1
    if errorlevel 1 (echo !date! !time! SELECT %%K FAILED>> "%LOG%\chain.log") else (echo !date! !time! SELECT %%K done>> "%LOG%\chain.log")
  )
)
"%PY%" "%RUN%" CLASSIFY > "%LOG%\classify2.log" 2>&1
"%PY%" "%RUN%" REPORT >> "%LOG%\classify2.log" 2>&1
echo !date! !time! SELECT CHAIN DONE>> "%LOG%\chain.log"
