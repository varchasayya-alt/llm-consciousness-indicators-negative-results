@echo off
rem C15-R2: R2-Z (pinned artifacts, no downloads) and R2-ENG (SMOKE2) for all three candidates, sequentially.
rem Launched through WMI so that it does not belong to the app's process tree. SMOKE2 material only (no G data).
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
echo %date% %time% ENG CHAIN start>> "%LOG%\chain.log"
"%PY%" "%RUN%" PINS > "%LOG%\pins_start.log" 2>&1
if errorlevel 1 (echo %date% %time% PINS FAILED - STOP>> "%LOG%\chain.log" & goto :eof)
for %%K in (qwen3-1.7b qwen3.5-2b qwen3-4b) do (
  echo %date% %time% Z %%K start>> "%LOG%\chain.log"
  "%PY%" "%RUN%" Z --model %%K > "%LOG%\z_%%K.log" 2>&1
  if errorlevel 1 (echo %date% %time% Z %%K FAILED>> "%LOG%\chain.log") else (echo %date% %time% Z %%K done>> "%LOG%\chain.log")
)
for %%K in (qwen3-1.7b qwen3.5-2b qwen3-4b) do (
  echo %date% %time% SMOKE2 %%K start>> "%LOG%\chain.log"
  "%PY%" "%RUN%" SMOKE2 --model %%K > "%LOG%\smoke2_%%K.log" 2>&1
  if errorlevel 1 (echo %date% %time% SMOKE2 %%K FAILED>> "%LOG%\chain.log") else (echo %date% %time% SMOKE2 %%K done>> "%LOG%\chain.log")
)
echo %date% %time% ENG CHAIN DONE>> "%LOG%\chain.log"
