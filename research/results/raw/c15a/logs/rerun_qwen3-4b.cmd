@echo off
rem Engineering restart of the C15-R A-stage SELECT for Qwen3-4B (the first worker was killed by a session
rem restart before writing any result). Identical committed code and data; unbuffered output only.
rem Launched through WMI so that it does not belong to the app's process tree.
set "ROOT=<REPO_ROOT>"
set "HF_HOME=%ROOT%\hf_cache"
set PYTHONUNBUFFERED=1
cd /d "%ROOT%"
set "LOG=%ROOT%\research\results\raw\c15a\logs"
echo %date% %time% RERUN SELECT qwen3-4b start>> "%LOG%\chain.log"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\research\experiments\c15\run_astage.py" SELECT --model qwen3-4b > "%LOG%\select_qwen3-4b.log" 2>&1
if errorlevel 1 (echo %date% %time% SELECT qwen3-4b FAILED>> "%LOG%\chain.log") else (echo %date% %time% SELECT qwen3-4b done>> "%LOG%\chain.log")
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\research\experiments\c15\run_astage.py" CHOOSE > "%LOG%\choose.log" 2>&1
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\research\experiments\c15\run_astage.py" REPORT >> "%LOG%\choose.log" 2>&1
echo %date% %time% CHAIN DONE>> "%LOG%\chain.log"
