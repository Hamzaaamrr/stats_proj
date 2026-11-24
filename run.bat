@echo off
REM Starts the driver monitor. Extra options are passed through, e.g.
REM   run.bat --driver D02 --test-mode --lighting dim --camera-position front
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing or invalid. Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py %*
pause
