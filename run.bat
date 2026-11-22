@echo off
REM Starts the driver monitor. Extra options are passed through, e.g.
REM   run.bat --driver D02 --test-mode --lighting dim --camera-position front
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python main.py %*
pause
