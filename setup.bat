@echo off
REM One-time setup on Windows: creates a virtual environment and installs everything.
cd /d "%~dp0"
py -3.11 -m venv .venv || py -3.12 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup finished. Start the app with:  run.bat
pause
