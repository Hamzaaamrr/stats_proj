@echo off
REM One-time setup on Windows: creates a virtual environment and installs everything.
setlocal
cd /d "%~dp0"
py -3.11 --version 2>nul | findstr /b /c:"Python 3.11." >nul
if not errorlevel 1 goto create311
py -3.12 --version 2>nul | findstr /b /c:"Python 3.12." >nul
if not errorlevel 1 goto create312
py -3.14 --version 2>nul | findstr /b /c:"Python 3.14." >nul
if not errorlevel 1 goto create314

echo Could not create a virtual environment with Python 3.11, 3.12, or 3.14.
echo Install one of these versions and make sure the Python launcher is available.
pause
exit /b 1

:create311
py -3.11 -m venv --clear .venv
if errorlevel 1 goto failed
goto install

:create312
py -3.12 -m venv --clear .venv
if errorlevel 1 goto failed
goto install

:create314
py -3.14 -m venv --clear .venv
if errorlevel 1 goto failed

:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Setup finished. Start the app with:  run.bat
pause
exit /b 0

:failed
echo.
echo Setup failed. Review the error above, then run setup.bat again.
pause
exit /b 1
