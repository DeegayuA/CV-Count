@echo off
setlocal
cd /d %~dp0

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Installing Python 3.14...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.14.0/python-3.14.0-amd64.exe' -OutFile 'python_installer.exe'"
    echo [INFO] Running installer... Please wait.
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe
    if %errorlevel% neq 0 (
        echo [ERROR] Installation failed. Please install Python manually.
        pause
        exit /b
    )
)

:: Try to find the specific 3.14 path if it exists
set PY_CMD=python
if exist "C:\Python314\python.exe" set PY_CMD=C:\Python314\python.exe


echo [INFO] Launching CV-Count...
%PY_CMD% counter.py
if %errorlevel% neq 0 (
    echo [ERROR] Application crashed.
    pause
)
