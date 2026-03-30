@echo off
title CV-Count Setup
echo =============================================
echo   CV-Count -- Installing dependencies...
echo =============================================
C:\Python314\python.exe -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
echo.
echo =============================================
echo   Starting CV-Count...
echo =============================================
C:\Python314\python.exe counter.py
pause
