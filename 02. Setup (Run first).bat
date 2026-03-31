@echo off
title CV-Count Smart Setup
echo =============================================
echo   CV-Count -- Initializing Setup...
echo =============================================

C:\Python314\python.exe setup_env.py

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Smart Setup failed.
    echo Please check your internet connection or Python path.
    pause
    exit /b 1
)

echo.
echo =============================================
echo   SETUP COMPLETE!
echo =============================================
echo You can now run the app via:
echo [ 03. Start CV Count App (Run after 02).bat ]

pause
