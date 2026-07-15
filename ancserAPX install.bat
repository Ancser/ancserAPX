@echo off
cd /d "%~dp0"
echo ==========================================
echo   Setting up ancserAPX Environment
echo ==========================================

:: 1. Check Python
echo [1/5] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from python.org and check "Add Python to PATH".
    echo.
    pause
    exit /b
)

:: 2. Create Virtual Environment if missing
if not exist ".venv" (
    echo [2/5] Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to create virtual environment. 
        echo Please ensure you have permission to write to this folder.
        pause
        exit /b
    )
) else (
    echo [2/5] Virtual environment (.venv) found.
)

:: 3. Activate and Install Dependencies
echo [3/5] Installing dependencies...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to activate virtual environment.
    echo Folder .venv\Scripts might be missing or corrupted.
    echo Try deleting the .venv folder and running this script again.
    pause
    exit /b
)

pip install --upgrade pip
echo Installing requirements from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies.
    echo Check your internet connection or proxy settings.
    pause
    exit /b
)

:: 4. Check Config
if not exist ".env" (
    echo [4/5] Creating .env template...
    echo APCA_API_KEY_ID=YOUR_KEY_HERE > .env
    echo APCA_API_SECRET_KEY=YOUR_SECRET_HERE >> .env
    echo.
    echo [IMPORTANT] Please edit '.env' file with your API keys!
) else (
    echo [4/5] .env configuration found.
)

:: 5. Install/update the timezone-aware daily Windows task.
echo [5/5] Installing daily task for 09:25 America/New_York...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CD%\scripts\install_windows_task.ps1"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The Windows scheduled task could not be installed.
    echo Re-run this installer with permission to create tasks.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo   Setup Complete!
echo   You can now run 'ancserAPX web.bat'
echo   Daily live check is installed for 09:25 New York time.
echo ==========================================
pause
