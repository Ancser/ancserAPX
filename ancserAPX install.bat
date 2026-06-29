@echo off
cd /d "%~dp0"
echo ==========================================
echo   Setting up ancserAPX Environment
echo ==========================================

:: 1. Check Python
echo [1/4] Checking Python...
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
    echo [2/4] Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to create virtual environment. 
        echo Please ensure you have permission to write to this folder.
        pause
        exit /b
    )
) else (
    echo [2/4] Virtual environment (.venv) found.
)

:: 3. Activate and Install Dependencies
echo [3/4] Installing dependencies...
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
    echo [4/4] Creating .env template...
    echo APCA_API_KEY_ID=YOUR_KEY_HERE > .env
    echo APCA_API_SECRET_KEY=YOUR_SECRET_HERE >> .env
    echo.
    echo [IMPORTANT] Please edit '.env' file with your API keys!
) else (
    echo [4/4] .env configuration found.
)

echo.
echo ==========================================
echo   Setup Complete!
echo   You can now run 'ancserAPX web.bat'
echo ==========================================
pause
