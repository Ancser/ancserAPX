@echo off
REM ============================================================
REM  ancserFX — one-click web launcher
REM  Port 7780 is deliberately BELOW 8000 so the TopstepX bot's
REM  "kill everything on 8000+" cleanup cannot touch this server.
REM  Change PORT below if 7780 is ever taken.
REM ============================================================
setlocal
cd /d "%~dp0"
set "PORT=7780"
set "HOST=127.0.0.1"

REM Use the project venv if it exists, otherwise fall back to global python
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [warn] .venv not found - using global python. Run "ancserFX install.bat" first if imports fail.
)

REM Bail out early if the port is already busy (avoids a confusing crash)
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo [error] Port %PORT% is already in use.
    echo Edit "ancserFX web.bat" and change PORT, then try again.
    pause
    exit /b 1
)

echo Starting ancserFX server on http://%HOST%:%PORT% ...
start "ancserFX server" cmd /k python -m uvicorn frontend.server:app --host %HOST% --port %PORT%

REM Give uvicorn a moment to boot, then open the browser
timeout /t 3 /nobreak >nul
start "" "http://%HOST%:%PORT%"

endlocal
