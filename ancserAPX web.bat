@echo off
REM ============================================================
REM  ancserAPX — one-click web launcher
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
    echo [warn] .venv not found - using global python. Run "ancserAPX install.bat" first if imports fail.
)

REM Clear an old APX server before starting again. This keeps the browser from
REM showing a stale UI served by a previous process on the same port.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    if not "%%p"=="0" (
        echo [info] Port %PORT% is already in use by PID %%p - stopping it...
        taskkill /PID %%p /F >nul 2>nul
    )
)

timeout /t 1 /nobreak >nul
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo [error] Port %PORT% is still in use and could not be cleared.
    pause
    exit /b 1
)

echo Starting ancserAPX server on http://%HOST%:%PORT% ...
start "ancserAPX server" cmd /k python -m uvicorn frontend.server:app --host %HOST% --port %PORT%

REM Give uvicorn a moment to boot, then open the browser
timeout /t 3 /nobreak >nul
start "" "http://%HOST%:%PORT%"

endlocal
