@echo off
setlocal
cd /d "%~dp0"
if not exist "logs" mkdir "logs"

:: Prefer the project interpreter; allow an explicit .env override.
set "PYTHON_EXEC=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXEC=%CD%\.venv\Scripts\python.exe"
if exist .env (
    for /f "tokens=1,* delims==" %%a in ('type .env ^| findstr /i "^PYTHON_EXEC="') do (
        set "PYTHON_EXEC=%%b"
    )
)

echo [%date% %time%] ancserAPX scheduled run starting >> "logs\daily_task.log"
"%PYTHON_EXEC%" -m backend.execution.scheduler --run-once --scheduled >> "logs\daily_task.log" 2>&1
set "RUN_EXIT=%ERRORLEVEL%"
echo [%date% %time%] ancserAPX scheduled run exit=%RUN_EXIT% >> "logs\daily_task.log"

:: Recalculate tomorrow's local trigger from 09:25 America/New_York. This keeps
:: hosts in Arizona, Europe, Asia, etc. correct through DST transition weeks.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CD%\scripts\install_windows_task.ps1" -Refresh >> "logs\daily_task.log" 2>&1
exit /b %RUN_EXIT%
