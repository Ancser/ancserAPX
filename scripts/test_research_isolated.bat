@echo off
setlocal

rem Offline-safety tests intentionally reject a Python process that already
rem imported production backtest/store modules. Run them in a clean process.
python -m pytest -q tests\test_research_v2_cli.py tests\test_research_v2_safety.py
if errorlevel 1 exit /b %errorlevel%

rem Run all remaining tests in a second process so fail-closed safety remains.
python -m pytest -q --ignore=tests\test_research_v2_cli.py --ignore=tests\test_research_v2_safety.py
exit /b %errorlevel%
