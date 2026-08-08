@echo off
rem Launch the folder scanner - find clips carrying subtitles or credits - in the
rem project's own .venv.
rem
rem Double-click it, or run it from a shell:
rem     RUN_scan.bat

setlocal
rem Run from the project root whatever directory the shell happens to be in - a
rem double-click from Explorer can start anywhere, and %~dp0 is where this file lives.
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo No .venv in "%CD%".
    echo Create it first:
    echo     powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

python scripts\scan_for_text.py %*
set "exit_code=%ERRORLEVEL%"

rem Double-clicked, the window closes the instant this returns and takes any traceback with
rem it - so hold it open when something went wrong, and only then.
if not "%exit_code%"=="0" (
    echo.
    echo scan_for_text.py exited with code %exit_code%.
    pause
)
exit /b %exit_code%
