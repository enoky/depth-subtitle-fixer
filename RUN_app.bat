@echo off
rem Launch the local Gradio app in the project's own .venv.
rem
rem Double-click it, or run it from a shell with any flag `dsf ui` takes:
rem     ui.bat --port 7861
rem     ui.bat --share

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

dsf ui %*
set "exit_code=%ERRORLEVEL%"

rem Double-clicked, the window closes the instant this returns and takes any traceback with
rem it - so hold it open when something went wrong, and only then.
if not "%exit_code%"=="0" (
    echo.
    echo dsf ui exited with code %exit_code%.
    pause
)
exit /b %exit_code%
