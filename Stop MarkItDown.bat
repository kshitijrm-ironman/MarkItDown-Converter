@echo off
REM Stops the background MarkItDown server started by "Run MarkItDown.bat".
setlocal EnableExtensions
set "PID_FILE=%TEMP%\markitdown_pid.txt"

if not exist "%PID_FILE%" (
    echo [INFO] No running MarkItDown server recorded.
    timeout /t 2 >nul
    exit /b 0
)

set "MD_PID="
for /f "usebackq delims=" %%p in ("%PID_FILE%") do set "MD_PID=%%p"

taskkill /PID %MD_PID% /T /F >nul 2>&1
if errorlevel 1 (
    echo [INFO] MarkItDown was not running ^(stale PID %MD_PID%^).
) else (
    echo [OK] MarkItDown stopped ^(PID %MD_PID%^).
)
del "%PID_FILE%" >nul 2>&1
timeout /t 2 >nul
endlocal
