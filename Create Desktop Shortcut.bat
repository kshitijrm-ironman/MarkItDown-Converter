@echo off
REM Puts a "MarkItDown" shortcut on the desktop that launches the app with the
REM logo icon and no console window. Run once.
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "assets\markitdown.ico" (
    echo [SETUP] Drawing the app icon...
    if exist "venv\Scripts\python.exe" (
        venv\Scripts\python.exe tools\make_icon.py
    ) else (
        echo [WARNING] venv not built yet - run "Run MarkItDown.bat" once first.
    )
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$w = New-Object -ComObject WScript.Shell;" ^
  "$s = $w.CreateShortcut([IO.Path]::Combine($w.SpecialFolders('Desktop'),'MarkItDown.lnk'));" ^
  "$s.TargetPath = '%CD%\MarkItDown.vbs';" ^
  "$s.WorkingDirectory = '%CD%';" ^
  "$s.IconLocation = '%CD%\assets\markitdown.ico';" ^
  "$s.Description = 'MarkItDown - convert documents, images, audio and URLs to Markdown';" ^
  "$s.Save()"

if errorlevel 1 (
    echo [ERROR] Could not create the shortcut.
) else (
    echo [OK] Desktop shortcut created.
)
pause
endlocal
