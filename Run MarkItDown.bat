@echo off
setlocal EnableExtensions
title MarkItDown App
cd /d "%~dp0"

REM ============================================================
REM  MarkItDown App - launcher
REM  Double-click to start. First run creates a venv and installs
REM  packages (takes a few minutes); later runs start in seconds.
REM ============================================================

REM Bump this when requirements.txt changes so existing venvs re-install.
set "DEPS_VERSION=4"
set "MARKER=venv\.deps_installed"
set "OCR_MARKER=venv\.ocr_installed"
set "PY="

echo.
echo ==================================================
echo   MarkItDown App - Document to Markdown converter
echo ==================================================
echo.

REM ---------- 1. Find Python 3.12 (preferred) or any python ----------
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
    set "PY=py -3.12"
    goto :python_found
)
python --version >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
    goto :python_found
)
echo [ERROR] Python is not installed or not found in PATH.
echo         Install Python 3.12 from https://www.python.org/downloads/
echo         and tick "Add python.exe to PATH" during setup.
echo.
pause
exit /b 1

:python_found
if exist "venv\Scripts\python.exe" goto :venv_ready
echo [SETUP] Creating virtual environment (Python 3.12 preferred, falls back to default python)...
%PY% -m venv venv
if errorlevel 1 (
    echo [ERROR] Could not create the virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment created.

:venv_ready
set "VPY=venv\Scripts\python.exe"

REM ---------- 2. Base dependencies (re-run when DEPS_VERSION changes) ----------
set "HAVE_DEPS="
if exist "%MARKER%" set /p HAVE_DEPS=<"%MARKER%"
if "%HAVE_DEPS%"=="%DEPS_VERSION%" goto :deps_ready

echo [SETUP] Installing/updating packages - this takes a few minutes the first time...
"%VPY%" -m pip install --upgrade pip --quiet
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Package installation failed. Check your internet connection and re-run.
    pause
    exit /b 1
)
echo [SETUP] Installing Chromium for Playwright (JavaScript-rendered web pages)...
"%VPY%" -m playwright install chromium
>"%MARKER%" echo %DEPS_VERSION%
echo [OK] Base packages installed.

:deps_ready

REM ---------- 3. Optional GPU extras (Unlimited-OCR + faster-whisper) ----------
if exist "%OCR_MARKER%" goto :ocr_done
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [SKIP] No NVIDIA GPU detected - Unlimited-OCR and faster-whisper ^(CUDA^) not installed.
    echo        Tesseract / Claude / Ollama modes still work.
    goto :ocr_done
)
echo.
echo [OPTIONAL] Unlimited-OCR (document parsing) + faster-whisper (audio/video transcription)
echo            run on your NVIDIA GPU. Needs ^>= 8 GB VRAM and downloads ~3 GB
echo            (torch CUDA wheel) plus ~6.7 GB of model weights on first use.
choice /C YN /T 10 /D N /M "Install GPU extras now? (auto-skips in 10s)"
if errorlevel 2 (
    echo [SKIP] Skipped. You can install later with:
    echo        venv\Scripts\python -m pip install -r requirements-ocr.txt
    goto :ocr_done
)
echo [SETUP] Installing PyTorch CUDA + Unlimited-OCR + faster-whisper dependencies...
"%VPY%" -m pip install -r requirements-ocr.txt
if errorlevel 1 (
    echo [WARNING] GPU extras install failed. The app will still run without them.
) else (
    >"%OCR_MARKER%" echo ok
    echo [OK] GPU extras installed.
)

:ocr_done

REM ---------- 4. External tools (informational only) ----------
echo.
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo [OK] Tesseract OCR found.
) else (
    echo [WARNING] Tesseract OCR not found - the free image OCR mode will not work.
    echo           Download: https://github.com/UB-Mannheim/tesseract/wiki
)
where ollama >nul 2>&1
if not errorlevel 1 (
    echo [OK] Ollama found.
) else (
    echo [WARNING] Ollama not found - the Llama local modes will not work.
    echo           Download: https://ollama.com/download
)

REM ---------- 5. Launch ----------
REM Pick a free localhost port so we never collide with another Streamlit app
REM (8501/8502 etc.). Set STREAMLIT_SERVER_PORT before running to force one.
REM The chosen port is written to %TEMP%\markitdown_port.txt (kept for the
REM browser auto-open below). Port is controlled here only - never in
REM .streamlit\config.toml - so it cannot conflict with this picker.
set "PORT_FILE=%TEMP%\markitdown_port.txt"
set "LAUNCH_LOG=%TEMP%\markitdown_launch.log"
if defined STREAMLIT_SERVER_PORT goto :port_ready
"%VPY%" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()" > "%PORT_FILE%" 2>nul
if errorlevel 1 goto :port_fallback
set /p STREAMLIT_SERVER_PORT=<"%PORT_FILE%"
if not defined STREAMLIT_SERVER_PORT goto :port_fallback
goto :port_ready

:port_fallback
echo [WARNING] Could not pick a free port automatically - letting Streamlit choose.
set "STREAMLIT_SERVER_PORT="
del "%PORT_FILE%" >nul 2>&1

:port_ready
echo.
echo [LAUNCH] Starting MarkItDown App... (Ctrl+C to stop)
if not defined STREAMLIT_SERVER_PORT goto :launch_default

>"%PORT_FILE%" echo %STREAMLIT_SERVER_PORT%
echo [INFO] Serving on http://localhost:%STREAMLIT_SERVER_PORT% (free port picked automatically)
echo [INFO] Your browser will open automatically once the server is up.
echo.
REM Background watcher: read the port back from the port file, poll until the
REM server answers HTTP 200, then open the default browser on that URL.
REM (config.toml sets server.headless=true so Streamlit itself does not also open one.)
start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=(Get-Content '%PORT_FILE%').Trim(); $u='http://localhost:'+$p; for($i=0;$i -lt 240;$i++){ try { if((Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2).StatusCode -eq 200){ Start-Process $u; Add-Content '%LAUNCH_LOG%' ((Get-Date -Format s)+' opened '+$u); exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }; Add-Content '%LAUNCH_LOG%' ((Get-Date -Format s)+' gave up waiting for '+$u)"
"%VPY%" -m streamlit run markitdown_app.py --server.port %STREAMLIT_SERVER_PORT%
goto :launched

:launch_default
echo [INFO] Port unknown - browser will not open automatically; use the URL Streamlit prints below.
echo.
"%VPY%" -m streamlit run markitdown_app.py

:launched
echo.
echo [INFO] App closed.
pause
endlocal
