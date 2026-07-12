@echo off
setlocal EnableDelayedExpansion
title JARVIS -- Starting
cls

echo.
echo  ============================================================
echo    J . A . R . V . I . S   --   Local AI Assistant
echo    Arc Cognition Core / Mark VII
echo  ============================================================
echo.

REM --- Verify WSL2 is available ---
where wsl.exe >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] WSL is not available on this machine.
    echo          Install WSL2: https://aka.ms/wsl2
    echo.
    pause
    exit /b 1
)

REM --- Verify Ubuntu distro exists ---
wsl.exe -d Ubuntu -- true >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Ubuntu WSL distro not found.
    echo          Run in PowerShell: wsl --install -d Ubuntu
    echo.
    pause
    exit /b 1
)

echo  [*] Contacting Ubuntu subsystem ...
echo  [*] Checking Ollama + JARVIS web server (first run may take 60 s)
echo  [*] To watch live progress, open a second terminal and run:
echo      wsl -d Ubuntu -- tail -f /tmp/jarvis-live.log
echo.

REM --- Run the startup orchestrator in WSL (blocks until server is up or fails) ---
wsl.exe -d Ubuntu -- bash /home/waitholdthis/Jarvis_Fable_Build/scripts/start-jarvis.sh
set RESULT=%ERRORLEVEL%

echo.
if %RESULT% neq 0 (
    echo  [ERROR] JARVIS did not start  (exit code %RESULT%)
    echo.
    echo  Diagnose with:
    echo    wsl -d Ubuntu -- tail -40 /tmp/jarvis-live.log
    echo    wsl -d Ubuntu -- tail -20 /tmp/ollama.log
    echo    wsl -d Ubuntu -- /home/waitholdthis/Jarvis_Fable_Build/.venv/bin/python -m jarvis doctor
    echo.
    echo  Common fixes:
    echo    Install Ollama in WSL:
    echo      wsl -d Ubuntu -- bash -c "curl -fsSL https://ollama.com/install.sh | sh"
    echo    Pull the default model:
    echo      wsl -d Ubuntu -- ollama pull qwen2.5:7b
    echo    Rebuild the Python venv:
    echo      wsl -d Ubuntu -- python3 -m venv /home/waitholdthis/Jarvis_Fable_Build/.venv
    echo.
    pause
    exit /b 1
)

REM --- Get the WSL2 IP (localhost forwarding is unreliable; use the real IP) ---
for /f "tokens=*" %%i in ('wsl.exe -d Ubuntu -- cat /tmp/jarvis-wsl-ip 2^>nul') do set WSL_IP=%%i
if "%WSL_IP%"=="" set WSL_IP=127.0.0.1

set JARVIS_URL=http://%WSL_IP%:8765

echo  [+] JARVIS is online.
echo.
echo  ╔══════════════════════════════════════╗
echo  ║  Open this URL in your browser:      ║
echo  ║  %JARVIS_URL%              ║
echo  ╚══════════════════════════════════════╝
echo.
echo  Opening browser automatically ...
start "" "%JARVIS_URL%"

echo.
echo  If the browser did not open, copy the URL above and paste it manually.
echo  This window will close in 8 seconds.
timeout /t 8 /nobreak >nul
exit /b 0
