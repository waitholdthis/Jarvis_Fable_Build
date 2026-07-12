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

echo  [+] JARVIS is online.  Opening browser ...
start "" "http://127.0.0.1:8765"
exit /b 0
