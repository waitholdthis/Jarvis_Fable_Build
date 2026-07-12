@echo off
setlocal EnableExtensions
title Jarvis - Ollama / Gemma 4
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

rem ===== Jarvis desktop configuration =====
set "JARVIS_DIR=C:\Users\User\Jarvis_Fable_Build"
set "JARVIS_PROVIDER=ollama"
set "JARVIS_API_BASE=http://127.0.0.1:11434/v1"
set "JARVIS_MODEL=gemma4:latest"
set "JARVIS_FAST_MODE=1"
set "OLLAMA_HOST=127.0.0.1:11434"
set "OLLAMA_EXE=C:\Users\User\AppData\Local\Programs\Ollama\ollama.exe"
set "PYTHON_EXE=%JARVIS_DIR%\.venv\Scripts\python.exe"
set "JARVIS_GUI=http://127.0.0.1:8765"

echo [Jarvis] Starting with Ollama model %JARVIS_MODEL%...

if not exist "%JARVIS_DIR%\pyproject.toml" (
    echo [ERROR] Jarvis was not found at "%JARVIS_DIR%".
    goto :fail
)

if not exist "%OLLAMA_EXE%" (
    where ollama.exe >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Ollama is not installed or is not on PATH.
        echo Install Ollama, then run: ollama pull %JARVIS_MODEL%
        goto :fail
    )
    set "OLLAMA_EXE=ollama.exe"
)

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Jarvis's local Python environment is missing:
    echo         "%PYTHON_EXE%"
    echo Recreate it from the project folder with:
    echo         python -m venv .venv
    echo         .venv\Scripts\python.exe -m pip install -e .
    goto :fail
)

cd /d "%JARVIS_DIR%" || goto :fail

"%OLLAMA_EXE%" list >nul 2>&1
if errorlevel 1 (
    echo [Jarvis] Ollama is not running; starting it now...
    start "Ollama Server" /min "%OLLAMA_EXE%" serve
    for /L %%N in (1,1,20) do (
        timeout /t 1 /nobreak >nul
        "%OLLAMA_EXE%" list >nul 2>&1 && goto :ollama_ready
    )
    echo [ERROR] Ollama did not become ready at %OLLAMA_HOST%.
    goto :fail
)

:ollama_ready
"%OLLAMA_EXE%" list | findstr /B /I /C:"%JARVIS_MODEL%" >nul
if errorlevel 1 (
    echo [ERROR] The required Ollama model "%JARVIS_MODEL%" is not installed.
    echo Install it with: ollama pull %JARVIS_MODEL%
    goto :fail
)

echo [Jarvis] Warming %JARVIS_MODEL% and keeping it in GPU memory for 30 minutes...
powershell.exe -NoProfile -Command "$body=@{model='%JARVIS_MODEL%';keep_alive='30m'}|ConvertTo-Json; Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/generate' -Method Post -ContentType 'application/json' -Body $body | Out-Null" >nul 2>&1
if errorlevel 1 echo [Jarvis] Warning: model warm-up failed; continuing with normal on-demand loading.

"%PYTHON_EXE%" -c "import jarvis, httpx, rich" >nul 2>&1
if errorlevel 1 (
    echo [Jarvis] Installing the local Jarvis package...
    "%PYTHON_EXE%" -m pip install -e "%JARVIS_DIR%"
    if errorlevel 1 goto :fail
)

echo [Jarvis] Ollama is ready. Starting the graphical dashboard...
echo [Jarvis] Your browser will open at %JARVIS_GUI%
echo.
start "" /B powershell.exe -NoProfile -WindowStyle Hidden -Command "$u='%JARVIS_GUI%'; for($i=0; $i -lt 60; $i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 1; if($r.StatusCode -eq 200){ Start-Process $u; exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }"
"%PYTHON_EXE%" -m jarvis serve --host 127.0.0.1 --port 8765
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [Jarvis] Dashboard stopped with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%

:fail
echo.
echo Jarvis could not start. Review the message above.
pause
exit /b 1
