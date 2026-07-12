@echo off
title JARVIS Launcher
echo Initializing JARVIS...
wsl.exe -d Ubuntu -- bash /home/waitholdthis/Jarvis_Fable_Build/scripts/start-jarvis.sh
if errorlevel 1 (
  echo.
  echo JARVIS could not start. Check /tmp/jarvis-live.log in Ubuntu.
  pause
  exit /b 1
)
start "" http://127.0.0.1:8765
exit /b 0
