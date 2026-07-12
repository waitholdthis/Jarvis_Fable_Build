#!/usr/bin/env bash
set -u

APP_DIR="/home/waitholdthis/Jarvis_Fable_Build"
URL="http://127.0.0.1:8765"
LOG="/tmp/jarvis-live.log"

if ! curl -fsS --max-time 1 "$URL/" >/dev/null 2>&1; then
  cd "$APP_DIR" || exit 1
  setsid .venv/bin/python -m jarvis serve --port 8765 >"$LOG" 2>&1 </dev/null &
fi

for _attempt in {1..30}; do
  if curl -fsS --max-time 1 "$URL/" >/dev/null 2>&1; then
    exit 0
  fi
  sleep 0.5
done

echo "JARVIS did not start. Review $LOG" >&2
exit 1
