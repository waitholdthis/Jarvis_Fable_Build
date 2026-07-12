#!/usr/bin/env bash
# JARVIS startup orchestrator — called by JARVIS.bat via WSL.
# Steps:
#   1. Load API keys from .env (if present)
#   2. Start Ollama if installed but not running
#   3. Create the Python venv and install deps if needed
#   4. Write a default config.toml if none exists
#   5. Launch the JARVIS web server and wait for it to respond
set -uo pipefail

# Ensure user-local binaries (ollama, etc.) are always on PATH
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:${LD_LIBRARY_PATH:-}"

APP_DIR="/home/waitholdthis/Jarvis_Fable_Build"
VENV="$APP_DIR/.venv"
PYTHON="$VENV/bin/python"
URL="http://127.0.0.1:8765"
LOG="/tmp/jarvis-live.log"
OLLAMA_LOG="/tmp/ollama.log"
ENV_FILE="$APP_DIR/.env"
JARVIS_HOME="${JARVIS_HOME:-$HOME/.jarvis}"
CONFIG="$JARVIS_HOME/config.toml"

log()  { printf '[JARVIS] %s\n'       "$*"; }
ok()   { printf '[JARVIS] \xE2\x9C\x93 %s\n' "$*"; }
warn() { printf '[JARVIS] ! %s\n'     "$*"; }
err()  { printf '[JARVIS] ERROR: %s\n' "$*" >&2; }

# ─── 1. Load .env ─────────────────────────────────────────────────────────────
# Supports KEY=VALUE, export KEY=VALUE, comments (#), and blank lines.
# Keeps secrets out of config files — put BRAVE_API_KEY, OPENAI_API_KEY,
# ANTHROPIC_API_KEY, SERPER_API_KEY, ALPHA_VANTAGE_KEY, TAVILY_API_KEY, etc. here.

if [[ -f "$ENV_FILE" ]]; then
    log "Loading API keys from $ENV_FILE ..."
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line//[[:space:]]/}" ]] && continue
        line="${line#export }"
        if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            export "$line"
        fi
    done < "$ENV_FILE"
    ok ".env loaded"
fi

# ─── 2. Ollama ────────────────────────────────────────────────────────────────

_ollama_ok() {
    curl -fsS --max-time 2 "http://127.0.0.1:11434/api/version" &>/dev/null
}

if command -v ollama &>/dev/null; then
    if _ollama_ok; then
        ok "Ollama already running"
    else
        log "Starting Ollama ..."
        setsid ollama serve >"$OLLAMA_LOG" 2>&1 </dev/null &
        for _i in {1..20}; do
            _ollama_ok && { ok "Ollama online"; break; }
            sleep 0.5
        done
        if ! _ollama_ok; then
            warn "Ollama did not respond — JARVIS will use whatever LLM backend is reachable"
            warn "Ollama log: $OLLAMA_LOG"
        fi
    fi
else
    warn "Ollama not installed — for local LLM inference run:"
    warn "  curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:7b"
fi

# ─── 3. Python venv ───────────────────────────────────────────────────────────

if [[ ! -x "$PYTHON" ]]; then
    log "Creating Python virtual environment at $VENV ..."
    python3 -m venv "$VENV"
    ok "Venv created"
fi

if ! "$PYTHON" -c "import jarvis" &>/dev/null 2>&1; then
    log "Installing JARVIS + dependencies (first run — takes 1-2 min) ..."
    cd "$APP_DIR"
    "$VENV/bin/pip" install -e ".[all]" --quiet --no-warn-script-location 2>&1 | tail -5
    ok "JARVIS installed"
fi

# ─── 4. Default config.toml ───────────────────────────────────────────────────

if [[ ! -f "$CONFIG" ]]; then
    log "Writing default config to $CONFIG ..."
    mkdir -p "$JARVIS_HOME"
    cat > "$CONFIG" <<'TOML'
# JARVIS configuration — edit to customise, then restart JARVIS.bat.

assistant_name = "Jarvis"

# Core feature flags (all safe to leave on)
internet_enabled  = true    # web search, fetch, API calls, real-time data
swarm_enabled     = true    # multi-agent council (zero cost until spawned)
workflows_enabled = true    # recordable step-based automation

# Autonomous background operation — off by default; opt in when ready
autonomy_enabled = false

# ── Optional modules ──────────────────────────────────────────────────────────
# Uncomment to enable. Each requires its optional pip extras ([vision], etc.)

# docker_enabled            = true   # pip install 'jarvis-assistant[docker]'
# knowledge_graph_enabled   = true
# fatigue_monitor_enabled   = true   # pip install 'jarvis-assistant[fatigue]'
# streaming_enabled         = true   # pip install 'jarvis-assistant[streaming]'

# LoRA calibration (fine-tune on your corrections)
# calibration_enabled       = true
# calibration_backend       = "ollama"
# calibration_base_model    = "qwen2.5:7b"
TOML
    ok "Default config written"
fi

# ─── 5. Start JARVIS web server ───────────────────────────────────────────────

WSL_IP=$(hostname -I | awk '{print $1}')
echo "$WSL_IP" > /tmp/jarvis-wsl-ip

if curl -fsS --max-time 1 "$URL/" &>/dev/null; then
    ok "JARVIS already running at $URL"
    exit 0
fi

log "Starting JARVIS web server on port 8765 ..."
cd "$APP_DIR"
# Bind to 0.0.0.0 so Windows can reach it via the WSL2 IP (localhost forwarding is unreliable)
WSL_IP=$(hostname -I | awk '{print $1}')
echo "$WSL_IP" > /tmp/jarvis-wsl-ip
setsid "$PYTHON" -m jarvis serve --host 0.0.0.0 --port 8765 >"$LOG" 2>&1 </dev/null &
SERVER_PID=$!

# ─── 6. Wait for ready (up to 20 s) ──────────────────────────────────────────

for _attempt in {1..40}; do
    if curl -fsS --max-time 1 "$URL/" &>/dev/null; then
        ok "JARVIS online → $URL"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        err "JARVIS process exited unexpectedly"
        err "--- last 20 lines of $LOG ---"
        tail -20 "$LOG" >&2 || true
        exit 1
    fi
    sleep 0.5
done

err "JARVIS did not respond after 20 seconds"
err "--- last 20 lines of $LOG ---"
tail -20 "$LOG" >&2 || true
exit 1
