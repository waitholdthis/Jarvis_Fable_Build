"""Local web UI, standard library only.

`jarvis serve` starts a single-user chat server bound to 127.0.0.1. The page
streams the agent's output over Server-Sent Events, shows tool calls as they
happen, and renders Allow/Deny buttons when a CONFIRM-tier tool needs the
user's permission — the browser equivalent of the CLI's y/N prompt.

Deliberately no web framework: http.server + SSE keeps the dependency
footprint at zero and works on any Python ≥ 3.10. This is a personal,
localhost-only interface; it binds 127.0.0.1 and serves one conversation.
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent import Agent

_KEEPALIVE_SECONDS = 15.0
_CONFIRM_TIMEOUT_SECONDS = 180.0


class WebUI:
    """Bridges one Agent to one browser: event fan-out + confirmations."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.agent.confirm = self.confirm  # route confirmations to the browser
        self.events: queue.Queue = queue.Queue()
        self._pending: dict[str, threading.Event] = {}
        self._decisions: dict[str, bool] = {}
        self._busy = threading.Lock()
        self.keepalive_seconds = _KEEPALIVE_SECONDS

    def push(self, event: dict) -> None:
        self.events.put(event)

    def reset_stream(self) -> queue.Queue:
        """A (re)connecting browser gets a fresh queue; stale readers exit."""
        self.events = queue.Queue()
        return self.events

    # ---- confirmation bridge (called from the agent worker thread) ---------

    def confirm(self, prompt: str) -> bool:
        cid = uuid.uuid4().hex
        done = threading.Event()
        self._pending[cid] = done
        self.push({"kind": "confirm", "id": cid, "prompt": prompt})
        answered = done.wait(timeout=_CONFIRM_TIMEOUT_SECONDS)
        self._pending.pop(cid, None)
        allow = self._decisions.pop(cid, False) if answered else False
        self.push({"kind": "confirm_resolved", "id": cid, "allow": allow})
        return allow

    def resolve_confirm(self, cid: str, allow: bool) -> bool:
        done = self._pending.get(cid)
        if done is None:
            return False
        self._decisions[cid] = allow
        done.set()
        return True

    # ---- turn execution -----------------------------------------------------

    def submit(self, message: str) -> bool:
        """Start a turn in a worker thread. False if a turn is running."""
        if not self._busy.acquire(blocking=False):
            return False
        threading.Thread(
            target=self._run_turn, args=(message,), daemon=True
        ).start()
        return True

    def _run_turn(self, message: str) -> None:
        try:
            for event in self.agent.run(message):
                payload = {"kind": event.kind, "text": event.text}
                if event.tool:
                    payload["tool"] = event.tool
                    payload["args"] = event.args
                self.push(payload)
        except Exception as exc:
            self.push({"kind": "error", "text": f"{type(exc).__name__}: {exc}"})
            self.push({"kind": "done", "text": ""})
        finally:
            self._busy.release()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def ui(self) -> WebUI:
        return self.server.ui  # type: ignore[attr-defined]

    def log_message(self, *args) -> None:  # silence per-request noise
        pass

    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 1_000_000:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return {}

    # ---- routes -------------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            body = _PAGE.replace(
                "{{NAME}}", self.ui.agent.config.assistant_name
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/events":
            self._stream_events()
        elif self.path == "/api/stats":
            self._send_json(self.ui.agent.memory.stats())
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path == "/api/chat":
            message = str(self._read_json().get("message", "")).strip()
            if not message:
                self._send_json({"error": "empty message"}, status=400)
            elif self.ui.submit(message):
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "a turn is already running"}, status=409)
        elif self.path == "/api/confirm":
            data = self._read_json()
            ok = self.ui.resolve_confirm(
                str(data.get("id", "")), bool(data.get("allow", False))
            )
            self._send_json({"ok": ok}, status=200 if ok else 404)
        else:
            self._send_json({"error": "not found"}, status=404)

    def _stream_events(self) -> None:
        q = self.ui.reset_stream()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while self.ui.events is q:  # a newer connection replaces us
                try:
                    item = q.get(timeout=self.ui.keepalive_seconds)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                data = json.dumps(item, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def make_server(agent: Agent, host: str = "127.0.0.1", port: int = 8765):
    server = ThreadingHTTPServer((host, port), _Handler)
    server.ui = WebUI(agent)  # type: ignore[attr-defined]
    return server


def serve(agent: Agent, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = make_server(agent, host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{NAME}}</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0d1117; color: #e6edf3;
         font: 15px/1.55 system-ui, -apple-system, sans-serif;
         display: flex; flex-direction: column; height: 100vh; }
  header { padding: 10px 16px; border-bottom: 1px solid #21262d;
           display: flex; align-items: baseline; gap: 10px; }
  header h1 { font-size: 16px; margin: 0; }
  header .sub { color: #7d8590; font-size: 12px; }
  #log { flex: 1; overflow-y: auto; padding: 16px; }
  .msg { max-width: 52rem; margin: 0 auto 12px; padding: 10px 14px;
         border-radius: 10px; white-space: pre-wrap; word-break: break-word; }
  .user { background: #1f6feb22; border: 1px solid #1f6feb55; }
  .bot  { background: #161b22; border: 1px solid #21262d; }
  .tool { max-width: 52rem; margin: 0 auto 8px; color: #7d8590;
          font: 12px/1.5 ui-monospace, monospace; white-space: pre-wrap; }
  .confirm { max-width: 52rem; margin: 0 auto 12px; padding: 10px 14px;
             border-radius: 10px; background: #3b2300; border: 1px solid #9e6a03; }
  .confirm button { margin: 6px 8px 0 0; padding: 4px 14px; border-radius: 6px;
                    border: 1px solid #30363d; cursor: pointer; }
  .allow { background: #238636; color: #fff; }
  .deny  { background: #21262d; color: #e6edf3; }
  .error { max-width: 52rem; margin: 0 auto 12px; color: #f85149; }
  form { display: flex; gap: 8px; padding: 12px 16px;
         border-top: 1px solid #21262d; }
  input[type=text] { flex: 1; padding: 10px 14px; border-radius: 10px;
                     border: 1px solid #30363d; background: #161b22;
                     color: #e6edf3; font-size: 15px; outline: none; }
  input[type=text]:focus { border-color: #1f6feb; }
  button[type=submit] { padding: 10px 20px; border-radius: 10px; border: 0;
                        background: #1f6feb; color: #fff; cursor: pointer; }
  button[type=submit]:disabled { opacity: .45; cursor: default; }
</style>
</head>
<body>
<header><h1>{{NAME}}</h1><span class="sub">local-first assistant · everything stays on this machine</span></header>
<div id="log"></div>
<form id="f">
  <input id="box" type="text" autocomplete="off"
         placeholder="Message {{NAME}}..." autofocus>
  <button id="send" type="submit">Send</button>
</form>
<script>
const log = document.getElementById('log');
const box = document.getElementById('box');
const send = document.getElementById('send');
let botEl = null;

function scroll() { log.scrollTop = log.scrollHeight; }
function add(cls, text) {
  const el = document.createElement('div');
  el.className = cls; el.textContent = text;
  log.appendChild(el); scroll(); return el;
}

function handle(ev) {
  if (ev.kind === 'text') {
    if (!botEl) botEl = add('msg bot', '');
    botEl.textContent += ev.text; scroll();
  } else if (ev.kind === 'tool_call') {
    if (botEl) {  // the fenced tool block was streamed into the bubble; hide it
      botEl.textContent =
        botEl.textContent.replace(/```tool[\\s\\S]*$/, '').trim();
      if (!botEl.textContent) botEl.remove();
    }
    botEl = null;
    add('tool', '\\u2192 ' + ev.tool + '(' + JSON.stringify(ev.args) + ')');
  } else if (ev.kind === 'tool_result') {
    const t = ev.text.length > 400 ? ev.text.slice(0, 400) + '\\u2026' : ev.text;
    add('tool', '\\u2190 ' + t);
  } else if (ev.kind === 'confirm') {
    const el = add('confirm', ev.prompt + ' ');
    for (const [label, allow] of [['Allow', true], ['Deny', false]]) {
      const b = document.createElement('button');
      b.textContent = label; b.className = allow ? 'allow' : 'deny';
      b.onclick = () => {
        fetch('/api/confirm', {method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({id: ev.id, allow})});
        el.querySelectorAll('button').forEach(x => x.disabled = true);
        el.append(' \\u2014 ' + label.toLowerCase() + 'ed');
      };
      el.appendChild(b);
    }
  } else if (ev.kind === 'error') {
    add('error', ev.text);
  } else if (ev.kind === 'done') {
    botEl = null; send.disabled = false; box.focus();
  }
}

new EventSource('/api/events').onmessage = e => handle(JSON.parse(e.data));

document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const message = box.value.trim();
  if (!message || send.disabled) return;
  add('msg user', message);
  box.value = ''; send.disabled = true;
  const r = await fetch('/api/chat', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message})});
  if (!r.ok) { add('error', 'send failed: HTTP ' + r.status); send.disabled = false; }
};
</script>
</body>
</html>
"""
