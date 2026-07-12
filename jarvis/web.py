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
from pathlib import Path

from .agent import Agent
from .ingest import ingest_path
from .llm import PROVIDER_PROFILES, cloud_client, detect
from .routines import get_routines

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

    def dashboard(self) -> dict:
        config = self.agent.config
        tools = [
            {"name": tool.name, "description": tool.description,
             "tier": tool.tier.value, "params": tool.params}
            for tool in self.agent.tools.tools.values()
        ]
        routines = [
            {"name": name, "schedule": spec.get("schedule", "manual")}
            for name, spec in get_routines(config).items()
        ]
        return {
            "assistant": config.assistant_name,
            "runtime": getattr(self.agent.llm, "runtime_name", "local"),
            "model": getattr(self.agent.llm, "model", "auto"),
            "workspace": str(config.workspace),
            "voice_enabled": config.voice_enabled,
            "provider": getattr(config, "provider", "local"),
            "providers": [
                {"id": "local", "label": "Local / Ollama", "configured": True},
                *[{"id": key, "label": spec["label"], "model": spec["model"],
                   "configured": bool(config.provider_key(key))}
                  for key, spec in PROVIDER_PROFILES.items() if key != "anthropic"],
            ],
            "tools": tools,
            "routines": routines,
            "recent": [
                {"role": role, "content": content}
                for role, content in self.agent.memory.recent(limit=8)
            ],
        }

    def action(self, action: str, value: str = "") -> tuple[dict, int]:
        if action == "clear_conversation":
            self.agent.messages.clear()
            return {"ok": True, "message": "Active conversation cleared."}, 200
        if action == "toggle_voice":
            self.agent.config.voice_enabled = not self.agent.config.voice_enabled
            state = self.agent.config.voice_enabled
            return {"ok": True, "voice_enabled": state,
                    "message": f"Voice output {'enabled' if state else 'disabled'}."}, 200
        if action == "switch_provider":
            provider = value.strip().lower()
            try:
                if provider == "local":
                    old_provider = self.agent.config.provider
                    old_model = self.agent.config.model
                    self.agent.config.provider = "local"
                    self.agent.config.model = ""
                    try:
                        client = detect(self.agent.config)
                    finally:
                        self.agent.config.model = old_model if old_provider == "local" else ""
                else:
                    client = cloud_client(self.agent.config, provider)
                self.agent.llm = client
                self.agent.config.provider = provider
                self.agent.config.model = client.model
                return {"ok": True, "provider": provider,
                        "message": f"Cognitive engine switched to {client.runtime_name} / {client.model}."}, 200
            except Exception as exc:
                return {"error": str(exc)}, 400
        if action == "forget":
            if not value.strip():
                return {"error": "Enter text to forget."}, 400
            removed = self.agent.memory.forget(value.strip())
            return {"ok": True, "removed": removed,
                    "message": f"Removed {removed} matching memory entries."}, 200
        if action == "ingest":
            if not value.strip():
                return {"error": "Enter a file or directory path."}, 400
            try:
                target = self.agent.tools.resolve_read_path(value.strip())
            except Exception as exc:
                return {"error": str(exc)}, 403
            if not target.exists():
                return {"error": f"Path not found: {target}"}, 404
            files, chunks = ingest_path(self.agent.memory, Path(target))
            return {"ok": True, "files": files, "chunks": chunks,
                    "message": f"Indexed {files} files and {chunks} chunks."}, 200
        if action == "run_routine":
            routine = get_routines(self.agent.config).get(value)
            if routine is None:
                return {"error": f"Unknown routine: {value}"}, 404
            if not self.submit(routine["prompt"]):
                return {"error": "JARVIS is already processing a directive."}, 409
            return {"ok": True, "streaming": True,
                    "message": f"Routine {value} initiated."}, 202
        return {"error": f"Unknown action: {action}"}, 404

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
        elif self.path == "/api/dashboard":
            self._send_json(self.ui.dashboard())
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
        elif self.path == "/api/action":
            data = self._read_json()
            payload, status = self.ui.action(
                str(data.get("action", "")), str(data.get("value", ""))
            )
            self._send_json(payload, status=status)
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
  :root { color-scheme: dark; --cyan:#58e6ff; --ice:#b9f6ff; --dim:#5c8590;
    --line:rgba(88,230,255,.18); --panel:rgba(5,18,25,.7); --amber:#ffb45b; }
  * { box-sizing:border-box; }
  html,body { margin:0; min-height:100%; overflow:hidden; }
  body { background:#02070a; color:#d8f7fb; font:14px/1.55 Inter,system-ui,sans-serif; }
  body:before { content:""; position:fixed; inset:0; pointer-events:none; z-index:20; opacity:.22;
    background:repeating-linear-gradient(0deg,transparent 0 3px,rgba(121,223,255,.025) 4px); }
  .world { position:relative; height:100dvh; overflow:hidden; isolation:isolate;
    background:radial-gradient(circle at 50% 45%,rgba(13,76,91,.18),transparent 32%),
      radial-gradient(ellipse at 50% 100%,#09202a 0,transparent 52%),#02070a; }
  .grid { position:absolute; inset:48% -20% -30%; z-index:-1; opacity:.28; transform:perspective(500px) rotateX(62deg);
    background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px);
    background-size:50px 50px; mask-image:linear-gradient(to bottom,transparent,#000 35%); }
  .flare { position:absolute; width:55vw; height:55vw; left:22.5vw; top:10%; z-index:-1; border-radius:50%;
    background:radial-gradient(circle,rgba(66,214,255,.06),transparent 65%); filter:blur(20px); }
  header { height:72px; padding:0 30px; display:flex; align-items:center; justify-content:space-between;
    border-bottom:1px solid var(--line); background:linear-gradient(90deg,rgba(3,15,20,.88),rgba(5,23,30,.48),rgba(3,15,20,.88)); }
  .brand { display:flex; align-items:center; gap:14px; }
  .mark { width:27px; height:27px; border:1px solid var(--cyan); transform:rotate(45deg); position:relative; box-shadow:0 0 18px #38dfff55; }
  .mark:after { content:""; position:absolute; inset:6px; border:1px solid var(--ice); background:#7beeff55; }
  h1 { margin:0; font:600 15px/1 Space Mono,monospace; letter-spacing:.32em; color:#e3fcff; }
  .eyebrow,.mono { font:10px/1.4 Space Mono,monospace; letter-spacing:.18em; text-transform:uppercase; color:var(--dim); }
  .status { display:flex; gap:24px; align-items:center; }
  .live { color:#9df3d4; } .live:before { content:""; display:inline-block; width:5px; height:5px; margin-right:8px; border-radius:50%; background:#6fffc9; box-shadow:0 0 10px #6fffc9; }
  main { height:calc(100dvh - 72px); display:grid; grid-template-columns:minmax(170px,240px) minmax(420px,1fr) minmax(190px,260px); gap:22px; padding:22px 28px 25px; }
  .rail { min-height:0; display:flex; flex-direction:column; gap:14px; }
  .panel { position:relative; border:1px solid var(--line); background:linear-gradient(145deg,rgba(8,26,34,.72),rgba(2,10,14,.54)); backdrop-filter:blur(12px); }
  .panel:before,.panel:after { content:""; position:absolute; width:9px; height:9px; border-color:var(--cyan); opacity:.7; }
  .panel:before { top:-1px; left:-1px; border-top:1px solid; border-left:1px solid; }
  .panel:after { right:-1px; bottom:-1px; border-right:1px solid; border-bottom:1px solid; }
  .panel-title { padding:12px 14px 10px; border-bottom:1px solid rgba(88,230,255,.09); }
  .datum { padding:13px 14px; border-bottom:1px solid rgba(88,230,255,.07); }
  .datum:last-child { border:0; } .datum strong { display:block; color:#d9faff; font:500 12px Space Mono,monospace; margin-top:4px; }
  .meter { height:2px; background:#132c34; margin-top:9px; overflow:hidden; } .meter i { display:block; height:100%; background:var(--cyan); box-shadow:0 0 8px var(--cyan); }
  .core-wrap { flex:1; min-height:210px; display:grid; place-items:center; overflow:hidden; }
  .core { position:relative; width:min(170px,70%); aspect-ratio:1; display:grid; place-items:center; }
  .ring { position:absolute; inset:0; border:1px solid rgba(88,230,255,.28); border-radius:50%; box-shadow:0 0 30px rgba(36,217,255,.06) inset; animation:spin 22s linear infinite; }
  .ring:before,.ring:after { content:""; position:absolute; inset:10%; border-radius:50%; border:1px dashed rgba(109,233,255,.32); }
  .ring:after { inset:24%; border-style:solid; border-color:rgba(88,230,255,.14); }
  .core.active .ring { animation-duration:3s; border-color:rgba(88,230,255,.65); box-shadow:0 0 35px rgba(88,230,255,.16) inset; }
  .orb { width:78%; aspect-ratio:1; border-radius:50%; filter:drop-shadow(0 0 14px #5ceaff) drop-shadow(0 0 35px #21caef66); animation:pulse 3.4s ease-in-out infinite; cursor:crosshair; }
  .orb-fallback { background:radial-gradient(circle at 40% 35%,#e8fdff 0,#71efff 8%,#128ba8 30%,#03151c 64%); box-shadow:0 0 18px #5ceaff,0 0 65px #21caef88; }
  .core-label { position:absolute; bottom:-3px; text-align:center; }
  @keyframes spin { to{transform:rotate(360deg)} } @keyframes pulse { 50%{transform:scale(1.07);filter:brightness(1.25)} }
  .conversation { min-height:0; display:flex; flex-direction:column; position:relative; }
  .conversation-head { height:46px; display:flex; align-items:center; justify-content:space-between; padding:0 17px; border-bottom:1px solid var(--line); }
  #log { flex:1; overflow:auto; padding:22px 24px 10px; scrollbar-width:thin; scrollbar-color:#1b6474 transparent; }
  .welcome { height:100%; display:grid; align-content:center; justify-items:center; text-align:center; color:#688f98; }
  .welcome .hello { color:#dffbff; font:300 clamp(20px,3vw,38px) Inter,sans-serif; letter-spacing:.03em; margin:15px 0 5px; }
  .welcome .line { width:70px; height:1px; background:var(--cyan); box-shadow:0 0 12px var(--cyan); }
  .msg { max-width:82%; margin:0 0 18px; padding:13px 15px; white-space:pre-wrap; word-break:break-word; border-left:1px solid var(--cyan); background:linear-gradient(90deg,rgba(23,90,106,.16),transparent); animation:arrive .35s ease-out; }
  .msg:before { display:block; margin-bottom:6px; font:9px Space Mono,monospace; letter-spacing:.18em; color:#52aaba; }
  .user { margin-left:auto; border-left:0; border-right:1px solid #6d9fa9; text-align:right; background:linear-gradient(270deg,rgba(70,112,122,.13),transparent); color:#bed7dc; }
  .user:before { content:"OPERATOR"; } .bot:before { content:"JARVIS // RESPONSE"; color:var(--cyan); }
  @keyframes arrive { from{opacity:0;transform:translateY(8px)} }
  .tool,.error,.notify,.confirm { max-width:92%; margin:0 0 12px; padding:10px 13px; font:11px/1.5 Space Mono,monospace; white-space:pre-wrap; border:1px solid var(--line); background:#051318bb; color:#78bac7; }
  .tool:before { content:"SYS  "; color:var(--cyan); } .error { color:#ff8b85;border-color:#ff6d6644; }
  .notify { color:#8cf2cc;border-color:#62eeb055; } .confirm { color:#ffd9a8;border-color:#ffb45b66;background:#251807cc; }
  .confirm button { margin:9px 8px 0 0; padding:6px 14px; border:1px solid currentColor; background:transparent; font:10px Space Mono,monospace; text-transform:uppercase; cursor:pointer; }
  .allow { color:#80f3c5; }.deny { color:#ff938c; }
  form { margin:0 18px 18px; min-height:58px; display:flex; align-items:center; gap:10px; border:1px solid rgba(88,230,255,.25); background:rgba(2,11,15,.83); padding:8px 9px 8px 16px; box-shadow:0 0 28px rgba(25,196,229,.04); }
  input[type=text] { min-width:0; flex:1; border:0; outline:0; background:transparent; color:#dffbff; font:14px Inter,sans-serif; }
  input::placeholder { color:#456c75; } button[type=submit] { width:42px;height:38px;border:1px solid #58e6ff66;background:#0b3340;color:var(--cyan);cursor:pointer;font-size:17px; }
  button[type=submit]:hover { background:#115064;box-shadow:0 0 20px #28dfff33; } button:disabled{opacity:.35;cursor:default;}
  .voice-button { position:relative; width:42px; height:38px; border:1px solid rgba(88,230,255,.28); background:#071b22; color:#83bbc5; cursor:pointer; display:grid; place-items:center; font-size:16px; }
  .voice-button:hover { color:var(--cyan); border-color:var(--cyan); } .voice-button.active { color:#071013; background:#62e8ff; border-color:#b8f7ff; box-shadow:0 0 22px #43dfff88; animation:micpulse 1.2s ease-in-out infinite; }
  .voice-button.speaking { color:#8fffd5; border-color:#62f2bd; box-shadow:0 0 16px #43e6ae55; }
  @keyframes micpulse { 50% { box-shadow:0 0 34px #43dfffcc; } }
  .voice-unavailable { display:none; }
  .quick { display:grid; gap:8px; padding:12px; } .quick button { text-align:left; color:#8db8c1; border:1px solid rgba(88,230,255,.1); background:rgba(10,35,43,.35); padding:9px 10px; font:10px Space Mono,monospace; cursor:pointer; }
  .quick button:hover { color:var(--cyan);border-color:rgba(88,230,255,.4); }
  .clock { padding:15px; font:300 26px Space Mono,monospace; color:#dffaff; } .clock small { display:block;font-size:9px;color:var(--dim);letter-spacing:.15em; }
  .nav-button { border:1px solid var(--line); background:#08212a; color:var(--cyan); padding:8px 12px; font:10px Space Mono,monospace; letter-spacing:.12em; cursor:pointer; }
  .dashboard { position:fixed; inset:72px 0 0; z-index:15; display:none; background:rgba(1,7,10,.96); backdrop-filter:blur(18px); padding:22px 28px 28px; overflow:auto; }
  .dashboard.open { display:block; animation:arrive .25s ease-out; }
  .dash-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:18px; }
  .dash-head h2 { margin:4px 0 0; font:300 25px Inter,sans-serif; letter-spacing:.08em; }
  .dash-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; max-width:1500px; margin:auto; }
  .module { grid-column:span 4; min-height:190px; } .module.wide { grid-column:span 8; } .module.full { grid-column:1/-1; }
  .module-body { padding:14px; } .module-copy { color:#749ba4; font-size:12px; margin:0 0 14px; }
  .control-row { display:flex; gap:8px; margin-top:9px; }
  .control-row input { min-width:0; flex:1; padding:9px 10px; color:#dffaff; background:#031117; border:1px solid var(--line); outline:0; }
  .action { padding:8px 11px; border:1px solid rgba(88,230,255,.32); color:var(--cyan); background:#09252e; font:10px Space Mono,monospace; cursor:pointer; }
  .action.danger { color:#ff938c; border-color:#ff938c55; background:#271013; }
  .tool-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:8px; max-height:300px; overflow:auto; }
  .tool-card { padding:10px; border:1px solid rgba(88,230,255,.1); background:#06161b; }
  .tool-card strong { display:block; color:#bceff6; font:11px Space Mono,monospace; } .tool-card span { color:#638b94; font-size:10px; }
  .badge { float:right; padding:2px 5px; border:1px solid #55d7ae55; color:#74eac2; font:8px Space Mono,monospace; } .badge.confirm { color:var(--amber);border-color:#ffb45b55; }
  .routine { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 0; border-bottom:1px solid rgba(88,230,255,.08); }
  .runtime-card { display:grid; grid-template-columns:1fr 1fr; gap:12px; } .runtime-card strong { display:block; margin-top:4px; font:11px Space Mono,monospace; color:#c9f7fc; overflow-wrap:anywhere; }
  .provider-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:8px; } .provider { padding:10px; border:1px solid rgba(88,230,255,.12); background:#05171d; color:#8db8c1; text-align:left; cursor:pointer; } .provider strong{display:block;color:#caf5fa;font:11px Space Mono,monospace}.provider small{font:9px Space Mono,monospace;color:#527b84}.provider.active{border-color:var(--cyan);box-shadow:0 0 15px #3ae3ff18}.provider:disabled{opacity:.35;cursor:not-allowed}
  #toast { position:fixed; z-index:30; right:25px; bottom:25px; max-width:360px; padding:12px 15px; border:1px solid var(--cyan); background:#061a21; color:#c9f8ff; font:11px Space Mono,monospace; opacity:0; transform:translateY(8px); pointer-events:none; transition:.2s; } #toast.show{opacity:1;transform:none;}
  @media(max-width:900px){ main{grid-template-columns:1fr minmax(0,2.3fr);padding:14px}.right{display:none}.status .mono{display:none} }
  @media(max-width:900px){.module,.module.wide{grid-column:span 6}}
  @media(max-width:620px){ header{height:61px;padding:0 17px}.status{gap:8px}main{height:calc(100dvh - 61px);display:block;padding:10px}.left{display:none}.conversation{height:100%}#log{padding:18px 14px}.msg{max-width:92%}form{margin:0 10px 10px}h1{font-size:13px}.eyebrow{display:none}.dashboard{inset:61px 0 0;padding:15px}.module,.module.wide{grid-column:1/-1}.dash-head h2{font-size:19px} }
  @media(prefers-reduced-motion:reduce){*{animation:none!important;scroll-behavior:auto!important}}
</style>
</head>
<body>
<div class="world"><div class="grid"></div><div class="flare"></div>
<header><div class="brand"><div class="mark"></div><div><h1>{{NAME}}</h1><div class="eyebrow">Cognitive interface // Mark VII</div></div></div><div class="status"><span class="mono">Encrypted local channel</span><span class="mono live">System online</span><button class="nav-button" id="dashboard-toggle">COMMAND CENTER</button></div></header>
<main>
 <aside class="rail left">
  <section class="panel"><div class="panel-title eyebrow">System integrity</div><div class="datum"><span class="eyebrow">Neural engine</span><strong id="engine">STANDBY</strong><div class="meter"><i style="width:82%"></i></div></div><div class="datum"><span class="eyebrow">Privacy protocol</span><strong>LOCAL // SECURE</strong></div></section>
  <section class="panel core-wrap"><div class="core" id="core"><div class="ring"></div><canvas class="orb" id="orb-canvas" aria-label="JARVIS animated cognition core"></canvas><div class="core-label eyebrow">Arc cognition core</div></div></section>
  <section class="panel"><div class="clock" id="clock">--:--<small>LOCAL NODE TIME</small></div></section>
 </aside>
 <section class="panel conversation"><div class="conversation-head"><span class="eyebrow">Active dialogue</span><span class="eyebrow" id="mode">Awaiting directive</span></div><div id="log"><div class="welcome" id="welcome"><div class="line"></div><div class="hello">Good evening.</div><div>All systems are standing by.<br>How may I assist you?</div></div></div><form id="f"><span class="eyebrow">CMD</span><input id="box" type="text" autocomplete="off" placeholder="Issue a directive..." autofocus><button id="speaker" class="voice-button" type="button" aria-label="Toggle spoken replies" title="Spoken replies enabled">◖))</button><button id="mic" class="voice-button" type="button" aria-label="Speak to JARVIS" title="Click to speak">●</button><button id="send" type="submit" aria-label="Send directive">⌁</button></form></section>
 <aside class="rail right">
  <section class="panel"><div class="panel-title eyebrow">Memory matrix</div><div class="datum"><span class="eyebrow">Semantic records</span><strong id="semantic">—</strong></div><div class="datum"><span class="eyebrow">Episodes indexed</span><strong id="episodic">—</strong></div></section>
  <section class="panel"><div class="panel-title eyebrow">Rapid directives</div><div class="quick"><button data-prompt="Run diagnostics and report whether every JARVIS subsystem is functioning correctly.">01 // RUN DIAGNOSTICS</button><button data-prompt="Open the situation room and give me a concise operational briefing.">02 // SITUATION ROOM</button><button data-prompt="Show mission control and identify the highest-leverage next checkpoint.">03 // MISSION CONTROL</button><button data-prompt="Run workspace radar on the workspace and summarize what changed.">04 // WORKSPACE RADAR</button></div></section>
  <section class="panel" style="flex:1"><div class="panel-title eyebrow">Telemetry</div><div class="datum"><span class="eyebrow">Network boundary</span><strong>AIR-GAP READY</strong></div><div class="datum"><span class="eyebrow">Tool broker</span><strong>POLICY GATED</strong></div><div class="datum"><span class="eyebrow">Stream state</span><strong id="stream">CONNECTED</strong></div></section>
 </aside>
</main>
<section class="dashboard" id="dashboard" aria-hidden="true"><div class="dash-head"><div><div class="eyebrow">JARVIS operations suite</div><h2>Command Center</h2></div><button class="nav-button" id="dashboard-close">RETURN TO DIALOGUE</button></div><div class="dash-grid">
 <article class="panel module"><div class="panel-title eyebrow">Runtime intelligence</div><div class="module-body runtime-card"><div><span class="eyebrow">Runtime</span><strong id="d-runtime">—</strong></div><div><span class="eyebrow">Model</span><strong id="d-model">—</strong></div><div style="grid-column:1/-1"><span class="eyebrow">Secure workspace</span><strong id="d-workspace">—</strong></div></div></article>
 <article class="panel module wide"><div class="panel-title eyebrow">Cognitive engine router</div><div class="module-body"><p class="module-copy">Switch JARVIS's reasoning engine while retaining the same memory, missions, tools, and personality. Keys remain in environment variables and are never returned to this page.</p><div class="provider-grid" id="provider-grid"></div></div></article>
 <article class="panel module"><div class="panel-title eyebrow">Voice systems</div><div class="module-body"><p class="module-copy">Control spoken-response output. Microphone capture remains available through the native voice interface.</p><strong class="mono" id="voice-state">VOICE OUTPUT —</strong><div class="control-row"><button class="action" data-action="toggle_voice">TOGGLE VOICE OUTPUT</button></div></div></article>
 <article class="panel module"><div class="panel-title eyebrow">Session control</div><div class="module-body"><p class="module-copy">Reset short-term dialogue context while preserving JARVIS's long-term memory.</p><button class="action danger" data-action="clear_conversation">CLEAR ACTIVE CONVERSATION</button></div></article>
 <article class="panel module wide"><div class="panel-title eyebrow">Knowledge ingestion</div><div class="module-body"><p class="module-copy">Index a text file or directory from your home folder into semantic memory.</p><div class="control-row"><input id="ingest-path" placeholder="~/Documents/notes"><button class="action" id="ingest-button">INDEX PATH</button></div></div></article>
 <article class="panel module"><div class="panel-title eyebrow">Memory redaction</div><div class="module-body"><p class="module-copy">Permanently remove memory entries matching a phrase or source.</p><div class="control-row"><input id="forget-text" placeholder="Text to forget"><button class="action danger" id="forget-button">FORGET</button></div></div></article>
 <article class="panel module"><div class="panel-title eyebrow">Automation routines</div><div class="module-body" id="routine-list"></div></article>
 <article class="panel module wide"><div class="panel-title eyebrow">Tool broker // Available capabilities</div><div class="module-body"><div class="tool-grid" id="tool-grid"></div></div></article>
</div></section><div id="toast"></div></div>
<script>
const log = document.getElementById('log');
const box = document.getElementById('box');
const send = document.getElementById('send');
const core = document.getElementById('core');
const mic = document.getElementById('mic');
const speaker = document.getElementById('speaker');
let botEl = null;
let dashboardData = null;
let recognition = null;
let listening = false;
let spokenReplies = localStorage.getItem('jarvis-spoken-replies') !== 'off';

function initOrb(){
  const canvas=document.getElementById('orb-canvas');
  const gl=canvas.getContext('webgl2',{alpha:true,premultipliedAlpha:false,antialias:true});
  if(!gl){canvas.classList.add('orb-fallback');return;}
  const vert=`#version 300 es
    precision highp float; out vec2 uv;
    void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p;gl_Position=vec4(p*2.0-1.0,0.0,1.0);}`;
  const frag=`#version 300 es
    precision highp float; in vec2 uv; out vec4 outColor;
    uniform vec2 resolution; uniform float time; uniform float hover; uniform float energy; uniform float rotation;
    float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1)),f.x),f.y);}
    float fbm(vec2 p){float v=0.0,a=.5;mat2 m=mat2(1.6,1.2,-1.2,1.6);for(int i=0;i<5;i++){v+=a*noise(p);p=m*p+.17;a*=.5;}return v;}
    void main(){
      vec2 p=(uv-.5)*2.0;p.x*=resolution.x/resolution.y;
      float c=cos(rotation),s=sin(rotation);p=mat2(c,-s,s,c)*p;
      float r=length(p),ang=atan(p.y,p.x);
      float warp=fbm(p*2.1+vec2(time*.13,-time*.1));
      float edge=.69+.075*sin(ang*3.0-time*1.2+warp*5.0)+.045*sin(ang*7.0+time*.8);
      p+=hover*.055*vec2(sin(p.y*11.0+time*2.0),cos(p.x*10.0-time*1.7));
      float field=fbm(p*3.0+vec2(time*.22,-time*.16));
      float plasma=.5+.5*sin(field*8.0-ang*2.0+time*1.15);
      vec3 cyan=vec3(.14,.88,1.0),ice=vec3(.72,.98,1.0),violet=vec3(.35,.16,.94),navy=vec3(.01,.035,.16);
      vec3 col=mix(violet,cyan,plasma);col=mix(navy,col,smoothstep(.05,.69,r));
      float rim=exp(-abs(r-edge)*25.0)*(1.0+energy*.9);
      float inner=exp(-r*3.3)*(.5+.55*field);
      float spark=pow(max(0.0,1.0-length(p-vec2(cos(time)*.43,sin(time)*.43))),16.0);
      col+=ice*(rim*1.45+inner*.55+spark*1.7);col*=.72+energy*.32+hover*.16;
      float alpha=smoothstep(edge+.075,edge-.025,r)*(.76+rim*.35);alpha*=smoothstep(.03,.16,r);
      outColor=vec4(col*alpha,alpha);
    }`;
  function makeShader(type,source){const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(shader));return shader;}
  try{
    const program=gl.createProgram();gl.attachShader(program,makeShader(gl.VERTEX_SHADER,vert));gl.attachShader(program,makeShader(gl.FRAGMENT_SHADER,frag));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));gl.useProgram(program);
    const uniforms={resolution:gl.getUniformLocation(program,'resolution'),time:gl.getUniformLocation(program,'time'),hover:gl.getUniformLocation(program,'hover'),energy:gl.getUniformLocation(program,'energy'),rotation:gl.getUniformLocation(program,'rotation')};
    let targetHover=0,currentHover=0,rot=0,last=performance.now();const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
    canvas.onpointermove=e=>{const b=canvas.getBoundingClientRect(),x=(e.clientX-b.left)/b.width-.5,y=(e.clientY-b.top)/b.height-.5;targetHover=Math.hypot(x,y)<.42?1:0;};canvas.onpointerleave=()=>targetHover=0;
    function resize(){const d=Math.min(devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*d)),h=Math.max(1,Math.round(canvas.clientHeight*d));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h);}}
    function draw(now){resize();const dt=Math.min((now-last)/1000,.05);last=now;currentHover+=(targetHover-currentHover)*.08;if(currentHover>.5)rot+=dt*.35;const active=core.classList.contains('active')?1:0;gl.uniform2f(uniforms.resolution,canvas.width,canvas.height);gl.uniform1f(uniforms.time,reduced?0:now/1000);gl.uniform1f(uniforms.hover,currentHover);gl.uniform1f(uniforms.energy,active);gl.uniform1f(uniforms.rotation,rot);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);gl.drawArrays(gl.TRIANGLES,0,3);requestAnimationFrame(draw);}
    requestAnimationFrame(draw);
  }catch(error){console.warn('Orb shader unavailable',error);canvas.classList.add('orb-fallback');}
}
initOrb();

function scroll() { log.scrollTop = log.scrollHeight; }
function add(cls, text) {
  document.getElementById('welcome')?.remove();
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
  } else if (ev.kind === 'notify') {
    const el = add('notify', '');
    const b = document.createElement('b');
    b.textContent = '\\ud83d\\udd14 ' + (ev.title || 'Notification') + ' ';
    el.appendChild(b);
    el.append(ev.text || '');
  } else if (ev.kind === 'error') {
    add('error', ev.text);
  } else if (ev.kind === 'done') {
    botEl = null; send.disabled = false; core.classList.remove('active');
    document.getElementById('engine').textContent='STANDBY';
    document.getElementById('mode').textContent='AWAITING DIRECTIVE';
    if (spokenReplies && ev.text) speakReply(ev.text); else box.focus();
  }
}

const events = new EventSource('/api/events');
events.onmessage = e => handle(JSON.parse(e.data));
events.onerror = () => document.getElementById('stream').textContent='RECONNECTING';
events.onopen = () => document.getElementById('stream').textContent='CONNECTED';

function updateClock(){document.getElementById('clock').firstChild.textContent=new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'})+' ';}
updateClock(); setInterval(updateClock,1000);
fetch('/api/stats').then(r=>r.json()).then(s=>{document.getElementById('semantic').textContent=String(s.semantic_chunks??0).padStart(4,'0');document.getElementById('episodic').textContent=String(s.episodic_entries??0).padStart(4,'0');}).catch(()=>{});
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{box.value=b.dataset.prompt;box.focus();});

function setSpeakerState(){speaker.classList.toggle('speaking',spokenReplies);speaker.title=spokenReplies?'Spoken replies enabled':'Spoken replies muted';speaker.setAttribute('aria-pressed',String(spokenReplies));}
function speakReply(text){
  if (!('speechSynthesis' in window)) return;
  speechSynthesis.cancel(); const utterance=new SpeechSynthesisUtterance(text);
  utterance.rate=.96; utterance.pitch=.88;
  const voices=speechSynthesis.getVoices(); const preferred=voices.find(v=>/Daniel|Google UK English Male|Microsoft David|Alex/i.test(v.name))||voices.find(v=>v.lang?.startsWith('en'));
  if(preferred) utterance.voice=preferred;
  utterance.onstart=()=>{speaker.classList.add('active');core.classList.add('active');document.getElementById('mode').textContent='VOICE OUTPUT ACTIVE';};
  utterance.onend=utterance.onerror=()=>{speaker.classList.remove('active');core.classList.remove('active');document.getElementById('mode').textContent='AWAITING DIRECTIVE';box.focus();};
  speechSynthesis.speak(utterance);
}
speaker.onclick=()=>{spokenReplies=!spokenReplies;localStorage.setItem('jarvis-spoken-replies',spokenReplies?'on':'off');if(!spokenReplies&&'speechSynthesis' in window)speechSynthesis.cancel();setSpeakerState();toast(spokenReplies?'Spoken replies enabled.':'Spoken replies muted.');};
setSpeakerState();

const SpeechRecognition=window.SpeechRecognition||window.webkitSpeechRecognition;
if(SpeechRecognition){
  recognition=new SpeechRecognition(); recognition.lang=navigator.language||'en-US'; recognition.interimResults=true; recognition.continuous=false;
  recognition.onstart=()=>{listening=true;mic.classList.add('active');core.classList.add('active');mic.setAttribute('aria-pressed','true');box.placeholder='Listening…';document.getElementById('mode').textContent='MICROPHONE ACTIVE // SPEAK NOW';if('speechSynthesis' in window)speechSynthesis.cancel();};
  recognition.onresult=e=>{let finalText='',interim='';for(let i=e.resultIndex;i<e.results.length;i++){const t=e.results[i][0].transcript;if(e.results[i].isFinal)finalText+=t;else interim+=t;}box.value=finalText||interim;if(finalText.trim())setTimeout(()=>document.getElementById('f').requestSubmit(),120);};
  recognition.onerror=e=>{if(e.error!=='aborted')toast(e.error==='not-allowed'?'Microphone permission was denied. Allow microphone access in your browser settings.':'Microphone error: '+e.error,true);};
  recognition.onend=()=>{listening=false;mic.classList.remove('active');if(!send.disabled)core.classList.remove('active');mic.setAttribute('aria-pressed','false');box.placeholder='Issue a directive...';if(!send.disabled)document.getElementById('mode').textContent='AWAITING DIRECTIVE';};
  mic.onclick=()=>{if(listening)recognition.stop();else{try{recognition.start();}catch(e){toast('Microphone is already active.',true);}}};
}else{
  mic.disabled=true;mic.classList.add('voice-unavailable');mic.title='Speech recognition is not supported by this browser';
}

function toast(message, bad=false){const t=document.getElementById('toast');t.textContent=message;t.style.borderColor=bad?'#ff7c75':'var(--cyan)';t.classList.add('show');setTimeout(()=>t.classList.remove('show'),3200);}
async function loadDashboard(){
  const r=await fetch('/api/dashboard'); dashboardData=await r.json();
  document.getElementById('d-runtime').textContent=dashboardData.runtime||'LOCAL';
  document.getElementById('d-model').textContent=dashboardData.model||'AUTO-DETECT';
  document.getElementById('d-workspace').textContent=dashboardData.workspace;
  document.getElementById('voice-state').textContent='VOICE OUTPUT '+(dashboardData.voice_enabled?'ENABLED':'DISABLED');
  document.getElementById('provider-grid').innerHTML=dashboardData.providers.map(p=>`<button class="provider ${p.id===dashboardData.provider?'active':''}" data-provider="${escapeHTML(p.id)}" ${p.configured?'':'disabled'}><strong>${escapeHTML(p.label)}</strong><small>${p.configured?(p.model||'AUTO DETECT'):'KEY NOT CONFIGURED'}</small></button>`).join('');
  document.querySelectorAll('[data-provider]').forEach(b=>b.onclick=()=>runAction('switch_provider',b.dataset.provider));
  document.getElementById('tool-grid').innerHTML=dashboardData.tools.map(t=>`<div class="tool-card"><span class="badge ${t.tier==='confirm'?'confirm':''}">${t.tier}</span><strong>${escapeHTML(t.name)}</strong><span>${escapeHTML(t.description)}</span></div>`).join('');
  document.getElementById('routine-list').innerHTML=dashboardData.routines.map(r=>`<div class="routine"><div><strong class="mono">${escapeHTML(r.name)}</strong><div class="eyebrow">${escapeHTML(r.schedule)}</div></div><button class="action routine-run" data-name="${escapeHTML(r.name)}">RUN</button></div>`).join('');
  document.querySelectorAll('.routine-run').forEach(b=>b.onclick=()=>runAction('run_routine',b.dataset.name));
}
function escapeHTML(v){const d=document.createElement('div');d.textContent=String(v);return d.innerHTML;}
async function runAction(action,value=''){
  const r=await fetch('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,value})});
  const data=await r.json(); toast(data.message||data.error,!r.ok);
  if(data.streaming){document.getElementById('dashboard').classList.remove('open');send.disabled=true;core.classList.add('active');}
  await loadDashboard(); fetch('/api/stats').then(r=>r.json()).then(s=>{document.getElementById('semantic').textContent=String(s.semantic_chunks??0).padStart(4,'0');document.getElementById('episodic').textContent=String(s.episodic_entries??0).padStart(4,'0');});
}
document.getElementById('dashboard-toggle').onclick=()=>{const d=document.getElementById('dashboard');d.classList.add('open');d.setAttribute('aria-hidden','false');loadDashboard().catch(()=>toast('Dashboard data unavailable',true));};
document.getElementById('dashboard-close').onclick=()=>{const d=document.getElementById('dashboard');d.classList.remove('open');d.setAttribute('aria-hidden','true');box.focus();};
document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>runAction(b.dataset.action));
document.getElementById('ingest-button').onclick=()=>runAction('ingest',document.getElementById('ingest-path').value);
document.getElementById('forget-button').onclick=()=>runAction('forget',document.getElementById('forget-text').value);

document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const message = box.value.trim();
  if (!message || send.disabled) return;
  add('msg user', message);
  box.value = ''; send.disabled = true; core.classList.add('active');
  document.getElementById('engine').textContent='PROCESSING';
  document.getElementById('mode').textContent='COGNITIVE STREAM ACTIVE';
  const r = await fetch('/api/chat', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message})});
  if (!r.ok) { add('error', 'send failed: HTTP ' + r.status); send.disabled = false; }
};
</script>
</body>
</html>
"""
