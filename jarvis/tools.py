"""Tool registry and built-in tools, with a risk-tier policy.

Every side effect the model can cause goes through this registry — the
"broker" from the architecture spec, scaled down to one process:

  SAFE     — read-only or scoped to the Jarvis workspace; auto-allowed.
  CONFIRM  — touches the wider system (shell, network, writes outside the
             workspace); requires explicit user confirmation per call.

File writes are confined to the workspace directory. File reads are confined
to the user's home tree. Shell commands always require confirmation and run
with a timeout. There is deliberately no "sudo tier": anything beyond this
should be done by the human.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import json
import platform
import re
import shutil
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from .config import Config
from .memory import Memory


class Tier(str, Enum):
    SAFE = "safe"
    CONFIRM = "confirm"


class PolicyError(PermissionError):
    """A tool call violated the sandbox policy."""


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # arg name -> human description
    func: Callable[..., str]
    tier: Tier = Tier.SAFE

    def signature(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.params.items())
        return f"{self.name}({args})"


@dataclass
class ToolRegistry:
    config: Config
    memory: Memory
    tools: dict[str, Tool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _register_builtins(self)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def describe_all(self) -> str:
        lines = []
        for tool in self.tools.values():
            flag = " (asks permission)" if tool.tier is Tier.CONFIRM else ""
            lines.append(f"- {tool.signature()}: {tool.description}{flag}")
        return "\n".join(lines)

    def run(self, name: str, args: dict) -> str:
        tool = self.get(name)
        if tool is None:
            known = ", ".join(sorted(self.tools))
            return f"ERROR: unknown tool '{name}'. Available tools: {known}"
        unknown_args = set(args) - set(tool.params)
        if unknown_args:
            return f"ERROR: unknown argument(s) {sorted(unknown_args)} for {tool.name}"
        try:
            result = tool.func(**args)
        except PolicyError as exc:
            return f"DENIED by policy: {exc}"
        except TypeError as exc:
            return f"ERROR: bad arguments for {tool.name}: {exc}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        return result if result else "(no output)"

    # ---- path policy -------------------------------------------------------

    def resolve_read_path(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.config.workspace / p
        p = p.resolve()
        home = Path.home().resolve()
        workspace = self.config.workspace.resolve()
        allowed = any(
            p == root or root in p.parents for root in (home, workspace)
        )
        if not allowed:
            raise PolicyError(
                f"reads are restricted to your home directory ({home}) "
                f"and the workspace ({workspace})"
            )
        return p

    def resolve_write_path(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.config.workspace / p
        p = p.resolve()
        workspace = self.config.workspace.resolve()
        if not (p == workspace or workspace in p.parents):
            raise PolicyError(
                f"writes are restricted to the workspace ({workspace}); "
                f"ask the user to move or copy files themselves"
            )
        return p


def parse_ddg_results(page: str, limit: int = 6) -> list[tuple[str, str, str]]:
    """Extract (title, url, snippet) triples from DuckDuckGo's HTML endpoint."""
    titles: list[tuple[str, str]] = []
    for m in re.finditer(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        page,
        re.DOTALL,
    ):
        url = _html.unescape(m.group(1))
        # DDG wraps results in a redirect: //duckduckgo.com/l/?uddg=<encoded>
        redirect = re.search(r"[?&]uddg=([^&]+)", url)
        if redirect:
            url = urllib.parse.unquote(redirect.group(1))
        title = _html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        titles.append((title, url))

    snippets = [
        _html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
        for m in re.finditer(
            r'class="result__snippet"[^>]*>(.*?)</a>', page, re.DOTALL
        )
    ]
    results = []
    for i, (title, url) in enumerate(titles[:limit]):
        snippet = snippets[i] if i < len(snippets) else ""
        results.append((title, url, snippet))
    return results


def register_scheduler_tools(reg: ToolRegistry, scheduler) -> None:
    """Reminder/automation tools; registered once a Scheduler exists."""
    from .scheduler import parse_when

    def schedule_task(when: str, message: str) -> str:
        parsed = parse_when(when)
        if parsed is None:
            return (
                f"ERROR: could not parse '{when}'. Supported forms: "
                "'in 20 minutes', 'at 18:30', 'tomorrow at 9:00', "
                "'every 30 minutes', 'every day at 08:00'"
            )
        due, repeat = parsed
        task_id = scheduler.add(due, payload=message, repeat=repeat)
        due_str = _dt.datetime.fromtimestamp(due).strftime("%Y-%m-%d %H:%M")
        return f"scheduled #{task_id} for {due_str}" + (
            " (repeating)" if repeat else ""
        )

    def list_scheduled() -> str:
        tasks = scheduler.list_pending()
        if not tasks:
            return "nothing scheduled"
        return "\n".join(task.describe() for task in tasks)

    def cancel_scheduled(task_id: str) -> str:
        try:
            numeric = int(str(task_id).lstrip("#"))
        except ValueError:
            return f"ERROR: '{task_id}' is not a task id"
        return (
            f"cancelled #{numeric}"
            if scheduler.cancel(numeric)
            else f"ERROR: no task #{numeric}"
        )

    reg.register(Tool(
        "schedule_task",
        "Schedule a reminder or recurring task for the user.",
        {"when": "e.g. 'in 20 minutes', 'every day at 08:00'",
         "message": "what to remind about"},
        schedule_task,
    ))
    reg.register(Tool(
        "list_scheduled", "List pending reminders and scheduled routines.",
        {}, list_scheduled,
    ))
    reg.register(Tool(
        "cancel_scheduled", "Cancel a scheduled task by its id.",
        {"task_id": "the task id, e.g. 3"}, cancel_scheduled,
    ))


def _register_builtins(reg: ToolRegistry) -> None:
    cfg = reg.config

    def current_time() -> str:
        now = _dt.datetime.now().astimezone()
        return now.strftime("%A, %Y-%m-%d %H:%M:%S %Z")

    def system_info() -> str:
        vm = shutil.disk_usage(Path.home())
        info = [
            f"os: {platform.system()} {platform.release()} ({platform.machine()})",
            f"python: {platform.python_version()}",
            f"hostname: {platform.node()}",
            f"cpu_count: {__import__('os').cpu_count()}",
            f"home_disk: {vm.free / 1e9:.1f} GB free of {vm.total / 1e9:.1f} GB",
            f"workspace: {cfg.workspace}",
        ]
        return "\n".join(info)

    def read_file(path: str) -> str:
        p = reg.resolve_read_path(path)
        if not p.is_file():
            return f"ERROR: not a file: {p}"
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > 20_000:
            text = text[:20_000] + f"\n... [truncated, {len(text)} chars total]"
        return text

    def write_file(path: str, content: str) -> str:
        p = reg.resolve_write_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {p}"

    def list_dir(path: str = ".") -> str:
        p = reg.resolve_read_path(path)
        if not p.is_dir():
            return f"ERROR: not a directory: {p}"
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        lines = [f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries[:200]]
        return f"{p}:\n" + "\n".join(lines) if lines else f"{p}: (empty)"

    def search_memory(query: str) -> str:
        hits = reg.memory.search(query, top_k=5)
        if not hits:
            return "no relevant memory found"
        return "\n---\n".join(
            f"[{h.source or h.kind}] (score {h.score:.2f})\n{h.content[:1500]}"
            for h in hits
        )

    def remember_fact(fact: str) -> str:
        reg.memory.remember(fact, source="conversation", kind="fact")
        return f"remembered: {fact}"

    def _load_json(name: str, default):
        path = cfg.workspace / ".jarvis" / name
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def _save_json(name: str, data) -> None:
        path = cfg.workspace / ".jarvis" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def create_mission(title: str, objective: str, steps: str) -> str:
        """Create a durable, checkpointed mission from newline-separated steps."""
        missions = _load_json("missions.json", {})
        key = title.strip().lower()
        items = [s.strip().lstrip("-0123456789. ") for s in steps.splitlines() if s.strip()]
        if not items:
            return "ERROR: provide at least one step, separated by newlines"
        missions[key] = {
            "title": title.strip(), "objective": objective.strip(),
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "steps": [{"task": item, "status": "pending", "note": ""} for item in items],
        }
        _save_json("missions.json", missions)
        return f"mission created: {title} ({len(items)} checkpoints)"

    def update_mission(title: str, step: str, status: str, note: str) -> str:
        missions = _load_json("missions.json", {})
        mission = missions.get(title.strip().lower())
        if mission is None:
            return f"ERROR: no mission named '{title}'"
        allowed = {"pending", "active", "blocked", "complete"}
        status = status.strip().lower()
        if status not in allowed:
            return f"ERROR: status must be one of {', '.join(sorted(allowed))}"
        try:
            index = int(step) - 1
            checkpoint = mission["steps"][index]
        except (ValueError, IndexError):
            return f"ERROR: step must be 1–{len(mission['steps'])}"
        checkpoint.update(status=status, note=note.strip(), updated=_dt.datetime.now().isoformat(timespec="seconds"))
        _save_json("missions.json", missions)
        return f"updated {title}, checkpoint {index + 1}: {status}"

    def mission_control() -> str:
        missions = _load_json("missions.json", {})
        if not missions:
            return "no active missions"
        sections = []
        for mission in missions.values():
            done = sum(s["status"] == "complete" for s in mission["steps"])
            lines = [f"MISSION: {mission['title']} — {done}/{len(mission['steps'])} complete",
                     f"Objective: {mission['objective']}"]
            lines.extend(f"{i}. [{s['status'].upper()}] {s['task']}" + (f" — {s['note']}" if s.get('note') else "")
                         for i, s in enumerate(mission["steps"], 1))
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def record_decision(decision: str, reasoning: str, alternatives: str) -> str:
        journal = _load_json("decisions.json", [])
        entry = {"timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                 "decision": decision.strip(), "reasoning": reasoning.strip(),
                 "alternatives": alternatives.strip()}
        journal.append(entry)
        _save_json("decisions.json", journal[-500:])
        reg.memory.remember(
            f"Decision: {entry['decision']}\nReasoning: {entry['reasoning']}\nAlternatives: {entry['alternatives']}",
            source="decision-journal", kind="fact",
        )
        return f"decision recorded at {entry['timestamp']}"

    def privacy_scan(text: str) -> str:
        """Local heuristic scan before content is copied, shared, or sent online."""
        patterns = {
            "email address": r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            "phone number": r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)",
            "US SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit-card-like number": r"\b(?:\d[ -]*?){13,19}\b",
            "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            "secret/token assignment": r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,;]{6,}",
        }
        findings = [(label, len(re.findall(pattern, text, re.IGNORECASE))) for label, pattern in patterns.items()]
        findings = [(label, count) for label, count in findings if count]
        if not findings:
            return "privacy scan: no common sensitive-data patterns detected (heuristic, not a guarantee)"
        return "PRIVACY WARNING\n" + "\n".join(f"- {label}: {count} match(es)" for label, count in findings)

    def workspace_radar(path: str = ".") -> str:
        root = reg.resolve_read_path(path)
        if not root.is_dir():
            return f"ERROR: not a directory: {root}"
        now = _dt.datetime.now().timestamp()
        changed = []
        for item in root.rglob("*"):
            if any(part in {".git", ".venv", "node_modules", "__pycache__"} for part in item.parts):
                continue
            if item.is_file():
                try:
                    age = now - item.stat().st_mtime
                except OSError:
                    continue
                if age <= 86400:
                    changed.append((item.stat().st_mtime, item, age))
        changed.sort(reverse=True)
        if not changed:
            return f"workspace radar: no files changed in the last 24 hours under {root}"
        lines = [f"workspace radar: {len(changed)} files changed in the last 24 hours (showing {min(30, len(changed))})"]
        for _, item, age in changed[:30]:
            relative = item.relative_to(root)
            when = f"{int(age // 60)}m ago" if age < 3600 else f"{age / 3600:.1f}h ago"
            lines.append(f"- {relative} — {when}")
        return "\n".join(lines)

    def situation_room() -> str:
        stats = reg.memory.stats()
        missions = mission_control()
        recent = reg.memory.recent(limit=6)
        dialogue = "\n".join(f"- {role}: {content[:180]}" for role, content in recent) or "(none)"
        return (f"SITUATION ROOM — {_dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}\n"
                f"Memory: {stats['episodic_entries']} episodes, {stats['semantic_chunks']} knowledge chunks, {stats['facts']} facts\n\n"
                f"{missions}\n\nRECENT SIGNALS\n{dialogue}")

    def run_diagnostics() -> str:
        """Run non-destructive health checks across the live JARVIS stack."""
        checks: list[tuple[str, str, str]] = []

        def check(name: str, status: str, detail: str) -> None:
            checks.append((name, status, detail.replace("\n", " ")[:300]))

        # Cognitive engine: local endpoints get a real connectivity probe;
        # cloud engines are configuration-checked without spending tokens.
        provider = getattr(cfg, "provider", "local").lower()
        if provider in ("", "local", "ollama", "auto"):
            try:
                from .llm import KNOWN_ENDPOINTS, _probe
                endpoints = ([cfg.api_base] if cfg.api_base else []) + [base for _, base in KNOWN_ENDPOINTS]
                live = None
                for endpoint in dict.fromkeys(endpoints):
                    try:
                        models = _probe(endpoint, timeout=1.5)
                        if models:
                            live = (endpoint, models)
                            break
                    except Exception:
                        continue
                if live:
                    check("Cognitive engine", "PASS", f"{live[0]} serving {len(live[1])} model(s)")
                else:
                    check("Cognitive engine", "FAIL", "no configured local model endpoint responded")
            except Exception as exc:
                check("Cognitive engine", "FAIL", f"probe error: {type(exc).__name__}: {exc}")
        else:
            key_present = bool(cfg.provider_key(provider))
            check("Cognitive engine", "PASS" if key_present else "FAIL",
                  f"{provider} configured; credential present" if key_present else f"{provider} credential missing")

        try:
            with reg.memory._lock:
                integrity = reg.memory._conn.execute("PRAGMA quick_check").fetchone()[0]
            stats = reg.memory.stats()
            check("Memory database", "PASS" if integrity == "ok" else "FAIL",
                  f"SQLite {integrity}; {stats['episodic_entries']} episodes, {stats['semantic_chunks']} chunks")
        except Exception as exc:
            check("Memory database", "FAIL", f"{type(exc).__name__}: {exc}")

        probe = cfg.workspace / ".jarvis-diagnostic-probe"
        try:
            cfg.workspace.mkdir(parents=True, exist_ok=True)
            probe.write_text("jarvis-health-check", encoding="utf-8")
            valid = probe.read_text(encoding="utf-8") == "jarvis-health-check"
            probe.unlink(missing_ok=True)
            check("Workspace I/O", "PASS" if valid else "FAIL", f"read/write verified at {cfg.workspace}")
        except Exception as exc:
            probe.unlink(missing_ok=True)
            check("Workspace I/O", "FAIL", f"{type(exc).__name__}: {exc}")

        try:
            disk = shutil.disk_usage(cfg.workspace)
            free_gb = disk.free / 1e9
            status = "PASS" if free_gb >= 5 else "WARN" if free_gb >= 1 else "FAIL"
            check("Storage", status, f"{free_gb:.1f} GB free of {disk.total / 1e9:.1f} GB")
        except Exception as exc:
            check("Storage", "FAIL", str(exc))

        safe = sum(tool.tier is Tier.SAFE for tool in reg.tools.values())
        gated = sum(tool.tier is Tier.CONFIRM for tool in reg.tools.values())
        check("Tool broker", "PASS" if safe else "FAIL", f"{safe} safe tools, {gated} permission-gated tools")

        scheduler = reg.get("list_scheduled")
        if scheduler:
            result = reg.run("list_scheduled", {})
            check("Scheduler", "FAIL" if result.startswith("ERROR:") else "PASS", result)
        else:
            check("Scheduler", "WARN", "scheduler is not attached to this session")

        try:
            from . import voice
            browser_voice = "browser microphone and speech available through the web dashboard"
            native_input = voice.asr_available()
            native_output = bool(
                __import__("importlib").util.find_spec("pyttsx3")
                or shutil.which("espeak") or shutil.which("spd-say") or platform.system() in ("Darwin", "Windows")
            )
            status = "PASS" if native_input and native_output else "WARN"
            check("Voice systems", status,
                  f"native input={'ready' if native_input else 'optional extras absent'}, native output={'ready' if native_output else 'unavailable'}; {browser_voice}")
        except Exception as exc:
            check("Voice systems", "WARN", f"voice probe unavailable: {exc}")

        configured_mcp = getattr(cfg, "mcp_servers", {}) or {}
        if not configured_mcp:
            check("MCP extensions", "PASS", "none configured (optional)")
        else:
            missing = []
            for name, spec in configured_mcp.items():
                command = spec.get("command", []) if isinstance(spec, dict) else []
                if not command or not shutil.which(str(command[0])):
                    missing.append(name)
            check("MCP extensions", "WARN" if missing else "PASS",
                  f"{len(configured_mcp) - len(missing)}/{len(configured_mcp)} commands available" + (f"; missing: {', '.join(missing)}" if missing else ""))

        counts = {level: sum(status == level for _, status, _ in checks) for level in ("PASS", "WARN", "FAIL")}
        overall = "OPERATIONAL" if not counts["FAIL"] else "DEGRADED"
        lines = [f"JARVIS DIAGNOSTIC REPORT — {overall}",
                 _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"), ""]
        lines.extend(f"[{status}] {name}: {detail}" for name, status, detail in checks)
        lines.extend(["", f"SUMMARY: {counts['PASS']} passed · {counts['WARN']} warnings · {counts['FAIL']} failed"])
        return "\n".join(lines)

    def shell(command: str) -> str:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=cfg.workspace,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > 12_000:
            out = out[:12_000] + "\n... [truncated]"
        return f"exit code {proc.returncode}\n{out.strip() or '(no output)'}"

    def web_search(query: str) -> str:
        import httpx

        response = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Jarvis local assistant)"},
            timeout=15,
            follow_redirects=True,
        )
        results = parse_ddg_results(response.text)
        if not results:
            return f"no results (HTTP {response.status_code})"
        return "\n\n".join(
            f"{title}\n{url}\n{snippet}" for title, url, snippet in results
        )

    def web_get(url: str) -> str:
        import httpx

        if not url.startswith(("http://", "https://")):
            return "ERROR: url must start with http:// or https://"
        response = httpx.get(url, timeout=20, follow_redirects=True)
        text = response.text
        if len(text) > 15_000:
            text = text[:15_000] + "\n... [truncated]"
        return f"HTTP {response.status_code}\n{text}"

    reg.register(Tool("current_time", "Current local date and time.", {}, current_time))
    reg.register(Tool("system_info", "OS, CPU, disk, and environment summary.", {}, system_info))
    reg.register(Tool(
        "read_file", "Read a text file (within your home directory).",
        {"path": "file path"}, read_file,
    ))
    reg.register(Tool(
        "write_file", "Write a text file inside the Jarvis workspace.",
        {"path": "file path", "content": "text to write"}, write_file,
    ))
    reg.register(Tool(
        "list_dir", "List a directory (within your home directory).",
        {"path": "directory path"}, list_dir,
    ))
    reg.register(Tool(
        "search_memory", "Search ingested documents and remembered facts.",
        {"query": "search query"}, search_memory,
    ))
    reg.register(Tool(
        "remember_fact", "Store a durable fact about the user or their setup.",
        {"fact": "the fact to remember"}, remember_fact,
    ))
    reg.register(Tool(
        "create_mission", "Create a durable mission with checkpointed steps that persists across sessions.",
        {"title": "mission name", "objective": "definition of success", "steps": "one checkpoint per line"}, create_mission,
    ))
    reg.register(Tool(
        "update_mission", "Update a mission checkpoint as pending, active, blocked, or complete.",
        {"title": "mission name", "step": "checkpoint number", "status": "pending, active, blocked, or complete", "note": "progress or blocker note"}, update_mission,
    ))
    reg.register(Tool("mission_control", "Show every persistent mission and checkpoint status.", {}, mission_control))
    reg.register(Tool(
        "record_decision", "Record a decision, rationale, and rejected alternatives in durable searchable memory.",
        {"decision": "decision made", "reasoning": "why", "alternatives": "options not chosen"}, record_decision,
    ))
    reg.register(Tool(
        "privacy_scan", "Locally inspect text for common secrets and personal data before sharing it.",
        {"text": "content to inspect"}, privacy_scan,
    ))
    reg.register(Tool(
        "workspace_radar", "Show files changed in the last 24 hours to recover project context quickly.",
        {"path": "directory path"}, workspace_radar,
    ))
    reg.register(Tool("situation_room", "Synthesize missions, memory health, and recent signals into one local briefing.", {}, situation_room))
    reg.register(Tool(
        "run_diagnostics",
        "Run JARVIS self-diagnostics across the reasoning engine, memory, workspace, storage, tools, scheduler, voice, and MCP extensions. Use whenever the user says run diagnostics or asks if everything is functioning.",
        {}, run_diagnostics,
    ))
    reg.register(Tool(
        "shell", "Run a shell command in the workspace (60 s timeout).",
        {"command": "the command"}, shell, tier=Tier.CONFIRM,
    ))
    reg.register(Tool(
        "web_search", "Search the web (DuckDuckGo) for current information.",
        {"query": "search terms"}, web_search, tier=Tier.CONFIRM,
    ))
    reg.register(Tool(
        "web_get", "Fetch a URL over HTTP(S).",
        {"url": "the URL"}, web_get, tier=Tier.CONFIRM,
    ))
