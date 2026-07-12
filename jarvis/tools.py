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
import platform
import shutil
import subprocess
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
        "shell", "Run a shell command in the workspace (60 s timeout).",
        {"command": "the command"}, shell, tier=Tier.CONFIRM,
    ))
    reg.register(Tool(
        "web_get", "Fetch a URL over HTTP(S).",
        {"url": "the URL"}, web_get, tier=Tier.CONFIRM,
    ))
