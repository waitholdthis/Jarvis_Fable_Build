"""Proactive monitoring, Complex Event Processing, and autonomic self-healing.

Blueprint Section 4: deploy state-triggered event engines at the OS level.
Rather than polling on user request, the monitor runs as a background daemon
that watches file systems and git repositories, identifies multi-event anomaly
patterns via a rolling event window (CEP), and triggers an OODA loop when
something goes wrong.

Self-healing workflow (SelfHealer):
  1. Detect a failing command (pytest, go build, npm test, cargo build, etc.)
  2. Capture the full error output.
  3. Identify the 3 most recently modified source files as context.
  4. Ask the LLM for a minimal unified diff that fixes the failure.
  5. Apply the patch in an isolated temp copy of the workspace.
  6. Re-run the command to verify the fix compiles and tests pass.
  7. If it passes, return the verified diff; caller decides whether to commit.

Filesystem watching uses Python's watchdog library when available (inotify /
FSEvents) and falls back to mtime polling every 2 s so the monitor works on
any system without mandatory extras.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ---- Event types ------------------------------------------------------------

@dataclass
class FsEvent:
    kind: str     # 'modified' | 'created' | 'deleted' | 'moved'
    path: str
    ts: float = field(default_factory=time.time)


@dataclass
class GitEvent:
    kind: str     # 'commit' | 'branch_change' | 'uncommitted_changes'
    repo: str
    detail: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class AnomalyEvent:
    pattern: str        # human-readable pattern name
    events: list        # the triggering events
    severity: str = "warn"   # 'info' | 'warn' | 'critical'
    ts: float = field(default_factory=time.time)


# ---- Event bus --------------------------------------------------------------

class EventBus:
    """Thread-safe in-process pub/sub.  Queues are bounded to avoid leaks."""

    def __init__(self, maxsize: int = 500) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def publish(self, event) -> None:
        with self._lock:
            for q in self._subs:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass  # drop oldest implicitly by not blocking


# ---- Source-file extensions monitored for churn detection -------------------

_CODE_EXTS = {".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".java",
              ".cs", ".rb", ".php", ".swift", ".kt", ".scala"}

_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache",
              "dist", "build", "target", ".tox"}


# ---- Filesystem watcher (watchdog or polling) --------------------------------

class _WatchdogHandler:
    """Bridges watchdog events into our EventBus."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus

    def dispatch(self, event) -> None:
        if event.is_directory:
            return
        kind_map = {
            "modified": "modified",
            "created": "created",
            "deleted": "deleted",
            "moved": "moved",
        }
        for suffix, kind in kind_map.items():
            if suffix in type(event).__name__.lower():
                self.bus.publish(FsEvent(kind, str(event.src_path)))
                break


class EventMonitor:
    """Watch filesystem paths, detect anomaly patterns, fire OODA callbacks.

    Usage:
        monitor = EventMonitor()
        monitor.watch(Path("/home/user/project"))
        monitor.on_anomaly = lambda a: print("⚠", a.pattern, a.events)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(self, window_s: float = 60.0) -> None:
        self.bus = EventBus()
        self._watch_paths: list[Path] = []
        self._running = False
        self._threads: list[threading.Thread] = []
        self._window_s = window_s
        self._recent: list[FsEvent] = []
        self.on_anomaly: Callable[[AnomalyEvent], None] | None = None
        self._subscriber = self.bus.subscribe()
        self._git_repos: list[Path] = []

    def watch(self, path: Path | str) -> None:
        p = Path(path)
        self._watch_paths.append(p)
        if (p / ".git").is_dir():
            self._git_repos.append(p)

    def start(self) -> None:
        self._running = True
        self._start_fs_watcher()
        for repo in self._git_repos:
            t = threading.Thread(target=self._git_poll_loop, args=(repo,), daemon=True)
            t.start()
            self._threads.append(t)
        cep = threading.Thread(target=self._cep_loop, daemon=True)
        cep.start()
        self._threads.append(cep)

    def stop(self) -> None:
        self._running = False

    def _start_fs_watcher(self) -> None:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class _Handler(FileSystemEventHandler):
                def __init__(self_, bus):
                    self_._bridge = _WatchdogHandler(bus)

                def on_any_event(self_, event):
                    self_._bridge.dispatch(event)

            observer = Observer()
            handler = _Handler(self.bus)
            for p in self._watch_paths:
                observer.schedule(handler, str(p), recursive=True)
            observer.start()
            self._threads.append(observer)
        except ImportError:
            t = threading.Thread(target=self._polling_loop, daemon=True)
            t.start()
            self._threads.append(t)

    def _polling_loop(self) -> None:
        snapshots: dict[str, float] = {}
        for p in self._watch_paths:
            for f in Path(p).rglob("*"):
                if f.is_file() and not self._skip(f):
                    try:
                        snapshots[str(f)] = f.stat().st_mtime
                    except OSError:
                        pass

        while self._running:
            time.sleep(2)
            for watch_root in self._watch_paths:
                for f in Path(watch_root).rglob("*"):
                    if not f.is_file() or self._skip(f):
                        continue
                    key = str(f)
                    try:
                        mtime = f.stat().st_mtime
                    except OSError:
                        if key in snapshots:
                            del snapshots[key]
                            self.bus.publish(FsEvent("deleted", key))
                        continue
                    if key not in snapshots:
                        snapshots[key] = mtime
                        self.bus.publish(FsEvent("created", key))
                    elif mtime != snapshots[key]:
                        snapshots[key] = mtime
                        self.bus.publish(FsEvent("modified", key))

    def _skip(self, path: Path) -> bool:
        return any(part in _SKIP_DIRS for part in path.parts)

    def _git_poll_loop(self, repo: Path) -> None:
        """Detect uncommitted changes every 10 s."""
        last_hash = ""
        while self._running:
            time.sleep(10)
            try:
                result = subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                )
                current_hash = result.stdout.strip()
                if current_hash and current_hash != last_hash:
                    if last_hash:
                        self.bus.publish(GitEvent("commit", str(repo), current_hash))
                    last_hash = current_hash

                dirty = subprocess.run(
                    ["git", "-C", str(repo), "status", "--porcelain"],
                    capture_output=True, text=True, timeout=5,
                )
                if dirty.stdout.strip():
                    self.bus.publish(GitEvent("uncommitted_changes", str(repo),
                                              dirty.stdout[:500]))
            except Exception:
                pass

    # ---- CEP (Complex Event Processing) ------------------------------------

    def _cep_loop(self) -> None:
        while self._running:
            try:
                event = self._subscriber.get(timeout=1)
            except queue.Empty:
                continue

            now = time.time()
            if isinstance(event, FsEvent):
                self._recent.append(event)
                self._recent = [e for e in self._recent if now - e.ts < self._window_s]
                self._check_patterns()

    def _check_patterns(self) -> None:
        if self.on_anomaly is None:
            return

        code_events = [
            e for e in self._recent
            if isinstance(e, FsEvent) and Path(e.path).suffix in _CODE_EXTS
        ]

        # Pattern: ≥5 source-file modifications in the rolling window → churn
        if len(code_events) >= 5:
            self.on_anomaly(AnomalyEvent(
                pattern="rapid-source-churn",
                events=code_events[-5:],
                severity="warn",
            ))
            self._recent.clear()

        # Pattern: same file modified ≥3 times → compile-loop thrash
        paths = [e.path for e in code_events if e.kind == "modified"]
        for path in set(paths):
            if paths.count(path) >= 3:
                self.on_anomaly(AnomalyEvent(
                    pattern="compile-loop-thrash",
                    events=[e for e in code_events if e.path == path],
                    severity="critical",
                ))
                self._recent = [e for e in self._recent if e.path != path]
                break


# ---- Self-healing -----------------------------------------------------------

@dataclass
class HealResult:
    success: bool
    original_error: str
    patch: str
    fixed: bool
    verification_output: str
    command: str


class SelfHealer:
    """Detect command failures and attempt LLM-generated programmatic fixes.

    The repair runs in an isolated tempdir copy of the workspace so the live
    environment is never mutated until the fix is verified.
    """

    def __init__(self, llm, workspace: Path) -> None:
        self.llm = llm
        self.workspace = Path(workspace)

    def attempt_repair(
        self,
        command: str,
        cwd: Path | None = None,
        timeout: int = 120,
    ) -> HealResult:
        """Run command; if it fails, generate a patch, verify it, return result."""
        cwd = cwd or self.workspace

        run = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            cwd=cwd, timeout=timeout,
        )
        original_error = (run.stdout + run.stderr).strip()

        if run.returncode == 0:
            return HealResult(True, "", "", True, original_error, command)

        # Gather recently modified source files for context (max 3)
        sources = sorted(
            [p for p in cwd.rglob("*")
             if p.is_file() and p.suffix in _CODE_EXTS and not self._skip(p)],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:3]
        file_context = "\n\n".join(
            f"--- {p.relative_to(cwd)} ---\n"
            + p.read_text(encoding="utf-8", errors="replace")[:3000]
            for p in sources
        )

        prompt = (
            f"The following command failed:\n\n  {command}\n\n"
            f"ERROR OUTPUT:\n{original_error[:2000]}\n\n"
            f"RELEVANT SOURCE FILES:\n{file_context}\n\n"
            "Produce a minimal unified diff (--- a/path +++ b/path) that fixes "
            "the error. Output ONLY the diff block, nothing else. "
            "Do not include explanations or markdown fences."
        )
        patch = self.llm.chat(
            [{"role": "user", "content": prompt}], temperature=0.1
        )

        # Apply and verify in a sandboxed copy
        with tempfile.TemporaryDirectory() as tmp_root:
            tmp = Path(tmp_root) / "sandbox"
            shutil.copytree(cwd, tmp, symlinks=True, ignore=shutil.ignore_patterns(
                ".venv", "node_modules", "__pycache__", "*.pyc",
            ))

            patch_result = subprocess.run(
                ["patch", "-p1", "--batch", "--silent"],
                input=patch, capture_output=True, text=True, cwd=tmp,
            )
            if patch_result.returncode != 0:
                return HealResult(
                    success=False,
                    original_error=original_error,
                    patch=patch,
                    fixed=False,
                    verification_output=f"patch failed: {patch_result.stderr.strip()}",
                    command=command,
                )

            verify = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                cwd=tmp, timeout=timeout,
            )
            verification_output = (verify.stdout + verify.stderr).strip()
            fixed = verify.returncode == 0

        return HealResult(
            success=fixed,
            original_error=original_error,
            patch=patch,
            fixed=fixed,
            verification_output=verification_output,
            command=command,
        )

    def _skip(self, path: Path) -> bool:
        return any(part in _SKIP_DIRS for part in path.parts)


# ---- Tool registration ------------------------------------------------------

def register_monitor_tools(registry, healer: SelfHealer) -> None:
    """Add self-heal tool to the tool registry."""
    from .tools import Tier, Tool

    def self_heal(command: str, cwd: str = "") -> str:
        path = Path(cwd).expanduser() if cwd else None
        try:
            result = healer.attempt_repair(command, cwd=path)
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out during self-heal attempt"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

        if result.success and not result.original_error:
            return f"Command passed on first run:\n{result.verification_output}"

        status = "FIXED ✓" if result.fixed else "UNFIXED ✗"
        lines = [
            f"self_heal({command!r}) — {status}",
            f"\nORIGINAL ERROR:\n{result.original_error[:1000]}",
        ]
        if result.patch:
            lines.append(f"\nGENERATED PATCH:\n{result.patch[:2000]}")
        lines.append(f"\nVERIFICATION:\n{result.verification_output[:1000]}")
        if result.fixed:
            lines.append("\nPatch verified in sandbox. Review the diff above and apply manually if approved.")
        return "\n".join(lines)

    registry.register(Tool(
        "self_heal",
        "Run a command; if it fails, generate and sandbox-verify a programmatic fix, then return the verified patch.",
        {"command": "the build/test command to run and fix",
         "cwd": "working directory (defaults to workspace)"},
        self_heal,
        tier=Tier.CONFIRM,
    ))
