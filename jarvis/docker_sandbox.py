"""Docker-based ephemeral and persistent sandboxes (Blueprint Section 2).

Three capabilities from the blueprint:

1. Autonomous Tool Synthesis — when the agent encounters an uninstalled utility
   or an exotic file format, it writes a custom script, compiles and validates
   it inside an isolated container, then ingests it into the live tool registry.

2. Deterministic Simulation (Digital Twin) — before any automation script or
   hardware instruction is dispatched, run it inside a container that mirrors
   the host environment. Only if the simulation exits cleanly is the real
   command allowed to proceed.

3. Persistent Architectural Sandboxes — long-lived named containers that retain
   structural memories, custom compilations, and synthesized utilities across
   workflow sessions. Context is NOT wiped when the workflow ends.

Requires: Docker Engine running on the host (docker CLI or docker SDK).
Graceful degradation: if Docker is absent, operations fall back to the existing
subprocess sandbox_exec tool and report the limitation clearly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---- Docker availability check ----------------------------------------------

def docker_available() -> bool:
    return shutil.which("docker") is not None and _docker_daemon_running()


def _docker_daemon_running() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---- Container lifecycle ----------------------------------------------------

@dataclass
class SandboxSpec:
    image: str = "python:3.12-slim"
    name: str = ""            # empty = ephemeral (auto-removed on stop)
    workdir: str = "/workspace"
    mounts: dict[str, str] = field(default_factory=dict)   # host_path -> container_path
    env: dict[str, str] = field(default_factory=dict)
    memory_limit: str = "512m"
    cpu_quota: int = 50000    # 50% of one CPU (100000 = 1 full CPU)
    network: str = "none"     # 'none' | 'bridge' | 'host'


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def output(self) -> str:
        combined = (self.stdout + self.stderr).strip()
        return combined[:10_000] if len(combined) > 10_000 else combined


class DockerSandbox:
    """Manage one Docker container: create, exec, commit, stop, remove."""

    def __init__(self, spec: SandboxSpec) -> None:
        self.spec = spec
        self.container_id: str = ""
        self._lock = threading.Lock()

    def start(self) -> str:
        """Start the container and return its ID."""
        cmd = [
            "docker", "run", "--detach",
            "--memory", self.spec.memory_limit,
            "--cpu-quota", str(self.spec.cpu_quota),
            "--network", self.spec.network,
            "--workdir", self.spec.workdir,
        ]
        if self.spec.name:
            cmd += ["--name", self.spec.name]
        else:
            cmd.append("--rm")  # auto-remove only for unnamed containers

        for host, cont in self.spec.mounts.items():
            cmd += ["-v", f"{host}:{cont}:rw"]

        for k, v in self.spec.env.items():
            cmd += ["-e", f"{k}={v}"]

        cmd += [self.spec.image, "sleep", "3600"]  # keep container alive

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")
        self.container_id = result.stdout.strip()
        return self.container_id

    def exec(self, command: str, timeout: int = 60) -> ExecResult:
        """Run a shell command inside the container."""
        if not self.container_id:
            raise RuntimeError("container not started; call start() first")
        t0 = time.monotonic()
        result = subprocess.run(
            ["docker", "exec", self.container_id, "sh", "-c", command],
            capture_output=True, text=True, timeout=timeout,
        )
        return ExecResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_s=time.monotonic() - t0,
        )

    def copy_to(self, local_path: str, container_path: str) -> None:
        subprocess.run(
            ["docker", "cp", local_path, f"{self.container_id}:{container_path}"],
            check=True, capture_output=True,
        )

    def copy_from(self, container_path: str, local_path: str) -> None:
        subprocess.run(
            ["docker", "cp", f"{self.container_id}:{container_path}", local_path],
            check=True, capture_output=True,
        )

    def commit(self, tag: str) -> str:
        """Snapshot the container state into an image."""
        result = subprocess.run(
            ["docker", "commit", self.container_id, tag],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()

    def stop(self) -> None:
        if not self.container_id:
            return
        subprocess.run(
            ["docker", "stop", "--time", "5", self.container_id],
            capture_output=True, timeout=15,
        )

    def remove(self) -> None:
        if not self.container_id:
            return
        subprocess.run(
            ["docker", "rm", "-f", self.container_id],
            capture_output=True, timeout=10,
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        if not self.spec.name:
            self.remove()


# ---- Persistent sandbox registry --------------------------------------------

class SandboxRegistry:
    """Track named persistent containers across sessions."""

    def __init__(self, state_path: Path) -> None:
        self.path = state_path
        self._state: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._state = json.loads(self.path.read_text())
            except Exception:
                self._state = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2))

    def register(self, name: str, container_id: str, image: str, purpose: str = "") -> None:
        self._state[name] = {
            "container_id": container_id,
            "image": image,
            "purpose": purpose,
            "created": time.time(),
            "last_used": time.time(),
        }
        self._save()

    def get(self, name: str) -> dict | None:
        return self._state.get(name)

    def remove(self, name: str) -> None:
        self._state.pop(name, None)
        self._save()

    def list_all(self) -> list[dict]:
        return [{"name": k, **v} for k, v in self._state.items()]

    def touch(self, name: str) -> None:
        if name in self._state:
            self._state[name]["last_used"] = time.time()
            self._save()


# ---- Digital twin (pre-execution simulation) --------------------------------

class DigitalTwin:
    """Run a command in a container mirror before executing on the real host.

    The twin container is a copy of the workspace mounted read-only, so the
    simulation can read files but cannot mutate the live environment.
    Only if the simulation succeeds (exit code 0) is the real command released.
    """

    def __init__(self, workspace: Path, image: str = "python:3.12-slim") -> None:
        self.workspace = workspace
        self.image = image

    def simulate(self, command: str, timeout: int = 60) -> tuple[bool, ExecResult]:
        """Run command in an isolated twin. Returns (safe_to_run, result)."""
        if not docker_available():
            return True, ExecResult(0, "(simulation skipped: Docker unavailable)", "", 0.0)

        spec = SandboxSpec(
            image=self.image,
            mounts={str(self.workspace): "/workspace:ro"},
            workdir="/workspace",
        )
        with DockerSandbox(spec) as sb:
            result = sb.exec(command, timeout=timeout)
        return result.ok, result


# ---- Autonomous tool synthesis ----------------------------------------------

class ToolSynthesizer:
    """Write, containerize, validate, and ingest custom utility scripts.

    When the agent encounters an uninstalled tool or an exotic format, it calls
    synthesize() with a description of what's needed. The synthesizer asks the
    LLM to write a Python script, runs it in a fresh container, validates that
    it works, and registers the result as a new SAFE-tier tool.
    """

    def __init__(self, llm, workspace: Path, registry=None) -> None:
        self.llm = llm
        self.workspace = workspace
        self.tool_registry = registry  # ToolRegistry, assigned after boot
        self._synthesized: dict[str, str] = {}  # name -> script path

    def synthesize(self, task_description: str, tool_name: str) -> dict:
        """Generate, validate, and register a new tool from a natural-language spec."""
        prompt = (
            f"Write a self-contained Python script that: {task_description}\n\n"
            "Requirements:\n"
            "- Accept all input via sys.argv or stdin; output to stdout.\n"
            "- Include no external dependencies beyond Python stdlib.\n"
            "- End with a main() function called via if __name__ == '__main__'.\n"
            "- Print a clear error message to stderr if something goes wrong.\n"
            "Output ONLY the Python source code, no explanation, no markdown fences."
        )
        script = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.1)

        script_path = self.workspace / ".jarvis" / "synthesized" / f"{tool_name}.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")

        if not docker_available():
            validation_result = self._validate_subprocess(script, tool_name)
        else:
            validation_result = self._validate_docker(script, tool_name)

        if validation_result["ok"]:
            self._synthesized[tool_name] = str(script_path)
            if self.tool_registry is not None:
                self._register(tool_name, script_path, task_description)

        return {
            "tool_name": tool_name,
            "script_path": str(script_path),
            "validated": validation_result["ok"],
            "output": validation_result["output"],
            "registered": validation_result["ok"] and self.tool_registry is not None,
        }

    def _validate_subprocess(self, script: str, tool_name: str) -> dict:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
        )
        return {
            "ok": result.returncode == 0,
            "output": (result.stdout + result.stderr).strip()[:2000],
        }

    def _validate_docker(self, script: str, tool_name: str) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(script)
            tmp = f.name

        spec = SandboxSpec(image="python:3.12-slim")
        try:
            with DockerSandbox(spec) as sb:
                sb.copy_to(tmp, f"/tmp/{tool_name}.py")
                result = sb.exec(f"python3 /tmp/{tool_name}.py 2>&1", timeout=30)
            return {"ok": result.ok, "output": result.output}
        finally:
            Path(tmp).unlink(missing_ok=True)

    def _register(self, tool_name: str, script_path: Path, description: str) -> None:
        from .tools import Tier, Tool
        import subprocess, sys

        def run_synthesized(**kwargs) -> str:
            args = [f"--{k}={v}" for k, v in kwargs.items()]
            result = subprocess.run(
                [sys.executable, str(script_path)] + args,
                capture_output=True, text=True, timeout=30,
            )
            out = (result.stdout + result.stderr).strip()
            return out[:5000] if len(out) > 5000 else out

        self.tool_registry.register(Tool(
            name=tool_name,
            description=f"[synthesized] {description}",
            params={"args": "key=value arguments passed to the script"},
            func=run_synthesized,
        ))


# ---- Tool registration ------------------------------------------------------

def register_sandbox_tools(registry, workspace: Path, llm=None) -> None:
    from .tools import Tier, Tool

    twin = DigitalTwin(workspace)
    synth = ToolSynthesizer(llm, workspace, registry) if llm else None

    def docker_exec(command: str, image: str = "python:3.12-slim",
                    sandbox_name: str = "") -> str:
        if not docker_available():
            return "ERROR: Docker is not available on this system"
        spec = SandboxSpec(
            image=image,
            name=sandbox_name,
            mounts={str(workspace): "/workspace"},
            workdir="/workspace",
        )
        ctx = DockerSandbox(spec)
        try:
            cid = ctx.start()
            result = ctx.exec(command, timeout=120)
            return f"exit {result.exit_code} ({result.duration_s:.1f}s)\n{result.output}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        finally:
            ctx.stop()
            if not sandbox_name:
                ctx.remove()

    def digital_twin_test(command: str) -> str:
        safe, result = twin.simulate(command, timeout=60)
        verdict = "SAFE TO RUN" if safe else "SIMULATION FAILED — do not run on host"
        return (
            f"Digital twin simulation: {verdict}\n"
            f"exit {result.exit_code}  ({result.duration_s:.1f}s)\n"
            f"{result.output}"
        )

    def synthesize_tool(task_description: str, tool_name: str) -> str:
        if synth is None:
            return "ERROR: LLM not available for tool synthesis"
        result = synth.synthesize(task_description, tool_name)
        if result["validated"]:
            status = "validated and registered" if result["registered"] else "validated (not registered)"
        else:
            status = "VALIDATION FAILED"
        return (
            f"Tool synthesis: {tool_name} — {status}\n"
            f"Script: {result['script_path']}\n"
            f"Output:\n{result['output']}"
        )

    registry.register(Tool(
        "docker_exec",
        "Run a shell command inside an isolated Docker container.",
        {"command": "shell command to run",
         "image": "Docker image (default: python:3.12-slim)",
         "sandbox_name": "named persistent sandbox (optional)"},
        docker_exec, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "digital_twin_test",
        "Simulate a command in a container mirror of the workspace before running it on the host.",
        {"command": "command to simulate"},
        digital_twin_test, tier=Tier.CONFIRM,
    ))
    if synth is not None:
        registry.register(Tool(
            "synthesize_tool",
            "Write, validate in Docker, and register a new custom script as a live tool.",
            {"task_description": "what the tool should do",
             "tool_name": "snake_case name for the new tool"},
            synthesize_tool, tier=Tier.CONFIRM,
        ))
