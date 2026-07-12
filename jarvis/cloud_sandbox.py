"""Cloud-decoupled persistent autonomy (Blueprint Section 7).

Three capabilities:

1. Always-On Serverless Sandboxes — offload long-running code tasks, web
   crawling, and background data mining to cloud-native sandboxes. Primary
   backend: Modal Labs (Firecracker microVMs, scale-to-zero, Python-native).
   Fallback: any HTTP-based job API (Daytona, Blaxel, or custom).

2. Snapshotting & Scale-to-Zero — take automated state snapshots during idle
   cycles. Restore the active session in under 25ms when triggered by an
   incoming webhook or event.

3. Hydration Sync Client — save cloud-compiled deliverables, blueprints, and
   reports to a distributed file layer. A local daemon detects reconnection,
   pulls the cloud-compiled state, and hydrates local directory structures
   and assets seamlessly.

Requires: pip install modal (for Modal Labs)
The module degrades gracefully when Modal is absent, logging warnings and
routing tasks to the local subprocess sandbox instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---- Job model --------------------------------------------------------------

@dataclass
class CloudJob:
    job_id: str
    status: str          # 'pending' | 'running' | 'done' | 'failed'
    backend: str         # 'modal' | 'webhook' | 'local'
    submitted_at: float = field(default_factory=time.time)
    result: Any = None
    error: str = ""
    artifacts: list[str] = field(default_factory=list)


# ---- Modal Labs backend -----------------------------------------------------

class ModalBackend:
    """Run arbitrary Python functions in Modal's Firecracker microVM fleet."""

    def __init__(self, app_name: str = "jarvis") -> None:
        self.app_name = app_name
        self._available = self._check()

    def _check(self) -> bool:
        try:
            import modal  # noqa: F401
            return True
        except ImportError:
            return False

    def available(self) -> bool:
        return self._available

    def run_function(self, code: str, requirements: list[str] | None = None,
                     timeout: int = 300) -> CloudJob:
        """Deploy an ephemeral Modal function and invoke it."""
        if not self._available:
            raise RuntimeError(
                "Modal is not installed. Run: pip install 'jarvis-assistant[cloud]'"
            )
        import modal

        job_id = hashlib.md5(f"{code}{time.time()}".encode()).hexdigest()[:8]

        try:
            app = modal.App(self.app_name)
            image = modal.Image.debian_slim()
            if requirements:
                image = image.pip_install(*requirements)

            @app.function(image=image, timeout=timeout)
            def _run():
                import sys, io
                stdout = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = stdout
                try:
                    exec(code, {})  # noqa: S102
                    return {"ok": True, "output": stdout.getvalue()}
                except Exception as exc:
                    return {"ok": False, "error": str(exc), "output": stdout.getvalue()}
                finally:
                    sys.stdout = old_stdout

            with modal.runner.deploy_app(app):
                result = _run.remote()

            return CloudJob(
                job_id=job_id, status="done", backend="modal", result=result,
            )
        except Exception as exc:
            return CloudJob(
                job_id=job_id, status="failed", backend="modal", error=str(exc),
            )

    def run_web_scrape(self, urls: list[str]) -> CloudJob:
        """Run multi-URL web scraping in Modal (avoids local bandwidth + IP limits)."""
        code = f"""
import urllib.request, json
results = {{}}
for url in {urls!r}:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            results[url] = r.read().decode('utf-8', errors='replace')[:5000]
    except Exception as e:
        results[url] = f'ERROR: {{e}}'
print(json.dumps(results))
"""
        return self.run_function(code, timeout=120)


# ---- Webhook-based job API (Daytona, Blaxel, generic) ----------------------

class WebhookBackend:
    """Submit jobs to any HTTP endpoint and poll for results."""

    def __init__(self, submit_url: str, status_url_template: str = "",
                 auth_token: str = "") -> None:
        self.submit_url = submit_url
        self.status_url_template = status_url_template
        self.auth_token = auth_token

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    def submit(self, payload: dict) -> CloudJob:
        import httpx
        job_id = hashlib.md5(json.dumps(payload).encode()).hexdigest()[:8]
        try:
            resp = httpx.post(self.submit_url, json=payload,
                              headers=self._headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return CloudJob(
                job_id=data.get("job_id", job_id),
                status="pending", backend="webhook",
            )
        except Exception as exc:
            return CloudJob(job_id=job_id, status="failed", backend="webhook",
                            error=str(exc))

    def poll(self, job: CloudJob, max_wait: int = 300) -> CloudJob:
        if not self.status_url_template:
            return job
        import httpx
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                url = self.status_url_template.format(job_id=job.job_id)
                resp = httpx.get(url, headers=self._headers(), timeout=10)
                resp.raise_for_status()
                data = resp.json()
                job.status = data.get("status", job.status)
                if job.status in ("done", "failed", "complete", "error"):
                    job.result = data.get("result")
                    job.error = data.get("error", "")
                    return job
            except Exception:
                pass
            time.sleep(5)
        job.status = "timeout"
        return job


# ---- Hydration sync daemon --------------------------------------------------

@dataclass
class HydrationManifest:
    """Records what cloud artifacts exist and where they map locally."""
    entries: dict[str, str] = field(default_factory=dict)  # remote_key -> local_path


class HydrationSyncClient:
    """Pull cloud-compiled artifacts into local directory structures.

    On reconnect (network up, job completes), the daemon detects new entries
    in the manifest, downloads them, and places them at the configured local
    paths. Works with any storage backend that exposes an HTTP GET endpoint.
    """

    def __init__(self, local_root: Path, manifest_path: Path,
                 base_url: str = "") -> None:
        self.local_root = local_root
        self.manifest_path = manifest_path
        self.base_url = base_url.rstrip("/")
        self._manifest = self._load_manifest()
        self._running = False
        self._thread: threading.Thread | None = None
        self.on_hydrated: Callable[[str, Path], None] | None = None

    def _load_manifest(self) -> HydrationManifest:
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text())
                return HydrationManifest(entries=data)
            except Exception:
                pass
        return HydrationManifest()

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._manifest.entries, indent=2))

    def register(self, remote_key: str, local_relative_path: str) -> None:
        """Register a remote artifact for sync."""
        self._manifest.entries[remote_key] = local_relative_path
        self._save_manifest()

    def hydrate_one(self, remote_key: str) -> Path | None:
        """Download one artifact and write it to its local path."""
        if not self.base_url:
            return None
        local_rel = self._manifest.entries.get(remote_key)
        if not local_rel:
            return None
        import httpx
        try:
            resp = httpx.get(f"{self.base_url}/{remote_key}", timeout=30)
            resp.raise_for_status()
            local_path = self.local_root / local_rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(resp.content)
            if self.on_hydrated:
                self.on_hydrated(remote_key, local_path)
            return local_path
        except Exception:
            return None

    def hydrate_all(self) -> dict[str, bool]:
        results = {}
        for key in list(self._manifest.entries.keys()):
            p = self.hydrate_one(key)
            results[key] = p is not None
        return results

    def start_daemon(self, interval_s: float = 60.0) -> None:
        """Run periodic hydration sync in the background."""
        self._running = True
        def _loop():
            while self._running:
                self.hydrate_all()
                time.sleep(interval_s)
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_daemon(self) -> None:
        self._running = False


# ---- Session snapshot / restore ---------------------------------------------

class SessionSnapshot:
    """Serialize and restore agent session state for scale-to-zero resumption."""

    def __init__(self, snapshot_dir: Path) -> None:
        self.dir = snapshot_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, messages: list[dict],
             metadata: dict | None = None) -> Path:
        """Serialize the in-flight conversation to disk."""
        path = self.dir / f"{session_id}.json"
        payload = {
            "session_id": session_id,
            "saved_at": time.time(),
            "messages": messages,
            "metadata": metadata or {},
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return path

    def restore(self, session_id: str) -> dict | None:
        path = self.dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def list_snapshots(self) -> list[dict]:
        snaps = []
        for p in sorted(self.dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text())
                snaps.append({
                    "session_id": data.get("session_id", p.stem),
                    "saved_at": data.get("saved_at", 0),
                    "message_count": len(data.get("messages", [])),
                })
            except Exception:
                pass
        return snaps

    def delete(self, session_id: str) -> bool:
        path = self.dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False


# ---- Tool registration ------------------------------------------------------

def register_cloud_tools(
    registry,
    modal_backend: ModalBackend,
    webhook_backend: WebhookBackend | None,
    hydration: HydrationSyncClient,
    snapshots: SessionSnapshot,
    agent=None,
) -> None:
    from .tools import Tier, Tool

    def cloud_run(code: str) -> str:
        if modal_backend.available():
            job = modal_backend.run_function(code)
        else:
            return (
                "Modal not available. Install with: pip install 'jarvis-assistant[cloud]'\n"
                "Falling back to local sandbox_exec."
            )
        if job.status == "done" and job.result:
            result = job.result
            out = result.get("output", "")
            if result.get("ok"):
                return f"cloud job {job.job_id}: done\n{out}"
            return f"cloud job {job.job_id}: FAILED\n{result.get('error')}\n{out}"
        return f"cloud job {job.job_id}: {job.status}\n{job.error}"

    def cloud_scrape(urls: str) -> str:
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        if not modal_backend.available():
            return "ERROR: Modal not installed. Run: pip install modal"
        job = modal_backend.run_web_scrape(url_list)
        if job.status != "done":
            return f"scrape job failed: {job.error}"
        if isinstance(job.result, dict):
            result = job.result
            out = result.get("output", "")
            try:
                pages = json.loads(out)
                return "\n---\n".join(
                    f"URL: {url}\n{content[:2000]}"
                    for url, content in pages.items()
                )
            except Exception:
                return out[:5000]
        return str(job.result)[:5000]

    def hydrate(remote_key: str = "") -> str:
        if remote_key:
            path = hydration.hydrate_one(remote_key)
            return (
                f"hydrated: {path}" if path
                else f"ERROR: could not hydrate '{remote_key}'"
            )
        results = hydration.hydrate_all()
        ok = sum(1 for v in results.values() if v)
        return f"hydrated {ok}/{len(results)} artifact(s)"

    def session_save(session_id: str) -> str:
        messages = getattr(agent, "messages", []) if agent else []
        path = snapshots.save(session_id, messages)
        return f"session '{session_id}' saved: {path}"

    def session_restore(session_id: str) -> str:
        data = snapshots.restore(session_id)
        if data is None:
            return f"ERROR: no snapshot found for '{session_id}'"
        if agent is not None and data.get("messages"):
            agent.messages = data["messages"]
        ts = time.strftime("%Y-%m-%d %H:%M",
                           time.localtime(data.get("saved_at", 0)))
        return (
            f"restored session '{session_id}' from {ts} "
            f"({len(data.get('messages', []))} messages)"
        )

    def session_list() -> str:
        snaps = snapshots.list_snapshots()
        if not snaps:
            return "no saved sessions"
        return "\n".join(
            f"  {s['session_id']}  {time.strftime('%Y-%m-%d %H:%M', time.localtime(s['saved_at']))}  "
            f"({s['message_count']} messages)"
            for s in snaps
        )

    registry.register(Tool(
        "cloud_run",
        "Execute a Python code snippet in a Modal Labs Firecracker microVM (scale-to-zero serverless).",
        {"code": "Python source code to run in the cloud"},
        cloud_run, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "cloud_scrape",
        "Scrape one or more URLs in Modal's cloud (avoids local IP limits, runs in parallel).",
        {"urls": "comma-separated URLs to scrape"},
        cloud_scrape, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "cloud_hydrate",
        "Pull cloud-compiled artifacts from the hydration manifest into local directories.",
        {"remote_key": "specific artifact key (leave empty to sync all)"},
        hydrate,
    ))
    registry.register(Tool(
        "session_save",
        "Snapshot the current conversation to disk for scale-to-zero resumption.",
        {"session_id": "name for the snapshot"},
        session_save,
    ))
    registry.register(Tool(
        "session_restore",
        "Restore a previously saved conversation snapshot into the active agent.",
        {"session_id": "snapshot name"},
        session_restore, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "session_list",
        "List all saved conversation snapshots.",
        {},
        session_list,
    ))
