"""Composable workflow engine: define, store, record, and execute multi-step plans.

Workflows are JARVIS's way of turning a successful ad-hoc task into a repeatable,
shareable procedure. A workflow is a JSON document (stored in ~/.jarvis/workflows/)
describing a sequence of steps JARVIS can execute autonomously.

Step kinds
----------
  tool          call any registered tool, with {var} interpolation in args
  llm           call the LLM with a prompt, capture the response
  parallel      run a list of sub-steps concurrently, wait for all to finish
  condition     if/else branching on a simple boolean test
  loop          repeat sub-steps up to max_iterations until a stop condition
  subworkflow   call another named workflow as an inline sub-procedure
  agent_talk    send a message to a named spawned agent (requires swarm)
  memory_store  write a value directly into JARVIS's semantic memory
  notify        emit a titled notification via the deliver callback
  sleep         pause for N seconds (rate-limiting, backoff)

Variable interpolation
----------------------
Every string value in step args/prompts supports {var_name} substitution.
Variables are populated from workflow-level 'vars', step-level 'output_var'
captures, and any inputs dict passed to WorkflowRunner.run().

Error handling
--------------
Each step supports on_error: 'fail' (default), 'continue', 'retry:N'.

Workflow recording
------------------
WorkflowRecorder wraps any agent session: each tool call is captured as a
workflow step. Calling recorder.stop(name) crystallises the session into a
reusable workflow definition that can be replayed later.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Workflow run state
# ---------------------------------------------------------------------------

@dataclass
class WorkflowRun:
    run_id: str
    workflow_name: str
    status: str                     # 'running' | 'done' | 'failed' | 'paused'
    started_at: float
    completed_at: float | None = None
    step_results: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    logs: list[str] = field(default_factory=list)

    def summary(self) -> str:
        dur = ""
        if self.completed_at:
            dur = f"  duration: {self.completed_at - self.started_at:.1f}s\n"
        return (
            f"workflow run  {self.run_id}  [{self.workflow_name}]\n"
            f"  status: {self.status}\n"
            + dur
            + (f"  error: {self.error}\n" if self.error else "")
            + f"  steps captured: {len(self.step_results)}\n"
            + (f"  logs ({len(self.logs)}): " + "; ".join(self.logs[-5:]) if self.logs else "")
        )


# ---------------------------------------------------------------------------
# Workflow store (JSON files on disk)
# ---------------------------------------------------------------------------

class WorkflowStore:
    """Persist workflow definitions as pretty-printed JSON files."""

    def __init__(self, workflows_dir: Path) -> None:
        self.dir = workflows_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, definition: dict) -> Path:
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(definition, indent=2, ensure_ascii=False))
        return path

    def load(self, name: str) -> dict | None:
        path = self.dir / f"{name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_all(self) -> list[dict]:
        results = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                wf = json.loads(path.read_text(encoding="utf-8"))
                results.append({
                    "name": path.stem,
                    "description": wf.get("description", ""),
                    "steps": len(wf.get("steps", [])),
                })
            except Exception:
                pass
        return results

    def delete(self, name: str) -> bool:
        path = self.dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False


# ---------------------------------------------------------------------------
# Variable interpolation and condition evaluation (no eval())
# ---------------------------------------------------------------------------

def _interpolate(template: Any, ctx: dict[str, Any]) -> str:
    if not isinstance(template, str):
        return str(template)
    for key, value in ctx.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def _interpolate_args(args: dict, ctx: dict) -> dict:
    return {k: _interpolate(v, ctx) for k, v in args.items()}


def _eval_condition(test: str, ctx: dict) -> bool:
    """Evaluate a simple boolean expression without eval()."""
    def resolve(token: str) -> str:
        token = token.strip().strip('"').strip("'")
        if token.startswith("{") and token.endswith("}"):
            return str(ctx.get(token[1:-1], ""))
        return token

    test = _interpolate(test.strip(), ctx)

    # X is [not] empty
    m = re.match(r"^(\S+)\s+is\s+(not\s+)?empty$", test)
    if m:
        val = resolve(m.group(1))
        negated = bool(m.group(2))
        return bool(val) if negated else not bool(val)

    # X contains Y
    m = re.match(r"^(\S+)\s+contains\s+(.+)$", test)
    if m:
        return resolve(m.group(2)) in resolve(m.group(1))

    # X == Y  /  X != Y
    m = re.match(r"^(\S+)\s*(==|!=)\s*(.+)$", test)
    if m:
        lhs, op, rhs = resolve(m.group(1)), m.group(2), resolve(m.group(3))
        return (lhs == rhs) if op == "==" else (lhs != rhs)

    # X > N  /  X < N
    m = re.match(r"^(\S+)\s*([><])\s*(\d+\.?\d*)$", test)
    if m:
        try:
            lhs = float(resolve(m.group(1)))
            rhs = float(m.group(3))
            return lhs > rhs if m.group(2) == ">" else lhs < rhs
        except ValueError:
            return False

    # len(X) > N
    m = re.match(r"^len\((\S+)\)\s*>\s*(\d+)$", test)
    if m:
        return len(resolve(m.group(1))) > int(m.group(2))

    return test.lower() not in ("false", "0", "no", "none", "")


# ---------------------------------------------------------------------------
# Workflow runner
# ---------------------------------------------------------------------------

class WorkflowRunner:
    """Execute workflow definitions step-by-step with variable capture."""

    def __init__(
        self,
        store: WorkflowStore,
        tools,
        llm,
        memory=None,
        swarm=None,
        deliver: Callable[[str, str], None] | None = None,
    ) -> None:
        self.store = store
        self.tools = tools
        self.llm = llm
        self.memory = memory
        self.swarm = swarm
        self.deliver = deliver
        self._runs: dict[str, WorkflowRun] = {}
        self._lock = threading.Lock()
        self.on_step: Callable[[str, str, str], None] | None = None

    # ---- public interface ---------------------------------------------------

    def run(self, name: str, inputs: dict | None = None) -> WorkflowRun:
        definition = self.store.load(name)
        if definition is None:
            return WorkflowRun(
                run_id=str(uuid.uuid4())[:8], workflow_name=name,
                status="failed", started_at=time.time(),
                error=f"workflow '{name}' not found",
            )

        run = WorkflowRun(
            run_id=str(uuid.uuid4())[:8],
            workflow_name=name,
            status="running",
            started_at=time.time(),
        )
        with self._lock:
            self._runs[run.run_id] = run

        ctx: dict[str, Any] = {}
        ctx.update(definition.get("vars", {}))
        ctx.update(inputs or {})

        try:
            self._execute_steps(run, definition.get("steps", []), ctx)
            run.status = "done"
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
        finally:
            run.completed_at = time.time()

        return run

    def run_async(self, name: str, inputs: dict | None = None) -> str:
        """Start a workflow in a background thread, return the run_id."""
        run_id = str(uuid.uuid4())[:8]
        placeholder = WorkflowRun(
            run_id=run_id, workflow_name=name,
            status="running", started_at=time.time(),
        )
        with self._lock:
            self._runs[run_id] = placeholder

        def _go() -> None:
            real = self.run(name, inputs)
            with self._lock:
                self._runs[run_id] = real

        threading.Thread(target=_go, daemon=True).start()
        return run_id

    def get_run(self, run_id: str) -> WorkflowRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self, n: int = 20) -> list[WorkflowRun]:
        with self._lock:
            return sorted(
                self._runs.values(), key=lambda r: r.started_at, reverse=True
            )[:n]

    # ---- step execution ----------------------------------------------------

    def _execute_steps(self, run: WorkflowRun, steps: list, ctx: dict) -> None:
        for step in steps:
            step_id = step.get("id", str(uuid.uuid4())[:6])
            kind = step.get("kind", "tool")
            on_error = step.get("on_error", "fail")
            run.logs.append(f"→ {step_id}")

            try:
                result = self._dispatch(run, step, ctx, kind)
            except Exception as exc:
                if on_error == "continue":
                    result = f"ERROR (continued): {exc}"
                elif isinstance(on_error, str) and on_error.startswith("retry:"):
                    max_retries = int(on_error.split(":", 1)[1])
                    result = f"ERROR: {exc}"
                    for _ in range(max_retries):
                        try:
                            result = self._dispatch(run, step, ctx, kind)
                            break
                        except Exception as e2:
                            result = f"ERROR: {e2}"
                    else:
                        raise RuntimeError(
                            f"step '{step_id}' failed after {max_retries} retries: {result}"
                        )
                else:
                    raise

            output_var = step.get("output_var")
            if output_var:
                ctx[output_var] = result
                run.step_results[output_var] = result

            run.logs.append(f"✓ {step_id}")
            if self.on_step:
                try:
                    self.on_step(run.run_id, step_id, str(result)[:200])
                except Exception:
                    pass

    def _dispatch(self, run: WorkflowRun, step: dict, ctx: dict, kind: str) -> str:
        if kind == "tool":
            name = step["tool"]
            args = _interpolate_args(step.get("args", {}), ctx)
            return self.tools.run(name, args)

        if kind == "llm":
            prompt = _interpolate(step["prompt"], ctx)
            temp = float(step.get("temperature", 0.7))
            return self.llm.chat([{"role": "user", "content": prompt}], temp)

        if kind == "parallel":
            sub_steps = step.get("steps", [])
            sub_ctx = dict(ctx)
            results: dict[str, str] = {}

            def _run_sub(s):
                sid = s.get("id", str(uuid.uuid4())[:6])
                res = self._dispatch(run, s, sub_ctx, s.get("kind", "tool"))
                ov = s.get("output_var")
                if ov:
                    ctx[ov] = res
                    run.step_results[ov] = res
                return sid, res

            with concurrent.futures.ThreadPoolExecutor() as ex:
                futures = {ex.submit(_run_sub, s): s for s in sub_steps}
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        sid, res = fut.result()
                        results[sid] = res
                    except Exception as exc:
                        s = futures[fut]
                        results[s.get("id", "?")] = f"ERROR: {exc}"

            return json.dumps(results)

        if kind == "condition":
            test = _interpolate(step.get("test", ""), ctx)
            passed = _eval_condition(test, ctx)
            branch = step.get("then" if passed else "else", [])
            if branch:
                self._execute_steps(run, branch, ctx)
            return f"condition: {'passed' if passed else 'failed'}"

        if kind == "loop":
            max_iter = int(step.get("max_iterations", 10))
            sub_steps = step.get("steps", [])
            stop_when = step.get("stop_when", "")
            i = 0
            for i in range(max_iter):
                ctx["_loop_index"] = str(i)
                self._execute_steps(run, sub_steps, ctx)
                if stop_when and _eval_condition(
                    _interpolate(stop_when, ctx), ctx
                ):
                    break
            return f"loop: {i + 1} iteration(s)"

        if kind == "subworkflow":
            sub_name = step["workflow"]
            sub_inputs = _interpolate_args(step.get("inputs", {}), ctx)
            sub_run = self.run(sub_name, sub_inputs)
            ctx.update(sub_run.step_results)
            run.step_results.update(sub_run.step_results)
            return f"subworkflow '{sub_name}': {sub_run.status}"

        if kind == "agent_talk":
            if self.swarm is None:
                return "ERROR: swarm not initialised"
            agent_name = step["agent"]
            message = _interpolate(step["message"], ctx)
            return self.swarm.talk(agent_name, message)

        if kind == "memory_store":
            content = _interpolate(step.get("content", ""), ctx)
            if self.memory:
                self.memory.store(
                    content,
                    kind=step.get("memory_kind", "workflow"),
                    source=step.get("source", "workflow"),
                )
            return f"stored {len(content)} chars"

        if kind == "notify":
            title = _interpolate(step.get("title", "JARVIS"), ctx)
            body = _interpolate(step.get("body", ""), ctx)
            if self.deliver:
                self.deliver(title, body)
            return f"notified: {title}"

        if kind == "sleep":
            secs = min(float(step.get("seconds", 1)), 3600)
            time.sleep(secs)
            return f"slept {secs:.1f}s"

        return f"ERROR: unknown step kind '{kind}'"


# ---------------------------------------------------------------------------
# Workflow recorder — crystallise a live session into a replayable workflow
# ---------------------------------------------------------------------------

class WorkflowRecorder:
    """Record tool calls from a live agent session into a workflow definition.

    Usage:
        recorder = WorkflowRecorder()
        recorder.start()
        # ... user runs a complex task; agent calls tools ...
        recorder.record_tool_call("read_file", {"path": "x.py"}, result)
        definition = recorder.stop("my_workflow")
        store.save("my_workflow", definition)
    """

    def __init__(self) -> None:
        self._steps: list[dict] = []
        self._recording = False
        self._start_time: float = 0.0

    @property
    def recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        self._steps.clear()
        self._recording = True
        self._start_time = time.time()

    def record_tool_call(
        self, tool_name: str, args: dict, result: str = ""
    ) -> None:
        if not self._recording:
            return
        step_id = f"step_{len(self._steps) + 1:02d}"
        self._steps.append({
            "id": step_id,
            "kind": "tool",
            "tool": tool_name,
            "args": args,
            "_result_preview": result[:120] if result else "",
        })

    def record_llm_call(self, prompt: str, output_var: str = "") -> None:
        if not self._recording:
            return
        step_id = f"step_{len(self._steps) + 1:02d}"
        step: dict = {"id": step_id, "kind": "llm", "prompt": prompt}
        if output_var:
            step["output_var"] = output_var
        self._steps.append(step)

    def stop(self, name: str, description: str = "") -> dict:
        self._recording = False
        return {
            "name": name,
            "description": description or f"Recorded workflow ({len(self._steps)} steps)",
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "vars": {},
            "steps": [
                {k: v for k, v in s.items() if not k.startswith("_")}
                for s in self._steps
            ],
        }

    def discard(self) -> None:
        self._steps.clear()
        self._recording = False


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_workflow_tools(
    registry,
    store: WorkflowStore,
    runner: WorkflowRunner,
    recorder: WorkflowRecorder,
) -> None:
    from .tools import Tier, Tool

    def workflow_create(name: str, steps_json: str, description: str = "") -> str:
        """Define a workflow from a JSON steps array."""
        try:
            steps = json.loads(steps_json)
            if not isinstance(steps, list):
                return "ERROR: steps_json must be a JSON array"
        except json.JSONDecodeError as exc:
            return f"ERROR: invalid JSON — {exc}"
        definition = {
            "name": name,
            "description": description,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "vars": {},
            "steps": steps,
        }
        path = store.save(name, definition)
        return f"workflow '{name}' saved ({len(steps)} steps) → {path}"

    def workflow_run(name: str, inputs_json: str = "") -> str:
        """Execute a named workflow synchronously and return the result."""
        inputs = {}
        if inputs_json.strip():
            try:
                inputs = json.loads(inputs_json)
            except json.JSONDecodeError as exc:
                return f"ERROR: invalid inputs JSON — {exc}"
        run = runner.run(name, inputs)
        return run.summary()

    def workflow_run_async(name: str, inputs_json: str = "") -> str:
        """Start a workflow in the background and return its run_id."""
        inputs = {}
        if inputs_json.strip():
            try:
                inputs = json.loads(inputs_json)
            except json.JSONDecodeError as exc:
                return f"ERROR: {exc}"
        run_id = runner.run_async(name, inputs)
        return f"workflow '{name}' started in background — run_id: {run_id}"

    def workflow_status(run_id: str) -> str:
        run = runner.get_run(run_id)
        if run is None:
            runs = runner.list_runs(5)
            recent = ", ".join(r.run_id for r in runs)
            return f"no run found for '{run_id}'. Recent runs: {recent or 'none'}"
        return run.summary()

    def workflow_list() -> str:
        workflows = store.list_all()
        if not workflows:
            return (
                "no workflows saved yet.\n\n"
                "Create one with workflow_create, or use workflow_record_start "
                "to capture a live session."
            )
        lines = [f"{len(workflows)} workflow(s):"]
        for wf in workflows:
            lines.append(f"  {wf['name']:<20} {wf['steps']} steps  {wf['description']}")
        recent = runner.list_runs(3)
        if recent:
            lines.append("\nRecent runs:")
            for r in recent:
                lines.append(f"  {r.run_id}  [{r.workflow_name}]  {r.status}")
        return "\n".join(lines)

    def workflow_delete(name: str) -> str:
        return f"workflow '{name}' deleted." if store.delete(name) else f"no workflow named '{name}'"

    def workflow_record_start() -> str:
        if recorder.recording:
            return "already recording — call workflow_record_stop first"
        recorder.start()
        return "recording started — all tool calls will be captured into a workflow"

    def workflow_record_stop(name: str, description: str = "") -> str:
        if not recorder.recording:
            return "not currently recording"
        definition = recorder.stop(name, description)
        path = store.save(name, definition)
        n = len(definition["steps"])
        return f"recording stopped — workflow '{name}' saved ({n} steps) → {path}"

    def workflow_record_discard() -> str:
        recorder.discard()
        return "recording discarded"

    registry.register(Tool(
        "workflow_create",
        "Define a new workflow from a JSON steps array. Steps can be: tool, llm, parallel, "
        "condition, loop, subworkflow, agent_talk, memory_store, notify, sleep.",
        {"name": "workflow name", "steps_json": "JSON array of step objects",
         "description": "(optional) human description"},
        workflow_create,
    ))
    registry.register(Tool(
        "workflow_run",
        "Execute a named workflow synchronously. Blocks until done.",
        {"name": "workflow name", "inputs_json": "(optional) JSON object of input variables"},
        workflow_run, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "workflow_run_async",
        "Start a named workflow in the background. Returns a run_id to check later.",
        {"name": "workflow name", "inputs_json": "(optional) JSON object of input variables"},
        workflow_run_async, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "workflow_status",
        "Check the status and step results of a workflow run by its run_id.",
        {"run_id": "run ID returned by workflow_run_async"},
        workflow_status,
    ))
    registry.register(Tool(
        "workflow_list",
        "List all saved workflows and recent runs.",
        {},
        workflow_list,
    ))
    registry.register(Tool(
        "workflow_delete",
        "Delete a saved workflow definition.",
        {"name": "workflow name"},
        workflow_delete,
    ))
    registry.register(Tool(
        "workflow_record_start",
        "Start recording the current session as a replayable workflow. "
        "All subsequent tool calls will be captured as workflow steps.",
        {},
        workflow_record_start,
    ))
    registry.register(Tool(
        "workflow_record_stop",
        "Stop recording and save the captured tool calls as a named workflow.",
        {"name": "name for the new workflow",
         "description": "(optional) description of what this workflow does"},
        workflow_record_stop,
    ))
    registry.register(Tool(
        "workflow_record_discard",
        "Discard the current recording without saving.",
        {},
        workflow_record_discard,
    ))
