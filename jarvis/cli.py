"""Interactive terminal UI: streaming chat, slash commands, tool confirmations."""

from __future__ import annotations

import threading
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import __version__
from .agent import Agent
from .config import Config
from .consolidate import maybe_consolidate
from .hub import hub_from_config, register_hub_tools
from .ingest import ingest_path
from .llm import LLMClient, NoRuntimeError, detect
from .mcp import register_mcp_tools
from .memory import Memory
from .monitor import EventMonitor, SelfHealer, register_monitor_tools
from .notify import desktop_notify
from .routines import get_routines, run_routine, sync_routine_schedules
from .router import IceEnsemble, PromptRouter, register_router_tools
from .scheduler import Scheduler
from .tools import ToolRegistry, register_scheduler_tools
from .vault import get_vault, register_vault_tools
from .vision import register_vision_tool
from .workspace_google import GoogleWorkspaceClient, register_google_tools
from . import voice

# Second-round blueprint modules (imported lazily to keep startup fast)
# face_id, docker_sandbox, knowledge_graph, fatigue, streaming,
# cloud_sandbox, reports, calibration

console = Console()

_HELP = """\
Talk normally, or use a command:
  /ingest <path>   index a file or directory into memory
  /memory          show memory statistics
  /forget <text>   delete memory entries matching text
  /routine [name]  run a routine (no name = list them)
  /tools           list available tools
  /voice           toggle spoken responses
  /listen          record ~6s from the microphone and send it (needs [voice])
  /new             clear the in-flight conversation (memory is kept)
  /help            this help
  /quit            exit\
"""


def _confirm(prompt: str) -> bool:
    console.print(f"[bold yellow]⚠ {prompt}[/bold yellow] \\[y/N] ", end="")
    try:
        return input().strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _connect(config: Config) -> LLMClient:
    with console.status("Looking for a local LLM runtime..."):
        client = detect(config)
    console.print(
        f"[green]✓[/green] {client.runtime_name} at [cyan]{client.api_base}[/cyan] "
        f"— model [bold]{client.model}[/bold]"
    )
    return client


def _bootstrap(
    config: Config,
    confirm=_confirm,
    consolidation: bool = True,
    start_scheduler: bool = True,
    on_notify=None,
):
    """Shared startup: LLM, memory, tools, MCP, scheduler, agent, hippocampus.

    Also bootstraps (when configured):
      - credential vault          (Section 12)
      - out-of-band hub           (Section 12)
      - filesystem event monitor  (Section 4)
      - multi-model router + ICE  (Section 8)
      - Google Workspace client   (Section 5)

    Returns (agent, memory, mcp_servers, scheduler). Raises NoRuntimeError
    when no LLM runtime is available; callers print it and exit.
    """
    config.ensure_dirs()
    llm = _connect(config)
    memory = Memory(config.db_path)
    tools = ToolRegistry(config, memory)
    mcp_servers = register_mcp_tools(
        tools, config.mcp_servers, notify=lambda t: console.print(f"[dim]{t}[/dim]")
    )
    register_vision_tool(tools, llm)

    scheduler = Scheduler(config.db_path)
    register_scheduler_tools(tools, scheduler)
    sync_routine_schedules(scheduler, get_routines(config))

    # ---- Section 12: Credential vault ---------------------------------------
    try:
        vault = get_vault(config.home)
        register_vault_tools(tools, vault)
    except Exception as exc:
        console.print(f"[dim](vault unavailable: {exc})[/dim]")

    # ---- Section 12: Out-of-band notification hub ---------------------------
    hub = hub_from_config(config)
    if hub is not None:
        register_hub_tools(tools, hub)
        console.print(f"[dim](hub: {type(hub).__name__} active)[/dim]")

    # ---- Section 8: Multi-model router + ICE ensemble -----------------------
    router = PromptRouter(default=llm)
    ice: IceEnsemble | None = None
    if config.model_tiers:
        from .llm import cloud_client
        tier_clients = {}
        for tier_name, spec in config.model_tiers.items():
            if isinstance(spec, dict) and spec.get("provider"):
                try:
                    tier_clients[tier_name] = cloud_client(
                        config, spec["provider"], spec.get("model", "")
                    )
                except Exception as exc:
                    console.print(f"[dim](tier '{tier_name}' unavailable: {exc})[/dim]")
        router = PromptRouter(default=llm, tiers=tier_clients)
        if config.ice_enabled:
            heavy = tier_clients.get("heavy", llm)
            ice = IceEnsemble(generator=heavy, tester=heavy, judge=heavy)
    register_router_tools(tools, router, ice)

    # ---- Section 4: Self-healer + filesystem monitor ------------------------
    healer = SelfHealer(llm=llm, workspace=config.workspace)
    register_monitor_tools(tools, healer)

    monitor: EventMonitor | None = None
    if config.monitor_enabled and config.monitor_paths:
        monitor = EventMonitor()
        for p in config.monitor_paths:
            import os
            monitor.watch(os.path.expanduser(str(p)))

        def _on_anomaly(event) -> None:
            msg = f"[bold red]⚡ ANOMALY[/bold red] {event.pattern}: {len(event.events)} event(s)"
            console.print(msg)
            desktop_notify("JARVIS Monitor", f"{event.pattern}: {len(event.events)} event(s)")
            if hub is not None:
                from .hub import HubMessage
                hub.send(HubMessage(
                    title="JARVIS Anomaly Detected",
                    body=f"Pattern: {event.pattern}\nEvents: {len(event.events)}",
                    kind="alert",
                ))

        monitor.on_anomaly = _on_anomaly
        monitor.start()
        console.print(
            f"[dim](monitor: watching {len(config.monitor_paths)} path(s))[/dim]"
        )

    # ---- Section 5: Google Workspace ----------------------------------------
    creds_path = config.google_credentials_path
    google_creds = Path(creds_path).expanduser() if creds_path else config.home / "google_credentials.json"
    if google_creds.exists():
        try:
            gws = GoogleWorkspaceClient(config.home)
            register_google_tools(tools, gws)
            console.print("[dim](Google Workspace: Gmail + Calendar tools active)[/dim]")
        except Exception as exc:
            console.print(f"[dim](Google Workspace unavailable: {exc})[/dim]")

    # ---- Section 1: Face recognition ----------------------------------------
    if config.face_recognition_enabled:
        try:
            from .face_id import FaceDatabase, FaceRecognizer, register_face_tools
            face_db_path = (
                Path(config.face_db_path).expanduser()
                if config.face_db_path
                else config.home / "faces.db"
            )
            face_db = FaceDatabase(face_db_path)
            face_rec = FaceRecognizer(face_db)
            register_face_tools(tools, face_rec, face_db)
            console.print("[dim](face recognition: active)[/dim]")
        except Exception as exc:
            console.print(f"[dim](face recognition unavailable: {exc})[/dim]")

    # ---- Section 2: Docker sandbox + digital twin + tool synthesis ----------
    if config.docker_enabled:
        try:
            from .docker_sandbox import (
                DigitalTwin, DockerSandbox, SandboxRegistry, SandboxSpec,
                ToolSynthesizer, register_sandbox_tools,
            )
            sandbox_registry = SandboxRegistry(config.home / "sandboxes.json")
            default_spec = SandboxSpec(image=config.docker_default_image)
            digital_twin = DigitalTwin(
                workspace=config.workspace, image=config.docker_default_image
            )
            synthesizer = ToolSynthesizer(
                llm=llm, workspace=config.workspace, registry=sandbox_registry
            )
            register_sandbox_tools(tools, default_spec, digital_twin, synthesizer)
            console.print("[dim](Docker sandbox: digital twin + tool synthesis active)[/dim]")
        except Exception as exc:
            console.print(f"[dim](Docker sandbox unavailable: {exc})[/dim]")

    # ---- Section 3: Knowledge graph + memory router -------------------------
    kg = None
    mem_router = None
    if config.knowledge_graph_enabled:
        try:
            from .knowledge_graph import (
                ASTIngester, KnowledgeGraph, MemoryRouter, register_graph_tools,
            )
            kg = KnowledgeGraph(config.home / "knowledge_graph.db")
            ast_ingester = ASTIngester(kg)
            mem_router = MemoryRouter(kg, memory)
            register_graph_tools(tools, kg, ast_ingester, mem_router)
            console.print("[dim](knowledge graph: AST ingestion + memory router active)[/dim]")
        except Exception as exc:
            console.print(f"[dim](knowledge graph unavailable: {exc})[/dim]")

    # ---- Section 4: Fatigue monitor (remaining signals) ---------------------
    fatigue_monitor = None
    if config.fatigue_monitor_enabled:
        try:
            from .fatigue import FatigueMonitor, register_fatigue_tools

            def _vision_posture(prompt: str) -> str:
                t = tools.get("see_screen")
                return t.func(prompt) if t else ""

            fatigue_monitor = FatigueMonitor(
                llm=llm,
                sample_interval_s=config.fatigue_sample_interval,
                vision_tool=_vision_posture,
            )

            def _on_fatigue(score) -> None:
                deliver(
                    f"{config.assistant_name} · Fatigue Alert",
                    score.summary(),
                )

            fatigue_monitor.on_alert = _on_fatigue
            register_fatigue_tools(tools, fatigue_monitor)
            fatigue_monitor.start()
            console.print("[dim](fatigue monitor: started)[/dim]")
        except Exception as exc:
            console.print(f"[dim](fatigue monitor unavailable: {exc})[/dim]")

    # ---- Section 6: Real-time streaming / OLAP ------------------------------
    if config.streaming_enabled:
        try:
            from .streaming import (
                MetricsEngine, OLAPStore, TelemetryIngestor, register_streaming_tools,
            )
            metrics_engine = MetricsEngine()
            olap_store = OLAPStore(config.home / "telemetry")
            TelemetryIngestor(metrics_engine, olap_store).start()
            register_streaming_tools(tools, metrics_engine, olap_store)

            def _on_metric_anomaly(event, snap) -> None:
                deliver(
                    f"{config.assistant_name} · Metric Anomaly",
                    f"{event.name}={event.value:.4f}  z={snap.get('z_score', 0):.2f}",
                )

            metrics_engine.on_anomaly(_on_metric_anomaly)
            console.print("[dim](streaming: metrics + OLAP active)[/dim]")
        except Exception as exc:
            console.print(f"[dim](streaming unavailable: {exc})[/dim]")

    # ---- Section 7: Cloud sandboxes + session snapshots ---------------------
    try:
        from .cloud_sandbox import (
            HydrationSyncClient, HydrationManifest, ModalBackend,
            SessionSnapshot, register_cloud_tools,
        )
        modal_backend = ModalBackend(config.modal_app_name)
        hydration = HydrationSyncClient(
            local_root=config.workspace,
            manifest_path=config.home / "hydration_manifest.json",
            base_url=config.hydration_base_url,
        )
        if config.hydration_base_url:
            hydration.start_daemon()
        snapshots = SessionSnapshot(config.home / "sessions")
        # agent is created below; pass as None then patch afterwards
        register_cloud_tools(tools, modal_backend, None, hydration, snapshots)
        console.print(
            f"[dim](cloud: {'Modal + ' if modal_backend.available() else ''}sessions + hydration active)[/dim]"
        )
    except Exception as exc:
        console.print(f"[dim](cloud sandbox unavailable: {exc})[/dim]")

    # ---- Section 11: Reports engine -----------------------------------------
    try:
        from .reports import (
            BackgroundReporter, BlueprintEngine, LatexPDFEngine,
            register_report_tools,
        )
        reports_dir = (
            Path(config.reports_output_dir).expanduser()
            if config.reports_output_dir
            else config.home / "reports"
        )
        latex_engine = LatexPDFEngine(reports_dir)
        blueprint_engine = BlueprintEngine(reports_dir)
        reporter = BackgroundReporter(reports_dir, llm, latex_engine)
        register_report_tools(tools, reporter, latex_engine, blueprint_engine)
        console.print("[dim](reports engine: LaTeX/PDF + SVG + background briefings active)[/dim]")
    except Exception as exc:
        console.print(f"[dim](reports engine unavailable: {exc})[/dim]")

    # ---- Internet access (web search, API calls, real-time data) -----------
    if config.internet_enabled:
        try:
            from .internet import register_internet_tools
            # vault is available from Section 12 bootstrap above (may be None)
            _vault = None
            try:
                _vault = get_vault(config.home)
            except Exception:
                pass
            register_internet_tools(tools, config.home, vault=_vault)
            console.print("[dim](internet: web search + API calls + real-time data ready)[/dim]")
        except Exception as exc:
            console.print(f"[dim](internet module unavailable: {exc})[/dim]")

    # ---- Workflow engine ----------------------------------------------------
    workflow_runner = None
    workflow_recorder = None
    if config.workflows_enabled:
        try:
            from .workflow import (
                WorkflowRecorder, WorkflowRunner, WorkflowStore,
                register_workflow_tools,
            )
            wf_store = WorkflowStore(config.home / "workflows")
            workflow_recorder = WorkflowRecorder()
            workflow_runner = WorkflowRunner(
                store=wf_store,
                tools=tools,
                llm=llm,
                memory=memory,
                deliver=None,  # patched to deliver() after it is defined below
            )
            register_workflow_tools(tools, wf_store, workflow_runner, workflow_recorder)
            console.print("[dim](workflow engine: ready)[/dim]")
        except Exception as exc:
            console.print(f"[dim](workflow engine unavailable: {exc})[/dim]")

    # ---- Agent swarm --------------------------------------------------------
    if config.swarm_enabled:
        try:
            from .swarm import SwarmOrchestrator, register_swarm_tools
            swarm = SwarmOrchestrator(llm=llm, memory=memory, tool_registry=tools)
            register_swarm_tools(tools, swarm)
            console.print("[dim](agent swarm: ready — use agent_spawn / agent_council)[/dim]")
        except Exception as exc:
            console.print(f"[dim](agent swarm unavailable: {exc})[/dim]")

    # ---- Section 12: LoRA calibration ---------------------------------------
    if config.calibration_enabled:
        try:
            from .calibration import (
                CalibrationJobConfig, CalibrationRunner, CalibrationStore,
                DiscrepancyEvaluator, register_calibration_tools,
            )
            cal_store = CalibrationStore(config.home / "calibration" / "pairs.jsonl")
            cal_evaluator = DiscrepancyEvaluator(llm)
            cal_job_config = CalibrationJobConfig(
                backend=config.calibration_backend,
                base_model=config.calibration_base_model,
                custom_command=config.calibration_custom_command,
                min_pairs=config.calibration_min_pairs,
            )
            cal_runner = CalibrationRunner(
                store=cal_store,
                evaluator=cal_evaluator,
                training_dir=config.home / "calibration",
                job_config=cal_job_config,
            )

            def _on_calibration_swap(model_name: str) -> None:
                deliver(
                    f"{config.assistant_name} · Calibration Complete",
                    f"Hot-swapping to calibrated model: {model_name}",
                )
                try:
                    llm.model = model_name
                except Exception:
                    pass

            cal_runner.on_hot_swap(_on_calibration_swap)
            register_calibration_tools(tools, cal_store, cal_runner, cal_evaluator)
            console.print("[dim](LoRA calibration: training pair collection active)[/dim]")
        except Exception as exc:
            console.print(f"[dim](calibration unavailable: {exc})[/dim]")

    # ---- Wire anomaly monitor → fatigue friction signal ---------------------
    if fatigue_monitor is not None and monitor is not None:
        _orig_on_anomaly = monitor.on_anomaly

        def _chained_anomaly(event) -> None:
            if _orig_on_anomaly:
                _orig_on_anomaly(event)
            fatigue_monitor.record_build_failure()

        monitor.on_anomaly = _chained_anomaly

    def deliver(title: str, body: str) -> None:
        console.print(f"\n[bold magenta]🔔 {title}[/bold magenta] {body}")
        desktop_notify(title, body)
        if hub is not None:
            from .hub import HubMessage
            hub.send(HubMessage(title=title, body=body, kind="info"))
        if config.voice_enabled:
            voice.speak(f"{title}. {body}")
        if on_notify is not None:
            on_notify(title, body)

    # Patch deliver into modules that were created before it was defined
    if workflow_runner is not None:
        workflow_runner.deliver = deliver

    # ---- Autonomous operation -----------------------------------------------
    if config.autonomy_enabled:
        try:
            from .autonomy import (
                AutonomousWorker, CuriosityEngine, GoalQueue,
                ReflectionEngine, register_autonomy_tools,
            )
            goal_queue = GoalQueue(config.home / "goals.db")
            curiosity_engine = CuriosityEngine(llm=llm, memory=memory, tools=tools)
            reflection_engine = ReflectionEngine(llm=llm, memory=memory)
            auto_worker = AutonomousWorker(
                goal_queue=goal_queue,
                curiosity=curiosity_engine,
                reflector=reflection_engine,
                llm=llm,
                tools=tools,
                memory=memory,
                workflow_runner=workflow_runner,
                poll_interval=config.autonomy_poll_interval,
                curiosity_interval_hours=config.autonomy_curiosity_interval_hours,
                reflection_interval_hours=config.autonomy_reflection_interval_hours,
            )
            auto_worker.on_progress = deliver
            register_autonomy_tools(
                tools, goal_queue, curiosity_engine, reflection_engine,
                auto_worker,
                agent_messages_ref=[],  # patched after agent is created
            )
            console.print("[dim](autonomous worker: ready)[/dim]")
        except Exception as exc:
            console.print(f"[dim](autonomous worker unavailable: {exc})[/dim]")
            auto_worker = None
    else:
        auto_worker = None

    def on_due(task) -> None:
        if task.kind == "routine":
            body = run_routine(config, llm, memory, tools, task.payload) or ""
            deliver(f"{config.assistant_name} · {task.payload}", body)
        else:
            deliver(f"{config.assistant_name} reminder", task.payload)

    scheduler.on_due = on_due
    if start_scheduler:
        scheduler.start()

    agent = Agent(config, llm, memory, tools, confirm=confirm)

    # Patch agent reference into modules that need it after creation
    if auto_worker is not None:
        auto_worker._session_messages = agent.messages
        auto_worker.start(agent.messages)
        console.print("[dim](autonomous worker: started)[/dim]")

    # Wire curiosity engine to observe assistant messages in real time
    if auto_worker is not None:
        _orig_log = memory.log

        def _observing_log(role: str, content: str) -> None:
            _orig_log(role, content)
            try:
                auto_worker.curiosity.observe_message(role, content)
            except Exception:
                pass

        memory.log = _observing_log

    # Wire workflow recorder into agent's tool-call stream
    if workflow_recorder is not None:
        _orig_execute = agent._execute

        def _recording_execute(name: str, args: dict) -> str:
            result = _orig_execute(name, args)
            if workflow_recorder.recording:
                workflow_recorder.record_tool_call(name, args, result)
            return result

        agent._execute = _recording_execute

    if consolidation and config.consolidation_enabled:
        def hippocampus():
            try:
                stored = maybe_consolidate(
                    memory,
                    llm,
                    min_new=config.consolidation_min_new,
                    min_interval_hours=config.consolidation_interval_hours,
                )
                if stored:
                    console.print(
                        f"[dim](memory consolidation: {stored} new fact(s) learned)[/dim]"
                    )
            except Exception:
                pass  # background job; never disturb the session

        threading.Thread(target=hippocampus, daemon=True).start()

    return agent, memory, mcp_servers, scheduler


def _teardown(memory: Memory, mcp_servers: list, scheduler: Scheduler | None = None) -> None:
    if scheduler is not None:
        scheduler.close()
    for server in mcp_servers:
        server.close()
    memory.close()


def _speak_if_enabled(config: Config, text: str) -> None:
    if config.voice_enabled and text:
        if not voice.speak(text):
            console.print("[dim](no TTS backend available; /voice to disable)[/dim]")


def run_turn(agent: Agent, config: Config, user_input: str) -> None:
    console.print()
    streamed_any = False
    final = ""
    for event in agent.run(user_input):
        if event.kind == "text":
            if not streamed_any:
                console.print(f"[bold blue]{config.assistant_name}:[/bold blue] ", end="")
                streamed_any = True
            console.print(event.text, end="", soft_wrap=True)
        elif event.kind == "tool_call":
            console.print(f"\n[dim]→ {event.tool}({event.args})[/dim]")
            streamed_any = False
        elif event.kind == "tool_result":
            preview = event.text if len(event.text) < 300 else event.text[:300] + "…"
            console.print(f"[dim]← {preview}[/dim]")
        elif event.kind == "done":
            final = event.text
    console.print("\n")
    _speak_if_enabled(config, final)


def repl(config: Config) -> int:
    config.ensure_dirs()
    console.print(
        Panel.fit(
            f"[bold]{config.assistant_name}[/bold] v{__version__} — local-first assistant\n"
            f"memory: {config.db_path}\nworkspace: {config.workspace}\n"
            "Type /help for commands.",
            border_style="blue",
        )
    )
    try:
        agent, memory, mcp_servers, scheduler = _bootstrap(config)
    except NoRuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    tools = agent.tools
    console.print(f"[dim]embedder: {memory.embedder.name}[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold green]you:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, _, arg = user_input.partition(" ")
            arg = arg.strip()
            if cmd in ("/quit", "/exit", "/q"):
                console.print("Goodbye.")
                break
            elif cmd == "/help":
                console.print(_HELP)
            elif cmd == "/ingest":
                if not arg:
                    console.print("usage: /ingest <path>")
                    continue
                target = Path(arg)
                if not target.expanduser().exists():
                    console.print(f"[red]not found: {target}[/red]")
                    continue
                with console.status(f"Ingesting {target}..."):
                    files, chunks = ingest_path(memory, target)
                console.print(f"[green]✓[/green] ingested {files} file(s), {chunks} chunk(s)")
            elif cmd == "/memory":
                for key, value in memory.stats().items():
                    console.print(f"  {key}: {value}")
            elif cmd == "/forget":
                if not arg:
                    console.print("usage: /forget <text>")
                    continue
                removed = memory.forget(arg)
                console.print(f"removed {removed} entr{'y' if removed == 1 else 'ies'}")
            elif cmd == "/routine":
                routines = get_routines(config)
                if not arg:
                    for name in routines:
                        console.print(f"  {name}")
                elif arg in routines:
                    with console.status(f"Running routine '{arg}'..."):
                        result = run_routine(config, agent.llm, memory, tools, arg)
                    console.print(result or "(no output)")
                else:
                    console.print(f"[red]no routine named '{arg}'[/red]")
            elif cmd == "/tools":
                console.print(tools.describe_all())
            elif cmd == "/voice":
                config.voice_enabled = not config.voice_enabled
                console.print(f"voice output: {'on' if config.voice_enabled else 'off'}")
            elif cmd == "/listen":
                if not voice.asr_available():
                    console.print(
                        "[red]voice input needs the extras: "
                        "pip install 'jarvis-assistant[voice]'[/red]"
                    )
                    continue
                console.print("[dim]listening (~6 s)...[/dim]")
                heard = voice.record_and_transcribe(model_size=config.whisper_model)
                if not heard:
                    console.print("[dim](heard nothing)[/dim]")
                    continue
                console.print(f"[bold green]you (voice):[/bold green] {heard}")
                run_turn(agent, config, heard)
            elif cmd == "/new":
                agent.messages.clear()
                console.print("conversation cleared (long-term memory kept)")
            else:
                console.print(f"unknown command {cmd}; /help for help")
            continue

        try:
            run_turn(agent, config, user_input)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
        except Exception as exc:
            console.print(f"\n[red]error: {type(exc).__name__}: {exc}[/red]")

    _teardown(memory, mcp_servers, scheduler)
    return 0


def voice_loop(config: Config, wake: bool = False) -> int:
    """Hands-free conversation: speak, get spoken answers. Ctrl-C to exit.

    With wake=True, utterances are ignored unless they start with the
    assistant's name ("Jarvis, ...") — fuzzy-matched on the transcript, so
    no separate wake-word model is needed.
    """
    if not voice.asr_available():
        console.print(
            "[red]voice mode needs the extras: "
            "pip install 'jarvis-assistant\\[voice]'[/red]"
        )
        return 1
    config.voice_enabled = True
    try:
        agent, memory, mcp_servers, scheduler = _bootstrap(config)
    except NoRuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    name = config.assistant_name
    console.print(
        f"[bold]{name} voice mode[/bold] — "
        + (f'say "{name}, ..." to give a command; ' if wake else "just talk; ")
        + 'say "goodbye" or press Ctrl-C to stop.\n'
    )
    with console.status("Loading speech model..."):
        voice.record_and_transcribe(seconds=0.1, model_size=config.whisper_model)

    try:
        while True:
            console.print("[dim]listening...[/dim]")
            heard = voice.listen_until_silence(model_size=config.whisper_model)
            if not heard:
                continue
            if wake:
                command = voice.strip_wake_word(heard, name)
                if command is None:
                    console.print(f"[dim](not addressed: {heard})[/dim]")
                    continue
                if not command:  # just the name: acknowledge, then listen
                    voice.speak("Yes?")
                    console.print("[dim]yes? listening...[/dim]")
                    command = voice.listen_until_silence(
                        model_size=config.whisper_model
                    )
                    if not command:
                        continue
                heard = command
            console.print(f"[bold green]you:[/bold green] {heard}")
            if heard.strip(" .!?").lower() in ("goodbye", "bye", "stop", "quit"):
                voice.speak("Goodbye.")
                break
            run_turn(agent, config, heard)
    except KeyboardInterrupt:
        console.print("\nGoodbye.")
    _teardown(memory, mcp_servers, scheduler)
    return 0


def serve_web(config: Config, port: int = 8765, host: str = "127.0.0.1") -> int:
    """Run the local web UI."""
    from .web import make_server

    # Scheduler notifications also land in the browser; the sink is swapped
    # in once the server (and its event queue) exists.
    sink = {"push": lambda event: None}
    try:
        agent, memory, mcp_servers, scheduler = _bootstrap(
            config,
            on_notify=lambda title, body: sink["push"](
                {"kind": "notify", "title": title, "text": body}
            ),
        )
    except NoRuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    server = make_server(agent, host=host, port=port)
    sink["push"] = server.ui.push
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    console.print(
        f"[green]✓[/green] web UI at [bold cyan]http://{display_host}:{port}[/bold cyan] "
        "(local machine only — Ctrl-C to stop)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\nStopped.")
    finally:
        server.server_close()
        _teardown(memory, mcp_servers, scheduler)
    return 0


def ask_once(config: Config, question: str) -> int:
    """One-shot mode: `jarvis ask "..."` for scripts and quick queries."""
    try:
        agent, memory, mcp_servers, scheduler = _bootstrap(
            config, consolidation=False, start_scheduler=False
        )
    except NoRuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    run_turn(agent, config, question)
    _teardown(memory, mcp_servers, scheduler)
    return 0


def routine_cmd(config: Config, name: str = "") -> int:
    """`jarvis routine [name]` / `jarvis briefing`: run or list routines."""
    routines = get_routines(config)
    if not name:
        console.print("[bold]routines:[/bold]")
        for routine_name, spec in routines.items():
            schedule = spec.get("schedule")
            extra = f" (scheduled: {schedule})" if schedule else ""
            console.print(f"  {routine_name}{extra}")
        return 0
    if name not in routines:
        console.print(f"[red]no routine named '{name}'[/red]")
        return 1
    try:
        agent, memory, mcp_servers, scheduler = _bootstrap(
            config, consolidation=False, start_scheduler=False
        )
    except NoRuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    with console.status(f"Running routine '{name}'..."):
        result = run_routine(config, agent.llm, memory, agent.tools, name)
    console.print(result or "(no output)")
    _speak_if_enabled(config, result or "")
    _teardown(memory, mcp_servers, scheduler)
    return 0


def consolidate_now(config: Config) -> int:
    """`jarvis consolidate`: force the hippocampus job to run immediately."""
    from .consolidate import consolidate

    config.ensure_dirs()
    try:
        llm = _connect(config)
    except NoRuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    memory = Memory(config.db_path)
    with console.status("Consolidating episodic memory into facts..."):
        stored = consolidate(memory, llm)
    console.print(
        f"[green]✓[/green] {stored} new fact(s) learned"
        if stored
        else "nothing new to consolidate"
    )
    memory.close()
    return 0
