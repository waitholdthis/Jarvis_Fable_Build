"""Interactive terminal UI: streaming chat, slash commands, tool confirmations."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import __version__
from .agent import Agent
from .config import Config
from .ingest import ingest_path
from .llm import LLMClient, NoRuntimeError, detect
from .memory import Memory
from .tools import ToolRegistry
from . import voice

console = Console()

_HELP = """\
Talk normally, or use a command:
  /ingest <path>   index a file or directory into memory
  /memory          show memory statistics
  /forget <text>   delete memory entries matching text
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
        llm = _connect(config)
    except NoRuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    memory = Memory(config.db_path)
    tools = ToolRegistry(config, memory)
    agent = Agent(config, llm, memory, tools, confirm=_confirm)
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

    memory.close()
    return 0


def voice_loop(config: Config) -> int:
    """Hands-free conversation: speak, get spoken answers. Ctrl-C to exit."""
    if not voice.asr_available():
        console.print(
            "[red]voice mode needs the extras: "
            "pip install 'jarvis-assistant\\[voice]'[/red]"
        )
        return 1
    config.ensure_dirs()
    config.voice_enabled = True
    try:
        llm = _connect(config)
    except NoRuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    memory = Memory(config.db_path)
    tools = ToolRegistry(config, memory)
    agent = Agent(config, llm, memory, tools, confirm=_confirm)

    console.print(
        f"[bold]{config.assistant_name} voice mode[/bold] — just talk; "
        'say "goodbye" or press Ctrl-C to stop.\n'
    )
    with console.status("Loading speech model..."):
        voice.record_and_transcribe(seconds=0.1, model_size=config.whisper_model)

    try:
        while True:
            console.print("[dim]listening...[/dim]")
            heard = voice.listen_until_silence(model_size=config.whisper_model)
            if not heard:
                continue
            console.print(f"[bold green]you:[/bold green] {heard}")
            if heard.strip(" .!?").lower() in ("goodbye", "bye", "stop", "quit"):
                voice.speak("Goodbye.")
                break
            run_turn(agent, config, heard)
    except KeyboardInterrupt:
        console.print("\nGoodbye.")
    memory.close()
    return 0


def serve_web(config: Config, port: int = 8765) -> int:
    """Run the local web UI (stdlib server, bound to 127.0.0.1)."""
    from .web import serve

    config.ensure_dirs()
    try:
        llm = _connect(config)
    except NoRuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    memory = Memory(config.db_path)
    tools = ToolRegistry(config, memory)
    agent = Agent(config, llm, memory, tools)
    console.print(
        f"[green]✓[/green] web UI at [bold cyan]http://127.0.0.1:{port}[/bold cyan] "
        "(local machine only — Ctrl-C to stop)"
    )
    try:
        serve(agent, host="127.0.0.1", port=port)
    except KeyboardInterrupt:
        console.print("\nStopped.")
    finally:
        memory.close()
    return 0


def ask_once(config: Config, question: str) -> int:
    """One-shot mode: `jarvis ask "..."` for scripts and quick queries."""
    config.ensure_dirs()
    try:
        llm = _connect(config)
    except NoRuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    memory = Memory(config.db_path)
    tools = ToolRegistry(config, memory)
    agent = Agent(config, llm, memory, tools, confirm=_confirm)
    run_turn(agent, config, question)
    memory.close()
    return 0
