"""Entry point: `jarvis` or `python -m jarvis`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="Local-first personal AI assistant that runs on any computer.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", help="interactive chat (default)")

    ask = sub.add_parser("ask", help="ask one question and exit")
    ask.add_argument("question", nargs="+", help="the question")

    ingest = sub.add_parser("ingest", help="index files into memory")
    ingest.add_argument("path", help="file or directory to ingest")

    serve = sub.add_parser("serve", help="run the local web UI")
    serve.add_argument("--port", type=int, default=8765, help="port (default 8765)")

    voice_cmd = sub.add_parser(
        "voice", help="hands-free voice conversation (needs [voice])"
    )
    voice_cmd.add_argument(
        "--wake",
        action="store_true",
        help='only respond when addressed by name ("Jarvis, ...")',
    )

    sub.add_parser(
        "consolidate", help="distill recent conversations into long-term facts now"
    )

    routine = sub.add_parser("routine", help="run a named routine (no name = list)")
    routine.add_argument("name", nargs="?", default="")

    sub.add_parser("briefing", help="run the built-in briefing routine")

    sub.add_parser("doctor", help="check runtimes, voice, and memory health")

    args = parser.parse_args(argv)

    from .config import Config

    config = Config.load()

    if args.command == "ask":
        from .cli import ask_once

        return ask_once(config, " ".join(args.question))

    if args.command == "ingest":
        from .cli import console
        from .ingest import ingest_path
        from .memory import Memory

        config.ensure_dirs()
        memory = Memory(config.db_path)
        files, chunks = ingest_path(memory, Path(args.path))
        console.print(f"ingested {files} file(s), {chunks} chunk(s) into {config.db_path}")
        memory.close()
        return 0

    if args.command == "serve":
        from .cli import serve_web

        return serve_web(config, port=args.port)

    if args.command == "voice":
        from .cli import voice_loop

        return voice_loop(config, wake=args.wake)

    if args.command == "consolidate":
        from .cli import consolidate_now

        return consolidate_now(config)

    if args.command == "routine":
        from .cli import routine_cmd

        return routine_cmd(config, args.name)

    if args.command == "briefing":
        from .cli import routine_cmd

        return routine_cmd(config, "briefing")

    if args.command == "doctor":
        return _doctor(config)

    from .cli import repl

    return repl(config)


def _doctor(config) -> int:
    """Report what this machine can and cannot do, without failing."""
    from .cli import console
    from .llm import NoRuntimeError, detect
    from .memory import Memory
    from . import voice

    console.print("[bold]jarvis doctor[/bold]")
    try:
        client = detect(config)
        console.print(
            f"  llm: [green]ok[/green] — {client.runtime_name} "
            f"({client.api_base}, model {client.model})"
        )
    except NoRuntimeError:
        console.print(
            "  llm: [red]none found[/red] — install Ollama/LM Studio or set JARVIS_API_BASE"
        )

    config.ensure_dirs()
    memory = Memory(config.db_path)
    stats = memory.stats()
    console.print(f"  memory: [green]ok[/green] — {stats['db_path']} "
                  f"({stats['semantic_chunks']} chunks, embedder: {stats['embedder']})")
    memory.close()

    if voice.asr_available():
        console.print("  voice input: [green]ok[/green]")
    else:
        console.print(
            "  voice input: [yellow]not installed[/yellow] "
            "(pip install 'jarvis-assistant\\[voice]')"
        )
    if config.mcp_servers:
        import shutil

        for name, spec in config.mcp_servers.items():
            command = spec.get("command") if isinstance(spec, dict) else None
            if not isinstance(command, list) or not command:
                console.print(f"  mcp '{name}': [red]missing command in config[/red]")
            elif shutil.which(str(command[0])):
                console.print(f"  mcp '{name}': [green]command found[/green] ({command[0]})")
            else:
                console.print(f"  mcp '{name}': [yellow]'{command[0]}' not on PATH[/yellow]")
    else:
        console.print("  mcp servers: none configured (optional)")

    console.print(f"  workspace: {config.workspace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
