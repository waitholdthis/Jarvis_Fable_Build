# Jarvis — a local-first AI assistant that runs on any computer

Jarvis is a personal assistant that lives entirely on your machine: your
conversations, your files, and your memory never leave it. It auto-detects
whatever LLM runtime you already have (Ollama, LM Studio, llamafile,
llama.cpp server, vLLM, Jan — anything OpenAI-compatible), works CPU-only,
and degrades gracefully: no GPU, no audio hardware, no extra ML libraries
required for the core experience.

This is the "runs anywhere" implementation of the full workstation-class
design in [`JARVIS_ARCHITECTURE_SPEC.md`](JARVIS_ARCHITECTURE_SPEC.md) — the
same architecture (tiered memory, hybrid retrieval, policy-gated agentic
tools), scaled from a dual-GPU node down to a laptop.

## What it does

- **Chat with any local model**, streaming, in your terminal.
- **Persistent memory** — an episodic log of every exchange plus a semantic
  store of facts and documents, in a single SQLite file.
- **RAG over your files** — `/ingest ~/notes` indexes text, markdown, code,
  configs, and logs; answers cite their `[source]`.
- **Agentic tool use** — the model can read files, write inside its
  workspace, search memory, remember facts, check system info, run shell
  commands, and fetch URLs. Risky tools (shell, network) always ask you
  first; file writes are confined to `~/.jarvis/workspace`; reads to your
  home directory. Works with *any* model — tool calling is prompt-based, no
  function-calling API needed.
- **Optional voice** — push-to-talk transcription (faster-whisper) and
  spoken replies (pyttsx3 or your OS's built-in speech), via the `[voice]`
  extra. `jarvis voice` is a fully hands-free loop: an energy-based VAD
  waits for you to speak, records until you go quiet, transcribes, and
  answers aloud. Without the extra, Jarvis is simply a text assistant.
- **Local web UI** — `jarvis serve` hosts a chat page at
  `http://127.0.0.1:8765` built entirely on the Python standard library
  (no web framework): streaming via Server-Sent Events, live tool-call
  visibility, and in-browser Allow/Deny buttons for permission-gated tools.
  Localhost-only by design.

## Quickstart

```bash
# 1. Have a model runtime. The zero-config path is Ollama:
#      https://ollama.com  →  ollama pull qwen2.5:7b
#    (LM Studio, llamafile, llama.cpp server, vLLM, and Jan also work.)

# 2. Install Jarvis
pip install -e .

# 3. Talk
jarvis
```

```
you: what's on this machine?
Jarvis: → system_info()
        You're on Linux 6.18 (x86_64), 16 CPUs, 412 GB free...

you: /ingest ~/Documents/notes
✓ ingested 37 file(s), 214 chunk(s)

you: what did I write about the postgres migration?
Jarvis: According to [notes/db-plan.md], you planned to...
```

### Commands

| Command | Effect |
|---|---|
| `jarvis` | interactive chat |
| `jarvis ask "..."` | one-shot question, for scripts |
| `jarvis ingest <path>` | index files into memory |
| `jarvis serve [--port N]` | local web UI at 127.0.0.1:8765 |
| `jarvis voice` | hands-free voice conversation (needs `[voice]`) |
| `jarvis doctor` | show what this machine supports |
| `/ingest` `/memory` `/forget` `/tools` `/voice` `/listen` `/new` | in-chat commands |

### Configuration

Zero config is the default. To override, set environment variables or create
`~/.jarvis/config.toml`:

```toml
api_base = "http://127.0.0.1:11434/v1"   # any OpenAI-compatible endpoint
model = "qwen2.5:7b"
temperature = 0.7
assistant_name = "Jarvis"
```

Environment: `JARVIS_API_BASE`, `JARVIS_MODEL`, `JARVIS_API_KEY`,
`JARVIS_HOME` (data directory, default `~/.jarvis`).

### Optional extras

```bash
pip install -e ".[voice]"   # microphone input + spoken replies
pip install -e ".[rag]"     # sentence-transformers embeddings (better recall)
pip install -e ".[dev]"     # pytest
```

Without `[rag]`, Jarvis uses a built-in dependency-free hashed embedder
combined with keyword scoring — weaker than neural embeddings, but functional
everywhere, instantly.

## Design

```
you ──► CLI (rich) ──► Agent loop ──► LLM (auto-detected runtime)
                          │  ▲
                 tool call│  │tool result
                          ▼  │
                   ToolRegistry ("broker")
                   SAFE: auto-allow · CONFIRM: ask you first
                   writes jailed to workspace, reads to $HOME
                          │
        ┌─────────────────┴──────────────┐
        ▼                                ▼
   Memory (SQLite)                  OS / shell / web
   episodic log + semantic vectors
   hybrid search: cosine + keywords
```

Principles carried down from the full spec:

1. **Local-first** — the only network calls are to your own LLM runtime,
   unless you explicitly approve a `web_get`.
2. **The model proposes, the broker disposes** — every side effect passes a
   policy check; shell and network are confirm-per-call.
3. **Memory is engineered, not prompted** — episodic and semantic tiers with
   provenance, a score floor that admits "nothing relevant found" instead of
   hallucinating, and `/forget` for right-to-forget.
4. **Graceful degradation** — no runtime? `doctor` tells you what to install.
   No voice stack? Text works. No embedding model? Hash embedder. Any OS,
   any Python ≥ 3.10.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite covers the memory store, chunking/ingestion, the tool policy
(path jails, confirmation tiers), the agent loop (scripted fake LLM), and
runtime detection — all offline.
