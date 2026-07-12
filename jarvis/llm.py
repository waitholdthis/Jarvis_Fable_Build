"""LLM access layer: one client, any OpenAI-compatible local runtime.

Ollama, LM Studio, llamafile, vLLM, llama.cpp's server, LocalAI, and Jan all
expose the OpenAI chat-completions API, so a single client covers every common
way people run models locally. `detect()` probes the well-known local ports so
the assistant works out of the box with zero configuration on any machine that
has one of these runtimes installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

import httpx

from .config import Config

# (name, base_url) pairs probed in order during auto-detection.
KNOWN_ENDPOINTS = [
    ("Ollama", "http://127.0.0.1:11434/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("llamafile/llama.cpp", "http://127.0.0.1:8080/v1"),
    ("vLLM", "http://127.0.0.1:8000/v1"),
    ("Jan", "http://127.0.0.1:1337/v1"),
]

# When a runtime hosts several models, prefer instruction-tuned generalists.
_PREFERRED_MODEL_HINTS = ("instruct", "qwen", "llama", "mistral", "gemma", "phi")


class NoRuntimeError(RuntimeError):
    """Raised when no local LLM runtime could be found."""


@dataclass
class LLMClient:
    api_base: str
    model: str
    api_key: str = ""
    runtime_name: str = "custom"
    timeout: float = 120.0

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat_stream(
        self, messages: list[dict], temperature: float = 0.7
    ) -> Iterator[str]:
        """Yield content deltas from a streaming chat completion."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST", url, json=payload, headers=self._headers()
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        return "".join(self.chat_stream(messages, temperature))


def _probe(base: str, timeout: float = 0.6) -> list[str]:
    """Return model ids served at an OpenAI-compatible base URL, or raise."""
    response = httpx.get(f"{base.rstrip('/')}/models", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]


def pick_model(models: list[str], requested: str = "") -> str:
    """Choose a model id from what the runtime offers."""
    if requested:
        for m in models:
            if m == requested or m.startswith(requested):
                return m
        return requested  # trust the user; runtime may lazy-load it (Ollama does)
    for hint in _PREFERRED_MODEL_HINTS:
        for m in models:
            if hint in m.lower():
                return m
    return models[0] if models else ""


def detect(config: Config) -> LLMClient:
    """Find a usable LLM runtime, honoring explicit configuration first."""
    candidates: list[tuple[str, str]] = []
    if config.api_base:
        candidates.append(("configured endpoint", config.api_base))
    candidates.extend(KNOWN_ENDPOINTS)

    errors: list[str] = []
    for name, base in candidates:
        try:
            models = _probe(base)
        except Exception as exc:  # connection refused, timeout, 4xx...
            errors.append(f"{name} ({base}): {type(exc).__name__}")
            continue
        model = pick_model(models, config.model)
        if not model:
            errors.append(f"{name} ({base}): reachable but serving no models")
            continue
        return LLMClient(
            api_base=base,
            model=model,
            api_key=config.api_key,
            runtime_name=name,
        )

    raise NoRuntimeError(
        "No local LLM runtime found. Install one of:\n"
        "  - Ollama (https://ollama.com) then run:  ollama pull qwen2.5:7b\n"
        "  - LM Studio (https://lmstudio.ai) and start its local server\n"
        "  - llamafile / llama.cpp server / vLLM on a local port\n"
        "or point Jarvis at any OpenAI-compatible endpoint:\n"
        "  export JARVIS_API_BASE=http://host:port/v1  JARVIS_MODEL=<model>\n\n"
        "Probes attempted:\n  " + "\n  ".join(errors)
    )
