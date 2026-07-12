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
    timeout: float = 300.0

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


@dataclass
class AnthropicClient:
    api_key: str
    model: str
    api_base: str = "https://api.anthropic.com/v1"
    runtime_name: str = "Claude"
    timeout: float = 300.0

    def chat_stream(self, messages: list[dict], temperature: float = 0.7) -> Iterator[str]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conversation = [m for m in messages if m["role"] in ("user", "assistant")]
        payload = {"model": self.model, "max_tokens": 8192, "messages": conversation,
                   "temperature": temperature, "stream": True}
        if system:
            payload["system"] = system
        headers = {"content-type": "application/json", "x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01"}
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", f"{self.api_base}/messages", json=payload, headers=headers) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "content_block_delta":
                        text = (event.get("delta") or {}).get("text")
                        if text:
                            yield text

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        return "".join(self.chat_stream(messages, temperature))


@dataclass
class GeminiClient:
    api_key: str
    model: str
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    runtime_name: str = "Gemini"
    timeout: float = 300.0

    def chat_stream(self, messages: list[dict], temperature: float = 0.7) -> Iterator[str]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] in ("user", "assistant")]
        payload = {"contents": contents, "generationConfig": {"temperature": temperature}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.api_base}/models/{self.model}:streamGenerateContent?alt=sse"
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, json=payload,
                               headers={"content-type": "application/json", "x-goog-api-key": self.api_key}) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        chunk = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    candidates = chunk.get("candidates") or []
                    if not candidates:
                        continue
                    for part in ((candidates[0].get("content") or {}).get("parts") or []):
                        if part.get("text"):
                            yield part["text"]

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        return "".join(self.chat_stream(messages, temperature))


PROVIDER_PROFILES = {
    "anthropic": {"label": "Claude", "model": "claude-sonnet-4-5"},
    "claude": {"label": "Claude", "model": "claude-sonnet-4-5"},
    "gemini": {"label": "Gemini", "model": "gemini-2.5-flash"},
    "perplexity": {"label": "Perplexity", "model": "sonar-pro"},
    "openai": {"label": "OpenAI", "model": "gpt-5.4"},
    "codex": {"label": "Codex", "model": "gpt-5.3-codex"},
}


def cloud_client(config: Config, provider: str, model: str = ""):
    """Create a cloud client while keeping credentials environment-only."""
    provider = provider.lower().strip()
    profile = PROVIDER_PROFILES.get(provider)
    if profile is None:
        raise ValueError(f"unknown provider: {provider}")
    key = config.provider_key(provider)
    if not key:
        env = {"claude": "ANTHROPIC_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
               "gemini": "GEMINI_API_KEY", "perplexity": "PERPLEXITY_API_KEY",
               "openai": "OPENAI_API_KEY", "codex": "OPENAI_API_KEY"}[provider]
        raise NoRuntimeError(f"{profile['label']} is not configured. Set {env} before starting JARVIS.")
    selected = model or profile["model"]
    if provider in ("claude", "anthropic"):
        return AnthropicClient(key, selected)
    if provider == "gemini":
        return GeminiClient(key, selected)
    bases = {"perplexity": "https://api.perplexity.ai/v1",
             "openai": "https://api.openai.com/v1", "codex": "https://api.openai.com/v1"}
    return LLMClient(bases[provider], selected, key, profile["label"])


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
    if config.provider.lower() not in ("", "local", "ollama", "auto"):
        return cloud_client(config, config.provider, config.model)
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
