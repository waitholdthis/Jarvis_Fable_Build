"""Configuration: sane defaults, a TOML file, and environment overrides.

Precedence (highest wins): environment variables > ~/.jarvis/config.toml > defaults.
Nothing here requires third-party packages, so `jarvis --help` works on a bare
Python install even before dependencies are set up.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _default_home() -> Path:
    return Path(os.environ.get("JARVIS_HOME", Path.home() / ".jarvis"))


@dataclass
class Config:
    # LLM endpoint. Empty api_base means "auto-detect a local runtime".
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.7
    max_context_chars: int = 24_000

    # Agent loop
    max_tool_iterations: int = 6

    # Persona
    assistant_name: str = "Jarvis"
    persona: str = (
        "You are {name}, a capable, concise personal assistant running fully "
        "on the user's own computer. You are direct, practical, and a little "
        "dry. You never invent facts about the user's files or system: when "
        "you need real data, you use a tool."
    )

    # Storage
    home: Path = field(default_factory=_default_home)
    workspace: Path = field(default_factory=lambda: _default_home() / "workspace")

    # Voice (only used when the optional [voice] extras are installed)
    voice_enabled: bool = False
    whisper_model: str = "base"

    @property
    def db_path(self) -> Path:
        return self.home / "memory.db"

    def ensure_dirs(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)

    @property
    def system_name(self) -> str:
        return self.assistant_name

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        path = cfg.home / "config.toml"
        if path.is_file():
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            for key, value in data.items():
                if hasattr(cfg, key):
                    if key in ("home", "workspace"):
                        value = Path(value).expanduser()
                    setattr(cfg, key, value)

        env_map = {
            "JARVIS_API_BASE": "api_base",
            "OPENAI_BASE_URL": "api_base",
            "JARVIS_API_KEY": "api_key",
            "OPENAI_API_KEY": "api_key",
            "JARVIS_MODEL": "model",
        }
        for env, attr in env_map.items():
            value = os.environ.get(env)
            if value and not getattr(cfg, attr):
                setattr(cfg, attr, value)
        if os.environ.get("JARVIS_API_BASE"):
            cfg.api_base = os.environ["JARVIS_API_BASE"]
        if os.environ.get("JARVIS_MODEL"):
            cfg.model = os.environ["JARVIS_MODEL"]
        return cfg
