from pathlib import Path

import pytest

from jarvis.config import Config
from jarvis.embeddings import HashEmbedder
from jarvis.memory import Memory
from jarvis.tools import PolicyError, Tier, ToolRegistry


@pytest.fixture
def registry(tmp_path):
    config = Config()
    config.home = tmp_path / "jarvis-home"
    config.workspace = tmp_path / "jarvis-home" / "workspace"
    config.ensure_dirs()
    memory = Memory(config.db_path, embedder=HashEmbedder())
    return ToolRegistry(config, memory)


def test_builtin_tools_registered(registry):
    for name in ("current_time", "read_file", "write_file", "shell", "search_memory"):
        assert registry.get(name) is not None
    assert registry.get("shell").tier is Tier.CONFIRM
    assert registry.get("web_get").tier is Tier.CONFIRM


def test_write_then_read_roundtrip_in_workspace(registry):
    out = registry.run("write_file", {"path": "notes/todo.txt", "content": "buy milk"})
    assert "wrote" in out
    back = registry.run("read_file", {"path": "notes/todo.txt"})
    assert back == "buy milk"


def test_write_outside_workspace_denied(registry, tmp_path):
    outside = tmp_path / "evil.txt"
    result = registry.run("write_file", {"path": str(outside), "content": "x"})
    assert result.startswith("DENIED by policy")
    assert not outside.exists()


def test_write_path_traversal_denied(registry):
    result = registry.run(
        "write_file", {"path": "../../../../etc/hax", "content": "x"}
    )
    assert result.startswith("DENIED by policy")


def test_read_outside_home_denied(registry):
    with pytest.raises(PolicyError):
        registry.resolve_read_path("/etc/passwd")


def test_unknown_tool_and_bad_args(registry):
    assert registry.run("teleport", {}).startswith("ERROR: unknown tool")
    result = registry.run("read_file", {"path": "a", "mode": "rw"})
    assert "unknown argument" in result


def test_remember_and_search_memory_tools(registry):
    registry.run("remember_fact", {"fact": "user prefers dark roast coffee"})
    found = registry.run("search_memory", {"query": "what coffee does the user like"})
    assert "dark roast" in found


def test_current_time_runs(registry):
    out = registry.run("current_time", {})
    assert any(ch.isdigit() for ch in out)
