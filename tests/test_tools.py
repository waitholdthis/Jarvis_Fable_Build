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


def test_mission_control_persists_checkpoints(registry):
    created = registry.run("create_mission", {
        "title": "Launch", "objective": "Ship safely",
        "steps": "Run tests\nBuild release\nVerify deployment",
    })
    assert "3 checkpoints" in created
    updated = registry.run("update_mission", {
        "title": "Launch", "step": "1", "status": "complete", "note": "green",
    })
    assert "complete" in updated
    control = registry.run("mission_control", {})
    assert "1/3 complete" in control
    assert "[COMPLETE] Run tests — green" in control


def test_decision_journal_becomes_searchable_memory(registry):
    result = registry.run("record_decision", {
        "decision": "Use SQLite", "reasoning": "Local-first reliability",
        "alternatives": "Hosted database",
    })
    assert "decision recorded" in result
    found = registry.run("search_memory", {"query": "why SQLite"})
    assert "Local-first reliability" in found


def test_privacy_scan_detects_sensitive_patterns(registry):
    clean = registry.run("privacy_scan", {"text": "ordinary project notes"})
    assert "no common sensitive" in clean
    warning = registry.run("privacy_scan", {
        "text": "Email me at person@example.com and use api_key=supersecret123"
    })
    assert "PRIVACY WARNING" in warning
    assert "email address" in warning
    assert "secret/token assignment" in warning


def test_workspace_radar_and_situation_room(registry):
    registry.run("write_file", {"path": "active.txt", "content": "in progress"})
    radar = registry.run("workspace_radar", {"path": "."})
    assert "active.txt" in radar
    situation = registry.run("situation_room", {})
    assert "SITUATION ROOM" in situation
    assert "Memory:" in situation


def test_self_diagnostics_reports_every_core_subsystem(registry, monkeypatch):
    import jarvis.llm as llm_mod

    monkeypatch.setattr(llm_mod, "_probe", lambda base, timeout=1.5: ["test-model"])
    report = registry.run("run_diagnostics", {})
    assert "JARVIS DIAGNOSTIC REPORT — OPERATIONAL" in report
    for subsystem in (
        "Cognitive engine", "Memory database", "Workspace I/O", "Storage",
        "Tool broker", "Scheduler", "Voice systems", "MCP extensions",
    ):
        assert subsystem in report
    assert "[PASS] Memory database" in report
    assert not (registry.config.workspace / ".jarvis-diagnostic-probe").exists()


def test_parse_ddg_results():
    from jarvis.tools import parse_ddg_results

    page = """
    <div class="result">
      <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=abc">Example <b>Title</b></a>
      <a class="result__snippet" href="#">A snippet about the &amp; result.</a>
    </div>
    <div class="result">
      <a class="result__a" href="https://direct.example.org/">Direct Result</a>
      <a class="result__snippet" href="#">Second snippet.</a>
    </div>
    """
    results = parse_ddg_results(page)
    assert results[0] == (
        "Example Title",
        "https://example.com/page",
        "A snippet about the & result.",
    )
    assert results[1][1] == "https://direct.example.org/"
    assert parse_ddg_results("<html>no results</html>") == []


def test_web_search_tool_registered_confirm_tier(registry):
    from jarvis.tools import Tier

    tool = registry.get("web_search")
    assert tool is not None
    assert tool.tier is Tier.CONFIRM
