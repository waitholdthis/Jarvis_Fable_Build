import sys
from pathlib import Path

import pytest

from jarvis.config import Config
from jarvis.embeddings import HashEmbedder
from jarvis.mcp import MCPError, MCPServer, register_mcp_tools
from jarvis.memory import Memory
from jarvis.tools import Tier, ToolRegistry

FAKE_SERVER = [sys.executable, str(Path(__file__).parent / "fake_mcp_server.py")]


@pytest.fixture
def server():
    srv = MCPServer("fake", FAKE_SERVER, timeout=10)
    srv.start()
    yield srv
    srv.close()


def test_handshake_lists_tools(server):
    names = [t["name"] for t in server.tools]
    assert names == ["echo", "fail"]


def test_call_tool_returns_text(server):
    assert server.call("echo", {"text": "hello"}) == "echo: hello"


def test_is_error_result_surfaces_as_error_string(server):
    result = server.call("fail", {})
    assert result.startswith("ERROR from fake.fail")
    assert "it broke" in result


def test_unknown_tool_raises(server):
    with pytest.raises(MCPError, match="unknown tool"):
        server.call("teleport", {})


def test_register_mcp_tools_into_registry(tmp_path):
    config = Config()
    config.home = tmp_path / "home"
    config.workspace = config.home / "workspace"
    config.ensure_dirs()
    memory = Memory(config.db_path, embedder=HashEmbedder())
    registry = ToolRegistry(config, memory)
    notes = []

    servers = register_mcp_tools(
        registry, {"fake": {"command": FAKE_SERVER}}, notify=notes.append
    )
    try:
        tool = registry.get("fake_echo")
        assert tool is not None
        assert tool.tier is Tier.CONFIRM  # external servers always confirm
        assert "text" in tool.params
        assert registry.run("fake_echo", {"text": "hi"}) == "echo: hi"
        assert any("2 tool(s) registered" in n for n in notes)
    finally:
        for srv in servers:
            srv.close()


def test_broken_server_is_skipped_not_fatal(tmp_path):
    config = Config()
    config.home = tmp_path / "home"
    config.workspace = config.home / "workspace"
    config.ensure_dirs()
    memory = Memory(config.db_path, embedder=HashEmbedder())
    registry = ToolRegistry(config, memory)
    notes = []

    servers = register_mcp_tools(
        registry,
        {
            "nocmd": {},  # missing command
            "badexe": {"command": ["/nonexistent/binary-xyz"]},
        },
        notify=notes.append,
    )
    assert servers == []
    assert any("missing command" in n for n in notes)
    assert any("failed to start" in n for n in notes)
