"""Minimal MCP (Model Context Protocol) client, stdio transport, stdlib only.

MCP is the emerging standard for exposing tools to AI assistants; hundreds of
servers exist (filesystem, git, browsers, home automation, ...). This module
implements just enough of the protocol for Jarvis to use them: spawn the
server process, speak JSON-RPC 2.0 over newline-delimited stdio, list its
tools, and call them.

Servers are declared in ~/.jarvis/config.toml:

    [mcp_servers.time]
    command = ["uvx", "mcp-server-time"]

    [mcp_servers.files]
    command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]

Every MCP tool is registered at the CONFIRM tier: external servers run
outside Jarvis's policy jail, so the user approves each call.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time

from . import __version__
from .tools import Tier, Tool, ToolRegistry

_PROTOCOL_VERSION = "2024-11-05"


class MCPError(RuntimeError):
    pass


class MCPServer:
    """One spawned MCP server process and its JSON-RPC session."""

    def __init__(self, name: str, command: list[str], timeout: float = 20.0):
        self.name = name
        self.command = command
        self.timeout = timeout
        self.tools: list[dict] = []
        self._id = 0
        self._lock = threading.Lock()
        self._responses: queue.Queue = queue.Queue()
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in message:  # responses only; notifications are ignored
                self._responses.put(message)

    def _send(self, message: dict) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise MCPError(f"{self.name}: server process is gone ({exc})")

    def _request(self, method: str, params: dict | None = None) -> dict:
        with self._lock:
            self._id += 1
            request_id = self._id
            message: dict = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params is not None:
                message["params"] = params
            self._send(message)

            deadline = time.time() + self.timeout
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise MCPError(f"{self.name}: timed out waiting for {method}")
                try:
                    response = self._responses.get(timeout=remaining)
                except queue.Empty:
                    raise MCPError(f"{self.name}: timed out waiting for {method}")
                if response.get("id") != request_id:
                    continue  # stale response from an earlier timeout
                if "error" in response:
                    err = response["error"]
                    raise MCPError(f"{self.name}: {err.get('message', err)}")
                return response.get("result") or {}

    def start(self) -> list[dict]:
        """Run the MCP handshake and fetch the tool list."""
        self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "jarvis", "version": __version__},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.tools = self._request("tools/list").get("tools", [])
        return self.tools

    def call(self, tool_name: str, arguments: dict) -> str:
        result = self._request(
            "tools/call", {"name": tool_name, "arguments": arguments}
        )
        parts = [
            c.get("text", "")
            for c in result.get("content", [])
            if c.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p) or json.dumps(result)
        if result.get("isError"):
            return f"ERROR from {self.name}.{tool_name}: {text}"
        return text

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


def _make_caller(server: MCPServer, tool_name: str):
    def caller(**kwargs) -> str:
        return server.call(tool_name, kwargs)

    return caller


def register_mcp_tools(
    registry: ToolRegistry, mcp_servers: dict, notify=None
) -> list[MCPServer]:
    """Spawn configured MCP servers and register their tools.

    Returns the live servers so the caller can close() them on exit. A server
    that fails to start is reported and skipped — a broken entry in the
    config must never take the assistant down.
    """
    notify = notify or (lambda text: None)
    started: list[MCPServer] = []
    for name, spec in (mcp_servers or {}).items():
        command = spec.get("command") if isinstance(spec, dict) else None
        if not isinstance(command, list) or not command:
            notify(f"mcp server '{name}': missing command = [...] in config; skipped")
            continue
        try:
            server = MCPServer(name, [str(c) for c in command])
            tools = server.start()
        except Exception as exc:
            notify(f"mcp server '{name}' failed to start: {exc}")
            continue

        for tool in tools:
            properties = (tool.get("inputSchema") or {}).get("properties") or {}
            params = {
                key: str(value.get("description") or value.get("type") or "value")
                for key, value in properties.items()
            }
            registry.register(
                Tool(
                    name=f"{name}_{tool['name']}",
                    description=f"[{name} MCP] {tool.get('description', '')}".strip(),
                    params=params,
                    func=_make_caller(server, tool["name"]),
                    tier=Tier.CONFIRM,
                )
            )
        started.append(server)
        notify(f"mcp server '{name}': {len(tools)} tool(s) registered")
    return started
