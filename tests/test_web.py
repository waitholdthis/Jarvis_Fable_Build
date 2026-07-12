import json
import threading
import time

import httpx
import pytest

from jarvis.agent import Agent
from jarvis.config import Config
from jarvis.embeddings import HashEmbedder
from jarvis.memory import Memory
from jarvis.tools import ToolRegistry
from jarvis.web import WebUI, make_server


class FakeLLM:
    runtime_name = "fake"
    model = "fake-model"
    api_base = "none"

    def __init__(self, responses):
        self.responses = list(responses)

    def chat_stream(self, messages, temperature=0.7):
        yield self.responses.pop(0)


def make_agent(tmp_path, responses):
    config = Config()
    config.home = tmp_path / "home"
    config.workspace = config.home / "workspace"
    config.ensure_dirs()
    memory = Memory(config.db_path, embedder=HashEmbedder())
    tools = ToolRegistry(config, memory)
    return Agent(config, FakeLLM(responses), memory, tools)


@pytest.fixture
def server(tmp_path):
    agent = make_agent(tmp_path, ["Hello from the web!"])
    srv = make_server(agent, host="127.0.0.1", port=0)
    srv.ui.keepalive_seconds = 0.2
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def read_events_until(base, kinds, timeout=5.0):
    """Read the SSE stream until one of the given kinds arrives."""
    seen = []
    deadline = time.time() + timeout
    with httpx.stream("GET", f"{base}/api/events", timeout=timeout) as response:
        for line in response.iter_lines():
            if time.time() > deadline:
                break
            if not line.startswith("data:"):
                continue
            event = json.loads(line[len("data:"):])
            seen.append(event)
            if event["kind"] in kinds:
                return seen
    return seen


def test_index_page_serves_html(server):
    _, base = server
    response = httpx.get(base + "/")
    assert response.status_code == 200
    assert "Jarvis" in response.text
    assert "EventSource" in response.text


def test_stats_endpoint(server):
    _, base = server
    response = httpx.get(base + "/api/stats")
    assert response.status_code == 200
    assert "semantic_chunks" in response.json()


def test_chat_streams_text_and_done(server):
    _, base = server
    result = {}

    def reader():
        result["events"] = read_events_until(base, kinds={"done"})

    thread = threading.Thread(target=reader)
    thread.start()
    time.sleep(0.3)  # let the SSE connection establish before submitting
    response = httpx.post(base + "/api/chat", json={"message": "hi"})
    assert response.status_code == 200
    thread.join(timeout=6)

    events = result["events"]
    kinds = [e["kind"] for e in events]
    assert "text" in kinds and kinds[-1] == "done"
    text = "".join(e["text"] for e in events if e["kind"] == "text")
    assert text == "Hello from the web!"


def test_empty_message_rejected(server):
    _, base = server
    response = httpx.post(base + "/api/chat", json={"message": "  "})
    assert response.status_code == 400


def test_unknown_route_404(server):
    _, base = server
    assert httpx.get(base + "/nope").status_code == 404
    assert httpx.post(base + "/api/nope", json={}).status_code == 404


def test_confirm_bridge_allow_and_timeout(tmp_path):
    agent = make_agent(tmp_path, [])
    ui = WebUI(agent)

    # Allow path: resolve from another thread while confirm() blocks.
    results = {}

    def confirmer():
        results["allow"] = ui.confirm("Run shell?")

    thread = threading.Thread(target=confirmer)
    thread.start()
    time.sleep(0.1)
    event = ui.events.get(timeout=2)
    assert event["kind"] == "confirm"
    assert ui.resolve_confirm(event["id"], True)
    thread.join(timeout=2)
    assert results["allow"] is True

    # Unknown id is rejected.
    assert not ui.resolve_confirm("bogus", True)


def test_busy_submission_conflicts(tmp_path):
    agent = make_agent(tmp_path, ["only one at a time"])
    ui = WebUI(agent)

    # Hold the busy lock as if a turn were in flight.
    assert ui._busy.acquire(blocking=False)
    try:
        assert ui.submit("second") is False
    finally:
        ui._busy.release()
