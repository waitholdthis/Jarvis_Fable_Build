import json

from jarvis.agent import Agent, parse_tool_call, strip_tool_block
from jarvis.config import Config
from jarvis.embeddings import HashEmbedder
from jarvis.memory import Memory
from jarvis.tools import ToolRegistry


def test_parse_tool_call_valid():
    text = 'Let me check.\n```tool\n{"tool": "read_file", "args": {"path": "a.txt"}}\n```'
    assert parse_tool_call(text) == ("read_file", {"path": "a.txt"})


def test_parse_tool_call_absent_or_malformed():
    assert parse_tool_call("just a normal answer") is None
    assert parse_tool_call("```tool\nnot json\n```") is None
    assert parse_tool_call('```tool\n{"args": {}}\n```') is None
    assert parse_tool_call('```tool\n{"tool": 5, "args": {}}\n```') is None


def test_strip_tool_block():
    text = 'Checking.\n```tool\n{"tool": "x", "args": {}}\n```'
    assert strip_tool_block(text) == "Checking."


class FakeLLM:
    """Scripted model: yields each canned response in order."""

    runtime_name = "fake"
    model = "fake-model"
    api_base = "none"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_stream(self, messages, temperature=0.7):
        self.calls.append([dict(m) for m in messages])
        yield self.responses.pop(0)


def make_agent(tmp_path, responses, confirm=lambda p: True):
    config = Config()
    config.home = tmp_path / "home"
    config.workspace = config.home / "workspace"
    config.ensure_dirs()
    memory = Memory(config.db_path, embedder=HashEmbedder())
    tools = ToolRegistry(config, memory)
    llm = FakeLLM(responses)
    return Agent(config, llm, memory, tools, confirm=confirm), memory, llm


def test_plain_answer_no_tools(tmp_path):
    agent, memory, _ = make_agent(tmp_path, ["Hello! How can I help?"])
    events = list(agent.run("hi"))
    kinds = [e.kind for e in events]
    assert kinds == ["text", "done"]
    assert events[-1].text == "Hello! How can I help?"
    # Both sides of the exchange land in episodic memory.
    assert memory.recent() == [("user", "hi"), ("assistant", "Hello! How can I help?")]


def test_tool_loop_executes_and_feeds_result_back(tmp_path):
    call = json.dumps({"tool": "write_file", "args": {"path": "x.txt", "content": "42"}})
    agent, _, llm = make_agent(
        tmp_path,
        [f"On it.\n```tool\n{call}\n```", "Done — saved 42 to x.txt."],
    )
    events = list(agent.run("save the number 42 to x.txt"))
    kinds = [e.kind for e in events]
    assert kinds == ["text", "tool_call", "tool_result", "text", "done"]
    assert "wrote" in events[2].text
    # Second LLM call must include the tool result.
    assert any("TOOL RESULT" in m["content"] for m in llm.calls[1])
    assert (agent.config.workspace / "x.txt").read_text() == "42"


def test_confirm_tier_denied_by_user(tmp_path):
    call = json.dumps({"tool": "shell", "args": {"command": "rm -rf /"}})
    agent, _, _ = make_agent(
        tmp_path,
        [f"```tool\n{call}\n```", "Understood, I won't run that."],
        confirm=lambda prompt: False,
    )
    events = list(agent.run("wipe my disk"))
    result_event = next(e for e in events if e.kind == "tool_result")
    assert result_event.text.startswith("DENIED")


def test_iteration_ceiling(tmp_path):
    call = '```tool\n{"tool": "current_time", "args": {}}\n```'
    responses = [call] * 10
    agent, _, _ = make_agent(tmp_path, responses)
    agent.config.max_tool_iterations = 3
    events = list(agent.run("loop forever"))
    assert sum(1 for e in events if e.kind == "tool_call") == 3
    assert "limit" in events[-1].text


def test_memory_context_reaches_system_prompt(tmp_path):
    agent, memory, llm = make_agent(tmp_path, ["Your wifi password is hunter2."])
    memory.remember("wifi password is hunter2", source="notes.md")
    list(agent.run("what is my wifi password?"))
    system = llm.calls[0][0]
    assert system["role"] == "system"
    assert "hunter2" in system["content"]
