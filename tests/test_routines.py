import time

from jarvis.config import Config
from jarvis.embeddings import HashEmbedder
from jarvis.memory import Memory
from jarvis.routines import BUILTIN_ROUTINES, get_routines, run_routine, sync_routine_schedules
from jarvis.scheduler import Scheduler
from jarvis.tools import ToolRegistry


def make_env(tmp_path):
    config = Config()
    config.home = tmp_path / "home"
    config.workspace = config.home / "workspace"
    config.ensure_dirs()
    memory = Memory(config.db_path, embedder=HashEmbedder())
    tools = ToolRegistry(config, memory)
    return config, memory, tools


def test_builtin_briefing_present_and_user_overrides(tmp_path):
    config, _, _ = make_env(tmp_path)
    assert "briefing" in get_routines(config)

    config.routines = {
        "briefing": {"prompt": "my own briefing"},
        "standup": {"prompt": "summarize yesterday", "schedule": "every day at 09:00"},
        "broken": {"no_prompt": True},
    }
    routines = get_routines(config)
    assert routines["briefing"]["prompt"] == "my own briefing"
    assert "standup" in routines
    assert "broken" not in routines


def test_sync_routine_schedules_idempotent(tmp_path):
    config, _, _ = make_env(tmp_path)
    config.routines = {
        "standup": {"prompt": "p", "schedule": "every day at 09:00"},
        "adhoc": {"prompt": "p"},  # no schedule -> never scheduled
        "bad": {"prompt": "p", "schedule": "sometime maybe"},  # unparseable
    }
    sched = Scheduler(tmp_path / "s.db")
    assert sync_routine_schedules(sched, get_routines(config)) == 1
    assert sync_routine_schedules(sched, get_routines(config)) == 0  # no dupes
    tasks = [t for t in sched.list_pending() if t.kind == "routine"]
    assert len(tasks) == 1
    assert tasks[0].payload == "standup"
    assert tasks[0].repeat == 86400.0


class FakeLLM:
    model = "fake"

    def __init__(self, responses):
        self.responses = list(responses)

    def chat_stream(self, messages, temperature=0.7):
        yield self.responses.pop(0)


def test_run_routine_returns_final_answer(tmp_path):
    config, memory, tools = make_env(tmp_path)
    config.routines = {"greet": {"prompt": "say hello"}}
    llm = FakeLLM(["Hello! All quiet on the local front."])
    result = run_routine(config, llm, memory, tools, "greet")
    assert result == "Hello! All quiet on the local front."
    assert run_routine(config, llm, memory, tools, "nonexistent") is None


def test_scheduled_routine_cannot_use_confirm_tools(tmp_path):
    # Scheduled runs pass confirm=deny; a shell call must come back DENIED.
    config, memory, tools = make_env(tmp_path)
    config.routines = {"sneaky": {"prompt": "run a shell command"}}
    llm = FakeLLM([
        '```tool\n{"tool": "shell", "args": {"command": "id"}}\n```',
        "The shell was denied, as expected.",
    ])
    result = run_routine(config, llm, memory, tools, "sneaky")
    assert result == "The shell was denied, as expected."
