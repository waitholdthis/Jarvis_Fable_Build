import datetime as dt
import time

from jarvis.config import Config
from jarvis.embeddings import HashEmbedder
from jarvis.memory import Memory
from jarvis.scheduler import Scheduler, parse_when
from jarvis.tools import ToolRegistry, register_scheduler_tools

NOW = time.mktime(dt.datetime(2026, 7, 12, 10, 0, 0).timetuple())


# ---- parse_when -------------------------------------------------------------

def test_parse_in_relative():
    assert parse_when("in 20 minutes", NOW) == (NOW + 1200, 0.0)
    assert parse_when("in 2 hours", NOW) == (NOW + 7200, 0.0)
    assert parse_when("in 1 day", NOW) == (NOW + 86400, 0.0)
    assert parse_when("in 90s", NOW) == (NOW + 90, 0.0)


def test_parse_at_wall_clock():
    due, repeat = parse_when("at 18:30", NOW)
    assert repeat == 0.0
    assert dt.datetime.fromtimestamp(due).strftime("%H:%M") == "18:30"
    assert due > NOW
    # A time already past today rolls to tomorrow.
    due2, _ = parse_when("at 09:00", NOW)
    assert dt.datetime.fromtimestamp(due2).day == dt.datetime.fromtimestamp(NOW).day + 1


def test_parse_tomorrow():
    due, _ = parse_when("tomorrow at 11:00", NOW)
    when = dt.datetime.fromtimestamp(due)
    assert when.day == dt.datetime.fromtimestamp(NOW).day + 1
    assert when.hour == 11


def test_parse_every_interval():
    assert parse_when("every 30 minutes", NOW) == (NOW + 1800, 1800.0)
    assert parse_when("every hour", NOW) == (NOW + 3600, 3600.0)
    assert parse_when("every 2 days", NOW) == (NOW + 172800, 172800.0)


def test_parse_every_day_at():
    due, repeat = parse_when("every day at 08:00", NOW)
    assert repeat == 86400.0
    assert dt.datetime.fromtimestamp(due).hour == 8


def test_parse_rejects_garbage_and_tight_loops():
    assert parse_when("whenever", NOW) is None
    assert parse_when("at 25:99", NOW) is None
    assert parse_when("every 5 seconds", NOW) is None  # sub-30 s refused
    assert parse_when("", NOW) is None


# ---- Scheduler --------------------------------------------------------------

def test_add_list_cancel(tmp_path):
    sched = Scheduler(tmp_path / "s.db")
    task_id = sched.add(NOW + 60, "drink water")
    tasks = sched.list_pending()
    assert len(tasks) == 1 and tasks[0].payload == "drink water"
    assert sched.cancel(task_id)
    assert not sched.cancel(task_id)
    assert sched.list_pending() == []


def test_one_shot_fires_once_and_is_deleted(tmp_path):
    sched = Scheduler(tmp_path / "s.db")
    fired = []
    sched.on_due = fired.append
    sched.add(NOW + 10, "stretch")
    assert sched.check_once(NOW) == []          # not due yet
    assert len(sched.check_once(NOW + 11)) == 1  # due
    assert fired[0].payload == "stretch"
    assert sched.check_once(NOW + 999) == []     # gone after firing
    assert sched.list_pending() == []


def test_repeating_task_reschedules_and_skips_backlog(tmp_path):
    sched = Scheduler(tmp_path / "s.db")
    sched.add(NOW + 100, "hydrate", repeat=3600)
    # Machine "slept" for 5 hours: fires once, next due is in the future.
    fired = sched.check_once(NOW + 5 * 3600)
    assert len(fired) == 1
    remaining = sched.list_pending()
    assert len(remaining) == 1
    assert remaining[0].due > NOW + 5 * 3600


def test_persistence_across_reopen(tmp_path):
    sched = Scheduler(tmp_path / "s.db")
    sched.add(NOW + 60, "call mom")
    sched.close()
    reopened = Scheduler(tmp_path / "s.db")
    assert reopened.list_pending()[0].payload == "call mom"


def test_broken_callback_does_not_kill_check(tmp_path):
    sched = Scheduler(tmp_path / "s.db")
    sched.on_due = lambda task: 1 / 0
    sched.add(NOW - 1, "a")
    sched.add(NOW - 1, "b")
    assert len(sched.check_once(NOW)) == 2


# ---- scheduler tools through the registry -----------------------------------

def test_scheduler_tools(tmp_path):
    config = Config()
    config.home = tmp_path / "home"
    config.workspace = config.home / "workspace"
    config.ensure_dirs()
    memory = Memory(config.db_path, embedder=HashEmbedder())
    registry = ToolRegistry(config, memory)
    sched = Scheduler(tmp_path / "s.db")
    register_scheduler_tools(registry, sched)

    out = registry.run("schedule_task", {"when": "in 10 minutes", "message": "tea"})
    assert out.startswith("scheduled #")
    assert "tea" in registry.run("list_scheduled", {})
    task_id = out.split("#")[1].split(" ")[0]
    assert registry.run("cancel_scheduled", {"task_id": task_id}).startswith("cancelled")
    assert registry.run("list_scheduled", {}) == "nothing scheduled"
    assert "ERROR" in registry.run("schedule_task", {"when": "someday", "message": "x"})
