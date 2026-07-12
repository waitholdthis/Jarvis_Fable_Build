"""Routines: named, optionally scheduled automations.

A routine is a prompt the agent runs on demand (`jarvis routine morning`,
`/routine morning`) or on a schedule declared in config:

    [routines.morning]
    prompt = "Give me a rundown of my day: schedule, reminders, and news from my notes."
    schedule = "every day at 08:00"

A built-in `briefing` routine exists out of the box. Scheduled routines run
with confirmations denied (nobody is present to approve), so they can only
use SAFE-tier tools — reads, memory, scheduling — never shell or network.
"""

from __future__ import annotations

from .scheduler import Scheduler, parse_when

BUILTIN_ROUTINES: dict[str, dict] = {
    "briefing": {
        "prompt": (
            "Prepare a spoken-style briefing for the user. Use tools to get "
            "real data: current_time for the date and time, list_scheduled "
            "for upcoming reminders, system_info if anything looks relevant, "
            "and search_memory for facts related to today's work. Then "
            "summarize in under 120 words: the time and date, anything "
            "scheduled soon, and one or two useful remembered facts. Do not "
            "invent events that are not in the data."
        ),
    },
}


def get_routines(config) -> dict[str, dict]:
    """Built-ins merged with (and overridable by) the user's config."""
    merged = dict(BUILTIN_ROUTINES)
    for name, spec in (getattr(config, "routines", None) or {}).items():
        if isinstance(spec, dict) and isinstance(spec.get("prompt"), str):
            merged[name] = spec
    return merged


def sync_routine_schedules(scheduler: Scheduler, routines: dict[str, dict]) -> int:
    """Ensure every routine with a `schedule` has exactly one pending task.

    Idempotent across restarts: existing tasks are matched by payload (the
    routine name). Returns the number of newly scheduled routines.
    """
    existing = {
        task.payload for task in scheduler.list_pending() if task.kind == "routine"
    }
    added = 0
    for name, spec in routines.items():
        expression = spec.get("schedule")
        if not expression or name in existing:
            continue
        parsed = parse_when(str(expression))
        if parsed is None:
            continue
        due, repeat = parsed
        scheduler.add(due, payload=name, kind="routine", repeat=repeat)
        added += 1
    return added


def run_routine(config, llm, memory, tools, name: str) -> str | None:
    """Execute one routine with a fresh Agent; returns its final answer.

    A fresh Agent keeps scheduled/background runs from clobbering the
    in-flight conversation history of an interactive session.
    """
    from .agent import Agent

    spec = get_routines(config).get(name)
    if spec is None:
        return None
    agent = Agent(config, llm, memory, tools, confirm=lambda prompt: False)
    final = ""
    for event in agent.run(spec["prompt"]):
        if event.kind == "done":
            final = event.text
    return final
