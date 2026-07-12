"""Autonomous operation: self-directed goal pursuit, curiosity, and reflection.

JARVIS can operate entirely without user input — pursuing goals from its queue,
filling knowledge gaps proactively, and reflecting on its own performance to
improve over time. This module is the "unsupervised" side of the architecture.

Three interlocking engines
--------------------------
GoalQueue        SQLite-backed priority queue of tasks JARVIS pursues autonomously.
                 Goals survive restarts. JARVIS works on one goal per cycle,
                 calling real tools, updating progress, and marking completion.

CuriosityEngine  Detects knowledge gaps (topics where JARVIS hedged or failed),
                 researches them using available tools, and stores findings in
                 memory. Runs once per curiosity_interval_hours (default: 12h).

ReflectionEngine After each conversation or on a schedule, reviews recent
                 interactions, identifies what went well and what failed, and
                 distils lessons into the semantic memory store. Forms the
                 closed feedback loop for continuous self-improvement.

AutonomousWorker Background daemon thread that orchestrates all three engines
                 on a configurable duty cycle. Sends progress notes to the hub
                 so the user can observe autonomous work from their phone.

Safety design
-------------
- CONFIRM-tier tools are always skipped in autonomous mode (no user present).
- Each autonomous step is one tool call or one LLM generation — no runaway loops.
- Goals have a max_steps ceiling; once reached the goal is paused for review.
- The worker idles at 60s poll interval when the queue is empty.
- Pause/resume is atomic and thread-safe.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Goal model and queue
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    id: str
    title: str
    description: str
    priority: int               # 1 (low) to 5 (critical)
    status: str                 # 'pending' | 'in_progress' | 'done' | 'failed' | 'paused'
    created_at: float
    progress_notes: list[str] = field(default_factory=list)
    max_steps: int = 20         # autonomous steps before pausing for human review
    steps_taken: int = 0
    deadline: float | None = None

    def summary(self, idx: int = 0) -> str:
        notes_preview = ""
        if self.progress_notes:
            notes_preview = f"\n    last: {self.progress_notes[-1][:80]}"
        deadline_str = ""
        if self.deadline:
            remaining = self.deadline - time.time()
            if remaining > 0:
                deadline_str = f"  deadline in {int(remaining / 3600)}h"
        return (
            f"  [{idx or self.id}] P{self.priority} [{self.status}] {self.title}"
            f"  (steps: {self.steps_taken}/{self.max_steps})"
            + deadline_str
            + notes_preview
        )


class GoalQueue:
    """Persistent priority queue backed by SQLite."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS goals (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        description TEXT DEFAULT '',
        priority INTEGER DEFAULT 3,
        status TEXT DEFAULT 'pending',
        created_at REAL,
        updated_at REAL,
        deadline REAL,
        max_steps INTEGER DEFAULT 20,
        steps_taken INTEGER DEFAULT 0,
        notes_json TEXT DEFAULT '[]'
    )
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(self._SCHEMA)
        self._conn.commit()

    def add(
        self,
        title: str,
        description: str = "",
        priority: int = 3,
        deadline: float | None = None,
        max_steps: int = 20,
    ) -> str:
        goal_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._conn.execute(
                "INSERT INTO goals VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (goal_id, title, description, priority, "pending",
                 time.time(), time.time(), deadline, max_steps, 0, "[]"),
            )
            self._conn.commit()
        return goal_id

    def pop_next(self) -> Goal | None:
        """Return highest-priority pending goal and set it in_progress."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id,title,description,priority,status,created_at,"
                "notes_json,max_steps,steps_taken,deadline "
                "FROM goals WHERE status='pending' "
                "ORDER BY priority DESC, created_at ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            gid, title, desc, pri, status, created_at, notes_json, max_s, taken, deadline = row
            notes = json.loads(notes_json or "[]")
            self._conn.execute(
                "UPDATE goals SET status='in_progress', updated_at=? WHERE id=?",
                (time.time(), gid),
            )
            self._conn.commit()
            return Goal(
                id=gid, title=title, description=desc, priority=pri,
                status="in_progress", created_at=created_at,
                progress_notes=notes, max_steps=max_s,
                steps_taken=taken, deadline=deadline,
            )

    def update(
        self,
        goal_id: str,
        status: str,
        note: str = "",
        steps_taken: int | None = None,
    ) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT notes_json, steps_taken FROM goals WHERE id=?", (goal_id,)
            ).fetchone()
            if row is None:
                return
            notes = json.loads(row[0] or "[]")
            taken = row[1]
            if note:
                notes.append(f"[{time.strftime('%H:%M:%S')}] {note}")
                notes = notes[-50:]
            if steps_taken is not None:
                taken = steps_taken
            self._conn.execute(
                "UPDATE goals SET status=?, updated_at=?, notes_json=?, steps_taken=? "
                "WHERE id=?",
                (status, time.time(), json.dumps(notes), taken, goal_id),
            )
            self._conn.commit()

    def complete(self, goal_id: str, summary: str = "") -> None:
        self.update(goal_id, "done", f"Completed: {summary}")

    def fail(self, goal_id: str, reason: str = "") -> None:
        self.update(goal_id, "failed", reason)

    def pause(self, goal_id: str, reason: str = "") -> None:
        self.update(goal_id, "paused", reason or "paused for review")

    def list_all(self, status: str | None = None) -> list[Goal]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT id,title,description,priority,status,created_at,"
                    "notes_json,max_steps,steps_taken,deadline "
                    "FROM goals WHERE status=? ORDER BY priority DESC, created_at ASC",
                    (status,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id,title,description,priority,status,created_at,"
                    "notes_json,max_steps,steps_taken,deadline "
                    "FROM goals ORDER BY priority DESC, created_at ASC"
                ).fetchall()
        return [
            Goal(
                id=r[0], title=r[1], description=r[2], priority=r[3],
                status=r[4], created_at=r[5],
                progress_notes=json.loads(r[6] or "[]"),
                max_steps=r[7], steps_taken=r[8], deadline=r[9],
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Curiosity engine — proactive knowledge gap detection and filling
# ---------------------------------------------------------------------------

_GAP_SIGNALS = [
    "i don't know", "i'm not sure", "i cannot", "i can't confirm",
    "i'm uncertain", "i don't have information", "i lack", "unclear",
    "you'd need to check", "i'm unable to verify", "outside my knowledge",
]

_FILL_PROMPT = (
    "You identified this knowledge gap: {gap}\n\n"
    "Research it using available tools and then write a concise factual summary "
    "of what you learned. If no tools are applicable, reason from first principles "
    "and note any uncertainty. Output ONLY the summary — no preamble."
)

_GAP_DETECT_PROMPT = (
    "Review this conversation excerpt:\n{excerpt}\n\n"
    "List up to 5 specific factual gaps or uncertainties JARVIS expressed. "
    "Output ONLY a JSON array of short gap descriptions, e.g. "
    '["what X does in context Y", "how Z works"]. '
    "If none, output []."
)


class CuriosityEngine:
    """Detect knowledge gaps from recent interactions and fill them proactively."""

    def __init__(self, llm, memory, tools) -> None:
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self._pending_gaps: list[str] = []
        self._lock = threading.Lock()

    def observe_message(self, role: str, content: str) -> None:
        """Flag hedging language in assistant messages."""
        if role != "assistant":
            return
        lower = content.lower()
        if any(sig in lower for sig in _GAP_SIGNALS):
            # Rough extraction: take a 30-word window around the hedge
            words = content.split()
            for i, w in enumerate(words):
                if any(sig in w.lower() for sig in _GAP_SIGNALS):
                    snippet = " ".join(words[max(0, i - 5):i + 15])
                    with self._lock:
                        self._pending_gaps.append(snippet[:120])
                    break

    def detect_gaps_from_history(self, messages: list[dict]) -> list[str]:
        """Use LLM to extract structured gaps from a conversation excerpt."""
        excerpt = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in messages[-12:]
        )
        prompt = _GAP_DETECT_PROMPT.format(excerpt=excerpt)
        try:
            raw = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.2)
            gaps = json.loads(raw.strip())
            if isinstance(gaps, list):
                return [str(g) for g in gaps if isinstance(g, str)]
        except Exception:
            pass
        return []

    def fill_one(self, gap: str) -> str:
        """Research one knowledge gap and store the finding in memory."""
        prompt = _FILL_PROMPT.format(gap=gap)
        try:
            summary = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.3)
        except Exception as exc:
            return f"fill_one failed: {exc}"
        if self.memory and len(summary) > 20:
            self.memory.store(
                f"[CURIOSITY] {gap}\n\n{summary}",
                kind="fact",
                source="curiosity_engine",
            )
        return summary[:200]

    def fill_pending(self, max_fill: int = 3) -> list[str]:
        """Fill up to max_fill pending gaps, return filled summaries."""
        with self._lock:
            to_fill, self._pending_gaps = (
                self._pending_gaps[:max_fill], self._pending_gaps[max_fill:]
            )
        results = []
        for gap in to_fill:
            result = self.fill_one(gap)
            results.append(f"GAP: {gap[:60]}\n→ {result}")
        return results

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending_gaps)


# ---------------------------------------------------------------------------
# Reflection engine — learn from what happened
# ---------------------------------------------------------------------------

_REFLECT_PROMPT = (
    "You are JARVIS reviewing your own recent performance.\n\n"
    "Conversation excerpt:\n{excerpt}\n\n"
    "Reflect on this interaction and output a JSON object with these keys:\n"
    "  went_well: list of things you handled effectively\n"
    "  failed: list of things that went wrong or could be better\n"
    "  facts_learned: list of specific facts you learned from this exchange\n"
    "  skills_to_improve: list of capabilities you should develop\n"
    "  patterns: any repeating patterns you notice in your behavior\n\n"
    "Output ONLY valid JSON. Be specific and honest — vague answers are useless."
)


@dataclass
class ReflectionEntry:
    session_id: str
    went_well: list[str]
    failed: list[str]
    facts_learned: list[str]
    skills_to_improve: list[str]
    patterns: list[str]
    ts: float = field(default_factory=time.time)

    def as_text(self) -> str:
        sections = []
        if self.went_well:
            sections.append("Went well:\n" + "\n".join(f"  + {x}" for x in self.went_well))
        if self.failed:
            sections.append("Failed / could improve:\n" + "\n".join(f"  - {x}" for x in self.failed))
        if self.facts_learned:
            sections.append("Facts learned:\n" + "\n".join(f"  • {x}" for x in self.facts_learned))
        if self.skills_to_improve:
            sections.append("Skills to develop:\n" + "\n".join(f"  → {x}" for x in self.skills_to_improve))
        if self.patterns:
            sections.append("Patterns:\n" + "\n".join(f"  ~ {x}" for x in self.patterns))
        return "\n\n".join(sections) or "(nothing to reflect on)"


class ReflectionEngine:
    """Analyse a conversation and distil lessons into memory."""

    def __init__(self, llm, memory) -> None:
        self.llm = llm
        self.memory = memory
        self._entries: list[ReflectionEntry] = []

    def reflect(self, messages: list[dict], session_id: str = "") -> ReflectionEntry:
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        excerpt = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}"
            for m in messages[-16:]
        )
        prompt = _REFLECT_PROMPT.format(excerpt=excerpt)
        entry = ReflectionEntry(
            session_id=session_id,
            went_well=[], failed=[], facts_learned=[],
            skills_to_improve=[], patterns=[],
        )
        try:
            raw = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.2)
            data = json.loads(raw.strip())
            entry.went_well = data.get("went_well", [])
            entry.failed = data.get("failed", [])
            entry.facts_learned = data.get("facts_learned", [])
            entry.skills_to_improve = data.get("skills_to_improve", [])
            entry.patterns = data.get("patterns", [])
        except Exception:
            pass

        self._entries.append(entry)

        # Persist facts and skill gaps into semantic memory
        if self.memory:
            if entry.facts_learned:
                self.memory.store(
                    "[REFLECTION FACTS]\n" + "\n".join(f"• {f}" for f in entry.facts_learned),
                    kind="fact", source="reflection",
                )
            if entry.skills_to_improve:
                self.memory.store(
                    "[SKILL GAPS]\n" + "\n".join(f"• {s}" for s in entry.skills_to_improve),
                    kind="fact", source="reflection",
                )

        return entry

    def last_n(self, n: int = 5) -> list[ReflectionEntry]:
        return self._entries[-n:]


# ---------------------------------------------------------------------------
# Autonomous worker — the background loop
# ---------------------------------------------------------------------------

_GOAL_PROMPT = (
    "You are JARVIS operating in autonomous mode.\n\n"
    "Active goal:\n"
    "  Title: {title}\n"
    "  Description: {description}\n"
    "  Priority: {priority}/5\n"
    "  Steps taken so far: {steps_taken}/{max_steps}\n"
    "  Progress notes:\n{notes}\n\n"
    "Available tools (SAFE-tier only — no user confirmation possible):\n"
    "{tool_descriptions}\n\n"
    "Take ONE concrete action toward this goal. Either:\n"
    "  • Call a tool using the ```tool ... ``` block format\n"
    "  • Write 'GOAL_COMPLETE: <brief summary>' if the goal is achieved\n"
    "  • Write 'GOAL_BLOCKED: <reason>' if you genuinely cannot proceed\n"
    "  • Write 'GOAL_NOTE: <observation>' to log progress without a tool call\n\n"
    "Be decisive. Do not ask for clarification. Make a reasonable assumption and act."
)


class AutonomousWorker:
    """Background daemon that pursues goals, fills knowledge gaps, and reflects."""

    def __init__(
        self,
        goal_queue: GoalQueue,
        curiosity: CuriosityEngine,
        reflector: ReflectionEngine,
        llm,
        tools,
        memory=None,
        workflow_runner=None,
        poll_interval: float = 60.0,
        curiosity_interval_hours: float = 12.0,
        reflection_interval_hours: float = 6.0,
    ) -> None:
        self.goals = goal_queue
        self.curiosity = curiosity
        self.reflector = reflector
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.workflow_runner = workflow_runner
        self.poll_interval = poll_interval
        self.curiosity_interval = curiosity_interval_hours * 3600
        self.reflection_interval = reflection_interval_hours * 3600

        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._last_curiosity = 0.0
        self._last_reflection = 0.0
        self._session_messages: list[dict] = []  # reference to main agent messages

        self.on_progress: Callable[[str, str], None] | None = None  # (title, body)

    # ---- lifecycle ----------------------------------------------------------

    def start(self, agent_messages: list[dict] | None = None) -> None:
        if agent_messages is not None:
            self._session_messages = agent_messages
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def is_running(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    # ---- main loop ----------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            time.sleep(self.poll_interval)

            if self._paused:
                continue

            now = time.time()

            # 1. Work on the highest-priority pending goal
            try:
                goal = self.goals.pop_next()
                if goal is not None:
                    self._work_on_goal(goal)
            except Exception as exc:
                self._notify("JARVIS Autonomy Error", f"goal worker: {exc}")

            # 2. Fill knowledge gaps (curiosity cycle)
            if now - self._last_curiosity >= self.curiosity_interval:
                try:
                    self._curiosity_cycle()
                    self._last_curiosity = now
                except Exception as exc:
                    self._notify("JARVIS Curiosity Error", str(exc))

            # 3. Reflect on recent interactions
            if (
                now - self._last_reflection >= self.reflection_interval
                and self._session_messages
            ):
                try:
                    self._reflection_cycle()
                    self._last_reflection = now
                except Exception as exc:
                    self._notify("JARVIS Reflection Error", str(exc))

    # ---- goal worker --------------------------------------------------------

    def _safe_tool_descriptions(self) -> str:
        from .tools import Tier
        lines = []
        for tool in self.tools.tools.values():
            if tool.tier is Tier.SAFE:
                args = ", ".join(f"{k}" for k in tool.params)
                lines.append(f"  {tool.name}({args}): {tool.description}")
        return "\n".join(lines) or "(no SAFE tools available)"

    def _work_on_goal(self, goal: Goal) -> None:
        from .agent import parse_tool_call, strip_tool_block
        from .tools import Tier

        if goal.steps_taken >= goal.max_steps:
            self.goals.pause(
                goal.id,
                f"reached max_steps ({goal.max_steps}) — paused for human review",
            )
            self._notify(
                f"Goal Paused: {goal.title}",
                f"Reached max steps ({goal.max_steps}). Review needed.",
            )
            return

        notes_text = "\n".join(f"  {n}" for n in goal.progress_notes[-5:]) or "  (none yet)"
        prompt = _GOAL_PROMPT.format(
            title=goal.title,
            description=goal.description or "(no description)",
            priority=goal.priority,
            steps_taken=goal.steps_taken,
            max_steps=goal.max_steps,
            notes=notes_text,
            tool_descriptions=self._safe_tool_descriptions(),
        )

        try:
            response = self.llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.4
            )
        except Exception as exc:
            self.goals.update(goal.id, "pending", f"LLM error: {exc}")
            return

        new_steps = goal.steps_taken + 1

        # Parse control signals
        if "GOAL_COMPLETE:" in response:
            summary = response.split("GOAL_COMPLETE:", 1)[1].strip()[:300]
            self.goals.complete(goal.id, summary)
            self._notify(f"Goal Complete: {goal.title}", summary)
            return

        if "GOAL_BLOCKED:" in response:
            reason = response.split("GOAL_BLOCKED:", 1)[1].strip()[:300]
            self.goals.pause(goal.id, f"BLOCKED: {reason}")
            self._notify(f"Goal Blocked: {goal.title}", reason)
            return

        if "GOAL_NOTE:" in response:
            note = response.split("GOAL_NOTE:", 1)[1].strip()[:300]
            self.goals.update(goal.id, "pending", note, steps_taken=new_steps)
            return

        # Parse tool call
        call = parse_tool_call(response)
        if call:
            name, args = call
            tool = self.tools.get(name)
            if tool is None:
                self.goals.update(
                    goal.id, "pending", f"tried unknown tool '{name}'",
                    steps_taken=new_steps,
                )
                return
            if tool.tier is Tier.CONFIRM:
                self.goals.update(
                    goal.id, "pending",
                    f"skipped CONFIRM-tier tool '{name}' (no user present)",
                    steps_taken=new_steps,
                )
                return
            result = self.tools.run(name, args)
            note = f"[{name}] → {result[:180]}"
            self.goals.update(goal.id, "pending", note, steps_taken=new_steps)

            # Store significant tool results in memory
            if self.memory and len(result) > 30:
                self.memory.store(
                    f"[AUTONOMOUS] Goal '{goal.title}' step {new_steps}: {note}",
                    kind="fact", source="autonomy",
                )
        else:
            # Pure reasoning step
            note = strip_tool_block(response)[:200]
            self.goals.update(goal.id, "pending", f"reasoning: {note}", steps_taken=new_steps)

    # ---- curiosity cycle ----------------------------------------------------

    def _curiosity_cycle(self) -> None:
        if self.curiosity.pending_count() == 0 and self._session_messages:
            gaps = self.curiosity.detect_gaps_from_history(self._session_messages)
            for gap in gaps:
                with self.curiosity._lock:
                    self.curiosity._pending_gaps.append(gap)

        filled = self.curiosity.fill_pending(max_fill=3)
        if filled:
            self._notify(
                "JARVIS Curiosity",
                f"Filled {len(filled)} knowledge gap(s). "
                + filled[0].split("\n→ ")[1][:100] if "→" in filled[0] else filled[0][:100],
            )

    # ---- reflection cycle ---------------------------------------------------

    def _reflection_cycle(self) -> None:
        messages = list(self._session_messages)
        if len(messages) < 4:
            return
        session_id = str(uuid.uuid4())[:8]
        entry = self.reflector.reflect(messages, session_id)
        if entry.failed or entry.skills_to_improve:
            body = ""
            if entry.failed:
                body += "Issues: " + "; ".join(entry.failed[:2]) + "\n"
            if entry.skills_to_improve:
                body += "To improve: " + "; ".join(entry.skills_to_improve[:2])
            self._notify("JARVIS Reflection", body[:200])

    # ---- notification -------------------------------------------------------

    def _notify(self, title: str, body: str) -> None:
        if self.on_progress:
            try:
                self.on_progress(title, body)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_autonomy_tools(
    registry,
    goal_queue: GoalQueue,
    curiosity: CuriosityEngine,
    reflector: ReflectionEngine,
    worker: AutonomousWorker,
    agent_messages_ref: list,
) -> None:
    from .tools import Tier, Tool

    # ---- goals --------------------------------------------------------------

    def goal_add(
        title: str,
        description: str = "",
        priority: str = "3",
        max_steps: str = "20",
    ) -> str:
        """Add a goal to JARVIS's autonomous work queue."""
        try:
            pri = max(1, min(5, int(priority)))
        except ValueError:
            pri = 3
        try:
            ms = int(max_steps)
        except ValueError:
            ms = 20
        goal_id = goal_queue.add(title, description, pri, max_steps=ms)
        return (
            f"goal added: [{goal_id}] P{pri} {title}\n"
            f"JARVIS will work on this autonomously in the background."
        )

    def goal_list(status: str = "") -> str:
        """List goals in the autonomous queue."""
        goals = goal_queue.list_all(status or None)
        if not goals:
            return "no goals" + (f" with status '{status}'" if status else "")
        lines = [f"{len(goals)} goal(s):"]
        for i, g in enumerate(goals, 1):
            lines.append(g.summary(i))
        return "\n".join(lines)

    def goal_update(
        goal_id: str,
        status: str = "",
        note: str = "",
    ) -> str:
        """Update a goal's status or add a progress note."""
        valid_statuses = ("pending", "in_progress", "done", "failed", "paused", "")
        if status and status not in valid_statuses:
            return f"invalid status '{status}'. Valid: {', '.join(s for s in valid_statuses if s)}"
        goal_queue.update(goal_id, status or "pending", note)
        return f"goal {goal_id} updated."

    def goal_complete(goal_id: str, summary: str = "") -> str:
        """Mark a goal as complete."""
        goal_queue.complete(goal_id, summary)
        return f"goal {goal_id} marked complete."

    def goal_delete(goal_id: str) -> str:
        """Remove a goal from the queue entirely."""
        goal_queue.fail(goal_id, "deleted by user")
        return f"goal {goal_id} removed."

    # ---- worker control -----------------------------------------------------

    def autonomy_status() -> str:
        """Report autonomous worker status, pending curiosity gaps, and recent reflections."""
        running_str = "running" if worker.is_running() else "stopped"
        paused_str = " (paused)" if worker._paused else ""
        pending = goal_queue.list_all("pending")
        in_progress = goal_queue.list_all("in_progress")
        done = goal_queue.list_all("done")
        lines = [
            f"Autonomous worker: {running_str}{paused_str}",
            f"  goals: {len(pending)} pending, {len(in_progress)} in_progress, {len(done)} done",
            f"  curiosity gaps pending: {curiosity.pending_count()}",
            f"  reflections logged: {len(reflector._entries)}",
            f"  poll interval: {worker.poll_interval:.0f}s",
        ]
        if in_progress:
            lines.append("  In progress:")
            for g in in_progress[:3]:
                lines.append(f"    {g.summary()}")
        return "\n".join(lines)

    def autonomy_pause() -> str:
        worker.pause()
        return "autonomous worker paused — goals will not be processed until resumed"

    def autonomy_resume() -> str:
        worker.resume()
        return "autonomous worker resumed"

    # ---- curiosity ----------------------------------------------------------

    def curiosity_research(topic: str) -> str:
        """Manually trigger curiosity research on a specific topic."""
        result = curiosity.fill_one(topic)
        return f"curiosity research on '{topic}':\n{result}"

    def curiosity_detect() -> str:
        """Scan recent conversation for knowledge gaps and queue them."""
        msgs = list(agent_messages_ref)
        if not msgs:
            return "no conversation history to analyse"
        gaps = curiosity.detect_gaps_from_history(msgs)
        if not gaps:
            return "no knowledge gaps detected in recent messages"
        with curiosity._lock:
            curiosity._pending_gaps.extend(gaps)
        return f"detected {len(gaps)} gap(s): " + "; ".join(gaps[:3])

    # ---- reflection ---------------------------------------------------------

    def reflect_now() -> str:
        """Trigger an immediate reflection cycle on the current session."""
        msgs = list(agent_messages_ref)
        if len(msgs) < 4:
            return "conversation too short for meaningful reflection (need ≥ 4 messages)"
        entry = reflector.reflect(msgs)
        return entry.as_text()

    def reflect_history() -> str:
        """Show the last few reflection summaries."""
        entries = reflector.last_n(3)
        if not entries:
            return "no reflections logged yet"
        parts = []
        for e in reversed(entries):
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.ts))
            parts.append(f"=== Reflection [{e.session_id}] {ts} ===\n{e.as_text()}")
        return "\n\n".join(parts)

    # ---- register -----------------------------------------------------------

    registry.register(Tool(
        "goal_add",
        "Add a task to JARVIS's autonomous work queue. JARVIS will pursue it "
        "independently using available tools, one step at a time, in the background.",
        {
            "title":       "short task title",
            "description": "(optional) detailed description of what needs to be done",
            "priority":    "(optional) 1=low to 5=critical (default: 3)",
            "max_steps":   "(optional) max autonomous steps before pausing for review (default: 20)",
        },
        goal_add,
    ))
    registry.register(Tool(
        "goal_list",
        "List goals in the autonomous queue with their status and progress.",
        {"status": "(optional) filter by status: pending, in_progress, done, failed, paused"},
        goal_list,
    ))
    registry.register(Tool(
        "goal_update",
        "Update a goal's status or add a progress note.",
        {
            "goal_id": "goal ID from goal_list",
            "status":  "(optional) new status",
            "note":    "(optional) progress note to append",
        },
        goal_update,
    ))
    registry.register(Tool(
        "goal_complete",
        "Mark an autonomous goal as completed.",
        {"goal_id": "goal ID", "summary": "(optional) completion summary"},
        goal_complete,
    ))
    registry.register(Tool(
        "goal_delete",
        "Remove a goal from the autonomous queue.",
        {"goal_id": "goal ID"},
        goal_delete,
    ))
    registry.register(Tool(
        "autonomy_status",
        "Show the autonomous worker's current state: active goals, curiosity gaps, and reflections.",
        {},
        autonomy_status,
    ))
    registry.register(Tool(
        "autonomy_pause",
        "Pause the autonomous background worker. Goals will not be processed until resumed.",
        {},
        autonomy_pause,
    ))
    registry.register(Tool(
        "autonomy_resume",
        "Resume the autonomous background worker after it has been paused.",
        {},
        autonomy_resume,
    ))
    registry.register(Tool(
        "curiosity_research",
        "Manually trigger curiosity research on a topic JARVIS doesn't know well. "
        "Finds information using available tools and stores the result in memory.",
        {"topic": "the topic or question to research"},
        curiosity_research,
    ))
    registry.register(Tool(
        "curiosity_detect",
        "Scan recent conversation history for knowledge gaps and queue them for autonomous research.",
        {},
        curiosity_detect,
    ))
    registry.register(Tool(
        "reflect_now",
        "Trigger an immediate reflection on the current session: what went well, "
        "what failed, facts learned, and skills to improve. Stores findings in memory.",
        {},
        reflect_now,
    ))
    registry.register(Tool(
        "reflect_history",
        "Show recent reflection summaries from past sessions.",
        {},
        reflect_history,
    ))
