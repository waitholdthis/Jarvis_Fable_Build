"""Persistent task scheduler: reminders and scheduled routines.

Tasks live in a SQLite table next to memory, so they survive restarts. A
background thread checks for due tasks and hands them to a frontend-supplied
callback (print + speak in the CLI, SSE event in the web UI, plus a native
desktop notification everywhere). Natural-language times are parsed by
parse_when(), a pure function:

    "in 20 minutes"          one-shot, +1200 s
    "at 18:30"               next occurrence of 18:30
    "tomorrow at 9:00"       explicit tomorrow
    "every 30 minutes"       repeating interval
    "every day at 08:00"     repeating daily at a wall-clock time
"""

from __future__ import annotations

import datetime as _dt
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_UNITS = {
    "s": 1, "sec": 1, "second": 1,
    "m": 60, "min": 60, "minute": 60,
    "h": 3600, "hr": 3600, "hour": 3600,
    "d": 86400, "day": 86400,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scheduled (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created REAL NOT NULL,
    due REAL NOT NULL,
    repeat REAL NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'reminder',
    payload TEXT NOT NULL
);
"""


def _next_wall_clock(hour: int, minute: int, now: float, force_tomorrow: bool = False) -> float:
    base = _dt.datetime.fromtimestamp(now)
    candidate = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if force_tomorrow or candidate.timestamp() <= now:
        candidate += _dt.timedelta(days=1)
    return candidate.timestamp()


def parse_when(text: str, now: float | None = None) -> tuple[float, float] | None:
    """Parse a natural-language time. Returns (due_ts, repeat_seconds) or None."""
    t = text.lower().strip()
    now = time.time() if now is None else now

    # "in 20 minutes", "in 2h", "in 90s", "in 1 day"
    m = re.match(r"^in\s+(\d+(?:\.\d+)?)\s*([a-z]+)$", t)
    if m:
        unit = m.group(2)
        if unit not in _UNITS:  # de-pluralize, but "s" alone means seconds
            unit = unit.rstrip("s")
        if unit in _UNITS:
            return now + float(m.group(1)) * _UNITS[unit], 0.0

    # "every day at 08:00" / "every morning at 8:00"
    m = re.match(r"^every\s+(?:day|morning|evening|night)\s+at\s+(\d{1,2}):(\d{2})$", t)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour < 24 and minute < 60:
            return _next_wall_clock(hour, minute, now), 86400.0

    # "every 30 minutes", "every hour", "every 2 days"
    m = re.match(r"^every\s+(\d+(?:\.\d+)?)?\s*([a-z]+)$", t)
    if m:
        unit = m.group(2).rstrip("s")
        if unit in _UNITS:
            interval = float(m.group(1) or 1) * _UNITS[unit]
            if interval >= 30:  # refuse sub-30 s loops
                return now + interval, interval

    # "tomorrow at 9:00" / "at 18:30"
    m = re.match(r"^(tomorrow\s+)?at\s+(\d{1,2}):(\d{2})$", t)
    if m:
        hour, minute = int(m.group(2)), int(m.group(3))
        if hour < 24 and minute < 60:
            return _next_wall_clock(hour, minute, now, force_tomorrow=bool(m.group(1))), 0.0

    return None


@dataclass
class ScheduledTask:
    id: int
    due: float
    repeat: float
    kind: str  # 'reminder' | 'routine'
    payload: str

    def describe(self) -> str:
        due_str = _dt.datetime.fromtimestamp(self.due).strftime("%Y-%m-%d %H:%M")
        repeat = ""
        if self.repeat:
            if self.repeat % 86400 == 0:
                repeat = f", repeats every {int(self.repeat // 86400)} day(s)"
            elif self.repeat % 3600 == 0:
                repeat = f", repeats every {int(self.repeat // 3600)} hour(s)"
            else:
                repeat = f", repeats every {int(self.repeat // 60)} minute(s)"
        return f"#{self.id} [{self.kind}] due {due_str}{repeat}: {self.payload}"


class Scheduler:
    """SQLite-backed scheduler with a polling thread."""

    def __init__(self, db_path: Path | str, poll_seconds: float = 5.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
        self.poll_seconds = poll_seconds
        self.on_due: Callable[[ScheduledTask], None] = lambda task: None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- CRUD ---------------------------------------------------------------

    def add(self, due: float, payload: str, kind: str = "reminder", repeat: float = 0.0) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO scheduled (created, due, repeat, kind, payload)"
                " VALUES (?, ?, ?, ?, ?)",
                (time.time(), due, repeat, kind, payload),
            )
            self._conn.commit()
            return cur.lastrowid

    def list_pending(self) -> list[ScheduledTask]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, due, repeat, kind, payload FROM scheduled ORDER BY due"
            ).fetchall()
        return [ScheduledTask(*row) for row in rows]

    def cancel(self, task_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM scheduled WHERE id = ?", (task_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ---- execution ------------------------------------------------------------

    def check_once(self, now: float | None = None) -> list[ScheduledTask]:
        """Fire every due task exactly once. Returns the tasks fired."""
        now = time.time() if now is None else now
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, due, repeat, kind, payload FROM scheduled WHERE due <= ?",
                (now,),
            ).fetchall()
        fired = []
        for row in rows:
            task = ScheduledTask(*row)
            with self._lock:
                if task.repeat > 0:
                    # Skip missed occurrences (machine was asleep): schedule
                    # the next one in the future, not a backlog of catch-ups.
                    next_due = task.due + task.repeat
                    while next_due <= now:
                        next_due += task.repeat
                    self._conn.execute(
                        "UPDATE scheduled SET due = ? WHERE id = ?", (next_due, task.id)
                    )
                else:
                    self._conn.execute("DELETE FROM scheduled WHERE id = ?", (task.id,))
                self._conn.commit()
            try:
                self.on_due(task)
            except Exception:
                pass  # a broken callback must not kill the scheduler
            fired.append(task)
        return fired

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()

        def loop():
            while not self._stop.wait(self.poll_seconds):
                self.check_once()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_seconds + 1)
            self._thread = None

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._conn.close()
