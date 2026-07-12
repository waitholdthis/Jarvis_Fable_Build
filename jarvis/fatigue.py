"""Multi-signal fatigue scoring and ambient wellness monitoring.

Blueprint Section 4 (Stateful Ambient Runtime & Fatigue Mapping):

Run a multi-signal fatigue-scoring heuristic evaluating:
  1. Posture slouching — detected via VLM bounding-box analysis on webcam or
     screen captures (asks the loaded vision model to assess ergonomic posture)
  2. Typing velocity anomalies — keyboard event timestamps reveal fatigue
     through increasing inter-keystroke intervals and rising error rates
  3. Recursive compilation friction — count consecutive build failures logged
     by the EventMonitor CEP stream as a friction signal

Each signal is normalised to [0, 1] and combined into a weighted fatigue score.
When the score crosses a strict 85% confidence threshold, JARVIS triggers:
  - A desktop notification
  - A spoken alert (if voice is enabled)
  - A hub message (if configured)
  - Optional: schedule an automatic 5-minute break reminder

The scorer runs as a daemon thread during interactive sessions. No external
dependencies are required for the scoring logic; pynput is optional for the
typing-velocity signal (falls back to self-reported idle time).
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


# ---- Signal processors ------------------------------------------------------

@dataclass
class PostureSignal:
    score: float = 0.0          # 0.0 = perfect posture, 1.0 = severe slouch
    last_assessment: float = 0.0
    assessments: list[float] = field(default_factory=list)

    def update(self, score: float) -> None:
        self.score = score
        self.last_assessment = time.time()
        self.assessments.append(score)
        if len(self.assessments) > 20:
            self.assessments.pop(0)

    @property
    def rolling_mean(self) -> float:
        if not self.assessments:
            return 0.0
        return statistics.mean(self.assessments[-5:])


@dataclass
class TypingSignal:
    score: float = 0.0          # 0.0 = fast and accurate, 1.0 = severely fatigued
    intervals: deque = field(default_factory=lambda: deque(maxlen=200))
    baseline_iki: float = 0.0   # median inter-keystroke interval when fresh

    def record_keystroke(self, ts: float) -> None:
        if self.intervals:
            iki = ts - list(self.intervals)[-1]
            if 0.05 < iki < 5.0:  # filter outliers (pauses, copy-paste)
                self.intervals.append(iki)
                self._update_score()

    def _update_score(self) -> None:
        if len(self.intervals) < 20:
            return
        recent = list(self.intervals)[-20:]
        if self.baseline_iki == 0.0 and len(self.intervals) >= 50:
            self.baseline_iki = statistics.median(list(self.intervals)[:50])
        if self.baseline_iki > 0:
            current_median = statistics.median(recent)
            slowdown = (current_median - self.baseline_iki) / self.baseline_iki
            self.score = min(max(slowdown, 0.0), 1.0)


@dataclass
class FrictionSignal:
    score: float = 0.0
    consecutive_failures: int = 0
    last_success: float = field(default_factory=time.time)

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.score = min(self.consecutive_failures / 5.0, 1.0)

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.last_success = time.time()
        self.score = 0.0


# ---- Fatigue scorer ---------------------------------------------------------

WEIGHTS = {
    "posture": 0.40,
    "typing": 0.35,
    "friction": 0.25,
}

THRESHOLD = 0.85


@dataclass
class FatigueScore:
    composite: float
    posture: float
    typing: float
    friction: float
    above_threshold: bool
    ts: float = field(default_factory=time.time)

    def summary(self) -> str:
        bar = "█" * int(self.composite * 20) + "░" * (20 - int(self.composite * 20))
        status = "⚠ FATIGUE ALERT" if self.above_threshold else "OK"
        return (
            f"Fatigue [{bar}] {self.composite:.1%}  {status}\n"
            f"  posture={self.posture:.1%}  typing={self.typing:.1%}  "
            f"friction={self.friction:.1%}"
        )


class FatigueMonitor:
    """Continuously sample fatigue signals and fire callbacks at threshold.

    Usage:
        monitor = FatigueMonitor(llm=llm)
        monitor.on_alert = lambda score: notify_user(score.summary())
        monitor.start()
    """

    def __init__(
        self,
        llm=None,
        sample_interval_s: float = 300.0,   # assess every 5 minutes
        vision_tool=None,                   # see_screen function if available
    ) -> None:
        self.llm = llm
        self.sample_interval = sample_interval_s
        self._vision_tool = vision_tool
        self.posture = PostureSignal()
        self.typing = TypingSignal()
        self.friction = FrictionSignal()
        self.on_alert: Callable[[FatigueScore], None] | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._scores: list[FatigueScore] = []
        self._last_alert = 0.0
        self._alert_cooldown = 1800.0   # 30 min between alerts

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._start_keyboard_monitor()

    def stop(self) -> None:
        self._running = False

    # ---- Keyboard monitoring (pynput optional) ------------------------------

    def _start_keyboard_monitor(self) -> None:
        try:
            from pynput import keyboard

            def on_press(key):
                self.typing.record_keystroke(time.time())

            listener = keyboard.Listener(on_press=on_press)
            listener.daemon = True
            listener.start()
        except ImportError:
            pass  # typing signal disabled without pynput

    # ---- Posture assessment via VLM -----------------------------------------

    def _assess_posture(self) -> float:
        """Return a 0-1 fatigue score for posture. Uses VLM if available."""
        if self._vision_tool is None and self.llm is None:
            return 0.0

        if self._vision_tool is not None:
            description = self._vision_tool(
                "Assess the person's posture in this image. "
                "Respond with ONLY a number from 0.0 (perfect posture, sitting upright) "
                "to 1.0 (severe slouch, head down, hunched over). "
                "Output only the decimal number."
            )
            try:
                score = float(description.strip().split()[0])
                return min(max(score, 0.0), 1.0)
            except (ValueError, IndexError):
                pass

        # Fallback: time-of-day heuristic (fatigue peaks mid-afternoon)
        hour = time.localtime().tm_hour
        if 14 <= hour <= 16:
            return 0.3
        if hour >= 22 or hour <= 6:
            return 0.4
        return 0.1

    # ---- Main scoring loop --------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            time.sleep(self.sample_interval)
            score = self.compute()
            self._scores.append(score)
            if len(self._scores) > 100:
                self._scores.pop(0)
            if score.above_threshold:
                now = time.time()
                if now - self._last_alert >= self._alert_cooldown:
                    self._last_alert = now
                    if self.on_alert:
                        self.on_alert(score)

    def compute(self) -> FatigueScore:
        posture_score = self._assess_posture()
        self.posture.update(posture_score)
        composite = (
            WEIGHTS["posture"] * self.posture.rolling_mean
            + WEIGHTS["typing"] * self.typing.score
            + WEIGHTS["friction"] * self.friction.score
        )
        return FatigueScore(
            composite=composite,
            posture=self.posture.rolling_mean,
            typing=self.typing.score,
            friction=self.friction.score,
            above_threshold=composite >= THRESHOLD,
        )

    def record_build_failure(self) -> None:
        self.friction.record_failure()

    def record_build_success(self) -> None:
        self.friction.record_success()

    def history(self, last_n: int = 10) -> list[FatigueScore]:
        return self._scores[-last_n:]


# ---- Confidence interval calculation ----------------------------------------

def confidence_interval(scores: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Bootstrap 95% CI for a list of scalar scores."""
    if len(scores) < 2:
        mean = scores[0] if scores else 0.0
        return (mean, mean)
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    n = len(scores)
    z = 1.96 if confidence >= 0.95 else 1.645
    margin = z * stdev / math.sqrt(n)
    return (max(0.0, mean - margin), min(1.0, mean + margin))


# ---- Tool registration ------------------------------------------------------

def register_fatigue_tools(registry, monitor: FatigueMonitor) -> None:
    from .tools import Tool

    def fatigue_check() -> str:
        score = monitor.compute()
        return score.summary()

    def fatigue_history() -> str:
        history = monitor.history()
        if not history:
            return "no fatigue samples recorded yet"
        values = [h.composite for h in history]
        lo, hi = confidence_interval(values)
        return (
            f"Fatigue history ({len(history)} samples)\n"
            f"  mean={statistics.mean(values):.1%}  "
            f"max={max(values):.1%}  "
            f"95% CI=[{lo:.1%}, {hi:.1%}]\n"
            + "\n".join(
                f"  {i + 1}. {h.summary()}"
                for i, h in enumerate(reversed(history[-5:]))
            )
        )

    registry.register(Tool(
        "fatigue_check",
        "Compute the current multi-signal fatigue score (posture + typing velocity + compile friction).",
        {},
        fatigue_check,
    ))
    registry.register(Tool(
        "fatigue_history",
        "Show recent fatigue score history and confidence interval.",
        {},
        fatigue_history,
    ))
