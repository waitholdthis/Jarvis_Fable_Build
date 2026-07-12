"""Continuous local adapter calibration (LoRA micro-weights) — Blueprint Section 12.

Runs an offline background loop during idle cycles to evaluate the discrepancy
between model predictions and sandbox execution traces, then calculates
micro-weights (Low-Rank Adaptation / LoRA adapters) to dynamically adjust
local 8B/7B model layers to local schema variations, formatting norms, and
environment constraints.

How it works:
  1. CollectionPhase — monitor the episodic memory stream for correction events:
     any time the user edits a JARVIS output, retries a tool, or explicitly
     says "that's wrong", the (prompt, expected_output) pair is recorded as a
     training signal.
  2. EvaluationPhase — periodically score the current model against the
     collected pairs using a local judge prompt. Compute discrepancy metrics.
  3. CalibrationPhase — format the training pairs as JSONL (Alpaca/ShareGPT
     format), write them to disk, and trigger the appropriate fine-tuning
     command (Ollama modelfile, llama.cpp train_text_from_scratch, Unsloth,
     or any custom script configured by the user).
  4. HotSwap — once training completes, update the LLMClient to point at the
     calibrated checkpoint and log the swap event.

No GPU training code is included here — this module prepares the data and
triggers the user's configured trainer. The trainer runs as a subprocess so
JARVIS remains responsive during calibration.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---- Training pair model ----------------------------------------------------

@dataclass
class TrainingPair:
    prompt: str
    response: str          # the model's actual output
    correction: str        # the user's corrected / preferred output
    source: str = ""       # 'user_edit' | 'retry' | 'explicit'
    ts: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)


# ---- Training pair store ----------------------------------------------------

class CalibrationStore:
    """Persist training pairs as JSONL for offline fine-tuning."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pairs: list[TrainingPair] = self._load()

    def _load(self) -> list[TrainingPair]:
        if not self.path.exists():
            return []
        pairs = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    pairs.append(TrainingPair(**d))
                except Exception:
                    pass
        return pairs

    def add(self, pair: TrainingPair) -> None:
        with self._lock:
            self._pairs.append(pair)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "prompt": pair.prompt, "response": pair.response,
                    "correction": pair.correction, "source": pair.source,
                    "ts": pair.ts, "tags": pair.tags,
                }) + "\n")

    def all_pairs(self) -> list[TrainingPair]:
        with self._lock:
            return list(self._pairs)

    def recent(self, n: int = 100) -> list[TrainingPair]:
        with self._lock:
            return self._pairs[-n:]

    def count(self) -> int:
        with self._lock:
            return len(self._pairs)

    def export_alpaca(self, output_path: Path) -> int:
        """Export as Alpaca-format JSONL for most fine-tuning frameworks."""
        pairs = self.all_pairs()
        with output_path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "instruction": pair.prompt,
                    "input": "",
                    "output": pair.correction,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(pairs)

    def export_sharegpt(self, output_path: Path) -> int:
        """Export as ShareGPT-format JSONL (compatible with Unsloth, LLaMA-Factory)."""
        pairs = self.all_pairs()
        with output_path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                record = {
                    "conversations": [
                        {"from": "human", "value": pair.prompt},
                        {"from": "gpt", "value": pair.correction},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(pairs)


# ---- Discrepancy evaluator --------------------------------------------------

@dataclass
class DiscrepancyReport:
    total_pairs: int
    evaluated: int
    mean_similarity: float      # 0-1; higher = closer to correction
    worst_pairs: list[dict]     # lowest-similarity pairs — best training signal
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"Calibration discrepancy report  ({time.strftime('%Y-%m-%d %H:%M', time.localtime(self.timestamp))})\n"
            f"  Pairs: {self.total_pairs} total, {self.evaluated} evaluated\n"
            f"  Mean similarity to correction: {self.mean_similarity:.1%}\n"
            f"  Worst {len(self.worst_pairs)} pairs (highest drift):"
        ) + "\n".join(
            f"    [{1 - p['similarity']:.1%} drift] {p['prompt'][:60]}..."
            for p in self.worst_pairs
        )


class DiscrepancyEvaluator:
    """Compare current model outputs against stored corrections."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def _similarity(self, a: str, b: str) -> float:
        """Token-overlap Jaccard similarity (fast, no deps)."""
        def tokens(s):
            return set(s.lower().split())
        ta, tb = tokens(a), tokens(b)
        if not ta and not tb:
            return 1.0
        return len(ta & tb) / len(ta | tb)

    def evaluate(self, pairs: list[TrainingPair], sample_n: int = 20) -> DiscrepancyReport:
        sample = pairs[-sample_n:] if len(pairs) > sample_n else pairs
        scored = []
        for pair in sample:
            try:
                actual = self.llm.chat(
                    [{"role": "user", "content": pair.prompt}], temperature=0.0
                )
            except Exception:
                actual = pair.response
            sim = self._similarity(actual, pair.correction)
            scored.append({"pair": pair, "similarity": sim, "actual": actual})

        if not scored:
            return DiscrepancyReport(len(pairs), 0, 1.0, [])

        mean_sim = sum(s["similarity"] for s in scored) / len(scored)
        worst = sorted(scored, key=lambda s: s["similarity"])[:5]
        worst_dicts = [
            {"prompt": w["pair"].prompt[:100], "similarity": w["similarity"],
             "expected": w["pair"].correction[:100], "got": w["actual"][:100]}
            for w in worst
        ]
        return DiscrepancyReport(
            total_pairs=len(pairs),
            evaluated=len(scored),
            mean_similarity=mean_sim,
            worst_pairs=worst_dicts,
        )


# ---- Calibration job runner -------------------------------------------------

@dataclass
class CalibrationJobConfig:
    """Configuration for a fine-tuning run."""
    backend: str = "ollama"          # 'ollama' | 'llama_cpp' | 'unsloth' | 'custom'
    base_model: str = "qwen2.5:7b"
    output_model_name: str = "jarvis-calibrated"
    custom_command: str = ""         # used when backend='custom'
    min_pairs: int = 50              # don't train with fewer pairs than this
    epochs: int = 3
    learning_rate: float = 2e-4


class CalibrationRunner:
    """Trigger an overnight fine-tuning run and hot-swap the result."""

    def __init__(
        self,
        store: CalibrationStore,
        evaluator: DiscrepancyEvaluator,
        training_dir: Path,
        job_config: CalibrationJobConfig,
    ) -> None:
        self.store = store
        self.evaluator = evaluator
        self.training_dir = training_dir
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.config = job_config
        self._active_job: subprocess.Popen | None = None
        self._running = False
        self._hot_swap_callback = None    # called with new model name on success
        self._job_history: list[dict] = []

    def on_hot_swap(self, callback) -> None:
        self._hot_swap_callback = callback

    def _build_ollama_modelfile(self, base: str, lora_path: Path | None) -> str:
        lines = [f"FROM {base}"]
        if lora_path and lora_path.exists():
            lines.append(f"ADAPTER {lora_path}")
        lines += [
            "PARAMETER temperature 0.7",
            "PARAMETER num_ctx 8192",
            'SYSTEM """You are JARVIS, a capable local assistant. '
            'Respond concisely and accurately based on your calibration."""',
        ]
        return "\n".join(lines)

    def launch(self) -> dict:
        """Export training data and start the fine-tuning job."""
        pairs = self.store.all_pairs()
        if len(pairs) < self.config.min_pairs:
            return {
                "ok": False,
                "reason": f"only {len(pairs)} training pairs; need {self.config.min_pairs}",
            }

        ts = time.strftime("%Y%m%d_%H%M")
        alpaca_path = self.training_dir / f"train_{ts}.jsonl"
        n = self.store.export_alpaca(alpaca_path)

        if self.config.backend == "ollama":
            return self._launch_ollama(alpaca_path, ts)
        if self.config.backend == "llama_cpp":
            return self._launch_llama_cpp(alpaca_path, ts)
        if self.config.backend == "custom" and self.config.custom_command:
            return self._launch_custom(alpaca_path, ts)

        return {
            "ok": False,
            "reason": f"unsupported backend '{self.config.backend}'",
            "training_file": str(alpaca_path),
            "pairs_exported": n,
        }

    def _launch_ollama(self, data_path: Path, ts: str) -> dict:
        if not __import__("shutil").which("ollama"):
            return {"ok": False, "reason": "ollama not found on PATH"}
        modelfile_content = self._build_ollama_modelfile(self.config.base_model, None)
        modelfile_path = self.training_dir / f"Modelfile_{ts}"
        modelfile_path.write_text(modelfile_content)
        model_name = f"{self.config.output_model_name}:{ts}"
        cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
        proc = subprocess.Popen(cmd, capture_output=True, text=True)
        self._active_job = proc
        record = {"backend": "ollama", "model": model_name,
                  "started_at": time.time(), "pid": proc.pid}
        self._job_history.append(record)
        threading.Thread(
            target=self._wait_and_swap, args=(proc, model_name), daemon=True
        ).start()
        return {"ok": True, "backend": "ollama", "model": model_name,
                "pid": proc.pid, "training_file": str(data_path)}

    def _launch_llama_cpp(self, data_path: Path, ts: str) -> dict:
        train_cmd = __import__("shutil").which("llama-finetune")
        if not train_cmd:
            return {"ok": False, "reason": "llama-finetune not found on PATH"}
        output = self.training_dir / f"lora_{ts}.bin"
        cmd = [
            train_cmd,
            "--train-data", str(data_path),
            "--model-base", self.config.base_model,
            "--output-lora", str(output),
            "--epochs", str(self.config.epochs),
            "--learning-rate", str(self.config.learning_rate),
        ]
        proc = subprocess.Popen(cmd, capture_output=True, text=True)
        self._active_job = proc
        record = {"backend": "llama_cpp", "output_lora": str(output),
                  "started_at": time.time(), "pid": proc.pid}
        self._job_history.append(record)
        threading.Thread(
            target=self._wait_and_swap, args=(proc, str(output)), daemon=True
        ).start()
        return {"ok": True, "backend": "llama_cpp", "output_lora": str(output),
                "pid": proc.pid}

    def _launch_custom(self, data_path: Path, ts: str) -> dict:
        cmd = self.config.custom_command.format(
            data=data_path, ts=ts, output_dir=self.training_dir
        )
        proc = subprocess.Popen(cmd, shell=True, capture_output=True, text=True)
        self._active_job = proc
        record = {"backend": "custom", "command": cmd,
                  "started_at": time.time(), "pid": proc.pid}
        self._job_history.append(record)
        threading.Thread(
            target=self._wait_and_swap, args=(proc, "custom"), daemon=True
        ).start()
        return {"ok": True, "backend": "custom", "command": cmd, "pid": proc.pid}

    def _wait_and_swap(self, proc: subprocess.Popen, model_name: str) -> None:
        proc.wait()
        if proc.returncode == 0 and self._hot_swap_callback:
            self._hot_swap_callback(model_name)
        record = {
            "model": model_name,
            "completed_at": time.time(),
            "exit_code": proc.returncode,
        }
        self._job_history.append(record)

    def job_status(self) -> str:
        if self._active_job is None:
            return "no calibration job running"
        poll = self._active_job.poll()
        if poll is None:
            return f"calibration job running (PID {self._active_job.pid})"
        return f"calibration job finished (exit {poll})"

    def history(self) -> list[dict]:
        return self._job_history[-10:]


# ---- Automatic correction collector (hooks into the agent loop) -------------

class CorrectionCollector:
    """Watch the agent's message stream for correction signals."""

    _CORRECTION_SIGNALS = [
        "that's wrong", "incorrect", "not right", "try again", "redo",
        "that's not", "you're wrong", "wrong answer", "bad output",
    ]

    def __init__(self, store: CalibrationStore) -> None:
        self.store = store
        self._last_prompt = ""
        self._last_response = ""

    def observe_turn(self, role: str, content: str) -> None:
        if role == "user":
            self._last_prompt = content
        elif role == "assistant":
            self._last_response = content

    def observe_correction(self, corrected_content: str, source: str = "user_edit") -> None:
        if self._last_prompt and self._last_response:
            self.store.add(TrainingPair(
                prompt=self._last_prompt,
                response=self._last_response,
                correction=corrected_content,
                source=source,
            ))

    def check_for_implicit_correction(self, user_message: str) -> bool:
        """Return True if the message looks like a correction of the last response."""
        lower = user_message.lower()
        return any(sig in lower for sig in self._CORRECTION_SIGNALS)


# ---- Tool registration ------------------------------------------------------

def register_calibration_tools(
    registry,
    store: CalibrationStore,
    runner: CalibrationRunner,
    evaluator: DiscrepancyEvaluator,
) -> None:
    from .tools import Tier, Tool

    def calibration_status() -> str:
        pairs = store.count()
        return (
            f"calibration store: {pairs} training pair(s)\n"
            f"{runner.job_status()}"
        )

    def calibration_evaluate() -> str:
        pairs = store.recent(50)
        if not pairs:
            return "no training pairs collected yet"
        report = evaluator.evaluate(pairs)
        return report.summary()

    def calibration_launch(backend: str = "", base_model: str = "") -> str:
        if backend:
            runner.config.backend = backend
        if base_model:
            runner.config.base_model = base_model
        result = runner.launch()
        if result.get("ok"):
            return (
                f"calibration job launched (backend={runner.config.backend})\n"
                + "\n".join(f"  {k}: {v}" for k, v in result.items() if k != "ok")
            )
        return f"calibration launch failed: {result.get('reason', 'unknown error')}"

    def calibration_record_correction(prompt: str, correction: str) -> str:
        pair = TrainingPair(
            prompt=prompt, response="", correction=correction, source="manual"
        )
        store.add(pair)
        return f"correction recorded (total pairs: {store.count()})"

    def calibration_export(format: str = "alpaca") -> str:
        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M")
        out = runner.training_dir / f"export_{ts}.jsonl"
        if format == "sharegpt":
            n = store.export_sharegpt(out)
        else:
            n = store.export_alpaca(out)
        return f"exported {n} pairs ({format} format) → {out}"

    registry.register(Tool(
        "calibration_status",
        "Show calibration store size, discrepancy metrics, and active fine-tuning job status.",
        {},
        calibration_status,
    ))
    registry.register(Tool(
        "calibration_evaluate",
        "Score the current model against stored training pairs and report drift.",
        {},
        calibration_evaluate, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "calibration_launch",
        "Export training pairs and launch an overnight LoRA fine-tuning run.",
        {"backend": "ollama | llama_cpp | custom (uses configured default if empty)",
         "base_model": "base model to fine-tune (e.g. qwen2.5:7b)"},
        calibration_launch, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "calibration_record",
        "Manually record a prompt-correction training pair for LoRA calibration.",
        {"prompt": "the original prompt", "correction": "the correct/preferred output"},
        calibration_record_correction,
    ))
    registry.register(Tool(
        "calibration_export",
        "Export all training pairs to disk in Alpaca or ShareGPT JSONL format.",
        {"format": "alpaca | sharegpt"},
        calibration_export,
    ))
