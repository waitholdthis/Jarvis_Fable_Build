"""Memory consolidation — the "hippocampus" job from the architecture spec.

Periodically replays the episodic log (raw conversation history) and asks the
LLM to distill durable facts, which are stored in semantic memory where
retrieval can find them in future conversations. A watermark in the meta
table tracks how far consolidation has progressed, so each exchange is only
ever processed once, and a timestamp prevents the job from running more than
once per interval.

Runs opportunistically in the background at startup (maybe_consolidate) and
on demand via `jarvis consolidate`.
"""

from __future__ import annotations

import json
import re
import time

from .memory import Memory

_WATERMARK_KEY = "consolidated_through"
_LAST_RUN_KEY = "consolidated_at"

_PROMPT = """You maintain long-term memory for a personal assistant. Below is a
transcript of recent exchanges between the user and the assistant.

Extract durable facts worth remembering across future conversations:
preferences, names, projects, machines, paths, recurring routines, decisions.
Rules:
- Each fact must be a single self-contained sentence, understandable with no
  other context (say "the user", never "he/she/they/I").
- Only include things likely to still be true and useful weeks from now.
- Skip small talk, one-off questions, and anything the assistant said that
  the user did not confirm.
- At most {max_facts} facts.

Respond with ONLY a JSON array of strings, e.g. ["fact one", "fact two"].
If there is nothing durable, respond with [].

Transcript:
{transcript}"""


def _extract_json_array(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item.strip() for item in data if isinstance(item, str) and item.strip()]


def consolidate(
    memory: Memory,
    llm,
    min_entries: int = 4,
    max_entries: int = 300,
    max_facts: int = 20,
) -> int:
    """Distill new episodic entries into semantic facts. Returns facts stored."""
    watermark = int(memory.get_meta(_WATERMARK_KEY, "0"))
    rows = memory.episodic_since(watermark, limit=max_entries)
    if len(rows) < min_entries:
        return 0

    transcript = "\n".join(
        f"{role}: {content[:600]}" for _id, role, content in rows
    )
    prompt = _PROMPT.format(max_facts=max_facts, transcript=transcript)
    response = llm.chat([{"role": "user", "content": prompt}], temperature=0.2)

    stored = 0
    for fact in _extract_json_array(response)[:max_facts]:
        if memory.fact_exists(fact):
            continue
        memory.remember(fact, source="consolidation", kind="fact")
        stored += 1

    # Advance the watermark even when nothing was extracted: these entries
    # have been considered, and reconsidering them would never add signal.
    memory.set_meta(_WATERMARK_KEY, str(rows[-1][0]))
    memory.set_meta(_LAST_RUN_KEY, str(time.time()))
    return stored


def maybe_consolidate(
    memory: Memory,
    llm,
    min_new: int = 12,
    min_interval_hours: float = 12.0,
) -> int:
    """Consolidate only if enough new material and enough time have passed."""
    last_run = float(memory.get_meta(_LAST_RUN_KEY, "0") or 0)
    if time.time() - last_run < min_interval_hours * 3600:
        return 0
    watermark = int(memory.get_meta(_WATERMARK_KEY, "0"))
    if len(memory.episodic_since(watermark, limit=min_new)) < min_new:
        return 0
    return consolidate(memory, llm)
