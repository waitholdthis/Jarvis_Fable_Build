import json
import time

from jarvis.consolidate import (
    _extract_json_array,
    consolidate,
    maybe_consolidate,
)
from jarvis.embeddings import HashEmbedder
from jarvis.memory import Memory


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def chat(self, messages, temperature=0.7):
        self.prompts.append(messages[-1]["content"])
        return self.responses.pop(0)


def make_memory(tmp_path):
    return Memory(tmp_path / "m.db", embedder=HashEmbedder())


def fill_episodic(mem, n=6):
    for i in range(n):
        mem.log("user", f"my favorite editor is helix, message {i}")
        mem.log("assistant", f"noted, message {i}")


def test_extract_json_array_variants():
    assert _extract_json_array('["a", "b"]') == ["a", "b"]
    assert _extract_json_array('Here you go:\n["x"]\nthanks') == ["x"]
    assert _extract_json_array("[]") == []
    assert _extract_json_array("no json at all") == []
    assert _extract_json_array('[1, "keep", null]') == ["keep"]


def test_consolidate_stores_facts_and_advances_watermark(tmp_path):
    mem = make_memory(tmp_path)
    fill_episodic(mem)
    llm = FakeLLM([json.dumps(["The user's favorite editor is Helix."])])

    stored = consolidate(mem, llm)
    assert stored == 1
    assert mem.stats()["facts"] == 1
    # The transcript reached the model.
    assert "favorite editor is helix" in llm.prompts[0]
    # Retrievable afterwards.
    hits = mem.search("which editor does the user like")
    assert hits and "Helix" in hits[0].content

    # Second run: watermark advanced, nothing new to process.
    llm2 = FakeLLM([json.dumps(["should never be asked"])])
    assert consolidate(mem, llm2) == 0
    assert llm2.prompts == []


def test_consolidate_dedupes_existing_facts(tmp_path):
    mem = make_memory(tmp_path)
    fill_episodic(mem)
    mem.remember("The user's favorite editor is Helix.", source="consolidation", kind="fact")
    llm = FakeLLM([json.dumps(["The user's favorite editor is Helix.", "The user has a cat."])])
    stored = consolidate(mem, llm)
    assert stored == 1
    assert mem.stats()["facts"] == 2


def test_consolidate_skips_when_too_few_entries(tmp_path):
    mem = make_memory(tmp_path)
    mem.log("user", "hi")
    llm = FakeLLM(["should not be called"])
    assert consolidate(mem, llm, min_entries=4) == 0
    assert llm.prompts == []


def test_consolidate_survives_malformed_llm_output(tmp_path):
    mem = make_memory(tmp_path)
    fill_episodic(mem)
    llm = FakeLLM(["I cannot produce JSON today, sorry."])
    assert consolidate(mem, llm) == 0
    # Watermark still advances: these entries were considered.
    llm2 = FakeLLM(["[]"])
    assert consolidate(mem, llm2) == 0
    assert llm2.prompts == []


def test_maybe_consolidate_respects_interval_and_volume(tmp_path):
    mem = make_memory(tmp_path)

    # Not enough new entries -> no run.
    fill_episodic(mem, n=2)
    llm = FakeLLM(["[]"])
    assert maybe_consolidate(mem, llm, min_new=12) == 0
    assert llm.prompts == []

    # Enough entries -> runs.
    fill_episodic(mem, n=10)
    llm = FakeLLM([json.dumps(["The user tests things thoroughly."])])
    assert maybe_consolidate(mem, llm, min_new=12) == 1

    # Ran moments ago -> interval gate blocks another run.
    fill_episodic(mem, n=10)
    llm = FakeLLM(["should not be called"])
    assert maybe_consolidate(mem, llm, min_new=12, min_interval_hours=12) == 0
    assert llm.prompts == []


def test_meta_roundtrip(tmp_path):
    mem = make_memory(tmp_path)
    assert mem.get_meta("nope", "fallback") == "fallback"
    mem.set_meta("k", "v1")
    mem.set_meta("k", "v2")
    assert mem.get_meta("k") == "v2"
