from jarvis.embeddings import HashEmbedder
from jarvis.memory import Memory


def make_memory(tmp_path):
    return Memory(tmp_path / "mem.db", embedder=HashEmbedder())


def test_episodic_log_and_recent(tmp_path):
    mem = make_memory(tmp_path)
    mem.log("user", "hello")
    mem.log("assistant", "hi there")
    recent = mem.recent()
    assert recent == [("user", "hello"), ("assistant", "hi there")]


def test_semantic_search_finds_relevant_chunk(tmp_path):
    mem = make_memory(tmp_path)
    mem.remember("The wifi password for the office is hunter2-secure", source="notes.md")
    mem.remember("Cats sleep roughly sixteen hours per day", source="trivia.md")
    hits = mem.search("what is the office wifi password")
    assert hits
    assert "hunter2" in hits[0].content
    assert hits[0].source == "notes.md"


def test_search_score_floor_filters_junk(tmp_path):
    mem = make_memory(tmp_path)
    mem.remember("Photosynthesis converts light into chemical energy", source="bio.md")
    hits = mem.search("zzzqx unrelated gibberish tokens", min_score=0.12)
    assert hits == []


def test_forget_removes_matching_rows(tmp_path):
    mem = make_memory(tmp_path)
    mem.remember("secret project codename is bluebird", source="secrets.md")
    mem.remember("lunch is at noon", source="calendar.md")
    removed = mem.forget("bluebird")
    assert removed == 1
    assert mem.search("project codename") == [] or all(
        "bluebird" not in h.content for h in mem.search("project codename")
    )


def test_stats(tmp_path):
    mem = make_memory(tmp_path)
    mem.log("user", "x")
    mem.remember("a fact", kind="fact")
    stats = mem.stats()
    assert stats["episodic_entries"] == 1
    assert stats["semantic_chunks"] == 1
    assert stats["facts"] == 1


def test_persistence_across_reopen(tmp_path):
    mem = make_memory(tmp_path)
    mem.remember("the deploy key lives in ~/.ssh/deploy", source="ops.md")
    mem.close()
    reopened = make_memory(tmp_path)
    hits = reopened.search("where is the deploy key")
    assert hits and "deploy" in hits[0].content
