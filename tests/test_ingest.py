from pathlib import Path

from jarvis.embeddings import HashEmbedder
from jarvis.ingest import chunk_text, ingest_path, looks_textual
from jarvis.memory import Memory


def test_chunk_short_text_is_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_chunk_empty():
    assert chunk_text("   \n  ") == []


def test_chunk_respects_paragraphs_and_size():
    paras = [f"Paragraph {i}. " + ("lorem ipsum " * 30) for i in range(10)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, chunk_chars=1000, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1200 for c in chunks)
    joined = " ".join(chunks)
    for i in range(10):
        assert f"Paragraph {i}." in joined


def test_chunk_handles_oversized_paragraph():
    text = "x" * 5000
    chunks = chunk_text(text, chunk_chars=1000, overlap_chars=100)
    assert all(len(c) <= 1000 for c in chunks)
    assert sum(len(c) for c in chunks) >= 5000  # overlap means >= original


def test_looks_textual(tmp_path):
    txt = tmp_path / "a.md"
    txt.write_text("# hi")
    binary = tmp_path / "a.bin"
    binary.write_bytes(b"\x00\x01\x02")
    exe = tmp_path / "prog"
    exe.write_bytes(b"\x7fELF\x00")
    assert looks_textual(txt)
    assert not looks_textual(binary)
    assert not looks_textual(exe)


def test_ingest_directory_and_reingest_replaces(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("The API server listens on port 8443.")
    (docs / "image.png").write_bytes(b"\x89PNG\x00\x00")
    (docs / ".git").mkdir()
    (docs / ".git" / "config").write_text("should be skipped")

    mem = Memory(tmp_path / "m.db", embedder=HashEmbedder())
    files, chunks = ingest_path(mem, docs)
    assert files == 1 and chunks == 1

    # Re-ingesting must not duplicate chunks.
    files, chunks = ingest_path(mem, docs)
    assert mem.stats()["semantic_chunks"] == 1

    hits = mem.search("which port does the api server use")
    assert hits and "8443" in hits[0].content
