"""Persistent memory on SQLite: episodic log + semantic (vector) store.

Two tiers, mirroring the architecture spec at any-computer scale:

  episodic  — an append-only transcript of every exchange, queryable by
              recency. Cheap, lossless, never blocks the hot path.
  semantic  — chunks of ingested documents plus distilled facts, each with an
              embedding. Searched with a hybrid score: cosine similarity +
              keyword overlap, so retrieval stays useful even with the
              dependency-free hash embedder.

Vectors are packed as float32 blobs; at personal-assistant scale (tens of
thousands of chunks) a full scan in Python is a few milliseconds — no vector
database required, which is exactly the point of "runs on any computer".
"""

from __future__ import annotations

import sqlite3
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .embeddings import Embedder, cosine, get_embedder, tokenize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS semantic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL DEFAULT 'doc',      -- 'doc' | 'fact'
    source TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    embedder TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodic_ts ON episodic (ts);
CREATE INDEX IF NOT EXISTS idx_semantic_source ON semantic (source);
"""


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


@dataclass
class Hit:
    content: str
    source: str
    kind: str
    score: float


class Memory:
    def __init__(self, db_path: Path | str, embedder: Embedder | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # The web UI touches memory from handler and worker threads; SQLite
        # connections are thread-bound by default, so opt out and serialize
        # access ourselves with a lock.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
        self.embedder = embedder or get_embedder()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- episodic tier ----------------------------------------------------

    def log(self, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO episodic (ts, role, content) VALUES (?, ?, ?)",
                (time.time(), role, content),
            )
            self._conn.commit()

    def recent(self, limit: int = 12) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM episodic ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return list(reversed(rows))

    def episodic_since(self, after_id: int, limit: int = 400) -> list[tuple[int, str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, content FROM episodic WHERE id > ?"
                " ORDER BY id ASC LIMIT ?",
                (after_id, limit),
            ).fetchall()
        return rows

    # ---- meta (key/value bookkeeping, e.g. consolidation watermark) --------

    def get_meta(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    # ---- semantic tier ----------------------------------------------------

    def remember(self, content: str, source: str = "", kind: str = "doc") -> int:
        vec = self.embedder.embed(content)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO semantic (ts, kind, source, content, embedder, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), kind, source, content, self.embedder.name, _pack(vec)),
            )
            self._conn.commit()
            return cur.lastrowid

    def search(self, query: str, top_k: int = 5, min_score: float = 0.12) -> list[Hit]:
        query_vec = self.embedder.embed(query)
        query_tokens = set(tokenize(query))
        with self._lock:
            rows = self._conn.execute(
                "SELECT content, source, kind, embedder, embedding FROM semantic"
            ).fetchall()

        hits: list[Hit] = []
        for content, source, kind, embedder_name, blob in rows:
            # Vectors from a different embedder aren't comparable; fall back
            # to keyword-only scoring for those rows.
            sim = (
                cosine(query_vec, _unpack(blob))
                if embedder_name == self.embedder.name
                else 0.0
            )
            doc_tokens = set(tokenize(content))
            overlap = (
                len(query_tokens & doc_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            score = 0.65 * sim + 0.35 * overlap
            if score >= min_score:
                hits.append(Hit(content=content, source=source, kind=kind, score=score))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def fact_exists(self, content: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM semantic WHERE kind = 'fact' AND content = ? LIMIT 1",
                (content,),
            ).fetchone()
        return row is not None

    def forget(self, pattern: str) -> int:
        """Delete semantic rows whose content or source matches a substring."""
        like = f"%{pattern}%"
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM semantic WHERE content LIKE ? OR source LIKE ?",
                (like, like),
            )
            self._conn.commit()
            return cur.rowcount

    def stats(self) -> dict:
        with self._lock:
            episodic = self._conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
            semantic = self._conn.execute("SELECT COUNT(*) FROM semantic").fetchone()[0]
            facts = self._conn.execute(
                "SELECT COUNT(*) FROM semantic WHERE kind = 'fact'"
            ).fetchone()[0]
        return {
            "episodic_entries": episodic,
            "semantic_chunks": semantic,
            "facts": facts,
            "embedder": self.embedder.name,
            "db_path": str(self.db_path),
        }
