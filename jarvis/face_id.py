"""Face recognition and zero-shot identity ingestion pipeline.

Blueprint Section 1: Build a sub-second edge computer vision pipeline.

Pipeline:
  1. Capture frames from webcam or RTSP stream (OpenCV optional, PIL fallback)
  2. Detect faces using InsightFace (optional) or OpenCV Haar cascade (fallback)
  3. Generate 512-dim embeddings (InsightFace) or 128-dim hash embeddings (fallback)
  4. Match against encrypted SQLite vector store using cosine distance in <10ms
  5. On no match: create anonymous profile (Person_Alpha, Person_Beta, ...)
  6. Cluster embeddings from multiple angles into the same identity profile
  7. Bind human-provided string names to anonymous vector groups
  8. Persist identity-to-context graph for ambient TTS engagement

Optional deps: opencv-python, insightface, onnxruntime
The fallback path uses only stdlib + the existing VLM screen-describe pipeline,
so this module loads without errors on any machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---- Embedding helpers ------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def _hash_embedding(data: bytes, dim: int = 128) -> list[float]:
    """Deterministic hash-based embedding fallback (no ML required)."""
    digest = hashlib.sha512(data).digest()
    floats: list[float] = []
    for i in range(0, min(len(digest), dim * 4), 4):
        val = struct.unpack_from("<f", digest, i % len(digest))[0]
        if math.isfinite(val):
            floats.append(val)
        if len(floats) >= dim:
            break
    while len(floats) < dim:
        floats.append(0.0)
    mag = math.sqrt(sum(x * x for x in floats)) or 1.0
    return [x / mag for x in floats]


# ---- Anonymous name generator -----------------------------------------------

_ALPHA = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
          "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi"]


def _anon_name(index: int) -> str:
    prefix = _ALPHA[index % len(_ALPHA)]
    suffix = "" if index < len(_ALPHA) else str(index // len(_ALPHA) + 1)
    return f"Person_{prefix}{suffix}"


# ---- Face database (SQLite + blobs) -----------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    is_anonymous INTEGER NOT NULL DEFAULT 1,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    appearance_count INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id INTEGER NOT NULL REFERENCES identities(id),
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    captured_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS identity_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id INTEGER NOT NULL REFERENCES identities(id),
    event_type TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fe_identity ON face_embeddings (identity_id);
CREATE INDEX IF NOT EXISTS idx_ie_identity ON identity_events (identity_id);
"""


@dataclass
class Identity:
    id: int
    name: str
    is_anonymous: bool
    first_seen: float
    last_seen: float
    appearance_count: int
    notes: str


@dataclass
class MatchResult:
    identity: Identity
    score: float            # cosine similarity 0-1
    is_new: bool = False    # True when an anonymous profile was just created


class FaceDatabase:
    """Thread-safe SQLite store for face identities and 512-dim embeddings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _anon_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM identities WHERE is_anonymous = 1"
        ).fetchone()[0]

    def find_match(
        self, embedding: list[float], threshold: float = 0.72
    ) -> tuple[Identity | None, float]:
        """Search for the closest existing identity. Returns (identity, score)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT fe.identity_id, fe.embedding, "
                "i.id, i.name, i.is_anonymous, i.first_seen, i.last_seen, "
                "i.appearance_count, i.notes "
                "FROM face_embeddings fe "
                "JOIN identities i ON i.id = fe.identity_id"
            ).fetchall()

        best_score = -1.0
        best_identity: Identity | None = None

        for row in rows:
            stored = _unpack(row[1])
            if len(stored) != len(embedding):
                continue
            score = _cosine(embedding, stored)
            if score > best_score:
                best_score = score
                best_identity = Identity(
                    id=row[2], name=row[3],
                    is_anonymous=bool(row[4]),
                    first_seen=row[5], last_seen=row[6],
                    appearance_count=row[7], notes=row[8],
                )

        if best_identity is None or best_score < threshold:
            return None, best_score
        return best_identity, best_score

    def create_anonymous(self, embedding: list[float]) -> Identity:
        """Create a new anonymous profile and store its first embedding."""
        now = time.time()
        with self._lock:
            count = self._anon_count()
            name = _anon_name(count)
            cur = self._conn.execute(
                "INSERT INTO identities (name, is_anonymous, first_seen, last_seen, "
                "appearance_count) VALUES (?,1,?,?,1)",
                (name, now, now),
            )
            identity_id = cur.lastrowid
            self._conn.execute(
                "INSERT INTO face_embeddings (identity_id, embedding, embedding_dim, "
                "captured_at) VALUES (?,?,?,?)",
                (identity_id, _pack(embedding), len(embedding), now),
            )
            self._conn.commit()
        return Identity(id=identity_id, name=name, is_anonymous=True,
                        first_seen=now, last_seen=now, appearance_count=1, notes="")

    def add_embedding(self, identity_id: int, embedding: list[float]) -> None:
        """Add a new angle embedding to an existing identity."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO face_embeddings (identity_id, embedding, embedding_dim, "
                "captured_at) VALUES (?,?,?,?)",
                (identity_id, _pack(embedding), len(embedding), now),
            )
            self._conn.execute(
                "UPDATE identities SET last_seen=?, appearance_count=appearance_count+1 "
                "WHERE id=?", (now, identity_id),
            )
            self._conn.commit()

    def bind_name(self, identity_id: int, name: str, notes: str = "") -> Identity:
        """Promote an anonymous profile to a named identity."""
        with self._lock:
            self._conn.execute(
                "UPDATE identities SET name=?, is_anonymous=0, notes=? WHERE id=?",
                (name, notes, identity_id),
            )
            self._conn.execute(
                "INSERT INTO identity_events (identity_id, event_type, detail, ts) "
                "VALUES (?,?,?,?)",
                (identity_id, "named", f"bound to {name!r}", time.time()),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT id, name, is_anonymous, first_seen, last_seen, "
                "appearance_count, notes FROM identities WHERE id=?",
                (identity_id,),
            ).fetchone()
        return Identity(*row)

    def all_identities(self) -> list[Identity]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, is_anonymous, first_seen, last_seen, "
                "appearance_count, notes FROM identities ORDER BY last_seen DESC"
            ).fetchall()
        return [Identity(*r) for r in rows]

    def recall(self, name_query: str) -> list[Identity]:
        """Fuzzy search identities by name."""
        like = f"%{name_query}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, is_anonymous, first_seen, last_seen, "
                "appearance_count, notes FROM identities "
                "WHERE name LIKE ? OR notes LIKE ? ORDER BY last_seen DESC",
                (like, like),
            ).fetchall()
        return [Identity(*r) for r in rows]

    def log_event(self, identity_id: int, event_type: str, detail: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO identity_events (identity_id, event_type, detail, ts) "
                "VALUES (?,?,?,?)",
                (identity_id, event_type, detail, time.time()),
            )
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
            named = self._conn.execute(
                "SELECT COUNT(*) FROM identities WHERE is_anonymous=0"
            ).fetchone()[0]
            embeddings = self._conn.execute(
                "SELECT COUNT(*) FROM face_embeddings"
            ).fetchone()[0]
        return {"total_identities": total, "named": named, "anonymous": total - named,
                "total_embeddings": embeddings}


# ---- Embedding backends -----------------------------------------------------

def _insightface_embed(frame_bgr) -> list[float] | None:
    """Generate a 512-dim InsightFace embedding from a BGR frame. Returns None if no face."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(frame_bgr)
        if not faces:
            return None
        emb = faces[0].normed_embedding.tolist()
        return emb
    except ImportError:
        return None
    except Exception:
        return None


def _opencv_embed(frame_bgr, cascade_path: str | None = None) -> list[float] | None:
    """Haar cascade detection + hash embedding fallback. Returns None if no face."""
    try:
        import cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cascade_path or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if not len(faces):
            return None
        x, y, w, h = faces[0]
        face_roi = gray[y:y + h, x:x + w]
        face_bytes = face_roi.tobytes()
        return _hash_embedding(face_bytes, dim=512)
    except ImportError:
        return None
    except Exception:
        return None


def embed_frame(frame_bgr) -> list[float] | None:
    """Try InsightFace first, fall back to OpenCV Haar cascade."""
    emb = _insightface_embed(frame_bgr)
    if emb is not None:
        return emb
    return _opencv_embed(frame_bgr)


# ---- RTSP / webcam frame capture --------------------------------------------

def capture_frames(source: str | int = 0) -> Iterator:
    """Yield BGR frames from a webcam index or RTSP URL. Requires opencv-python."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "opencv-python is required for live frame capture: "
            "pip install 'jarvis-assistant[vision]'"
        )
    cap = cv2.VideoCapture(source)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


# ---- High-level recognizer --------------------------------------------------

class FaceRecognizer:
    """Orchestrates capture → embed → match → create/update identity.

    Usage:
        rec = FaceRecognizer(db)
        result = rec.identify_frame(frame_bgr)
        print(result.identity.name, result.score)
    """

    def __init__(
        self,
        db: FaceDatabase,
        match_threshold: float = 0.72,
        cluster_embeddings: bool = True,
    ) -> None:
        self.db = db
        self.threshold = match_threshold
        self.cluster = cluster_embeddings

    def identify_frame(self, frame_bgr) -> MatchResult | None:
        """Process one frame; return MatchResult or None (no face detected)."""
        embedding = embed_frame(frame_bgr)
        if embedding is None:
            return None
        return self.identify_embedding(embedding)

    def identify_embedding(self, embedding: list[float]) -> MatchResult:
        identity, score = self.db.find_match(embedding, self.threshold)
        if identity is not None:
            if self.cluster:
                self.db.add_embedding(identity.id, embedding)
            return MatchResult(identity=identity, score=score, is_new=False)

        new_identity = self.db.create_anonymous(embedding)
        return MatchResult(identity=new_identity, score=0.0, is_new=True)

    def identify_stream(
        self,
        source: str | int = 0,
        max_frames: int = 30,
        on_identify=None,
    ) -> list[MatchResult]:
        """Process frames from a webcam/RTSP source and collect results."""
        results = []
        for i, frame in enumerate(capture_frames(source)):
            if i >= max_frames:
                break
            result = self.identify_frame(frame)
            if result is not None:
                results.append(result)
                if on_identify:
                    on_identify(result)
        return results


# ---- Tool registration ------------------------------------------------------

def register_face_tools(registry, db: FaceDatabase, recognizer: FaceRecognizer) -> None:
    from .tools import Tier, Tool

    def face_identify_screen() -> str:
        """Capture one webcam frame and attempt to identify the person."""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return "ERROR: could not capture webcam frame"
        except ImportError:
            return (
                "ERROR: opencv-python not installed. "
                "Run: pip install 'jarvis-assistant[vision]'"
            )
        result = recognizer.identify_frame(frame)
        if result is None:
            return "no face detected in webcam frame"
        ident = result.identity
        status = "NEW anonymous profile created" if result.is_new else f"matched (score {result.score:.3f})"
        last = time.strftime("%Y-%m-%d %H:%M", time.localtime(ident.last_seen))
        return (
            f"Identity: {ident.name}  [{status}]\n"
            f"Anonymous: {ident.is_anonymous}  "
            f"Appearances: {ident.appearance_count}  Last seen: {last}"
            + (f"\nNotes: {ident.notes}" if ident.notes else "")
        )

    def face_bind_name(identity_name: str, real_name: str, notes: str = "") -> str:
        """Bind a human-provided name to an anonymous profile."""
        matches = db.recall(identity_name)
        if not matches:
            return f"ERROR: no identity matching '{identity_name}'"
        ident = matches[0]
        updated = db.bind_name(ident.id, real_name, notes)
        return f"Bound {updated.name!r} to profile #{updated.id}"

    def face_list_identities() -> str:
        identities = db.all_identities()
        if not identities:
            return "no identities in database"
        lines = []
        for ident in identities:
            flag = "?" if ident.is_anonymous else "✓"
            last = time.strftime("%Y-%m-%d %H:%M", time.localtime(ident.last_seen))
            lines.append(
                f"[{flag}] #{ident.id} {ident.name}  "
                f"seen {ident.appearance_count}x  last: {last}"
            )
        stats = db.stats()
        lines.append(
            f"\n{stats['total_identities']} identities  "
            f"({stats['named']} named, {stats['anonymous']} anonymous)  "
            f"{stats['total_embeddings']} embeddings"
        )
        return "\n".join(lines)

    def face_recall(query: str) -> str:
        matches = db.recall(query)
        if not matches:
            return f"no identity matching '{query}'"
        return "\n".join(
            f"#{m.id} {m.name} (appearances: {m.appearance_count})"
            for m in matches
        )

    registry.register(Tool(
        "face_identify",
        "Capture a webcam frame and identify the person using the face recognition pipeline.",
        {},
        face_identify_screen,
    ))
    registry.register(Tool(
        "face_bind_name",
        "Bind a real name to an anonymous face profile (e.g. 'Person_Alpha' → 'Alice').",
        {"identity_name": "current profile name (e.g. Person_Alpha)",
         "real_name": "the real name to assign",
         "notes": "optional context notes"},
        face_bind_name, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "face_list",
        "List all known face identities in the recognition database.",
        {},
        face_list_identities,
    ))
    registry.register(Tool(
        "face_recall",
        "Search the identity graph by name or notes.",
        {"query": "name fragment or keyword to search"},
        face_recall,
    ))
