"""Embeddings with a zero-dependency fallback.

Preference order:
  1. sentence-transformers (installed via the [rag] extra) — best quality.
  2. HashEmbedder — a pure-Python hashed bag-of-words vector. No dependencies,
     deterministic, instant. Combined with the keyword scoring in memory
     search it gives serviceable retrieval on any machine.

Both produce plain list[float] so the storage layer never cares which is used.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Words too common to carry meaning; keeps the hashed space for signal.
_STOPWORDS = frozenset(
    "the a an and or of to in is are was were be been it this that with for on"
    " at as by from i you he she they we my your not do does did have has had"
    " what which who how when where why can will would could should".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class HashEmbedder:
    """Feature-hashed unigram+bigram vector with sublinear TF and L2 norm."""

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _slot(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    def embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for a, b in zip(tokens, tokens[1:]):
            bigram = f"{a}__{b}"
            counts[bigram] = counts.get(bigram, 0) + 1

        vec = [0.0] * self.dim
        for token, count in counts.items():
            index, sign = self._slot(token)
            vec[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dep

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


def get_embedder() -> Embedder:
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return HashEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))  # inputs are already L2-normalized
