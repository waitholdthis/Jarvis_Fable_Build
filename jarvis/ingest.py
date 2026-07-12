"""Document ingestion: turn files and directories into searchable memory.

Plain-text formats only (code, markdown, config, logs, csv...) so ingestion
needs no parsers or binary-format dependencies. Chunking splits on paragraph
boundaries with overlap, and every chunk is prefixed with its source path so
retrieved chunks carry provenance into the prompt.
"""

from __future__ import annotations

from pathlib import Path

from .memory import Memory

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".bash",
    ".zsh", ".fish", ".ps1", ".sql", ".html", ".css", ".xml", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".log", ".csv", ".tsv",
}

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


def looks_textual(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.suffix:
        return False
    try:  # extensionless files (Makefile, LICENSE...): sniff for binary bytes
        return b"\x00" not in path.open("rb").read(1024)
    except OSError:
        return False


def chunk_text(text: str, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    """Split text into ~chunk_chars pieces, preferring paragraph boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # Hard-split paragraphs that alone exceed the chunk size.
        while len(para) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:chunk_chars])
            para = para[chunk_chars - overlap_chars:]
        if current and len(current) + len(para) + 2 > chunk_chars:
            chunks.append(current)
            current = current[-overlap_chars:] if overlap_chars else ""
        current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current)
    return chunks


def ingest_file(memory: Memory, path: Path, max_bytes: int = 2_000_000) -> int:
    """Ingest one file; returns the number of chunks stored."""
    if path.stat().st_size > max_bytes or not looks_textual(path):
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0

    source = str(path)
    memory.forget(source)  # re-ingesting a file replaces its old chunks
    count = 0
    for chunk in chunk_text(text):
        memory.remember(f"[{path.name}] {chunk}", source=source, kind="doc")
        count += 1
    return count


def ingest_path(memory: Memory, target: Path) -> tuple[int, int]:
    """Ingest a file or directory tree. Returns (files, chunks)."""
    target = target.expanduser().resolve()
    if target.is_file():
        chunks = ingest_file(memory, target)
        return (1 if chunks else 0), chunks

    files = 0
    chunks = 0
    for path in sorted(target.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            added = ingest_file(memory, path)
            if added:
                files += 1
                chunks += added
    return files, chunks
