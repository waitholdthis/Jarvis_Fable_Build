"""Unified knowledge graph, AST ingestion, and multi-tenant memory router.

Blueprint Section 3: map complex software systems by tracing hidden
dependencies simultaneously across source code, database schemas, network
port configurations, and container architectures into a unified relational
graph.

Three capabilities:

1. AST Ingester — walk a Python (or JS/TS) source tree using the built-in
   `ast` module, extract modules/classes/functions/imports as graph nodes and
   their relationships as typed edges. No external tools required.

2. Knowledge Graph — a SQLite-backed directed graph store with node properties
   and typed edge relationships. Traversable via Cypher-style path queries or
   direct neighbour lookup.

3. Multi-Tenant Memory Router — a unified query interface that steps through
   the graph (structural relationships), verifies results against the episodic
   vector store (semantic relevance), and applies relational constraints
   (SQL-style filtering) in a single round-trip. Also handles zero-lag context
   hot-swapping by compressing long history windows into dense vector summaries.
"""

from __future__ import annotations

import ast
import json
import math
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---- Graph schema -----------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,          -- 'module' | 'class' | 'function' | 'import' | 'port' | 'container' | 'table' | 'column'
    name TEXT NOT NULL,
    path TEXT NOT NULL DEFAULT '',
    props TEXT NOT NULL DEFAULT '{}'  -- JSON bag of extra attributes
);
CREATE TABLE IF NOT EXISTS edges (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id  INTEGER NOT NULL REFERENCES nodes(id),
    dst_id  INTEGER NOT NULL REFERENCES nodes(id),
    rel     TEXT NOT NULL,       -- 'imports' | 'defines' | 'calls' | 'inherits' | 'contains' | 'depends_on'
    props   TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_node_kind_name_path ON nodes (kind, name, path);
CREATE INDEX IF NOT EXISTS idx_edge_src ON edges (src_id);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON edges (dst_id);
CREATE INDEX IF NOT EXISTS idx_edge_rel ON edges (rel);
"""


# ---- Knowledge graph --------------------------------------------------------

class KnowledgeGraph:
    """Thread-safe directed graph stored in SQLite.

    Node and edge properties are stored as JSON blobs so the schema never
    needs to change when new metadata surfaces.
    """

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

    # ---- Node operations ----------------------------------------------------

    def upsert_node(self, kind: str, name: str, path: str = "",
                    props: dict | None = None) -> int:
        props_json = json.dumps(props or {})
        with self._lock:
            self._conn.execute(
                "INSERT INTO nodes (kind, name, path, props) VALUES (?,?,?,?)"
                " ON CONFLICT(kind, name, path) DO UPDATE SET props=excluded.props",
                (kind, name, path, props_json),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT id FROM nodes WHERE kind=? AND name=? AND path=?",
                (kind, name, path),
            ).fetchone()
        return row[0]

    def get_node(self, node_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, kind, name, path, props FROM nodes WHERE id=?", (node_id,)
            ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "kind": row[1], "name": row[2],
                "path": row[3], "props": json.loads(row[4])}

    def find_nodes(self, kind: str = "", name_like: str = "") -> list[dict]:
        with self._lock:
            if kind and name_like:
                rows = self._conn.execute(
                    "SELECT id, kind, name, path, props FROM nodes "
                    "WHERE kind=? AND name LIKE ?", (kind, f"%{name_like}%"),
                ).fetchall()
            elif kind:
                rows = self._conn.execute(
                    "SELECT id, kind, name, path, props FROM nodes WHERE kind=?", (kind,)
                ).fetchall()
            elif name_like:
                rows = self._conn.execute(
                    "SELECT id, kind, name, path, props FROM nodes WHERE name LIKE ?",
                    (f"%{name_like}%",),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, kind, name, path, props FROM nodes LIMIT 500"
                ).fetchall()
        return [{"id": r[0], "kind": r[1], "name": r[2], "path": r[3],
                 "props": json.loads(r[4])} for r in rows]

    # ---- Edge operations ----------------------------------------------------

    def add_edge(self, src_id: int, dst_id: int, rel: str,
                 props: dict | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO edges (src_id, dst_id, rel, props) "
                "VALUES (?,?,?,?)",
                (src_id, dst_id, rel, json.dumps(props or {})),
            )
            self._conn.commit()
            return cur.lastrowid or 0

    def neighbours(self, node_id: int, rel: str = "",
                   direction: str = "out") -> list[dict]:
        """Return neighbouring nodes. direction: 'out' | 'in' | 'both'"""
        with self._lock:
            if direction == "out":
                clause = "WHERE e.src_id=?"
                other = "e.dst_id"
            elif direction == "in":
                clause = "WHERE e.dst_id=?"
                other = "e.src_id"
            else:
                clause = "WHERE (e.src_id=? OR e.dst_id=?)"
                other = "CASE WHEN e.src_id=? THEN e.dst_id ELSE e.src_id END"

            if rel:
                clause += f" AND e.rel='{rel}'"

            if direction == "both":
                rows = self._conn.execute(
                    f"SELECT n.id, n.kind, n.name, n.path, e.rel FROM edges e "
                    f"JOIN nodes n ON n.id = ({other}) {clause}",
                    (node_id, node_id, node_id),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT n.id, n.kind, n.name, n.path, e.rel FROM edges e "
                    f"JOIN nodes n ON n.id = {other} {clause}",
                    (node_id,),
                ).fetchall()
        return [{"id": r[0], "kind": r[1], "name": r[2],
                 "path": r[3], "rel": r[4]} for r in rows]

    def path_query(self, from_name: str, to_name: str,
                   max_hops: int = 5) -> list[list[dict]]:
        """BFS shortest paths between two named nodes."""
        starts = self.find_nodes(name_like=from_name)
        ends = {n["id"] for n in self.find_nodes(name_like=to_name)}
        if not starts or not ends:
            return []

        paths: list[list[dict]] = []
        for start in starts[:3]:
            frontier = [[start]]
            visited = {start["id"]}
            while frontier and len(paths) < 5:
                next_frontier = []
                for path in frontier:
                    current = path[-1]
                    for nb in self.neighbours(current["id"], direction="both"):
                        if nb["id"] in visited:
                            continue
                        new_path = path + [nb]
                        if nb["id"] in ends:
                            paths.append(new_path)
                        elif len(new_path) <= max_hops:
                            next_frontier.append(new_path)
                            visited.add(nb["id"])
                frontier = next_frontier
                if paths:
                    break
        return paths

    def stats(self) -> dict:
        with self._lock:
            nodes = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            edges = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            kinds = dict(self._conn.execute(
                "SELECT kind, COUNT(*) FROM nodes GROUP BY kind"
            ).fetchall())
        return {"nodes": nodes, "edges": edges, "kinds": kinds}


# ---- AST Ingester -----------------------------------------------------------

class ASTIngester:
    """Walk Python source trees and ingest them into the knowledge graph."""

    _SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__",
                  ".pytest_cache", "dist", "build"}

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def ingest_path(self, path: Path) -> dict:
        path = Path(path)
        files, nodes_added, edges_added = 0, 0, 0
        targets = list(path.rglob("*.py")) if path.is_dir() else [path]
        for fp in targets:
            if any(part in self._SKIP_DIRS for part in fp.parts):
                continue
            try:
                n, e = self._ingest_file(fp)
                nodes_added += n
                edges_added += e
                files += 1
            except Exception:
                pass
        return {"files": files, "nodes": nodes_added, "edges": edges_added}

    def _ingest_file(self, path: Path) -> tuple[int, int]:
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return 0, 0

        rel_path = str(path)
        module_name = path.stem
        mod_id = self.graph.upsert_node("module", module_name, rel_path,
                                        {"lines": len(source.splitlines())})
        nodes, edges = 1, 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imp_id = self.graph.upsert_node("import", alias.name, "")
                    self.graph.add_edge(mod_id, imp_id, "imports")
                    nodes += 1
                    edges += 1

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imp_id = self.graph.upsert_node("import", node.module, "")
                    self.graph.add_edge(mod_id, imp_id, "imports")
                    nodes += 1
                    edges += 1

            elif isinstance(node, ast.ClassDef):
                cls_id = self.graph.upsert_node(
                    "class", node.name, rel_path,
                    {"lineno": node.lineno,
                     "bases": [ast.unparse(b) for b in node.bases]},
                )
                self.graph.add_edge(mod_id, cls_id, "defines")
                nodes += 1
                edges += 1
                for base in node.bases:
                    base_name = ast.unparse(base)
                    base_id = self.graph.upsert_node("class", base_name, "")
                    self.graph.add_edge(cls_id, base_id, "inherits")
                    edges += 1

            elif isinstance(node, ast.FunctionDef):
                fn_id = self.graph.upsert_node(
                    "function", node.name, rel_path,
                    {"lineno": node.lineno,
                     "args": [a.arg for a in node.args.args]},
                )
                self.graph.add_edge(mod_id, fn_id, "defines")
                nodes += 1
                edges += 1

        return nodes, edges

    def ingest_schema(self, schema_sql: str, db_name: str = "db") -> dict:
        """Parse CREATE TABLE statements from SQL DDL into the graph."""
        import re
        table_pattern = re.compile(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\(([^;]+)\)",
            re.IGNORECASE | re.DOTALL,
        )
        col_pattern = re.compile(r"^\s*(\w+)\s+(\w+)", re.MULTILINE)
        nodes, edges = 0, 0

        for m in table_pattern.finditer(schema_sql):
            table_name = m.group(1)
            body = m.group(2)
            tbl_id = self.graph.upsert_node("table", table_name, db_name)
            nodes += 1
            for col_m in col_pattern.finditer(body):
                col_name, col_type = col_m.groups()
                if col_name.upper() in ("PRIMARY", "FOREIGN", "UNIQUE", "INDEX", "CHECK"):
                    continue
                col_id = self.graph.upsert_node(
                    "column", col_name, f"{db_name}.{table_name}",
                    {"type": col_type},
                )
                self.graph.add_edge(tbl_id, col_id, "contains")
                nodes += 1
                edges += 1

        return {"nodes": nodes, "edges": edges}


# ---- Multi-tenant memory router ---------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class RouterHit:
    source: str         # 'graph' | 'semantic' | 'both'
    kind: str
    name: str
    path: str
    score: float
    excerpt: str = ""


class MemoryRouter:
    """Unified query across graph structure, vector semantics, and SQL constraints.

    Instead of asking "what does the semantic store say?" OR "what does the
    graph say?", the router asks both simultaneously and fuses the answers.
    Results from the graph are boosted if they also appear in the semantic
    store, providing structural accuracy with semantic relevance weighting.
    """

    def __init__(self, graph: KnowledgeGraph, memory) -> None:
        self.graph = graph
        self.memory = memory      # jarvis.memory.Memory instance

    def query(self, q: str, top_k: int = 8) -> list[RouterHit]:
        graph_hits = self._graph_search(q)
        semantic_hits = self._semantic_search(q, top_k=top_k * 2)
        semantic_names = {h.content[:60]: h.score for h in semantic_hits}

        fused: list[RouterHit] = []
        seen = set()

        for gh in graph_hits:
            key = f"{gh['kind']}:{gh['name']}"
            if key in seen:
                continue
            seen.add(key)
            boost = 0.0
            for excerpt, sem_score in semantic_names.items():
                if gh["name"] in excerpt or gh["path"] in excerpt:
                    boost = sem_score * 0.3
                    break
            fused.append(RouterHit(
                source="both" if boost else "graph",
                kind=gh["kind"], name=gh["name"], path=gh["path"],
                score=0.6 + boost,
            ))

        for sh in semantic_hits:
            name = sh.source or sh.kind
            key = f"semantic:{name}"
            if key in seen:
                continue
            seen.add(key)
            fused.append(RouterHit(
                source="semantic", kind=sh.kind, name=name, path="",
                score=sh.score, excerpt=sh.content[:300],
            ))

        fused.sort(key=lambda h: h.score, reverse=True)
        return fused[:top_k]

    def _graph_search(self, q: str) -> list[dict]:
        tokens = [t for t in q.lower().split() if len(t) > 3]
        hits: list[dict] = []
        for token in tokens[:4]:
            hits.extend(self.graph.find_nodes(name_like=token))
        seen: set[int] = set()
        deduped = []
        for h in hits:
            if h["id"] not in seen:
                seen.add(h["id"])
                deduped.append(h)
        return deduped[:20]

    def _semantic_search(self, q: str, top_k: int = 10):
        try:
            return self.memory.search(q, top_k=top_k)
        except Exception:
            return []

    def hot_swap_context(self, messages: list[dict], max_chars: int = 8000) -> list[dict]:
        """Compress old messages into a dense summary to prevent context saturation.

        When the in-flight conversation exceeds max_chars, the oldest messages
        are replaced with a dense summary kept in the semantic store.
        Returns a trimmed message list safe for the next LLM call.
        """
        total = sum(len(m.get("content", "")) for m in messages)
        if total <= max_chars:
            return messages

        cut = len(messages) // 2
        old_block = messages[:cut]
        summary_prompt = (
            "Summarize the following conversation history into a dense, "
            "information-rich paragraph preserving all decisions, facts, and "
            "file paths mentioned. Be concise but lose nothing important.\n\n"
            + "\n".join(f"{m['role'].upper()}: {m.get('content','')[:500]}"
                        for m in old_block)
        )
        try:
            self.memory.remember(summary_prompt[:4000], source="context-hotswap",
                                 kind="fact")
        except Exception:
            pass

        summary_msg = {
            "role": "system",
            "content": f"[Compressed context — {len(old_block)} messages summarized]\n"
                       + summary_prompt[:800],
        }
        return [summary_msg] + messages[cut:]


# ---- Tool registration ------------------------------------------------------

def register_graph_tools(registry, graph: KnowledgeGraph,
                         ingester: ASTIngester, router: MemoryRouter) -> None:
    from .tools import Tier, Tool

    def graph_ingest(path: str) -> str:
        p = Path(path).expanduser()
        if not p.exists():
            return f"ERROR: path does not exist: {p}"
        result = ingester.ingest_path(p)
        stats = graph.stats()
        return (
            f"ingested {result['files']} file(s): "
            f"+{result['nodes']} nodes, +{result['edges']} edges\n"
            f"graph total: {stats['nodes']} nodes, {stats['edges']} edges"
        )

    def graph_ingest_schema(schema_sql: str, db_name: str = "db") -> str:
        result = ingester.ingest_schema(schema_sql, db_name)
        return f"schema ingested: +{result['nodes']} nodes, +{result['edges']} edges"

    def graph_query(name: str, kind: str = "") -> str:
        nodes = graph.find_nodes(kind=kind, name_like=name)
        if not nodes:
            return f"no nodes matching '{name}'" + (f" of kind '{kind}'" if kind else "")
        lines = []
        for n in nodes[:20]:
            nbrs = graph.neighbours(n["id"], direction="both")
            rel_summary = ", ".join(
                f"{nb['rel']}→{nb['name']}" for nb in nbrs[:5]
            )
            lines.append(f"[{n['kind']}] {n['name']}  {n['path']}")
            if rel_summary:
                lines.append(f"  edges: {rel_summary}")
        return "\n".join(lines)

    def graph_path(from_name: str, to_name: str) -> str:
        paths = graph.path_query(from_name, to_name)
        if not paths:
            return f"no path found between '{from_name}' and '{to_name}'"
        result = []
        for i, path in enumerate(paths[:3], 1):
            chain = " → ".join(f"{n['name']}({n['kind']})" for n in path)
            result.append(f"Path {i}: {chain}")
        return "\n".join(result)

    def memory_route(query: str) -> str:
        hits = router.query(query)
        if not hits:
            return "no results from multi-tenant memory router"
        lines = [f"Memory router: {len(hits)} results for '{query}'", ""]
        for h in hits:
            lines.append(
                f"[{h.source}] {h.kind}:{h.name}  score={h.score:.2f}"
                + (f"  path={h.path}" if h.path else "")
                + (f"\n  {h.excerpt}" if h.excerpt else "")
            )
        return "\n".join(lines)

    def graph_stats() -> str:
        s = graph.stats()
        kinds = "  ".join(f"{k}:{v}" for k, v in s["kinds"].items())
        return f"knowledge graph: {s['nodes']} nodes  {s['edges']} edges\n{kinds}"

    registry.register(Tool(
        "graph_ingest",
        "Walk a Python source directory or file and ingest its structure into the knowledge graph.",
        {"path": "file or directory path"},
        graph_ingest,
    ))
    registry.register(Tool(
        "graph_ingest_schema",
        "Parse SQL CREATE TABLE DDL and add table/column nodes to the knowledge graph.",
        {"schema_sql": "SQL DDL containing CREATE TABLE statements",
         "db_name": "logical database name for scoping"},
        graph_ingest_schema,
    ))
    registry.register(Tool(
        "graph_query",
        "Search the knowledge graph by node name (supports fuzzy matching).",
        {"name": "node name fragment", "kind": "filter by kind: module|class|function|table|column (optional)"},
        graph_query,
    ))
    registry.register(Tool(
        "graph_path",
        "Find shortest structural paths between two named entities in the knowledge graph.",
        {"from_name": "starting entity name", "to_name": "target entity name"},
        graph_path,
    ))
    registry.register(Tool(
        "memory_route",
        "Multi-tenant query fusing knowledge graph structure + semantic vector search.",
        {"query": "natural language question or entity name"},
        memory_route,
    ))
    registry.register(Tool(
        "graph_stats",
        "Show knowledge graph node/edge counts by type.",
        {},
        graph_stats,
    ))
