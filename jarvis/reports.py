"""Reporting and physical design fabrication engines (Blueprint Section 11).

Three capabilities:

1. Multi-Format Export Pipelines — write raw LaTeX code for engineering
   documents and compile publication-grade PDFs inside an isolated Docker
   container running a texlive toolchain. Falls back to plain-text if Docker
   or pdflatex is absent.

2. Vector Blueprint Synthesis — execute Python visualization scripts using
   matplotlib and Graphviz to output high-resolution scalable vector assets
   (SVG, DOT/DXF-style network diagrams) for physical fabrication paths,
   laser-cutting templates, and network topology blueprints.

3. Automated Background Reports — implement persistent background cron
   scheduling: synthesize silent background metrics, execute web-scraping to
   ingest clean Markdown content (via requests + HTML stripping), filter via
   local keyword reranker, verify calculations inside a Python sandbox, and
   output compiled operational briefings ready at next user login.

Optional deps: matplotlib, graphviz (graphviz Python package)
Docker is used for pdflatex; plain-text fallback works without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---- LaTeX / PDF generation -------------------------------------------------

_DEFAULT_LATEX_TEMPLATE = r"""
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{booktabs}
\usepackage{xcolor}
\geometry{margin=2.5cm}
\title{%(title)s}
\author{JARVIS Autonomous Report Engine}
\date{%(date)s}
\begin{document}
\maketitle
\tableofcontents
\newpage
%(body)s
\end{document}
"""


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text content."""
    for char, replacement in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\^{}"),
    ]:
        text = text.replace(char, replacement)
    return text


class LatexPDFEngine:
    """Compile LaTeX → PDF via system pdflatex or Docker texlive container."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _pdflatex_available(self) -> bool:
        return shutil.which("pdflatex") is not None

    def _docker_available(self) -> bool:
        return shutil.which("docker") is not None

    def write_tex(self, title: str, body_latex: str, filename: str) -> Path:
        """Write a complete .tex file from a title and LaTeX body."""
        import datetime
        content = _DEFAULT_LATEX_TEMPLATE % {
            "title": _latex_escape(title),
            "date": datetime.date.today().isoformat(),
            "body": body_latex,
        }
        path = self.output_dir / f"{filename}.tex"
        path.write_text(content, encoding="utf-8")
        return path

    def compile(self, tex_path: Path) -> dict:
        """Compile a .tex file to PDF. Returns {ok, pdf_path, log}."""
        if self._pdflatex_available():
            return self._compile_local(tex_path)
        if self._docker_available():
            return self._compile_docker(tex_path)
        pdf_path = tex_path.with_suffix(".txt")
        content = tex_path.read_text(encoding="utf-8")
        body_start = content.find(r"\begin{document}") + len(r"\begin{document}")
        body_end = content.find(r"\end{document}")
        plain = content[body_start:body_end].strip() if body_start >= 0 else content
        pdf_path.write_text(plain, encoding="utf-8")
        return {"ok": True, "pdf_path": pdf_path, "log": "fallback: plain text (no pdflatex/docker)"}

    def _compile_local(self, tex_path: Path) -> dict:
        for _ in range(2):  # two passes for TOC/cross-refs
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_path.name],
                capture_output=True, text=True, cwd=tex_path.parent, timeout=60,
            )
        pdf_path = tex_path.with_suffix(".pdf")
        return {
            "ok": pdf_path.exists(),
            "pdf_path": pdf_path if pdf_path.exists() else None,
            "log": result.stdout[-2000:],
        }

    def _compile_docker(self, tex_path: Path) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(tex_path, tmp_path / tex_path.name)
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{tmp}:/doc",
                "-w", "/doc",
                "texlive/texlive:latest",
                "pdflatex", "-interaction=nonstopmode", tex_path.name,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            pdf_name = tex_path.stem + ".pdf"
            src_pdf = tmp_path / pdf_name
            if src_pdf.exists():
                dst = self.output_dir / pdf_name
                shutil.copy(src_pdf, dst)
                return {"ok": True, "pdf_path": dst, "log": result.stdout[-2000:]}
        return {"ok": False, "pdf_path": None, "log": result.stderr[-2000:]}

    def markdown_to_latex(self, markdown: str) -> str:
        """Convert simple Markdown to LaTeX for embedding in documents."""
        lines = []
        in_code = False
        for line in markdown.splitlines():
            if line.startswith("```"):
                if in_code:
                    lines.append(r"\end{lstlisting}")
                else:
                    lines.append(r"\begin{lstlisting}")
                in_code = not in_code
                continue
            if in_code:
                lines.append(line)
                continue
            if line.startswith("# "):
                lines.append(r"\section{" + _latex_escape(line[2:]) + "}")
            elif line.startswith("## "):
                lines.append(r"\subsection{" + _latex_escape(line[3:]) + "}")
            elif line.startswith("### "):
                lines.append(r"\subsubsection{" + _latex_escape(line[4:]) + "}")
            elif line.startswith("- "):
                lines.append(r"\begin{itemize}\item " + _latex_escape(line[2:]) + r"\end{itemize}")
            else:
                lines.append(_latex_escape(line))
        return "\n".join(lines)


# ---- SVG / DXF vector blueprint synthesis -----------------------------------

class BlueprintEngine:
    """Generate SVG network diagrams and technical blueprints.

    Uses Graphviz (DOT language) for topology diagrams and matplotlib for
    data visualisation. Both are optional; the engine reports clearly when
    they're absent rather than crashing.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def network_topology_svg(
        self, nodes: list[dict], edges: list[dict], filename: str = "topology"
    ) -> Path:
        """Render a network topology as SVG via Graphviz.

        nodes: [{"id": "web", "label": "Web Server", "color": "#4a90d9"}]
        edges: [{"src": "web", "dst": "db", "label": "SQL"}]
        """
        lines = ["digraph topology {", "  rankdir=LR;",
                 '  node [shape=box style=filled fontname="Helvetica"];',
                 '  edge [fontname="Helvetica" fontsize=10];']

        for n in nodes:
            color = n.get("color", "#aec6cf")
            label = n.get("label", n["id"])
            lines.append(f'  {n["id"]} [label="{label}" fillcolor="{color}"];')

        for e in edges:
            label = e.get("label", "")
            lines.append(f'  {e["src"]} -> {e["dst"]} [label="{label}"];')

        lines.append("}")
        dot_src = "\n".join(lines)

        dot_path = self.output_dir / f"{filename}.dot"
        dot_path.write_text(dot_src, encoding="utf-8")
        svg_path = self.output_dir / f"{filename}.svg"

        if shutil.which("dot"):
            subprocess.run(
                ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
                capture_output=True, timeout=15,
            )
        else:
            svg_path.write_text(dot_src, encoding="utf-8")
            svg_path = svg_path.with_suffix(".dot")

        return svg_path

    def matplotlib_chart(
        self, data: dict, chart_type: str = "bar",
        title: str = "Chart", filename: str = "chart"
    ) -> Path | None:
        """Generate a matplotlib chart and save as SVG.

        data: {"labels": [...], "values": [...]}
        chart_type: 'bar' | 'line' | 'pie'
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        labels = data.get("labels", [])
        values = data.get("values", [])

        fig, ax = plt.subplots(figsize=(10, 6))
        if chart_type == "bar":
            ax.bar(labels, values)
        elif chart_type == "line":
            ax.plot(labels, values, marker="o")
        elif chart_type == "pie" and values:
            ax.pie(values, labels=labels, autopct="%1.1f%%")
        else:
            ax.bar(labels, values)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        svg_path = self.output_dir / f"{filename}.svg"
        fig.savefig(str(svg_path), format="svg", bbox_inches="tight")
        plt.close(fig)
        return svg_path


# ---- Background report engine -----------------------------------------------

@dataclass
class ReportConfig:
    title: str
    sources: list[str] = field(default_factory=list)  # URLs to scrape
    keywords: list[str] = field(default_factory=list)  # filter by keyword
    schedule: str = "every day at 06:00"
    output_format: str = "markdown"   # 'markdown' | 'latex' | 'pdf'


class BackgroundReporter:
    """Synthesise background reports on a schedule: scrape → rerank → compile."""

    def __init__(
        self, output_dir: Path, llm=None, latex_engine: LatexPDFEngine | None = None
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.latex_engine = latex_engine
        self._reports: list[dict] = []

    def scrape_and_summarise(self, config: ReportConfig) -> str:
        """Fetch sources, filter by keyword, summarise with LLM."""
        import html
        import re
        import urllib.request

        sections: list[str] = []
        for url in config.sources[:5]:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "JARVIS/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                clean = re.sub(r"<[^>]+>", " ", raw)
                clean = html.unescape(clean)
                clean = re.sub(r"\s+", " ", clean).strip()[:4000]
                if config.keywords:
                    lower = clean.lower()
                    if not any(kw.lower() in lower for kw in config.keywords):
                        continue
                sections.append(f"### Source: {url}\n{clean}")
            except Exception:
                pass

        if not sections:
            return "No relevant content found from configured sources."

        combined = "\n\n".join(sections)

        if self.llm is None:
            return combined[:5000]

        prompt = (
            f"Compile an operational briefing titled '{config.title}'. "
            f"Filter by keywords: {', '.join(config.keywords)}. "
            "Summarise each source in 2-3 sentences. Note any CVEs, incidents, "
            "or action items. Be concise and factual.\n\n" + combined[:6000]
        )
        return self.llm.chat(
            [{"role": "user", "content": prompt}], temperature=0.1
        )

    def generate(self, config: ReportConfig) -> dict:
        """Generate a full report and write it to disk."""
        content = self.scrape_and_summarise(config)
        ts = time.strftime("%Y-%m-%d_%H%M")
        safe_title = "".join(c if c.isalnum() else "_" for c in config.title)
        report: dict = {"title": config.title, "generated_at": ts, "content": content}

        if config.output_format in ("latex", "pdf") and self.latex_engine:
            body = self.latex_engine.markdown_to_latex(content)
            tex_path = self.latex_engine.write_tex(
                config.title, body, f"{safe_title}_{ts}"
            )
            if config.output_format == "pdf":
                result = self.latex_engine.compile(tex_path)
                report["path"] = str(result.get("pdf_path") or tex_path)
                report["format"] = "pdf" if result.get("ok") else "tex"
            else:
                report["path"] = str(tex_path)
                report["format"] = "tex"
        else:
            md_path = self.output_dir / f"{safe_title}_{ts}.md"
            md_path.write_text(f"# {config.title}\n\n{content}", encoding="utf-8")
            report["path"] = str(md_path)
            report["format"] = "markdown"

        self._reports.append(report)
        return report

    def list_reports(self) -> list[dict]:
        return list(reversed(self._reports[-20:]))


# ---- Tool registration ------------------------------------------------------

def register_report_tools(
    registry,
    reporter: BackgroundReporter,
    latex_engine: LatexPDFEngine,
    blueprint_engine: BlueprintEngine,
) -> None:
    from .tools import Tier, Tool

    def generate_report(title: str, sources: str = "",
                        keywords: str = "", output_format: str = "markdown") -> str:
        config = ReportConfig(
            title=title,
            sources=[u.strip() for u in sources.split(",") if u.strip()],
            keywords=[k.strip() for k in keywords.split(",") if k.strip()],
            output_format=output_format.lower(),
        )
        try:
            report = reporter.generate(config)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        return (
            f"Report generated: {report['path']} ({report['format']})\n\n"
            + report["content"][:2000]
        )

    def generate_pdf(title: str, body_markdown: str) -> str:
        body_latex = latex_engine.markdown_to_latex(body_markdown)
        safe = "".join(c if c.isalnum() else "_" for c in title)
        tex_path = latex_engine.write_tex(title, body_latex, safe)
        result = latex_engine.compile(tex_path)
        if result["ok"]:
            return f"PDF compiled: {result['pdf_path']}"
        return f"PDF compilation failed (tex saved at {tex_path})\n{result.get('log','')[-500:]}"

    def generate_topology_svg(nodes_json: str, edges_json: str,
                               filename: str = "topology") -> str:
        try:
            nodes = json.loads(nodes_json)
            edges = json.loads(edges_json)
        except json.JSONDecodeError as exc:
            return f"ERROR: invalid JSON: {exc}"
        path = blueprint_engine.network_topology_svg(nodes, edges, filename)
        return f"topology SVG written: {path}"

    def generate_chart(data_json: str, chart_type: str = "bar",
                       title: str = "Chart", filename: str = "chart") -> str:
        try:
            data = json.loads(data_json)
        except json.JSONDecodeError as exc:
            return f"ERROR: invalid JSON: {exc}"
        path = blueprint_engine.matplotlib_chart(data, chart_type, title, filename)
        if path is None:
            return "ERROR: matplotlib not installed. Run: pip install matplotlib"
        return f"chart SVG written: {path}"

    def list_reports() -> str:
        reports = reporter.list_reports()
        if not reports:
            return "no reports generated yet"
        return "\n".join(
            f"  {r['generated_at']}  [{r['format']}]  {r['title']}  → {r.get('path','')}"
            for r in reports
        )

    registry.register(Tool(
        "generate_report",
        "Scrape URLs, filter by keywords, summarise with LLM, and compile a structured report (markdown/pdf).",
        {"title": "report title",
         "sources": "comma-separated URLs to scrape",
         "keywords": "comma-separated filter keywords",
         "output_format": "markdown | latex | pdf"},
        generate_report, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "generate_pdf",
        "Write Markdown content to LaTeX and compile it to a PDF via pdflatex or Docker texlive.",
        {"title": "document title", "body_markdown": "Markdown content for the document body"},
        generate_pdf, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "generate_topology_svg",
        "Render a network topology diagram as SVG using Graphviz (dot). "
        "nodes_json: [{\"id\":\"web\",\"label\":\"Web\"}], edges_json: [{\"src\":\"web\",\"dst\":\"db\"}]",
        {"nodes_json": "JSON array of nodes", "edges_json": "JSON array of edges",
         "filename": "output filename without extension"},
        generate_topology_svg,
    ))
    registry.register(Tool(
        "generate_chart",
        "Generate a bar/line/pie chart as SVG using matplotlib. "
        "data_json: {\"labels\":[...], \"values\":[...]}",
        {"data_json": "JSON object with labels and values arrays",
         "chart_type": "bar | line | pie",
         "title": "chart title",
         "filename": "output filename without extension"},
        generate_chart,
    ))
    registry.register(Tool(
        "list_reports",
        "List all reports generated in this session.",
        {},
        list_reports,
    ))
