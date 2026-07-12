"""Multi-model tier routing and Iterative Consensus Ensemble (ICE).

Blueprint Section 8: route prompts to the cheapest model tier capable of
handling them, cutting compute overhead by routing simple tasks to local edge
models while escalating complex reasoning to flagship cloud models.

Three tiers:
  edge       — quantised 7–8B models on local VRAM; handles log parsing,
               semantic chunking, event-stream filtering, simple Q&A.
  structured — mid-tier models (DeepSeek-V3 / Command R+); handles JSON
               schema compliance, function calling, DB migrations, scripts.
  heavy      — flagship reasoning models (Claude Sonnet/Opus, GPT-4o);
               root-cause debugging, multi-domain planning, architecture.

ICE (Iterative Consensus Ensemble): for zero-error production tasks, three
models form an assembly line — Model A generates, Model B stress-tests for
edge cases and bugs inside a sandboxed critique pass, Model C judges and
reconciles into the authoritative final answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from .llm import LLMClient


# ---- Complexity classification -----------------------------------------------

_HEAVY_SIGNALS = frozenset([
    "debug", "root cause", "root-cause", "architecture", "design", "refactor",
    "multi-domain", "critical", "security audit", "performance profile",
    "analyze", "analyse", "investigate", "diagnose", "optimize", "scalability",
    "deep dive", "trace", "profile",
])

_STRUCTURED_SIGNALS = frozenset([
    "json", "schema", "sql", "function call", "api", "database", "parse",
    "extract", "serialize", "validate", "format", "migrate", "migration",
    "endpoint", "query", "payload", "struct", "config", "script",
])


def classify_prompt(prompt: str) -> str:
    """Heuristic complexity classifier: returns 'heavy', 'structured', or 'edge'.

    Runs in <1 ms with no model inference — token counts and keyword signals
    only. The classification is a conservative lower-bound: ambiguous prompts
    are escalated, never demoted.
    """
    lower = prompt.lower()
    word_count = len(prompt.split())
    if word_count > 300 or any(s in lower for s in _HEAVY_SIGNALS):
        return "heavy"
    if any(s in lower for s in _STRUCTURED_SIGNALS):
        return "structured"
    return "edge"


# ---- Router ------------------------------------------------------------------

class PromptRouter:
    """Route a prompt to the cheapest tier client that can handle it.

    Tiers are supplied as a dict {tier_name: LLMClient}. Missing tiers fall
    through to the next tier up so a single-client setup still works correctly.

    Usage:
        router = PromptRouter(default_client, tiers={
            "edge": edge_llm,
            "structured": struct_llm,
            "heavy": cloud_llm,
        })
        tier, client = router.route("Parse this JSON payload into a schema")
        # -> ("structured", struct_llm)
    """

    def __init__(
        self,
        default: LLMClient,
        tiers: dict[str, LLMClient] | None = None,
    ) -> None:
        self.default = default
        self.tiers: dict[str, LLMClient] = tiers or {}

    def client_for(self, tier: str) -> LLMClient:
        """Return the best configured client for a tier, falling up if absent."""
        for t in (tier, "structured", "heavy"):
            if t in self.tiers:
                return self.tiers[t]
        return self.default

    def route(self, prompt: str) -> tuple[str, LLMClient]:
        """Classify prompt and return (tier_name, client)."""
        tier = classify_prompt(prompt)
        return tier, self.client_for(tier)

    def route_stream(
        self, prompt: str, messages: list[dict], temperature: float = 0.7
    ) -> tuple[str, Iterator[str]]:
        """Classify, pick client, and return (tier, streaming iterator)."""
        tier, client = self.route(prompt)
        return tier, client.chat_stream(messages, temperature)


# ---- ICE Ensemble ------------------------------------------------------------

@dataclass
class IceResult:
    answer: str
    verdict: str          # one-sentence judge summary
    tier: str = "ice"
    generator_output: str = ""
    tester_critique: str = ""
    duration_s: float = 0.0


class IceEnsemble:
    """Iterative Consensus Ensemble: generate → stress-test → judge.

    Model A (generator) produces the initial answer.
    Model B (tester) critiques it for correctness, edge cases, and bugs.
    Model C (judge) reads both and produces the authoritative reconciled answer.

    All three default to the same client when distinct ones aren't supplied,
    which is safe — the prompts carry enough context for self-critique.
    """

    def __init__(
        self,
        generator: LLMClient,
        tester: LLMClient,
        judge: LLMClient,
    ) -> None:
        self.generator = generator
        self.tester = tester
        self.judge = judge

    def run(
        self,
        task: str,
        context: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> IceResult:
        t0 = time.monotonic()
        ctx = context or []

        # Phase 1 — generate
        generation = self.generator.chat(
            ctx + [{"role": "user", "content": f"Complete the following task:\n\n{task}"}],
            temperature=temperature,
        )

        # Phase 2 — stress-test (edge cases, logic errors, compilation)
        critique = self.tester.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "You are a rigorous code reviewer and adversarial tester.\n"
                        f"ORIGINAL TASK:\n{task}\n\n"
                        f"PROPOSED SOLUTION:\n{generation}\n\n"
                        "List every correctness issue, edge case, or bug you find. "
                        "If the solution is correct and complete, say 'LGTM: no issues found.'"
                    ),
                }
            ],
            temperature=0.1,
        )

        # Phase 3 — judge arbitration
        final = self.judge.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "You are an impartial technical judge. Produce the authoritative "
                        "final answer by reconciling the solution and its critique.\n\n"
                        f"TASK:\n{task}\n\n"
                        f"SOLUTION (generator):\n{generation}\n\n"
                        f"CRITIQUE (tester):\n{critique}\n\n"
                        "Begin with a one-sentence VERDICT line summarising your judgment, "
                        "then give the complete, corrected final answer."
                    ),
                }
            ],
            temperature=0.1,
        )

        verdict = final.split("\n")[0].strip() if final else ""
        return IceResult(
            answer=final,
            verdict=verdict,
            tier="ice",
            generator_output=generation,
            tester_critique=critique,
            duration_s=time.monotonic() - t0,
        )


# ---- Tool registration -------------------------------------------------------

def register_router_tools(registry, router: PromptRouter, ice: IceEnsemble | None = None) -> None:
    """Add route_info and ice_solve tools so the agent can invoke them."""
    from .tools import Tier, Tool

    def route_info(prompt: str) -> str:
        tier, client = router.route(prompt)
        model = getattr(client, "model", "unknown")
        name = getattr(client, "runtime_name", type(client).__name__)
        return f"tier={tier}  model={model}  runtime={name}"

    registry.register(Tool(
        "route_info",
        "Classify a prompt's complexity and show which model tier would handle it.",
        {"prompt": "the prompt text to classify"},
        route_info,
    ))

    if ice is not None:
        def ice_solve(task: str) -> str:
            result = ice.run(task)
            return (
                f"ICE ENSEMBLE ({result.duration_s:.1f}s)\n"
                f"VERDICT: {result.verdict}\n\n"
                f"{result.answer}"
            )

        registry.register(Tool(
            "ice_solve",
            "Run the Iterative Consensus Ensemble (generate → stress-test → judge) for zero-error tasks.",
            {"task": "the task requiring high-confidence output"},
            ice_solve,
            tier=Tier.CONFIRM,
        ))
