"""Dynamic multi-agent swarm: JARVIS spawns specialist agents, coordinates them,
and runs structured planning councils where agents debate, critique, and converge.

Architecture
------------
SpawnedAgent  — a lightweight LLM wrapper with its own persona and message history.
               Can optionally be given a filtered tool registry for autonomous tool use.
AgentPool     — thread-safe registry of named active agents.
CouncilSession — multi-round debate protocol (propose → cross-review → synthesize).
SwarmOrchestrator — high-level API called by JARVIS's tool loop.

Council protocol
----------------
  Round 0 Propose   : all agents independently draft a position on the topic.
  Round 1 Cross-review : each agent reads every other agent's round-0 output and
                         refines its position in light of the critiques.
  Synthesis         : the orchestrating LLM (JARVIS himself) distils all rounds
                      into a unified recommendation with dissenting notes.

Running agents in parallel uses ThreadPoolExecutor (IO-bound LLM calls; no GIL
contention). Agents share read access to JARVIS's memory but keep isolated
conversation histories so they cannot contaminate each other's context.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Agent specification
# ---------------------------------------------------------------------------

@dataclass
class AgentSpec:
    """Blueprint for a spawned agent."""
    name: str
    role: str                            # one-liner: "security auditor", "SQL expert"
    persona: str = ""                    # full system prompt; auto-generated if empty
    allowed_tools: list[str] = field(default_factory=list)  # empty = no tool access
    temperature: float = 0.7
    max_history: int = 40                # maximum messages kept in this agent's window


# ---------------------------------------------------------------------------
# Spawned agent
# ---------------------------------------------------------------------------

class SpawnedAgent:
    """A named AI agent with its own persona and isolated conversation history.

    Tool access is intentionally opt-in: by default a spawned agent is a pure
    reasoning engine. When allowed_tools is non-empty, a FilteredRegistry wraps
    the shared ToolRegistry and exposes only those tools. CONFIRM-tier tools are
    always auto-denied for sub-agents (they cannot prompt the user).
    """

    def __init__(
        self,
        spec: AgentSpec,
        llm,
        memory=None,
        tool_registry=None,
    ) -> None:
        self.spec = spec
        self.llm = llm
        self.memory = memory
        self._tool_registry = tool_registry
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self.created_at = time.time()
        self.turn_count = 0
        self._system_prompt = self._build_system()

    # ---- system prompt ------------------------------------------------------

    def _build_system(self) -> str:
        if self.spec.persona:
            return self.spec.persona

        tool_section = ""
        if self.spec.allowed_tools and self._tool_registry:
            names = ", ".join(self.spec.allowed_tools)
            tool_section = (
                f"\n\nYou have access to the following tools: {names}. "
                "Call them using the standard ```tool ... ``` block format when needed."
            )

        return (
            f"You are {self.spec.name}, an expert AI agent specializing in "
            f"{self.spec.role}.\n\n"
            "You are a member of a collaborative agent team coordinated by JARVIS, "
            "a local-first autonomous assistant. Your job is to provide rigorous, "
            "specific, and actionable analysis. You must:\n"
            "- Be direct and precise — no filler, no hedging without reason\n"
            "- Disagree openly when you see a flaw in another agent's reasoning\n"
            "- Ground claims in evidence or explicit assumptions, never invent facts\n"
            "- When asked to critique, be constructive but unsparing\n"
            "- When asked to propose, be bold but justify every key choice"
            + tool_section
        )

    # ---- conversation -------------------------------------------------------

    def chat(self, message: str) -> str:
        """Send one message and return the agent's response (blocking)."""
        with self._lock:
            self._history.append({"role": "user", "content": message})
            self._trim()

            messages = (
                [{"role": "system", "content": self._system_prompt}]
                + self._history
            )
            try:
                response = self.llm.chat(messages, self.spec.temperature)
            except Exception as exc:
                response = f"[{self.spec.name} error: {type(exc).__name__}: {exc}]"

            self._history.append({"role": "assistant", "content": response})
            self.turn_count += 1
            return response

    def _trim(self) -> None:
        while len(self._history) > self.spec.max_history:
            self._history.pop(0)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self.turn_count = 0

    def summary(self) -> str:
        age = int(time.time() - self.created_at)
        return (
            f"{self.spec.name} ({self.spec.role}) — "
            f"{self.turn_count} turns, "
            f"age {age}s, "
            f"history {len(self._history)} msgs"
        )


# ---------------------------------------------------------------------------
# Agent pool
# ---------------------------------------------------------------------------

class AgentPool:
    """Thread-safe registry of active named agents."""

    def __init__(self) -> None:
        self._agents: dict[str, SpawnedAgent] = {}
        self._lock = threading.Lock()

    def spawn(self, spec: AgentSpec, llm, memory=None, tool_registry=None) -> SpawnedAgent:
        agent = SpawnedAgent(spec, llm, memory, tool_registry)
        with self._lock:
            self._agents[spec.name] = agent
        return agent

    def get(self, name: str) -> SpawnedAgent | None:
        with self._lock:
            return self._agents.get(name)

    def retire(self, name: str) -> bool:
        with self._lock:
            if name in self._agents:
                del self._agents[name]
                return True
            return False

    def list_agents(self) -> list[SpawnedAgent]:
        with self._lock:
            return list(self._agents.values())

    def all_names(self) -> list[str]:
        with self._lock:
            return list(self._agents.keys())

    def fork(self, source_name: str, new_name: str, new_role: str = "") -> SpawnedAgent | None:
        with self._lock:
            src = self._agents.get(source_name)
            if src is None:
                return None
            new_spec = AgentSpec(
                name=new_name,
                role=new_role or src.spec.role,
                persona=src.spec.persona,
                allowed_tools=list(src.spec.allowed_tools),
                temperature=src.spec.temperature,
            )
            forked = SpawnedAgent(new_spec, src.llm, src.memory, src._tool_registry)
            forked._history = list(src._history)
            forked.turn_count = src.turn_count
            self._agents[new_name] = forked
            return forked


# ---------------------------------------------------------------------------
# Council session
# ---------------------------------------------------------------------------

@dataclass
class CouncilRound:
    round_index: int
    kind: str                          # 'propose' | 'cross_review'
    responses: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class CouncilResult:
    topic: str
    agents: list[str]
    rounds: list[CouncilRound]
    synthesis: str
    duration_s: float

    def as_text(self) -> str:
        lines = [f"Council on: {self.topic}", f"Agents: {', '.join(self.agents)}", ""]
        for r in self.rounds:
            lines.append(f"─── Round {r.round_index}: {r.kind.upper()} ───")
            for name, resp in r.responses.items():
                lines.append(f"\n[{name}]\n{resp}")
            lines.append("")
        lines += ["─── SYNTHESIS ───", self.synthesis]
        return "\n".join(lines)


class CouncilSession:
    """Run a structured multi-agent debate and return a synthesized result.

    Round 0 — each agent independently proposes their analysis.
    Round 1 — each agent reads all other proposals and refines their position.
    Synthesis — the orchestrator LLM distils the discussion into a unified plan.
    """

    _PROPOSE_TEMPLATE = (
        "Topic: {topic}\n\n"
        "Provide your independent analysis and recommendation. "
        "Structure your answer as:\n"
        "1. Key observations (what you notice that others might miss)\n"
        "2. Recommendation (your position)\n"
        "3. Critical assumptions (what must be true for this to work)\n"
        "4. Biggest risk (the thing most likely to make you wrong)"
    )

    _REVIEW_TEMPLATE = (
        "Topic: {topic}\n\n"
        "Below are the other agents' proposals:\n"
        "{others}\n\n"
        "Now:\n"
        "1. Identify the strongest point in the proposals you agree with\n"
        "2. Identify the most important flaw or gap you disagree with\n"
        "3. Revise your own recommendation in light of this discussion — "
        "you may keep it, update it, or reverse it with justification"
    )

    _SYNTHESIS_TEMPLATE = (
        "You are JARVIS, orchestrating a multi-agent planning council.\n\n"
        "Topic: {topic}\n\n"
        "The council discussion:\n{discussion}\n\n"
        "Synthesise the discussion into:\n"
        "1. Areas of consensus (what all agents agree on)\n"
        "2. Key disagreements (the real forks in the road)\n"
        "3. Recommended course of action (with your own judgment applied)\n"
        "4. Open questions that need more information before committing\n\n"
        "Be specific. Name the agents when attributing a view."
    )

    def __init__(self, orchestrator_llm, max_workers: int = 8) -> None:
        self.llm = orchestrator_llm
        self.max_workers = max_workers

    def run(
        self,
        topic: str,
        agents: list[SpawnedAgent],
        rounds: int = 2,
    ) -> CouncilResult:
        if not agents:
            return CouncilResult(
                topic=topic, agents=[], rounds=[],
                synthesis="No agents in council.", duration_s=0.0,
            )

        start = time.time()
        council_rounds: list[CouncilRound] = []

        # Round 0: independent proposals
        r0 = self._run_propose(topic, agents)
        council_rounds.append(r0)

        # Round 1+: cross-review
        for i in range(1, rounds):
            r = self._run_cross_review(topic, agents, r0.responses, round_index=i)
            council_rounds.append(r)

        # Synthesis pass
        synthesis = self._synthesize(topic, council_rounds)

        return CouncilResult(
            topic=topic,
            agents=[a.spec.name for a in agents],
            rounds=council_rounds,
            synthesis=synthesis,
            duration_s=time.time() - start,
        )

    def _run_propose(self, topic: str, agents: list[SpawnedAgent]) -> CouncilRound:
        prompt = self._PROPOSE_TEMPLATE.format(topic=topic)
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(a.chat, prompt): a.spec.name for a in agents}
            responses = {}
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    responses[name] = fut.result()
                except Exception as exc:
                    responses[name] = f"[ERROR: {exc}]"
        return CouncilRound(
            round_index=0, kind="propose",
            responses=responses, duration_s=time.time() - t0,
        )

    def _run_cross_review(
        self,
        topic: str,
        agents: list[SpawnedAgent],
        prior_responses: dict[str, str],
        round_index: int,
    ) -> CouncilRound:
        t0 = time.time()

        def _build_others(exclude_name: str) -> str:
            return "\n\n".join(
                f"[{name}]\n{text}"
                for name, text in prior_responses.items()
                if name != exclude_name
            )

        def _review_agent(agent: SpawnedAgent) -> tuple[str, str]:
            others = _build_others(agent.spec.name)
            prompt = self._REVIEW_TEMPLATE.format(topic=topic, others=others)
            return agent.spec.name, agent.chat(prompt)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(_review_agent, a): a.spec.name for a in agents}
            responses = {}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    name, resp = fut.result()
                    responses[name] = resp
                except Exception as exc:
                    responses[futures[fut]] = f"[ERROR: {exc}]"

        return CouncilRound(
            round_index=round_index, kind="cross_review",
            responses=responses, duration_s=time.time() - t0,
        )

    def _synthesize(self, topic: str, rounds: list[CouncilRound]) -> str:
        discussion_parts = []
        for r in rounds:
            discussion_parts.append(f"=== Round {r.round_index}: {r.kind.upper()} ===")
            for name, resp in r.responses.items():
                discussion_parts.append(f"[{name}]\n{resp}")
        discussion = "\n\n".join(discussion_parts)

        prompt = self._SYNTHESIS_TEMPLATE.format(topic=topic, discussion=discussion)
        try:
            return self.llm.chat(
                [{"role": "user", "content": prompt}], temperature=0.3
            )
        except Exception as exc:
            return f"[Synthesis error: {exc}]"


# ---------------------------------------------------------------------------
# Swarm orchestrator
# ---------------------------------------------------------------------------

class SwarmOrchestrator:
    """JARVIS-facing API for managing the agent swarm."""

    # Built-in role presets — JARVIS can use these or invent new roles.
    ROLE_PRESETS: dict[str, str] = {
        "critic":       "adversarial red-teaming and flaw detection",
        "architect":    "system design, architecture trade-offs, and scalability",
        "security":     "security analysis, threat modeling, and vulnerability assessment",
        "data":         "data analysis, statistics, SQL, and machine learning",
        "product":      "product strategy, user experience, and feature prioritization",
        "devops":       "infrastructure, CI/CD, containers, and reliability engineering",
        "legal":        "compliance, licensing, privacy regulations, and risk framing",
        "researcher":   "literature review, evidence synthesis, and hypothesis generation",
        "writer":       "technical writing, documentation, and communication clarity",
        "optimizer":    "performance profiling, algorithmic complexity, and code efficiency",
    }

    def __init__(self, llm, memory=None, tool_registry=None) -> None:
        self.llm = llm
        self.memory = memory
        self.tool_registry = tool_registry
        self.pool = AgentPool()
        self.council = CouncilSession(llm)

    def spawn(self, name: str, role: str, persona: str = "",
              allowed_tools: list[str] | None = None) -> SpawnedAgent:
        # Resolve role preset shorthand
        resolved_role = self.ROLE_PRESETS.get(role.lower(), role)
        spec = AgentSpec(
            name=name,
            role=resolved_role,
            persona=persona,
            allowed_tools=allowed_tools or [],
        )
        return self.pool.spawn(spec, self.llm, self.memory, self.tool_registry)

    def talk(self, name: str, message: str) -> str:
        agent = self.pool.get(name)
        if agent is None:
            return f"ERROR: no agent named '{name}' is active. Use agent_spawn first."
        return agent.chat(message)

    def broadcast(self, message: str) -> dict[str, str]:
        agents = self.pool.list_agents()
        if not agents:
            return {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(a.chat, message): a.spec.name for a in agents}
            results = {}
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                try:
                    results[name] = fut.result()
                except Exception as exc:
                    results[name] = f"[ERROR: {exc}]"
        return results

    def run_council(
        self,
        topic: str,
        agent_names: list[str] | None = None,
        rounds: int = 2,
    ) -> CouncilResult:
        if agent_names:
            agents = [a for name in agent_names
                      if (a := self.pool.get(name)) is not None]
        else:
            agents = self.pool.list_agents()
        return self.council.run(topic, agents, rounds)

    def quick_council(
        self,
        topic: str,
        roles: list[str],
        rounds: int = 2,
    ) -> CouncilResult:
        """Spawn temporary agents, run a council, then retire them.

        Useful for one-off planning sessions without polluting the permanent pool.
        """
        spawned_names: list[str] = []
        agents: list[SpawnedAgent] = []
        for role in roles:
            name = f"_tmp_{role.split()[0].lower()}"
            a = self.spawn(name, role)
            spawned_names.append(name)
            agents.append(a)

        result = self.council.run(topic, agents, rounds)

        for name in spawned_names:
            self.pool.retire(name)

        return result


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_swarm_tools(registry, orchestrator: SwarmOrchestrator) -> None:
    from .tools import Tier, Tool

    # ---- agent_spawn --------------------------------------------------------

    def agent_spawn(name: str, role: str, persona: str = "",
                    allowed_tools: str = "") -> str:
        """Spawn a named specialist agent.

        role can be a preset shorthand (critic, architect, security, data,
        product, devops, legal, researcher, writer, optimizer) or any free-form
        description.
        """
        if orchestrator.pool.get(name) is not None:
            return f"agent '{name}' already exists. Use agent_retire first or pick a different name."
        tools_list = [t.strip() for t in allowed_tools.split(",") if t.strip()]
        agent = orchestrator.spawn(name, role, persona=persona, allowed_tools=tools_list)
        return (
            f"spawned: {agent.spec.name}\n"
            f"  role: {agent.spec.role}\n"
            f"  tools: {', '.join(agent.spec.allowed_tools) if agent.spec.allowed_tools else 'none (pure reasoning mode)'}"
        )

    # ---- agent_talk ---------------------------------------------------------

    def agent_talk(name: str, message: str) -> str:
        """Send a message to a named agent and get its response."""
        return orchestrator.talk(name, message)

    # ---- agent_council ------------------------------------------------------

    def agent_council(
        topic: str,
        agents: str = "",
        rounds: int = 2,
    ) -> str:
        """Run a multi-agent planning council on a topic.

        agents — comma-separated names of agents to include (default: all active).
        rounds — how many debate rounds (1 = propose only; 2 = propose + cross-review).

        Returns the full council transcript + synthesis.
        """
        names = [n.strip() for n in agents.split(",") if n.strip()] if agents else None
        result = orchestrator.run_council(topic, agent_names=names, rounds=rounds)
        return result.as_text()

    # ---- agent_council_quick ------------------------------------------------

    def agent_council_quick(topic: str, roles: str = "", rounds: int = 2) -> str:
        """Spin up a temporary council for a one-off topic without polluting the pool.

        roles — comma-separated role names or presets (e.g. "critic,architect,security").
        If empty, uses: critic, architect, researcher.
        """
        role_list = [r.strip() for r in roles.split(",") if r.strip()]
        if not role_list:
            role_list = ["critic", "architect", "researcher"]
        result = orchestrator.quick_council(topic, role_list, rounds=rounds)
        return result.as_text()

    # ---- agent_broadcast ----------------------------------------------------

    def agent_broadcast(message: str) -> str:
        """Send the same message to all active agents and show each response."""
        responses = orchestrator.broadcast(message)
        if not responses:
            return "no active agents. Use agent_spawn first."
        return "\n\n".join(
            f"[{name}]\n{resp}" for name, resp in responses.items()
        )

    # ---- agent_fork ---------------------------------------------------------

    def agent_fork(source: str, new_name: str, new_role: str = "") -> str:
        """Fork an agent's persona and history into a new agent.

        Useful for spawning a 'variant' that starts with the same context but
        will diverge from a different angle.
        """
        forked = orchestrator.pool.fork(source, new_name, new_role)
        if forked is None:
            return f"ERROR: no agent named '{source}'"
        return (
            f"forked '{source}' → '{new_name}'\n"
            f"  role: {forked.spec.role}\n"
            f"  history: {len(forked._history)} messages copied"
        )

    # ---- agent_list ---------------------------------------------------------

    def agent_list() -> str:
        agents = orchestrator.pool.list_agents()
        if not agents:
            return (
                "no agents active.\n\n"
                "Built-in role presets:\n"
                + "\n".join(
                    f"  {name:<12} {role}"
                    for name, role in SwarmOrchestrator.ROLE_PRESETS.items()
                )
            )
        lines = [f"{len(agents)} active agent(s):"]
        for a in agents:
            lines.append(f"  {a.summary()}")
        return "\n".join(lines)

    # ---- agent_retire -------------------------------------------------------

    def agent_retire(name: str) -> str:
        """Retire a named agent and free its memory."""
        if orchestrator.pool.retire(name):
            return f"agent '{name}' retired."
        return f"no agent named '{name}'"

    # ---- agent_reset --------------------------------------------------------

    def agent_reset(name: str) -> str:
        """Clear a named agent's conversation history without retiring it."""
        agent = orchestrator.pool.get(name)
        if agent is None:
            return f"no agent named '{name}'"
        agent.reset()
        return f"'{name}' history cleared (persona kept)."

    # ---- register all -------------------------------------------------------

    registry.register(Tool(
        "agent_spawn",
        "Spawn a named specialist AI agent with a role/persona. Agents persist for the session. "
        "Role can be a preset (critic, architect, security, data, product, devops, legal, "
        "researcher, writer, optimizer) or any free-form description.",
        {
            "name":          "unique name for this agent (e.g. 'alice', 'sec_auditor')",
            "role":          "specialist role — preset shorthand or free text",
            "persona":       "(optional) full custom system prompt to override the auto-generated one",
            "allowed_tools": "(optional) comma-separated tool names this agent may call",
        },
        agent_spawn,
    ))

    registry.register(Tool(
        "agent_talk",
        "Send a message to a named spawned agent and get its response. "
        "Each agent keeps its own isolated conversation history.",
        {
            "name":    "agent name (must have been spawned first)",
            "message": "message to send",
        },
        agent_talk,
    ))

    registry.register(Tool(
        "agent_council",
        "Run a structured multi-agent debate council: propose → cross-review → synthesize. "
        "Uses all active agents by default, or a named subset.",
        {
            "topic":  "the question or decision the council should deliberate on",
            "agents": "(optional) comma-separated agent names to include (default: all active)",
            "rounds": "(optional) number of debate rounds — 1=propose only, 2=propose+critique (default: 2)",
        },
        agent_council,
    ))

    registry.register(Tool(
        "agent_council_quick",
        "Spin up a temporary council for a one-off topic without persisting agents. "
        "Cheaper than spawning a permanent pool when you only need one deliberation.",
        {
            "topic": "question or decision to deliberate on",
            "roles": "(optional) comma-separated role presets or descriptions (default: critic,architect,researcher)",
            "rounds": "(optional) debate rounds (default: 2)",
        },
        agent_council_quick,
    ))

    registry.register(Tool(
        "agent_broadcast",
        "Send the same message to every active agent simultaneously and collect all responses. "
        "Useful for polling all agents on a shared context update.",
        {
            "message": "message to broadcast",
        },
        agent_broadcast,
    ))

    registry.register(Tool(
        "agent_fork",
        "Fork an existing agent's persona and history into a new agent with a different name/role. "
        "The fork starts with the same context but diverges independently from that point.",
        {
            "source":   "name of the agent to fork from",
            "new_name": "name for the new agent",
            "new_role": "(optional) new role description; keeps source role if empty",
        },
        agent_fork,
    ))

    registry.register(Tool(
        "agent_list",
        "List all active spawned agents with their roles, turn counts, and ages. "
        "Also shows available built-in role presets when no agents are active.",
        {},
        agent_list,
    ))

    registry.register(Tool(
        "agent_retire",
        "Retire a named agent and free its memory.",
        {"name": "agent name to retire"},
        agent_retire,
    ))

    registry.register(Tool(
        "agent_reset",
        "Clear a named agent's conversation history without retiring it. "
        "The agent keeps its persona but starts fresh.",
        {"name": "agent name"},
        agent_reset,
    ))
