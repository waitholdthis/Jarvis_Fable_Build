"""The agent loop: prompt assembly, streaming, and prompt-based tool calling.

Tool calls use a fenced block the model writes into its answer:

    ```tool
    {"tool": "read_file", "args": {"path": "notes.txt"}}
    ```

This deliberately avoids native function-calling APIs: it works identically on
every runtime and every model, including small quantized ones, which is what
"runs on any computer" demands. The loop is: generate -> if a tool block is
present, execute it (confirming with the user when policy requires) -> feed
the result back -> repeat, up to a fixed iteration ceiling.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .config import Config
from .llm import LLMClient
from .memory import Memory
from .tools import Tier, ToolRegistry

_TOOL_BLOCK_RE = re.compile(r"```tool\s*\n(.*?)```", re.DOTALL)

_SYSTEM_TEMPLATE = """{persona}

Current date: {date}

## Tools
You may call ONE tool per response by ending your reply with a fenced block:

```tool
{{"tool": "<name>", "args": {{"<arg>": "<value>"}}}}
```

Available tools:
{tools}

Rules:
- Use a tool whenever the answer depends on the user's real files, system
  state, memory, or the current time. Never guess such things.
- After a "TOOL RESULT" message, either call another tool or give the final
  answer. Do not repeat identical tool calls.
- If a tool is denied or errors, adapt or tell the user plainly.
- For ordinary conversation, just answer: no tool block.

## Memory
{memory_context}"""


@dataclass
class AgentEvent:
    kind: str  # 'text' | 'tool_call' | 'tool_result' | 'done'
    text: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract the first well-formed tool call block from model output."""
    match = _TOOL_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    name = payload.get("tool")
    args = payload.get("args", {})
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return name, args


def strip_tool_block(text: str) -> str:
    return _TOOL_BLOCK_RE.sub("", text).strip()


class Agent:
    def __init__(
        self,
        config: Config,
        llm: LLMClient,
        memory: Memory,
        tools: ToolRegistry,
        confirm: Callable[[str], bool] = lambda prompt: False,
    ):
        self.config = config
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.confirm = confirm  # UI supplies this; default denies everything
        self.messages: list[dict] = []  # in-flight conversation (tier T0/T1)

    # ---- prompt assembly ----------------------------------------------------

    def _memory_context(self, user_input: str) -> str:
        parts: list[str] = []
        hits = self.memory.search(user_input, top_k=4)
        if hits:
            parts.append("Relevant knowledge (cite the [source] when used):")
            for h in hits:
                snippet = h.content[:1200]
                parts.append(f"[{h.source or h.kind}]\n{snippet}")
        if not parts:
            parts.append("(no stored knowledge relevant to this message)")
        return "\n\n".join(parts)

    def _system_prompt(self, user_input: str) -> str:
        import datetime

        return _SYSTEM_TEMPLATE.format(
            persona=self.config.persona.format(name=self.config.assistant_name),
            date=datetime.date.today().isoformat(),
            tools=self.tools.describe_all(),
            memory_context=self._memory_context(user_input),
        )

    def _trim_history(self) -> None:
        budget = self.config.max_context_chars
        total = sum(len(m["content"]) for m in self.messages)
        while len(self.messages) > 4 and total > budget:
            dropped = self.messages.pop(0)
            total -= len(dropped["content"])

    # ---- the loop -------------------------------------------------------------

    def run(self, user_input: str) -> Iterator[AgentEvent]:
        self.memory.log("user", user_input)
        self.messages.append({"role": "user", "content": user_input})
        self._trim_history()
        system = {"role": "system", "content": self._system_prompt(user_input)}

        final_answer = ""
        for _ in range(self.config.max_tool_iterations):
            response = ""
            for delta in self.llm.chat_stream(
                [system] + self.messages, self.config.temperature
            ):
                response += delta
                yield AgentEvent("text", text=delta)
            self.messages.append({"role": "assistant", "content": response})

            call = parse_tool_call(response)
            if call is None:
                final_answer = response
                break

            name, args = call
            yield AgentEvent("tool_call", tool=name, args=args)
            result = self._execute(name, args)
            yield AgentEvent("tool_result", tool=name, text=result)
            self.messages.append(
                {"role": "user", "content": f"TOOL RESULT ({name}):\n{result}"}
            )
        else:
            final_answer = "(stopped: reached the tool-iteration limit)"
            yield AgentEvent("text", text="\n" + final_answer)

        self.memory.log("assistant", strip_tool_block(final_answer))
        yield AgentEvent("done", text=strip_tool_block(final_answer))

    def _execute(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is not None and tool.tier is Tier.CONFIRM:
            pretty = json.dumps(args, ensure_ascii=False)
            if not self.confirm(f"Allow {name} with args {pretty}?"):
                return "DENIED: the user declined this action."
        return self.tools.run(name, args)
