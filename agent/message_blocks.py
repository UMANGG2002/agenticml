"""Provider-agnostic message shapes agent/llm_agent.py's loop depends on.

The agent loop calls `client.chat.completions.create(...)` (OpenAI client
via LiteLLM proxy) and parses the response into these blocks internally.
Test fakes return OpenAI-shaped responses that the loop converts the same
way. The loop only ever reads `.type`, `.text`, `.name`/`.input`/`.id`,
and `.stop_reason` — it never imports a provider SDK's own types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class AgentMessage:
    content: list[Any]
    stop_reason: str
