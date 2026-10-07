"""A scripted fake matching the surface agent/llm_agent.py calls on a real
anthropic.Anthropic() client — just enough to drive the agentic loop in
tests without any network access. Not a test module itself (no test_
prefix), just a shared fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str


class _FakeMessages:
    def __init__(self, responses: list[FakeMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeMessage:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeAnthropicClient ran out of scripted responses")
        return self._responses.pop(0)


class FakeAnthropicClient:
    """Scripted stand-in for anthropic.Anthropic(): each call to
    `messages.create()` pops the next response, in order."""

    def __init__(self, responses: list[FakeMessage]) -> None:
        self.messages = _FakeMessages(responses)
