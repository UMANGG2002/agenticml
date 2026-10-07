"""A scripted fake matching the surface agent/llm_agent.py calls on a real
OpenAI() client — just enough to drive the agentic loop in tests without
any network access. Not a test module itself (no test_ prefix), just a
shared fixture.

The agent loop calls `client.chat.completions.create(...)` and gets back an
object with `.choices[0].message` containing `.content` (text or None) and
`.tool_calls` (list or None). This fake produces those shapes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class FakeFunction:
    name: str
    arguments: str  # JSON string


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeChatMessage:
    content: str | None
    tool_calls: list[FakeToolCall] | None = None
    role: str = "assistant"


@dataclass
class FakeChoice:
    message: FakeChatMessage
    finish_reason: str = "stop"


@dataclass
class FakeCompletion:
    choices: list[FakeChoice] = field(default_factory=list)


class _FakeCompletions:
    def __init__(self, responses: list[FakeCompletion]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeCompletion:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        return self._responses.pop(0)


class FakeLLMClient:
    """Scripted stand-in for OpenAI(): each call to
    `chat.completions.create()` pops the next response, in order."""

    def __init__(self, responses: list[FakeCompletion]) -> None:
        self._completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self._completions)


# ---------------------------------------------------------------------------
# Convenience helpers for building responses
# ---------------------------------------------------------------------------

def make_text_response(text: str) -> FakeCompletion:
    """Build a FakeCompletion with a plain text response (end_turn)."""
    return FakeCompletion(
        choices=[FakeChoice(
            message=FakeChatMessage(content=text, tool_calls=None),
            finish_reason="stop",
        )]
    )


def make_tool_response(tool_calls: list[dict[str, Any]], text: str | None = None) -> FakeCompletion:
    """Build a FakeCompletion with tool calls.

    Each entry in tool_calls should be {"id": ..., "name": ..., "input": ...}.
    """
    fake_tool_calls = [
        FakeToolCall(
            id=tc["id"],
            function=FakeFunction(
                name=tc["name"],
                arguments=json.dumps(tc["input"]),
            ),
        )
        for tc in tool_calls
    ]
    return FakeCompletion(
        choices=[FakeChoice(
            message=FakeChatMessage(content=text, tool_calls=fake_tool_calls),
            finish_reason="stop",
        )]
    )
