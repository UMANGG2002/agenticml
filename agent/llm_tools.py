"""Adapts agent/schemas.py's TOOL_REGISTRY to OpenAI's chat-completions tool
format, and defines the synthetic `ask_user` tool the agent calls whenever it
needs a human decision — the LLM-driven counterpart to
agent/orchestrator.py's PendingQuestion, except the model decides when to
raise it instead of a hardcoded gate.
"""

from __future__ import annotations

from typing import Any

from agent.schemas import TOOL_REGISTRY

ASK_USER_TOOL_NAME = "ask_user"

_ASK_USER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The question to put to the user."},
        "reason": {"type": "string", "description": "Why the agent can't proceed without an answer."},
        "context": {
            "type": "object",
            "description": "Supporting evidence — e.g. target candidates, leakage findings.",
        },
    },
    "required": ["question", "reason"],
}


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure `additionalProperties: false` so `strict: true` tool calls validate exactly."""
    schema = dict(schema)
    schema.setdefault("additionalProperties", False)
    return schema


def build_tools() -> list[dict[str, Any]]:
    """The full `tools` list for the OpenAI chat completions API: every tool in
    TOOL_REGISTRY plus the synthetic `ask_user` escape hatch."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.description,
                "parameters": _strict_schema(tool.parameters),
                "strict": True,
            },
        }
        for name, tool in TOOL_REGISTRY.items()
    ]
    tools.append(
        {
            "type": "function",
            "function": {
                "name": ASK_USER_TOOL_NAME,
                "description": (
                    "Ask the human a question when you cannot proceed safely without their input — "
                    "e.g. the target column is ambiguous, or a critical leakage finding needs a decision."
                ),
                "parameters": _strict_schema(_ASK_USER_SCHEMA),
                "strict": True,
            },
        }
    )
    return tools
