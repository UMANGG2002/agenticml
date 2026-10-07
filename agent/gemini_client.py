
"""Gemini (via a LiteLLM proxy) as an alternate LLM backend for
agent/llm_agent.py's loop — using Google's own google-genai SDK.

Implements the same `client.messages.create(...)` surface that
run_agent() calls on anthropic.Anthropic(). The agent loop itself
has no provider-specific code and does not know which provider it
is talking to.

IMPORTANT:
    This adapter intentionally uses MANUAL function calling.

    Gemini returns function calls to this adapter, but the adapter
    does NOT let google-genai's Automatic Function Calling (AFC)
    execute tools automatically.

    Tool execution remains owned by run_agent(), which preserves the
    provider-neutral architecture and allows the application to apply
    authorization, RBAC, tenant isolation, auditing, retries, timeouts,
    rate limits, etc. before executing a capability.

Credentials are resolved by the caller (see apps/api/routes/agent.py)
and passed in explicitly — never read from the environment here.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from google import genai
from google.genai import types

from agent.message_blocks import AgentMessage, TextBlock, ToolUseBlock


def make_litellm_gemini_client(
    *,
    api_key: str = "",
    base_url: str = "",
) -> genai.Client:
    """Create a google-genai client pointed at the LiteLLM proxy.

    The caller owns credential resolution. No environment variables
    are read here.

    Args:
        api_key: API key forwarded to the LiteLLM proxy.
        base_url: LiteLLM endpoint, e.g. ".../litellm".

    Returns:
        Configured google-genai Client.
    """
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            base_url=base_url,
        ),
    )


def _anthropic_tools_to_genai(
    tools: list[dict[str, Any]],
) -> list[types.Tool]:
    """Convert Anthropic-style tool definitions to Gemini tools.

    Example input:

        {
            "name": "get_weather",
            "description": "Get weather information.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }

    Gemini expects FunctionDeclaration objects containing the JSON
    schema under parameters_json_schema.
    """
    if not tools:
        return []

    declarations = [
        types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters_json_schema=tool["input_schema"],
        )
        for tool in tools
    ]

    return [
        types.Tool(
            function_declarations=declarations,
        )
    ]


def _parse_tool_result_content(
    content: str,
) -> dict[str, Any]:
    """Convert our tool_result content into Gemini's response dict.

    Successful tools normally return JSON strings.

    Example:
        '{"temperature": 32}'

    becomes:
        {"temperature": 32}

    Non-JSON error strings become:

        {"error": "Error: something failed"}
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"error": content}

    if isinstance(parsed, dict):
        return parsed

    return {"result": parsed}


def _anthropic_messages_to_genai(
    messages: list[dict[str, Any]],
) -> list[types.Content]:
    """Translate Anthropic-shaped messages into google-genai Content.

    The adapter supports:

    - user text
    - assistant text
    - assistant tool_use
    - user tool_result

    Anthropic tool_result contains only the tool_use_id, while Gemini's
    function_response requires the function name.

    Therefore we remember:

        tool_use_id -> function_name

    while walking assistant messages.
    """
    contents: list[types.Content] = []

    # Maps Anthropic tool_use IDs to Gemini function names.
    id_to_name: dict[str, str] = {}

    for message in messages:
        role = message["role"]
        content = message["content"]

        # ------------------------------------------------------------
        # Simple string message
        # ------------------------------------------------------------
        if isinstance(content, str):
            genai_role = "model" if role == "assistant" else "user"

            contents.append(
                types.Content(
                    role=genai_role,
                    parts=[
                        types.Part.from_text(
                            text=content,
                        )
                    ],
                )
            )
            continue

        # ------------------------------------------------------------
        # Assistant message
        # ------------------------------------------------------------
        if role == "assistant":
            parts: list[types.Part] = []

            for block in content:
                block_type = getattr(block, "type", None)

                # Normal assistant text.
                if block_type == "text":
                    parts.append(
                        types.Part.from_text(
                            text=block.text,
                        )
                    )

                # Assistant requested a tool.
                elif block_type == "tool_use":
                    id_to_name[block.id] = block.name

                    parts.append(
                        types.Part.from_function_call(
                            name=block.name,
                            args=block.input,
                        )
                    )

            if parts:
                contents.append(
                    types.Content(
                        role="model",
                        parts=parts,
                    )
                )

            continue

        # ------------------------------------------------------------
        # User message
        # ------------------------------------------------------------
        # The user message can contain tool_result blocks generated by
        # run_agent() after it executes a tool.
        tool_results = [
            block
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "tool_result"
        ]

        if tool_results:
            parts: list[types.Part] = []

            for result in tool_results:
                tool_use_id = result["tool_use_id"]

                function_name = id_to_name.get(
                    tool_use_id,
                    "unknown_tool",
                )

                parts.append(
                    types.Part.from_function_response(
                        name=function_name,
                        response=_parse_tool_result_content(
                            str(result["content"])
                        ),
                    )
                )

            contents.append(
                types.Content(
                    role="user",
                    parts=parts,
                )
            )

            continue

        # ------------------------------------------------------------
        # Normal user text blocks
        # ------------------------------------------------------------
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        ]

        if text_parts:
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text="\n".join(text_parts),
                        )
                    ],
                )
            )

    return contents


class GeminiLiteLLMClient:
    """Drop-in replacement for anthropic.Anthropic().

    The public interface intentionally remains:

        client.messages.create(...)

    This allows run_agent() to remain provider-neutral.

    IMPORTANT:
        Automatic Function Calling is explicitly disabled.

        Gemini only generates the function call.

        run_agent() is responsible for:
            1. receiving the function call
            2. validating it
            3. authorizing it
            4. executing the tool
            5. adding the tool result
            6. calling the model again
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
    ) -> None:
        self._client = make_litellm_gemini_client(
            api_key=api_key,
            base_url=base_url,
        )

        # Preserve the Anthropic-style API:
        #
        #     client.messages.create(...)
        #
        # so run_agent() does not need provider-specific code.
        self.messages = self

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> AgentMessage:
        """Generate one model response.

        This method performs ONE model generation.

        Tool execution is intentionally NOT performed here.
        """

        genai_contents = _anthropic_messages_to_genai(
            messages
        )

        genai_tools = _anthropic_tools_to_genai(
            tools
        )

        response = self._client.models.generate_content(
            model=model,
            contents=genai_contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                tools=genai_tools or None,

                # ----------------------------------------------------
                # IMPORTANT
                # ----------------------------------------------------
                # Disable Google's Automatic Function Calling.
                #
                # We want:
                #
                #     Gemini
                #       ↓
                #     function_call
                #       ↓
                #     run_agent()
                #       ↓
                #     authorization / execution
                #
                # NOT:
                #
                #     Gemini
                #       ↓
                #     Google SDK automatically executes our tool
                #
                automatic_function_calling=(
                    types.AutomaticFunctionCallingConfig(
                        disable=True,
                    )
                ),
            ),
        )

        # ------------------------------------------------------------
        # Gemini requested one or more tools
        # ------------------------------------------------------------
        function_calls = response.function_calls

        if function_calls:
            content: list[Any] = []

            for call in function_calls:
                tool_id = (
                    call.id
                    or f"call_{uuid.uuid4().hex[:12]}"
                )

                content.append(
                    ToolUseBlock(
                        id=tool_id,
                        name=call.name,
                        input=dict(call.args)
                        if call.args
                        else {},
                    )
                )

            return AgentMessage(
                content=content,
                stop_reason="tool_use",
            )

        # ------------------------------------------------------------
        # Normal text response
        # ------------------------------------------------------------
        text = response.text

        content = (
            [TextBlock(text=text)]
            if text
            else []
        )

        # Gemini's candidate finish reason is an enum.
        finish_reason = (
            response.candidates[0].finish_reason
            if response.candidates
            else None
        )

        stop_reason = (
            "max_tokens"
            if finish_reason == types.FinishReason.MAX_TOKENS
            else "end_turn"
        )

        return AgentMessage(
            content=content,
            stop_reason=stop_reason,
        )
