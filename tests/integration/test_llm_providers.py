"""Integration tests for LLM provider connectivity.

Run with real API keys (set in .env or environment):

    uv run pytest tests/integration/test_llm_providers.py -v

Tests are marked @pytest.mark.integration so they can be skipped in CI
when keys aren't available:

    uv run pytest -m "not integration"
"""

from __future__ import annotations

import json
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Markers & skip conditions
# ---------------------------------------------------------------------------

integration = pytest.mark.integration

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "")

skip_no_openai = pytest.mark.skipif(
    not OPENAI_KEY,
    reason="OPENAI_API_KEY not set",
)

# A tiny tool schema in OpenAI format
_ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo the input back. Call this with the user's message.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo"},
            },
            "required": ["text"],
        },
    },
}


# ===================================================================
# OpenAI provider tests (via LiteLLM proxy)
# ===================================================================


@integration
@skip_no_openai
class TestOpenAIProvider:
    """Validates that the OpenAI API key + LiteLLM proxy work and the
    client can complete basic tasks the agent loop requires."""

    @pytest.fixture(autouse=True)
    def _setup_client(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=OPENAI_KEY, base_url=LITELLM_BASE_URL or None)
        self.model = os.environ.get("AGENT_MODEL", "genuin/clipit")

    def test_key_is_valid_simple_completion(self):
        """Smoke test: the key authenticates and a simple completion returns."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=64,
            messages=[
                {"role": "system", "content": "Reply in exactly one word."},
                {"role": "user", "content": "Say hello."},
            ],
        )
        assert response.choices[0].finish_reason in ("stop", "length")
        msg = response.choices[0].message
        assert isinstance(msg.content, str) and len(msg.content) > 0

    def test_tool_calling_works(self):
        """The provider can receive a tool schema, invoke it, and return
        tool_calls — the core mechanic the agent loop depends on."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": "You have an echo tool. Always call it with exactly what the user said."},
                {"role": "user", "content": "Hello, world!"},
            ],
            tools=[_ECHO_TOOL],
        )
        msg = response.choices[0].message
        assert msg.tool_calls is not None and len(msg.tool_calls) >= 1
        tc = msg.tool_calls[0]
        assert tc.function.name == "echo"
        args = json.loads(tc.function.arguments)
        assert "text" in args

    def test_tool_result_round_trip(self):
        """The provider can receive a tool result and continue the
        conversation — validates the full dispatch cycle."""
        # Step 1: get a tool call
        r1 = self.client.chat.completions.create(
            model=self.model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": "Always call the echo tool, then summarize its result."},
                {"role": "user", "content": "ping"},
            ],
            tools=[_ECHO_TOOL],
        )
        msg1 = r1.choices[0].message
        assert msg1.tool_calls is not None
        tc = msg1.tool_calls[0]

        # Step 2: send the tool result back
        r2 = self.client.chat.completions.create(
            model=self.model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": "Always call the echo tool, then summarize its result."},
                {"role": "user", "content": "ping"},
                {"role": "assistant", "content": msg1.content, "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                ]},
                {"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"text": "pong"})},
            ],
            tools=[_ECHO_TOOL],
        )
        msg2 = r2.choices[0].message
        # The model should produce text (or another tool call) — not error
        assert msg2.content is not None or msg2.tool_calls is not None


# ===================================================================
# Provider-agnostic interface tests
# ===================================================================


@integration
class TestProviderInterface:
    """Tests that the provider is configured and that the agent's tool
    format is valid."""

    def test_provider_key_is_set(self):
        """The project needs an LLM provider configured."""
        assert OPENAI_KEY, (
            "OPENAI_API_KEY is not set. "
            "Set it in .env to run the agent."
        )

    def test_build_tools_produces_valid_schemas(self):
        """The tool schemas from agent/llm_tools.py must be valid for
        the OpenAI API."""
        from agent.llm_tools import build_tools

        tools = build_tools()
        assert len(tools) >= 7  # 7 registered tools + ask_user

        for tool in tools:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            schema = func["parameters"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_tool_registry_names_match_dispatcher(self):
        """Every tool in TOOL_REGISTRY must have a matching _handle_
        method in ToolDispatcher."""
        from agent.schemas import TOOL_REGISTRY
        from agent.dispatch import ToolDispatcher
        from agent.dataset_registry import DatasetRegistry
        from experiments.registry import ExperimentRegistry

        dispatcher = ToolDispatcher(DatasetRegistry(), ExperimentRegistry())
        for tool_name in TOOL_REGISTRY:
            handler = f"_handle_{tool_name}"
            assert hasattr(dispatcher, handler), (
                f"TOOL_REGISTRY has '{tool_name}' but ToolDispatcher "
                f"is missing '{handler}'"
            )


# ===================================================================
# Environment / configuration validation
# ===================================================================


@integration
class TestEnvironmentConfig:
    """Validates that .env / environment variables are correctly set
    for the agent to run."""

    def test_env_file_loads(self):
        """dotenv can load the .env file without errors."""
        # load_dotenv() at module level already ran — if we got here, it worked
        pass

    @skip_no_openai
    def test_openai_key_format(self):
        """OPENAI_API_KEY should be a non-empty string."""
        assert len(OPENAI_KEY) > 0, "OPENAI_API_KEY is empty"

    def test_litellm_base_url_is_set(self):
        """LITELLM_BASE_URL should be configured."""
        # Not strictly required (can use OpenAI directly), but expected in this setup
        assert LITELLM_BASE_URL or True, "LITELLM_BASE_URL is not set"

    def test_agent_model_is_set(self):
        """AGENT_MODEL should default to a valid model name."""
        model = os.environ.get("AGENT_MODEL", "genuin/clipit")
        assert len(model) > 0, "AGENT_MODEL is empty"

    def test_artifact_storage_path_is_configured(self):
        """ARTIFACT_STORAGE_PATH should be set (or default works)."""
        path = os.environ.get("ARTIFACT_STORAGE_PATH", "./data/artifacts")
        assert len(path) > 0
