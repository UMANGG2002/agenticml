import numpy as np
import pandas as pd
import pytest
from google.genai import types

from agent.dataset_registry import DatasetRegistry
from agent.gemini_client import (
    GeminiLiteLLMClient,
    _anthropic_messages_to_genai,
    _anthropic_tools_to_genai,
    make_litellm_gemini_client,
)
from agent.llm_agent import run_agent
from agent.message_blocks import TextBlock, ToolUseBlock
from agent.state import AgentState
from experiments.registry import ExperimentRegistry


def _text_response(text: str, finish_reason=types.FinishReason.STOP) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
                finish_reason=finish_reason,
            )
        ]
    )


def _function_call_response(name: str, args: dict, call_id: str | None = None) -> types.GenerateContentResponse:
    part = types.Part.from_function_call(name=name, args=args)
    if call_id:
        part.function_call.id = call_id
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))])


def _client_with(responses, monkeypatch):
    client = GeminiLiteLLMClient(api_key="test-key", base_url="http://fake.local")
    queue = list(responses)

    def _fake_generate_content(**kwargs):
        if not queue:
            raise AssertionError("ran out of scripted responses")
        return queue.pop(0)

    monkeypatch.setattr(client._client.models, "generate_content", _fake_generate_content)
    return client


class TestClientConstruction:
    def test_make_litellm_gemini_client_builds_a_real_genai_client(self):
        client = make_litellm_gemini_client(api_key="k", base_url="http://fake.local")
        assert callable(getattr(client.models, "generate_content", None))


class TestToolTranslation:
    def test_tool_schema_becomes_a_function_declaration(self):
        tools = [{"name": "profile_columns", "description": "profile it", "input_schema": {"type": "object"}}]
        result = _anthropic_tools_to_genai(tools)
        assert len(result) == 1
        assert result[0].function_declarations[0].name == "profile_columns"
        assert result[0].function_declarations[0].parameters_json_schema == {"type": "object"}

    def test_empty_tools_list_returns_empty(self):
        assert _anthropic_tools_to_genai([]) == []


class TestMessageTranslation:
    def test_plain_string_user_message(self):
        result = _anthropic_messages_to_genai([{"role": "user", "content": "hello"}])
        assert result[0].role == "user"
        assert result[0].parts[0].text == "hello"

    def test_assistant_text_and_tool_use_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    TextBlock(text="Let me check."),
                    ToolUseBlock(id="call_1", name="profile_columns", input={"dataset_version_id": "v1"}),
                ],
            }
        ]
        result = _anthropic_messages_to_genai(messages)
        assert result[0].role == "model"
        assert result[0].parts[0].text == "Let me check."
        assert result[0].parts[1].function_call.name == "profile_columns"
        assert dict(result[0].parts[1].function_call.args) == {"dataset_version_id": "v1"}

    def test_tool_result_resolves_to_the_matching_function_name(self):
        messages = [
            {
                "role": "assistant",
                "content": [ToolUseBlock(id="call_1", name="profile_columns", input={})],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": '{"n_rows": 5}', "is_error": False}
                ],
            },
        ]
        result = _anthropic_messages_to_genai(messages)
        function_response_content = result[1].parts[0].function_response
        assert function_response_content.name == "profile_columns"
        assert function_response_content.response == {"n_rows": 5}

    def test_error_tool_result_is_wrapped_as_a_dict(self):
        messages = [
            {"role": "assistant", "content": [ToolUseBlock(id="call_1", name="detect_leakage", input={})]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "Error: boom", "is_error": True}
                ],
            },
        ]
        result = _anthropic_messages_to_genai(messages)
        assert result[1].parts[0].function_response.response == {"error": "Error: boom"}


class TestGeminiLiteLLMClientCreate:
    def test_text_only_response_maps_to_end_turn(self, monkeypatch):
        client = _client_with([_text_response("hi there")], monkeypatch)
        result = client.create(model="m", max_tokens=100, system="sys", tools=[], messages=[{"role": "user", "content": "hi"}])
        assert result.stop_reason == "end_turn"
        assert result.content[0].type == "text"
        assert result.content[0].text == "hi there"

    def test_function_call_response_maps_to_tool_use(self, monkeypatch):
        client = _client_with(
            [_function_call_response("profile_columns", {"dataset_version_id": "v1"}, call_id="call_9")], monkeypatch
        )
        result = client.create(model="m", max_tokens=100, system="sys", tools=[], messages=[])
        assert result.stop_reason == "tool_use"
        block = result.content[0]
        assert block.type == "tool_use"
        assert block.id == "call_9"
        assert block.name == "profile_columns"
        assert block.input == {"dataset_version_id": "v1"}

    def test_missing_call_id_gets_a_generated_one(self, monkeypatch):
        client = _client_with([_function_call_response("profile_columns", {})], monkeypatch)
        result = client.create(model="m", max_tokens=100, system="sys", tools=[], messages=[])
        assert result.content[0].id.startswith("call_")

    def test_max_tokens_finish_reason_maps_correctly(self, monkeypatch):
        client = _client_with([_text_response("partial...", finish_reason=types.FinishReason.MAX_TOKENS)], monkeypatch)
        result = client.create(model="m", max_tokens=1, system="sys", tools=[], messages=[])
        assert result.stop_reason == "max_tokens"


class TestRunAgentWithGeminiClient:
    @pytest.fixture
    def context(self):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({"x1": rng.normal(size=n), "churned": (rng.normal(size=n) > 0).astype(int)})
        dataset_registry = DatasetRegistry()
        version_id = dataset_registry.register(df, dataset_version_id="v1")
        experiment_registry = ExperimentRegistry()
        state = AgentState(project_id="p1", dataset_version_id=version_id, objective="predict churn")
        return state, dataset_registry, experiment_registry

    def test_full_loop_with_gemini_backed_client(self, context, monkeypatch):
        state, dataset_registry, experiment_registry = context
        client = _client_with(
            [
                _function_call_response("profile_columns", {"dataset_version_id": "v1"}, call_id="call_1"),
                _text_response("Profiled it."),
            ],
            monkeypatch,
        )

        result = run_agent(client, state, dataset_registry, experiment_registry, artifact_dir=None)

        assert result.status == "done"
        assert result.final_text == "Profiled it."
        assert result.tool_calls == [{"name": "profile_columns", "input": {"dataset_version_id": "v1"}}]
