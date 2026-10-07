import numpy as np
import pandas as pd
import pytest

from agent.dataset_registry import DatasetRegistry
from agent.llm_agent import build_resume_messages, run_agent
from agent.state import AgentState
from experiments.registry import ExperimentRegistry
from tests.unit.fake_llm_client import FakeLLMClient, make_text_response, make_tool_response


def _churn_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "churned": (rng.normal(size=n) > 0).astype(int),
        }
    )


@pytest.fixture
def context():
    df = _churn_df()
    dataset_registry = DatasetRegistry()
    version_id = dataset_registry.register(df, dataset_version_id="v1")
    experiment_registry = ExperimentRegistry()
    state = AgentState(project_id="p1", dataset_version_id=version_id, objective="predict churn")
    return state, dataset_registry, experiment_registry


class TestDoneFlow:
    def test_end_turn_with_no_tool_calls_returns_done(self, context):
        state, dataset_registry, experiment_registry = context
        client = FakeLLMClient([
            make_text_response("No data yet, nothing to do."),
        ])

        result = run_agent(client, state, dataset_registry, experiment_registry)

        assert result.status == "done"
        assert result.final_text == "No data yet, nothing to do."
        assert result.tool_calls == []


class TestToolCallingFlow:
    def test_dispatches_a_real_tool_and_continues(self, context):
        state, dataset_registry, experiment_registry = context
        client = FakeLLMClient([
            make_tool_response([{"id": "call_1", "name": "profile_columns", "input": {"dataset_version_id": "v1"}}]),
            make_text_response("Profiled the dataset."),
        ])

        result = run_agent(client, state, dataset_registry, experiment_registry)

        assert result.status == "done"
        assert result.tool_calls == [{"name": "profile_columns", "input": {"dataset_version_id": "v1"}}]
        # the tool_result made it back into the conversation
        tool_result_msg = result.messages[-2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["tool_use_id"] == "call_1"
        assert tool_result_msg["content"][0]["is_error"] is False
        assert '"n_rows": 200' in tool_result_msg["content"][0]["content"]

    def test_multiple_tool_calls_in_one_turn_all_get_results(self, context):
        state, dataset_registry, experiment_registry = context
        client = FakeLLMClient([
            make_tool_response([
                {"id": "call_1", "name": "profile_columns", "input": {"dataset_version_id": "v1"}},
                {"id": "call_2", "name": "check_data_quality", "input": {"dataset_version_id": "v1"}},
            ]),
            make_text_response("done"),
        ])

        result = run_agent(client, state, dataset_registry, experiment_registry)

        tool_result_msg = result.messages[-2]
        ids = {r["tool_use_id"] for r in tool_result_msg["content"]}
        assert ids == {"call_1", "call_2"}


class TestAskUserFlow:
    def test_ask_user_stops_the_loop_with_a_pending_question(self, context):
        state, dataset_registry, experiment_registry = context
        client = FakeLLMClient([
            make_tool_response([{
                "id": "ask_1",
                "name": "ask_user",
                "input": {
                    "question": "Which column is the target?",
                    "reason": "detect_target was ambiguous",
                    "context": {"candidates": ["x1", "x2"]},
                },
            }]),
        ])

        result = run_agent(client, state, dataset_registry, experiment_registry)

        assert result.status == "awaiting_user_input"
        assert result.ask_user_tool_use_id == "ask_1"
        assert result.pending_question.question == "Which column is the target?"
        assert result.pending_question.context["candidates"] == ["x1", "x2"]
        # only one API call was made — ask_user stops the loop, it isn't dispatched
        assert len(client.chat.completions.calls) == 1

    def test_resume_appends_the_answer_as_a_tool_result(self, context):
        state, dataset_registry, experiment_registry = context
        client = FakeLLMClient([
            make_tool_response([{
                "id": "ask_1",
                "name": "ask_user",
                "input": {"question": "Which target?", "reason": "ambiguous"},
            }]),
        ])
        first = run_agent(client, state, dataset_registry, experiment_registry)

        resumed_messages = build_resume_messages(first, answer="churned")

        assert resumed_messages[-1]["role"] == "user"
        assert resumed_messages[-1]["content"][0]["tool_use_id"] == "ask_1"
        assert resumed_messages[-1]["content"][0]["content"] == "churned"

        client2 = FakeLLMClient([
            make_text_response("Proceeding with churned as target."),
        ])
        second = run_agent(client2, state, dataset_registry, experiment_registry, messages=resumed_messages)
        assert second.status == "done"

    def test_resume_raises_when_not_awaiting_user_input(self, context):
        state, dataset_registry, experiment_registry = context
        client = FakeLLMClient([
            make_text_response("done"),
        ])
        done_result = run_agent(client, state, dataset_registry, experiment_registry)
        with pytest.raises(ValueError, match="awaiting_user_input"):
            build_resume_messages(done_result, answer="anything")


class TestToolErrorHandling:
    def test_tool_error_is_reported_not_raised(self, context):
        state, dataset_registry, experiment_registry = context
        client = FakeLLMClient([
            make_tool_response([{
                "id": "call_1",
                "name": "detect_leakage",
                "input": {"dataset_version_id": "does_not_exist", "target": "y"},
            }]),
            make_text_response("Hit an error, stopping."),
        ])

        result = run_agent(client, state, dataset_registry, experiment_registry)

        assert result.status == "done"
        tool_result_msg = result.messages[-2]
        assert tool_result_msg["content"][0]["is_error"] is True
        assert "Error" in tool_result_msg["content"][0]["content"]


class TestMaxTurns:
    def test_stops_at_max_turns_if_the_model_never_finishes(self, context):
        state, dataset_registry, experiment_registry = context
        infinite_tool_calls = [
            make_tool_response([{"id": f"call_{i}", "name": "profile_columns", "input": {"dataset_version_id": "v1"}}])
            for i in range(10)
        ]
        client = FakeLLMClient(infinite_tool_calls)

        result = run_agent(client, state, dataset_registry, experiment_registry, max_turns=3)

        assert result.status == "max_turns_reached"
        assert len(client.chat.completions.calls) == 3


class TestRealPipelineThroughTheLoop:
    def test_profile_then_run_experiment_produces_a_real_recorded_experiment(self, context):
        state, dataset_registry, experiment_registry = context
        client = FakeLLMClient([
            make_tool_response([{"id": "c1", "name": "profile_columns", "input": {"dataset_version_id": "v1"}}]),
            make_tool_response([{
                "id": "c2",
                "name": "run_experiment",
                "input": {
                    "dataset_version_id": "v1",
                    "target": "churned",
                    "task_type": "binary_classification",
                    "hypothesis": "establish a logistic regression baseline",
                    "candidate_models": ["logistic_regression"],
                },
            }]),
            make_text_response("Baseline established."),
        ])

        result = run_agent(client, state, dataset_registry, experiment_registry, artifact_dir=None)

        assert result.status == "done"
        assert len(experiment_registry.all()) == 1
        recorded = experiment_registry.all()[0]
        assert recorded.hypothesis == "establish a logistic regression baseline"
        assert recorded.model == "logistic_regression"
