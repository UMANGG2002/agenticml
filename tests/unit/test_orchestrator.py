import numpy as np
import pandas as pd
import pytest

from agent.dataset_registry import DatasetRegistry
from agent.orchestrator import run_data_audit
from agent.state import AgentState


def _state(dataset_version_id: str, objective: str = "predict churn") -> AgentState:
    return AgentState(project_id="p1", dataset_version_id=dataset_version_id, objective=objective)


class TestConfidentTargetGoesStraightThrough:
    def test_ready_for_automl_when_target_confident_and_no_leakage(self):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "feature_a": rng.normal(size=n),
                "feature_b": rng.normal(size=n),
                "churned": rng.integers(0, 2, size=n),
            }
        )
        registry = DatasetRegistry()
        version_id = registry.register(df)
        state = _state(version_id)

        result = run_data_audit(registry, state)

        assert result.status == "ready_for_automl"
        assert result.pending_question is None
        assert result.state.target_column == "churned"
        assert result.state.task_type == "binary_classification"
        assert result.leakage is not None
        assert result.leakage.has_critical_issues is False

    def test_does_not_mutate_the_input_state(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=100), "churned": rng.integers(0, 2, size=100)})
        registry = DatasetRegistry()
        version_id = registry.register(df)
        state = _state(version_id)

        run_data_audit(registry, state)

        assert state.target_column is None  # original AgentState instance is untouched


class TestAmbiguousTargetPausesForConfirmation:
    def test_pauses_with_candidates_in_context(self):
        rng = np.random.default_rng(1)
        n = 300
        df = pd.DataFrame(
            {
                "reading_a": rng.normal(size=n),
                "reading_b": rng.normal(size=n),
                "reading_c": rng.integers(0, 3, size=n),
            }
        )
        registry = DatasetRegistry()
        version_id = registry.register(df)
        state = _state(version_id, objective="analyze sensor readings")

        result = run_data_audit(registry, state)

        assert result.status == "paused_for_target_confirmation"
        assert result.pending_question is not None
        assert result.state.target_column is None
        candidate_columns = {c["column"] for c in result.pending_question.context["candidates"]}
        assert candidate_columns.issubset(set(df.columns))

    def test_resumes_after_confirmed_target_is_supplied(self):
        rng = np.random.default_rng(1)
        n = 300
        df = pd.DataFrame(
            {
                "reading_a": rng.normal(size=n),
                "reading_b": rng.normal(size=n),
                "reading_c": rng.integers(0, 3, size=n),
            }
        )
        registry = DatasetRegistry()
        version_id = registry.register(df)
        state = _state(version_id, objective="analyze sensor readings")

        first = run_data_audit(registry, state)
        assert first.status == "paused_for_target_confirmation"

        second = run_data_audit(registry, state, confirmed_target="reading_c")

        assert second.status == "ready_for_automl"
        assert second.state.target_column == "reading_c"
        assert second.pending_question is None

    def test_state_target_column_short_circuits_future_calls(self):
        # Once state.target_column is set (e.g. loaded from a persisted run),
        # the orchestrator should not re-ask even if detection is ambiguous.
        rng = np.random.default_rng(1)
        n = 300
        df = pd.DataFrame(
            {
                "reading_a": rng.normal(size=n),
                "reading_b": rng.normal(size=n),
                "reading_c": rng.integers(0, 3, size=n),
            }
        )
        registry = DatasetRegistry()
        version_id = registry.register(df)
        state = _state(version_id).model_copy(update={"target_column": "reading_c"})

        result = run_data_audit(registry, state)

        assert result.status == "ready_for_automl"
        assert result.state.target_column == "reading_c"


class TestCriticalLeakageBlocksProgress:
    def test_blocked_on_leakage_with_critical_findings_in_context(self):
        n = 200
        target = [i % 2 for i in range(n)]
        df = pd.DataFrame(
            {
                "leaky_copy": target,
                "feature_a": range(n),
                "churned": target,
            }
        )
        registry = DatasetRegistry()
        version_id = registry.register(df)
        state = _state(version_id)

        result = run_data_audit(registry, state)

        assert result.status == "blocked_on_leakage"
        assert result.pending_question is not None
        assert result.leakage.has_critical_issues is True
        finding_columns = [c for f in result.pending_question.context["findings"] for c in f["columns"]]
        assert "leaky_copy" in finding_columns
        # The target is still resolved and recorded even though we're blocked.
        assert result.state.target_column == "churned"

    def test_post_outcome_columns_are_forwarded_to_leakage_check(self):
        n = 50
        target = [i % 2 for i in range(n)]
        df = pd.DataFrame(
            {
                "income_after_application": range(n),
                "churned": target,
            }
        )
        registry = DatasetRegistry()
        version_id = registry.register(df)
        state = _state(version_id).model_copy(update={"target_column": "churned"})

        result = run_data_audit(registry, state, post_outcome_columns=["income_after_application"])

        assert result.status == "blocked_on_leakage"
        finding_types = [f["type"] for f in result.pending_question.context["findings"]]
        assert "post_outcome_feature" in finding_types


class TestDatasetLookup:
    def test_uses_the_dataframe_registered_under_state_dataset_version_id(self):
        registry = DatasetRegistry()
        df_a = pd.DataFrame({"a": range(10), "churned": [0, 1] * 5})
        df_b = pd.DataFrame({"totally_different_columns": range(10)})
        id_a = registry.register(df_a)
        registry.register(df_b)  # a second, unrelated dataset in the same registry

        state = _state(id_a)
        result = run_data_audit(registry, state)

        assert any(c.name == "churned" for c in result.profile.columns)

    def test_unknown_dataset_version_id_raises(self):
        registry = DatasetRegistry()
        state = _state("does_not_exist")
        with pytest.raises(KeyError):
            run_data_audit(registry, state)
