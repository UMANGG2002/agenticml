import numpy as np
import pandas as pd
import pytest

from agent.dataset_registry import DatasetRegistry
from agent.dispatch import ToolDispatcher
from experiments.registry import ExperimentRegistry


@pytest.fixture
def df():
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
def dispatcher(df, tmp_path):
    dataset_registry = DatasetRegistry()
    version_id = dataset_registry.register(df, dataset_version_id="v1")
    experiment_registry = ExperimentRegistry()
    return ToolDispatcher(dataset_registry, experiment_registry, artifact_dir=tmp_path), version_id, experiment_registry


class TestDataDetectiveDispatch:
    def test_inspect_dataset(self, dispatcher, df):
        d, version_id, _ = dispatcher
        result = d.dispatch("inspect_dataset", {"dataset_version_id": version_id})
        assert result["n_rows"] == len(df)

    def test_profile_columns(self, dispatcher, df):
        d, version_id, _ = dispatcher
        result = d.dispatch("profile_columns", {"dataset_version_id": version_id})
        assert result["n_rows"] == len(df)
        assert len(result["columns"]) == df.shape[1]

    def test_detect_target(self, dispatcher):
        d, version_id, _ = dispatcher
        result = d.dispatch("detect_target", {"dataset_version_id": version_id, "objective": "predict churn"})
        assert result["recommended"]["column"] == "churned"

    def test_check_data_quality(self, dispatcher):
        d, version_id, _ = dispatcher
        result = d.dispatch("check_data_quality", {"dataset_version_id": version_id})
        assert "issues" in result

    def test_detect_leakage(self, dispatcher):
        d, version_id, _ = dispatcher
        result = d.dispatch("detect_leakage", {"dataset_version_id": version_id, "target": "churned"})
        assert result["has_critical_issues"] is False

    def test_create_split(self, dispatcher, df):
        d, version_id, _ = dispatcher
        result = d.dispatch(
            "create_split",
            {"dataset_version_id": version_id, "strategy": "stratified", "target": "churned", "test_size": 0.2},
        )
        assert len(result["train_indices"]) + len(result["test_indices"]) == len(df)


class TestRunExperimentDispatch:
    def test_records_an_experiment_in_the_registry(self, dispatcher):
        d, version_id, experiment_registry = dispatcher
        result = d.dispatch(
            "run_experiment",
            {
                "dataset_version_id": version_id,
                "target": "churned",
                "task_type": "binary_classification",
                "hypothesis": "establish a baseline",
                "candidate_models": ["logistic_regression"],
            },
        )
        assert result["experiment_id"] in [e.experiment_id for e in experiment_registry.all()]
        assert result["hypothesis"] == "establish a baseline"

    def test_parent_child_linkage_through_dispatch(self, dispatcher):
        d, version_id, experiment_registry = dispatcher
        parent = d.dispatch(
            "run_experiment",
            {
                "dataset_version_id": version_id,
                "target": "churned",
                "task_type": "binary_classification",
                "hypothesis": "baseline",
                "candidate_models": ["logistic_regression"],
            },
        )
        child = d.dispatch(
            "run_experiment",
            {
                "dataset_version_id": version_id,
                "target": "churned",
                "task_type": "binary_classification",
                "hypothesis": "try a random forest",
                "parent_experiment_id": parent["experiment_id"],
                "candidate_models": ["random_forest"],
            },
        )
        assert child["parent_experiment_id"] == parent["experiment_id"]
        assert child["conclusion"] in {"supported", "rejected", "inconclusive"}


class TestUnknownTool:
    def test_unknown_tool_name_raises(self, dispatcher):
        d, version_id, _ = dispatcher
        with pytest.raises(ValueError, match="no handler"):
            d.dispatch("not_a_real_tool", {"dataset_version_id": version_id})
