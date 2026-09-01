import pytest

from experiments.registry import ExperimentRegistry
from experiments.schemas import ExperimentRecord


def _record(experiment_id, parent_experiment_id=None, metric="f1", primary_score=0.8, model="logistic_regression"):
    return ExperimentRecord(
        experiment_id=experiment_id,
        parent_experiment_id=parent_experiment_id,
        hypothesis="baseline" if parent_experiment_id is None else "some change",
        split_strategy="stratified",
        features=["x"],
        preprocessing="median-impute + scale (numeric); most-frequent-impute + one-hot (categorical)",
        model=model,
        hyperparameters={},
        seed=42,
        metric=metric,
        metrics={"cv_mean": primary_score},
        primary_score=primary_score,
        runtime_seconds=1.0,
        artifact_path=None,
        created_at="2026-01-01T00:00:00+00:00",
    )


class TestRecordAndGet:
    def test_roundtrip(self):
        registry = ExperimentRegistry()
        record = _record("exp_1")
        registry.record(record)
        assert registry.get("exp_1") == record

    def test_duplicate_id_raises(self):
        registry = ExperimentRegistry()
        registry.record(_record("exp_1"))
        with pytest.raises(ValueError):
            registry.record(_record("exp_1"))

    def test_unknown_parent_raises(self):
        registry = ExperimentRegistry()
        with pytest.raises(KeyError):
            registry.record(_record("exp_2", parent_experiment_id="does_not_exist"))

    def test_get_unknown_id_raises(self):
        registry = ExperimentRegistry()
        with pytest.raises(KeyError):
            registry.get("does_not_exist")


class TestParentChildGraph:
    def test_children_of_root(self):
        registry = ExperimentRegistry()
        registry.record(_record("root"))
        registry.record(_record("child_a", parent_experiment_id="root"))
        registry.record(_record("child_b", parent_experiment_id="root"))
        registry.record(_record("grandchild", parent_experiment_id="child_a"))

        children = {c.experiment_id for c in registry.children("root")}
        assert children == {"child_a", "child_b"}

    def test_children_of_leaf_is_empty(self):
        registry = ExperimentRegistry()
        registry.record(_record("root"))
        assert registry.children("root") == []

    def test_children_of_unknown_id_raises(self):
        registry = ExperimentRegistry()
        with pytest.raises(KeyError):
            registry.children("does_not_exist")

    def test_lineage_is_root_to_leaf(self):
        registry = ExperimentRegistry()
        registry.record(_record("root"))
        registry.record(_record("mid", parent_experiment_id="root"))
        registry.record(_record("leaf", parent_experiment_id="mid"))

        lineage = [e.experiment_id for e in registry.lineage("leaf")]
        assert lineage == ["root", "mid", "leaf"]

    def test_lineage_of_root_is_itself(self):
        registry = ExperimentRegistry()
        registry.record(_record("root"))
        assert [e.experiment_id for e in registry.lineage("root")] == ["root"]

    def test_all_returns_every_recorded_experiment(self):
        registry = ExperimentRegistry()
        registry.record(_record("a"))
        registry.record(_record("b", parent_experiment_id="a"))
        assert {e.experiment_id for e in registry.all()} == {"a", "b"}


class TestBest:
    def test_picks_highest_for_higher_is_better_metric(self):
        registry = ExperimentRegistry()
        registry.record(_record("low", metric="f1", primary_score=0.5))
        registry.record(_record("high", metric="f1", primary_score=0.9, parent_experiment_id="low"))
        assert registry.best("f1").experiment_id == "high"

    def test_picks_lowest_for_lower_is_better_metric(self):
        registry = ExperimentRegistry()
        registry.record(_record("high_rmse", metric="rmse", primary_score=5.0))
        registry.record(_record("low_rmse", metric="rmse", primary_score=1.0, parent_experiment_id="high_rmse"))
        assert registry.best("rmse").experiment_id == "low_rmse"

    def test_ignores_experiments_with_a_different_metric(self):
        registry = ExperimentRegistry()
        registry.record(_record("f1_exp", metric="f1", primary_score=0.5))
        registry.record(_record("rmse_exp", metric="rmse", primary_score=0.1))
        assert registry.best("f1").experiment_id == "f1_exp"

    def test_returns_none_when_no_matching_metric_recorded(self):
        registry = ExperimentRegistry()
        registry.record(_record("f1_exp", metric="f1", primary_score=0.5))
        assert registry.best("rmse") is None
