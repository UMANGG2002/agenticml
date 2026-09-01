from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from experiments.registry import ExperimentRegistry
from experiments.runner import _compare, run_experiment
from experiments.schemas import ExperimentRecord
from tools.data.split import create_split
from tools.ml.automl import run_automl


def _classification_df(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


class TestHypothesisRequired:
    def test_blank_hypothesis_raises(self):
        df = _classification_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        with pytest.raises(ValueError, match="hypothesis"):
            run_experiment(
                df,
                target="y",
                task_type="binary_classification",
                split=split,
                hypothesis="   ",
                registry=ExperimentRegistry(),
                artifact_dir=None,
            )


class TestBaselineExperiment:
    def test_recorded_with_no_conclusion(self):
        df = _classification_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        registry = ExperimentRegistry()

        record = run_experiment(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            hypothesis="establish a baseline",
            registry=registry,
            cv_folds=3,
            seed=42,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        assert record.conclusion is None
        assert record.next_action is None
        assert registry.get(record.experiment_id) == record
        # created_at is a real, parseable timestamp
        datetime.fromisoformat(record.created_at)

    def test_metrics_include_test_score_when_split_has_test_set(self):
        df = _classification_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        registry = ExperimentRegistry()

        record = run_experiment(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            hypothesis="baseline",
            registry=registry,
            cv_folds=3,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        assert "test_score" in record.metrics
        assert record.primary_score == record.metrics["test_score"]

    def test_unknown_parent_raises(self):
        df = _classification_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        with pytest.raises(KeyError):
            run_experiment(
                df,
                target="y",
                task_type="binary_classification",
                split=split,
                hypothesis="a change",
                registry=ExperimentRegistry(),
                parent_experiment_id="does_not_exist",
                artifact_dir=None,
            )


class TestParentChildLinkage:
    def test_child_is_linked_and_visible_via_registry(self):
        df = _classification_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        registry = ExperimentRegistry()

        parent = run_experiment(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            hypothesis="baseline",
            registry=registry,
            cv_folds=3,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )
        child = run_experiment(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            hypothesis="try random forest instead",
            registry=registry,
            parent_experiment_id=parent.experiment_id,
            cv_folds=3,
            artifact_dir=None,
            candidate_models=["random_forest"],
        )

        assert child.parent_experiment_id == parent.experiment_id
        assert [c.experiment_id for c in registry.children(parent.experiment_id)] == [child.experiment_id]
        assert child.conclusion in {"supported", "rejected", "inconclusive"}
        assert child.next_action is not None


class TestReproducibility:
    def test_same_inputs_and_seed_yield_identical_scores(self):
        df = _classification_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        kwargs = dict(
            target="y",
            task_type="binary_classification",
            split=split,
            hypothesis="baseline",
            cv_folds=3,
            seed=42,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )
        first = run_experiment(df, registry=ExperimentRegistry(), **kwargs)
        second = run_experiment(df, registry=ExperimentRegistry(), **kwargs)

        assert first.primary_score == second.primary_score
        assert first.model == second.model
        assert first.hyperparameters == second.hyperparameters
        assert first.metrics == second.metrics


class TestFeatureBookkeeping:
    def test_features_and_exclusions_come_from_automl_profile_selection(self):
        from tools.data.profile import profile_columns

        n = 100
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "id": [f"ID{i}" for i in range(n)],
                "x": rng.normal(size=n),
                "y": [i % 2 for i in range(n)],
            }
        )
        profile = profile_columns(df)
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        record = run_experiment(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            hypothesis="baseline",
            registry=ExperimentRegistry(),
            profile=profile,
            cv_folds=3,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        assert record.features == ["x"]
        assert "id" in record.excluded_features


class TestCompareConclusionLogic:
    def _parent(self, metric="f1", primary_score=0.7):
        return ExperimentRecord(
            experiment_id="parent",
            hypothesis="baseline",
            split_strategy="stratified",
            features=["x"],
            preprocessing="p",
            model="logistic_regression",
            hyperparameters={},
            seed=42,
            metric=metric,
            metrics={"cv_mean": primary_score},
            primary_score=primary_score,
            runtime_seconds=1.0,
            artifact_path=None,
            created_at="2026-01-01T00:00:00+00:00",
        )

    def test_supported_when_higher_is_better_metric_improves(self):
        parent = self._parent(metric="f1", primary_score=0.70)
        assert _compare(parent, "f1", child_score=0.75, minimum_improvement=0.01) == "supported"

    def test_rejected_when_higher_is_better_metric_worsens(self):
        parent = self._parent(metric="f1", primary_score=0.70)
        assert _compare(parent, "f1", child_score=0.60, minimum_improvement=0.01) == "rejected"

    def test_inconclusive_when_change_is_within_noise(self):
        parent = self._parent(metric="f1", primary_score=0.70)
        assert _compare(parent, "f1", child_score=0.701, minimum_improvement=0.01) == "inconclusive"

    def test_supported_when_lower_is_better_metric_decreases(self):
        parent = self._parent(metric="rmse", primary_score=5.0)
        assert _compare(parent, "rmse", child_score=4.0, minimum_improvement=0.1) == "supported"

    def test_rejected_when_lower_is_better_metric_increases(self):
        parent = self._parent(metric="rmse", primary_score=5.0)
        assert _compare(parent, "rmse", child_score=6.0, minimum_improvement=0.1) == "rejected"

    def test_inconclusive_when_metric_name_differs_from_parent(self):
        parent = self._parent(metric="f1", primary_score=0.70)
        assert _compare(parent, "accuracy", child_score=0.95, minimum_improvement=0.01) == "inconclusive"


class TestArtifactPersistence:
    def test_artifact_path_is_forwarded_from_automl(self, tmp_path):
        df = _classification_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        record = run_experiment(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            hypothesis="baseline",
            registry=ExperimentRegistry(),
            cv_folds=3,
            artifact_dir=tmp_path,
            candidate_models=["logistic_regression"],
        )

        assert record.artifact_path is not None
        from pathlib import Path

        assert Path(record.artifact_path).exists()

    def test_matches_a_plain_run_automl_call(self):
        # run_experiment should not change AutoML's own behavior — same
        # inputs/seed produce the same winning model and score either way.
        df = _classification_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        direct = run_automl(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            cv_folds=3,
            seed=42,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )
        via_runner = run_experiment(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            hypothesis="baseline",
            registry=ExperimentRegistry(),
            cv_folds=3,
            seed=42,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        assert direct.cv_mean == via_runner.metrics["cv_mean"]
        assert direct.best_model == via_runner.model
        assert direct.best_params == via_runner.hyperparameters
