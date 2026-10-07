import numpy as np
import pandas as pd
import pytest

from experiments.artifacts import load_artifact, refit_from_experiment, refit_from_registry
from experiments.registry import ExperimentRegistry
from experiments.runner import run_experiment
from tools.data.split import create_split
from tools.ml.automl import evaluate_holdout


def _classification_df(seed=0, n=300):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


class TestLoadArtifact:
    def test_reads_expected_keys(self, tmp_path):
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
            seed=42,
            artifact_dir=tmp_path,
            candidate_models=["logistic_regression"],
        )

        metadata = load_artifact(record.artifact_path)
        assert metadata["model_name"] == "logistic_regression"
        assert metadata["target"] == "y"
        assert metadata["feature_columns"] == record.features
        assert metadata["seed"] == 42


class TestRefitReproducesTheOriginalScore:
    def test_classification_refit_matches_recorded_test_score(self, tmp_path):
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
            seed=42,
            artifact_dir=tmp_path,
        )

        refit = refit_from_experiment(record, df, split)
        test_df = df.loc[split.test_indices]
        score = evaluate_holdout(refit, test_df[record.features], test_df["y"], record.metric)

        assert score == pytest.approx(record.metrics["test_score"])

    def test_regression_refit_matches_recorded_test_score(self, tmp_path):
        rng = np.random.default_rng(1)
        n = 300
        x = rng.normal(size=n)
        y = 3 * x + rng.normal(scale=0.1, size=n)
        df = pd.DataFrame({"x": x, "target_col": y})
        split = create_split(df, "random", test_size=0.2, seed=42)
        registry = ExperimentRegistry()

        record = run_experiment(
            df,
            target="target_col",
            task_type="regression",
            split=split,
            hypothesis="baseline",
            registry=registry,
            cv_folds=3,
            seed=42,
            artifact_dir=tmp_path,
            candidate_models=["linear_regression"],
        )

        refit = refit_from_experiment(record, df, split)
        test_df = df.loc[split.test_indices]
        score = evaluate_holdout(refit, test_df[record.features], test_df["target_col"], record.metric)

        assert score == pytest.approx(record.metrics["test_score"])

    def test_refit_is_deterministic_across_calls(self, tmp_path):
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
            seed=42,
            artifact_dir=tmp_path,
            candidate_models=["random_forest"],
        )

        test_df = df.loc[split.test_indices]
        first = refit_from_experiment(record, df, split).predict(test_df[record.features])
        second = refit_from_experiment(record, df, split).predict(test_df[record.features])

        assert list(first) == list(second)


class TestErrorHandling:
    def test_raises_when_experiment_has_no_artifact(self):
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
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        with pytest.raises(ValueError, match="no saved artifact"):
            refit_from_experiment(record, df, split)


class TestRefitFromRegistry:
    def test_looks_up_and_refits_by_experiment_id(self, tmp_path):
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
            seed=42,
            artifact_dir=tmp_path,
            candidate_models=["logistic_regression"],
        )

        refit = refit_from_registry(registry, record.experiment_id, df, split)
        test_df = df.loc[split.test_indices]
        score = evaluate_holdout(refit, test_df[record.features], test_df["y"], record.metric)
        assert score == pytest.approx(record.metrics["test_score"])

    def test_unknown_experiment_id_raises(self, tmp_path):
        df = _classification_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        with pytest.raises(KeyError):
            refit_from_registry(ExperimentRegistry(), "does_not_exist", df, split)
