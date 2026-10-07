import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.data.profile import profile_columns
from tools.data.schemas import SplitResult
from tools.data.split import create_split
from tools.ml.automl import run_automl


class TestSplitLeakageGuards:
    def test_overlapping_train_test_indices_raises(self):
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        split = SplitResult(strategy="random", train_indices=[0, 1, 2, 3, 4], test_indices=[4, 5, 6], seed=42)
        with pytest.raises(ValueError, match="overlap"):
            run_automl(df, target="y", task_type="binary_classification", split=split, artifact_dir=None)

    def test_overlapping_train_validation_indices_raises(self):
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        split = SplitResult(
            strategy="random",
            train_indices=[0, 1, 2, 3, 4],
            validation_indices=[3, 4],
            test_indices=[5, 6],
            seed=42,
        )
        with pytest.raises(ValueError, match="overlap"):
            run_automl(df, target="y", task_type="binary_classification", split=split, artifact_dir=None)

    def test_selection_is_unaffected_by_corrupting_test_rows(self):
        rng = np.random.default_rng(2)
        n = 300
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = (x1 + x2 > 0).astype(int)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        kwargs = dict(
            target="y",
            task_type="binary_classification",
            split=split,
            cv_folds=3,
            seed=42,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )
        original = run_automl(df, **kwargs)

        corrupted = df.copy()
        # Flip labels and scramble features only on the held-out test rows.
        corrupted.loc[split.test_indices, "y"] = 1 - corrupted.loc[split.test_indices, "y"]
        corrupted.loc[split.test_indices, "x1"] = rng.normal(size=len(split.test_indices)) * 1000

        result = run_automl(corrupted, **kwargs)

        assert result.cv_mean == original.cv_mean
        assert result.cv_std == original.cv_std
        assert result.score == original.score
        assert result.best_params == original.best_params
        assert result.best_model == original.best_model
        # The test set itself changed, so its score is allowed to differ.
        assert result.test_score != original.test_score


class TestGoldenPaths:
    def test_binary_classification(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 300
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = (x1 + x2 > 0).astype(int)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        result = run_automl(
            df, target="y", task_type="binary_classification", split=split, cv_folds=3, seed=42, artifact_dir=tmp_path
        )

        assert result.metric == "f1"
        assert result.score > 0.85
        assert result.test_score is not None and result.test_score > 0.8
        assert result.experiment_id.startswith("exp_")
        assert result.artifact_path is not None
        assert Path(result.artifact_path).exists()

        with open(result.artifact_path) as f:
            artifact = json.load(f)
        assert artifact["model_name"] == result.best_model
        assert artifact["params"] == result.best_params
        assert artifact["feature_columns"] == result.feature_columns

    def test_regression(self, tmp_path):
        rng = np.random.default_rng(1)
        n = 300
        x = rng.normal(size=n)
        y = 3 * x + rng.normal(scale=0.1, size=n)
        df = pd.DataFrame({"x": x, "target": y})
        split = create_split(df, "random", test_size=0.2, seed=42)

        result = run_automl(
            df,
            target="target",
            task_type="regression",
            split=split,
            cv_folds=3,
            seed=42,
            artifact_dir=tmp_path,
            candidate_models=["linear_regression"],
        )

        assert result.metric == "rmse"
        assert result.best_model == "linear_regression"
        assert result.score < 0.5
        assert result.test_score is not None and result.test_score < 0.5

    def test_multiclass_uses_f1_macro_by_default(self):
        rng = np.random.default_rng(6)
        n = 300
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = np.select([x1 + x2 < -0.5, x1 + x2 < 0.5], [0, 1], default=2)
        df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        result = run_automl(
            df,
            target="y",
            task_type="multiclass_classification",
            split=split,
            cv_folds=3,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        assert result.metric == "f1_macro"


class TestFeatureSelection:
    def test_profile_excludes_identifier_and_constant_columns(self):
        n = 100
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "id": [f"ID{i}" for i in range(n)],
                "constant_col": [1] * n,
                "x": rng.normal(size=n),
                "y": [i % 2 for i in range(n)],
            }
        )
        profile = profile_columns(df)
        split = create_split(df, "random", test_size=0.2, seed=42)

        result = run_automl(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            profile=profile,
            cv_folds=3,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        assert "id" in result.excluded_columns
        assert "constant_col" in result.excluded_columns
        assert result.feature_columns == ["x"]

    def test_no_usable_features_raises(self):
        n = 50
        df = pd.DataFrame({"id": [f"ID{i}" for i in range(n)], "y": [i % 2 for i in range(n)]})
        profile = profile_columns(df)
        split = create_split(df, "random", test_size=0.2, seed=42)

        with pytest.raises(ValueError, match="no usable feature columns"):
            run_automl(
                df, target="y", task_type="binary_classification", split=split, profile=profile, artifact_dir=None
            )

    def test_explicit_feature_columns_are_trusted_as_is(self):
        n = 100
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "id": [f"ID{i}" for i in range(n)],
                "x": rng.normal(size=n),
                "y": [i % 2 for i in range(n)],
            }
        )
        split = create_split(df, "random", test_size=0.2, seed=42)

        result = run_automl(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            feature_columns=["id", "x"],
            cv_folds=3,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        assert set(result.feature_columns) == {"id", "x"}
        assert result.excluded_columns == []


class TestRobustness:
    def test_handles_missing_values_via_imputation(self):
        n = 200
        rng = np.random.default_rng(4)
        x1 = rng.normal(size=n)
        x1[::10] = np.nan
        cat = rng.choice(["a", "b", "c"], size=n).astype(object)
        cat[::15] = None
        y = rng.integers(0, 2, size=n)
        df = pd.DataFrame({"x1": x1, "cat": cat, "y": y})
        split = create_split(df, "random", test_size=0.2, seed=42)

        result = run_automl(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            cv_folds=3,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        assert result.cv_mean is not None

    def test_validation_score_computed_when_split_has_validation_indices(self):
        rng = np.random.default_rng(5)
        n = 400
        x1 = rng.normal(size=n)
        y = (x1 > 0).astype(int)
        df = pd.DataFrame({"x1": x1, "y": y})
        split = create_split(df, "stratified", test_size=0.2, validation_size=0.1, target="y", seed=42)

        result = run_automl(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            cv_folds=3,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        assert result.validation_score is not None
        assert result.test_score is not None


class TestValidation:
    def test_unknown_target_raises(self):
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        split = create_split(df, "random", test_size=0.2, seed=42)
        with pytest.raises(ValueError, match="not found"):
            run_automl(df, target="does_not_exist", task_type="binary_classification", split=split, artifact_dir=None)

    def test_unknown_task_type_raises(self):
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        split = create_split(df, "random", test_size=0.2, seed=42)
        with pytest.raises(ValueError, match="task_type"):
            run_automl(df, target="y", task_type="not_a_real_task", split=split, artifact_dir=None)

    def test_unknown_metric_raises(self):
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        split = create_split(df, "random", test_size=0.2, seed=42)
        with pytest.raises(ValueError, match="metric"):
            run_automl(
                df, target="y", task_type="binary_classification", split=split, metric="not_a_metric", artifact_dir=None
            )

    def test_unknown_candidate_models_raises(self):
        df = pd.DataFrame({"x": range(10), "y": [0, 1] * 5})
        split = create_split(df, "random", test_size=0.2, seed=42)
        with pytest.raises(ValueError, match="none of"):
            run_automl(
                df,
                target="y",
                task_type="binary_classification",
                split=split,
                candidate_models=["not_a_real_model"],
                artifact_dir=None,
            )


class TestCandidateBookkeeping:
    def test_candidates_cover_the_full_param_grid(self):
        n = 200
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.integers(0, 2, size=n)})
        split = create_split(df, "random", test_size=0.2, seed=42)

        result = run_automl(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            cv_folds=3,
            artifact_dir=None,
            candidate_models=["logistic_regression"],
        )

        assert len(result.candidates) == 2  # two param sets in the default logistic_regression grid
        assert all(c.model_name == "logistic_regression" for c in result.candidates)

    def test_best_candidate_matches_top_of_ranking(self):
        n = 200
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.integers(0, 2, size=n)})
        split = create_split(df, "random", test_size=0.2, seed=42)

        result = run_automl(
            df, target="y", task_type="binary_classification", split=split, cv_folds=3, artifact_dir=None
        )

        best_among_candidates = max(c.cv_mean for c in result.candidates)
        assert result.cv_mean == pytest.approx(best_among_candidates)
