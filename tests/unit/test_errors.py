import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools.ml.errors import analyze_classification_errors, analyze_regression_errors


def _fit_classifier(df, feature_columns, target):
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
    pipeline.fit(df[feature_columns], df[target])
    return pipeline


def _fit_regressor(df, feature_columns, target):
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    pipeline.fit(df[feature_columns], df[target])
    return pipeline


class TestClassificationErrors:
    def test_perfectly_separable_data_has_no_errors(self):
        n = 200
        rng = np.random.default_rng(0)
        x = rng.normal(size=n)
        y = (x > 0).astype(int)  # trivially separable, LogisticRegression should nail it
        df = pd.DataFrame({"x": x, "y": y})
        pipeline = _fit_classifier(df, ["x"], "y")

        result = analyze_classification_errors(pipeline, df, "y", ["x"])

        assert result.n_rows == n
        assert sum(m.support for m in result.per_class_metrics) == n
        for m in result.per_class_metrics:
            assert m.precision == 1.0
            assert m.recall == 1.0
        assert result.false_positives == []
        assert result.false_negatives == []
        # off-diagonal cells are all zero
        for i, row in enumerate(result.confusion_matrix.matrix):
            for j, count in enumerate(row):
                if i != j:
                    assert count == 0

    def test_noisy_data_produces_false_positives_and_negatives(self):
        n = 300
        rng = np.random.default_rng(1)
        x = rng.normal(size=n)
        noise = rng.normal(scale=2.0, size=n)
        y = ((x + noise) > 0).astype(int)
        df = pd.DataFrame({"x": x, "y": y})
        pipeline = _fit_classifier(df, ["x"], "y")

        result = analyze_classification_errors(pipeline, df, "y", ["x"], max_examples=3)

        assert len(result.false_positives) <= 3
        assert len(result.false_negatives) <= 3
        assert (len(result.false_positives) + len(result.false_negatives)) > 0
        for example in result.false_positives + result.false_negatives:
            assert example.true_value != example.predicted_value

    def test_threshold_curve_present_for_binary_task(self):
        n = 200
        rng = np.random.default_rng(2)
        x = rng.normal(size=n)
        y = (x > 0).astype(int)
        df = pd.DataFrame({"x": x, "y": y})
        pipeline = _fit_classifier(df, ["x"], "y")

        result = analyze_classification_errors(pipeline, df, "y", ["x"])

        assert len(result.threshold_curve) == 9
        thresholds = [t.threshold for t in result.threshold_curve]
        assert thresholds == sorted(thresholds)

    def test_threshold_curve_empty_for_multiclass_task(self):
        n = 300
        rng = np.random.default_rng(3)
        x = rng.normal(size=n)
        y = np.select([x < -0.5, x < 0.5], [0, 1], default=2)
        df = pd.DataFrame({"x": x, "y": y})
        pipeline = _fit_classifier(df, ["x"], "y")

        result = analyze_classification_errors(pipeline, df, "y", ["x"])

        assert result.threshold_curve == []
        assert len(result.per_class_metrics) == 3

    def test_subgroup_metrics_and_small_group_warning(self):
        n = 200
        rng = np.random.default_rng(4)
        x = rng.normal(size=n)
        y = (x > 0).astype(int)
        group = ["big"] * 190 + ["tiny"] * 10
        df = pd.DataFrame({"x": x, "y": y, "region": group})
        pipeline = _fit_classifier(df, ["x"], "y")

        result = analyze_classification_errors(pipeline, df, "y", ["x"], subgroup_columns=["region"])

        by_group = {m.group: m for m in result.subgroup_metrics}
        assert by_group["big"].n == 190
        assert by_group["big"].warning is None
        assert by_group["tiny"].n == 10
        assert by_group["tiny"].warning == "small sample size"

    def test_values_are_json_safe_native_types(self):
        n = 300
        rng = np.random.default_rng(1)
        x = rng.normal(size=n)
        noise = rng.normal(scale=2.0, size=n)
        y = ((x + noise) > 0).astype(int)
        df = pd.DataFrame({"x": x, "y": y})
        pipeline = _fit_classifier(df, ["x"], "y")

        result = analyze_classification_errors(pipeline, df, "y", ["x"])

        for example in result.false_positives + result.false_negatives:
            assert isinstance(example.true_value, (int, float, str, type(None)))
            assert not isinstance(example.true_value, np.generic)


class TestRegressionErrors:
    def test_perfect_fit_has_near_zero_residuals(self):
        n = 200
        rng = np.random.default_rng(0)
        x = rng.normal(size=n)
        y = 3 * x + 1  # exact linear relationship, no noise
        df = pd.DataFrame({"x": x, "y": y})
        pipeline = _fit_regressor(df, ["x"], "y")

        result = analyze_regression_errors(pipeline, df, "y", ["x"])

        assert abs(result.residual_mean) < 1e-6
        assert result.residual_std < 1e-6

    def test_extreme_errors_are_sorted_by_largest_absolute_residual(self):
        n = 200
        rng = np.random.default_rng(5)
        x = rng.normal(size=n)
        y = 3 * x + 1
        df = pd.DataFrame({"x": x, "y": y})
        # inject one huge outlier
        df.loc[0, "y"] = df.loc[0, "y"] + 1000
        pipeline = _fit_regressor(df, ["x"], "y")

        result = analyze_regression_errors(pipeline, df, "y", ["x"], max_examples=5)

        assert result.extreme_errors[0].index == 0
        residuals = [abs(e.residual) for e in result.extreme_errors]
        assert residuals == sorted(residuals, reverse=True)

    def test_subgroup_rmse_reflects_group_specific_noise(self):
        n = 300
        rng = np.random.default_rng(6)
        x = rng.normal(size=n)
        group = np.array(["clean"] * 150 + ["noisy"] * 150)
        noise = np.concatenate([np.zeros(150), rng.normal(scale=10.0, size=150)])
        y = 3 * x + 1 + noise
        df = pd.DataFrame({"x": x, "y": y, "segment": group})
        pipeline = _fit_regressor(df, ["x"], "y")

        result = analyze_regression_errors(pipeline, df, "y", ["x"], subgroup_columns=["segment"])

        by_group = {m.group: m for m in result.subgroup_metrics}
        assert by_group["noisy"].metric_value > by_group["clean"].metric_value

    def test_n_rows_matches_input(self):
        n = 150
        rng = np.random.default_rng(7)
        x = rng.normal(size=n)
        df = pd.DataFrame({"x": x, "y": 2 * x})
        pipeline = _fit_regressor(df, ["x"], "y")

        result = analyze_regression_errors(pipeline, df, "y", ["x"])
        assert result.n_rows == n
