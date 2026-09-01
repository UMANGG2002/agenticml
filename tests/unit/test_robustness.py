import numpy as np
import pandas as pd

from tools.data.split import create_split
from tools.validation.robustness import run_robustness_checks


def _stable_df(n=400, seed=0):
    """A cleanly separable dataset — the model should be robust on this."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def _check(report, name):
    return next((c for c in report.checks if c.name == name), None)


def _base_kwargs(df, split, reference_score, features=("x1", "x2")):
    return dict(
        target="y",
        task_type="binary_classification",
        split=split,
        reference_score=reference_score,
        metric="f1",
        feature_columns=list(features),
        model_name="logistic_regression",
        seeds=[1, 7, 13],
        cv_folds=3,
    )


class TestMultiSeed:
    def test_stable_model_passes_multi_seed(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(df, **_base_kwargs(df, split, reference_score=0.95))

        check = _check(report, "multi_seed")
        assert check is not None
        assert check.passed is True
        assert check.values["std"] <= 0.03
        assert set(check.values) >= {"mean", "std", "min", "max"}

    def test_noisy_model_shows_real_cross_split_variance(self):
        # Multi-seed must re-split, not just re-fit: a deterministic
        # estimator ignores random_state, so a fixed split would report
        # std == 0 no matter how unstable the model actually is.
        n = 200
        rng = np.random.default_rng(9)
        x1 = rng.normal(size=n)
        noise = rng.normal(scale=3.0, size=n)
        y = ((x1 + noise) > 0).astype(int)
        df = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "y": y})
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        report = run_robustness_checks(df, **_base_kwargs(df, split, reference_score=0.55))

        check = _check(report, "multi_seed")
        assert check.values["std"] > 0.0
        assert check.values["min"] < check.values["max"]

    def test_tight_threshold_fails_and_records_severity(self):
        n = 200
        rng = np.random.default_rng(9)
        x1 = rng.normal(size=n)
        noise = rng.normal(scale=3.0, size=n)
        y = ((x1 + noise) > 0).astype(int)
        df = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "y": y})
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        report = run_robustness_checks(
            df, seed_std_threshold=0.0, **_base_kwargs(df, split, reference_score=0.55)
        )

        check = _check(report, "multi_seed")
        assert check.passed is False
        assert check.severity == "medium"


class TestCvVariance:
    def test_skipped_when_cv_std_not_supplied(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(df, **_base_kwargs(df, split, reference_score=0.95))

        assert _check(report, "cv_variance") is None
        assert any("cv_variance" in s for s in report.checks_skipped)

    def test_low_cv_std_passes(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(df, cv_std=0.01, **_base_kwargs(df, split, reference_score=0.95))

        check = _check(report, "cv_variance")
        assert check.passed is True

    def test_high_cv_std_fails(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(df, cv_std=0.25, **_base_kwargs(df, split, reference_score=0.95))

        check = _check(report, "cv_variance")
        assert check.passed is False
        assert "above" in check.evidence


class TestTemporalValidation:
    def test_skipped_when_no_time_column_supplied(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(df, **_base_kwargs(df, split, reference_score=0.95))

        assert _check(report, "temporal_validation") is None
        assert any("temporal_validation" in s for s in report.checks_skipped)

    def test_skipped_when_named_column_missing_from_dataset(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(
            df, time_column="not_a_real_column", **_base_kwargs(df, split, reference_score=0.95)
        )

        assert _check(report, "temporal_validation") is None
        assert any("not in dataset" in s for s in report.checks_skipped)

    def test_detects_temporal_degradation_when_relationship_drifts(self):
        # The x->y relationship flips halfway through time, so a model
        # trained only on the early period should fail on the later one.
        n = 400
        rng = np.random.default_rng(1)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        early = (x1 + x2 > 0).astype(int)
        late = (x1 + x2 <= 0).astype(int)
        y = np.concatenate([early[: n // 2], late[n // 2 :]])
        df = pd.DataFrame(
            {"x1": x1, "x2": x2, "y": y, "event_time": pd.date_range("2024-01-01", periods=n, freq="D")}
        )
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        report = run_robustness_checks(
            df, time_column="event_time", **_base_kwargs(df, split, reference_score=0.95)
        )

        check = _check(report, "temporal_validation")
        assert check is not None
        assert check.passed is False
        assert check.values["degradation"] > 0.05

    def test_stable_relationship_passes_temporal_validation(self):
        n = 400
        df = _stable_df(n=n)
        df["event_time"] = pd.date_range("2024-01-01", periods=n, freq="D")
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        report = run_robustness_checks(
            df, time_column="event_time", **_base_kwargs(df, split, reference_score=0.93)
        )

        check = _check(report, "temporal_validation")
        assert check.passed is True


class TestGroupValidation:
    def test_skipped_when_no_group_column_supplied(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(df, **_base_kwargs(df, split, reference_score=0.95))

        assert _check(report, "group_validation") is None
        assert any("group_validation" in s for s in report.checks_skipped)

    def test_runs_when_group_column_supplied(self):
        n = 400
        df = _stable_df(n=n)
        df["customer_id"] = [i % 40 for i in range(n)]
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        report = run_robustness_checks(
            df, group_column="customer_id", **_base_kwargs(df, split, reference_score=0.93)
        )

        check = _check(report, "group_validation")
        assert check is not None
        assert "score" in check.values


class TestNoiseSensitivity:
    def test_runs_on_numeric_features(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(df, **_base_kwargs(df, split, reference_score=0.93))

        check = _check(report, "noise_sensitivity")
        assert check is not None
        assert check.values["noise_scale"] == 0.1

    def test_skipped_when_no_numeric_features(self):
        n = 200
        rng = np.random.default_rng(2)
        cat = rng.choice(["a", "b", "c"], size=n)
        y = (cat == "a").astype(int)
        df = pd.DataFrame({"cat": cat, "y": y})
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)

        report = run_robustness_checks(
            df,
            target="y",
            task_type="binary_classification",
            split=split,
            reference_score=0.9,
            metric="f1",
            feature_columns=["cat"],
            model_name="logistic_regression",
            seeds=[1, 7],
            cv_folds=3,
        )

        assert _check(report, "noise_sensitivity") is None
        assert any("noise_sensitivity" in s for s in report.checks_skipped)

    def test_heavy_noise_degrades_score(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(
            df, noise_scale=5.0, **_base_kwargs(df, split, reference_score=0.95)
        )

        check = _check(report, "noise_sensitivity")
        assert check.passed is False
        assert check.values["degradation"] > 0.05


class TestReportBookkeeping:
    def test_n_failed_matches_failed_checks(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(df, cv_std=0.25, **_base_kwargs(df, split, reference_score=0.95))

        assert report.n_failed == len(report.failed)
        assert report.n_failed == sum(1 for c in report.checks if not c.passed)

    def test_metric_and_reference_score_are_echoed(self):
        df = _stable_df()
        split = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        report = run_robustness_checks(df, **_base_kwargs(df, split, reference_score=0.93))

        assert report.metric == "f1"
        assert report.reference_score == 0.93

    def test_lower_is_better_metric_degradation_direction(self):
        # For rmse, a HIGHER observed value is worse. Using a deliberately
        # optimistic reference should register as degradation, not improvement.
        n = 300
        rng = np.random.default_rng(3)
        x = rng.normal(size=n)
        y = 3 * x + rng.normal(scale=1.0, size=n)
        df = pd.DataFrame({"x": x, "y": y})
        split = create_split(df, "random", test_size=0.2, seed=42)

        report = run_robustness_checks(
            df,
            target="y",
            task_type="regression",
            split=split,
            reference_score=0.0,  # impossibly good rmse
            metric="rmse",
            feature_columns=["x"],
            model_name="linear_regression",
            seeds=[1, 7],
            cv_folds=3,
        )

        check = _check(report, "noise_sensitivity")
        assert check.values["degradation"] > 0  # worse than the reference
