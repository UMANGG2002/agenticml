import numpy as np
import pandas as pd
import pytest

from tools.data.leakage import detect_leakage


def _finding_types(result):
    return [f.type for f in result.findings]


class TestTargetDerivedAndCorrelation:
    def test_raises_when_target_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError):
            detect_leakage(df, target="does_not_exist")

    def test_exact_duplicate_of_target_is_critical(self):
        target = [0, 1] * 50
        df = pd.DataFrame({"leaky": target, "target": target, "noise": range(100)})
        result = detect_leakage(df, target="target")
        finding = next(f for f in result.findings if f.type == "target_derived_feature")
        assert finding.severity == "critical"
        assert "leaky" in finding.columns

    def test_near_perfect_linear_correlation_is_critical(self):
        rng = np.random.default_rng(0)
        target = rng.normal(size=500)
        feature = target * 2.0  # perfect linear transform -> corr == 1.0
        df = pd.DataFrame({"scaled_target": feature, "target": target})
        result = detect_leakage(df, target="target")
        finding = next(f for f in result.findings if f.type == "target_derived_feature")
        assert finding.metric_value == pytest.approx(1.0, abs=1e-6)

    def test_moderately_correlated_feature_is_suspicious_not_critical(self):
        rng = np.random.default_rng(1)
        n = 2000
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        r = 0.96
        feature = r * x + np.sqrt(1 - r**2) * y
        df = pd.DataFrame({"feature_moderate": feature, "target": x})
        result = detect_leakage(df, target="target", correlation_threshold=0.9, near_perfect_threshold=0.995)
        finding = next(f for f in result.findings if f.columns == ["feature_moderate"])
        assert finding.type == "suspicious_correlation"
        assert finding.severity == "high"
        assert 0.9 <= abs(finding.metric_value) < 0.995

    def test_feature_nonnull_only_for_one_target_class_does_not_warn_or_crash(self):
        # Real-world pattern (e.g. Titanic's "body" column, only recorded for
        # non-survivors): among the rows where the feature is non-null, the
        # target has zero variance, making Pearson correlation undefined.
        import warnings

        n = 100
        target = [0] * 60 + [1] * 40
        feature = list(range(60)) + [None] * 40  # non-null only when target == 0
        df = pd.DataFrame({"feature": feature, "target": target})
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = detect_leakage(df, target="target")
        assert "feature" not in [c for f in result.findings for c in f.columns]

    def test_uncorrelated_feature_produces_no_finding(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {
                "unrelated": rng.normal(size=300),
                "target": rng.normal(size=300),
            }
        )
        result = detect_leakage(df, target="target")
        assert _finding_types(result) == []

    def test_categorical_feature_perfectly_predicting_target(self):
        target = ["A", "B"] * 50
        feature = ["cat1" if t == "A" else "cat2" for t in target]
        df = pd.DataFrame({"mapped_category": feature, "target": target})
        result = detect_leakage(df, target="target")
        finding = next(f for f in result.findings if f.type == "target_derived_feature")
        assert finding.metric_name == "purity"
        assert finding.metric_value == pytest.approx(1.0)

    def test_high_cardinality_categorical_feature_not_flagged_as_leakage(self):
        # Near-unique categorical features would trivially reach ~100% purity;
        # they should be excluded from the purity check, not flagged.
        n = 200
        df = pd.DataFrame(
            {
                "near_unique_id": [f"id_{i}" for i in range(n)],
                "target": ["A", "B"] * (n // 2),
            }
        )
        result = detect_leakage(df, target="target")
        assert "near_unique_id" not in [c for f in result.findings for c in f.columns]


class TestPostOutcome:
    def test_skipped_when_no_metadata_supplied(self):
        df = pd.DataFrame({"a": range(10), "target": range(10)})
        result = detect_leakage(df, target="target")
        assert any("post_outcome_feature" in s for s in result.checks_skipped)
        assert "post_outcome_feature" not in _finding_types(result)

    def test_flags_declared_post_outcome_columns(self):
        df = pd.DataFrame(
            {
                "income_after_application": range(10),
                "age": range(10),
                "target": range(10),
            }
        )
        result = detect_leakage(df, target="target", post_outcome_columns=["income_after_application"])
        finding = next(f for f in result.findings if f.type == "post_outcome_feature")
        assert finding.severity == "critical"
        assert finding.columns == ["income_after_application"]

    def test_has_critical_issues_true_when_post_outcome_present(self):
        df = pd.DataFrame({"income_after_application": range(10), "target": range(10)})
        result = detect_leakage(df, target="target", post_outcome_columns=["income_after_application"])
        assert result.has_critical_issues is True


class TestDuplicateAndNearDuplicate:
    def test_duplicate_rows_flagged(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "target": [0, 0, 1, 1]})
        result = detect_leakage(df, target="target")
        finding = next(f for f in result.findings if f.type == "duplicate_row_contamination")
        assert finding.metric_value > 0

    def test_no_duplicate_finding_when_all_rows_unique(self):
        df = pd.DataFrame({"a": range(50), "target": range(50)})
        result = detect_leakage(df, target="target")
        assert "duplicate_row_contamination" not in _finding_types(result)

    def test_conflicting_labels_on_identical_features_flagged(self):
        df = pd.DataFrame(
            {
                "a": [1, 1, 2, 3, 4],
                "b": ["x", "x", "y", "z", "w"],
                "target": [0, 1, 0, 1, 0],
            }
        )
        result = detect_leakage(df, target="target")
        finding = next(f for f in result.findings if f.type == "near_duplicate_contamination")
        assert finding.severity == "high"
        assert finding.metric_value == 2  # both conflicting rows counted

    def test_no_near_duplicate_finding_when_consistent(self):
        df = pd.DataFrame({"a": [1, 1, 2], "target": [0, 0, 1]})
        result = detect_leakage(df, target="target")
        assert "near_duplicate_contamination" not in _finding_types(result)


class TestTrainTestContamination:
    def test_skipped_when_only_one_side_supplied(self):
        df = pd.DataFrame({"a": range(10), "target": range(10)})
        result = detect_leakage(df, target="target", train_df=df)
        assert any("train_test_contamination" in s for s in result.checks_skipped)

    def test_flags_overlap_between_train_and_test(self):
        train_df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
        test_df = pd.DataFrame({"a": [3, 4, 5, 6], "target": [0, 1, 1, 0]})
        df = pd.concat([train_df, test_df], ignore_index=True)
        result = detect_leakage(df, target="target", train_df=train_df, test_df=test_df)
        finding = next(f for f in result.findings if f.type == "train_test_contamination")
        assert finding.severity == "critical"
        assert finding.metric_value == pytest.approx(50.0)  # 2 of 4 test rows overlap

    def test_no_finding_when_splits_are_disjoint(self):
        train_df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        test_df = pd.DataFrame({"a": [3, 4], "target": [0, 1]})
        df = pd.concat([train_df, test_df], ignore_index=True)
        result = detect_leakage(df, target="target", train_df=train_df, test_df=test_df)
        assert "train_test_contamination" not in _finding_types(result)


class TestChecksPerformedBookkeeping:
    def test_checks_performed_and_skipped_are_disjoint(self):
        df = pd.DataFrame({"a": range(10), "target": range(10)})
        result = detect_leakage(df, target="target")
        performed_names = set(result.checks_performed)
        skipped_names = {s.split(" ")[0] for s in result.checks_skipped}
        assert performed_names.isdisjoint(skipped_names)

    def test_has_critical_issues_false_on_clean_data(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"a": rng.normal(size=200), "target": rng.normal(size=200)})
        result = detect_leakage(df, target="target")
        assert result.has_critical_issues is False
