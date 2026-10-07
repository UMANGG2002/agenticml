"""Unit tests for tools/validation/critic.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from experiments.schemas import ExperimentRecord
from tools.validation.critic import (
    CriticIssue,
    CriticReview,
    _check_cv_variance,
    _check_leakage,
    _check_metric_appropriateness,
    _check_overfitting,
    _check_reproducibility,
    _check_seed_sensitivity,
    _check_subgroup_disparities,
    _compute_confidence,
    _derive_verdict,
    _std,
    run_critic,
)


# ---------------------------------------------------------------------------
# Helpers — mock factories
# ---------------------------------------------------------------------------


def _make_experiment(**overrides) -> ExperimentRecord:
    """Return an ExperimentRecord with sensible defaults.

    Any keyword argument overrides the corresponding field.
    """
    defaults = dict(
        experiment_id="exp-001",
        hypothesis="Test hypothesis",
        objective="Test objective",
        dataset_version_id="ds-v1",
        split_strategy="stratified_kfold",
        features=["f1", "f2", "f3"],
        excluded_features=[],
        preprocessing="standard_scaler",
        model="random_forest",
        hyperparameters={"n_estimators": 100},
        seed=42,
        metric="f1",
        metrics={"train_score": 0.90, "test_score": 0.88, "cv_mean": 0.87, "cv_std": 0.01},
        primary_score=0.88,
        runtime_seconds=12.5,
        artifact_path="/artifacts/exp-001",
        conclusion="supported",
        next_action="deploy",
        created_at="2025-01-01T00:00:00",
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


def _make_leakage(**attrs) -> SimpleNamespace:
    """Return a mock LeakageDetectionResult."""
    defaults = dict(leaked_features=[], duplicate_rows=0, target_leakage=False)
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def _make_robustness(**attrs) -> SimpleNamespace:
    """Return a mock RobustnessReport."""
    defaults = dict(seed_std=None, seed_scores=None, passed=True)
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


# ===================================================================
# CriticIssue / CriticReview model tests
# ===================================================================


class TestModels:
    def test_critic_issue_creation(self):
        issue = CriticIssue(
            type="overfitting",
            severity="high",
            evidence="Train-test gap too large.",
            recommendation="Add regularization.",
        )
        assert issue.type == "overfitting"
        assert issue.severity == "high"

    def test_critic_issue_rejects_invalid_severity(self):
        with pytest.raises(Exception):
            CriticIssue(
                type="test",
                severity="extreme",  # not in the Literal
                evidence="e",
                recommendation="r",
            )

    def test_critic_review_creation(self):
        review = CriticReview(
            experiment_id="exp-001",
            verdict="approve",
            confidence=0.95,
            issues=[],
            required_next_actions=[],
            summary="All good.",
        )
        assert review.verdict == "approve"
        assert review.confidence == 0.95
        assert review.issues == []

    def test_critic_review_confidence_bounds(self):
        with pytest.raises(Exception):
            CriticReview(
                experiment_id="exp-001",
                verdict="approve",
                confidence=1.5,  # exceeds max
                issues=[],
                required_next_actions=[],
                summary="s",
            )

    def test_critic_review_rejects_invalid_verdict(self):
        with pytest.raises(Exception):
            CriticReview(
                experiment_id="exp-001",
                verdict="maybe",
                confidence=0.5,
                issues=[],
                required_next_actions=[],
                summary="s",
            )


# ===================================================================
# run_critic — integration-level tests
# ===================================================================


class TestRunCriticClean:
    """Experiment with no problems should be approved."""

    def test_clean_experiment_approved(self):
        exp = _make_experiment()
        leakage = _make_leakage()
        robustness = _make_robustness(seed_scores=[0.87, 0.88, 0.87], seed_std=0.005)

        review = run_critic(
            experiment=exp,
            leakage_result=leakage,
            robustness_report=robustness,
        )

        assert review.verdict == "approve"
        assert review.experiment_id == "exp-001"
        # No critical or high issues
        high_or_critical = [i for i in review.issues if i.severity in ("critical", "high")]
        assert len(high_or_critical) == 0

    def test_clean_experiment_high_confidence(self):
        exp = _make_experiment()
        leakage = _make_leakage()
        robustness = _make_robustness(seed_scores=[0.87, 0.88, 0.87], seed_std=0.005)

        review = run_critic(
            experiment=exp,
            leakage_result=leakage,
            robustness_report=robustness,
        )

        # With leakage + robustness supplied (no error_analysis), confidence
        # loses 0.10 for missing error_analysis but nothing else severe.
        assert review.confidence >= 0.80


class TestRunCriticLeakage:
    """Leakage should produce a reject verdict."""

    def test_target_leakage_rejects(self):
        exp = _make_experiment()
        leakage = _make_leakage(target_leakage=True)
        robustness = _make_robustness(seed_scores=[0.99, 0.99], seed_std=0.001)

        review = run_critic(
            experiment=exp,
            leakage_result=leakage,
            robustness_report=robustness,
        )

        assert review.verdict == "reject"
        types = {i.type for i in review.issues}
        assert "leakage_risk" in types

    def test_leaked_features_produces_issue(self):
        exp = _make_experiment()
        leakage = _make_leakage(leaked_features=["future_price", "label_encoded"])

        review = run_critic(experiment=exp, leakage_result=leakage)

        leakage_issues = [i for i in review.issues if i.type == "leakage_risk"]
        assert len(leakage_issues) >= 1
        assert "2" in leakage_issues[0].evidence  # 2 features

    def test_many_leaked_features_high_severity(self):
        exp = _make_experiment()
        leakage = _make_leakage(leaked_features=["a", "b", "c", "d"])

        review = run_critic(experiment=exp, leakage_result=leakage)

        leakage_issues = [i for i in review.issues if i.type == "leakage_risk"]
        assert any(i.severity == "high" for i in leakage_issues)

    def test_duplicate_rows_produces_issue(self):
        exp = _make_experiment()
        leakage = _make_leakage(duplicate_rows=100)

        review = run_critic(experiment=exp, leakage_result=leakage)

        dup_issues = [i for i in review.issues if "duplicate" in i.evidence.lower()]
        assert len(dup_issues) >= 1
        assert dup_issues[0].severity == "high"  # >50 duplicates

    def test_no_leakage_result_flags_not_checked(self):
        exp = _make_experiment()

        review = run_critic(experiment=exp, leakage_result=None)

        types = {i.type for i in review.issues}
        assert "leakage_not_checked" in types


class TestRunCriticCVVariance:
    """High CV variance should produce conditional approval."""

    def test_high_cv_std_conditional(self):
        exp = _make_experiment(
            metrics={"train_score": 0.85, "test_score": 0.83, "cv_mean": 0.83, "cv_std": 0.10},
        )
        leakage = _make_leakage()
        robustness = _make_robustness(seed_scores=[0.80, 0.82], seed_std=0.01)

        review = run_critic(
            experiment=exp,
            leakage_result=leakage,
            robustness_report=robustness,
        )

        # cv_std=0.10 is >= threshold*3 (0.03*3=0.09), so severity="high"
        # which means verdict="conditional"
        assert review.verdict == "conditional"
        cv_issues = [i for i in review.issues if i.type == "high_cv_variance"]
        assert len(cv_issues) == 1
        assert cv_issues[0].severity == "high"

    def test_moderate_cv_std_still_approve(self):
        exp = _make_experiment(
            metrics={"train_score": 0.85, "test_score": 0.83, "cv_mean": 0.83, "cv_std": 0.035},
        )
        leakage = _make_leakage()
        robustness = _make_robustness(seed_scores=[0.82, 0.83], seed_std=0.005)

        review = run_critic(
            experiment=exp,
            leakage_result=leakage,
            robustness_report=robustness,
        )

        # cv_std=0.035 >= threshold (0.03) but < 2*threshold, so severity="low"
        cv_issues = [i for i in review.issues if i.type == "high_cv_variance"]
        assert len(cv_issues) == 1
        assert cv_issues[0].severity == "low"
        # Low severity alone does not block approval
        assert review.verdict == "approve"

    def test_no_cv_std_no_issue(self):
        exp = _make_experiment(metrics={"train_score": 0.85, "test_score": 0.83})
        issues = _check_cv_variance(exp, cv_std_threshold=0.03)
        assert len(issues) == 0


class TestRunCriticOverfitting:
    """Overfitting (large train-test gap) detection."""

    def test_large_gap_detected(self):
        exp = _make_experiment(
            metric="f1",
            metrics={"train_score": 0.99, "test_score": 0.70},
        )

        issues = _check_overfitting(exp, threshold=0.05)
        assert len(issues) == 1
        assert issues[0].type == "overfitting"
        # gap = 0.29, threshold * 3 = 0.15 → critical
        assert issues[0].severity == "critical"

    def test_moderate_gap(self):
        exp = _make_experiment(
            metric="f1",
            metrics={"train_score": 0.90, "test_score": 0.82},
        )

        issues = _check_overfitting(exp, threshold=0.05)
        assert len(issues) == 1
        # gap = 0.08, >= threshold but < 2*threshold → medium
        assert issues[0].severity == "medium"

    def test_no_gap_no_issue(self):
        exp = _make_experiment(
            metric="f1",
            metrics={"train_score": 0.88, "test_score": 0.88},
        )
        issues = _check_overfitting(exp, threshold=0.05)
        assert len(issues) == 0

    def test_lower_is_better_metric(self):
        """For error metrics (rmse), overfitting means train << test."""
        exp = _make_experiment(
            metric="rmse",
            metrics={"train_score": 0.10, "test_score": 0.30},
        )
        issues = _check_overfitting(exp, threshold=0.05)
        assert len(issues) == 1
        assert issues[0].type == "overfitting"
        # gap = 0.30 - 0.10 = 0.20 → critical
        assert issues[0].severity == "critical"

    def test_missing_scores_no_issue(self):
        exp = _make_experiment(metrics={"cv_mean": 0.85})
        issues = _check_overfitting(exp, threshold=0.05)
        assert len(issues) == 0

    def test_overfitting_causes_reject_via_run_critic(self):
        """Critical overfitting through the full run_critic pipeline."""
        exp = _make_experiment(
            metric="f1",
            metrics={"train_score": 0.99, "test_score": 0.70, "cv_mean": 0.70, "cv_std": 0.01},
        )
        leakage = _make_leakage()
        robustness = _make_robustness(seed_scores=[0.69, 0.71], seed_std=0.01)

        review = run_critic(
            experiment=exp,
            leakage_result=leakage,
            robustness_report=robustness,
        )
        assert review.verdict == "reject"


# ===================================================================
# Confidence scoring
# ===================================================================


class TestConfidence:
    def test_all_inputs_provided_highest_confidence(self):
        issues = []
        conf = _compute_confidence(
            issues,
            has_leakage_result=True,
            has_robustness_report=True,
            has_error_analysis=True,
        )
        assert conf == 1.0

    def test_missing_all_inputs_lowers_confidence(self):
        issues = []
        conf = _compute_confidence(
            issues,
            has_leakage_result=False,
            has_robustness_report=False,
            has_error_analysis=False,
        )
        # Missing: -0.15 -0.15 -0.10 = 0.60
        assert conf == 0.60

    def test_issues_further_reduce_confidence(self):
        issues = [
            CriticIssue(type="x", severity="critical", evidence="e", recommendation="r"),
            CriticIssue(type="y", severity="high", evidence="e", recommendation="r"),
        ]
        conf = _compute_confidence(
            issues,
            has_leakage_result=True,
            has_robustness_report=True,
            has_error_analysis=True,
        )
        # 1.0 - 0.15 (critical) - 0.10 (high) = 0.75
        assert conf == 0.75

    def test_confidence_never_below_zero(self):
        many_issues = [
            CriticIssue(type=f"issue_{i}", severity="critical", evidence="e", recommendation="r")
            for i in range(20)
        ]
        conf = _compute_confidence(
            many_issues,
            has_leakage_result=False,
            has_robustness_report=False,
            has_error_analysis=False,
        )
        assert conf == 0.0

    def test_more_inputs_means_higher_confidence(self):
        issues = []
        conf_partial = _compute_confidence(
            issues,
            has_leakage_result=True,
            has_robustness_report=False,
            has_error_analysis=False,
        )
        conf_full = _compute_confidence(
            issues,
            has_leakage_result=True,
            has_robustness_report=True,
            has_error_analysis=True,
        )
        assert conf_full > conf_partial


# ===================================================================
# Individual check functions
# ===================================================================


class TestCheckLeakage:
    def test_none_input_flags_not_checked(self):
        issues = _check_leakage(None)
        assert len(issues) == 1
        assert issues[0].type == "leakage_not_checked"

    def test_clean_leakage_no_issues(self):
        leakage = _make_leakage()
        issues = _check_leakage(leakage)
        assert len(issues) == 0

    def test_target_leakage_critical(self):
        leakage = _make_leakage(target_leakage=True)
        issues = _check_leakage(leakage)
        assert any(i.severity == "critical" for i in issues)


class TestCheckSeedSensitivity:
    def test_none_report_flags_not_checked(self):
        issues = _check_seed_sensitivity(None)
        assert len(issues) == 1
        assert issues[0].type == "robustness_not_checked"

    def test_stable_seeds_no_issue(self):
        rob = _make_robustness(seed_scores=[0.85, 0.86, 0.85], seed_std=0.005)
        issues = _check_seed_sensitivity(rob)
        assert len(issues) == 0

    def test_high_seed_range_flags_issue(self):
        rob = _make_robustness(seed_scores=[0.70, 0.85, 0.90], seed_std=0.08)
        issues = _check_seed_sensitivity(rob)
        assert len(issues) == 1
        assert issues[0].type == "seed_sensitivity"
        assert issues[0].severity == "high"  # range = 0.20 > 0.10

    def test_moderate_seed_range(self):
        rob = _make_robustness(seed_scores=[0.83, 0.90], seed_std=0.04)
        issues = _check_seed_sensitivity(rob)
        assert len(issues) == 1
        assert issues[0].severity == "medium"  # range = 0.07, between 0.05 and 0.10

    def test_passed_false_with_no_scores(self):
        rob = _make_robustness(passed=False)
        issues = _check_seed_sensitivity(rob)
        assert any(i.type == "robustness_failed" for i in issues)


class TestCheckReproducibility:
    def test_seed_present_no_issue(self):
        exp = _make_experiment(seed=42)
        issues = _check_reproducibility(exp)
        assert len(issues) == 0

    def test_seed_always_present(self):
        """ExperimentRecord.seed is required (int), so the reproducibility
        check should always pass for any valid ExperimentRecord."""
        exp = _make_experiment(seed=42)
        issues = _check_reproducibility(exp)
        assert len(issues) == 0


class TestCheckMetricAppropriateness:
    def test_valid_metric_no_issue(self):
        exp = _make_experiment(metric="f1")
        issues = _check_metric_appropriateness(exp, error_analysis=None)
        assert not any(i.type == "inappropriate_metric" for i in issues)

    def test_missing_metric_flags_issue(self):
        exp = _make_experiment(metric="")
        issues = _check_metric_appropriateness(exp, error_analysis=None)
        assert any(i.type == "metric_unspecified" for i in issues)

    def test_regression_metric_on_classification_flags_issue(self):
        from tools.ml.schemas import ClassificationErrorAnalysis

        exp = _make_experiment(metric="rmse")
        mock_ea = MagicMock(spec=ClassificationErrorAnalysis)
        mock_ea.class_distribution = None
        issues = _check_metric_appropriateness(exp, error_analysis=mock_ea)
        assert any(i.type == "inappropriate_metric" for i in issues)


class TestCheckSubgroupDisparities:
    def test_no_error_analysis_no_issue(self):
        issues = _check_subgroup_disparities(None, min_subgroup_size=20, disparity_threshold=0.10)
        assert len(issues) == 0

    def test_disparity_detected(self):
        mock_ea = SimpleNamespace(
            subgroup_metrics={
                "group_a": {"accuracy": 0.95, "n": 100},
                "group_b": {"accuracy": 0.70, "n": 50},
            }
        )
        issues = _check_subgroup_disparities(mock_ea, min_subgroup_size=20, disparity_threshold=0.10)
        assert len(issues) == 1
        assert issues[0].type == "subgroup_disparity"
        assert "group_a" in issues[0].evidence
        assert "group_b" in issues[0].evidence

    def test_small_disparity_no_issue(self):
        mock_ea = SimpleNamespace(
            subgroup_metrics={
                "group_a": {"accuracy": 0.90, "n": 100},
                "group_b": {"accuracy": 0.88, "n": 50},
            }
        )
        issues = _check_subgroup_disparities(mock_ea, min_subgroup_size=20, disparity_threshold=0.10)
        assert len(issues) == 0


# ===================================================================
# Verdict derivation
# ===================================================================


class TestDeriveVerdict:
    def test_no_issues_approve(self):
        assert _derive_verdict([]) == "approve"

    def test_low_medium_approve(self):
        issues = [
            CriticIssue(type="x", severity="low", evidence="e", recommendation="r"),
            CriticIssue(type="y", severity="medium", evidence="e", recommendation="r"),
        ]
        assert _derive_verdict(issues) == "approve"

    def test_high_conditional(self):
        issues = [
            CriticIssue(type="x", severity="high", evidence="e", recommendation="r"),
        ]
        assert _derive_verdict(issues) == "conditional"

    def test_critical_reject(self):
        issues = [
            CriticIssue(type="x", severity="critical", evidence="e", recommendation="r"),
        ]
        assert _derive_verdict(issues) == "reject"


# ===================================================================
# Helper: _std
# ===================================================================


class TestStd:
    def test_single_value(self):
        assert _std([5.0]) == 0.0

    def test_identical_values(self):
        assert _std([3.0, 3.0, 3.0]) == 0.0

    def test_known_values(self):
        # Population std of [2, 4, 4, 4, 5, 5, 7, 9] = 2.0
        result = _std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(result - 2.0) < 1e-9

    def test_empty_list(self):
        assert _std([]) == 0.0


# ===================================================================
# Summary / required actions
# ===================================================================


class TestSummaryAndActions:
    def test_clean_summary(self):
        exp = _make_experiment()
        leakage = _make_leakage()
        robustness = _make_robustness(seed_scores=[0.87, 0.88], seed_std=0.005)

        review = run_critic(experiment=exp, leakage_result=leakage, robustness_report=robustness)
        assert "exp-001" in review.summary

    def test_reject_has_required_actions(self):
        exp = _make_experiment()
        leakage = _make_leakage(target_leakage=True)
        robustness = _make_robustness(seed_scores=[0.99, 0.99], seed_std=0.001)

        review = run_critic(experiment=exp, leakage_result=leakage, robustness_report=robustness)
        assert review.verdict == "reject"
        assert len(review.required_next_actions) > 0
        assert any("leakage" in a.lower() for a in review.required_next_actions)

    def test_approve_no_required_actions_for_low_issues(self):
        exp = _make_experiment()
        leakage = _make_leakage()
        robustness = _make_robustness(seed_scores=[0.87, 0.88], seed_std=0.005)

        review = run_critic(experiment=exp, leakage_result=leakage, robustness_report=robustness)
        # Required actions only come from high/critical issues
        # An approved review should have none
        assert len(review.required_next_actions) == 0
