"""Deterministic critic / falsification layer for the Agentic ML Engineer.

This module independently reviews an experiment's results and checks for
methodological problems.  It is purely deterministic -- no LLM calls -- so
every verdict is reproducible and auditable.

Implements the checks described in spec section 12:
  1. Data-leakage warnings
  2. Overfitting (train-vs-test gap)
  3. Cross-validation variance
  4. Random-seed sensitivity
  5. Metric-selection appropriateness
  6. Subgroup performance disparities
  7. Reproducibility (seed recorded)

Usage::

    from critic import run_critic
    review = run_critic(
        experiment=exp,
        leakage_result=leakage,
        robustness_report=robustness,
    )
    if review.verdict == "reject":
        ...
"""

from __future__ import annotations

import math
from typing import Literal, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Project-internal imports (lazy-friendly: only referenced inside run_critic)
# ---------------------------------------------------------------------------
from experiments.schemas import ExperimentRecord
from tools.data.schemas import LeakageDetectionResult
from tools.ml.schemas import (
    ClassificationErrorAnalysis,
    RegressionErrorAnalysis,
    TaskType,
)
from tools.validation.schemas import RobustnessReport

# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class CriticIssue(BaseModel):
    """A single issue identified by the critic."""

    type: str  # e.g. "overfitting", "high_cv_variance", "leakage_risk"
    severity: Literal["low", "medium", "high", "critical"]
    evidence: str
    recommendation: str


class CriticReview(BaseModel):
    """Complete review produced by the critic for one experiment."""

    experiment_id: str
    verdict: Literal["approve", "conditional", "reject"]
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[CriticIssue]
    required_next_actions: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Metric-appropriateness lookup
# ---------------------------------------------------------------------------

# Maps task types to the metrics considered appropriate.
_APPROPRIATE_METRICS: dict[str, set[str]] = {
    "binary_classification": {
        "accuracy",
        "f1",
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
        "auc",
        "average_precision",
        "log_loss",
        "brier_score_loss",
        "balanced_accuracy",
        "mcc",
        "matthews_corrcoef",
    },
    "multiclass_classification": {
        "accuracy",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "f1",
        "precision_macro",
        "recall_macro",
        "log_loss",
        "balanced_accuracy",
        "mcc",
        "matthews_corrcoef",
    },
    "regression": {
        "rmse",
        "root_mean_squared_error",
        "mse",
        "mean_squared_error",
        "mae",
        "mean_absolute_error",
        "r2",
        "r2_score",
        "mape",
        "mean_absolute_percentage_error",
        "explained_variance",
        "median_absolute_error",
    },
}

# Metrics where lower is better (used when interpreting gaps).
_LOWER_IS_BETTER: set[str] = {
    "rmse",
    "root_mean_squared_error",
    "mse",
    "mean_squared_error",
    "mae",
    "mean_absolute_error",
    "mape",
    "mean_absolute_percentage_error",
    "median_absolute_error",
    "log_loss",
    "brier_score_loss",
}

# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def _check_leakage(
    leakage_result: LeakageDetectionResult | None,
) -> list[CriticIssue]:
    """Check 1: data-leakage warnings."""
    if leakage_result is None:
        return [
            CriticIssue(
                type="leakage_not_checked",
                severity="medium",
                evidence="No leakage detection result was provided for review.",
                recommendation=(
                    "Run detect_leakage() and supply the result to the critic."
                ),
            )
        ]

    issues: list[CriticIssue] = []

    # Check for detected leakage signals on the result object.
    # LeakageDetectionResult may expose leaked features, duplicates, or
    # target-leakage flags.  We inspect the attributes that are commonly
    # present, falling back gracefully when a field is absent.
    leaked_features: list[str] = getattr(leakage_result, "leaked_features", []) or []
    duplicate_rows: int = getattr(leakage_result, "duplicate_rows", 0) or 0
    target_leakage: bool = getattr(leakage_result, "target_leakage", False)

    if target_leakage:
        issues.append(
            CriticIssue(
                type="leakage_risk",
                severity="critical",
                evidence="Target leakage detected: information from the target variable is present in the features.",
                recommendation=(
                    "Remove features that encode or proxy the target before "
                    "training.  Re-run the experiment after remediation."
                ),
            )
        )

    if leaked_features:
        severity: Literal["low", "medium", "high", "critical"] = (
            "high" if len(leaked_features) >= 3 else "medium"
        )
        issues.append(
            CriticIssue(
                type="leakage_risk",
                severity=severity,
                evidence=(
                    f"Potential leakage detected in {len(leaked_features)} "
                    f"feature(s): {', '.join(leaked_features[:10])}."
                ),
                recommendation=(
                    "Investigate the flagged features.  Remove or re-engineer "
                    "any that would not be available at prediction time."
                ),
            )
        )

    if duplicate_rows > 0:
        severity = "high" if duplicate_rows > 50 else "medium"
        issues.append(
            CriticIssue(
                type="leakage_risk",
                severity=severity,
                evidence=(
                    f"{duplicate_rows} duplicate rows found across "
                    "train/test splits, risking data leakage."
                ),
                recommendation=(
                    "De-duplicate the dataset before splitting, or ensure "
                    "duplicates are confined to a single split."
                ),
            )
        )

    return issues


def _check_overfitting(
    experiment: ExperimentRecord,
    threshold: float,
) -> list[CriticIssue]:
    """Check 2: train-vs-test gap indicates overfitting."""
    issues: list[CriticIssue] = []
    metrics: dict = experiment.metrics or {}
    metric_name: str = experiment.metric or ""

    train_score = metrics.get("train_score")
    test_score = metrics.get("test_score") or metrics.get("cv_mean")

    if train_score is None or test_score is None:
        return issues  # Cannot evaluate without both scores.

    lower_better = metric_name.lower() in _LOWER_IS_BETTER

    if lower_better:
        # For error metrics, overfitting means train error << test error.
        gap = test_score - train_score
    else:
        # For score metrics, overfitting means train >> test.
        gap = train_score - test_score

    if gap <= 0:
        return issues  # No overfitting signal.

    abs_gap = abs(gap)

    if abs_gap >= threshold * 3:
        severity: Literal["low", "medium", "high", "critical"] = "critical"
    elif abs_gap >= threshold * 2:
        severity = "high"
    elif abs_gap >= threshold:
        severity = "medium"
    else:
        return issues

    issues.append(
        CriticIssue(
            type="overfitting",
            severity=severity,
            evidence=(
                f"Train-test gap of {abs_gap:.4f} "
                f"(train={train_score:.4f}, test={test_score:.4f}) "
                f"exceeds threshold {threshold:.4f}."
            ),
            recommendation=(
                "Consider stronger regularization, feature selection, "
                "or reducing model complexity.  Verify that the train/test "
                "split is representative."
            ),
        )
    )

    return issues


def _check_cv_variance(
    experiment: ExperimentRecord,
    cv_std_threshold: float,
) -> list[CriticIssue]:
    """Check 3: cross-validation variance."""
    issues: list[CriticIssue] = []
    metrics: dict = experiment.metrics or {}

    cv_std = metrics.get("cv_std")
    cv_mean = metrics.get("cv_mean")

    if cv_std is None:
        return issues

    if cv_std < cv_std_threshold:
        return issues

    # Compute coefficient of variation when possible.
    cv_coeff = (cv_std / abs(cv_mean)) if (cv_mean and cv_mean != 0) else None

    if cv_std >= cv_std_threshold * 3:
        severity: Literal["low", "medium", "high", "critical"] = "high"
    elif cv_std >= cv_std_threshold * 2:
        severity = "medium"
    else:
        severity = "low"

    evidence = f"CV standard deviation is {cv_std:.4f} (threshold {cv_std_threshold:.4f})."
    if cv_coeff is not None:
        evidence += f" Coefficient of variation: {cv_coeff:.2%}."

    issues.append(
        CriticIssue(
            type="high_cv_variance",
            severity=severity,
            evidence=evidence,
            recommendation=(
                "High fold-to-fold variance suggests the model is sensitive "
                "to the data partition.  Try more folds, stratified splitting, "
                "or investigate heterogeneous subpopulations."
            ),
        )
    )

    return issues


def _check_seed_sensitivity(
    robustness_report: RobustnessReport | None,
) -> list[CriticIssue]:
    """Check 4: random-seed sensitivity (via robustness report)."""
    if robustness_report is None:
        return [
            CriticIssue(
                type="robustness_not_checked",
                severity="medium",
                evidence="No robustness report was provided for review.",
                recommendation=(
                    "Run run_robustness_checks() and supply the report to the critic."
                ),
            )
        ]

    issues: list[CriticIssue] = []

    # Inspect seed-sensitivity fields that RobustnessReport exposes.
    seed_std = getattr(robustness_report, "seed_std", None)
    seed_scores: list[float] | None = getattr(robustness_report, "seed_scores", None)
    passed: bool | None = getattr(robustness_report, "passed", None)

    if seed_scores and len(seed_scores) >= 2:
        score_range = max(seed_scores) - min(seed_scores)
        computed_std = _std(seed_scores)
        seed_std = seed_std if seed_std is not None else computed_std

        if score_range > 0.10:
            severity: Literal["low", "medium", "high", "critical"] = "high"
        elif score_range > 0.05:
            severity = "medium"
        else:
            severity = "low"

        if severity != "low":
            issues.append(
                CriticIssue(
                    type="seed_sensitivity",
                    severity=severity,
                    evidence=(
                        f"Score range across seeds: {score_range:.4f} "
                        f"(std={seed_std:.4f}).  "
                        f"Scores: {[round(s, 4) for s in seed_scores]}."
                    ),
                    recommendation=(
                        "Results vary significantly with the random seed.  "
                        "Report averaged metrics, increase training data, "
                        "or use a more stable algorithm."
                    ),
                )
            )
    elif seed_std is not None and seed_std > 0.03:
        issues.append(
            CriticIssue(
                type="seed_sensitivity",
                severity="medium" if seed_std < 0.06 else "high",
                evidence=f"Seed standard deviation is {seed_std:.4f}.",
                recommendation=(
                    "Results vary with the random seed.  Average over "
                    "multiple seeds and report the mean +/- std."
                ),
            )
        )

    # If robustness report has an explicit pass/fail and it failed.
    if passed is False and not issues:
        issues.append(
            CriticIssue(
                type="robustness_failed",
                severity="high",
                evidence="The robustness report indicates the experiment did not pass robustness checks.",
                recommendation=(
                    "Investigate the robustness report details.  The model "
                    "may be unreliable under perturbation."
                ),
            )
        )

    return issues


def _check_metric_appropriateness(
    experiment: ExperimentRecord,
    error_analysis: ClassificationErrorAnalysis | RegressionErrorAnalysis | None,
) -> list[CriticIssue]:
    """Check 5: is the chosen metric appropriate for the task type?"""
    issues: list[CriticIssue] = []
    metric_name = (experiment.metric or "").lower().strip()

    if not metric_name:
        issues.append(
            CriticIssue(
                type="metric_unspecified",
                severity="medium",
                evidence="No primary metric is specified for the experiment.",
                recommendation="Specify a primary evaluation metric.",
            )
        )
        return issues

    # Infer task type from error_analysis or from the metric name itself.
    task_type_str: str | None = None
    if error_analysis is not None:
        if isinstance(error_analysis, ClassificationErrorAnalysis):
            task_type_str = "binary_classification"  # conservative default
        elif isinstance(error_analysis, RegressionErrorAnalysis):
            task_type_str = "regression"

    if task_type_str is None:
        # Heuristic: infer from metric name.
        if metric_name in _APPROPRIATE_METRICS.get("regression", set()):
            task_type_str = "regression"
        elif metric_name in _APPROPRIATE_METRICS.get("binary_classification", set()):
            task_type_str = "binary_classification"

    if task_type_str is None:
        return issues  # Cannot determine task type; skip this check.

    # Check against lookup.
    appropriate = _APPROPRIATE_METRICS.get(task_type_str, set())
    # Also check any classification variant.
    if task_type_str == "binary_classification":
        appropriate = appropriate | _APPROPRIATE_METRICS.get(
            "multiclass_classification", set()
        )
    elif task_type_str == "multiclass_classification":
        appropriate = appropriate | _APPROPRIATE_METRICS.get(
            "binary_classification", set()
        )

    if appropriate and metric_name not in appropriate:
        issues.append(
            CriticIssue(
                type="inappropriate_metric",
                severity="medium",
                evidence=(
                    f"Metric '{metric_name}' is not in the standard set for "
                    f"task type '{task_type_str}'.  "
                    f"Appropriate metrics include: {', '.join(sorted(appropriate)[:6])}."
                ),
                recommendation=(
                    "Verify the metric is meaningful for this task.  "
                    "Consider switching to a standard metric."
                ),
            )
        )

    # Special case: accuracy on imbalanced classification.
    if metric_name == "accuracy" and error_analysis is not None:
        class_distribution = getattr(error_analysis, "class_distribution", None)
        if class_distribution and isinstance(class_distribution, dict):
            counts = list(class_distribution.values())
            if counts and max(counts) > 0:
                imbalance_ratio = min(counts) / max(counts)
                if imbalance_ratio < 0.3:
                    issues.append(
                        CriticIssue(
                            type="inappropriate_metric",
                            severity="high",
                            evidence=(
                                f"Accuracy used on an imbalanced dataset "
                                f"(minority/majority ratio: {imbalance_ratio:.2f}).  "
                                "Accuracy can be misleading here."
                            ),
                            recommendation=(
                                "Use balanced_accuracy, F1, ROC-AUC, or "
                                "average precision instead of raw accuracy."
                            ),
                        )
                    )

    return issues


def _check_subgroup_disparities(
    error_analysis: ClassificationErrorAnalysis | RegressionErrorAnalysis | None,
    min_subgroup_size: int,
    disparity_threshold: float,
) -> list[CriticIssue]:
    """Check 6: subgroup performance disparities."""
    if error_analysis is None:
        return []

    issues: list[CriticIssue] = []

    # Look for per-subgroup metrics on the error analysis object.
    subgroup_metrics: dict | None = getattr(
        error_analysis, "subgroup_metrics", None
    ) or getattr(error_analysis, "group_metrics", None)

    if not subgroup_metrics or not isinstance(subgroup_metrics, dict):
        return issues

    # subgroup_metrics expected shape: {group_name: {metric: value, "n": count}}
    # or {group_name: score_value}
    scores: dict[str, float] = {}
    for group, value in subgroup_metrics.items():
        if isinstance(value, (int, float)):
            scores[group] = float(value)
        elif isinstance(value, dict):
            n = value.get("n", value.get("count", min_subgroup_size))
            if n < min_subgroup_size:
                continue
            # Pick the first numeric score available.
            for k, v in value.items():
                if k not in ("n", "count", "size") and isinstance(v, (int, float)):
                    scores[group] = float(v)
                    break

    if len(scores) < 2:
        return issues

    best = max(scores.values())
    worst = min(scores.values())
    disparity = best - worst

    if disparity < disparity_threshold:
        return issues

    best_group = [g for g, s in scores.items() if s == best][0]
    worst_group = [g for g, s in scores.items() if s == worst][0]

    if disparity >= disparity_threshold * 3:
        severity: Literal["low", "medium", "high", "critical"] = "high"
    elif disparity >= disparity_threshold * 2:
        severity = "medium"
    else:
        severity = "low"

    issues.append(
        CriticIssue(
            type="subgroup_disparity",
            severity=severity,
            evidence=(
                f"Performance disparity of {disparity:.4f} between "
                f"best group '{best_group}' ({best:.4f}) and "
                f"worst group '{worst_group}' ({worst:.4f})."
            ),
            recommendation=(
                "Investigate why the model performs worse on certain "
                "subgroups.  Consider stratified training, re-sampling, "
                "or fairness-aware algorithms."
            ),
        )
    )

    return issues


def _check_reproducibility(
    experiment: ExperimentRecord,
) -> list[CriticIssue]:
    """Check 7: reproducibility -- is a random seed recorded?"""
    issues: list[CriticIssue] = []

    if experiment.seed is None:
        issues.append(
            CriticIssue(
                type="reproducibility",
                severity="medium",
                evidence="No random seed is recorded for this experiment.",
                recommendation=(
                    "Always record the random seed to enable exact "
                    "reproduction of results."
                ),
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _std(values: list[float]) -> float:
    """Population standard deviation (no numpy dependency)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return math.sqrt(variance)


def _derive_verdict(
    issues: list[CriticIssue],
) -> Literal["approve", "conditional", "reject"]:
    """Determine the overall verdict from the issue list."""
    severities = {issue.severity for issue in issues}
    if "critical" in severities:
        return "reject"
    if "high" in severities:
        return "conditional"
    return "approve"


def _compute_confidence(
    issues: list[CriticIssue],
    has_leakage_result: bool,
    has_robustness_report: bool,
    has_error_analysis: bool,
) -> float:
    """Compute a confidence score for the review.

    Confidence reflects how thoroughly we were able to evaluate the
    experiment, not how good the experiment is.  Missing inputs lower
    confidence because we cannot rule out problems.
    """
    confidence = 1.0

    # Penalize for missing optional inputs.
    if not has_leakage_result:
        confidence -= 0.15
    if not has_robustness_report:
        confidence -= 0.15
    if not has_error_analysis:
        confidence -= 0.10

    # Penalize for each issue found (more issues = less certain the
    # experiment is sound).
    severity_penalty = {
        "low": 0.02,
        "medium": 0.05,
        "high": 0.10,
        "critical": 0.15,
    }
    for issue in issues:
        confidence -= severity_penalty.get(issue.severity, 0.05)

    return max(0.0, min(1.0, round(confidence, 4)))


def _build_summary(
    experiment: ExperimentRecord,
    verdict: str,
    issues: list[CriticIssue],
) -> str:
    """Build a human-readable summary line."""
    if not issues:
        return (
            f"Experiment {experiment.experiment_id}: APPROVED.  "
            "No issues detected."
        )

    severity_counts: dict[str, int] = {}
    for issue in issues:
        severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

    parts = [f"{count} {sev}" for sev, count in sorted(severity_counts.items())]
    issue_summary = ", ".join(parts)
    types = ", ".join(sorted({i.type for i in issues}))

    return (
        f"Experiment {experiment.experiment_id}: {verdict.upper()}.  "
        f"Found {len(issues)} issue(s) ({issue_summary}).  "
        f"Types: {types}."
    )


def _build_required_actions(
    issues: list[CriticIssue],
    verdict: str,
) -> list[str]:
    """Collect required next actions from high/critical issues."""
    actions: list[str] = []
    for issue in issues:
        if issue.severity in ("high", "critical"):
            actions.append(f"[{issue.type}] {issue.recommendation}")
    return actions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_critic(
    experiment: ExperimentRecord,
    *,
    leakage_result: LeakageDetectionResult | None = None,
    robustness_report: RobustnessReport | None = None,
    error_analysis: (
        ClassificationErrorAnalysis | RegressionErrorAnalysis | None
    ) = None,
    all_experiments: list[ExperimentRecord] | None = None,
    overfitting_threshold: float = 0.05,
    cv_std_threshold: float = 0.03,
    min_subgroup_size: int = 20,
    subgroup_disparity_threshold: float = 0.10,
) -> CriticReview:
    """Run the deterministic critic on a single experiment.

    The critic inspects the experiment record and any supplementary
    analysis results, producing a :class:`CriticReview` that lists
    every issue found and an overall verdict.

    Parameters
    ----------
    experiment:
        The experiment record to review.
    leakage_result:
        Output of ``detect_leakage()``.  If ``None`` the critic flags
        that leakage was not checked.
    robustness_report:
        Output of ``run_robustness_checks()``.  If ``None`` the critic
        flags that robustness was not checked.
    error_analysis:
        Output of ``analyze_classification_errors()`` or
        ``analyze_regression_errors()``.  Used for metric
        appropriateness and subgroup checks.
    all_experiments:
        Full experiment history.  Reserved for future comparative
        checks (e.g. detecting plateau or regression across lineage).
    overfitting_threshold:
        Maximum acceptable absolute gap between train and test scores.
    cv_std_threshold:
        Maximum acceptable cross-validation standard deviation.
    min_subgroup_size:
        Minimum number of samples for a subgroup to be included in
        the disparity check.
    subgroup_disparity_threshold:
        Maximum acceptable performance gap between the best and worst
        subgroups.

    Returns
    -------
    CriticReview
        The complete review with verdict, issues, and recommended
        actions.
    """
    issues: list[CriticIssue] = []

    # 1. Data leakage
    issues.extend(_check_leakage(leakage_result))

    # 2. Overfitting
    issues.extend(_check_overfitting(experiment, overfitting_threshold))

    # 3. CV variance
    issues.extend(_check_cv_variance(experiment, cv_std_threshold))

    # 4. Seed sensitivity
    issues.extend(_check_seed_sensitivity(robustness_report))

    # 5. Metric appropriateness
    issues.extend(_check_metric_appropriateness(experiment, error_analysis))

    # 6. Subgroup disparities
    issues.extend(
        _check_subgroup_disparities(
            error_analysis, min_subgroup_size, subgroup_disparity_threshold
        )
    )

    # 7. Reproducibility
    issues.extend(_check_reproducibility(experiment))

    # Derive verdict and confidence.
    verdict = _derive_verdict(issues)
    confidence = _compute_confidence(
        issues,
        has_leakage_result=leakage_result is not None,
        has_robustness_report=robustness_report is not None,
        has_error_analysis=error_analysis is not None,
    )
    summary = _build_summary(experiment, verdict, issues)
    required_actions = _build_required_actions(issues, verdict)

    return CriticReview(
        experiment_id=experiment.experiment_id,
        verdict=verdict,
        confidence=confidence,
        issues=issues,
        required_next_actions=required_actions,
        summary=summary,
    )
