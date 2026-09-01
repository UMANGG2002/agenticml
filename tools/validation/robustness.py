"""Robustness checks (docs/spec.md section 17).

Only runs the checks the task actually supports — a dataset with no time
column can't be temporally validated, and saying so explicitly (via
`checks_skipped`) is more honest than silently reporting a pass.

Like `detect_leakage`, every check returns evidence and a severity rather
than a bare boolean, and nothing here decides what to do about a failure —
that's the critic's job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.data.schemas import SplitResult
from tools.data.split import create_split
from tools.ml.automl import is_higher_better, run_automl
from tools.ml.schemas import TaskType
from tools.validation.schemas import RobustnessCheck, RobustnessReport, SeedRun


def run_robustness_checks(
    df: pd.DataFrame,
    target: str,
    task_type: TaskType,
    split: SplitResult,
    *,
    reference_score: float,
    metric: str,
    feature_columns: list[str],
    model_name: str,
    seeds: list[int] | None = None,
    cv_std: float | None = None,
    time_column: str | None = None,
    group_column: str | None = None,
    noise_scale: float = 0.1,
    cv_std_threshold: float = 0.05,
    seed_std_threshold: float = 0.03,
    degradation_threshold: float = 0.05,
    cv_folds: int = 3,
    base_seed: int = 42,
) -> RobustnessReport:
    """Re-run the winning model under varied conditions and report where it degrades.

    Args:
        reference_score: the score to compare every check against (typically
            the experiment's held-out test score).
        metric: the metric `reference_score` is expressed in.
        model_name: the single candidate model to re-run (the winner) — checks
            re-fit this model rather than re-searching the whole registry.
        seeds: seeds for multi-seed evaluation. Defaults to 5 seeds.
        cv_std: cross-validation std from the original run, if available.
        time_column / group_column: enable temporal / group validation when given.
    """
    checks: list[RobustnessCheck] = []
    checks_skipped: list[str] = []
    seeds = seeds if seeds is not None else [1, 7, 13, 42, 99]
    higher_better = is_higher_better(metric)

    automl_kwargs = dict(
        target=target,
        task_type=task_type,
        feature_columns=feature_columns,
        metric=metric,
        cv_folds=cv_folds,
        artifact_dir=None,
        candidate_models=[model_name],
    )

    _check_cv_variance(cv_std, cv_std_threshold, checks, checks_skipped, metric)
    _check_multi_seed(
        df, split, seeds, reference_score, metric, seed_std_threshold, automl_kwargs,
        target, group_column, time_column, checks,
    )
    _check_alternate_split_strategy(
        df, reference_score, metric, higher_better, degradation_threshold,
        time_column, group_column, automl_kwargs, base_seed, checks, checks_skipped,
    )
    _check_noise_sensitivity(
        df, split, feature_columns, reference_score, metric, higher_better,
        degradation_threshold, noise_scale, automl_kwargs, base_seed, checks, checks_skipped,
    )

    return RobustnessReport(
        metric=metric,
        reference_score=reference_score,
        checks=checks,
        checks_skipped=checks_skipped,
        n_failed=sum(1 for c in checks if not c.passed),
    )


def _degraded(reference: float, observed: float, higher_better: bool) -> float:
    """How much worse `observed` is than `reference` (negative means better)."""
    return (reference - observed) if higher_better else (observed - reference)


def _check_cv_variance(
    cv_std: float | None,
    threshold: float,
    checks: list[RobustnessCheck],
    checks_skipped: list[str],
    metric: str,
) -> None:
    if cv_std is None:
        checks_skipped.append("cv_variance (cv_std not supplied)")
        return
    passed = cv_std <= threshold
    checks.append(
        RobustnessCheck(
            name="cv_variance",
            metric=metric,
            passed=passed,
            severity=None if passed else "medium",
            evidence=(
                f"cross-validation std is {cv_std:.4f} "
                f"({'within' if passed else 'above'} the {threshold} threshold)."
            ),
            values={"cv_std": cv_std, "threshold": threshold},
        )
    )


def _check_multi_seed(
    df: pd.DataFrame,
    split: SplitResult,
    seeds: list[int],
    reference_score: float,
    metric: str,
    threshold: float,
    automl_kwargs: dict,
    target: str,
    group_column: str | None,
    time_column: str | None,
    checks: list[RobustnessCheck],
) -> None:
    # Re-split per seed, not just re-fit. Deterministic estimators ignore
    # random_state entirely, so reusing one fixed split would make this
    # check vacuously pass with std == 0 regardless of the model.
    runs: list[SeedRun] = []
    for seed in seeds:
        seed_split = create_split(
            df,
            split.strategy,
            test_size=0.2,
            target=target if split.strategy == "stratified" else None,
            group_column=group_column,
            time_column=time_column,
            seed=seed,
        )
        result = run_automl(df, split=seed_split, seed=seed, **automl_kwargs)
        score = result.test_score if result.test_score is not None else result.cv_mean
        runs.append(SeedRun(seed=seed, score=score))

    scores = [r.score for r in runs]
    mean, std = float(np.mean(scores)), float(np.std(scores))
    passed = std <= threshold
    checks.append(
        RobustnessCheck(
            name="multi_seed",
            metric=metric,
            passed=passed,
            severity=None if passed else "medium",
            evidence=(
                f"{len(seeds)}-seed mean {mean:.4f}, std {std:.4f} "
                f"(reference {reference_score:.4f}); std is "
                f"{'within' if passed else 'above'} the {threshold} threshold."
            ),
            values={"mean": mean, "std": std, "min": min(scores), "max": max(scores), "threshold": threshold},
        )
    )


def _check_alternate_split_strategy(
    df: pd.DataFrame,
    reference_score: float,
    metric: str,
    higher_better: bool,
    threshold: float,
    time_column: str | None,
    group_column: str | None,
    automl_kwargs: dict,
    base_seed: int,
    checks: list[RobustnessCheck],
    checks_skipped: list[str],
) -> None:
    """Temporal and group validation — the checks that most often expose
    optimistic random-split scores."""
    for name, strategy, column, kwarg in (
        ("temporal_validation", "time", time_column, "time_column"),
        ("group_validation", "group", group_column, "group_column"),
    ):
        if column is None:
            checks_skipped.append(f"{name} ({kwarg} not supplied)")
            continue
        if column not in df.columns:
            checks_skipped.append(f"{name} (column '{column}' not in dataset)")
            continue

        alt_split = create_split(df, strategy, test_size=0.2, seed=base_seed, **{kwarg: column})
        result = run_automl(df, split=alt_split, seed=base_seed, **automl_kwargs)
        score = result.test_score if result.test_score is not None else result.cv_mean

        degradation = _degraded(reference_score, score, higher_better)
        passed = degradation <= threshold
        checks.append(
            RobustnessCheck(
                name=name,
                metric=metric,
                passed=passed,
                severity=None if passed else ("high" if degradation > threshold * 2 else "medium"),
                evidence=(
                    f"{strategy}-split score {score:.4f} vs reference {reference_score:.4f} "
                    f"(degradation {degradation:+.4f}, threshold {threshold})."
                ),
                values={"score": score, "degradation": degradation, "threshold": threshold},
            )
        )


def _check_noise_sensitivity(
    df: pd.DataFrame,
    split: SplitResult,
    feature_columns: list[str],
    reference_score: float,
    metric: str,
    higher_better: bool,
    threshold: float,
    noise_scale: float,
    automl_kwargs: dict,
    base_seed: int,
    checks: list[RobustnessCheck],
    checks_skipped: list[str],
) -> None:
    numeric_features = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_features:
        checks_skipped.append("noise_sensitivity (no numeric feature columns)")
        return

    rng = np.random.default_rng(base_seed)
    noisy = df.copy()
    for col in numeric_features:
        col_std = df[col].std()
        if col_std and np.isfinite(col_std):
            noisy[col] = df[col] + rng.normal(scale=noise_scale * col_std, size=len(df))

    result = run_automl(noisy, split=split, seed=base_seed, **automl_kwargs)
    score = result.test_score if result.test_score is not None else result.cv_mean

    degradation = _degraded(reference_score, score, higher_better)
    passed = degradation <= threshold
    checks.append(
        RobustnessCheck(
            name="noise_sensitivity",
            metric=metric,
            passed=passed,
            severity=None if passed else "medium",
            evidence=(
                f"with {noise_scale:.0%} feature noise added, score {score:.4f} vs reference "
                f"{reference_score:.4f} (degradation {degradation:+.4f}, threshold {threshold})."
            ),
            values={"score": score, "degradation": degradation, "noise_scale": noise_scale, "threshold": threshold},
        )
    )
