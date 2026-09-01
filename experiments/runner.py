"""Experiment runner (docs/spec.md section 10, Prompt 6).

`run_experiment` wraps `tools/ml/automl.run_automl` with the bookkeeping
that turns a one-off model search into a tracked, comparable experiment:
a required hypothesis (spec principle 3 — no experiment without a reason),
a parent/child link, and a conclusion decided by comparing scores against
the parent, never by re-optimizing against the test set.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from experiments.registry import ExperimentRegistry
from experiments.schemas import Conclusion, ExperimentRecord
from tools.data.schemas import DatasetProfile, SplitResult
from tools.ml.automl import is_higher_better, run_automl
from tools.ml.schemas import TaskType

_PREPROCESSING_DESCRIPTION = "median-impute + scale (numeric); most-frequent-impute + one-hot (categorical)"

_DEFAULT_NEXT_ACTION: dict[Conclusion, str] = {
    "supported": "run multi-seed validation to confirm robustness",
    "rejected": "form a different hypothesis — this change did not help",
    "inconclusive": "effect was within noise — consider a larger change or more data",
}


def run_experiment(
    df: pd.DataFrame,
    target: str,
    task_type: TaskType,
    split: SplitResult,
    *,
    hypothesis: str,
    registry: ExperimentRegistry,
    parent_experiment_id: str | None = None,
    objective: str | None = None,
    profile: DatasetProfile | None = None,
    feature_columns: list[str] | None = None,
    metric: str | None = None,
    cv_folds: int = 5,
    seed: int = 42,
    dataset_version_id: str | None = None,
    artifact_dir: str | Path | None = "./data/artifacts",
    candidate_models: list[str] | None = None,
    minimum_improvement: float = 0.002,
) -> ExperimentRecord:
    """Run one AutoML search and record it as a tracked, comparable experiment.

    Raises:
        ValueError: if `hypothesis` is blank.
        KeyError: if `parent_experiment_id` is given but not already recorded in `registry`.
    """
    if not hypothesis.strip():
        raise ValueError("hypothesis must be a non-empty string — every experiment needs a reason")

    parent = registry.get(parent_experiment_id) if parent_experiment_id else None

    automl_result = run_automl(
        df,
        target=target,
        task_type=task_type,
        split=split,
        profile=profile,
        feature_columns=feature_columns,
        metric=metric,
        cv_folds=cv_folds,
        seed=seed,
        dataset_version_id=dataset_version_id,
        artifact_dir=artifact_dir,
        candidate_models=candidate_models,
    )

    metrics = {"cv_mean": automl_result.cv_mean, "cv_std": automl_result.cv_std}
    if automl_result.validation_score is not None:
        metrics["validation_score"] = automl_result.validation_score
    if automl_result.test_score is not None:
        metrics["test_score"] = automl_result.test_score

    # Prefer the held-out test score for comparison — it's the only number
    # never touched during model selection. Falls back to cv_mean when no
    # test split was given.
    primary_score = automl_result.test_score if automl_result.test_score is not None else automl_result.cv_mean

    conclusion: Conclusion | None = None
    next_action: str | None = None
    if parent is not None:
        conclusion = _compare(parent, automl_result.metric, primary_score, minimum_improvement)
        next_action = _DEFAULT_NEXT_ACTION[conclusion]

    record = ExperimentRecord(
        experiment_id=automl_result.experiment_id,
        parent_experiment_id=parent_experiment_id,
        hypothesis=hypothesis,
        objective=objective,
        dataset_version_id=dataset_version_id,
        split_strategy=split.strategy,
        features=automl_result.feature_columns,
        excluded_features=automl_result.excluded_columns,
        preprocessing=_PREPROCESSING_DESCRIPTION,
        model=automl_result.best_model,
        hyperparameters=automl_result.best_params,
        seed=seed,
        metric=automl_result.metric,
        metrics=metrics,
        primary_score=primary_score,
        runtime_seconds=automl_result.runtime_seconds,
        artifact_path=automl_result.artifact_path,
        conclusion=conclusion,
        next_action=next_action,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    registry.record(record)
    return record


def _compare(parent: ExperimentRecord, metric: str, child_score: float, minimum_improvement: float) -> Conclusion:
    if parent.metric != metric:
        return "inconclusive"
    higher_better = is_higher_better(metric)
    delta = child_score - parent.primary_score
    if not higher_better:
        delta = -delta
    if delta >= minimum_improvement:
        return "supported"
    if delta <= -minimum_improvement:
        return "rejected"
    return "inconclusive"
