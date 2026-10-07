"""AutoML tool: a deterministic, reproducible baseline model search.

`run_automl` receives an explicit train/validation/test split up front
(see tools/data/split.py) and structurally never touches test rows during
model search:

- every candidate is scored purely by cross-validation *within the
  training rows*
- train/validation/test index sets are checked for overlap up front and
  the call is refused if they leak into each other
- the held-out test set is touched exactly once, after the best candidate
  has already been selected on training data alone, purely to report a
  number that is never fed back into selection

Not the product: this wraps a handful of scikit-learn estimators with a
small hand-rolled hyperparameter grid, not a tuned production model. It
exists to give the agent a first baseline to critique and improve on
(docs/spec.md section 9).
"""

from __future__ import annotations

import inspect
import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tools.data.schemas import DatasetProfile, SplitResult
from tools.ml.labels import positive_label
from tools.ml.schemas import AutoMLResult, ModelCandidateResult, TaskType

# metric_name -> (sklearn scoring string, higher_is_better)
_METRIC_REGISTRY: dict[str, tuple[str, bool]] = {
    "f1": ("f1", True),
    "f1_macro": ("f1_macro", True),
    "accuracy": ("accuracy", True),
    "roc_auc": ("roc_auc", True),
    "rmse": ("neg_root_mean_squared_error", False),
    "mae": ("neg_mean_absolute_error", False),
    "r2": ("r2", True),
}


def is_higher_better(metric: str) -> bool:
    """Whether a larger value of `metric` is better (e.g. False for rmse/mae)."""
    try:
        return _METRIC_REGISTRY[metric][1]
    except KeyError:
        raise ValueError(f"unknown metric: '{metric}'") from None


_DEFAULT_METRIC: dict[TaskType, str] = {
    "binary_classification": "f1",
    "multiclass_classification": "f1_macro",
    "regression": "rmse",
}

_EXCLUDED_INFERRED_TYPES = {"identifier", "constant", "empty", "text"}

_METRIC_FUNCTIONS = {
    # "f1" (binary) needs a pos_label — handled separately in evaluate_holdout
    # since it depends on the fitted pipeline's classes_.
    "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    "rmse": lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred),
    "mae": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
    "r2": lambda y_true, y_pred: r2_score(y_true, y_pred),
}


def _default_candidates(task_type: TaskType) -> dict[str, tuple[type, list[dict[str, Any]]]]:
    if task_type == "regression":
        return {
            "linear_regression": (LinearRegression, [{}]),
            "random_forest": (
                RandomForestRegressor,
                [{"n_estimators": 100, "max_depth": None}, {"n_estimators": 200, "max_depth": 8}],
            ),
            "gradient_boosting": (
                GradientBoostingRegressor,
                [{"n_estimators": 100, "learning_rate": 0.1}, {"n_estimators": 200, "learning_rate": 0.05}],
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression,
            [{"C": 1.0, "max_iter": 1000}, {"C": 0.1, "max_iter": 1000}],
        ),
        "random_forest": (
            RandomForestClassifier,
            [{"n_estimators": 100, "max_depth": None}, {"n_estimators": 200, "max_depth": 8}],
        ),
        "gradient_boosting": (
            GradientBoostingClassifier,
            [{"n_estimators": 100, "learning_rate": 0.1}, {"n_estimators": 200, "learning_rate": 0.05}],
        ),
    }


def _instantiate(estimator_class: type, params: dict[str, Any], seed: int) -> Any:
    kwargs = dict(params)
    if "random_state" in inspect.signature(estimator_class.__init__).parameters:
        kwargs["random_state"] = seed
    return estimator_class(**kwargs)


def _select_feature_columns(
    df: pd.DataFrame,
    target: str,
    profile: DatasetProfile | None,
    feature_columns: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Resolve which columns to model on.

    If `feature_columns` is given explicitly, it is trusted as-is. Otherwise,
    when a DatasetProfile is available, identifier/constant/empty/text
    columns are dropped automatically — the same audit that fed
    `detect_leakage` also shapes what AutoML is allowed to train on.
    """
    if feature_columns is not None:
        return [c for c in feature_columns if c != target], []

    excluded: list[str] = []
    selected: list[str] = []
    for col in df.columns:
        if col == target:
            continue
        if profile is not None:
            col_profile = profile.get(col)
            if col_profile is not None and col_profile.inferred_type in _EXCLUDED_INFERRED_TYPES:
                excluded.append(col)
                continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            excluded.append(col)
            continue
        selected.append(col)
    return selected, excluded


def _build_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_cols = df[feature_columns].select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers)


def build_pipeline(
    model_name: str,
    params: dict[str, Any],
    seed: int,
    task_type: TaskType,
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> Pipeline:
    """Build (unfit) the same preprocessing + estimator pipeline `run_automl` would use
    for this model/params/task_type combination.

    Exists so a saved JSON artifact (model_name, params, seed, feature_columns) can be
    turned back into a real pipeline and refit — see experiments/artifacts.py.
    """
    registry = _default_candidates(task_type)
    if model_name not in registry:
        raise ValueError(f"unknown model '{model_name}' for task_type '{task_type}'")
    estimator_class, _ = registry[model_name]
    return Pipeline(
        [
            ("preprocess", _build_preprocessor(train_df, feature_columns)),
            ("model", _instantiate(estimator_class, params, seed)),
        ]
    )


def evaluate_holdout(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, metric: str) -> float:
    """Score a fitted pipeline on held-out rows using `metric`."""
    if metric not in _METRIC_REGISTRY:
        raise ValueError(f"unknown metric: '{metric}'")
    if metric == "roc_auc":
        y_score = pipeline.predict_proba(X)[:, 1]
        return float(roc_auc_score(y, y_score))
    y_pred = pipeline.predict(X)
    if metric == "f1":
        return float(f1_score(y, y_pred, pos_label=pipeline.classes_[-1]))
    return float(_METRIC_FUNCTIONS[metric](y, y_pred))


def run_automl(
    df: pd.DataFrame,
    target: str,
    task_type: TaskType,
    split: SplitResult,
    *,
    profile: DatasetProfile | None = None,
    feature_columns: list[str] | None = None,
    metric: str | None = None,
    cv_folds: int = 5,
    seed: int = 42,
    dataset_version_id: str | None = None,
    artifact_dir: str | Path | None = "./data/artifacts",
    candidate_models: list[str] | None = None,
) -> AutoMLResult:
    """Search a handful of model/hyperparameter candidates via cross-validation on `split.train_indices` only.

    Raises:
        ValueError: if `target`/`task_type`/`metric`/`candidate_models` are invalid, if the split's
            train/test or train/validation index sets overlap, or if no usable feature columns remain.
    """
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in dataset")
    if task_type not in _DEFAULT_METRIC:
        raise ValueError(f"unknown task_type: '{task_type}'")

    train_idx = set(split.train_indices)
    test_idx = set(split.test_indices)
    val_idx = set(split.validation_indices)
    if train_idx & test_idx:
        raise ValueError("train and test indices overlap — refusing to run AutoML with a leaking split")
    if train_idx & val_idx:
        raise ValueError("train and validation indices overlap — refusing to run AutoML with a leaking split")

    metric = metric or _DEFAULT_METRIC[task_type]
    if metric not in _METRIC_REGISTRY:
        raise ValueError(f"unknown metric: '{metric}'")
    sklearn_scoring, higher_is_better = _METRIC_REGISTRY[metric]
    cv_scoring: str | Any = sklearn_scoring
    if metric == "f1":
        # sklearn's "f1" scorer defaults to pos_label=1, which breaks for
        # non-{0,1} binary targets (e.g. "<=50K"/">50K") — score explicitly
        # against the conventional positive label instead.
        cv_scoring = make_scorer(f1_score, pos_label=positive_label(df[target]))

    resolved_features, excluded = _select_feature_columns(df, target, profile, feature_columns)
    if not resolved_features:
        raise ValueError("no usable feature columns remain after exclusions")

    train_df = df.loc[split.train_indices]
    test_df = df.loc[split.test_indices] if split.test_indices else None
    val_df = df.loc[split.validation_indices] if split.validation_indices else None

    X_train, y_train = train_df[resolved_features], train_df[target]

    is_classification = task_type != "regression"
    cv = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        if is_classification
        else KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    )

    registry = _default_candidates(task_type)
    if candidate_models is not None:
        registry = {name: spec for name, spec in registry.items() if name in candidate_models}
        if not registry:
            raise ValueError(f"none of {candidate_models} are known candidate models for task '{task_type}'")

    started = time.perf_counter()
    candidates: list[ModelCandidateResult] = []
    best: dict[str, Any] | None = None

    for model_name, (estimator_class, param_grid) in registry.items():
        for params in param_grid:
            pipeline = Pipeline(
                [
                    ("preprocess", _build_preprocessor(train_df, resolved_features)),
                    ("model", _instantiate(estimator_class, params, seed)),
                ]
            )

            fit_started = time.perf_counter()
            raw_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=cv_scoring)
            fit_time = time.perf_counter() - fit_started

            sign = -1.0 if sklearn_scoring.startswith("neg_") else 1.0
            scores = (sign * raw_scores).tolist()
            cv_mean = float(np.mean(scores))
            cv_std = float(np.std(scores))

            candidates.append(
                ModelCandidateResult(
                    model_name=model_name,
                    params=params,
                    cv_mean=cv_mean,
                    cv_std=cv_std,
                    cv_scores=scores,
                    fit_time_seconds=fit_time,
                )
            )

            rank_key = cv_mean if higher_is_better else -cv_mean
            if best is None or rank_key > best["rank_key"]:
                best = {
                    "rank_key": rank_key,
                    "model_name": model_name,
                    "params": params,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                }

    assert best is not None  # registry is non-empty here — guarded above

    best_pipeline = build_pipeline(best["model_name"], best["params"], seed, task_type, train_df, resolved_features)
    best_pipeline.fit(X_train, y_train)

    validation_score = None
    if val_df is not None and len(val_df):
        validation_score = evaluate_holdout(best_pipeline, val_df[resolved_features], val_df[target], metric)

    test_score = None
    if test_df is not None and len(test_df):
        test_score = evaluate_holdout(best_pipeline, test_df[resolved_features], test_df[target], metric)

    runtime_seconds = time.perf_counter() - started
    experiment_id = f"exp_{uuid.uuid4().hex[:12]}"

    artifact_path = None
    if artifact_dir is not None:
        experiment_dir = Path(artifact_dir) / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = str(experiment_dir / "model.json")
        with open(artifact_path, "w") as f:
            json.dump(
                {
                    "experiment_id": experiment_id,
                    "task_type": task_type,
                    "metric": metric,
                    "model_name": best["model_name"],
                    "params": best["params"],
                    "feature_columns": resolved_features,
                    "target": target,
                    "cv_mean": best["cv_mean"],
                    "cv_std": best["cv_std"],
                    "seed": seed,
                },
                f,
                indent=2,
            )

    return AutoMLResult(
        experiment_id=experiment_id,
        dataset_version_id=dataset_version_id,
        task_type=task_type,
        metric=metric,
        feature_columns=resolved_features,
        excluded_columns=excluded,
        best_model=best["model_name"],
        best_params=best["params"],
        score=best["cv_mean"],
        cv_mean=best["cv_mean"],
        cv_std=best["cv_std"],
        validation_score=validation_score,
        test_score=test_score,
        candidates=candidates,
        runtime_seconds=runtime_seconds,
        artifact_path=artifact_path,
        seed=seed,
    )
