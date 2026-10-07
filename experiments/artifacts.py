"""Reconstruct a usable fitted model from a recorded experiment.

Artifacts are saved as plain JSON metadata (tools/ml/automl.py), not a
pickled model — no joblib/pickle involved, so artifacts are inspectable,
portable, and never execute arbitrary code on load. The trade-off: getting
a fitted model back means refitting it, not deserializing it. Since the
artifact records the exact model, hyperparameters, features, and seed,
that refit is deterministic and reproduces the original fit exactly —
callers just need to supply the same training rows the experiment used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from experiments.registry import ExperimentRegistry
from experiments.schemas import ExperimentRecord
from tools.data.schemas import SplitResult
from tools.ml.automl import build_pipeline


def load_artifact(artifact_path: str | Path) -> dict:
    """Read a saved AutoML artifact's JSON metadata."""
    with open(artifact_path) as f:
        return json.load(f)


def refit_from_experiment(experiment: ExperimentRecord, df: pd.DataFrame, split: SplitResult) -> Pipeline:
    """Refit the exact model `experiment` recorded, from its JSON artifact plus the training data.

    `df` and `split` must be the same dataset/split the experiment was originally run
    against — neither is persisted on `ExperimentRecord`, so it's the caller's
    responsibility to supply the matching ones (e.g. the same `create_split(...)` call).

    Raises:
        ValueError: if the experiment has no saved artifact to refit from.
    """
    if experiment.artifact_path is None:
        raise ValueError(f"experiment '{experiment.experiment_id}' has no saved artifact to refit from")

    metadata = load_artifact(experiment.artifact_path)
    train_df = df.loc[split.train_indices]
    feature_columns = metadata["feature_columns"]

    pipeline = build_pipeline(
        model_name=metadata["model_name"],
        params=metadata["params"],
        seed=metadata["seed"],
        task_type=metadata["task_type"],
        train_df=train_df,
        feature_columns=feature_columns,
    )
    pipeline.fit(train_df[feature_columns], train_df[metadata["target"]])
    return pipeline


def refit_from_registry(
    registry: ExperimentRegistry, experiment_id: str, df: pd.DataFrame, split: SplitResult
) -> Pipeline:
    """Convenience wrapper: look up `experiment_id` in `registry`, then refit it."""
    return refit_from_experiment(registry.get(experiment_id), df, split)
