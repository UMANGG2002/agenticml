"""Typed record for a single tracked experiment (docs/spec.md section 10)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Conclusion = Literal["supported", "rejected", "inconclusive"]


class ExperimentRecord(BaseModel):
    experiment_id: str
    parent_experiment_id: str | None = None

    hypothesis: str
    objective: str | None = None

    dataset_version_id: str | None = None
    split_strategy: str
    features: list[str]
    excluded_features: list[str] = Field(default_factory=list)
    preprocessing: str

    model: str
    hyperparameters: dict[str, Any]
    seed: int

    metric: str
    metrics: dict[str, float]
    primary_score: float

    runtime_seconds: float
    artifact_path: str | None

    conclusion: Conclusion | None = None
    next_action: str | None = None

    created_at: str
