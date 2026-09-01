"""Typed output for the AutoML tool."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal["binary_classification", "multiclass_classification", "regression"]


class ModelCandidateResult(BaseModel):
    model_name: str
    params: dict[str, Any]
    cv_mean: float
    cv_std: float
    cv_scores: list[float]
    fit_time_seconds: float


class AutoMLResult(BaseModel):
    experiment_id: str
    dataset_version_id: str | None
    task_type: TaskType
    metric: str
    feature_columns: list[str]
    excluded_columns: list[str] = Field(default_factory=list)

    best_model: str
    best_params: dict[str, Any]
    score: float
    cv_mean: float
    cv_std: float

    validation_score: float | None = None
    test_score: float | None = None

    candidates: list[ModelCandidateResult]
    runtime_seconds: float
    artifact_path: str | None
    seed: int


# --------------------------------------------------------------------------
# error analysis
# --------------------------------------------------------------------------


class ConfusionMatrix(BaseModel):
    labels: list[str]
    matrix: list[list[int]]  # matrix[i][j] = count of true label i predicted as label j


class ClassMetrics(BaseModel):
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class ThresholdPoint(BaseModel):
    threshold: float
    precision: float
    recall: float
    f1: float


class SubgroupMetric(BaseModel):
    column: str
    group: str
    n: int
    metric_name: str
    metric_value: float
    warning: str | None = None


class ErrorExample(BaseModel):
    index: int
    true_value: Any
    predicted_value: Any
    residual: float | None = None


class ClassificationErrorAnalysis(BaseModel):
    n_rows: int
    confusion_matrix: ConfusionMatrix
    per_class_metrics: list[ClassMetrics]
    threshold_curve: list[ThresholdPoint] = Field(default_factory=list)
    subgroup_metrics: list[SubgroupMetric] = Field(default_factory=list)
    false_positives: list[ErrorExample] = Field(default_factory=list)
    false_negatives: list[ErrorExample] = Field(default_factory=list)


class RegressionErrorAnalysis(BaseModel):
    n_rows: int
    residual_mean: float
    residual_std: float
    residual_min: float
    residual_max: float
    residual_q1: float
    residual_q3: float
    subgroup_metrics: list[SubgroupMetric] = Field(default_factory=list)
    extreme_errors: list[ErrorExample] = Field(default_factory=list)
