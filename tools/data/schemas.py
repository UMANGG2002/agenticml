"""Typed, deterministic output schemas for the Data Detective tools.

These are the contracts the agent (and the API) reason over — every dataset
tool returns one of these models rather than free-form text or a raw dict.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

InferredColumnType = Literal[
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "identifier",
    "constant",
    "empty",
]

TaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "unknown",
]

LeakageType = Literal[
    "target_derived_feature",
    "suspicious_correlation",
    "post_outcome_feature",
    "duplicate_row_contamination",
    "near_duplicate_contamination",
    "train_test_contamination",
]

Severity = Literal["low", "medium", "high", "critical"]

SplitStrategy = Literal["random", "stratified", "group", "time"]


# --------------------------------------------------------------------------
# inspect_dataset
# --------------------------------------------------------------------------


class DatasetInspection(BaseModel):
    n_rows: int
    n_columns: int
    columns: list[str]
    dtypes: dict[str, str]
    sample: list[dict[str, Any]]
    memory_usage_bytes: int
    duplicate_row_count: int


# --------------------------------------------------------------------------
# profile_columns
# --------------------------------------------------------------------------


class NumericStats(BaseModel):
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    q1: float | None = None
    q3: float | None = None
    zero_count: int = 0
    negative_count: int = 0
    infinite_count: int = 0


class CategoricalStats(BaseModel):
    top_values: dict[str, int]
    n_categories: int
    is_high_cardinality: bool


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    inferred_type: InferredColumnType
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    suspicious_values: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    n_rows: int
    columns: list[ColumnProfile]

    def get(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)


# --------------------------------------------------------------------------
# detect_target
# --------------------------------------------------------------------------


class TargetCandidate(BaseModel):
    column: str
    confidence: float
    reasons: list[str]
    inferred_task_type: TaskType


class TargetDetectionResult(BaseModel):
    candidates: list[TargetCandidate]
    recommended: TargetCandidate | None
    requires_user_confirmation: bool
    message: str | None = None


# --------------------------------------------------------------------------
# check_data_quality
# --------------------------------------------------------------------------


class DataQualityIssue(BaseModel):
    type: str
    severity: Severity
    columns: list[str]
    message: str


class DataQualityReport(BaseModel):
    issues: list[DataQualityIssue]
    n_critical: int
    n_high: int
    n_medium: int
    n_low: int


# --------------------------------------------------------------------------
# detect_leakage
# --------------------------------------------------------------------------


class LeakageFinding(BaseModel):
    type: LeakageType
    severity: Severity
    columns: list[str]
    evidence: str
    metric_name: str | None = None
    metric_value: float | None = None


class LeakageDetectionResult(BaseModel):
    findings: list[LeakageFinding]
    checks_performed: list[str]
    checks_skipped: list[str]
    has_critical_issues: bool


# --------------------------------------------------------------------------
# create_split
# --------------------------------------------------------------------------


class SplitResult(BaseModel):
    strategy: SplitStrategy
    train_indices: list[int]
    validation_indices: list[int] = Field(default_factory=list)
    test_indices: list[int]
    seed: int
    warnings: list[str] = Field(default_factory=list)
