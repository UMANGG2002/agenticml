"""Data Detective — dataset inspection, column profiling, and target detection.

All functions here are deterministic and read-only: they never mutate the
input DataFrame and never guess silently where guessing would be unsafe
(see `detect_target`).
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from tools.data.schemas import (
    CategoricalStats,
    ColumnProfile,
    DatasetInspection,
    DatasetProfile,
    InferredColumnType,
    NumericStats,
    TargetCandidate,
    TargetDetectionResult,
    TaskType,
)

_PLACEHOLDER_MISSING_TOKENS = {
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "?",
    "-",
    "--",
    "unknown",
    "missing",
    "",
}

_TARGET_LIKE_NAMES = {
    "target",
    "label",
    "labels",
    "y",
    "class",
    "outcome",
    "churn",
    "result",
    "response",
    "fraud",
    "default",
    "survived",
    "approved",
    "status",
}

_TARGET_LIKE_KEYWORDS = (
    "target",
    "label",
    "churn",
    "fraud",
    "default",
    "outcome",
    "flag",
    "is_",
    "result",
    "class",
)

_IDENTIFIER_KEYWORDS = ("id", "uuid", "guid", "index", "key")


def _json_safe(value):
    """Convert a pandas/numpy scalar into a JSON/Pydantic-safe native value."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating,)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def inspect_dataset(df: pd.DataFrame, sample_size: int = 5) -> DatasetInspection:
    """Structural overview of a dataset: shape, dtypes, sample rows, duplicates."""
    sample_records = df.head(sample_size).to_dict(orient="records")
    sample = [{k: _json_safe(v) for k, v in row.items()} for row in sample_records]

    return DatasetInspection(
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        columns=list(df.columns.astype(str)),
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        sample=sample,
        memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
        duplicate_row_count=int(df.duplicated().sum()),
    )


def _infer_column_type(series: pd.Series, n_rows: int) -> InferredColumnType:
    non_null = series.dropna()

    if n_rows == 0 or non_null.empty:
        return "empty"
    if non_null.nunique() <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    unique_pct = non_null.nunique() / len(non_null) * 100
    if unique_pct >= 95 and len(non_null) > 20:
        return "identifier"

    avg_len = non_null.astype(str).str.len().mean()
    if avg_len is not None and avg_len > 30:
        return "text"
    return "categorical"


def _numeric_stats(series: pd.Series) -> NumericStats:
    non_null = pd.to_numeric(series.dropna(), errors="coerce")
    finite = non_null[np.isfinite(non_null)]
    if finite.empty:
        return NumericStats(
            infinite_count=int(np.isinf(non_null).sum()),
        )
    return NumericStats(
        min=float(finite.min()),
        max=float(finite.max()),
        mean=float(finite.mean()),
        median=float(finite.median()),
        std=float(finite.std()) if len(finite) > 1 else 0.0,
        q1=float(finite.quantile(0.25)),
        q3=float(finite.quantile(0.75)),
        zero_count=int((finite == 0).sum()),
        negative_count=int((finite < 0).sum()),
        infinite_count=int(np.isinf(non_null).sum()),
    )


def _categorical_stats(series: pd.Series, high_cardinality_threshold: int = 50) -> CategoricalStats:
    non_null = series.dropna().astype(str)
    counts = non_null.value_counts().head(10)
    n_categories = int(non_null.nunique())
    return CategoricalStats(
        top_values={str(k): int(v) for k, v in counts.items()},
        n_categories=n_categories,
        is_high_cardinality=n_categories > high_cardinality_threshold,
    )


def _suspicious_placeholder_values(series: pd.Series) -> list[str]:
    non_null = series.dropna()
    is_stringlike = (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )
    if non_null.empty or not is_stringlike:
        return []
    found = set()
    for value in non_null.astype(str).unique():
        if value.strip().lower() in _PLACEHOLDER_MISSING_TOKENS:
            found.add(value)
    return sorted(found)


def profile_columns(df: pd.DataFrame) -> DatasetProfile:
    """Per-column descriptive statistics, inferred type, and quality warnings."""
    n_rows = int(df.shape[0])
    columns: list[ColumnProfile] = []

    for name in df.columns:
        series = df[name]
        missing_count = int(series.isna().sum())
        missing_pct = (missing_count / n_rows * 100) if n_rows else 0.0
        unique_count = int(series.dropna().nunique())
        unique_pct = (unique_count / n_rows * 100) if n_rows else 0.0

        inferred_type = _infer_column_type(series, n_rows)

        numeric_stats = None
        categorical_stats = None
        if inferred_type == "numeric":
            numeric_stats = _numeric_stats(series)
        elif inferred_type in ("categorical", "text", "identifier"):
            categorical_stats = _categorical_stats(series)

        suspicious_values = _suspicious_placeholder_values(series)

        warnings: list[str] = []
        if suspicious_values:
            warnings.append(
                f"contains {len(suspicious_values)} placeholder-like value(s) not recognized "
                f"as missing by pandas: {suspicious_values}"
            )
        if missing_pct > 50:
            warnings.append(f"{missing_pct:.1f}% missing — consider dropping or imputing carefully")
        if inferred_type == "constant":
            warnings.append("column has a single distinct value and carries no signal")
        if inferred_type == "empty":
            warnings.append("column is entirely missing")
        if numeric_stats and numeric_stats.infinite_count > 0:
            warnings.append(f"contains {numeric_stats.infinite_count} infinite value(s)")
        if categorical_stats and categorical_stats.is_high_cardinality:
            warnings.append(
                f"high-cardinality categorical ({categorical_stats.n_categories} categories) — "
                "consider target/frequency encoding rather than one-hot"
            )
        if inferred_type == "identifier":
            warnings.append("looks like an identifier column — exclude from modeling features")

        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_percentage=round(missing_pct, 4),
                unique_count=unique_count,
                unique_percentage=round(unique_pct, 4),
                numeric_stats=numeric_stats,
                categorical_stats=categorical_stats,
                suspicious_values=suspicious_values,
                warnings=warnings,
            )
        )

    return DatasetProfile(n_rows=n_rows, columns=columns)


def _infer_task_type(col: ColumnProfile) -> TaskType:
    if col.inferred_type == "boolean":
        return "binary_classification"
    if col.inferred_type == "categorical":
        if col.unique_count == 2:
            return "binary_classification"
        if col.unique_count >= 3:
            return "multiclass_classification"
        return "unknown"
    if col.inferred_type == "numeric":
        if col.unique_count == 2:
            return "binary_classification"
        if col.unique_count <= 20 and col.unique_percentage < 5:
            return "multiclass_classification"
        return "regression"
    return "unknown"


def detect_target(
    df: pd.DataFrame,
    profile: DatasetProfile | None = None,
    objective: str | None = None,
) -> TargetDetectionResult:
    """Propose candidate target columns with confidence and reasons.

    Never silently commits to a target: `requires_user_confirmation` is True
    whenever the top candidate is not clearly and confidently the best one.
    """
    if profile is None:
        profile = profile_columns(df)

    candidates: list[TargetCandidate] = []
    last_column_name = profile.columns[-1].name if profile.columns else None
    objective_lower = objective.lower() if objective else None

    for col in profile.columns:
        if col.inferred_type in ("empty", "constant", "identifier"):
            continue

        score = 0.0
        reasons: list[str] = []
        name_lower = col.name.lower()

        if name_lower in _TARGET_LIKE_NAMES:
            score += 0.4
            reasons.append(f"column name '{col.name}' matches a common target name")
        elif any(kw in name_lower for kw in _TARGET_LIKE_KEYWORDS):
            score += 0.25
            reasons.append(f"column name '{col.name}' contains a target-like keyword")

        if any(kw == name_lower or name_lower.endswith(f"_{kw}") for kw in _IDENTIFIER_KEYWORDS):
            score -= 0.3
            reasons.append(f"column name '{col.name}' looks like an identifier, unlikely to be a target")

        if col.name == last_column_name:
            score += 0.15
            reasons.append("column is the last column in the dataset (common target convention)")

        if col.missing_percentage == 0:
            score += 0.05
            reasons.append("column has no missing values")
        elif col.missing_percentage > 5:
            score -= 0.2
            reasons.append(f"column has {col.missing_percentage:.1f}% missing values, unusual for a target")

        if col.inferred_type in ("categorical", "boolean"):
            if col.unique_count == 2:
                score += 0.2
                reasons.append("exactly 2 unique values suggests a binary classification target")
            elif 3 <= col.unique_count <= 20:
                score += 0.1
                reasons.append(f"{col.unique_count} unique values suggests a multiclass classification target")
        elif col.inferred_type == "numeric":
            if col.unique_count == 2:
                score += 0.15
                reasons.append("numeric column with exactly 2 unique values suggests a binary flag target")
            elif col.unique_percentage > 30:
                score += 0.1
                reasons.append("high-cardinality numeric column suggests a regression target")

        if objective_lower:
            # Whole-word overlap only: short substrings (e.g. "m", "h", "n")
            # would otherwise match almost any sentence via `in`.
            objective_words = set(re.findall(r"[a-z0-9]+", objective_lower))
            name_tokens = [t for t in re.split(r"[_\s]+", name_lower) if len(t) >= 3]
            matched = [t for t in name_tokens if t in objective_words]
            if matched:
                score += 0.2
                reasons.append(f"column name overlaps with the stated objective: '{objective}' (matched: {matched})")

        score = max(0.0, min(1.0, score))
        if score <= 0:
            continue

        candidates.append(
            TargetCandidate(
                column=col.name,
                confidence=round(score, 3),
                reasons=reasons,
                inferred_task_type=_infer_task_type(col),
            )
        )

    candidates.sort(key=lambda c: c.confidence, reverse=True)

    if not candidates:
        return TargetDetectionResult(
            candidates=[],
            recommended=None,
            requires_user_confirmation=True,
            message="No plausible target column could be identified — ask the user to specify one.",
        )

    recommended = candidates[0]
    second_best = candidates[1].confidence if len(candidates) > 1 else 0.0
    confident = recommended.confidence >= 0.6 and (recommended.confidence - second_best) >= 0.15

    return TargetDetectionResult(
        candidates=candidates,
        recommended=recommended,
        requires_user_confirmation=not confident,
        message=None if confident else "Target confidence is low or ambiguous — ask the user to confirm.",
    )
