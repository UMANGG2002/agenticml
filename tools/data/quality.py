"""Data Detective — aggregate, actionable data-quality checks.

Builds on top of `profile_columns` output rather than re-scanning the
DataFrame column-by-column; only duplicate-row detection touches the raw
DataFrame directly.
"""

from __future__ import annotations

import pandas as pd

from tools.data.profile import profile_columns
from tools.data.schemas import DataQualityIssue, DataQualityReport, DatasetProfile


def check_data_quality(df: pd.DataFrame, profile: DatasetProfile | None = None) -> DataQualityReport:
    if profile is None:
        profile = profile_columns(df)

    issues: list[DataQualityIssue] = []
    n_rows = profile.n_rows

    empty_cols = [c.name for c in profile.columns if c.inferred_type == "empty"]
    if empty_cols:
        issues.append(
            DataQualityIssue(
                type="empty_columns",
                severity="high",
                columns=empty_cols,
                message=f"{len(empty_cols)} column(s) are entirely missing and carry no information.",
            )
        )

    constant_cols = [c.name for c in profile.columns if c.inferred_type == "constant"]
    if constant_cols:
        issues.append(
            DataQualityIssue(
                type="constant_columns",
                severity="medium",
                columns=constant_cols,
                message=f"{len(constant_cols)} column(s) have a single distinct value and carry no signal.",
            )
        )

    high_missing = [
        c.name for c in profile.columns if c.inferred_type != "empty" and c.missing_percentage > 50
    ]
    if high_missing:
        issues.append(
            DataQualityIssue(
                type="high_missing_columns",
                severity="medium",
                columns=high_missing,
                message=f"{len(high_missing)} column(s) are more than 50% missing.",
            )
        )

    identifier_cols = [c.name for c in profile.columns if c.inferred_type == "identifier"]
    if identifier_cols:
        issues.append(
            DataQualityIssue(
                type="identifier_columns",
                severity="low",
                columns=identifier_cols,
                message=(
                    f"{len(identifier_cols)} column(s) look like identifiers "
                    "and should typically be excluded from model features."
                ),
            )
        )

    placeholder_cols = [c.name for c in profile.columns if c.suspicious_values]
    if placeholder_cols:
        issues.append(
            DataQualityIssue(
                type="placeholder_missing_values",
                severity="medium",
                columns=placeholder_cols,
                message=(
                    f"{len(placeholder_cols)} column(s) contain placeholder-like strings "
                    "(e.g. 'NA', '?', 'unknown') that pandas does not treat as missing."
                ),
            )
        )

    high_card_cols = [
        c.name for c in profile.columns if c.categorical_stats and c.categorical_stats.is_high_cardinality
    ]
    if high_card_cols:
        issues.append(
            DataQualityIssue(
                type="high_cardinality_categorical",
                severity="low",
                columns=high_card_cols,
                message=(
                    f"{len(high_card_cols)} categorical column(s) have high cardinality — "
                    "prefer target/frequency encoding over one-hot."
                ),
            )
        )

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = (dup_count / n_rows * 100) if n_rows else 0.0
        severity = "high" if dup_pct >= 5 else "medium"
        issues.append(
            DataQualityIssue(
                type="duplicate_rows",
                severity=severity,
                columns=[],
                message=f"{dup_count} duplicate row(s) found ({dup_pct:.1f}% of the dataset).",
            )
        )

    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in issues:
        counts[issue.severity] += 1

    return DataQualityReport(
        issues=issues,
        n_critical=counts["critical"],
        n_high=counts["high"],
        n_medium=counts["medium"],
        n_low=counts["low"],
    )
