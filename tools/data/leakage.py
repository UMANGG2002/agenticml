"""Data Detective — leakage detection.

Every check returns *evidence*, not just a boolean. Nothing here mutates or
drops columns automatically — the agent (or the user) decides what to do
with the findings.
"""

from __future__ import annotations

import pandas as pd

from tools.data.schemas import LeakageDetectionResult, LeakageFinding


def detect_leakage(
    df: pd.DataFrame,
    target: str,
    *,
    post_outcome_columns: list[str] | None = None,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    correlation_threshold: float = 0.95,
    near_perfect_threshold: float = 0.999,
) -> LeakageDetectionResult:
    """Detect likely data leakage between features and the target.

    Args:
        df: the dataset to audit (features + target).
        target: name of the target column.
        post_outcome_columns: columns explicitly known (via domain rules or
            metadata) to be recorded only after the outcome occurred. If not
            supplied, the post-outcome check is skipped rather than guessed.
        train_df / test_df: if both are supplied, checks for row overlap
            between the two splits. If either is missing, the check is
            skipped.
        correlation_threshold: |corr| at or above this is a "suspicious
            correlation" finding.
        near_perfect_threshold: |corr| at or above this (or perfect
            feature/target purity) is treated as a "target-derived feature".
    """
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in dataset")

    findings: list[LeakageFinding] = []
    checks_performed: list[str] = []
    checks_skipped: list[str] = []

    feature_columns = [c for c in df.columns if c != target]

    _check_target_derived_and_correlated(
        df, target, feature_columns, correlation_threshold, near_perfect_threshold, findings
    )
    checks_performed.append("target_derived_feature")
    checks_performed.append("suspicious_correlation")

    if post_outcome_columns:
        _check_post_outcome(post_outcome_columns, feature_columns, findings)
        checks_performed.append("post_outcome_feature")
    else:
        checks_skipped.append("post_outcome_feature (no post_outcome_columns metadata supplied)")

    _check_duplicate_rows(df, findings)
    checks_performed.append("duplicate_row_contamination")

    _check_near_duplicate_conflicting_labels(df, target, feature_columns, findings)
    checks_performed.append("near_duplicate_contamination")

    if train_df is not None and test_df is not None:
        _check_train_test_contamination(train_df, test_df, findings)
        checks_performed.append("train_test_contamination")
    else:
        checks_skipped.append("train_test_contamination (train_df and test_df were not both supplied)")

    return LeakageDetectionResult(
        findings=findings,
        checks_performed=checks_performed,
        checks_skipped=checks_skipped,
        has_critical_issues=any(f.severity == "critical" for f in findings),
    )


def _check_target_derived_and_correlated(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    correlation_threshold: float,
    near_perfect_threshold: float,
    findings: list[LeakageFinding],
) -> None:
    target_series = df[target]
    target_is_numeric = pd.api.types.is_numeric_dtype(target_series)

    for col in feature_columns:
        series = df[col]

        # Exact duplicate of the target (allowing for dtype differences).
        if len(series) == len(target_series) and series.astype(str).equals(target_series.astype(str)):
            findings.append(
                LeakageFinding(
                    type="target_derived_feature",
                    severity="critical",
                    columns=[col],
                    evidence=f"'{col}' is identical to the target column '{target}' for every row.",
                )
            )
            continue

        if target_is_numeric and pd.api.types.is_numeric_dtype(series):
            paired = pd.concat([series, target_series], axis=1).dropna()
            if len(paired) < 3:
                continue
            # A zero-variance side (e.g. a feature that's only ever non-null
            # for one target class) makes Pearson correlation undefined —
            # pandas would emit a RuntimeWarning and return NaN. Skip
            # up front rather than let numpy divide by zero.
            if paired.iloc[:, 0].std() == 0 or paired.iloc[:, 1].std() == 0:
                continue
            corr = paired.iloc[:, 0].corr(paired.iloc[:, 1])
            if corr is None or pd.isna(corr):
                continue
            abs_corr = abs(corr)
            if abs_corr >= near_perfect_threshold:
                findings.append(
                    LeakageFinding(
                        type="target_derived_feature",
                        severity="critical",
                        columns=[col],
                        evidence=f"'{col}' has |correlation|={abs_corr:.4f} with target '{target}' (near-perfect).",
                        metric_name="pearson_correlation",
                        metric_value=float(corr),
                    )
                )
            elif abs_corr >= correlation_threshold:
                findings.append(
                    LeakageFinding(
                        type="suspicious_correlation",
                        severity="high",
                        columns=[col],
                        evidence=f"'{col}' has |correlation|={abs_corr:.4f} with target '{target}'.",
                        metric_name="pearson_correlation",
                        metric_value=float(corr),
                    )
                )
        elif not target_is_numeric:
            purity = _feature_purity_against_target(series, target_series)
            if purity is None:
                continue
            if purity >= near_perfect_threshold:
                findings.append(
                    LeakageFinding(
                        type="target_derived_feature",
                        severity="critical",
                        columns=[col],
                        evidence=(
                            f"'{col}' values map to a single target class {purity * 100:.1f}% of the "
                            f"time — it near-perfectly predicts '{target}'."
                        ),
                        metric_name="purity",
                        metric_value=float(purity),
                    )
                )
            elif purity >= correlation_threshold:
                findings.append(
                    LeakageFinding(
                        type="suspicious_correlation",
                        severity="high",
                        columns=[col],
                        evidence=(
                            f"'{col}' values map to a single target class {purity * 100:.1f}% of the time."
                        ),
                        metric_name="purity",
                        metric_value=float(purity),
                    )
                )


def _feature_purity_against_target(feature: pd.Series, target: pd.Series) -> float | None:
    """Fraction of rows where the feature value maps to a single (dominant) target value.

    Only evaluated for low/medium-cardinality features — high-cardinality
    features (e.g. near-unique identifiers) trivially reach ~100% purity and
    would otherwise flood this check with false positives.
    """
    paired = pd.concat([feature, target], axis=1).dropna()
    paired.columns = ["feature", "target"]
    if paired.empty:
        return None
    n_unique_feature = paired["feature"].nunique()
    if n_unique_feature <= 1 or n_unique_feature > max(50, int(len(paired) * 0.2)):
        return None

    group_sizes = paired.groupby("feature")["target"].size()
    dominant_counts = paired.groupby("feature")["target"].agg(lambda s: s.value_counts().iloc[0])
    return float((dominant_counts.sum()) / group_sizes.sum())


def _check_post_outcome(
    post_outcome_columns: list[str], feature_columns: list[str], findings: list[LeakageFinding]
) -> None:
    present = [c for c in post_outcome_columns if c in feature_columns]
    if present:
        findings.append(
            LeakageFinding(
                type="post_outcome_feature",
                severity="critical",
                columns=present,
                evidence=(
                    f"{len(present)} column(s) are declared (via domain metadata) as only "
                    f"available after the outcome occurred: {present}."
                ),
            )
        )


def _check_duplicate_rows(df: pd.DataFrame, findings: list[LeakageFinding]) -> None:
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return
    pct = dup_count / len(df) * 100 if len(df) else 0.0
    severity = "high" if pct >= 5 else "medium"
    findings.append(
        LeakageFinding(
            type="duplicate_row_contamination",
            severity=severity,
            columns=[],
            evidence=f"{dup_count} exact duplicate row(s) found ({pct:.1f}% of the dataset).",
            metric_name="duplicate_row_percentage",
            metric_value=round(pct, 4),
        )
    )


def _check_near_duplicate_conflicting_labels(
    df: pd.DataFrame, target: str, feature_columns: list[str], findings: list[LeakageFinding]
) -> None:
    if not feature_columns:
        return
    hashable = df[feature_columns].astype(str)
    grouped_targets = df[target].groupby([hashable[c] for c in feature_columns]).nunique()
    conflicting_groups = int((grouped_targets > 1).sum())
    if conflicting_groups == 0:
        return
    conflicting_rows = int(
        df.groupby(feature_columns, dropna=False)[target].transform("nunique").gt(1).sum()
    )
    findings.append(
        LeakageFinding(
            type="near_duplicate_contamination",
            severity="high",
            columns=feature_columns,
            evidence=(
                f"{conflicting_groups} group(s) of rows with identical features "
                f"({conflicting_rows} row(s) total) have inconsistent target values — "
                "likely label noise or duplicated-but-mislabeled examples."
            ),
            metric_name="conflicting_row_count",
            metric_value=float(conflicting_rows),
        )
    )


def _check_train_test_contamination(
    train_df: pd.DataFrame, test_df: pd.DataFrame, findings: list[LeakageFinding]
) -> None:
    common_columns = [c for c in train_df.columns if c in test_df.columns]
    if not common_columns:
        return
    train_hashes = set(map(tuple, train_df[common_columns].astype(str).values.tolist()))
    test_rows = list(map(tuple, test_df[common_columns].astype(str).values.tolist()))
    overlap = sum(1 for row in test_rows if row in train_hashes)
    if overlap == 0:
        return
    pct = overlap / len(test_rows) * 100 if test_rows else 0.0
    findings.append(
        LeakageFinding(
            type="train_test_contamination",
            severity="critical",
            columns=common_columns,
            evidence=f"{overlap} test row(s) ({pct:.1f}% of the test set) also appear in the training set.",
            metric_name="overlap_percentage",
            metric_value=round(pct, 4),
        )
    )
