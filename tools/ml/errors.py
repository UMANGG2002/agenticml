"""Deterministic error analysis over a fitted model's predictions
(docs/spec.md section 16). No LLM here — every finding is evidence for the
agent's hypothesis engine to read, not a conclusion in itself.

Both functions take an already-fitted pipeline (e.g. from an
AutoMLResult's artifact) and the rows to evaluate it on — they don't fit
anything themselves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from tools.ml.labels import positive_label
from tools.ml.schemas import (
    ClassificationErrorAnalysis,
    ClassMetrics,
    ConfusionMatrix,
    ErrorExample,
    RegressionErrorAnalysis,
    SubgroupMetric,
    ThresholdPoint,
)

_MIN_SUBGROUP_SIZE = 20
_THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def analyze_classification_errors(
    pipeline: Any,
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    *,
    subgroup_columns: list[str] | None = None,
    max_examples: int = 5,
) -> ClassificationErrorAnalysis:
    """Confusion matrix, per-class metrics, threshold curve, subgroup accuracy, and example errors.

    For binary tasks, the alphabetically/numerically *last* label (e.g. 1
    over 0, "Yes" over "No") is treated as positive for the threshold curve
    and false-positive/negative examples — there's no explicit "positive
    class" metadata to go on otherwise.
    """
    X = df[feature_columns]
    y_true = df[target]
    y_pred = pd.Series(pipeline.predict(X), index=df.index)

    labels = sorted(pd.unique(pd.concat([y_true, y_pred])), key=str)
    label_strs = [str(label) for label in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = [
        ClassMetrics(
            label=str(labels[i]),
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i in range(len(labels))
    ]

    is_binary = len(labels) == 2 and hasattr(pipeline, "predict_proba")
    threshold_curve: list[ThresholdPoint] = []
    false_positives: list[ErrorExample] = []
    false_negatives: list[ErrorExample] = []

    if is_binary:
        pos_label = positive_label(labels)
        neg_label = labels[0] if pos_label == labels[1] else labels[1]
        proba = pipeline.predict_proba(X)[:, list(pipeline.classes_).index(pos_label)]
        y_true_binary = (y_true == pos_label).astype(int)

        for t in _THRESHOLD_GRID:
            y_pred_t = (proba >= t).astype(int)
            p, r, f, _ = precision_recall_fscore_support(y_true_binary, y_pred_t, average="binary", zero_division=0)
            threshold_curve.append(ThresholdPoint(threshold=t, precision=float(p), recall=float(r), f1=float(f)))

        fp_mask = (y_pred == pos_label) & (y_true == neg_label)
        fn_mask = (y_pred == neg_label) & (y_true == pos_label)
        for idx in df.index[fp_mask][:max_examples]:
            false_positives.append(
                ErrorExample(index=int(idx), true_value=_json_safe(neg_label), predicted_value=_json_safe(pos_label))
            )
        for idx in df.index[fn_mask][:max_examples]:
            false_negatives.append(
                ErrorExample(index=int(idx), true_value=_json_safe(pos_label), predicted_value=_json_safe(neg_label))
            )

    subgroup_metrics: list[SubgroupMetric] = []
    if subgroup_columns:
        correct = (y_true.to_numpy() == y_pred.to_numpy())
        for col in subgroup_columns:
            for group_value, group_index in df.groupby(col, dropna=False).groups.items():
                positions = df.index.get_indexer(group_index)
                n = len(positions)
                accuracy = float(correct[positions].mean())
                subgroup_metrics.append(
                    SubgroupMetric(
                        column=col,
                        group=str(group_value),
                        n=n,
                        metric_name="accuracy",
                        metric_value=accuracy,
                        warning="small sample size" if n < _MIN_SUBGROUP_SIZE else None,
                    )
                )

    return ClassificationErrorAnalysis(
        n_rows=len(df),
        confusion_matrix=ConfusionMatrix(labels=label_strs, matrix=cm.tolist()),
        per_class_metrics=per_class,
        threshold_curve=threshold_curve,
        subgroup_metrics=subgroup_metrics,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def analyze_regression_errors(
    pipeline: Any,
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    *,
    subgroup_columns: list[str] | None = None,
    max_examples: int = 5,
) -> RegressionErrorAnalysis:
    """Residual distribution, subgroup RMSE, and the largest-error examples."""
    X = df[feature_columns]
    y_true = df[target].to_numpy(dtype=float)
    y_pred = np.asarray(pipeline.predict(X), dtype=float)
    residuals = y_true - y_pred

    subgroup_metrics: list[SubgroupMetric] = []
    if subgroup_columns:
        for col in subgroup_columns:
            for group_value, group_index in df.groupby(col, dropna=False).groups.items():
                positions = df.index.get_indexer(group_index)
                n = len(positions)
                rmse = float(np.sqrt(np.mean(residuals[positions] ** 2)))
                subgroup_metrics.append(
                    SubgroupMetric(
                        column=col,
                        group=str(group_value),
                        n=n,
                        metric_name="rmse",
                        metric_value=rmse,
                        warning="small sample size" if n < _MIN_SUBGROUP_SIZE else None,
                    )
                )

    order = np.argsort(-np.abs(residuals))[:max_examples]
    extreme_errors = [
        ErrorExample(
            index=int(df.index[i]),
            true_value=_json_safe(float(y_true[i])),
            predicted_value=_json_safe(float(y_pred[i])),
            residual=float(residuals[i]),
        )
        for i in order
    ]

    return RegressionErrorAnalysis(
        n_rows=len(df),
        residual_mean=float(np.mean(residuals)),
        residual_std=float(np.std(residuals)),
        residual_min=float(np.min(residuals)),
        residual_max=float(np.max(residuals)),
        residual_q1=float(np.percentile(residuals, 25)),
        residual_q3=float(np.percentile(residuals, 75)),
        subgroup_metrics=subgroup_metrics,
        extreme_errors=extreme_errors,
    )
