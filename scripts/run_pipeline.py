#!/usr/bin/env python3
"""Run the Data Detective audit + AutoML baseline against any CSV.

This wires together everything built so far: profile -> detect/confirm
target -> leakage check (gated) -> split -> AutoML search. It's a thin CLI
around agent/orchestrator.py + tools/ml/automl.py — nothing here is new
logic, just a runnable entry point.

Examples:
    # First pass: just see what the agent finds (often pauses for input)
    uv run python scripts/run_pipeline.py dataset/train.csv --objective "predict price range"

    # Target was ambiguous -> confirm it explicitly
    uv run python scripts/run_pipeline.py dataset/train.csv \\
        --objective "predict price range" --target price_range

    # Leakage was found -> tell it which columns are legitimately post-outcome
    uv run python scripts/run_pipeline.py dataset/titanic.csv \\
        --objective "predict survival" --target survived \\
        --post-outcome-columns boat,body

    # Or force past a leakage block after reviewing it yourself, dropping columns
    uv run python scripts/run_pipeline.py dataset/titanic.csv \\
        --objective "predict survival" --target survived \\
        --exclude boat,body,name --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from agent.dataset_registry import DatasetRegistry
from agent.orchestrator import run_data_audit
from agent.state import AgentState
from tools.data.split import create_split
from tools.ml.automl import run_automl


def _csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="path to the dataset CSV")
    parser.add_argument("--objective", required=True, help="plain-English description of the ML goal")
    parser.add_argument("--target", default=None, help="target column, if already known")
    parser.add_argument(
        "--post-outcome-columns", default=None, help="comma-separated columns known to be recorded after the outcome"
    )
    parser.add_argument(
        "--exclude", default=None, help="comma-separated columns to force-drop from AutoML's feature set"
    )
    parser.add_argument(
        "--force", action="store_true", help="proceed to AutoML even if leakage checks found critical issues"
    )
    parser.add_argument("--split", choices=["random", "stratified", "group", "time"], default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=None)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--metric", default=None, help="override the default metric for the task type")
    parser.add_argument("--candidate-models", default=None, help="comma-separated subset of models to try")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, default=Path("data/artifacts"))
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"error: '{args.csv}' does not exist", file=sys.stderr)
        raise SystemExit(1)

    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv} — {df.shape[0]} rows x {df.shape[1]} columns")

    registry = DatasetRegistry()
    dataset_version_id = registry.register(df)
    state = AgentState(project_id="cli-run", dataset_version_id=dataset_version_id, objective=args.objective)

    post_outcome_columns = _csv_list(args.post_outcome_columns)
    audit = run_data_audit(registry, state, confirmed_target=args.target, post_outcome_columns=post_outcome_columns)

    print(f"\n=== data audit: {audit.status} ===")
    if audit.quality.issues:
        print("data quality issues:")
        for issue in audit.quality.issues:
            print(f"  [{issue.severity}] {issue.type}: {issue.columns} -- {issue.message}")

    if audit.status == "paused_for_target_confirmation":
        print(f"\n{audit.pending_question.question}")
        print("candidates:")
        for c in audit.pending_question.context["candidates"]:
            print(f"  {c['column']:20s} confidence={c['confidence']:.2f}  {c['reasons']}")
        print(f"\nRe-run with --target <column> (dataset: {args.csv})")
        raise SystemExit(2)

    if audit.status == "blocked_on_leakage" and not args.force:
        print("\ncritical leakage findings:")
        for f in audit.leakage.findings:
            if f.severity == "critical":
                print(f"  [{f.type}] {f.columns} -- {f.evidence}")
        print(
            "\nRe-run with --post-outcome-columns (if these are legitimately post-outcome), "
            "--exclude to drop them from AutoML's features, or --force to proceed anyway."
        )
        raise SystemExit(3)

    if audit.status == "blocked_on_leakage" and args.force:
        print("\n--force set: proceeding to AutoML despite critical leakage findings above.")

    target = audit.state.target_column
    task_type = audit.state.task_type
    if task_type is None:
        print(f"error: could not infer a task type for target '{target}' — pass --target explicitly", file=sys.stderr)
        raise SystemExit(1)

    excluded_by_flag = set(_csv_list(args.exclude) or [])
    feature_columns = None
    if excluded_by_flag:
        feature_columns = [
            c for c in df.columns if c != target and c not in excluded_by_flag
        ]

    split_strategy = args.split or ("stratified" if task_type != "regression" else "random")
    split = create_split(
        df,
        split_strategy,
        test_size=args.test_size,
        validation_size=args.validation_size,
        target=target if split_strategy == "stratified" else None,
        group_column=args.group_column,
        time_column=args.time_column,
        seed=args.seed,
    )
    if split.warnings:
        print("\nsplit warnings:", split.warnings)

    result = run_automl(
        df,
        target=target,
        task_type=task_type,
        split=split,
        profile=audit.profile if feature_columns is None else None,
        feature_columns=feature_columns,
        metric=args.metric,
        cv_folds=args.cv_folds,
        seed=args.seed,
        dataset_version_id=dataset_version_id,
        artifact_dir=args.artifact_dir,
        candidate_models=_csv_list(args.candidate_models),
    )

    print(f"\n=== automl: {result.experiment_id} ===")
    print(f"target={target!r}  task_type={task_type}  metric={result.metric}")
    print(f"features used ({len(result.feature_columns)}): {result.feature_columns}")
    if result.excluded_columns:
        print(f"features excluded: {result.excluded_columns}")
    print(f"best_model={result.best_model}  params={result.best_params}")
    print(f"cv_mean={result.cv_mean:.4f}  cv_std={result.cv_std:.4f}")
    if result.validation_score is not None:
        print(f"validation_score={result.validation_score:.4f}")
    if result.test_score is not None:
        print(f"test_score={result.test_score:.4f}")
    print(f"runtime_seconds={result.runtime_seconds:.2f}")
    print(f"artifact_path={result.artifact_path}")

    print("\ncandidates:")
    for c in sorted(result.candidates, key=lambda c: c.cv_mean, reverse=True):
        print(f"  {c.model_name:20s} {json.dumps(c.params):40s} cv_mean={c.cv_mean:.4f} cv_std={c.cv_std:.4f}")


if __name__ == "__main__":
    main()
