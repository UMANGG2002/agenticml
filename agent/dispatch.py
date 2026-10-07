"""Bridges LLM tool_use calls to the actual deterministic tools.

Every dispatch resolves `dataset_version_id` to a real DataFrame via
DatasetRegistry — the LLM only ever sees an id, never raw data (the
contract defined in agent/schemas.py). Results are the same Pydantic
models the deterministic orchestrator (agent/orchestrator.py) uses,
dumped to plain dicts for the tool_result content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.dataset_registry import DatasetRegistry
from experiments.artifacts import refit_from_experiment
from experiments.registry import ExperimentRegistry
from experiments.runner import run_experiment
from tools.data.leakage import detect_leakage
from tools.data.profile import detect_target, inspect_dataset, profile_columns
from tools.data.quality import check_data_quality
from tools.data.split import create_split
from tools.ml.errors import analyze_classification_errors, analyze_regression_errors
from tools.validation.critic import run_critic
from tools.validation.robustness import run_robustness_checks


class ToolDispatcher:
    def __init__(
        self,
        dataset_registry: DatasetRegistry,
        experiment_registry: ExperimentRegistry,
        artifact_dir: str | Path | None = "./data/artifacts",
    ) -> None:
        self._datasets = dataset_registry
        self._experiments = experiment_registry
        self._artifact_dir = artifact_dir
        # Cache for leakage results and robustness reports (keyed by experiment_id)
        self._leakage_cache: dict[str, Any] = {}
        self._robustness_cache: dict[str, Any] = {}
        self._error_analysis_cache: dict[str, Any] = {}

    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            raise ValueError(f"no handler registered for tool '{tool_name}'")
        return handler(tool_input)

    def _handle_inspect_dataset(self, args: dict[str, Any]) -> dict[str, Any]:
        df = self._datasets.get(args["dataset_version_id"])
        return inspect_dataset(df).model_dump()

    def _handle_profile_columns(self, args: dict[str, Any]) -> dict[str, Any]:
        df = self._datasets.get(args["dataset_version_id"])
        return profile_columns(df).model_dump()

    def _handle_detect_target(self, args: dict[str, Any]) -> dict[str, Any]:
        df = self._datasets.get(args["dataset_version_id"])
        return detect_target(df, objective=args.get("objective")).model_dump()

    def _handle_check_data_quality(self, args: dict[str, Any]) -> dict[str, Any]:
        df = self._datasets.get(args["dataset_version_id"])
        return check_data_quality(df).model_dump()

    def _handle_detect_leakage(self, args: dict[str, Any]) -> dict[str, Any]:
        df = self._datasets.get(args["dataset_version_id"])
        result = detect_leakage(
            df, target=args["target"], post_outcome_columns=args.get("post_outcome_columns")
        )
        # Cache for later use by the critic
        self._leakage_cache[args["dataset_version_id"]] = result
        return result.model_dump()

    def _handle_create_split(self, args: dict[str, Any]) -> dict[str, Any]:
        df = self._datasets.get(args["dataset_version_id"])
        return create_split(
            df,
            strategy=args["strategy"],
            test_size=args.get("test_size", 0.2),
            validation_size=args.get("validation_size"),
            target=args.get("target"),
            group_column=args.get("group_column"),
            time_column=args.get("time_column"),
            seed=args.get("seed", 42),
        ).model_dump()

    def _handle_run_experiment(self, args: dict[str, Any]) -> dict[str, Any]:
        df = self._datasets.get(args["dataset_version_id"])
        split_strategy = args.get("split_strategy", "random")
        split = create_split(
            df,
            strategy=split_strategy,
            test_size=args.get("test_size", 0.2),
            target=args["target"] if split_strategy == "stratified" else None,
            seed=args.get("seed", 42),
        )
        record = run_experiment(
            df,
            target=args["target"],
            task_type=args["task_type"],
            split=split,
            hypothesis=args["hypothesis"],
            registry=self._experiments,
            parent_experiment_id=args.get("parent_experiment_id"),
            profile=profile_columns(df),
            feature_columns=args.get("feature_columns"),
            metric=args.get("metric"),
            candidate_models=args.get("candidate_models"),
            seed=args.get("seed", 42),
            dataset_version_id=args["dataset_version_id"],
            artifact_dir=self._artifact_dir,
        )
        return record.model_dump()

    def _handle_analyze_errors(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run error analysis on a completed experiment by refitting the model."""
        experiment_id = args["experiment_id"]
        dataset_version_id = args["dataset_version_id"]
        subgroup_columns = args.get("subgroup_columns")

        experiment = self._experiments.get(experiment_id)
        df = self._datasets.get(dataset_version_id)

        # Recreate the split used in the experiment
        split = create_split(
            df,
            strategy=experiment.split_strategy or "random",
            test_size=0.2,
            target=experiment.features[0] if experiment.split_strategy == "stratified" else None,
            seed=experiment.seed or 42,
        )

        # Refit the model from its artifact
        pipeline = refit_from_experiment(experiment, df, split)
        feature_columns = experiment.features or [c for c in df.columns if c != experiment.metric]

        # Determine task type and run appropriate analysis
        test_df = df.loc[split.test_indices]
        target = df.columns[df.columns.isin([experiment.objective]) | (df.columns == experiment.model)][0] if experiment.objective else experiment.features[0]

        # Infer target from experiment record
        # The target isn't stored directly on ExperimentRecord, but we can find it
        # from the artifact metadata
        from experiments.artifacts import load_artifact
        if experiment.artifact_path:
            metadata = load_artifact(experiment.artifact_path)
            target = metadata.get("target", "")
            feature_columns = metadata.get("feature_columns", feature_columns)

        task_type = metadata.get("task_type", "binary_classification") if experiment.artifact_path else "binary_classification"

        if "classification" in task_type:
            result = analyze_classification_errors(
                pipeline, test_df, target, feature_columns,
                subgroup_columns=subgroup_columns,
            )
        else:
            result = analyze_regression_errors(
                pipeline, test_df, target, feature_columns,
                subgroup_columns=subgroup_columns,
            )

        self._error_analysis_cache[experiment_id] = result
        return result.model_dump()

    def _handle_run_robustness_checks(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run robustness checks on a completed experiment."""
        experiment_id = args["experiment_id"]
        dataset_version_id = args["dataset_version_id"]

        experiment = self._experiments.get(experiment_id)
        df = self._datasets.get(dataset_version_id)

        # Recreate the split
        split = create_split(
            df,
            strategy=experiment.split_strategy or "random",
            test_size=0.2,
            seed=experiment.seed or 42,
        )

        # Get target and feature info from artifact
        from experiments.artifacts import load_artifact
        target = ""
        feature_columns = experiment.features or []
        task_type = "binary_classification"
        if experiment.artifact_path:
            metadata = load_artifact(experiment.artifact_path)
            target = metadata.get("target", "")
            feature_columns = metadata.get("feature_columns", feature_columns)
            task_type = metadata.get("task_type", task_type)

        report = run_robustness_checks(
            df,
            target=target,
            task_type=task_type,
            split=split,
            reference_score=experiment.primary_score,
            metric=experiment.metric or "f1",
            feature_columns=feature_columns,
            model_name=experiment.model,
            seeds=args.get("seeds"),
            cv_std=experiment.metrics.get("cv_std") if experiment.metrics else None,
            time_column=args.get("time_column"),
            group_column=args.get("group_column"),
        )

        self._robustness_cache[experiment_id] = report
        return report.model_dump()

    def _handle_run_critic(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run the critic on a completed experiment, using cached analysis results."""
        experiment_id = args["experiment_id"]
        dataset_version_id = args["dataset_version_id"]

        experiment = self._experiments.get(experiment_id)

        review = run_critic(
            experiment,
            leakage_result=self._leakage_cache.get(dataset_version_id),
            robustness_report=self._robustness_cache.get(experiment_id),
            error_analysis=self._error_analysis_cache.get(experiment_id),
            all_experiments=list(self._experiments.all()),
        )

        return review.model_dump()
