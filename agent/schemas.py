"""The tool-calling contract between the agent and the deterministic tools.

Each entry in TOOL_REGISTRY pairs:
- an *Args* model — what the caller must supply. Its `model_json_schema()`
  becomes the function-calling `parameters` schema.
- a *Returns* model — the typed Pydantic output from tools/data/*. Its
  `model_json_schema()` tells the caller what shape to expect back.

Tool arguments are always addressed by `dataset_version_id`, never a raw
DataFrame, so the contract stays JSON-serializable end to end. Resolving
that id to an actual DataFrame is the orchestrator's job (see
agent/dataset_registry.py), not the caller's.

This registry has no LLM wiring yet — it exists so the contract is defined
and typed *before* an LLM is put in the loop (docs/spec.md section 28:
"Do not parse free-form text when a decision can be structured.").
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from experiments.schemas import ExperimentRecord
from tools.data.schemas import (
    DatasetInspection,
    DatasetProfile,
    DataQualityReport,
    LeakageDetectionResult,
    SplitResult,
    TargetDetectionResult,
)
from tools.ml.schemas import ClassificationErrorAnalysis, RegressionErrorAnalysis
from tools.validation.critic import CriticReview
from tools.validation.schemas import RobustnessReport


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]


class InspectDatasetArgs(BaseModel):
    dataset_version_id: str


class ProfileColumnsArgs(BaseModel):
    dataset_version_id: str


class DetectTargetArgs(BaseModel):
    dataset_version_id: str
    objective: str | None = None


class CheckDataQualityArgs(BaseModel):
    dataset_version_id: str


class DetectLeakageArgs(BaseModel):
    dataset_version_id: str
    target: str
    post_outcome_columns: list[str] | None = None


class CreateSplitArgs(BaseModel):
    dataset_version_id: str
    strategy: str
    test_size: float = 0.2
    validation_size: float | None = None
    target: str | None = None
    group_column: str | None = None
    time_column: str | None = None
    seed: int = 42


class RunExperimentArgs(BaseModel):
    dataset_version_id: str
    target: str
    task_type: str
    hypothesis: str
    parent_experiment_id: str | None = None
    split_strategy: str = "random"
    test_size: float = 0.2
    feature_columns: list[str] | None = None
    metric: str | None = None
    candidate_models: list[str] | None = None
    seed: int = 42


class AnalyzeErrorsArgs(BaseModel):
    experiment_id: str
    dataset_version_id: str
    subgroup_columns: list[str] | None = None


class RunRobustnessChecksArgs(BaseModel):
    experiment_id: str
    dataset_version_id: str
    seeds: list[int] | None = None
    time_column: str | None = None
    group_column: str | None = None


class RunCriticArgs(BaseModel):
    experiment_id: str
    dataset_version_id: str


class GenerateReportArgs(BaseModel):
    """Generate a final report for the current project."""
    pass  # Uses state — no extra args needed


class GetDomainKnowledgeArgs(BaseModel):
    query: str
    top_k: int = 5


def _tool(
    name: str, description: str, args_model: type[BaseModel], returns_model: type[BaseModel]
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters=args_model.model_json_schema(),
        returns=returns_model.model_json_schema(),
    )


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "inspect_dataset": _tool(
        "inspect_dataset",
        "Structural overview of a dataset: shape, dtypes, sample rows, duplicate row count.",
        InspectDatasetArgs,
        DatasetInspection,
    ),
    "profile_columns": _tool(
        "profile_columns",
        "Per-column statistics, inferred type, and quality warnings.",
        ProfileColumnsArgs,
        DatasetProfile,
    ),
    "detect_target": _tool(
        "detect_target",
        "Propose candidate target columns with confidence and reasons. Never silently "
        "commits to a target — check `requires_user_confirmation` before proceeding.",
        DetectTargetArgs,
        TargetDetectionResult,
    ),
    "check_data_quality": _tool(
        "check_data_quality",
        "Aggregated, actionable data-quality issues by severity.",
        CheckDataQualityArgs,
        DataQualityReport,
    ),
    "detect_leakage": _tool(
        "detect_leakage",
        "Detect target-derived features, suspicious correlation, post-outcome features, "
        "duplicate/near-duplicate contamination, and train/test contamination. Check "
        "`has_critical_issues` before proceeding to modeling.",
        DetectLeakageArgs,
        LeakageDetectionResult,
    ),
    "create_split": _tool(
        "create_split",
        "Create a train/validation/test split using a random, stratified, group, or "
        "time-based strategy.",
        CreateSplitArgs,
        SplitResult,
    ),
    "run_experiment": _tool(
        "run_experiment",
        "Run an AutoML search as a tracked experiment. `hypothesis` is required — every "
        "experiment needs a reason. Pass `parent_experiment_id` to compare against a prior "
        "experiment; the recorded `conclusion` (supported/rejected/inconclusive) is decided "
        "by comparing held-out scores, never by re-optimizing against the test set.",
        RunExperimentArgs,
        ExperimentRecord,
    ),
    "analyze_errors": _tool(
        "analyze_errors",
        "Run error analysis on a completed experiment: confusion matrix, per-class metrics, "
        "threshold curve (binary classification), subgroup performance, and example errors. "
        "For regression: residual distribution, subgroup RMSE, and extreme-error examples. "
        "Requires a completed experiment_id to refit the model from its artifact.",
        AnalyzeErrorsArgs,
        ClassificationErrorAnalysis,  # or RegressionErrorAnalysis — union at runtime
    ),
    "run_robustness_checks": _tool(
        "run_robustness_checks",
        "Run robustness checks on a completed experiment: CV variance, multi-seed evaluation, "
        "alternate split strategy (temporal/group validation), and noise sensitivity. "
        "Returns evidence of where the model degrades under varied conditions.",
        RunRobustnessChecksArgs,
        RobustnessReport,
    ),
    "run_critic": _tool(
        "run_critic",
        "Run the deterministic critic/falsification layer on a completed experiment. "
        "Checks for leakage, overfitting, CV variance, seed sensitivity, metric "
        "appropriateness, subgroup disparities, and reproducibility. Returns a verdict "
        "(approve/conditional/reject) with evidence. A critical rejection blocks model promotion.",
        RunCriticArgs,
        CriticReview,
    ),
}


class AgentDecision(BaseModel):
    """A single structured agent decision (docs/spec.md section 28)."""

    action: str
    hypothesis: str | None = None
    reason: str
    tool: str | None = None
    parameters: dict[str, Any] = {}
    expected_metric: dict[str, Any] | None = None
    estimated_cost: float | None = None


class PendingQuestion(BaseModel):
    """Raised when the orchestrator cannot proceed without human input."""

    reason: str
    question: str
    context: dict[str, Any] = {}
