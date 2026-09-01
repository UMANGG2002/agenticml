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

from tools.data.schemas import (
    DatasetInspection,
    DatasetProfile,
    DataQualityReport,
    LeakageDetectionResult,
    SplitResult,
    TargetDetectionResult,
)


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
