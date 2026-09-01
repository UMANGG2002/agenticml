"""Deterministic orchestrator for the data-audit phase of the MVP flow
(docs/spec.md section 29, steps 4-9): profile -> detect/confirm target ->
leakage check -> ready for AutoML.

No LLM call happens here yet. Every branch is a plain policy function over
typed tool output (agent/policies.py), so the gating logic — pause for
target confirmation, block on critical leakage — is deterministic and
unit-testable before an LLM is put in the loop.

Usage:
    registry = DatasetRegistry()
    dataset_version_id = registry.register(df)
    state = AgentState(project_id="p1", dataset_version_id=dataset_version_id, objective="predict churn")

    result = run_data_audit(registry, state)
    if result.status == "paused_for_target_confirmation":
        ...  # surface result.pending_question to the user, then call again:
        result = run_data_audit(registry, state, confirmed_target="churned")

    if result.status == "blocked_on_leakage":
        ...  # surface result.pending_question / result.leakage.findings

    if result.status == "ready_for_automl":
        ...  # next: tools/ml/automl.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.dataset_registry import DatasetRegistry
from agent.policies import leakage_blocks_progress, target_confirmation_required
from agent.schemas import PendingQuestion
from agent.state import AgentState
from tools.data.leakage import detect_leakage
from tools.data.profile import detect_target, profile_columns
from tools.data.quality import check_data_quality
from tools.data.schemas import (
    DataQualityReport,
    DatasetProfile,
    LeakageDetectionResult,
    TargetDetectionResult,
)

RunStatus = Literal["paused_for_target_confirmation", "blocked_on_leakage", "ready_for_automl"]


@dataclass
class RunResult:
    status: RunStatus
    state: AgentState
    profile: DatasetProfile
    quality: DataQualityReport
    target_detection: TargetDetectionResult
    leakage: LeakageDetectionResult | None = None
    pending_question: PendingQuestion | None = None


def run_data_audit(
    registry: DatasetRegistry,
    state: AgentState,
    *,
    confirmed_target: str | None = None,
    post_outcome_columns: list[str] | None = None,
) -> RunResult:
    """Run the profile -> target -> leakage sequence for `state.dataset_version_id`.

    Re-entrant by design: call it once, and if it comes back
    `paused_for_target_confirmation`, ask the user, then call it again with
    `confirmed_target` set. Everything before the pause point is cheap and
    deterministic, so re-running it is not wasted work.
    """
    df = registry.get(state.dataset_version_id)

    profile = profile_columns(df)
    quality = check_data_quality(df, profile=profile)
    target_detection = detect_target(df, profile=profile, objective=state.objective)

    target_column = confirmed_target or state.target_column
    if target_column is None:
        if target_confirmation_required(target_detection):
            return RunResult(
                status="paused_for_target_confirmation",
                state=state,
                profile=profile,
                quality=quality,
                target_detection=target_detection,
                pending_question=PendingQuestion(
                    reason=target_detection.message or "target column is ambiguous",
                    question="Which column is the prediction target?",
                    context={"candidates": [c.model_dump() for c in target_detection.candidates[:5]]},
                ),
            )
        target_column = target_detection.recommended.column

    state = state.model_copy(
        update={
            "target_column": target_column,
            "task_type": _task_type_for(target_column, target_detection),
        }
    )

    leakage = detect_leakage(df, target=target_column, post_outcome_columns=post_outcome_columns)

    if leakage_blocks_progress(leakage):
        critical_findings = [f.model_dump() for f in leakage.findings if f.severity == "critical"]
        return RunResult(
            status="blocked_on_leakage",
            state=state,
            profile=profile,
            quality=quality,
            target_detection=target_detection,
            leakage=leakage,
            pending_question=PendingQuestion(
                reason="critical leakage findings must be resolved before modeling",
                question=(
                    "Review the leakage findings below — exclude the flagged columns, "
                    "or confirm they are safe to keep?"
                ),
                context={"findings": critical_findings},
            ),
        )

    return RunResult(
        status="ready_for_automl",
        state=state,
        profile=profile,
        quality=quality,
        target_detection=target_detection,
        leakage=leakage,
    )


def _task_type_for(target_column: str, detection: TargetDetectionResult) -> str | None:
    match = next((c for c in detection.candidates if c.column == target_column), None)
    return match.inferred_task_type if match else None
