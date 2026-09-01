"""Explicit, typed agent run state (docs/spec.md section 7).

This is the source of truth for a run — not the LLM conversation history.
Currently held in memory by the caller; will move behind PostgreSQL once
db/models.py exists (docs/spec.md section 24).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Budget(BaseModel):
    max_experiments: int = 30
    max_runtime_seconds: int = 1800
    max_compute_cost: float = 5.0
    minimum_improvement: float = 0.002


class AgentState(BaseModel):
    project_id: str
    dataset_version_id: str
    objective: str
    task_type: str | None = None
    target_column: str | None = None

    domain_constraints: list[str] = Field(default_factory=list)
    business_constraints: list[str] = Field(default_factory=list)

    baseline_experiment_id: str | None = None
    best_experiment_id: str | None = None

    hypotheses: list[str] = Field(default_factory=list)
    completed_experiments: list[str] = Field(default_factory=list)
    rejected_experiments: list[str] = Field(default_factory=list)

    budget: Budget = Field(default_factory=Budget)
    stopping_reason: str | None = None
