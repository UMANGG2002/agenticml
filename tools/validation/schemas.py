"""Typed output for the validation layer (robustness, fairness, calibration, drift)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high", "critical"]


class SeedRun(BaseModel):
    seed: int
    score: float


class RobustnessCheck(BaseModel):
    name: str
    metric: str
    passed: bool
    severity: Severity | None = None
    evidence: str
    values: dict[str, float] = Field(default_factory=dict)


class RobustnessReport(BaseModel):
    metric: str
    reference_score: float
    checks: list[RobustnessCheck]
    checks_skipped: list[str] = Field(default_factory=list)
    n_failed: int

    @property
    def failed(self) -> list[RobustnessCheck]:
        return [c for c in self.checks if not c.passed]
