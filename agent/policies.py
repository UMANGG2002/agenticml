"""Pure gating policies over typed tool output.

Kept separate from agent/orchestrator.py so the *rule* ("what counts as
blocking") and the *sequencing* (which tool runs next) can change and be
tested independently. No I/O, no state mutation — every function here is a
plain predicate.
"""

from __future__ import annotations

from tools.data.schemas import DataQualityReport, LeakageDetectionResult, TargetDetectionResult


def target_confirmation_required(result: TargetDetectionResult) -> bool:
    return result.requires_user_confirmation


def leakage_blocks_progress(result: LeakageDetectionResult) -> bool:
    return result.has_critical_issues


def quality_blocks_progress(report: DataQualityReport, max_allowed_critical: int = 0) -> bool:
    return report.n_critical > max_allowed_critical
