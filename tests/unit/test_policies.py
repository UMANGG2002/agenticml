from agent.policies import leakage_blocks_progress, quality_blocks_progress, target_confirmation_required
from tools.data.schemas import (
    DataQualityReport,
    LeakageDetectionResult,
    TargetCandidate,
    TargetDetectionResult,
)


def _target_result(requires_confirmation: bool) -> TargetDetectionResult:
    candidate = TargetCandidate(column="y", confidence=0.9, reasons=[], inferred_task_type="binary_classification")
    return TargetDetectionResult(
        candidates=[candidate],
        recommended=candidate,
        requires_user_confirmation=requires_confirmation,
    )


def test_target_confirmation_required_reflects_flag():
    assert target_confirmation_required(_target_result(True)) is True
    assert target_confirmation_required(_target_result(False)) is False


def test_leakage_blocks_progress_reflects_flag():
    clean = LeakageDetectionResult(findings=[], checks_performed=[], checks_skipped=[], has_critical_issues=False)
    dirty = LeakageDetectionResult(findings=[], checks_performed=[], checks_skipped=[], has_critical_issues=True)
    assert leakage_blocks_progress(clean) is False
    assert leakage_blocks_progress(dirty) is True


def test_quality_blocks_progress_uses_critical_count():
    clean = DataQualityReport(issues=[], n_critical=0, n_high=0, n_medium=0, n_low=0)
    dirty = DataQualityReport(issues=[], n_critical=2, n_high=0, n_medium=0, n_low=0)
    assert quality_blocks_progress(clean) is False
    assert quality_blocks_progress(dirty) is True
    assert quality_blocks_progress(dirty, max_allowed_critical=2) is False
