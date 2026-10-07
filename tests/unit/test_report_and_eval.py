"""Unit tests for tools/report.py and evaluation/agent_eval.py."""

from __future__ import annotations

import sys
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Lightweight stubs for domain models that live outside the outputs directory.
# We patch sys.modules so that the report / agent_eval imports resolve
# without needing the full project on the Python path.
# ---------------------------------------------------------------------------


class _StubBudget(BaseModel):
    max_experiments: int = 30
    max_runtime_seconds: int = 3600
    max_compute_cost: float = 10.0
    minimum_improvement: float = 0.001


class _StubAgentState(BaseModel):
    project_id: str = "proj-1"
    dataset_version_id: str = "ds-v1"
    objective: str = "Maximise accuracy"
    task_type: str | None = None
    target_column: str | None = None
    domain_constraints: list[str] | None = None
    business_constraints: list[str] | None = None
    baseline_experiment_id: str | None = None
    best_experiment_id: str | None = None
    hypotheses: list[str] = Field(default_factory=list)
    completed_experiments: list[str] = Field(default_factory=list)
    rejected_experiments: list[str] = Field(default_factory=list)
    budget: _StubBudget | None = None
    stopping_reason: str | None = None


class _StubExperimentRecord(BaseModel):
    experiment_id: str = "exp-1"
    hypothesis: str = "baseline"
    objective: str = "Maximise accuracy"
    dataset_version_id: str = "ds-v1"
    split_strategy: str = "random"
    features: list[str] = Field(default_factory=list)
    excluded_features: list[str] = Field(default_factory=list)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    model: str = "xgboost"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    metric: str = "accuracy"
    metrics: dict[str, float] = Field(default_factory=dict)
    primary_score: float | None = None
    runtime_seconds: float = 1.0
    artifact_path: str | None = None
    conclusion: str | None = None
    next_action: str | None = None
    created_at: str = "2025-01-01T00:00:00Z"


class _ColumnProfile(BaseModel):
    name: str = "col"
    inferred_type: str = "numeric"
    missing_count: int = 0


class _DatasetProfile(BaseModel):
    n_rows: int = 100
    columns: list[_ColumnProfile] = Field(default_factory=list)


class _DataQualityIssue(BaseModel):
    severity: str = "high"
    type: str = "missing_values"
    columns: list[str] = Field(default_factory=list)
    message: str = "Too many missing values"


class _DataQualityReport(BaseModel):
    n_critical: int = 0
    n_high: int = 0
    n_medium: int = 0
    n_low: int = 0
    issues: list[_DataQualityIssue] = Field(default_factory=list)


class _LeakageFinding(BaseModel):
    severity: str = "critical"
    type: str = "target_leakage"
    columns: list[str] = Field(default_factory=list)
    evidence: str = "Perfect correlation"


class _LeakageDetectionResult(BaseModel):
    has_critical_issues: bool = False
    findings: list[_LeakageFinding] = Field(default_factory=list)


class _RobustnessCheck(BaseModel):
    name: str = "multi_seed"
    passed: bool = True
    evidence: str = "stddev < 0.01"


class _RobustnessReport(BaseModel):
    metric: str = "accuracy"
    reference_score: float = 0.90
    n_failed: int = 0
    checks: list[_RobustnessCheck] = Field(default_factory=list)
    checks_skipped: list[str] = Field(default_factory=list)


class _CriticReview(BaseModel):
    verdict: str = "unknown"
    confidence: float = 0.0
    issues: list[dict[str, Any]] = Field(default_factory=list)
    required_next_actions: list[str] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Patch sys.modules so that the source files can be imported.
# ---------------------------------------------------------------------------

_agent_state_mod = MagicMock()
_agent_state_mod.AgentState = _StubAgentState

_experiment_schemas_mod = MagicMock()
_experiment_schemas_mod.ExperimentRecord = _StubExperimentRecord

_data_schemas_mod = MagicMock()
_data_schemas_mod.DataQualityReport = _DataQualityReport
_data_schemas_mod.DatasetProfile = _DatasetProfile
_data_schemas_mod.LeakageDetectionResult = _LeakageDetectionResult

_validation_schemas_mod = MagicMock()
_validation_schemas_mod.RobustnessReport = _RobustnessReport

_validation_critic_mod = MagicMock()
_validation_critic_mod.CriticReview = _CriticReview

_leakage_mod = MagicMock()
_robustness_mod = MagicMock()

sys.modules.setdefault("agent", MagicMock())
sys.modules.setdefault("agent.state", _agent_state_mod)
sys.modules.setdefault("experiments", MagicMock())
sys.modules.setdefault("experiments.schemas", _experiment_schemas_mod)
sys.modules.setdefault("tools", MagicMock())
sys.modules.setdefault("tools.data", MagicMock())
sys.modules.setdefault("tools.data.schemas", _data_schemas_mod)
sys.modules.setdefault("tools.data.leakage", _leakage_mod)
sys.modules.setdefault("tools.validation", MagicMock())
sys.modules.setdefault("tools.validation.schemas", _validation_schemas_mod)
sys.modules.setdefault("tools.validation.critic", _validation_critic_mod)
sys.modules.setdefault("tools.validation.robustness", _robustness_mod)

# Now add the outputs directory to the path so we can import the modules.
sys.path.insert(0, "/mnt/user-data/outputs")

from report import (  # noqa: E402
    FinalReport,
    ReportSection,
    _NOT_AVAILABLE,
    _section_baseline,
    _section_best_model,
    _section_critic,
    _section_data_quality,
    _section_data_summary,
    _section_deployment_recommendation,
    _section_leakage,
    _section_limitations,
    _section_objective,
    _section_robustness,
    generate_report,
)
from agent_eval import (  # noqa: E402
    AgentQualityMetrics,
    _evaluate_conclusions,
    _evaluate_hypotheses,
    _evaluate_invalid_experiments,
    _evaluate_stopping,
    _evaluate_tool_selection,
    evaluate_agent_run,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_state(**overrides) -> _StubAgentState:
    defaults = dict(
        project_id="proj-1",
        dataset_version_id="ds-v1",
        objective="Maximise accuracy",
    )
    defaults.update(overrides)
    return _StubAgentState(**defaults)


def _make_experiment(**overrides) -> _StubExperimentRecord:
    defaults = dict(
        experiment_id="exp-1",
        hypothesis="baseline",
        model="xgboost",
        metric="accuracy",
        metrics={"accuracy": 0.85},
        primary_score=0.85,
        runtime_seconds=2.0,
        conclusion="supported",
    )
    defaults.update(overrides)
    return _StubExperimentRecord(**defaults)


# =========================================================================
# REPORT TESTS
# =========================================================================


class TestGenerateReportMinimalState:
    """generate_report() with most inputs None should still return all sections."""

    def test_all_sections_present(self):
        state = _make_state()
        report = generate_report(state)

        assert isinstance(report, FinalReport)
        titles = [s.title for s in report.sections]
        expected_titles = [
            "Objective",
            "Data Summary",
            "Data Quality Findings",
            "Leakage Findings",
            "Baseline Results",
            "Experiment Summary",
            "Best Model",
            "Robustness",
            "Critic Review",
            "Limitations",
            "Deployment Recommendation",
            "Reproducibility Information",
        ]
        assert titles == expected_titles

    def test_unavailable_sections_contain_marker(self):
        state = _make_state()
        report = generate_report(state)

        # These sections should say "Not available" when inputs are None.
        na_titles = {
            "Data Summary",
            "Data Quality Findings",
            "Leakage Findings",
            "Baseline Results",
            "Experiment Summary",
            "Best Model",
            "Robustness",
            "Critic Review",
        }
        for section in report.sections:
            if section.title in na_titles:
                assert _NOT_AVAILABLE in section.content, (
                    f"Section '{section.title}' should contain the not-available marker"
                )

    def test_project_id_and_objective_set(self):
        state = _make_state(project_id="test-proj", objective="Minimise loss")
        report = generate_report(state)

        assert report.project_id == "test-proj"
        assert report.objective == "Minimise loss"

    def test_generated_at_is_iso(self):
        state = _make_state()
        report = generate_report(state)
        # Should parse without error
        datetime.fromisoformat(report.generated_at)


class TestGenerateReportFullInputs:
    """generate_report() with all inputs provided should populate sections."""

    @pytest.fixture()
    def full_report(self):
        state = _make_state(
            target_column="target",
            task_type="classification",
            domain_constraints=["Must be interpretable"],
            business_constraints=["Latency < 50ms"],
            baseline_experiment_id="exp-base",
            hypotheses=["h1", "h2"],
            budget=_StubBudget(),
        )

        baseline = _make_experiment(experiment_id="exp-base", model="logistic_regression",
                                    primary_score=0.80, metrics={"accuracy": 0.80})
        best = _make_experiment(experiment_id="exp-best", primary_score=0.92,
                                metrics={"accuracy": 0.92, "f1": 0.88},
                                hyperparameters={"n_estimators": 200},
                                features=["f1", "f2"], excluded_features=["leak_col"],
                                artifact_path="/models/best.pkl")

        profile = _DatasetProfile(
            n_rows=1000,
            columns=[
                _ColumnProfile(name="f1", inferred_type="numeric", missing_count=10),
                _ColumnProfile(name="f2", inferred_type="categorical", missing_count=0),
                _ColumnProfile(name="target", inferred_type="binary", missing_count=0),
            ],
        )
        quality = _DataQualityReport(
            n_critical=0, n_high=1, n_medium=0, n_low=0,
            issues=[_DataQualityIssue(severity="high", type="missing_values",
                                      columns=["f1"], message="10 missing")],
        )
        leakage = _LeakageDetectionResult(
            has_critical_issues=True,
            findings=[_LeakageFinding(severity="critical", type="target_leakage",
                                      columns=["leak_col"], evidence="r=0.99")],
        )
        robustness = _RobustnessReport(
            metric="accuracy", reference_score=0.92, n_failed=0,
            checks=[_RobustnessCheck(name="multi_seed", passed=True,
                                      evidence="stddev=0.003")],
        )
        critic = _CriticReview(
            verdict="approve", confidence=0.95,
            summary="Model looks solid.",
            issues=[{"severity": "low", "type": "minor", "evidence": "small sample"}],
            required_next_actions=["Monitor for drift"],
        )

        return generate_report(
            state,
            profile=profile,
            quality_report=quality,
            leakage_result=leakage,
            experiments=[baseline, best],
            best_experiment=best,
            robustness_report=robustness,
            critic_review=critic,
        )

    def test_no_section_is_not_available(self, full_report):
        for section in full_report.sections:
            assert _NOT_AVAILABLE not in section.content, (
                f"Section '{section.title}' should be populated, not 'Not available'"
            )

    def test_objective_section_has_constraints(self, full_report):
        obj = next(s for s in full_report.sections if s.title == "Objective")
        assert "interpretable" in obj.content.lower()
        assert "Latency" in obj.content

    def test_data_summary_row_count(self, full_report):
        ds = next(s for s in full_report.sections if s.title == "Data Summary")
        assert "1,000" in ds.content

    def test_leakage_critical_flag(self, full_report):
        lk = next(s for s in full_report.sections if s.title == "Leakage Findings")
        assert "Critical leakage detected" in lk.content

    def test_best_model_hyperparams(self, full_report):
        bm = next(s for s in full_report.sections if s.title == "Best Model")
        assert "n_estimators" in bm.content
        assert "200" in bm.content

    def test_deployment_ready(self, full_report):
        dep = next(s for s in full_report.sections if s.title == "Deployment Recommendation")
        assert "Ready for deployment" in dep.content

    def test_critic_section_populated(self, full_report):
        cr = next(s for s in full_report.sections if s.title == "Critic Review")
        assert "approve" in cr.content
        assert "95%" in cr.content
        assert "Monitor for drift" in cr.content


class TestFinalReportToMarkdown:
    def test_produces_valid_markdown(self):
        report = FinalReport(
            project_id="proj-1",
            objective="Maximise accuracy",
            generated_at="2025-01-01T00:00:00+00:00",
            sections=[
                ReportSection(title="First", content="Content A"),
                ReportSection(title="Second", content="Content B"),
            ],
        )
        md = report.to_markdown()

        assert md.startswith("# Agentic ML Report")
        assert "## First" in md
        assert "## Second" in md
        assert "Content A" in md
        assert "Content B" in md
        assert "proj-1" in md

    def test_empty_sections_list(self):
        report = FinalReport(
            project_id="p", objective="o",
            generated_at="t", sections=[],
        )
        md = report.to_markdown()
        assert "# Agentic ML Report" in md
        # No section headers
        assert "##" not in md


# ---------------------------------------------------------------------------
# Section builder edge-case tests
# ---------------------------------------------------------------------------


class TestSectionBuilders:
    def test_data_summary_zero_rows(self):
        profile = _DatasetProfile(n_rows=0, columns=[])
        section = _section_data_summary(profile)
        assert "0" in section.content

    def test_data_quality_no_issues(self):
        report = _DataQualityReport(issues=[])
        section = _section_data_quality(report)
        assert "No data-quality issues" in section.content

    def test_leakage_no_findings(self):
        result = _LeakageDetectionResult(findings=[])
        section = _section_leakage(result)
        assert "No leakage detected" in section.content

    def test_baseline_missing_experiment(self):
        """baseline_experiment_id set but no matching record."""
        state = _make_state(baseline_experiment_id="missing-id")
        section = _section_baseline(state, [_make_experiment(experiment_id="other")])
        assert _NOT_AVAILABLE in section.content

    def test_best_model_none(self):
        section = _section_best_model(None)
        assert _NOT_AVAILABLE in section.content

    def test_robustness_with_failures(self):
        report = _RobustnessReport(
            metric="accuracy", reference_score=0.90, n_failed=1,
            checks=[
                _RobustnessCheck(name="multi_seed", passed=True, evidence="ok"),
                _RobustnessCheck(name="temporal", passed=False, evidence="drift"),
            ],
            checks_skipped=["perturbation"],
        )
        section = _section_robustness(report)
        assert "1 check(s) failed" in section.content
        assert "**NO**" in section.content
        assert "perturbation" in section.content

    def test_critic_with_dict_issues(self):
        review = _CriticReview(
            verdict="reject", confidence=0.4,
            issues=[{"severity": "high", "type": "overfitting", "evidence": "val gap"}],
        )
        section = _section_critic(review)
        assert "overfitting" in section.content
        assert "val gap" in section.content

    def test_limitations_single_model(self):
        state = _make_state()
        exps = [_make_experiment(model="xgboost")]
        section = _section_limitations(state, exps, None, None)
        assert "one model family" in section.content.lower()

    def test_limitations_critic_rejected(self):
        state = _make_state()
        critic = _CriticReview(verdict="reject")
        section = _section_limitations(state, None, None, critic)
        assert "reject" in section.content

    def test_limitations_no_issues(self):
        state = _make_state()
        section = _section_limitations(state, None, None, None)
        assert "No major limitations" in section.content

    def test_deployment_no_evidence(self):
        section = _section_deployment_recommendation(None, None)
        assert "Insufficient evidence" in section.content

    def test_deployment_critic_rejected(self):
        critic = _CriticReview(verdict="reject")
        section = _section_deployment_recommendation(critic, None)
        assert "Do not deploy" in section.content

    def test_deployment_conditional_robustness_pass(self):
        critic = _CriticReview(verdict="conditional_approval")
        robust = _RobustnessReport(n_failed=0)
        section = _section_deployment_recommendation(critic, robust)
        assert "Conditional deployment" in section.content

    def test_deployment_conditional_robustness_fail(self):
        critic = _CriticReview(verdict="conditional_approval")
        robust = _RobustnessReport(n_failed=2)
        section = _section_deployment_recommendation(critic, robust)
        assert "Do not deploy yet" in section.content

    def test_deployment_approved_but_robustness_fails(self):
        critic = _CriticReview(verdict="approve")
        robust = _RobustnessReport(n_failed=1)
        section = _section_deployment_recommendation(critic, robust)
        assert "Investigate before deploying" in section.content


# =========================================================================
# AGENT_EVAL TESTS
# =========================================================================


class TestEvaluateAgentRunEmpty:
    """evaluate_agent_run() with no experiments should yield all-None metrics."""

    def test_all_none(self):
        metrics = evaluate_agent_run([])
        assert metrics.correct_tool_selection_rate is None
        assert metrics.valid_hypothesis_rate is None
        assert metrics.invalid_experiment_rate is None
        assert metrics.correct_conclusion_rate is None
        assert metrics.stopping_quality is None
        assert metrics.leakage_detection_rate is None


class TestToolSelectionRate:
    def test_all_valid(self):
        calls = [{"name": "run_experiment"}, {"name": "detect_leakage"}]
        assert _evaluate_tool_selection(calls) == 1.0

    def test_all_invalid(self):
        calls = [{"name": "do_magic"}, {"name": "hack_prod"}]
        assert _evaluate_tool_selection(calls) == 0.0

    def test_mixed(self):
        calls = [
            {"name": "run_experiment"},
            {"name": "unknown_tool"},
            {"name": "profile_columns"},
            {"name": "bogus"},
        ]
        assert _evaluate_tool_selection(calls) == pytest.approx(0.5)

    def test_empty_returns_none(self):
        assert _evaluate_tool_selection([]) is None

    def test_missing_name_key(self):
        calls = [{"args": "x"}, {"name": "run_experiment"}]
        # Missing "name" key -> tc.get("name") is None, not in valid set
        assert _evaluate_tool_selection(calls) == pytest.approx(0.5)


class TestHypothesisValidation:
    def test_all_unique(self):
        exps = [
            _make_experiment(hypothesis="Try XGBoost"),
            _make_experiment(hypothesis="Try LightGBM"),
        ]
        assert _evaluate_hypotheses(exps) == 1.0

    def test_duplicates_count_as_invalid(self):
        exps = [
            _make_experiment(hypothesis="Try XGBoost"),
            _make_experiment(hypothesis="Try XGBoost"),  # duplicate
            _make_experiment(hypothesis="Try LightGBM"),
        ]
        # 2 valid out of 3
        assert _evaluate_hypotheses(exps) == pytest.approx(2 / 3)

    def test_empty_hypothesis_invalid(self):
        exps = [
            _make_experiment(hypothesis=""),
            _make_experiment(hypothesis="   "),
            _make_experiment(hypothesis="Valid idea"),
        ]
        assert _evaluate_hypotheses(exps) == pytest.approx(1 / 3)

    def test_empty_list(self):
        assert _evaluate_hypotheses([]) is None


class TestStoppingQuality:
    def test_early_stop(self):
        """Using < 30% of budget should be 'early'."""
        exps = [_make_experiment() for _ in range(2)]
        result = _evaluate_stopping(exps, budget_max_experiments=10)
        assert result == "early"

    def test_good_stop(self):
        """Using 30-99% of budget should generally be 'good'."""
        exps = [_make_experiment(primary_score=0.80 + i * 0.01) for i in range(5)]
        result = _evaluate_stopping(exps, budget_max_experiments=10)
        assert result == "good"

    def test_late_stop_stagnation(self):
        """Full budget with no improvement in last quartile should be 'late'."""
        # 10 experiments, all same score, budget = 10
        exps = [_make_experiment(primary_score=0.85) for _ in range(10)]
        result = _evaluate_stopping(exps, budget_max_experiments=10)
        assert result == "late"

    def test_full_budget_with_improvement_is_good(self):
        """Full budget but last quartile still improving should be 'good'."""
        exps = [_make_experiment(primary_score=0.80 + i * 0.01) for i in range(10)]
        result = _evaluate_stopping(exps, budget_max_experiments=10)
        assert result == "good"

    def test_empty_experiments(self):
        assert _evaluate_stopping([], budget_max_experiments=10) is None

    def test_zero_budget(self):
        """Zero budget => utilisation = 1.0, avoids division by zero."""
        exps = [_make_experiment(primary_score=0.85)]
        # utilisation = 1/0 handled as 1.0 in code; n=1 so tail check may not trigger "late"
        result = _evaluate_stopping(exps, budget_max_experiments=0)
        assert result in ("good", "late")


class TestConclusionCorrectness:
    def test_correct_conclusions(self):
        exps = [
            _make_experiment(experiment_id="e1", primary_score=0.80, conclusion="supported"),
            _make_experiment(experiment_id="e2", primary_score=0.85, conclusion="supported"),
            _make_experiment(experiment_id="e3", primary_score=0.82, conclusion="rejected"),
        ]
        rate = _evaluate_conclusions(exps)
        # e1: best_so_far is None, so improved=False. conclusion="supported" -> wrong.
        #     but best_so_far becomes 0.80
        # e2: 0.85 > 0.80, improved=True, conclusion="supported" -> correct
        # e3: 0.82 < 0.85, improved=False, conclusion="rejected" -> correct
        # scorable = 3, correct = 2
        assert rate == pytest.approx(2 / 3)

    def test_single_experiment_returns_none(self):
        exps = [_make_experiment()]
        assert _evaluate_conclusions(exps) is None

    def test_no_scores_returns_none(self):
        exps = [
            _make_experiment(primary_score=None, conclusion="supported"),
            _make_experiment(primary_score=None, conclusion="rejected"),
        ]
        assert _evaluate_conclusions(exps) is None

    def test_inconclusive_when_no_improvement(self):
        exps = [
            _make_experiment(experiment_id="e1", primary_score=0.90, conclusion="supported"),
            _make_experiment(experiment_id="e2", primary_score=0.88, conclusion="inconclusive"),
        ]
        rate = _evaluate_conclusions(exps)
        # e1: not improved (first), conclusion "supported" -> wrong
        # e2: 0.88 < 0.90, not improved, "inconclusive" -> correct
        # correct=1, scorable=2
        assert rate == pytest.approx(0.5)


class TestInvalidExperiments:
    def test_all_valid(self):
        exps = [_make_experiment(primary_score=0.9), _make_experiment(primary_score=0.8)]
        assert _evaluate_invalid_experiments(exps) == 0.0

    def test_some_invalid(self):
        exps = [
            _make_experiment(primary_score=0.9),
            _make_experiment(primary_score=None),
        ]
        assert _evaluate_invalid_experiments(exps) == pytest.approx(0.5)

    def test_empty(self):
        assert _evaluate_invalid_experiments([]) is None


class TestAgentQualityMetricsModel:
    def test_to_dict(self):
        m = AgentQualityMetrics(correct_tool_selection_rate=0.8)
        d = m.to_dict()
        assert d["correct_tool_selection_rate"] == 0.8
        assert d["valid_hypothesis_rate"] is None

    def test_to_markdown(self):
        m = AgentQualityMetrics(
            correct_tool_selection_rate=1.0,
            stopping_quality="good",
        )
        md = m.to_markdown()
        assert "Agent Quality Metrics" in md
        assert "100.00%" in md
        assert "good" in md.lower()


class TestEvaluateAgentRunIntegration:
    """Integration: evaluate_agent_run with realistic inputs."""

    def test_with_experiments_and_tools(self):
        exps = [
            _make_experiment(experiment_id="e1", hypothesis="baseline",
                             primary_score=0.80, conclusion="supported"),
            _make_experiment(experiment_id="e2", hypothesis="try lightgbm",
                             primary_score=0.85, conclusion="supported"),
            _make_experiment(experiment_id="e3", hypothesis="add features",
                             primary_score=0.83, conclusion="rejected"),
        ]
        tools = [
            {"name": "run_experiment"},
            {"name": "run_experiment"},
            {"name": "detect_leakage"},
            {"name": "unknown"},
        ]

        metrics = evaluate_agent_run(
            exps,
            tool_calls=tools,
            budget_max_experiments=10,
        )

        assert metrics.correct_tool_selection_rate == pytest.approx(0.75)
        assert metrics.valid_hypothesis_rate == 1.0
        assert metrics.invalid_experiment_rate == 0.0
        assert metrics.correct_conclusion_rate is not None
        assert metrics.stopping_quality == "good"  # 3/10 = 0.30 -> "good" (30% boundary)
        assert metrics.leakage_detection_rate is None

    def test_boundary_early_vs_good(self):
        """Exactly 30% utilisation should NOT be early (< 0.30 required)."""
        exps = [_make_experiment(primary_score=0.80 + i * 0.01) for i in range(3)]
        metrics = evaluate_agent_run(exps, budget_max_experiments=10)
        assert metrics.stopping_quality == "good"
