"""Final report generator for the Agentic ML Engineer (spec §26 / §35).

Consumes the typed outputs from a completed ML engineering run and produces
a structured, markdown-formatted report covering every phase of the pipeline:
data profiling, quality checks, leakage detection, experimentation,
robustness validation, critic review, and deployment recommendation.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agent.state import AgentState
from experiments.schemas import ExperimentRecord
from tools.data.schemas import (
    DataQualityReport,
    DatasetProfile,
    LeakageDetectionResult,
)
from tools.validation.schemas import RobustnessReport

# CriticReview is defined downstream in tools.validation.critic.
# Import defensively so the module still loads if that file has not been
# created yet (the report gracefully handles critic_review=None).
try:
    from tools.validation.critic import CriticReview
except ImportError:  # pragma: no cover

    class CriticReview(BaseModel):  # type: ignore[no-redef]
        """Minimal stand-in until the real CriticReview is implemented."""

        verdict: str = "unknown"
        confidence: float = 0.0
        issues: list[dict[str, Any]] = Field(default_factory=list)
        required_next_actions: list[str] = Field(default_factory=list)
        summary: str = ""


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class ReportSection(BaseModel):
    """A single titled section of the final report."""

    title: str
    content: str  # markdown-formatted text


class FinalReport(BaseModel):
    """Complete final report for an Agentic ML engineering run."""

    project_id: str
    objective: str
    generated_at: str
    sections: list[ReportSection]

    def to_markdown(self) -> str:
        """Render the full report as a single markdown document."""
        lines: list[str] = [
            f"# Agentic ML Report — {self.project_id}",
            "",
            f"*Generated at {self.generated_at}*",
            "",
        ]
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section builders — each returns a ReportSection or None
# ---------------------------------------------------------------------------

_NOT_AVAILABLE = "*Not available.*"


def _section_objective(state: AgentState) -> ReportSection:
    lines = [state.objective]
    if state.target_column:
        lines.append(f"\n**Target column:** `{state.target_column}`")
    if state.task_type:
        lines.append(f"**Task type:** {state.task_type}")
    if state.domain_constraints:
        lines.append("\n**Domain constraints:**")
        for dc in state.domain_constraints:
            lines.append(f"- {dc}")
    if state.business_constraints:
        lines.append("\n**Business constraints:**")
        for bc in state.business_constraints:
            lines.append(f"- {bc}")
    return ReportSection(title="Objective", content="\n".join(lines))


def _section_data_summary(profile: DatasetProfile | None) -> ReportSection:
    if profile is None:
        return ReportSection(title="Data Summary", content=_NOT_AVAILABLE)

    n_columns = len(profile.columns)
    type_counts: Counter[str] = Counter()
    for col in profile.columns:
        type_counts[col.inferred_type] += 1

    type_breakdown = ", ".join(
        f"{count} {typ}" for typ, count in type_counts.most_common()
    )

    total_missing = sum(c.missing_count for c in profile.columns)
    total_cells = profile.n_rows * n_columns
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

    lines = [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Rows | {profile.n_rows:,} |",
        f"| Columns | {n_columns:,} |",
        f"| Column types | {type_breakdown} |",
        f"| Overall missingness | {missing_pct:.1f}% |",
    ]
    return ReportSection(title="Data Summary", content="\n".join(lines))


def _section_data_quality(report: DataQualityReport | None) -> ReportSection:
    if report is None:
        return ReportSection(title="Data Quality Findings", content=_NOT_AVAILABLE)

    if not report.issues:
        return ReportSection(
            title="Data Quality Findings",
            content="No data-quality issues detected.",
        )

    lines = [
        f"**{report.n_critical} critical, {report.n_high} high, "
        f"{report.n_medium} medium, {report.n_low} low** issue(s) found.",
        "",
        "| Severity | Type | Columns | Detail |",
        "|----------|------|---------|--------|",
    ]
    for issue in report.issues:
        cols = ", ".join(f"`{c}`" for c in issue.columns) if issue.columns else "—"
        lines.append(f"| {issue.severity} | {issue.type} | {cols} | {issue.message} |")

    return ReportSection(title="Data Quality Findings", content="\n".join(lines))


def _section_leakage(result: LeakageDetectionResult | None) -> ReportSection:
    if result is None:
        return ReportSection(title="Leakage Findings", content=_NOT_AVAILABLE)

    if not result.findings:
        return ReportSection(
            title="Leakage Findings",
            content="No leakage detected.",
        )

    status = "**Critical leakage detected.**" if result.has_critical_issues else ""
    lines = [status, ""] if status else []
    lines.extend([
        "| Severity | Type | Columns | Evidence |",
        "|----------|------|---------|----------|",
    ])
    for f in result.findings:
        cols = ", ".join(f"`{c}`" for c in f.columns) if f.columns else "—"
        lines.append(f"| {f.severity} | {f.type} | {cols} | {f.evidence} |")

    return ReportSection(title="Leakage Findings", content="\n".join(lines))


def _section_baseline(
    state: AgentState,
    experiments: list[ExperimentRecord] | None,
) -> ReportSection:
    if not experiments or not state.baseline_experiment_id:
        return ReportSection(title="Baseline Results", content=_NOT_AVAILABLE)

    baseline = next(
        (e for e in experiments if e.experiment_id == state.baseline_experiment_id),
        None,
    )
    if baseline is None:
        return ReportSection(title="Baseline Results", content=_NOT_AVAILABLE)

    lines = [
        f"**Model:** {baseline.model}",
        f"**Primary metric ({baseline.metric}):** {baseline.primary_score:.4f}",
    ]
    if len(baseline.metrics) > 1:
        lines.append("\nAll metrics:")
        for name, value in sorted(baseline.metrics.items()):
            lines.append(f"- {name}: {value:.4f}")

    return ReportSection(title="Baseline Results", content="\n".join(lines))


def _section_experiment_summary(
    state: AgentState,
    experiments: list[ExperimentRecord] | None,
) -> ReportSection:
    if not experiments:
        return ReportSection(title="Experiment Summary", content=_NOT_AVAILABLE)

    conclusions: Counter[str | None] = Counter(e.conclusion for e in experiments)
    supported = conclusions.get("supported", 0)
    rejected = conclusions.get("rejected", 0)
    inconclusive = conclusions.get("inconclusive", 0)
    pending = conclusions.get(None, 0)

    total_runtime = sum(e.runtime_seconds for e in experiments)
    unique_models = {e.model for e in experiments}

    lines = [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total experiments | {len(experiments)} |",
        f"| Hypotheses tested | {len(state.hypotheses)} |",
        f"| Supported | {supported} |",
        f"| Rejected | {rejected} |",
        f"| Inconclusive | {inconclusive} |",
    ]
    if pending:
        lines.append(f"| Pending conclusion | {pending} |")
    lines.extend([
        f"| Distinct models tried | {len(unique_models)} |",
        f"| Total runtime | {total_runtime:.1f}s |",
    ])

    return ReportSection(title="Experiment Summary", content="\n".join(lines))


def _section_best_model(best: ExperimentRecord | None) -> ReportSection:
    if best is None:
        return ReportSection(title="Best Model", content=_NOT_AVAILABLE)

    lines = [
        f"**Model:** {best.model}",
        f"**Experiment:** `{best.experiment_id}`",
        f"**Hypothesis:** {best.hypothesis}",
        f"**Primary metric ({best.metric}):** {best.primary_score:.4f}",
    ]
    if best.metrics:
        lines.append("\n**All metrics:**")
        for name, value in sorted(best.metrics.items()):
            lines.append(f"- {name}: {value:.4f}")
    if best.hyperparameters:
        lines.append("\n**Hyperparameters:**")
        for k, v in sorted(best.hyperparameters.items()):
            lines.append(f"- `{k}`: {v}")
    if best.features:
        lines.append(f"\n**Features used:** {len(best.features)}")
    if best.excluded_features:
        excluded = ", ".join(f"`{f}`" for f in best.excluded_features)
        lines.append(f"**Excluded features:** {excluded}")

    return ReportSection(title="Best Model", content="\n".join(lines))


def _section_robustness(report: RobustnessReport | None) -> ReportSection:
    if report is None:
        return ReportSection(title="Robustness", content=_NOT_AVAILABLE)

    status = "All checks passed." if report.n_failed == 0 else (
        f"**{report.n_failed} check(s) failed.**"
    )
    lines = [
        f"**Reference score ({report.metric}):** {report.reference_score:.4f}",
        f"**Status:** {status}",
        "",
        "| Check | Passed | Evidence |",
        "|-------|--------|----------|",
    ]
    for check in report.checks:
        icon = "yes" if check.passed else "**NO**"
        lines.append(f"| {check.name} | {icon} | {check.evidence} |")

    if report.checks_skipped:
        lines.append(
            f"\n*Skipped checks:* {', '.join(report.checks_skipped)}"
        )

    return ReportSection(title="Robustness", content="\n".join(lines))


def _section_critic(review: CriticReview | None) -> ReportSection:
    if review is None:
        return ReportSection(title="Critic Review", content=_NOT_AVAILABLE)

    lines = [
        f"**Verdict:** {review.verdict}",
        f"**Confidence:** {review.confidence:.0%}",
    ]
    if review.summary:
        lines.append(f"\n{review.summary}")

    if review.issues:
        lines.append("\n**Issues:**")
        for issue in review.issues:
            if isinstance(issue, dict):
                severity = issue.get("severity", "unknown")
                issue_type = issue.get("type", "unspecified")
                evidence = issue.get("evidence", "")
                lines.append(f"- [{severity}] {issue_type}: {evidence}")
            else:
                lines.append(f"- {issue}")

    if review.required_next_actions:
        lines.append("\n**Required next actions:**")
        for action in review.required_next_actions:
            lines.append(f"1. {action}")

    return ReportSection(title="Critic Review", content="\n".join(lines))


def _section_limitations(
    state: AgentState,
    experiments: list[ExperimentRecord] | None,
    robustness_report: RobustnessReport | None,
    critic_review: CriticReview | None,
) -> ReportSection:
    limitations: list[str] = []

    if experiments:
        models_tried = {e.model for e in experiments}
        if len(models_tried) == 1:
            limitations.append(
                "Only one model family was evaluated; other algorithms may perform better."
            )
        rejected = [e for e in experiments if e.conclusion == "rejected"]
        if len(rejected) > len(experiments) * 0.6:
            limitations.append(
                "A majority of hypotheses were rejected, suggesting the feature space "
                "or modelling approach may need rethinking."
            )

    if robustness_report and robustness_report.n_failed > 0:
        failed_names = [c.name for c in robustness_report.checks if not c.passed]
        limitations.append(
            f"Robustness checks failed: {', '.join(failed_names)}. "
            "Model stability should be investigated before deployment."
        )

    if critic_review and critic_review.verdict in ("reject", "conditional_approval"):
        limitations.append(
            f"Critic verdict is '{critic_review.verdict}' — "
            "outstanding issues must be resolved."
        )

    if state.stopping_reason:
        limitations.append(f"Run stopped because: {state.stopping_reason}")

    if not limitations:
        limitations.append("No major limitations identified during this run.")

    content = "\n".join(f"- {lim}" for lim in limitations)
    return ReportSection(title="Limitations", content=content)


def _section_deployment_recommendation(
    critic_review: CriticReview | None,
    robustness_report: RobustnessReport | None,
) -> ReportSection:
    if critic_review is None and robustness_report is None:
        return ReportSection(
            title="Deployment Recommendation",
            content=(
                "Insufficient evidence to make a deployment recommendation. "
                "Run the critic review and robustness checks first."
            ),
        )

    robustness_ok = (
        robustness_report is not None and robustness_report.n_failed == 0
    )
    critic_approved = (
        critic_review is not None
        and critic_review.verdict in ("approve", "approved", "approval")
    )
    critic_conditional = (
        critic_review is not None
        and critic_review.verdict in (
            "conditional_approval", "conditional_approve", "conditional",
        )
    )
    critic_rejected = (
        critic_review is not None
        and critic_review.verdict in ("reject", "rejected")
    )

    if critic_rejected:
        recommendation = (
            "**Do not deploy.** The critic has rejected the current model. "
            "Address the required next actions before reconsidering."
        )
    elif critic_conditional:
        if robustness_ok:
            recommendation = (
                "**Conditional deployment.** The model passes robustness checks "
                "but the critic has flagged issues that should be addressed. "
                "Deploy only if the flagged conditions are acceptable for the "
                "target environment."
            )
        else:
            recommendation = (
                "**Do not deploy yet.** The critic gave conditional approval but "
                "robustness checks have failed. Resolve both before deploying."
            )
    elif critic_approved and robustness_ok:
        recommendation = (
            "**Ready for deployment.** The model has been approved by the critic "
            "and passes all robustness checks."
        )
    elif critic_approved and not robustness_ok:
        recommendation = (
            "**Investigate before deploying.** The critic approved the model but "
            "robustness checks have failed. Review the failed checks."
        )
    elif robustness_ok:
        recommendation = (
            "**Robustness checks pass.** However, no critic review is available. "
            "Run the critic before deploying to production."
        )
    else:
        recommendation = (
            "**Do not deploy yet.** Robustness checks have failed and/or the "
            "critic review is missing."
        )

    lines = [recommendation]

    if critic_review and critic_review.required_next_actions:
        lines.append("\n**Outstanding actions:**")
        for action in critic_review.required_next_actions:
            lines.append(f"1. {action}")

    return ReportSection(title="Deployment Recommendation", content="\n".join(lines))


def _section_reproducibility(
    state: AgentState,
    experiments: list[ExperimentRecord] | None,
    best: ExperimentRecord | None,
) -> ReportSection:
    lines: list[str] = []

    lines.append(f"**Project ID:** `{state.project_id}`")
    lines.append(f"**Dataset version:** `{state.dataset_version_id}`")

    if best:
        lines.append(f"**Best experiment ID:** `{best.experiment_id}`")
        lines.append(f"**Seed:** {best.seed}")
        if best.artifact_path:
            lines.append(f"**Model artifact:** `{best.artifact_path}`")
        if best.dataset_version_id:
            lines.append(f"**Dataset version (experiment):** `{best.dataset_version_id}`")
        lines.append(f"**Created at:** {best.created_at}")

    if experiments:
        ids = [e.experiment_id for e in experiments]
        lines.append(f"\n**All experiment IDs ({len(ids)}):**")
        for eid in ids:
            lines.append(f"- `{eid}`")

    if state.budget:
        lines.append("\n**Budget:**")
        lines.append(f"- Max experiments: {state.budget.max_experiments}")
        lines.append(f"- Max runtime: {state.budget.max_runtime_seconds}s")
        lines.append(f"- Max compute cost: ${state.budget.max_compute_cost:.2f}")
        lines.append(f"- Minimum improvement: {state.budget.minimum_improvement}")

    return ReportSection(title="Reproducibility Information", content="\n".join(lines))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_report(
    state: AgentState,
    *,
    profile: DatasetProfile | None = None,
    quality_report: DataQualityReport | None = None,
    leakage_result: LeakageDetectionResult | None = None,
    experiments: list[ExperimentRecord] | None = None,
    best_experiment: ExperimentRecord | None = None,
    robustness_report: RobustnessReport | None = None,
    critic_review: CriticReview | None = None,
) -> FinalReport:
    """Generate the final report for a completed Agentic ML run.

    Every parameter after *state* is optional; sections whose data is
    unavailable are included with a "Not available" note so the reader
    knows what was skipped rather than silently omitting information.

    Args:
        state: The agent's accumulated run state.
        profile: Column-level dataset profile.
        quality_report: Data-quality audit results.
        leakage_result: Target-leakage detection results.
        experiments: All experiment records from the run.
        best_experiment: The single best experiment selected by the agent.
        robustness_report: Multi-seed / temporal / perturbation robustness.
        critic_review: Independent critic verdict and issues.

    Returns:
        A ``FinalReport`` whose ``to_markdown()`` renders the full document.
    """
    sections: list[ReportSection] = [
        _section_objective(state),
        _section_data_summary(profile),
        _section_data_quality(quality_report),
        _section_leakage(leakage_result),
        _section_baseline(state, experiments),
        _section_experiment_summary(state, experiments),
        _section_best_model(best_experiment),
        _section_robustness(robustness_report),
        _section_critic(critic_review),
        _section_limitations(state, experiments, robustness_report, critic_review),
        _section_deployment_recommendation(critic_review, robustness_report),
        _section_reproducibility(state, experiments, best_experiment),
    ]

    return FinalReport(
        project_id=state.project_id,
        objective=state.objective,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sections=sections,
    )
