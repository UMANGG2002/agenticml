"""
Agent quality metrics for evaluating AME agent decision-making.

Measures how well the agent selects tools, forms hypotheses, avoids
invalid experiments, draws correct conclusions, and knows when to stop.

Spec reference: section 32 — Agent Quality Evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from experiments.schemas import ExperimentRecord
from tools.data.leakage import detect_leakage
from tools.validation.robustness import run_robustness_checks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known valid tool names — used to judge correct tool selection.
# ---------------------------------------------------------------------------

_VALID_TOOL_NAMES: set[str] = {
    "create_split",
    "detect_leakage",
    "profile_columns",
    "check_data_quality",
    "run_automl",
    "run_robustness_checks",
    "run_experiment",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AgentQualityMetrics(BaseModel):
    """Quality metrics for a single agent run.

    Each rate is a float in ``[0, 1]`` or ``None`` when the metric could not
    be computed (e.g. no tool calls were recorded).

    Attributes
    ----------
    correct_tool_selection_rate:
        Fraction of tool calls that targeted a tool from the known valid set.
    valid_hypothesis_rate:
        Fraction of experiments whose stated hypothesis was non-trivial and
        testable (heuristic check: non-empty, not duplicated).
    invalid_experiment_rate:
        Fraction of experiments that raised an error or produced no score.
    correct_conclusion_rate:
        Fraction of experiments whose recorded conclusion is consistent with
        the observed score change.
    stopping_quality:
        One of ``"good"``, ``"early"``, ``"late"`` — how well the agent used
        its experiment budget.
    leakage_detection_rate:
        If ground-truth leakage labels are available, the fraction that the
        agent's leakage detector correctly identified.
    """

    correct_tool_selection_rate: float | None = None
    valid_hypothesis_rate: float | None = None
    invalid_experiment_rate: float | None = None
    correct_conclusion_rate: float | None = None
    stopping_quality: str | None = None  # "good", "early", "late"
    leakage_detection_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for logging or storage."""
        return self.model_dump(exclude_none=False)

    def to_markdown(self) -> str:
        """Render the metrics as a compact Markdown summary."""
        lines = ["### Agent Quality Metrics\n"]
        for field_name, value in self.model_dump().items():
            label = field_name.replace("_", " ").title()
            if value is None:
                formatted = "—"
            elif isinstance(value, float):
                formatted = f"{value:.2%}"
            else:
                formatted = str(value)
            lines.append(f"- **{label}**: {formatted}")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_tool_selection(tool_calls: list[dict]) -> float | None:
    """Return the fraction of tool calls whose name is in the valid set.

    Parameters
    ----------
    tool_calls:
        Each dict is expected to have at least a ``"name"`` key.

    Returns
    -------
    float or None
        ``None`` when *tool_calls* is empty.
    """
    if not tool_calls:
        return None
    correct = sum(1 for tc in tool_calls if tc.get("name") in _VALID_TOOL_NAMES)
    return correct / len(tool_calls)


def _evaluate_hypotheses(experiments: list[ExperimentRecord]) -> float | None:
    """Rate the fraction of experiments with a valid (non-trivial) hypothesis.

    A hypothesis is considered *invalid* when it is:
    - empty or whitespace-only, or
    - an exact duplicate of a previous hypothesis in the same run.
    """
    if not experiments:
        return None

    seen: set[str] = set()
    valid_count = 0

    for exp in experiments:
        hyp = getattr(exp, "hypothesis", None) or ""
        hyp = hyp.strip()
        if hyp and hyp not in seen:
            valid_count += 1
            seen.add(hyp)

    return valid_count / len(experiments)


def _evaluate_invalid_experiments(experiments: list[ExperimentRecord]) -> float | None:
    """Return the fraction of experiments that failed or produced no score."""
    if not experiments:
        return None

    invalid = 0
    for exp in experiments:
        score = getattr(exp, "primary_score", None)
        if score is None:
            invalid += 1

    return invalid / len(experiments)


def _evaluate_conclusions(experiments: list[ExperimentRecord]) -> float | None:
    """Rate the fraction of experiments whose conclusion aligns with the score.

    Heuristic: if the experiment improved the score (relative to the previous
    best) the conclusion should contain a positive-sentiment keyword; if it
    did not, the conclusion should *not* claim improvement.
    """
    if len(experiments) < 2:
        return None

    # ExperimentRecord.conclusion is Literal["supported", "rejected", "inconclusive"]
    # "supported" means the hypothesis was confirmed (score improved vs parent)
    # "rejected" means it did not improve
    correct = 0
    best_so_far: float | None = None

    for exp in experiments:
        score = getattr(exp, "primary_score", None)
        conclusion = getattr(exp, "conclusion", None) or ""

        if score is None or not conclusion:
            continue

        improved = False
        if best_so_far is not None:
            improved = score > best_so_far

        if best_so_far is None or score > best_so_far:
            best_so_far = score

        if improved and conclusion == "supported":
            correct += 1
        elif not improved and conclusion in ("rejected", "inconclusive"):
            correct += 1

    scorable = sum(
        1
        for e in experiments
        if getattr(e, "primary_score", None) is not None
        and getattr(e, "conclusion", None)
    )
    if scorable == 0:
        return None
    return correct / scorable


def _evaluate_stopping(
    experiments: list[ExperimentRecord],
    budget_max_experiments: int,
) -> str | None:
    """Judge whether the agent stopped at a reasonable point.

    Rules of thumb:
    - ``"early"`` — used less than 30 % of the budget.
    - ``"late"``  — used the full budget *and* the last quartile of
      experiments showed no improvement.
    - ``"good"``  — everything else.
    """
    if not experiments:
        return None

    n = len(experiments)
    utilisation = n / budget_max_experiments if budget_max_experiments > 0 else 1.0

    if utilisation < 0.30:
        return "early"

    # Check for stagnation in the final quartile.
    if utilisation >= 1.0:
        quarter = max(1, n // 4)
        tail_scores = [
            getattr(e, "primary_score", None) for e in experiments[-quarter:]
        ]
        tail_scores = [s for s in tail_scores if s is not None]
        if len(tail_scores) >= 2:
            improvement = max(tail_scores) - min(tail_scores)
            if abs(improvement) < 1e-6:
                return "late"

    return "good"


def _evaluate_leakage_detection(
    leakage_result: Any | None,
) -> float | None:
    """Return the fraction of leakage issues the agent detected.

    If *leakage_result* is ``None`` or has no ``issues`` attribute the
    metric cannot be computed and ``None`` is returned.
    """
    if leakage_result is None:
        return None

    issues = getattr(leakage_result, "issues", None)
    if not issues:
        return None

    detected = sum(1 for issue in issues if getattr(issue, "detected", False))
    return detected / len(issues)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_agent_run(
    experiments: list[ExperimentRecord],
    *,
    leakage_result: Any | None = None,
    robustness_report: Any | None = None,
    tool_calls: list[dict] | None = None,
    budget_max_experiments: int = 30,
) -> AgentQualityMetrics:
    """Evaluate the quality of an agent's decisions across a single run.

    Parameters
    ----------
    experiments:
        Ordered list of experiments the agent executed.
    leakage_result:
        Output of ``detect_leakage``, if available.
    robustness_report:
        Output of ``run_robustness_checks``, if available.
    tool_calls:
        Raw tool-call records (each a dict with at least ``"name"``).
    budget_max_experiments:
        The experiment budget the agent was given; used to judge stopping.

    Returns
    -------
    AgentQualityMetrics
        Populated metrics.  Fields that cannot be computed are ``None``.
    """
    metrics = AgentQualityMetrics(
        correct_tool_selection_rate=_evaluate_tool_selection(tool_calls or []),
        valid_hypothesis_rate=_evaluate_hypotheses(experiments),
        invalid_experiment_rate=_evaluate_invalid_experiments(experiments),
        correct_conclusion_rate=_evaluate_conclusions(experiments),
        stopping_quality=_evaluate_stopping(experiments, budget_max_experiments),
        leakage_detection_rate=_evaluate_leakage_detection(leakage_result),
    )

    logger.info("Agent quality evaluation complete: %s", metrics.to_dict())
    return metrics
