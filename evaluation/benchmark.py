"""
Benchmark runner comparing AME (Agentic ML Engineer) vs plain AutoML.

Runs the same datasets through both a conventional AutoML baseline and the full
AME pipeline (audit → automl → error analysis → robustness → critic), then
collects side-by-side metrics so improvements can be measured objectively.

Spec reference: section 32 — Evaluation & Benchmarking.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agent.dataset_registry import DatasetRegistry
from agent.state import AgentState
from experiments.registry import ExperimentRegistry
from experiments.runner import run_experiment
from experiments.schemas import ExperimentRecord
from tools.data.leakage import detect_leakage
from tools.data.profile import profile_columns
from tools.data.quality import check_data_quality
from tools.data.split import create_split
from tools.ml.automl import run_automl
from tools.validation.robustness import run_robustness_checks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

_DEFAULT_METRICS: dict[str, str] = {
    "binary_classification": "roc_auc",
    "multiclass_classification": "f1_macro",
    "regression": "rmse",
}


class BenchmarkDataset(BaseModel):
    """Description of a single dataset used in the benchmark suite."""

    name: str
    path: str
    target: str
    task_type: str  # "binary_classification", "multiclass_classification", "regression"
    description: str = ""

    def default_metric(self) -> str:
        """Return the canonical metric for this task type."""
        return _DEFAULT_METRICS.get(self.task_type, "rmse")


class BenchmarkResult(BaseModel):
    """Side-by-side result for one dataset: AutoML baseline vs AME pipeline."""

    dataset_name: str
    task_type: str

    # AutoML baseline scores
    automl_score: float
    automl_metric: str
    automl_runtime_seconds: float
    automl_model: str = ""

    # AME scores (populated only when the AME pipeline was run)
    ame_score: float | None = None
    ame_metric: str | None = None
    ame_runtime_seconds: float | None = None
    ame_n_experiments: int | None = None
    ame_leakage_detected: bool = False
    ame_robustness_passed: bool | None = None
    ame_critic_verdict: str | None = None

    # Derived comparison values
    delta: float | None = None  # ame_score - automl_score
    relative_improvement: float | None = None  # delta / |automl_score| * 100

    def compute_delta(self) -> None:
        """Populate *delta* and *relative_improvement* from the raw scores."""
        if self.ame_score is not None:
            self.delta = self.ame_score - self.automl_score
            if abs(self.automl_score) > 1e-12:
                self.relative_improvement = (self.delta / abs(self.automl_score)) * 100.0
            else:
                self.relative_improvement = None


class BenchmarkSuite(BaseModel):
    """Aggregated results across the full benchmark suite."""

    results: list[BenchmarkResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def add_result(self, result: BenchmarkResult) -> None:
        """Append a single result and refresh the aggregate summary."""
        result.compute_delta()
        self.results.append(result)
        self._update_summary()

    def to_markdown(self) -> str:
        """Render the full results table plus summary block as Markdown."""
        if not self.results:
            return "_No benchmark results recorded._"

        header = (
            "| Dataset | Task | AutoML Score | AME Score | Delta | Rel % | "
            "Leakage | Robust | Critic |\n"
            "|---------|------|-------------:|----------:|------:|------:|"
            "---------|--------|--------|\n"
        )
        rows: list[str] = []
        for r in self.results:
            ame = f"{r.ame_score:.4f}" if r.ame_score is not None else "—"
            delta = f"{r.delta:+.4f}" if r.delta is not None else "—"
            rel = f"{r.relative_improvement:+.1f}%" if r.relative_improvement is not None else "—"
            leak = "Yes" if r.ame_leakage_detected else "No"
            robust = {True: "Pass", False: "Fail", None: "—"}[r.ame_robustness_passed]
            critic = r.ame_critic_verdict or "—"
            rows.append(
                f"| {r.dataset_name} | {r.task_type} | {r.automl_score:.4f} "
                f"| {ame} | {delta} | {rel} | {leak} | {robust} | {critic} |"
            )

        summary_lines: list[str] = ["\n### Summary\n"]
        for key, value in self.summary.items():
            label = key.replace("_", " ").title()
            if isinstance(value, float):
                summary_lines.append(f"- **{label}**: {value:.4f}")
            else:
                summary_lines.append(f"- **{label}**: {value}")

        return header + "\n".join(rows) + "\n" + "\n".join(summary_lines) + "\n"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_summary(self) -> None:
        """Recompute aggregate statistics from the current result list."""
        n = len(self.results)
        if n == 0:
            self.summary = {}
            return

        automl_scores = [r.automl_score for r in self.results]
        ame_scores = [r.ame_score for r in self.results if r.ame_score is not None]
        deltas = [r.delta for r in self.results if r.delta is not None]

        self.summary = {
            "n_datasets": n,
            "mean_automl_score": float(np.mean(automl_scores)),
            "mean_ame_score": float(np.mean(ame_scores)) if ame_scores else None,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "median_delta": float(np.median(deltas)) if deltas else None,
            "ame_wins": sum(1 for d in deltas if d > 0),
            "ame_losses": sum(1 for d in deltas if d < 0),
            "ties": sum(1 for d in deltas if d == 0),
            "leakage_detected_count": sum(
                1 for r in self.results if r.ame_leakage_detected
            ),
            "robustness_pass_count": sum(
                1 for r in self.results if r.ame_robustness_passed is True
            ),
        }


# ---------------------------------------------------------------------------
# Built-in benchmark datasets
# ---------------------------------------------------------------------------

BENCHMARK_DATASETS: list[BenchmarkDataset] = [
    BenchmarkDataset(
        name="titanic",
        path="evaluation/datasets/titanic.csv",
        target="Survived",
        task_type="binary_classification",
        description="Titanic survival prediction — small binary classification baseline.",
    ),
    BenchmarkDataset(
        name="iris",
        path="evaluation/datasets/iris.csv",
        target="species",
        task_type="multiclass_classification",
        description="Iris flower classification — canonical multiclass task.",
    ),
    BenchmarkDataset(
        name="boston_housing",
        path="evaluation/datasets/boston_housing.csv",
        target="MEDV",
        task_type="regression",
        description="Boston housing price regression — small tabular regression baseline.",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dataset(dataset: BenchmarkDataset) -> pd.DataFrame:
    """Load a benchmark dataset from its declared path.

    Raises ``FileNotFoundError`` if the CSV is missing.
    """
    try:
        df = pd.read_csv(dataset.path)
    except FileNotFoundError:
        logger.error("Dataset file not found: %s", dataset.path)
        raise
    logger.info(
        "Loaded dataset '%s' — %d rows, %d columns",
        dataset.name,
        len(df),
        len(df.columns),
    )
    return df


def _resolve_metric(task_type: str, metric: str | None) -> str:
    """Pick the scoring metric, falling back to the task-type default."""
    if metric is not None:
        return metric
    return _DEFAULT_METRICS.get(task_type, "rmse")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_automl_baseline(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    *,
    metric: str | None = None,
    seed: int = 42,
) -> BenchmarkResult:
    """Run plain AutoML on *df* and return the baseline result.

    This executes only ``run_automl`` — no auditing, no robustness checks,
    no critic loop — so the result represents what a user would get from a
    traditional AutoML tool.

    Parameters
    ----------
    df:
        Full dataframe (will be split internally).
    target:
        Name of the target column.
    task_type:
        One of ``"binary_classification"``, ``"multiclass_classification"``,
        or ``"regression"``.
    metric:
        Scoring metric.  Defaults to the canonical metric for *task_type*.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    BenchmarkResult
        Populated with ``automl_*`` fields only; ``ame_*`` fields are ``None``.
    """
    resolved_metric = _resolve_metric(task_type, metric)

    split = create_split(
        df, strategy="random", test_size=0.2, seed=seed,
    )

    logger.info("Running AutoML baseline (metric=%s) …", resolved_metric)
    profile = profile_columns(df)
    t0 = time.perf_counter()
    automl_result = run_automl(
        df,
        target=target,
        task_type=task_type,
        split=split,
        profile=profile,
        metric=resolved_metric,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0

    return BenchmarkResult(
        dataset_name=df.attrs.get("name", "unknown"),
        task_type=task_type,
        automl_score=automl_result.score,
        automl_metric=resolved_metric,
        automl_runtime_seconds=elapsed,
        automl_model=automl_result.best_model,
    )


def run_ame_pipeline(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    objective: str,
    *,
    metric: str | None = None,
    seed: int = 42,
    max_experiments: int = 5,
) -> BenchmarkResult:
    """Run the full AME pipeline on *df* and return the enriched result.

    Pipeline stages executed in order:

    1. **Data audit** — profiling, quality checks, leakage detection.
    2. **Experiment** — hypothesis-driven experiment via the experiment runner.
    3. **Robustness checks** — perturbation / invariance tests.
    4. **Critic verdict** — a pass / needs-work / fail judgement.

    Parameters
    ----------
    df:
        Full dataframe.
    target:
        Target column name.
    task_type:
        Task type string.
    objective:
        Free-text objective for the agent (used by the critic).
    metric:
        Scoring metric.
    seed:
        Random seed.
    max_experiments:
        Cap on the number of experiments the AME loop may run.

    Returns
    -------
    BenchmarkResult
        Both ``automl_*`` and ``ame_*`` fields populated, plus ``delta``.
    """
    resolved_metric = _resolve_metric(task_type, metric)
    dataset_name = df.attrs.get("name", "unknown")

    # ---- 1. Data audit ----
    logger.info("[AME] Stage 1 — data audit for '%s'", dataset_name)
    profile = profile_columns(df)
    quality_report = check_data_quality(df)

    split = create_split(
        df, strategy="random", test_size=0.2, seed=seed,
    )

    leakage_result = detect_leakage(df, target=target)
    leakage_detected = leakage_result.has_critical_issues

    # ---- 2. First run plain AutoML for the baseline score ----
    logger.info("[AME] Stage 2 — baseline AutoML")
    t_baseline = time.perf_counter()
    automl_result = run_automl(
        df,
        target=target,
        task_type=task_type,
        split=split,
        profile=profile,
        metric=resolved_metric,
        seed=seed,
    )
    baseline_elapsed = time.perf_counter() - t_baseline
    baseline_score = automl_result.score

    # ---- 3. AME experiment loop ----
    logger.info("[AME] Stage 3 — AME experiment loop (max %d)", max_experiments)
    registry = ExperimentRegistry()
    t_ame = time.perf_counter()
    best_ame_record: ExperimentRecord | None = None

    for i in range(max_experiments):
        hypothesis = f"Experiment {i+1}: explore model space with seed variation"
        try:
            record = run_experiment(
                df,
                target=target,
                task_type=task_type,
                split=split,
                hypothesis=hypothesis,
                registry=registry,
                parent_experiment_id=best_ame_record.experiment_id if best_ame_record else None,
                profile=profile,
                metric=resolved_metric,
                seed=seed + i,
                dataset_version_id=f"bench_{dataset_name}",
            )
        except Exception:
            logger.exception("Experiment %d failed for '%s'", i + 1, dataset_name)
            continue

        if best_ame_record is None or (record.primary_score > best_ame_record.primary_score):
            best_ame_record = record

    ame_elapsed = time.perf_counter() - t_ame
    ame_score = best_ame_record.primary_score if best_ame_record else None
    ame_n = len(list(registry.all()))

    # ---- 4. Robustness checks ----
    robustness_passed: bool | None = None
    if best_ame_record is not None:
        logger.info("[AME] Stage 4 — robustness checks")
        feature_columns = best_ame_record.features or [
            c for c in df.columns if c != target
        ]
        try:
            rob_report = run_robustness_checks(
                df,
                target=target,
                task_type=task_type,
                split=split,
                reference_score=best_ame_record.primary_score,
                metric=resolved_metric,
                feature_columns=feature_columns,
                model_name=best_ame_record.model,
            )
            robustness_passed = rob_report.n_failed == 0
        except Exception:
            logger.exception("Robustness checks failed for '%s'", dataset_name)

    # ---- 5. Critic verdict ----
    logger.info("[AME] Stage 5 — critic verdict")
    critic_verdict = _compute_critic_verdict(
        ame_score,
        baseline_score,
        leakage_detected,
        robustness_passed,
        task_type,
    )

    result = BenchmarkResult(
        dataset_name=dataset_name,
        task_type=task_type,
        # baseline
        automl_score=baseline_score,
        automl_metric=resolved_metric,
        automl_runtime_seconds=baseline_elapsed,
        automl_model=automl_result.best_model,
        # AME
        ame_score=ame_score,
        ame_metric=resolved_metric,
        ame_runtime_seconds=ame_elapsed,
        ame_n_experiments=ame_n,
        ame_leakage_detected=leakage_detected,
        ame_robustness_passed=robustness_passed,
        ame_critic_verdict=critic_verdict,
    )
    result.compute_delta()
    return result


def _compute_critic_verdict(
    ame_score: float | None,
    baseline_score: float,
    leakage_detected: bool,
    robustness_passed: bool | None,
    task_type: str,
) -> str:
    """Produce a simple critic verdict based on AME run outcomes.

    Verdicts
    --------
    - ``"pass"`` — AME matched or improved on the baseline with no integrity
      issues.
    - ``"needs-work"`` — score improved but leakage or robustness problems
      were flagged.
    - ``"fail"`` — AME scored worse than the baseline, or critical integrity
      failures.
    """
    if leakage_detected:
        return "fail"

    if ame_score is None:
        return "fail"

    higher_is_better = task_type != "regression"
    if higher_is_better:
        improved = ame_score >= baseline_score
    else:
        improved = ame_score <= baseline_score

    if not improved:
        return "fail"

    if robustness_passed is False:
        return "needs-work"

    return "pass"


# ---------------------------------------------------------------------------
# Top-level benchmark driver
# ---------------------------------------------------------------------------


def run_benchmark(
    datasets: list[BenchmarkDataset] | None = None,
    *,
    seed: int = 42,
    max_experiments_per_dataset: int = 5,
) -> BenchmarkSuite:
    """Run the full benchmark suite across all given datasets.

    For each dataset the runner executes:

    1. A plain AutoML baseline (``run_automl_baseline``).
    2. The full AME pipeline (``run_ame_pipeline``).
    3. Computes deltas and aggregates them into a ``BenchmarkSuite``.

    Parameters
    ----------
    datasets:
        List of datasets to benchmark.  Defaults to ``BENCHMARK_DATASETS``.
    seed:
        Global random seed.
    max_experiments_per_dataset:
        Maximum experiments per dataset in the AME loop.

    Returns
    -------
    BenchmarkSuite
        Aggregated results with per-dataset rows and a summary dict.
    """
    if datasets is None:
        datasets = BENCHMARK_DATASETS

    suite = BenchmarkSuite()

    for ds in datasets:
        logger.info("=== Benchmarking dataset: %s ===", ds.name)
        try:
            df = _load_dataset(ds)
            df.attrs["name"] = ds.name
        except FileNotFoundError:
            logger.warning("Skipping '%s' — file not found.", ds.name)
            continue

        # --- AutoML baseline ---
        try:
            baseline = run_automl_baseline(
                df,
                target=ds.target,
                task_type=ds.task_type,
                seed=seed,
            )
        except Exception:
            logger.exception("AutoML baseline failed for '%s'", ds.name)
            continue

        # --- AME pipeline ---
        try:
            ame_result = run_ame_pipeline(
                df,
                target=ds.target,
                task_type=ds.task_type,
                objective=ds.description or f"Predict {ds.target}",
                seed=seed,
                max_experiments=max_experiments_per_dataset,
            )
        except Exception:
            logger.exception("AME pipeline failed for '%s'", ds.name)
            # Still record the baseline even if AME fails.
            baseline.dataset_name = ds.name
            suite.add_result(baseline)
            continue

        # Merge: keep the AME result (it already includes baseline numbers).
        suite.add_result(ame_result)

    logger.info("Benchmark complete — %d datasets evaluated.", len(suite.results))
    return suite
