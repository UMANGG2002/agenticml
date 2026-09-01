# ML Agent — Build Spec

An agentic ML engineer: one LLM-driven agent, a deterministic tool library, an experiment ledger, and an adversarial critic pass. Not a multi-agent orchestrator. Not "GPT + AutoML" — GPT decides, Python computes, AutoML is one tool among many.

Use this doc as the working spec while building with Claude Code. Build in the stage order given. Do not skip ahead to later stages before earlier ones pass their acceptance checks.

---

## 0. Non-negotiable design rules

1. **LLM does reasoning and tool selection only.** It never computes a metric, never trains a model by generating ad-hoc code, never invents a number. All computation happens in deterministic Python tool functions; the LLM calls them and interprets structured JSON results.
2. **Every model change requires a hypothesis before it requires a run.** No experiment without: observation → hypothesis → predicted effect → run → result → conclusion. This is the entire novelty of the project — enforce it in the system prompt and in the data model.
3. **Every experiment is logged**, win or lose, to an append-only ledger. Rejected hypotheses are as valuable as accepted ones — they're what lets the agent answer "why did we stop using SMOTE."
4. **A critic pass is mandatory before any model is called final.** Same LLM, a separate adversarial system prompt, explicit job is to find reasons the result is wrong (leakage, overfitting, instability, subgroup failure) — not to congratulate the main loop.
5. **Domain knowledge is retrieved, not assumed.** If the user uploads business rules / feature definitions / policy docs, the agent must retrieve and cite them before accepting or rejecting a feature — it does not reason about the domain from parametric knowledge alone.
6. **A budget stops the loop.** Max experiments, max wall-clock time, or a diminishing-returns threshold. Without this the agent runs forever and the demo looks broken.
7. **Never touch the test set** except for the final, single evaluation. All experimentation happens via CV on train data.

---



## 1. Tech stack


| Layer               | Choice                                                                                                              | Why                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Agent framework     | Anthropic Python SDK, direct tool-use loop (no LangGraph for v1)                                                    | Full control over the loop; fewer abstractions to debug when something goes wrong                  |
| Backend             | Python + FastAPI                                                                                                    | Standard, fast to stand up an API around the agent loop                                            |
| ML                  | scikit-learn, XGBoost, LightGBM                                                                                     | Cover linear, tree, and boosting baselines                                                         |
| AutoML tool         | AutoGluon (or FLAML if lighter weight is preferred)                                                                 | Used as one callable tool, and separately as the baseline in the final benchmark                   |
| Interpretability    | SHAP                                                                                                                | Feature importance and per-prediction explanation                                                  |
| Experiment tracking | SQLite for v1 (upgrade to Postgres only if needed)                                                                  | Zero setup, one file, fully sufficient for the ledger schema below                                 |
| Domain RAG          | Simple embedding store — `chromadb` or even a flat JSON + cosine similarity over `sentence-transformers` embeddings | Don't over-engineer retrieval for v1; a few hundred doc chunks doesn't need a production vector DB |
| Frontend            | Skip for v1. CLI + Jupyter/markdown report output.                                                                  | UI is not what gets you interview signal; the loop and the evidence do                             |


Model to call: use the current Claude model available via the API (Claude Sonnet). Do not hardcode a model string that will go stale — read it from a config value.

---



## 2. Project structure

```
ml-agent/
├── agent/
│   ├── loop.py              # main reasoning loop
│   ├── critic.py            # adversarial critic pass
│   ├── system_prompts.py    # main + critic system prompts
│   └── budget.py            # stopping-rule logic
├── tools/
│   ├── data_tools.py        # inspect_dataset, detect_leakage, profile_columns
│   ├── split_tools.py       # create_split, check_distribution_shift
│   ├── model_tools.py       # train_baseline, train_model, run_automl
│   ├── eval_tools.py        # cross_validate, evaluate_model, check_calibration
│   ├── explain_tools.py     # analyze_errors, run_ablation, shap_importance
│   ├── domain_tools.py      # domain_retrieve
│   └── registry.py          # tool schema definitions passed to the API
├── ledger/
│   ├── schema.sql
│   ├── db.py                 # read/write helpers
│   └── models.py              # dataclasses for Experiment, Hypothesis, etc.
├── domain_kb/
│   ├── ingest.py              # chunk + embed uploaded docs
│   └── retrieve.py
├── eval/
│   ├── run_benchmark.py       # agent vs AutoGluon across OpenML datasets
│   └── datasets.py            # dataset registry for the benchmark
├── reports/
│   └── report_generator.py    # turns ledger rows into a final markdown report
├── api/
│   └── main.py                 # FastAPI entrypoint
├── cli.py                       # `python cli.py run --data churn.csv --goal "predict churn" --metric f1`
├── config.py
└── README.md
```

---



## 3. Experiment ledger — schema

SQLite, one file per project run (`experiments.db`). This is the core artifact of the whole system — get this right first.

```sql
CREATE TABLE datasets (
    id TEXT PRIMARY KEY,
    source_path TEXT,
    n_rows INTEGER,
    n_cols INTEGER,
    target_column TEXT,
    problem_type TEXT,           -- classification | regression
    created_at TIMESTAMP
);

CREATE TABLE data_audit (
    id TEXT PRIMARY KEY,
    dataset_id TEXT REFERENCES datasets(id),
    missing_values_json TEXT,
    leakage_flags_json TEXT,     -- [{feature, reason, risk}]
    class_imbalance_json TEXT,
    distribution_shift_json TEXT,
    excluded_features_json TEXT, -- features removed and why
    created_at TIMESTAMP
);

CREATE TABLE hypotheses (
    id TEXT PRIMARY KEY,
    dataset_id TEXT REFERENCES datasets(id),
    parent_hypothesis_id TEXT REFERENCES hypotheses(id),  -- nullable, for the experiment graph
    observation TEXT NOT NULL,
    hypothesis_text TEXT NOT NULL,
    predicted_effect TEXT,
    status TEXT,                 -- proposed | running | supported | rejected
    created_at TIMESTAMP
);

CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT REFERENCES hypotheses(id),
    model_type TEXT,
    config_json TEXT,            -- hyperparameters / feature changes
    cv_metric_mean REAL,
    cv_metric_std REAL,
    metric_name TEXT,
    baseline_metric REAL,        -- what it's being compared against
    delta REAL,
    confidence REAL,             -- e.g. from a paired CV significance test
    conclusion TEXT,             -- supported | rejected | inconclusive
    runtime_seconds REAL,
    random_seed INTEGER,
    created_at TIMESTAMP
);

CREATE TABLE critic_reviews (
    id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(id),
    leakage_check TEXT,
    stability_check TEXT,        -- across CV folds / seeds
    subgroup_performance_json TEXT,
    calibration_check TEXT,
    verdict TEXT,                -- approved | rejected | flagged
    warnings_json TEXT,
    created_at TIMESTAMP
);

CREATE TABLE domain_citations (
    id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(id),
    doc_chunk_id TEXT,
    doc_source TEXT,
    quoted_rule TEXT,
    applied_to_feature TEXT,
    created_at TIMESTAMP
);
```

Every experiment must be traceable back to a hypothesis and, where relevant, a domain citation. If a query like "why did we stop using SMOTE" can't be answered by a `SELECT` against these tables, the ledger is incomplete — fix it before moving on.

---



## 4. Tool library — exact function signatures

Implement these as plain Python functions first, then wrap each in the Anthropic tool-use JSON schema. Keep the LLM-facing description short and behavior-focused; keep validation and error handling inside the function, not the prompt.

```python
# data_tools.py
def inspect_dataset(path: str) -> dict:
    """Returns: n_rows, n_cols, dtypes, target_candidates, sample_rows."""

def detect_target(path: str, user_goal: str) -> dict:
    """Returns: target_column, problem_type, confidence, reasoning."""

def profile_columns(path: str) -> dict:
    """Per-column: missing_pct, cardinality, dtype, outlier_flag."""

def detect_leakage(path: str, target: str) -> dict:
    """Returns: suspicious_features: [{feature, correlation, reason, risk}]."""

def detect_class_imbalance(path: str, target: str) -> dict:
    """Returns: class_distribution, imbalance_ratio, recommendation."""

# split_tools.py
def create_split(path: str, strategy: str, target: str) -> dict:
    """strategy: random | temporal | grouped. Returns: train_path, test_path, split_metadata."""

def check_distribution_shift(train_path: str, test_path: str) -> dict:
    """Per-feature shift score (e.g. PSI). Returns: shifted_features, severity."""

# model_tools.py
def train_baseline(train_path: str, target: str, model_type: str) -> dict:
    """model_type: logistic_regression | random_forest | xgboost | lightgbm.
    Returns: model_id, cv_metrics."""

def run_automl(train_path: str, target: str, time_budget_seconds: int) -> dict:
    """Calls AutoGluon/FLAML. Returns: best_model_id, leaderboard, cv_metrics."""

def train_model(train_path: str, target: str, model_type: str, config: dict) -> dict:
    """Trains with specific hyperparameters/features. Returns: model_id, cv_metrics."""

# eval_tools.py
def cross_validate(model_id: str, folds: int, metric: str) -> dict:
    """Returns: mean, std, per_fold_scores."""

def evaluate_model(model_id: str, test_path: str, metric: str) -> dict:
    """Single evaluation against held-out test set. Use only when finalizing."""

def check_calibration(model_id: str, data_path: str) -> dict:
    """Returns: calibration_curve_points, brier_score."""

def compare_experiments(experiment_id_a: str, experiment_id_b: str) -> dict:
    """Paired significance test on CV folds. Returns: delta, p_value, verdict."""

# explain_tools.py
def analyze_errors(model_id: str, data_path: str) -> dict:
    """Returns: worst_predictions, error_patterns_by_segment."""

def run_ablation(model_id: str, feature_group: list[str]) -> dict:
    """Retrains without the given features. Returns: metric_drop."""

def shap_importance(model_id: str, data_path: str) -> dict:
    """Returns: top_features, per_feature_shap_summary."""

def check_subgroup_performance(model_id: str, data_path: str, subgroup_cols: list[str]) -> dict:
    """Returns: metric per subgroup, worst-performing subgroup, gap size."""

# domain_tools.py
def domain_retrieve(query: str, top_k: int = 5) -> dict:
    """Returns: chunks: [{source, text, score}]. Empty if no domain docs uploaded."""

# ledger tools (used by the agent to read its own history, not just write to it)
def save_experiment(hypothesis: dict, experiment: dict) -> str:
    """Writes to ledger. Returns experiment_id."""

def get_past_experiments(dataset_id: str, filter: dict = None) -> list[dict]:
    """Lets the agent check what's already been tried before proposing a duplicate."""
```

Rule: the agent must call `get_past_experiments` before proposing a new hypothesis on a dataset it's already worked on. Put this in the system prompt as a hard requirement, not a suggestion.

---



## 5. Main agent loop — system prompt (draft, refine in code)

```
You are an autonomous ML engineer. You have tools for data inspection, domain
knowledge retrieval, model training (including AutoML), evaluation, and
explanation. You do not compute metrics or write training code yourself —
you call tools and reason about their structured results.

Before proposing any experiment:
1. Check get_past_experiments — do not repeat a rejected hypothesis.
2. State an observation (what you currently see in the data or results).
3. State a hypothesis (a specific, falsifiable claim).
4. State the predicted effect (what you expect to change and roughly by how much).
5. Check domain_retrieve for any relevant business rule or constraint before
   including or excluding a feature. If domain docs exist and you skip this
   step, you are wrong by default.
6. Estimate the cost of the experiment (time/compute) before running it.

Never modify or evaluate against the test set except for the single final
evaluation, explicitly requested by the user or triggered by the stopping rule.

Every experiment must be saved to the ledger via save_experiment, including
rejected and inconclusive ones.

Stop proposing new experiments when any of the following is true:
- the user's target metric is reached,
- the experiment budget (count or time) is exhausted,
- the last 3 experiments each improved the metric by less than the configured
  minimum improvement threshold.

When you stop, produce a final report: best model, full experiment
history with hypotheses and outcomes, features excluded and why (with domain
citations where applicable), and known limitations or unresolved risks
(e.g. "leakage possible in feature X, could not confirm").
```

---



## 6. Critic pass — system prompt (separate call, same model)

Run this **once**, after the main loop proposes a final model, before it's accepted.

```
You are an adversarial ML reviewer. You did not build this model — your job
is to find reasons it should not be trusted. Assume the main agent's
self-report may be optimistic. Do not soften your findings to be polite.

Check, using the available tools:
1. Leakage — re-run detect_leakage independently; check the final feature
   set against the domain rules retrieved during training.
2. Stability — retrain across at least 3 random seeds; flag if the metric
   varies by more than the configured tolerance.
3. Subgroup performance — check performance across any available
   demographic/categorical subgroups; flag any subgroup underperforming the
   overall metric by more than the configured tolerance.
4. Calibration — check_calibration; flag if predicted probabilities are
   poorly calibrated for a use case that needs them (e.g. risk scoring).
5. Distribution shift — confirm check_distribution_shift was run and no
   material shift between train and test was ignored.

Return a verdict: approved | rejected | flagged, with specific numeric
evidence for each check, not qualitative reassurance. If rejected or
flagged, state exactly what the main agent should try next.
```

Write this to `critic_reviews`. If verdict is `rejected`, hand control back to the main loop with the critic's findings appended as new observations — do not silently retry.

---



## 7. Domain knowledge layer

- Ingest: user uploads `.md`/`.txt`/`.pdf` docs (policies, feature definitions, business rules). Chunk by heading/paragraph (~300-500 tokens per chunk), embed with a small local sentence-transformer model, store in `chromadb` (or a flat file — no need for anything heavier at this scale).
- Retrieval is triggered by the agent explicitly calling `domain_retrieve`, not injected automatically into every prompt — this keeps the agent's reasoning about *why* it looked something up visible in the trace.
- Every feature inclusion/exclusion decision that cites a domain rule must be logged in `domain_citations`, linking the experiment to the specific doc chunk. This is what makes "the agent caught a business-rule violation AutoML would have missed" a checkable claim rather than an assertion.

---



## 8. Stopping rule / budget

Implement as a small pure function, not a prompt instruction alone (the LLM should also see this rule, but it should be enforced in code as a hard stop):

```python
def should_stop(experiments: list[dict], config: dict) -> tuple[bool, str]:
    if len(experiments) >= config["max_experiments"]:
        return True, "max_experiments_reached"
    if elapsed_seconds(experiments) >= config["max_runtime_seconds"]:
        return True, "max_runtime_reached"
    if best_metric(experiments) >= config["target_metric"]:
        return True, "target_metric_reached"
    last_n = experiments[-config["patience"]:]
    if len(last_n) == config["patience"] and all(
        e["delta"] < config["min_improvement"] for e in last_n
    ):
        return True, "diminishing_returns"
    return False, ""
```

---



## 9. Benchmark — the evidence section

This is what turns the project from a demo into a claim. Run after stage 4 passes.

- Dataset pool: 15-20 OpenML/Kaggle tabular datasets, mix of classification/regression, at least 3 with a known or injected leakage feature, at least 2 with a stated synthetic "business constraint" doc for the domain layer to catch.
- Run both:
  - **Baseline**: AutoGluon alone, same time budget.
  - **Agent**: full loop (data audit → domain retrieval → AutoML tool → experiment loop → critic), same time budget.
- Record per dataset: final metric, number of experiments run, leakage caught (yes/no), domain-rule violations caught (yes/no), reproducibility across 3 seeds (std dev).
- Report as a single markdown table. Do not editorialize beyond the numbers — let the table make the claim.

Target claim shape for the README:

> On N/20 datasets the agent matched or beat the AutoGluon baseline on [metric]. On the 3 datasets with injected leakage features, the agent's critic pass caught leakage in X/3 cases; the AutoGluon-only baseline has no mechanism to catch this and used the leaky feature in all 3.

---



## 10. Build order and acceptance checks

**Stage 1 (week 1) — plumbing**

- CLI takes a CSV + goal string + metric name.
- `inspect_dataset`, `detect_target`, `train_baseline`, `run_automl` wired up.
- Agent produces one report: baseline metric, dataset summary.
- ✅ Acceptance: running the CLI on a sample CSV produces a correct baseline metric with no manual intervention.

**Stage 2 (weeks 2-3) — the loop**

- Hypothesis → experiment → evidence loop implemented against the ledger schema.
- `get_past_experiments` enforced before new proposals.
- Domain RAG wired up (`domain_retrieve`, `domain_citations`).
- ✅ Acceptance: `SELECT` against the ledger can answer "what was tried, in what order, and why was each accepted/rejected" for a full run, unaided by prompt/chat history.

**Stage 3 (week 4) — critic + stopping**

- Critic pass implemented as a separate system prompt/call.
- Stopping rule enforced in code, not just prompt.
- ✅ Acceptance: feed the agent a dataset with an intentionally injected leaky feature; critic must catch and flag it, and the final report must reflect the rejection.

**Stage 4 (weeks 5-6) — benchmark**

- Run the full benchmark suite (Section 9).
- Write results into `README.md` as the evidence section.
- ✅ Acceptance: the benchmark table exists, is reproducible via `eval/run_benchmark.py`, and the leakage/domain-rule claim in the README is backed by the table, not asserted separately.

Do not start Stage 4 before Stage 3's acceptance check passes — a benchmark without a working critic just measures AutoGluon.

---



## 11. What to explicitly not build

- No multi-agent orchestration layer, no agent-to-agent handoff protocol.
- No deployment/serving/monitoring/CI-CD — out of scope for this project.
- No custom UI for v1 — CLI + generated markdown report is sufficient.
- No "research agent that searches papers" — nice-to-have, cut for v1; add only after stage 4 is done and benchmarked.
- No dependency on a single hardcoded model string — read the model name from config so it can be swapped without code changes.

