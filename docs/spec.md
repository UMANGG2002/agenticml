# Agentic ML Engineer — End-to-End Vibe Coding Specification

## 0. Project definition

### Working name
**Agentic ML Engineer (AME)**

### One-line description
An autonomous ML engineering agent that uses AutoML as one tool, but goes beyond model search by understanding the ML objective, auditing data, retrieving domain knowledge, designing experiments, challenging results, checking robustness/fairness/drift, and producing reproducible evidence-backed recommendations.

### Core research question
> Can an agentic ML engineering workflow improve the quality, reliability, domain alignment, and experimental efficiency of ML solutions beyond conventional AutoML?

### Core distinction

**Traditional AutoML**
```text
Dataset → preprocessing → model search → HPO → best model
```

**AME**
```text
Goal
 ↓
Problem formulation
 ↓
Domain/context retrieval
 ↓
Data audit
 ↓
Leakage/bias checks
 ↓
AutoML baseline
 ↓
Error analysis
 ↓
Hypothesis generation
 ↓
Experiment design
 ↓
Experiment execution
 ↓
Critique / falsification
 ↓
Robustness + fairness + drift checks
 ↓
Research-guided next experiments
 ↓
Final model + evidence + reproducibility report
```

---

# 1. What is in scope

## Must-have MVP

1. CSV upload
2. Dataset profiling
3. Target detection/confirmation
4. Problem-type detection
5. Data-quality checks
6. Leakage detection
7. Train/validation/test strategy
8. AutoML baseline
9. Agent-generated experiment hypotheses
10. Tool-based experiment execution
11. Experiment tracking
12. Model comparison
13. Error analysis
14. Critic/reviewer pass
15. Domain-knowledge RAG
16. Final report
17. Reproducible experiment artifacts
18. Human approval before expensive/risky actions

## Phase-2

- Web/research search
- Fairness analysis
- Temporal validation
- Distribution-shift tests
- Calibration
- Feature ablation
- Multiple seeds
- Model registry
- MLflow integration
- Git integration
- Scheduled monitoring
- Deployment suggestions

## Explicitly NOT promised

Do not claim that the system:
- guarantees the globally best model
- solves causality automatically
- eliminates bias
- replaces ML engineers
- understands every domain without supplied knowledge
- guarantees production safety
- proves a model is causally valid

The system **detects, evaluates, warns, recommends, and gathers evidence**.

---

# 2. Product principles

1. **AutoML is a tool, not the product.**
2. **The LLM reasons; deterministic code computes.**
3. **Every experiment needs a hypothesis or explicit reason.**
4. **Never use the test set for iterative optimization.**
5. **Every result must be reproducible.**
6. **Never claim improvement without a valid comparison.**
7. **Domain knowledge must affect decisions, not merely appear in chat.**
8. **The critic must be able to reject the main agent's conclusions.**
9. **Every important decision must have evidence.**
10. **The agent must stop when the budget or stopping criteria are reached.**

---

# 3. High-level architecture

```text
                         ┌─────────────────────┐
                         │       Web UI        │
                         │ Next.js + React     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │    API Gateway      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Agent Orchestrator│
                         │ OpenAI Responses /  │
                         │ Agents SDK           │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
       Domain/RAG Tools       ML Tools              Research Tools
              │                     │                     │
              ▼                     ▼                     ▼
       Vector store             Python ML             Web/search
                              execution sandbox
              │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Experiment Store    │
                         │ PostgreSQL + object  │
                         │ storage             │
                         └─────────────────────┘
```

---

# 4. Recommended stack

## Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS
- Plotly or another interactive chart library

## Backend
- Python
- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL

## Agent
- OpenAI Responses API / Agents SDK
- Function tools for ML operations
- Structured outputs for decisions
- Tracing/evaluation for agent behavior

## ML
- pandas
- numpy
- scikit-learn
- XGBoost
- LightGBM
- Optuna
- SHAP
- scipy
- imbalanced-learn where appropriate

## Experiment tracking
- PostgreSQL initially
- MLflow optionally
- Object storage for datasets/models/reports

## Execution
- Isolated Docker/container sandbox
- Resource/time limits
- No unrestricted host access

## Retrieval
- Start with PostgreSQL + embeddings/vector extension or a simple vector store
- Store document chunks with metadata
- Retrieve only relevant context

## Testing
- pytest
- unit tests
- integration tests
- agent trajectory/evaluation tests

---

# 5. Repository structure

Use this structure:

```text
agentic-ml-engineer/
│
├── apps/
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   └── dependencies.py
│   │
│   └── web/
│       ├── app/
│       ├── components/
│       └── lib/
│
├── agent/
│   ├── orchestrator.py
│   ├── prompts/
│   │   ├── system.md
│   │   ├── planner.md
│   │   └── critic.md
│   ├── policies.py
│   ├── state.py
│   └── schemas.py
│
├── tools/
│   ├── data/
│   │   ├── profile.py
│   │   ├── quality.py
│   │   ├── leakage.py
│   │   └── split.py
│   │
│   ├── ml/
│   │   ├── automl.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── tuning.py
│   │   ├── ablation.py
│   │   └── errors.py
│   │
│   ├── validation/
│   │   ├── robustness.py
│   │   ├── fairness.py
│   │   ├── calibration.py
│   │   └── drift.py
│   │
│   └── research/
│       └── search.py
│
├── domain/
│   ├── ingestion.py
│   ├── retrieval.py
│   └── knowledge.py
│
├── experiments/
│   ├── runner.py
│   ├── registry.py
│   └── artifacts.py
│
├── db/
│   ├── models.py
│   ├── session.py
│   └── migrations/
│
├── evaluation/
│   ├── benchmark.py
│   ├── agent_eval.py
│   └── datasets/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── docker/
├── scripts/
├── docs/
├── .env.example
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

# 6. Core agent architecture

Start with **one orchestrator agent**, not many agents.

The orchestrator can call specialized tools.

```text
                   ORCHESTRATOR
                        │
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
    DATA TOOLS       ML TOOLS       KNOWLEDGE TOOLS
       │                │                │
       └────────────────┼────────────────┘
                        ▼
                    EXPERIMENT
                        │
                        ▼
                      CRITIC
```

Only introduce subagents when there is a measured reason.

Possible later subagents:

- Data Detective
- Researcher
- Experiment Scientist
- Critic
- MLOps Reviewer

---

# 7. Agent state

Maintain explicit state.

```python
class AgentState:
    project_id: str
    dataset_version_id: str
    objective: str
    task_type: str | None
    target_column: str | None

    domain_constraints: list
    business_constraints: list

    baseline_experiment_id: str | None
    best_experiment_id: str | None

    hypotheses: list
    completed_experiments: list
    rejected_experiments: list

    budget:
        max_experiments
        max_runtime_seconds
        max_compute_cost

    stopping_reason: str | None
```

Do not rely on conversation history alone as the source of truth.

Persist important state in PostgreSQL.

---

# 8. Tool catalogue

## Dataset tools

### `inspect_dataset`
Input:
```json
{
  "dataset_version_id": "..."
}
```

Returns:
- rows
- columns
- dtypes
- sample
- memory usage
- duplicate count

### `profile_columns`

Returns:
- missing percentage
- unique count
- numeric statistics
- categorical distribution
- suspicious values

### `detect_target`

Returns candidate target columns with confidence and reasons.

Never silently choose an uncertain target. Ask the user when confidence is low.

### `detect_leakage`

Checks:
- target-derived columns
- suspicious correlations
- post-outcome fields
- temporal leakage
- duplicate/near-duplicate contamination
- train/test contamination

This tool should return **evidence**, not just a boolean.

### `create_split`

Supports:
- random split
- stratified split
- group split
- time-based split

The split strategy must depend on the problem.

---

# 9. AutoML tool

Implement a wrapper:

```python
run_automl(
    dataset_version_id,
    target,
    task_type,
    metric,
    time_budget,
    constraints
)
```

The AutoML layer can use an existing framework.

Its job:

```text
preprocessing
model candidates
hyperparameter search
cross-validation
baseline comparison
best candidate
```

Return:

```json
{
  "experiment_id": "exp_001",
  "best_model": "xgboost",
  "metric": "f1",
  "score": 0.84,
  "cv_mean": 0.84,
  "cv_std": 0.012,
  "runtime_seconds": 91
}
```

AutoML must **not** be allowed to touch the held-out test set during optimization.

---

# 10. Experiment tool

Every experiment must have:

```text
experiment_id
parent_experiment_id
hypothesis
objective
dataset_version
split_strategy
features
preprocessing
model
hyperparameters
seed
metrics
runtime
artifacts
conclusion
next_action
```

Example:

```json
{
  "experiment_id": "exp_017",
  "parent_experiment_id": "exp_014",
  "hypothesis": "Threshold optimization will improve F1 because recall is currently limiting performance.",
  "change": {
    "decision_threshold": 0.37
  },
  "metrics": {
    "f1": 0.871,
    "precision": 0.86,
    "recall": 0.883
  },
  "conclusion": "supported",
  "next_action": "run multi-seed validation"
}
```

---

# 11. Hypothesis engine

The agent should not randomly generate experiments.

Use this structure:

```text
Observation
    ↓
Weakness
    ↓
Hypothesis
    ↓
Expected effect
    ↓
Experiment
    ↓
Success criterion
```

Example:

```text
Observation:
Minority recall = 0.61

Weakness:
Poor minority-class detection

Hypothesis:
Class weighting or threshold optimization may improve recall.

Expected effect:
F1 increases without unacceptable precision loss.

Experiment:
Compare:
1. baseline
2. class weights
3. threshold optimization

Success criterion:
F1 improvement >= configured minimum
and no unacceptable precision degradation.
```

---

# 12. Critic / falsification layer

The critic must independently inspect results.

It should check:

1. Data leakage
2. Test-set contamination
3. Overfitting
4. Cross-validation variance
5. Random-seed sensitivity
6. Metric selection
7. Temporal validity
8. Subgroup performance
9. Calibration
10. Distribution shift
11. Feature dependence
12. Reproducibility

Critic output:

```json
{
  "verdict": "reject",
  "confidence": 0.89,
  "issues": [
    {
      "type": "temporal_generalization",
      "severity": "high",
      "evidence": "Random split F1=0.89, temporal F1=0.67"
    }
  ],
  "required_next_actions": [
    "run time-based validation"
  ]
}
```

A critic rejection must be able to block model promotion.

---

# 13. Domain knowledge layer

Domain knowledge is not "GPT knows the domain."

The system should support user-provided knowledge:

```text
domain/
├── business_rules
├── feature_definitions
├── policies
├── documentation
├── research
└── expert_notes
```

Pipeline:

```text
documents
 ↓
parse
 ↓
chunk
 ↓
embed/index
 ↓
retrieve relevant chunks
 ↓
agent decision
```

Each retrieved item should contain:

```text
source
document_id
chunk_id
text
metadata
timestamp/version
```

The agent must cite the knowledge source internally in its decision record.

---

# 14. Domain knowledge must affect ML behavior

Do NOT use RAG only for answering questions.

Domain rules should become machine-readable constraints.

Example:

```json
{
  "rule": "income_after_application_date_must_not_be_used",
  "type": "feature_constraint",
  "severity": "critical"
}
```

The agent can then:

```text
domain rule
 ↓
feature exclusion
 ↓
experiment constraint
 ↓
evaluation constraint
```

Another example:

```text
Business says:
False negatives cost 5x more than false positives.

Agent changes:
- evaluation metric
- threshold analysis
- expected-cost calculation
- model-selection decision
```

---

# 15. Research capability

When the agent reaches a plateau:

```text
current best
 ↓
error analysis
 ↓
research question
 ↓
search relevant literature/docs
 ↓
extract candidate technique
 ↓
create hypothesis
 ↓
run experiment
```

Research should never automatically become "truth."

The agent must convert retrieved information into a **testable hypothesis**.

---

# 16. Error analysis

For classification:
- confusion matrix
- per-class precision/recall/F1
- threshold analysis
- subgroup performance
- false-positive examples
- false-negative examples

For regression:
- residual distribution
- residual vs prediction
- residual vs features
- subgroup RMSE/MAE
- extreme-error analysis

For time series:
- temporal residuals
- recent-vs-historical performance
- seasonality errors
- future holdout

---

# 17. Robustness layer

Run only checks relevant to the task.

Possible checks:

```text
multi-seed evaluation
cross-validation variance
temporal validation
group validation
noise sensitivity
feature perturbation
missing-feature sensitivity
distribution shift
calibration
subgroup performance
```

Return evidence:

```text
Baseline F1: 0.871
5-seed mean: 0.866
5-seed std: 0.009
Temporal F1: 0.801
```

The final report should distinguish:

- validation performance
- robustness performance
- test performance

---

# 18. Fairness layer

Only activate when protected/sensitive attributes are available and legally/ethically appropriate.

Evaluate:
- group metrics
- TPR/FNR differences
- FPR differences
- calibration differences where applicable

The system should report disparities and recommendations, not claim that one metric universally defines fairness.

---

# 19. Causality layer

Do not claim automatic causal inference.

Instead:

```text
If user asks:
"Will changing X cause Y?"
```

Agent responds with a causal warning and may recommend:

- causal DAG
- confounder analysis
- experimental design
- causal inference methods

Predictive feature importance is not causal evidence.

---

# 20. Deployment/MLOps layer

Later add:

```text
model registry
 ↓
versioning
 ↓
deployment recommendation
 ↓
monitoring plan
 ↓
drift detection
 ↓
performance monitoring
 ↓
retraining trigger
```

The agent can generate:

```text
Deployment readiness:
CONDITIONAL

Required before deployment:
1. temporal validation
2. subgroup review
3. calibration check
4. monitoring for feature drift
5. rollback strategy
```

It should not deploy production models without explicit human approval.

---

# 21. Safety and execution controls

Never allow unrestricted LLM-generated code to run on the API server.

Use a sandbox.

Restrictions:
- CPU limit
- RAM limit
- runtime limit
- disk limit
- network disabled by default
- filesystem isolation
- allowed package list
- process limit
- output-size limit

Every execution gets:
- experiment ID
- container ID
- code snapshot
- dependency snapshot
- seed
- logs
- exit code

---

# 22. Budget system

Project configuration:

```json
{
  "max_experiments": 30,
  "max_runtime_minutes": 30,
  "max_compute_cost": 5,
  "minimum_improvement": 0.002
}
```

Agent must reason about cost.

Stop when:
- target is achieved
- budget exhausted
- repeated experiments show negligible gains
- critic finds unresolved critical issue
- user stops the run

---

# 23. Experiment graph

Represent experiments as a directed graph.

```text
Dataset v1
   ↓
Baseline #1
   ├── Class weights #2
   │       ↓
   │   Threshold #4
   │
   └── Feature engineering #3
           ↓
       XGBoost #5
           ↓
       Ablation #6
```

Relationships:
- parent
- derived-from
- invalidated-by
- improves-on
- rejects
- confirms
- depends-on

This graph becomes the agent's long-term experiment memory.

---

# 24. Database schema

Minimum tables:

```text
users
projects
datasets
dataset_versions
domain_documents
domain_chunks
experiments
experiment_metrics
hypotheses
models
model_versions
artifacts
agent_runs
tool_calls
critic_reviews
constraints
```

Important indexes:
- project_id
- dataset_version_id
- experiment_id
- parent_experiment_id
- created_at
- metric name

---

# 25. API endpoints

Minimum:

```text
POST   /projects
GET    /projects/{id}

POST   /projects/{id}/datasets
GET    /datasets/{id}/profile

POST   /projects/{id}/knowledge
GET    /projects/{id}/knowledge

POST   /projects/{id}/runs
GET    /runs/{id}
POST   /runs/{id}/stop

GET    /projects/{id}/experiments
GET    /experiments/{id}

GET    /projects/{id}/models
GET    /projects/{id}/report
```

Use WebSocket/SSE for live agent progress.

---

# 26. Frontend screens

## Project dashboard

Show:

```text
Dataset
Objective
Current best model
Best metric
Experiment count
Runtime
Budget remaining
Agent status
```

## Agent console

Show:

```text
Observation
Hypothesis
Tool call
Result
Decision
Next action
```

## Experiment timeline

Show:
- experiment number
- hypothesis
- metric
- status
- parent
- reason

## Model comparison

Show:
- metric table
- CV mean/std
- runtime
- model complexity
- robustness
- fairness
- calibration

## Domain knowledge

Show:
- uploaded documents
- extracted rules
- source references
- active constraints

## Final report

Sections:
1. Objective
2. Data
3. Domain constraints
4. Data-quality findings
5. Leakage findings
6. Baseline
7. Experiments
8. Best model
9. Robustness
10. Fairness
11. Limitations
12. Deployment recommendation
13. Reproducibility information

---

# 27. Agent system prompt

Use a strict system prompt along these lines:

```text
You are an autonomous machine-learning engineering agent.

Your goal is not merely to maximize a validation metric.
Your goal is to produce a reliable, reproducible, domain-aligned ML solution.

Rules:

1. Understand the user's objective before training.
2. Confirm the target when uncertain.
3. Inspect the dataset before modeling.
4. Detect possible leakage before trusting results.
5. Never use the held-out test set for iterative optimization.
6. Treat domain rules as constraints.
7. Every experiment must have a reason or hypothesis.
8. Inspect previous experiments before proposing a new one.
9. Prefer experiments that address known weaknesses.
10. Do not claim causality from predictive relationships.
11. Do not claim fairness without evaluation.
12. Do not claim robustness without appropriate validation.
13. Record every experiment.
14. Never hide failed experiments.
15. Prefer reproducible experiments.
16. Respect compute/time/experiment budgets.
17. Ask for human input when requirements are ambiguous or decisions are high-impact.
18. A critic rejection must be addressed before model promotion.
19. AutoML is one tool in the workflow, not the final authority.
20. Stop when the target is achieved, the budget is exhausted, or further experiments are unlikely to provide meaningful improvement.

For each experiment record:
- observation
- hypothesis
- expected outcome
- exact change
- dataset/split
- metric
- result
- conclusion
- confidence
- next action
```

---

# 28. Structured agent decisions

Do not parse free-form text when a decision can be structured.

Example:

```json
{
  "action": "run_experiment",
  "hypothesis": "...",
  "reason": "...",
  "tool": "run_experiment",
  "parameters": {},
  "expected_metric": {
    "name": "f1",
    "minimum_improvement": 0.01
  },
  "estimated_cost": 0.42
}
```

Use schema validation.

---

# 29. MVP execution flow

Implement exactly this first:

```text
1. User creates project
2. User uploads CSV
3. User describes objective
4. Agent profiles dataset
5. Agent detects/asks for target
6. Agent detects task type
7. Agent performs leakage checks
8. Agent proposes split
9. User approves if ambiguous
10. Agent runs AutoML baseline
11. Agent analyzes errors
12. Agent creates hypothesis
13. Agent runs experiment
14. Agent compares result
15. Critic reviews
16. Agent decides next experiment
17. Repeat until stopping condition
18. Run final validation
19. Generate report
20. Save model + experiment artifacts
```

---

# 30. Vibe-coding rules

When coding with an AI coding assistant:

### Rule 1
Never ask:

> "Build the whole application."

Ask for one vertical slice.

Good:

> "Implement `detect_leakage()` with tests and a typed result schema."

### Rule 2
Always request:
- exact files changed
- assumptions
- tests
- error handling
- no unrelated refactors

### Rule 3
Build in this order:

```text
database
 ↓
schemas
 ↓
deterministic ML tools
 ↓
experiment runner
 ↓
agent tool wrappers
 ↓
agent loop
 ↓
API
 ↓
UI
```

### Rule 4
Do not build the UI first.

The agent's deterministic ML workflow is the core product.

---

# 31. First coding prompts

## Prompt 1 — repository

```text
Create the repository structure described in docs/spec.md.

Do not implement business logic yet.

Create:
- Python backend
- Next.js frontend
- PostgreSQL configuration
- Docker Compose
- pytest configuration
- environment template

Keep the project runnable.
```

## Prompt 2 — database

```text
Implement the SQLAlchemy database models specified in docs/spec.md.

Requirements:
- typed models
- foreign keys
- timestamps
- migrations
- indexes
- tests

Do not implement agent logic.
```

## Prompt 3 — dataset profiler

```text
Implement inspect_dataset(), profile_columns(), and detect_target().

Requirements:
- deterministic
- typed Pydantic outputs
- handle missing/corrupt data
- never modify the source dataset
- tests for numeric, categorical, missing, duplicate, and empty columns
```

## Prompt 4 — leakage detector

```text
Implement detect_leakage().

Detect:
- target-derived features
- suspiciously perfect relationships
- post-outcome features when temporal metadata exists
- duplicate contamination
- train/test contamination

Return evidence and severity.

Do not automatically delete features.
```

## Prompt 5 — AutoML

```text
Implement run_automl() as a tool.

It must:
- receive an explicit dataset version
- receive target and task
- use only training data for optimization
- maintain a separate final test set
- record configuration
- record metrics
- save model artifacts
- return an experiment ID

Add tests proving test data is not used during tuning.
```

## Prompt 6 — experiment runner

```text
Implement the experiment registry and run_experiment().

Every experiment must record:
hypothesis, parent, dataset version, split, model, parameters, metrics, runtime, seed, artifact paths, conclusion.

Add tests for reproducibility and parent/child relationships.
```

## Prompt 7 — agent

```text
Implement the ML Scientist orchestrator using the system prompt and tool schemas in docs/spec.md.

The agent must:
- inspect state
- select tools
- record tool calls
- record hypotheses
- respect budgets
- never directly execute arbitrary code
- use the sandbox execution tool
- stop on configured conditions
```

## Prompt 8 — critic

```text
Implement an independent critic pass.

The critic must review an experiment without changing its result and return:
verdict, confidence, issues, evidence, required_next_actions.

A critical rejection must prevent model promotion.
```

---

# 32. Evaluation benchmark

This is mandatory for a serious final project.

Create a benchmark of 20–50 public datasets across:

```text
binary classification
multiclass classification
regression
imbalanced classification
temporal data
grouped data
datasets with missing values
datasets containing potential leakage
```

Compare:

```text
A = conventional AutoML
B = AME without domain knowledge
C = AME with domain knowledge
```

Optional:

```text
D = human-designed baseline
```

Measure:

## Model quality
- F1
- PR-AUC
- ROC-AUC
- RMSE/MAE as appropriate

## Reliability
- leakage detection rate
- robustness
- calibration
- subgroup performance
- temporal generalization

## Efficiency
- wall-clock time
- number of experiments
- compute cost

## Agent quality
- correct tool selection
- valid hypotheses
- invalid experiment rate
- correct conclusions
- stopping quality

---

# 33. Ablation study

To prove which components matter, remove them one at a time.

```text
Full system
   ↓
- domain knowledge
   ↓
- critic
   ↓
- research
   ↓
- autonomous hypothesis loop
   ↓
- error analysis
```

Compare results.

This gives evidence for:

> "The improvement comes from component X."

rather than simply:

> "GPT made it better."

---

# 34. Key metrics for your research paper

Use:

### Performance gain
```text
ΔMetric = AgentMetric - AutoMLMetric
```

### Relative improvement
```text
Relative improvement =
(Agent - AutoML) / |AutoML|
```

### Experiment efficiency
```text
metric improvement / experiment count
```

### Compute efficiency
```text
metric improvement / compute cost
```

### Reliability
Track:
- leakage caught
- invalid claims caught
- robustness failures
- reproducibility failures

Report median and distribution, not only the single best dataset.

---

# 35. Example final report

```text
PROJECT: Customer Churn

OBJECTIVE
Predict customers likely to churn.

BUSINESS CONSTRAINT
False negatives are significantly more costly than false positives.

DATA
1,200,000 rows
82 features

DATA AUDIT
✓ Missing values handled
✓ Duplicate contamination checked
⚠ 2 potential leakage features excluded

BASELINE
AutoML best model: XGBoost
F1: 0.842

AGENTIC EXPERIMENTS
17 experiments
6 hypotheses supported
7 rejected
4 inconclusive

BEST MODEL
XGBoost + engineered behavioral features
F1: 0.871

ROBUSTNESS
5-seed mean F1: 0.866
Std: 0.009

TEMPORAL VALIDATION
F1: 0.801

CRITIC
CONDITIONAL APPROVAL

REASON
Strong random-split performance, but temporal degradation
requires investigation before production deployment.

RECOMMENDATION
Do not deploy yet.
Run temporal feature investigation and monitoring design.

REPRODUCIBILITY
Dataset version: v4
Experiment: exp_017
Seed: 42
Environment hash: ...
Code artifact: ...
Model artifact: ...
```

---

# 36. What makes this novel

Do not position novelty as:

> "We connected GPT to AutoML."

That is weak.

Position the contribution around:

### 1. Autonomous evidence-driven experimentation
The agent identifies weaknesses, forms hypotheses, runs experiments, evaluates evidence, and chooses the next experiment.

### 2. Domain-aware ML reasoning
Retrieved domain rules become explicit ML constraints.

### 3. Critic/falsification loop
The system actively attempts to invalidate promising results.

### 4. Experiment graph memory
The system remembers why experiments were performed and what was learned.

### 5. Evidence-based stopping
The agent balances improvement against compute/time and uncertainty.

### 6. End-to-end ML engineering
It extends beyond model selection into data quality, robustness, fairness, deployment readiness, and reproducibility.

Your claims must ultimately be supported by benchmark results.

---

# 37. What NOT to build first

Do not start with:
- multi-agent swarm
- fancy UI
- Kubernetes
- autonomous production deployment
- dozens of ML algorithms
- custom LLM training
- complex vector database
- reinforcement learning for the agent

First prove:

```text
Dataset
 ↓
Audit
 ↓
AutoML
 ↓
Hypothesis
 ↓
Experiment
 ↓
Critic
 ↓
Better decision
```

If that loop works reliably, the rest is engineering.

---

# 38. Definition of done for MVP

The MVP is complete when:

- [ ] User uploads a dataset
- [ ] User provides objective
- [ ] Agent profiles dataset
- [ ] Agent identifies/asks for target
- [ ] Leakage analysis runs
- [ ] AutoML baseline runs
- [ ] Experiment is recorded
- [ ] Agent identifies a weakness
- [ ] Agent generates a hypothesis
- [ ] Agent runs a targeted experiment
- [ ] Experiment is compared correctly
- [ ] Critic reviews result
- [ ] Agent can reject its own promising result
- [ ] Domain document can be uploaded
- [ ] Domain rule can affect a decision
- [ ] Test set remains isolated
- [ ] All experiments are reproducible
- [ ] Budget limits work
- [ ] Final report is generated
- [ ] Benchmark comparison with AutoML exists

---

# 39. Final project positioning

Use this description:

> **Agentic ML Engineer is an AI-powered machine-learning experimentation system that uses AutoML as a model-search component while addressing limitations of conventional AutoML through problem formulation, domain knowledge retrieval, data-quality and leakage analysis, autonomous hypothesis-driven experimentation, adversarial model criticism, robustness evaluation, and reproducible experiment memory.**

Short version:

> **AutoML finds models. Agentic ML Engineer investigates whether those models should be trusted.**

That is the central idea to keep throughout implementation.

---

# 40. Recommended build sequence

```text
PHASE 1
Project skeleton
       ↓
PHASE 2
Database + experiment schema
       ↓
PHASE 3
Dataset profiler
       ↓
PHASE 4
Leakage + split tools
       ↓
PHASE 5
AutoML wrapper
       ↓
PHASE 6
Experiment runner
       ↓
PHASE 7
Agent tool calling
       ↓
PHASE 8
Hypothesis loop
       ↓
PHASE 9
Critic
       ↓
PHASE 10
Domain RAG
       ↓
PHASE 11
Robustness/fairness
       ↓
PHASE 12
Frontend
       ↓
PHASE 13
Benchmark
       ↓
PHASE 14
Ablation study
       ↓
PHASE 15
Final report / paper / demo
```

## Golden rule

**Every time you add a feature, ask:**

> Does this make the agent better at making, testing, explaining, or validating ML decisions?

If not, postpone it.