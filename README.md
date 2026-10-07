# Agentic ML Engineer (AME)

An autonomous ML engineering agent that uses AutoML as one tool, but goes beyond model
search by understanding the ML objective, auditing data, retrieving domain knowledge,
designing experiments, challenging results, checking robustness/fairness/drift, and
producing reproducible evidence-backed recommendations.

> AutoML finds models. Agentic ML Engineer investigates whether those models should be trusted.

Full specification: [docs/spec.md](docs/spec.md).

## Repository layout

```text
apps/api/         FastAPI backend
apps/web/          Next.js frontend
agent/              Orchestrator, prompts, state
tools/data/         Dataset profiling, leakage detection, splitting
tools/ml/           AutoML wrapper, training, tuning, error analysis
tools/validation/   Robustness, fairness, calibration, drift
tools/research/     Literature/web search tool
domain/             Domain knowledge ingestion + retrieval (RAG)
experiments/        Experiment runner, registry, artifacts
db/                 SQLAlchemy models, session, migrations
evaluation/         Benchmark + agent evaluation harness
tests/              unit / integration / e2e
```

## Getting started (backend)

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest
```

## Getting started (frontend)

```bash
cd apps/web
npm install
npm run dev
```

## Build order

This project is built bottom-up, per [docs/spec.md](docs/spec.md#30-vibe-coding-rules):

```text
database → schemas → deterministic ML tools → experiment runner
  → agent tool wrappers → agent loop → API → UI
```

The first deterministic tools implemented are the **Data Detective** tools in
[tools/data/](tools/data/): `inspect_dataset`, `profile_columns`, `detect_target`,
and `detect_leakage`. See [tools/data/README.md](tools/data/README.md).
# agenticml
