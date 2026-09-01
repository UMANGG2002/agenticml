# AME pipeline: how it runs today

This documents the audit-to-baseline pipeline as actually built and tested — not the full
spec (see [spec.md](spec.md)), just what exists right now: `tools/data/*`, `tools/ml/automl.py`,
`tools/ml/errors.py`, `experiments/*`, and `agent/*`. 151 passing tests back the claims below.

## Quickstart

Run it on any CSV. Most first passes pause — that's intentional, not a bug.

```bash
uv run python scripts/run_pipeline.py dataset/train.csv \
  --objective "predict mobile phone price range"
```

**Target gate.** If the target was ambiguous, the run exits with a ranked candidate list and a
reason for each score. Confirm it and re-run:

```bash
uv run python scripts/run_pipeline.py dataset/train.csv \
  --objective "predict mobile phone price range" \
  --target price_range
```

**Leakage gate.** If a critical leakage finding blocks the run, tell it which columns are
legitimately post-outcome (they get excluded and flagged), or force past it after reviewing:

```bash
uv run python scripts/run_pipeline.py dataset/titanic.csv \
  --objective "predict survival" --target survived \
  --post-outcome-columns boat,body
```

A clean run prints the winning model, its params, cross-validated and held-out test scores, and
where the fitted pipeline was saved (`data/artifacts/<experiment_id>/model.joblib`).

## Overall flow

Everything up to `run_automl` is built and tested today. Everything after the dashed arrow —
error analysis, hypotheses, the experiment loop, the critic — is the next phase, per
[spec.md](spec.md). The two diamonds are human-in-the-loop gates: neither the target column nor
a leakage-critical dataset gets silently decided, both pause the run and hand control back to
the caller.

```mermaid
flowchart LR
    A["Dataset +<br/>Objective"] --> B["Data Detective<br/>profile · quality · detect_target"]
    B --> C{"Target<br/>confirmed?"}
    C -- "ambiguous: ask user, re-run" --> C
    C -- confirmed --> D["detect_leakage<br/>(evidence, not a verdict)"]
    D --> E{"Critical<br/>leakage?"}
    E -- "blocked: resolve or exclude cols" --> E
    E -- clean --> F["create_split<br/>random · stratified · group · time"]
    F --> G[["run_automl<br/>(baseline)"]]
    G -.->|"test_score recorded,<br/>never used to pick the model"| H

    subgraph planned [planned - next phase]
        direction LR
        H["Error<br/>analysis"] --> I["Hypothesis<br/>engine"]
        I --> J["Experiment<br/>loop"]
        J -- iterate --> J
        J --> K{"Critic<br/>approves?"}
        K --> L["Final report<br/>+ artifacts"]
    end

    classDef built fill:#fff,stroke:#1B1F1D,stroke-width:1.5px,color:#1B1F1D;
    classDef gate fill:#fff,stroke:#B5701F,stroke-width:1.8px,color:#B5701F;
    classDef autom fill:#2F6F5E,stroke:#2F6F5E,color:#fff;
    classDef plan fill:none,stroke:#9AA098,stroke-width:1.3px,stroke-dasharray:4 3,color:#7A8078;

    class A,B,D,F built
    class C,E gate
    class G autom
    class H,I,J,K,L plan
    style planned fill:none,stroke:#9AA098,stroke-width:1px,stroke-dasharray:4 3
```

## Module architecture

The orchestrator calls the Data Detective and AutoML tools directly as plain Python functions
today. `agent/schemas.py` already defines the JSON-schema contract an LLM would use to call the
same tools — it *mirrors* them, but nothing in the live call path goes through it yet.
`agent/dataset_registry.py` is the one module built purely as a placeholder: swapping it for the
Postgres `dataset_versions` table later changes no call site, only this one file.

```mermaid
flowchart TD
    CLI["scripts/run_pipeline.py"] --> ORC
    API["apps/api<br/>(planned)"]:::plan --> ORC

    ORC[["agent/orchestrator.py<br/>run_data_audit()"]]:::autom

    ORC --> STATE["agent/state.py<br/>AgentState - run state,<br/>not conversation history"]:::built
    ORC --> REG["agent/dataset_registry.py<br/>dataset_version_id to DataFrame<br/>(stands in for future DB)"]:::built
    ORC --> POL["agent/policies.py<br/>gating predicates"]:::built
    ORC --> DATA["tools/data/*<br/>profile · quality · leakage · split"]:::built

    ORC -->|"if status is ready_for_automl"| AUTOML
    AUTOML[["tools/ml/automl.py<br/>run_automl()"]]:::autom

    DATA -->|DatasetProfile| AUTOML
    DATA -->|SplitResult| AUTOML

    AUTOML --> ART["artifact store<br/>model.joblib + AutoMLResult"]:::built

    SCHEMAS["agent/schemas.py<br/>TOOL_REGISTRY<br/>(JSON-schema tool contract)"]:::plan
    SCHEMAS -. "mirrors - not called yet" .-> DATA

    classDef built fill:#fff,stroke:#1B1F1D,stroke-width:1.5px,color:#1B1F1D;
    classDef autom fill:#2F6F5E,stroke:#2F6F5E,color:#fff;
    classDef plan fill:none,stroke:#9AA098,stroke-width:1.3px,stroke-dasharray:4 3,color:#7A8078;
```

## Status

| Piece | State |
|---|---|
| `tools/data/*` — inspect, profile, quality, target detection, leakage, split | built & tested |
| `agent/*` — state, dataset registry, policies, orchestrator, tool-schema contract | built & tested |
| `tools/ml/automl.py` — baseline model search with structural test-set isolation | built & tested |
| `experiments/*` — experiment registry, parent/child graph, hypothesis-vs-parent conclusions | built & tested |
| `tools/ml/errors.py` — confusion matrix, per-class metrics, thresholds, subgroup performance, residuals | built & tested |
| `scripts/fetch_dataset.py`, `scripts/run_pipeline.py`, `ztest.py` | built |
| Robustness/fairness checks, rule-based hypothesis engine, deterministic critic | planned (spec §11, §17–18) |
| DB persistence (replacing the in-memory dataset/experiment registries) | planned (spec §24) |
| `agent/schemas.py` TOOL_REGISTRY wired into an actual LLM call loop | planned |
| Domain knowledge — plain `.md` files only, no RAG/vector store for now | scope decision |
