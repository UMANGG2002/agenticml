"""Download a public dataset, run the Data Detective audit on it, and run AutoML.

Quick end-to-end smoke test, same shape as scripts/run_pipeline.py but as a
plain script (no argparse) for poking at things interactively — see z.py
for the same idea against a local CSV.
"""

from pathlib import Path

import pandas as pd

from scripts.fetch_dataset import REGISTRY
from agent.dataset_registry import DatasetRegistry
from agent.orchestrator import run_data_audit
from agent.state import AgentState
from tools.data.split import create_split
from tools.ml.automl import run_automl

DATASET_NAME = "titanic"
OBJECTIVE = "predict passenger survival"
# Titanic's boat/body columns are only recorded after the outcome is known —
# see docs/architecture.md. Declaring them here lets detect_leakage catch it.
POST_OUTCOME_COLUMNS = ["boat", "body"]

csv_path = Path("evaluation/datasets") / f"{DATASET_NAME}.csv"
if csv_path.exists():
    print(f"using cached {csv_path}")
    df = pd.read_csv(csv_path)
else:
    spec = REGISTRY[DATASET_NAME]
    print(f"downloading '{DATASET_NAME}' ({spec.description}) ...")
    df = spec.loader()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"saved {df.shape[0]} rows x {df.shape[1]} cols -> {csv_path}")

print(f"\n=== inspect: {csv_path} ===")
print(df.shape, list(df.columns))

registry = DatasetRegistry()
dataset_version_id = registry.register(df)
state = AgentState(project_id="ztest", dataset_version_id=dataset_version_id, objective=OBJECTIVE)

audit = run_data_audit(registry, state, post_outcome_columns=POST_OUTCOME_COLUMNS)
print(f"\n=== audit: {audit.status} ===")
for issue in audit.quality.issues:
    print(f"  [{issue.severity}] {issue.type}: {issue.columns}")

if audit.status == "paused_for_target_confirmation":
    raise SystemExit(f"target ambiguous — set target explicitly: {audit.pending_question.context}")

if audit.status == "blocked_on_leakage":
    critical_cols = [c for f in audit.leakage.findings if f.severity == "critical" for c in f.columns]
    print(f"  critical leakage columns: {critical_cols} — excluding them and proceeding")
else:
    critical_cols = []

target = audit.state.target_column
task_type = audit.state.task_type

# Drop the leakage-critical columns, then let run_automl's profile-based
# selection also drop identifiers/text/constants (e.g. name, ticket) —
# see docs/architecture.md's module diagram: profile flows into AutoML.
modeling_df = df.drop(columns=critical_cols)
split = create_split(modeling_df, "stratified", test_size=0.2, target=target, seed=42)

result = run_automl(
    modeling_df,
    target=target,
    task_type=task_type,
    split=split,
    profile=audit.profile,
    cv_folds=5,
    seed=42,
    dataset_version_id=dataset_version_id,
    artifact_dir="data/artifacts",
)

print(f"\n=== automl: {result.experiment_id} ===")
print(f"target={target!r}  task_type={task_type}  metric={result.metric}")
print(f"features: {result.feature_columns}")
print(f"best_model={result.best_model}  params={result.best_params}")
print(f"cv_mean={result.cv_mean:.4f}  cv_std={result.cv_std:.4f}  test_score={result.test_score:.4f}")
print(f"artifact_path={result.artifact_path}")
