"""In-memory experiment registry: the parent/child experiment graph
(docs/spec.md section 23) that becomes the agent's long-term experiment
memory.

Like agent/dataset_registry.py, this stands in for a future Postgres
`experiments` table (docs/spec.md section 24) — callers only ever address
experiments by `experiment_id`, so swapping the backing store later
requires no change at the call sites.
"""

from __future__ import annotations

from experiments.schemas import ExperimentRecord


class ExperimentRegistry:
    def __init__(self) -> None:
        self._store: dict[str, ExperimentRecord] = {}

    def record(self, experiment: ExperimentRecord) -> str:
        if experiment.experiment_id in self._store:
            raise ValueError(f"experiment_id '{experiment.experiment_id}' is already recorded")
        if experiment.parent_experiment_id is not None and experiment.parent_experiment_id not in self._store:
            raise KeyError(f"parent experiment '{experiment.parent_experiment_id}' not found")
        self._store[experiment.experiment_id] = experiment
        return experiment.experiment_id

    def get(self, experiment_id: str) -> ExperimentRecord:
        try:
            return self._store[experiment_id]
        except KeyError:
            raise KeyError(f"unknown experiment_id: '{experiment_id}'") from None

    def children(self, experiment_id: str) -> list[ExperimentRecord]:
        self.get(experiment_id)  # raise if experiment_id itself is unknown
        return [e for e in self._store.values() if e.parent_experiment_id == experiment_id]

    def lineage(self, experiment_id: str) -> list[ExperimentRecord]:
        """The chain from the root ancestor down to `experiment_id`, inclusive."""
        chain = [self.get(experiment_id)]
        while chain[-1].parent_experiment_id is not None:
            chain.append(self.get(chain[-1].parent_experiment_id))
        return list(reversed(chain))

    def all(self) -> list[ExperimentRecord]:
        return list(self._store.values())

    def best(self, metric: str) -> ExperimentRecord | None:
        """The best-scoring recorded experiment using `metric`, respecting its direction.

        Only considers experiments recorded against this exact metric name —
        comparing across different metrics isn't meaningful.
        """
        from tools.ml.automl import is_higher_better

        candidates = [e for e in self._store.values() if e.metric == metric]
        if not candidates:
            return None
        higher_better = is_higher_better(metric)
        return max(candidates, key=lambda e: e.primary_score if higher_better else -e.primary_score)
