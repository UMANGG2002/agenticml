"""In-memory dataset_version_id -> DataFrame store.

Stands in for the PostgreSQL `dataset_versions` table + object storage
described in docs/spec.md section 24, until that layer is built. The
orchestrator and tool wrappers only ever address datasets by
`dataset_version_id`, never a raw DataFrame, so swapping this out for a
DB/object-storage-backed implementation later requires no change at their
call sites.
"""

from __future__ import annotations

import uuid

import pandas as pd


class DatasetRegistry:
    def __init__(self) -> None:
        self._store: dict[str, pd.DataFrame] = {}

    def register(self, df: pd.DataFrame, dataset_version_id: str | None = None) -> str:
        version_id = dataset_version_id or f"dsv_{uuid.uuid4().hex[:12]}"
        if version_id in self._store:
            raise ValueError(f"dataset_version_id '{version_id}' is already registered")
        self._store[version_id] = df
        return version_id

    def get(self, dataset_version_id: str) -> pd.DataFrame:
        try:
            return self._store[dataset_version_id]
        except KeyError:
            raise KeyError(f"unknown dataset_version_id: '{dataset_version_id}'") from None
