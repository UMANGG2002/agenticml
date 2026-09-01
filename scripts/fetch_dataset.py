#!/usr/bin/env python3
"""Fetch a small, hand-picked registry of public ML datasets for local
testing and the evaluation benchmark (docs/spec.md section 32).

No Kaggle credentials required — datasets come from OpenML (via
scikit-learn's `fetch_openml`) or scikit-learn's bundled loaders, both of
which just need plain internet access.

Usage:
    uv run python scripts/fetch_dataset.py --list
    uv run python scripts/fetch_dataset.py adult_income
    uv run python scripts/fetch_dataset.py titanic --out dataset/titanic.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("evaluation/datasets")


@dataclass(frozen=True)
class DatasetSpec:
    description: str
    task_type: str
    target: str
    loader: Callable[[], pd.DataFrame]


def _load_openml(name: str, version: int | str = "active") -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name=name, version=version, as_frame=True, parser="pandas")
    return data.frame


def _load_california_housing() -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing

    return fetch_california_housing(as_frame=True).frame


REGISTRY: dict[str, DatasetSpec] = {
    "adult_income": DatasetSpec(
        description="UCI Adult / Census Income — ~48.8k rows, mixed numeric/categorical, real '?' missing values.",
        task_type="binary_classification",
        target="class",
        loader=lambda: _load_openml("adult", version=2),
    ),
    "titanic": DatasetSpec(
        description="Titanic survival — ~1.3k rows, missing age/cabin, leakage-prone name/ticket/cabin columns.",
        task_type="binary_classification",
        target="survived",
        loader=lambda: _load_openml("titanic", version=1),
    ),
    "california_housing": DatasetSpec(
        description="California housing prices — ~20.6k rows, all-numeric regression target.",
        task_type="regression",
        target="MedHouseVal",
        loader=_load_california_housing,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a public dataset for local testing / benchmarking.")
    parser.add_argument("name", nargs="?", choices=sorted(REGISTRY), help="dataset to fetch")
    parser.add_argument(
        "--out", type=Path, default=None, help="output CSV path (default: evaluation/datasets/<name>.csv)"
    )
    parser.add_argument("--list", action="store_true", help="list available datasets and exit")
    args = parser.parse_args()

    if args.list or args.name is None:
        for name, spec in sorted(REGISTRY.items()):
            print(f"{name:20s} [{spec.task_type:24s}] target={spec.target!r:20s} {spec.description}")
        return

    spec = REGISTRY[args.name]
    out_path = args.out or (DEFAULT_OUTPUT_DIR / f"{args.name}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching '{args.name}' ({spec.description}) ...")
    df = spec.loader()
    df.to_csv(out_path, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} cols -> {out_path}")
    print(f"task_type={spec.task_type} target={spec.target!r}")


if __name__ == "__main__":
    main()
