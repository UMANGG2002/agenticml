"""Data Detective — train/validation/test split strategies.

The split strategy must depend on the problem, not default to random:
- `random` for i.i.d. tabular data
- `stratified` to preserve class balance in classification targets
- `group` to keep rows sharing a group key (e.g. customer_id) on one side
- `time` to prevent training on the future and testing on the past
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from tools.data.schemas import SplitResult, SplitStrategy


def create_split(
    df: pd.DataFrame,
    strategy: SplitStrategy,
    *,
    test_size: float = 0.2,
    validation_size: float | None = None,
    target: str | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
    seed: int = 42,
) -> SplitResult:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    if validation_size is not None and not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1 (exclusive)")
    if validation_size is not None and test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")

    n_rows = len(df)
    if n_rows < 2:
        raise ValueError("dataset must have at least 2 rows to split")

    warnings: list[str] = []
    index = df.index.to_numpy()

    if strategy == "random":
        train_idx, test_idx = _random_split(index, test_size, seed)
    elif strategy == "stratified":
        if target is None:
            raise ValueError("stratified split requires `target`")
        train_idx, test_idx, strat_warning = _stratified_split(df, index, target, test_size, seed)
        if strat_warning:
            warnings.append(strat_warning)
    elif strategy == "group":
        if group_column is None:
            raise ValueError("group split requires `group_column`")
        train_idx, test_idx = _group_split(df, index, group_column, test_size, seed)
    elif strategy == "time":
        if time_column is None:
            raise ValueError("time split requires `time_column`")
        train_idx, test_idx = _time_split(df, time_column, test_size)
    else:  # pragma: no cover - guarded by the SplitStrategy literal type
        raise ValueError(f"unknown split strategy: {strategy}")

    validation_idx: np.ndarray = np.array([], dtype=index.dtype)
    if validation_size is not None:
        remaining_frac = 1 - test_size
        val_frac_of_train = validation_size / remaining_frac
        train_df = df.loc[train_idx]

        if strategy == "random":
            train_idx, validation_idx = _random_split(train_idx, val_frac_of_train, seed)
        elif strategy == "stratified":
            train_idx, validation_idx, strat_warning = _stratified_split(
                train_df, train_idx, target, val_frac_of_train, seed
            )
            if strat_warning:
                warnings.append(strat_warning)
        elif strategy == "group":
            train_idx, validation_idx = _group_split(train_df, train_idx, group_column, val_frac_of_train, seed)
        elif strategy == "time":
            train_idx, validation_idx = _time_split(train_df, time_column, val_frac_of_train)

    if strategy == "group" and group_column is not None:
        _warn_if_group_overlap(df, group_column, train_idx, test_idx, validation_idx, warnings)

    return SplitResult(
        strategy=strategy,
        train_indices=[int(i) for i in train_idx],
        validation_indices=[int(i) for i in validation_idx],
        test_indices=[int(i) for i in test_idx],
        seed=seed,
        warnings=warnings,
    )


def _random_split(index: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = train_test_split(index, test_size=test_size, random_state=seed, shuffle=True)
    return train_idx, test_idx


def _stratified_split(
    df: pd.DataFrame, index: np.ndarray, target: str, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, str | None]:
    subset = df.loc[index]
    class_counts = subset[target].value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0

    # Stratification requires at least 2 members per class per side.
    if min_class_count < 2 or (min_class_count * test_size) < 1:
        train_idx, test_idx = _random_split(index, test_size, seed)
        return (
            train_idx,
            test_idx,
            (
                f"target class '{class_counts.idxmin() if not class_counts.empty else target}' has too few "
                f"members ({min_class_count}) to stratify — fell back to a random split."
            ),
        )

    train_idx, test_idx = train_test_split(
        index, test_size=test_size, random_state=seed, shuffle=True, stratify=subset[target]
    )
    return train_idx, test_idx, None


def _group_split(
    df: pd.DataFrame, index: np.ndarray, group_column: str, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    subset = df.loc[index]
    n_groups = subset[group_column].nunique()
    if n_groups < 2:
        raise ValueError(f"group column '{group_column}' must have at least 2 distinct groups to split on")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_pos, test_pos = next(splitter.split(subset, groups=subset[group_column]))
    return index[train_pos], index[test_pos]


def _time_split(df: pd.DataFrame, time_column: str, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    subset = df[[time_column]].copy()
    subset["__order__"] = np.arange(len(subset))
    ordered = subset.sort_values(time_column, kind="mergesort")
    n_test = max(1, int(round(len(ordered) * test_size)))
    test_idx = ordered.index.to_numpy()[-n_test:]
    train_idx = ordered.index.to_numpy()[:-n_test]
    return train_idx, test_idx


def _warn_if_group_overlap(
    df: pd.DataFrame,
    group_column: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    validation_idx: np.ndarray,
    warnings: list[str],
) -> None:
    train_groups = set(df.loc[train_idx, group_column])
    test_groups = set(df.loc[test_idx, group_column])
    overlap = train_groups & test_groups
    if overlap:
        warnings.append(f"{len(overlap)} group(s) unexpectedly appear in both train and test: {sorted(overlap)[:5]}")
    if len(validation_idx):
        val_groups = set(df.loc[validation_idx, group_column])
        val_overlap = (train_groups | test_groups) & val_groups
        if val_overlap:
            warnings.append(
                f"{len(val_overlap)} group(s) unexpectedly appear in both validation and another split: "
                f"{sorted(val_overlap)[:5]}"
            )
