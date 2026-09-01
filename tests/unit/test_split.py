import numpy as np
import pandas as pd
import pytest

from tools.data.split import create_split


def _all_indices(result):
    return set(result.train_indices) | set(result.validation_indices) | set(result.test_indices)


class TestRandomSplit:
    def test_partitions_all_rows_with_no_overlap(self):
        df = pd.DataFrame({"x": range(100)})
        result = create_split(df, "random", test_size=0.2, seed=42)
        assert set(result.train_indices).isdisjoint(result.test_indices)
        assert _all_indices(result) == set(df.index)

    def test_test_size_is_approximately_respected(self):
        df = pd.DataFrame({"x": range(1000)})
        result = create_split(df, "random", test_size=0.2, seed=42)
        assert 190 <= len(result.test_indices) <= 210

    def test_seed_is_reproducible(self):
        df = pd.DataFrame({"x": range(100)})
        r1 = create_split(df, "random", test_size=0.2, seed=7)
        r2 = create_split(df, "random", test_size=0.2, seed=7)
        assert r1.train_indices == r2.train_indices
        assert r1.test_indices == r2.test_indices

    def test_different_seeds_generally_differ(self):
        df = pd.DataFrame({"x": range(200)})
        r1 = create_split(df, "random", test_size=0.3, seed=1)
        r2 = create_split(df, "random", test_size=0.3, seed=2)
        assert r1.test_indices != r2.test_indices

    def test_with_validation_size_three_way_split(self):
        df = pd.DataFrame({"x": range(1000)})
        result = create_split(df, "random", test_size=0.2, validation_size=0.1, seed=42)
        all_idx = _all_indices(result)
        assert all_idx == set(df.index)
        assert set(result.train_indices).isdisjoint(result.validation_indices)
        assert set(result.train_indices).isdisjoint(result.test_indices)
        assert set(result.validation_indices).isdisjoint(result.test_indices)
        assert 80 <= len(result.validation_indices) <= 120

    def test_invalid_test_size_raises(self):
        df = pd.DataFrame({"x": range(10)})
        with pytest.raises(ValueError):
            create_split(df, "random", test_size=1.5)

    def test_test_plus_validation_too_large_raises(self):
        df = pd.DataFrame({"x": range(10)})
        with pytest.raises(ValueError):
            create_split(df, "random", test_size=0.6, validation_size=0.6)

    def test_too_few_rows_raises(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError):
            create_split(df, "random")


class TestStratifiedSplit:
    def test_requires_target(self):
        df = pd.DataFrame({"x": range(10)})
        with pytest.raises(ValueError):
            create_split(df, "stratified")

    def test_preserves_class_balance(self):
        df = pd.DataFrame({"x": range(200), "y": ([0] * 150) + ([1] * 50)})
        result = create_split(df, "stratified", test_size=0.25, target="y", seed=42)
        test_labels = df.loc[result.test_indices, "y"]
        train_labels = df.loc[result.train_indices, "y"]
        assert test_labels.mean() == pytest.approx(train_labels.mean(), abs=0.03)

    def test_falls_back_to_random_with_warning_when_class_too_rare(self):
        df = pd.DataFrame({"x": range(20), "y": [0] * 19 + [1]})
        result = create_split(df, "stratified", test_size=0.2, target="y", seed=42)
        assert result.warnings
        assert "fell back to a random split" in result.warnings[0]


class TestGroupSplit:
    def test_requires_group_column(self):
        df = pd.DataFrame({"x": range(10)})
        with pytest.raises(ValueError):
            create_split(df, "group")

    def test_no_group_appears_in_both_train_and_test(self):
        df = pd.DataFrame({"x": range(100), "customer_id": [i % 20 for i in range(100)]})
        result = create_split(df, "group", test_size=0.25, group_column="customer_id", seed=42)
        train_groups = set(df.loc[result.train_indices, "customer_id"])
        test_groups = set(df.loc[result.test_indices, "customer_id"])
        assert train_groups.isdisjoint(test_groups)
        assert result.warnings == []

    def test_raises_when_fewer_than_two_groups(self):
        df = pd.DataFrame({"x": range(10), "g": [1] * 10})
        with pytest.raises(ValueError):
            create_split(df, "group", group_column="g")


class TestTimeSplit:
    def test_requires_time_column(self):
        df = pd.DataFrame({"x": range(10)})
        with pytest.raises(ValueError):
            create_split(df, "time")

    def test_test_set_is_strictly_later_than_train_set(self):
        df = pd.DataFrame(
            {
                "x": range(100),
                "event_time": pd.date_range("2024-01-01", periods=100, freq="D"),
            }
        )
        result = create_split(df, "time", test_size=0.2, time_column="event_time", seed=42)
        max_train_time = df.loc[result.train_indices, "event_time"].max()
        min_test_time = df.loc[result.test_indices, "event_time"].min()
        assert max_train_time < min_test_time

    def test_respects_test_size(self):
        df = pd.DataFrame(
            {
                "x": range(100),
                "event_time": pd.date_range("2024-01-01", periods=100, freq="D"),
            }
        )
        result = create_split(df, "time", test_size=0.2, time_column="event_time", seed=42)
        assert len(result.test_indices) == 20


class TestSplitResultCompleteness:
    def test_train_test_and_validation_cover_dataset_exactly_once(self):
        df = pd.DataFrame({"x": range(500), "y": np.tile([0, 1], 250)})
        result = create_split(df, "stratified", test_size=0.2, validation_size=0.1, target="y", seed=42)
        combined = result.train_indices + result.validation_indices + result.test_indices
        assert sorted(combined) == sorted(df.index.tolist())
        assert len(combined) == len(set(combined))
