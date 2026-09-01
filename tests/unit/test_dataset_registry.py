import pandas as pd
import pytest

from agent.dataset_registry import DatasetRegistry


def test_register_and_get_roundtrip():
    registry = DatasetRegistry()
    df = pd.DataFrame({"a": [1, 2, 3]})
    version_id = registry.register(df)
    pd.testing.assert_frame_equal(registry.get(version_id), df)


def test_register_generates_unique_ids():
    registry = DatasetRegistry()
    df = pd.DataFrame({"a": [1]})
    id1 = registry.register(df)
    id2 = registry.register(df)
    assert id1 != id2


def test_register_with_explicit_id():
    registry = DatasetRegistry()
    df = pd.DataFrame({"a": [1]})
    version_id = registry.register(df, dataset_version_id="v1")
    assert version_id == "v1"


def test_register_rejects_duplicate_id():
    registry = DatasetRegistry()
    df = pd.DataFrame({"a": [1]})
    registry.register(df, dataset_version_id="v1")
    with pytest.raises(ValueError):
        registry.register(df, dataset_version_id="v1")


def test_get_unknown_id_raises_key_error():
    registry = DatasetRegistry()
    with pytest.raises(KeyError):
        registry.get("does_not_exist")
