import numpy as np
import pandas as pd
import pytest

from tools.data.profile import detect_target, inspect_dataset, profile_columns


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(100)],
            "age": [20 + (i % 50) for i in range(100)],
            "country": ["US", "UK", "DE", "FR"] * 25,
            "signup_date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "is_active": [True, False] * 50,
            "notes": [None] * 100,
            "plan": ["basic"] * 100,
            "churned": [0, 1] * 50,
        }
    )


class TestInspectDataset:
    def test_shape_and_columns(self, mixed_df):
        result = inspect_dataset(mixed_df)
        assert result.n_rows == 100
        assert result.n_columns == 8
        assert "customer_id" in result.columns

    def test_dtypes_are_stringified(self, mixed_df):
        result = inspect_dataset(mixed_df)
        assert isinstance(result.dtypes["age"], str)

    def test_sample_is_json_safe(self, mixed_df):
        result = inspect_dataset(mixed_df, sample_size=3)
        assert len(result.sample) == 3
        # None values (NaN) must serialize as None, not NaN floats.
        assert result.sample[0]["notes"] is None

    def test_duplicate_row_count(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        result = inspect_dataset(df)
        assert result.duplicate_row_count == 1

    def test_does_not_mutate_input(self, mixed_df):
        original = mixed_df.copy(deep=True)
        inspect_dataset(mixed_df)
        pd.testing.assert_frame_equal(mixed_df, original)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = inspect_dataset(df)
        assert result.n_rows == 0
        assert result.n_columns == 0
        assert result.sample == []


class TestProfileColumns:
    def test_numeric_column_stats(self, mixed_df):
        profile = profile_columns(mixed_df)
        age = profile.get("age")
        assert age.inferred_type == "numeric"
        assert age.numeric_stats is not None
        assert age.numeric_stats.min == 20
        assert age.numeric_stats.max == 69

    def test_categorical_column_stats(self, mixed_df):
        profile = profile_columns(mixed_df)
        country = profile.get("country")
        assert country.inferred_type == "categorical"
        assert country.categorical_stats is not None
        assert country.categorical_stats.n_categories == 4

    def test_boolean_column(self, mixed_df):
        profile = profile_columns(mixed_df)
        is_active = profile.get("is_active")
        assert is_active.inferred_type == "boolean"

    def test_datetime_column(self, mixed_df):
        profile = profile_columns(mixed_df)
        signup = profile.get("signup_date")
        assert signup.inferred_type == "datetime"

    def test_fully_missing_column(self, mixed_df):
        profile = profile_columns(mixed_df)
        notes = profile.get("notes")
        assert notes.inferred_type == "empty"
        assert notes.missing_percentage == 100
        assert "entirely missing" in " ".join(notes.warnings)

    def test_constant_column(self, mixed_df):
        profile = profile_columns(mixed_df)
        plan = profile.get("plan")
        assert plan.inferred_type == "constant"
        assert any("single distinct value" in w for w in plan.warnings)

    def test_identifier_column(self, mixed_df):
        profile = profile_columns(mixed_df)
        cust_id = profile.get("customer_id")
        assert cust_id.inferred_type == "identifier"
        assert any("identifier" in w for w in cust_id.warnings)

    def test_missing_percentage_computed(self):
        df = pd.DataFrame({"x": [1, 2, None, None]})
        profile = profile_columns(df)
        x = profile.get("x")
        assert x.missing_count == 2
        assert x.missing_percentage == 50.0

    def test_placeholder_missing_values_detected(self):
        df = pd.DataFrame({"x": ["a", "b", "N/A", "unknown", "c"]})
        profile = profile_columns(df)
        x = profile.get("x")
        assert "N/A" in x.suspicious_values
        assert "unknown" in x.suspicious_values
        assert any("placeholder" in w for w in x.warnings)

    def test_infinite_values_flagged(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.inf, -np.inf, 3.0]})
        profile = profile_columns(df)
        x = profile.get("x")
        assert x.numeric_stats.infinite_count == 2
        assert any("infinite" in w for w in x.warnings)

    def test_high_cardinality_categorical_flagged(self):
        # Partially-repeating values so the column stays "categorical"
        # rather than being classified as an "identifier".
        df = pd.DataFrame({"x": [f"cat_{i % 80}" for i in range(200)]})
        profile = profile_columns(df)
        x = profile.get("x")
        assert x.inferred_type == "categorical"
        assert x.categorical_stats.is_high_cardinality

    def test_empty_dataframe(self):
        profile = profile_columns(pd.DataFrame())
        assert profile.n_rows == 0
        assert profile.columns == []

    def test_all_rows_missing_in_column_of_empty_frame(self):
        df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
        profile = profile_columns(df)
        a = profile.get("a")
        assert a.inferred_type == "empty"


class TestDetectTarget:
    def test_prefers_explicit_target_like_name(self):
        df = pd.DataFrame(
            {
                "feature_a": range(50),
                "feature_b": list(range(25)) * 2,
                "churned": [0, 1] * 25,
            }
        )
        result = detect_target(df)
        assert result.recommended is not None
        assert result.recommended.column == "churned"

    def test_binary_numeric_target_inferred_as_classification(self):
        df = pd.DataFrame({"x": range(20), "target": [0, 1] * 10})
        result = detect_target(df)
        assert result.recommended.column == "target"
        assert result.recommended.inferred_task_type == "binary_classification"

    def test_excludes_identifier_columns(self):
        df = pd.DataFrame(
            {
                "user_id": [f"U{i}" for i in range(50)],
                "score": np.random.default_rng(0).normal(size=50),
            }
        )
        result = detect_target(df)
        columns = [c.column for c in result.candidates]
        assert "user_id" not in columns

    def test_low_confidence_requires_confirmation(self):
        df = pd.DataFrame(
            {
                "a": range(30),
                "b": range(30, 60),
                "c": list(range(15)) * 2,
            }
        )
        result = detect_target(df)
        assert result.requires_user_confirmation is True

    def test_no_candidates_when_all_columns_unsuitable(self):
        df = pd.DataFrame({"id": [f"ID{i}" for i in range(30)], "constant": [1] * 30})
        result = detect_target(df)
        assert result.candidates == []
        assert result.recommended is None
        assert result.requires_user_confirmation is True

    def test_objective_text_boosts_matching_column(self):
        df = pd.DataFrame(
            {
                "feature_a": range(40),
                "revenue_next_month": np.random.default_rng(1).normal(size=40),
            }
        )
        without_objective = detect_target(df)
        with_objective = detect_target(df, objective="predict revenue next month")
        rev_without = next(c for c in without_objective.candidates if c.column == "revenue_next_month")
        rev_with = next(c for c in with_objective.candidates if c.column == "revenue_next_month")
        assert rev_with.confidence > rev_without.confidence

    def test_reuses_precomputed_profile(self, monkeypatch):
        df = pd.DataFrame({"x": range(10), "target": [0, 1] * 5})
        profile = profile_columns(df)

        called = {"count": 0}

        import tools.data.profile as profile_module

        original = profile_module.profile_columns

        def spy(*args, **kwargs):
            called["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(profile_module, "profile_columns", spy)
        detect_target(df, profile=profile)
        assert called["count"] == 0
