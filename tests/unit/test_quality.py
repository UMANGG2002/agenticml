import pandas as pd

from tools.data.quality import check_data_quality


def test_clean_dataset_has_no_issues():
    df = pd.DataFrame(
        {
            "age": range(50),
            "country": ["US", "UK"] * 25,
            "churned": [0, 1] * 25,
        }
    )
    report = check_data_quality(df)
    assert report.issues == []
    assert report.n_critical == report.n_high == report.n_medium == report.n_low == 0


def test_empty_column_flagged_high():
    df = pd.DataFrame({"a": range(10), "b": [None] * 10})
    report = check_data_quality(df)
    issue = next(i for i in report.issues if i.type == "empty_columns")
    assert issue.severity == "high"
    assert issue.columns == ["b"]


def test_constant_column_flagged():
    df = pd.DataFrame({"a": range(10), "b": [1] * 10})
    report = check_data_quality(df)
    issue = next(i for i in report.issues if i.type == "constant_columns")
    assert "b" in issue.columns


def test_high_missing_column_flagged():
    df = pd.DataFrame({"a": range(10), "b": [1, 2, 3] + [None] * 7})
    report = check_data_quality(df)
    issue = next(i for i in report.issues if i.type == "high_missing_columns")
    assert "b" in issue.columns


def test_duplicate_rows_flagged():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [1, 1, 2, 3]})
    report = check_data_quality(df)
    issue = next(i for i in report.issues if i.type == "duplicate_rows")
    assert "1 duplicate" in issue.message


def test_duplicate_rows_severity_scales_with_percentage():
    # 6 out of 100 rows are duplicates (6%) -> "high"
    rows = [{"a": i, "b": i} for i in range(94)] + [{"a": 0, "b": 0}] * 6
    df = pd.DataFrame(rows)
    report = check_data_quality(df)
    issue = next(i for i in report.issues if i.type == "duplicate_rows")
    assert issue.severity == "high"


def test_placeholder_missing_values_flagged():
    df = pd.DataFrame({"a": ["x", "y", "NA", "z"], "b": range(4)})
    report = check_data_quality(df)
    issue = next(i for i in report.issues if i.type == "placeholder_missing_values")
    assert "a" in issue.columns


def test_severity_counts_are_consistent():
    df = pd.DataFrame({"a": [None] * 10, "b": [1] * 10, "c": range(10)})
    report = check_data_quality(df)
    total = report.n_critical + report.n_high + report.n_medium + report.n_low
    assert total == len(report.issues)


def test_reuses_precomputed_profile(monkeypatch):
    import tools.data.quality as quality_module

    df = pd.DataFrame({"a": range(10), "b": [1] * 10})
    profile = quality_module.profile_columns(df)

    called = {"count": 0}
    original = quality_module.profile_columns

    def spy(*args, **kwargs):
        called["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(quality_module, "profile_columns", spy)
    check_data_quality(df, profile=profile)
    assert called["count"] == 0
