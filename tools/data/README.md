# Data Detective

Deterministic, typed dataset tools. No LLM calls happen in this module —
everything here is plain pandas/numpy/sklearn so results are reproducible
and unit-testable. The agent calls these as tools and reasons over their
structured output.

| Function | File | Purpose |
|---|---|---|
| `inspect_dataset(df)` | [profile.py](profile.py) | Shape, dtypes, sample rows, duplicate row count |
| `profile_columns(df)` | [profile.py](profile.py) | Per-column stats, inferred type, warnings |
| `detect_target(df, profile=None, objective=None)` | [profile.py](profile.py) | Candidate target columns with confidence + reasons — never silently picks one |
| `check_data_quality(df, profile=None)` | [quality.py](quality.py) | Aggregated, actionable data-quality issues by severity |
| `detect_leakage(df, target, ...)` | [leakage.py](leakage.py) | Target-derived features, suspicious correlation, post-outcome features, duplicate/near-duplicate contamination, train/test contamination |
| `create_split(df, strategy, ...)` | [split.py](split.py) | random / stratified / group / time-based splitting |

## Design rules

1. **Read-only.** No function here mutates the input DataFrame or drops columns automatically.
2. **Evidence, not verdicts.** `detect_leakage` and `check_data_quality` return findings with
   evidence and severity — the agent (or critic) decides what to do about them.
3. **Never guess the target silently.** `detect_target` always returns
   `requires_user_confirmation=True` unless the top candidate is both confident and clearly
   separated from the runner-up.
4. **Typed I/O.** All return values are Pydantic models from [schemas.py](schemas.py), validated
   at the boundary — no free-form dicts.

## Example

```python
import pandas as pd
from tools.data.profile import inspect_dataset, profile_columns, detect_target
from tools.data.quality import check_data_quality
from tools.data.leakage import detect_leakage

df = pd.read_csv("customers.csv")

inspection = inspect_dataset(df)
profile = profile_columns(df)
quality = check_data_quality(df, profile=profile)
target_result = detect_target(df, profile=profile, objective="predict customer churn")

if target_result.requires_user_confirmation:
    ...  # ask the user which column is the target

leakage = detect_leakage(df, target=target_result.recommended.column)
if leakage.has_critical_issues:
    ...  # surface findings before running AutoML
```
