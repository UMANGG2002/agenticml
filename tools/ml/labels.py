"""Shared convention for picking the positive class of a binary target.

There's no explicit "positive class" metadata to go on, so by convention the
alphabetically/numerically last label (e.g. 1 over 0, "Yes" over "No",
">50K" over "<=50K") is treated as positive. AutoML's scoring and error
analysis's threshold curve/false-positive-negative examples both rely on
this and must agree, or their numbers won't line up.
"""

from __future__ import annotations

from typing import Any, Iterable


def positive_label(values: Iterable[Any]) -> Any:
    """Return the label treated as positive for a binary target."""
    return sorted(set(values), key=str)[-1]
