"""Conservative paired-query statistics."""

from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Iterable


def paired_difference(a: Iterable[float], b: Iterable[float]) -> dict[str, float | int | None]:
    """Summarize paired differences; no p-value is claimed without scipy."""
    differences = [float(x) - float(y) for x, y in zip(a, b)]
    if not differences:
        return {"n": 0, "mean_difference": None, "std_difference": None}
    avg = mean(differences)
    variance = sum((value - avg) ** 2 for value in differences) / max(len(differences) - 1, 1)
    return {
        "n": len(differences),
        "mean_difference": avg,
        "std_difference": sqrt(variance),
    }
