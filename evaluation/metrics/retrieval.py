"""Retrieval metrics for ranked binary or graded relevance results."""

from __future__ import annotations

import math
from statistics import mean, median, stdev
from typing import Any, Iterable


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(v) for v in values]
    if not values:
        return {"mean": None, "median": None, "std": None, "n": 0}
    return {
        "mean": mean(values),
        "median": median(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def query_metrics(retrieved: list[dict[str, Any]], relevant: set[str],
                  graded: dict[str, float] | None = None) -> dict[str, Any]:
    """Calculate metrics for one ranked query, without assuming unique results."""
    # FAISS indexes may contain several chunks for one function. Retrieval
    # metrics operate on ranked nodes, so retain the first occurrence only.
    unique = []
    seen = set()
    for item in retrieved:
        item_id = str(item.get("function_id", ""))
        if item_id and item_id not in seen:
            seen.add(item_id)
            unique.append(item)
    ids = [str(item.get("function_id", "")) for item in unique]
    binary = [1.0 if item.get("relevant") or item.get("function_id") in relevant else 0.0
              for item in unique]
    if graded:
        gains = [float(graded.get(item_id, 0.0)) for item_id in ids]
    else:
        gains = binary
    first = next((i + 1 for i, value in enumerate(binary) if value), None)
    result: dict[str, Any] = {
        "first_relevant_rank": first,
        "mrr": 1.0 / first if first else 0.0,
        "num_relevant": len(relevant),
    }
    for k in (1, 5, 10, 20):
        top = ids[:k]
        top_binary = binary[:k]
        hits = sum(top_binary)
        result[f"recall_at_{k}"] = hits / len(relevant) if relevant else None
        result[f"precision_at_{k}"] = hits / k if k else None
        result[f"hit_at_{k}"] = 1.0 if hits else 0.0
        result[f"ndcg_at_{k}"] = _ndcg(gains[:k], gains, k)
    return result


def _ndcg(gains: list[float], all_gains: list[float], k: int) -> float:
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains[:k]))
    ideal = sorted(all_gains, reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query rows with mean, median, and standard deviation."""
    metrics = {}
    names = [key for key in rows[0] if key.startswith(("recall_at_", "precision_at_",
                                                        "ndcg_at_", "hit_at_"))] if rows else []
    names += ["mrr"]
    for name in dict.fromkeys(names):
        values = [row[name] for row in rows if row.get(name) is not None]
        metrics[name] = _summary(values)
    return metrics
