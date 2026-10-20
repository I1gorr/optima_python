"""Retrieval metrics and paired-bootstrap uncertainty for the research runner.

Per-query Recall@k, Precision@k, Hit@k, nDCG@k and reciprocal rank are computed
by the project's shared ``evaluation.metrics.retrieval.query_metrics`` (whose
nDCG ideal-ranking bug was fixed for this work), so there is a single metric
implementation in the repository.

Conventions:

* Ranked lists are de-duplicated by function ID (first occurrence).
* ``Precision@k`` divides by ``k`` even when fewer than ``k`` results exist.
* ``MRR`` is computed over the returned top-k list (cutoff = the run's ``top_k``).
* A query without any relevant function (missing or empty judgment) is
  *excluded* from every aggregate and reported in ``skipped``; it is never
  counted as a zero.
* Graded relevance: ``relevance`` maps ID -> grade (> 0 is relevant).  nDCG uses
  the grades; recall/precision/hit/MRR use ``grade > 0``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from evaluation.metrics.retrieval import query_metrics

K_VALUES: Tuple[int, ...] = (1, 3, 5, 10)
METRIC_NAMES: Tuple[str, ...] = tuple(
    [f"recall_at_{k}" for k in K_VALUES] + [f"precision_at_{k}" for k in K_VALUES]
    + [f"hit_at_{k}" for k in K_VALUES] + [f"ndcg_at_{k}" for k in K_VALUES] + ["mrr"])
HEADLINE_METRICS: Tuple[str, ...] = ("mrr", "ndcg_at_10", "recall_at_10", "recall_at_5", "recall_at_1")


def relevance_map(query: Mapping[str, Any]) -> Dict[str, float]:
    """Judgments of a benchmark query as ``{function_id: grade}`` (grades > 0 only)."""
    graded = query.get("relevance")
    if isinstance(graded, Mapping) and graded:
        return {str(k): float(v) for k, v in graded.items() if v and float(v) > 0}
    ids = query.get("relevant_functions") or []
    return {str(i): 1.0 for i in ids}


def evaluate_query(ranked_ids: Sequence[str], relevance: Mapping[str, float]) -> Optional[Dict[str, Any]]:
    """Metrics for one query, or ``None`` when it has no relevant function."""
    relevant = {i for i, g in relevance.items() if g and g > 0}
    if not relevant:
        return None
    is_graded = len({float(g) for g in relevance.values()}) > 1
    row = query_metrics([{"function_id": i} for i in ranked_ids], relevant,
                        graded=dict(relevance) if is_graded else None)
    keep = {k: row[k] for k in METRIC_NAMES if k in row}
    keep["first_relevant_rank"] = row["first_relevant_rank"]
    keep["num_relevant"] = row["num_relevant"]
    return keep


def mean_metrics(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    rows = list(rows)
    out: Dict[str, Optional[float]] = {}
    for name in METRIC_NAMES:
        values = [r[name] for r in rows if r.get(name) is not None]
        out[name] = float(np.mean(values)) if values else None
    out["n_queries"] = len(rows)
    return out


def by_category(rows: Sequence[Mapping[str, Any]], categories: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row, category in zip(rows, categories):
        grouped[category or "uncategorised"].append(row)
    return {cat: mean_metrics(items) for cat, items in sorted(grouped.items())}


# ------------------------------------------------------------------ bootstrap
def _group_codes(groups: Optional[Sequence[Any]], n: int) -> np.ndarray:
    if groups is None:
        return np.arange(n)
    mapping: Dict[Any, int] = {}
    return np.array([mapping.setdefault(g, len(mapping)) for g in groups])


def bootstrap_mean_ci(values: Sequence[float], groups: Optional[Sequence[Any]] = None,
                      n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> Dict[str, Any]:
    """Percentile bootstrap CI for a mean.  With ``groups`` it is a cluster bootstrap
    (whole groups -- e.g. queries about the same target function -- are resampled)."""
    arr = np.asarray(list(values), dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    codes = _group_codes(groups, n)
    g = int(codes.max()) + 1
    sums = np.bincount(codes, weights=arr, minlength=g)
    counts = np.bincount(codes, minlength=g).astype(float)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot).astype(float)
    estimates = (weights @ sums) / (weights @ counts)
    low, high = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(arr.mean()), "ci_low": float(low), "ci_high": float(high), "n": n,
            "n_groups": g}


def paired_bootstrap(candidate: Sequence[float], reference: Sequence[float],
                     groups: Optional[Sequence[Any]] = None, n_boot: int = 10000,
                     seed: int = 0, alpha: float = 0.05) -> Dict[str, Any]:
    """Paired bootstrap of ``mean(candidate - reference)`` over queries.

    ``ci_excludes_zero`` is a statement about this benchmark's query sample only; it
    is not a claim of significance beyond it (queries are not an i.i.d. draw from
    all possible developer questions).  Pass ``groups`` to account for several
    queries sharing a target function.
    """
    a = np.asarray(list(candidate), dtype=float)
    b = np.asarray(list(reference), dtype=float)
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diff = a - b
    result = bootstrap_mean_ci(diff, groups, n_boot, seed, alpha)
    result.update({
        "mean_difference": result.pop("mean"),
        "wins": int((diff > 1e-12).sum()), "losses": int((diff < -1e-12).sum()),
        "ties": int((np.abs(diff) <= 1e-12).sum()),
    })
    low, high = result["ci_low"], result["ci_high"]
    result["ci_excludes_zero"] = bool(low is not None and (low > 0 or high < 0))
    return result


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    items = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(items)) if items else None
