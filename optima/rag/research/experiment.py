"""Experiment runner and result storage.

``run_pipeline`` executes one :class:`PipelineConfig` against one benchmark and
returns a self-describing, JSON-serialisable record (config + hash, models,
dataset/index metadata, metrics, per-query rankings with provenance, timings,
resources, errors).  ``ExperimentStore`` writes each record under a unique
experiment ID and refuses to overwrite earlier results.

Result layout::

    <root>/<suite_id>/
        suite.json                      suite spec + environment + benchmark hashes
        runs/<experiment_id>/result.json
        runs/<experiment_id>/per_query.jsonl.gz
        aggregate.csv per_category.csv comparisons.csv ...   (written by reporting.py)
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import platform
import statistics
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .benchmark import Benchmark, validate_benchmark
from .config import PipelineConfig
from .corpus import Corpus, load_corpus
from .dense import EmbeddingCache, EmbeddingUnavailable
from .metrics import (HEADLINE_METRICS, METRIC_NAMES, bootstrap_mean_ci, by_category,
                      evaluate_query, mean_metrics, relevance_map)
from .pipeline import QueryResult, ResearchPipeline
from .rerank import RerankerUnavailable

logger = logging.getLogger(__name__)

RESULT_SCHEMA_VERSION = 1


# ------------------------------------------------------------------ resources
class ResourceMonitor:
    """Samples process RSS in a background thread; records GPU peak if CUDA is used."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_rss_mb = 0.0
        self.start_rss_mb = 0.0
        self.gpu_peak_mb: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = None

    def _rss_mb(self) -> float:
        return self._proc.memory_info().rss / (1024 * 1024) if self._proc else 0.0

    def __enter__(self) -> "ResourceMonitor":
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
        except ImportError:
            self._proc = None
        self.start_rss_mb = self.peak_rss_mb = self._rss_mb()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # torch optional
            pass
        if self._proc is not None:
            def sample() -> None:
                while not self._stop.wait(self.interval):
                    self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())
            self._thread = threading.Thread(target=sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self.peak_rss_mb = max(self.peak_rss_mb, self._rss_mb())
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            pass

    def as_dict(self) -> Dict[str, Any]:
        return {"rss_start_mb": round(self.start_rss_mb, 1), "rss_peak_mb": round(self.peak_rss_mb, 1),
                "gpu_peak_allocated_mb": None if self.gpu_peak_mb is None else round(self.gpu_peak_mb, 1),
                "note": "RSS is process-wide (includes cached corpora/models from earlier runs)"}


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                            "cpu_count": os.cpu_count()}
    for module in ("numpy", "torch", "sentence_transformers", "transformers", "faiss"):
        try:
            info[module] = __import__(module).__version__
        except Exception:
            info[module] = None
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import subprocess
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[3]).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info


# --------------------------------------------------------------------- stats
def _latency_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {"mean": round(float(arr.mean()), 4), "median": round(float(np.median(arr)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4), "max": round(float(arr.max()), 4),
            "n": int(arr.size)}


def _serialise_result(query: Dict[str, Any], qr: QueryResult, metrics: Optional[Dict[str, Any]],
                      relevance: Dict[str, float]) -> Dict[str, Any]:
    return {
        "query_id": query["query_id"], "category": query.get("category", ""),
        "seed": query.get("seed"), "query": query["query"], "relevance": relevance,
        "metrics": metrics,
        "ranking": [asdict(item) for item in qr.results],
        "timings_ms": qr.timings_ms, "counts": qr.counts,
        "rerank_applied": qr.rerank_applied, "rerank_error": qr.rerank_error, "error": qr.error,
    }


# ------------------------------------------------------------------ one run
def run_pipeline(config: PipelineConfig, benchmark: Benchmark, corpus: Optional[Corpus] = None,
                 encoder=None, reranker=None, embedding_cache: Optional[EmbeddingCache] = None,
                 suite_id: str = "adhoc", label: str = "", group: str = "",
                 bootstrap_samples: int = 2000, progress: bool = False) -> Dict[str, Any]:
    """Run one configuration over a benchmark.  Raises only for unusable setups
    (bad data, embedding model unavailable, reranker unavailable); per-query
    failures are recorded in the result instead."""
    config = config.validate()
    started = datetime.now(timezone.utc)
    with ResourceMonitor() as monitor:
        pipeline = ResearchPipeline(config, corpus=corpus, encoder=encoder, reranker=reranker,
                                    embedding_cache=embedding_cache)
        t0 = time.perf_counter()
        build_info = pipeline.build()
        build_seconds = time.perf_counter() - t0
        corpus = pipeline.corpus
        validation = validate_benchmark(benchmark, corpus)
        invalid_ids = validation["invalid_ids"]

        rows: List[Optional[Dict[str, Any]]] = []
        per_query: List[Dict[str, Any]] = []
        skipped: List[Dict[str, str]] = []
        errors: List[Dict[str, str]] = []
        stage_ms: Dict[str, List[float]] = {}
        pool_before: List[float] = []
        pool_after: List[float] = []
        rerank_failures = 0
        evaluated_queries: List[Dict[str, Any]] = []
        for index, query in enumerate(benchmark.queries, 1):
            relevance = relevance_map(query)
            known = {i: g for i, g in relevance.items() if i in corpus.records}
            if not relevance:
                skipped.append({"query_id": query["query_id"], "reason": "no_relevant_judgments"})
                continue
            if not known:
                skipped.append({"query_id": query["query_id"],
                                "reason": "relevant_ids_not_in_corpus"})
                continue
            result = pipeline.search(query["query"])
            if result.error:
                errors.append({"query_id": query["query_id"], "error": result.error})
            if result.rerank_error:
                rerank_failures += 1
                errors.append({"query_id": query["query_id"],
                               "error": f"rerank: {result.rerank_error}"})
            ranked = [item.function_id for item in result.results]
            # Judgments naming IDs absent from the corpus stay in the denominator:
            # they are unretrievable and honestly count as misses.
            metrics = evaluate_query(ranked, relevance)
            rows.append(metrics)
            evaluated_queries.append(query)
            for stage, value in result.timings_ms.items():
                stage_ms.setdefault(stage, []).append(value)
            if result.pool_before_graph:
                rel = {i for i, g in relevance.items() if g > 0}
                pool_before.append(len(rel & set(result.pool_before_graph)) / len(rel))
                pool_after.append(len(rel & set(result.pool_after_graph)) / len(rel))
            per_query.append(_serialise_result(query, result, metrics, relevance))
            if progress and index % 50 == 0:
                logger.info("%s: %d/%d queries", label or config.name, index, len(benchmark.queries))

        valid_rows = [r for r in rows if r is not None]
        categories = [q.get("category", "") for q in evaluated_queries]
        groups = [q.get("seed") or q["query_id"] for q in evaluated_queries]
        overall = mean_metrics(valid_rows)
        confidence = {
            name: bootstrap_mean_ci([r[name] for r in valid_rows], groups, bootstrap_samples)
            for name in HEADLINE_METRICS if valid_rows}
    finished = datetime.now(timezone.utc)

    enrichment = corpus.meta.get("enrichment", {})
    run_hash = config.config_hash()
    experiment_id = f"exp-{started.strftime('%Y%m%dT%H%M%S')}-{run_hash}-{os.urandom(2).hex()}"
    status = "ok" if not errors else "partial"
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "suite_id": suite_id, "group": group, "label": label or config.name,
        "timestamp": started.isoformat(timespec="seconds"),
        "duration_seconds": round((finished - started).total_seconds(), 2),
        "status": status,
        "pipeline": config.name,
        "config": config.to_dict(), "config_hash": run_hash,
        "embedding_model": build_info.get("encoder"),
        "enrichment_model": config.enrichment_model,
        "enrichment_run": corpus.meta.get("enrichment_run"),
        "reranker": build_info.get("reranker"),
        "query_set_version": benchmark.query_set_version,
        "ground_truth_version": benchmark.ground_truth_version,
        "benchmark_id": benchmark.benchmark_id,
        "dataset": {
            "project": corpus.meta.get("project"), "functions": len(corpus),
            "base_json": corpus.meta.get("base_json"), "base_sha256": corpus.meta.get("base_sha256"),
            "enriched_json": corpus.meta.get("enriched_json"),
            "enriched_sha256": corpus.meta.get("enriched_sha256"),
            "duplicate_ids": corpus.meta.get("duplicate_ids"),
            "enriched_only_ids": corpus.meta.get("enriched_only_ids"),
            "base_sha256_mismatch": corpus.meta.get("base_sha256_mismatch", False),
            "benchmark_validation": {k: v for k, v in validation.items() if k != "invalid_ids"},
            "invalid_relevant_ids": {k: v for k, v in list(invalid_ids.items())[:20]},
        },
        "enrichment_coverage": {
            "status_counts": enrichment.get("status_counts"),
            "semantic_coverage": enrichment.get("semantic_coverage"),
            "failed_or_incomplete": enrichment.get("failed_or_incomplete"),
        } if config.enriched_json else None,
        "enrichment_cost": enrichment.get("usage_totals") if config.enriched_json else None,
        "index": {k: build_info[k] for k in ("indexes", "index_bytes_total", "documents", "graph")},
        "metrics": {
            "k_values": [1, 3, 5, 10], "top_k": config.top_k,
            "n_queries_total": len(benchmark.queries), "n_queries_evaluated": len(valid_rows),
            "skipped": skipped,
            "overall": overall, "confidence_95": confidence,
            "by_category": by_category(valid_rows, categories) if valid_rows else {},
            "candidate_pool": {
                "recall_before_graph": float(np.mean(pool_before)) if pool_before else None,
                "recall_after_graph": float(np.mean(pool_after)) if pool_after else None,
                "mean_candidates_before_graph": _mean_count(per_query, "first_stage_candidates"),
                "mean_candidates_after_graph": _mean_count(per_query, "candidates_after_graph"),
                "mean_graph_added": _mean_count(per_query, "graph_added"),
            },
        },
        "timing": {
            "build_seconds": round(build_seconds, 3),
            "build_breakdown_s": {k: round(v, 4) for k, v in build_info["timings"].items()},
            "embedding_cache_hits": {k: v.get("cache_hit") for k, v in build_info["indexes"].items()
                                     if k.startswith("dense:")},
            "query_ms_by_stage": {stage: _latency_stats(v) for stage, v in stage_ms.items()},
        },
        "resources": monitor.as_dict(),
        "rerank": {"enabled": config.rerank.enabled,
                   "applied_queries": sum(1 for q in per_query if q["rerank_applied"]),
                   "failed_queries": rerank_failures} if config.rerank.enabled else None,
        "errors": errors,
    }
    result["_per_query"] = per_query  # stripped by the store; kept in memory for comparisons
    return result


def _mean_count(per_query: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [q["counts"][key] for q in per_query if key in q.get("counts", {})]
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------- store
class ExperimentStore:
    """Writes runs under ``<root>/<suite_id>/runs/<experiment_id>/`` without overwriting."""

    def __init__(self, root: Path, suite_id: str):
        self.suite_dir = Path(root) / suite_id
        self.runs_dir = self.suite_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save_suite_info(self, info: Dict[str, Any]) -> Path:
        path = self.suite_dir / "suite.json"
        if path.exists():  # never overwrite: a re-run of the same suite id gets a new file
            path = self.suite_dir / f"suite-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
        path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        return path

    def save_run(self, result: Dict[str, Any]) -> Path:
        per_query = result.pop("_per_query", [])
        run_dir = self.runs_dir / result["experiment_id"]
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing experiment {run_dir}")
        run_dir.mkdir(parents=True)
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str),
                                             encoding="utf-8")
        with gzip.open(run_dir / "per_query.jsonl.gz", "wt", encoding="utf-8") as handle:
            for row in per_query:
                handle.write(json.dumps(row, default=str) + "\n")
        result["_per_query"] = per_query
        return run_dir

    def load_runs(self) -> List[Dict[str, Any]]:
        runs = []
        for path in sorted(self.runs_dir.glob("*/result.json")):
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        return runs

    @staticmethod
    def load_per_query(run_dir: Path) -> List[Dict[str, Any]]:
        with gzip.open(Path(run_dir) / "per_query.jsonl.gz", "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def failed_run_record(config: PipelineConfig, label: str, group: str, suite_id: str,
                      error: BaseException) -> Dict[str, Any]:
    """A result stub for a run that could not execute (model unavailable, bad data...)."""
    kind = ("embedding_unavailable" if isinstance(error, EmbeddingUnavailable)
            else "reranker_unavailable" if isinstance(error, RerankerUnavailable)
            else "error")
    now = datetime.now(timezone.utc)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": f"exp-{now.strftime('%Y%m%dT%H%M%S')}-{config.config_hash()}-{os.urandom(2).hex()}",
        "suite_id": suite_id, "group": group, "label": label, "pipeline": config.name,
        "timestamp": now.isoformat(timespec="seconds"), "status": "failed",
        "failure_kind": kind, "error": f"{type(error).__name__}: {error}",
        "config": config.to_dict(), "config_hash": config.config_hash(),
        "enrichment_model": config.enrichment_model, "metrics": None,
    }
