"""Suite runner: executes an ablation plan over datasets with shared caches.

A suite spec (JSON)::

    {
      "suite_id": "chess-main",
      "output_root": "experiments",
      "embedding_cache": "experiments/_embedding_cache",
      "bootstrap_samples": 10000,
      "groups": ["pipelines", "enrichment", "representations", "hybrid", "graph", "rerank", "embedding"],
      "embedding_models": ["bge-small", "bge-base"],
      "primary_embedding": "bge-base",
      "rerankers": ["ms-marco-minilm"],
      "embedding_prefixes": {"nomic": {"query": "search_query: ", "document": "search_document: "}},
      "datasets": [{
        "name": "chess",
        "base_json": "optima_outputs/base/chess-engine-in-cpp-master/base.json",
        "benchmark": "benchmarks/chess/structural_v1.json",
        "enriched_dir": "output/chess Engine",          // or "enriched": {"slug": "path"}
        "primary_enrichment": "qwen-qwen2.5-coder-14b-instruct-chess",
        "enrichment_models": null                         // null = every file found
      }]
    }

Identical configurations that appear in several groups are executed once and the
result is re-recorded under each label (flagged ``reused_from``).  A run that
cannot execute (embedding model or reranker unavailable) is recorded as
``failed`` with the reason and the suite continues -- it is never replaced by a
different model.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .benchmark import load_benchmark
from .config import ConfigError
from .corpus import Corpus, load_corpus
from .dense import EmbeddingCache, EmbeddingUnavailable, get_encoder
from .experiment import ExperimentStore, environment_info, failed_run_record, run_pipeline
from .plan import (ALL_GROUPS, DEFAULT_GROUPS, PlanBuilder, RunSpec, discover_enriched,
                   enriched_base_mismatch)
from .reporting import write_suite_reports
from .rerank import RerankerUnavailable

logger = logging.getLogger(__name__)


def load_suite_spec(path: Path) -> Dict[str, Any]:
    try:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read suite spec {path}: {exc}") from exc
    for key in ("suite_id", "datasets", "embedding_models"):
        if key not in spec:
            raise ConfigError(f"suite spec {path} is missing '{key}'")
    unknown = [g for g in spec.get("groups", ALL_GROUPS) if g not in ALL_GROUPS]
    if unknown:
        raise ConfigError(f"unknown ablation group(s) {unknown}; choose from {ALL_GROUPS}")
    return spec


class _Caches:
    """Keeps at most two corpora and one encoder alive at a time."""

    def __init__(self, embedding_cache: Optional[EmbeddingCache]):
        self.embedding_cache = embedding_cache
        self._corpora: Dict[Tuple[str, Optional[str]], Corpus] = {}
        self._encoder_key: Optional[Tuple[str, str, str]] = None
        self._encoder = None
        self.encoder_errors: Dict[Tuple[str, str, str], str] = {}
        self.corpus_load_seconds: Dict[str, float] = {}

    def corpus(self, base: str, enriched: Optional[str]) -> Corpus:
        key = (base, enriched)
        if key not in self._corpora:
            import time
            t0 = time.perf_counter()
            while len(self._corpora) >= 2:
                self._corpora.pop(next(iter(self._corpora)))
            self._corpora[key] = load_corpus(Path(base), Path(enriched) if enriched else None)
            self.corpus_load_seconds[f"{base}|{enriched}"] = time.perf_counter() - t0
        return self._corpora[key]

    def encoder(self, alias: str, query_prefix: str, document_prefix: str):
        key = (alias, query_prefix, document_prefix)
        if key in self.encoder_errors:
            raise EmbeddingUnavailable(self.encoder_errors[key])
        if self._encoder_key != key:
            self._encoder = None  # release the previous model before loading another
            try:
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                self._encoder = get_encoder(alias, query_prefix, document_prefix)
            except EmbeddingUnavailable as exc:
                self.encoder_errors[key] = str(exc)
                self._encoder_key = None
                raise
            self._encoder_key = key
        return self._encoder


def _routing_profiles(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Profiles inline (``routing_profiles``) or from a JSON file (``routing_profiles_file`` -> {"profiles": {...}})."""
    if spec.get("routing_profiles"):
        return dict(spec["routing_profiles"])
    if spec.get("routing_profiles_file"):
        data = json.loads(Path(spec["routing_profiles_file"]).read_text(encoding="utf-8"))
        return dict(data.get("profiles", data))
    return {}


def _build_specs(spec: Dict[str, Any], dataset: Dict[str, Any]) -> List[RunSpec]:
    if dataset.get("enriched"):
        enriched = {k: str(v) for k, v in dataset["enriched"].items()}
    else:
        enriched = {k: str(v) for k, v in discover_enriched(Path(dataset["enriched_dir"])).items()}
    from .corpus import file_sha256
    base_sha = file_sha256(Path(dataset["base_json"]))
    skipped = dataset.setdefault("_skipped_enriched", {})
    for name, path in sorted(enriched.items()):
        reason = enriched_base_mismatch(Path(path), base_sha)
        if reason:
            logger.error("skipping enriched file %s for dataset %s: %s", path, dataset["name"], reason)
            skipped[name] = {"file": path, "reason": reason}
    enriched = {k: v for k, v in enriched.items() if k not in skipped}
    wanted = dataset.get("enrichment_models")
    if wanted:
        enriched = {k: v for k, v in enriched.items() if k in wanted}
    if not enriched:
        raise ConfigError(f"dataset {dataset['name']!r}: no enriched JSON files found")
    primary = dataset.get("primary_enrichment") or sorted(enriched)[0]
    builder = PlanBuilder(
        dataset["base_json"], enriched, primary, spec["embedding_models"],
        spec.get("primary_embedding") or spec["embedding_models"][0],
        spec.get("rerankers", []), spec.get("embedding_prefixes"),
        rerank_pool=int(spec.get("rerank_pool", 30)), routing_profiles=_routing_profiles(spec))
    return builder.build(spec.get("groups", DEFAULT_GROUPS))


def run_suite(spec: Dict[str, Any], root: Optional[Path] = None, resume: bool = False,
              only_groups: Optional[List[str]] = None, dry_run: bool = False) -> Dict[str, Any]:
    root = Path(root or spec.get("output_root", "experiments"))
    suite_id = spec["suite_id"]
    cache_dir = spec.get("embedding_cache")
    caches = _Caches(EmbeddingCache(Path(cache_dir) if cache_dir else root / "_embedding_cache"))
    store = ExperimentStore(root, suite_id)
    all_specs: List[Tuple[Dict[str, Any], RunSpec]] = []
    dataset_info: List[Dict[str, Any]] = []
    benchmarks = {}
    for dataset in spec["datasets"]:
        bench = load_benchmark(Path(dataset["benchmark"]))
        benchmarks[dataset["name"]] = bench
        specs = [s for s in _build_specs(spec, dataset)
                 if not only_groups or s.group in only_groups]
        all_specs += [(dataset, s) for s in specs]
        dataset_info.append({
            "name": dataset["name"], "benchmark": dataset["benchmark"],
            "benchmark_id": bench.benchmark_id, "query_set_version": bench.query_set_version,
            "ground_truth_version": bench.ground_truth_version, "n_queries": len(bench.queries),
            "counts": bench.meta.get("counts"), "unsupported": bench.meta.get("unsupported_categories"),
            "planned_runs": len(specs)})
    if dry_run:
        for dataset, run_spec in all_specs:
            print(f"[{dataset['name']}] {run_spec.label}  (ref: {run_spec.reference})")
        print(f"{len(all_specs)} runs planned")
        return {"planned": len(all_specs)}

    done: Dict[Tuple[str, str], bool] = {}
    if resume:
        for run in store.load_runs():
            if run.get("status") in ("ok", "partial"):
                done[(run.get("dataset_name") or "", run["label"])] = True
    # order by enrichment file to limit corpus reloads; keep plan order otherwise
    order = sorted(range(len(all_specs)),
                   key=lambda i: (all_specs[i][0]["name"], all_specs[i][1].config.enriched_json or "", i))
    executed: Dict[Tuple[str, str, str, Optional[str]], Dict[str, Any]] = {}
    summary = {"ok": 0, "partial": 0, "failed": 0, "reused": 0, "skipped_resume": 0}
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for position, index in enumerate(order, 1):
        dataset, run_spec = all_specs[index]
        config, bench = run_spec.config, benchmarks[dataset["name"]]
        logger.info("[%d/%d] %s :: %s", position, len(order), dataset["name"], run_spec.label)
        key = (dataset["name"], config.config_hash(), config.base_json, config.enriched_json)
        if key in executed:
            reused = copy.deepcopy(executed[key])
            reused.update({"dataset_name": dataset["name"], "experiment_id": reused["experiment_id"] + f"-r{os.urandom(1).hex()}",
                           "label": run_spec.label, "group": run_spec.group,
                           "reference": run_spec.reference, "reused_from": executed[key]["experiment_id"],
                           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            store.save_run(reused)
            summary["reused"] += 1
            continue
        try:
            corpus = caches.corpus(config.base_json, config.enriched_json)
            if done.get((dataset["name"], run_spec.label)):
                summary["skipped_resume"] += 1
                continue
            encoder = None
            if config.retrieval.dense:
                encoder = caches.encoder(config.embedding_model, config.retrieval.query_prefix,
                                         config.retrieval.document_prefix)
            result = run_pipeline(
                config, bench, corpus=corpus, encoder=encoder, embedding_cache=caches.embedding_cache,
                suite_id=suite_id, label=run_spec.label, group=run_spec.group,
                bootstrap_samples=int(spec.get("bootstrap_samples_per_run", 2000)))
        except (EmbeddingUnavailable, RerankerUnavailable) as exc:
            logger.error("run %s could not execute: %s", run_spec.label, exc)
            result = failed_run_record(config, run_spec.label, run_spec.group, suite_id, exc)
        except Exception as exc:  # a broken run must not abort the suite, but must be visible
            logger.exception("run %s failed", run_spec.label)
            result = failed_run_record(config, run_spec.label, run_spec.group, suite_id, exc)
        result["reference"] = run_spec.reference
        result["tags"] = run_spec.tags
        result["dataset_name"] = dataset["name"]
        store.save_run(result)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        if result["status"] in ("ok", "partial"):
            executed[key] = {k: v for k, v in result.items() if k != "_per_query"}
            executed[key]["_per_query"] = result["_per_query"]
        result.pop("_per_query", None)

    suite_info = {
        "suite_id": suite_id, "started": started,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": spec, "environment": environment_info(), "datasets": dataset_info,
        "corpus_load_seconds": caches.corpus_load_seconds, "run_summary": summary,
        "benchmark_files": {d["name"]: d["benchmark"] for d in spec["datasets"]},
    }
    for info in dataset_info:  # enrich with corpus facts for the report
        ds = next(d for d in spec["datasets"] if d["name"] == info["name"])
        try:
            info["functions"] = caches.corpus(ds["base_json"], None).meta["functions"]
        except Exception:
            info["functions"] = None
    store.save_suite_info(suite_info)
    paths = write_suite_reports(store, suite_info, [s for _, s in all_specs],
                                int(spec.get("bootstrap_samples", 10000)))
    return {"summary": summary, "reports": {k: str(v) for k, v in paths.items()},
            "suite_dir": str(store.suite_dir)}
