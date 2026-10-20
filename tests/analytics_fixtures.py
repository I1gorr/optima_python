"""Small synthetic experiment store for the analytics tests (no real data, no models)."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

QUALITY = ["recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "precision_at_1", "precision_at_3", "precision_at_5",
           "precision_at_10", "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10", "mrr"]
CATEGORIES = ["exact_name"] * 4 + ["behavioral"] * 4 + ["caller"] * 4          # 12 queries, 3 families
QUERY_IDS = [f"q{i:02d}" for i in range(12)]
SEEDS = [f"fn{i // 2}" for i in range(12)]                                      # pairs share a seed (cluster)
BASELINE_RR = [1.0, 1.0, 0.5, 1.0, 0.2, 0.0, 0.25, 0.0, 0.0, 0.1, 0.0, 0.2]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_base(path: Path, content: str = "{}") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return sha(path)


def config(name: str, emb: str, *, reps=("raw",), dense=True, bm25=False, exact=False, graph=False, rerank=None,
           enriched_json: Optional[str] = None, enrichment_model: str = "none", base_json: str = "base.json") -> Dict[str, Any]:
    return {"name": name, "base_json": base_json, "enriched_json": enriched_json, "enrichment_model": enrichment_model,
            "embedding_model": emb, "top_k": 10,
            "retrieval": {"representations": list(reps), "dense": dense, "bm25": bm25, "exact_match": exact, "rrf_k": 60},
            "graph": ({"enabled": True, "relations": ["callers", "callees"], "max_depth": 1, "graph_weight": 0.5} if graph else {"enabled": False, "relations": []}),
            "rerank": ({"enabled": True, "model": rerank, "pool_size": 30} if rerank else {"enabled": False})}


def write_run(suite_dir: Path, eid: str, label: str, cfg: Dict[str, Any], rr: List[float], *, group: str = "grid", suite_id: str = "toy-suite",
              dataset: str = "toy", base_sha: str = "b" * 64, status: str = "ok", enr_model: Optional[str] = None,
              coverage: Optional[float] = None, mismatch: bool = False, ts: str = "2026-01-01T00:00:00+00:00",
              chash: Optional[str] = None, query_ms: float = 5.0, with_per_query: bool = True, tags: Optional[dict] = None) -> Path:
    run_dir = suite_dir / "runs" / eid
    run_dir.mkdir(parents=True, exist_ok=True)
    n = len(rr)
    overall = {k: (sum(rr) / n if n else 0.0) for k in QUALITY}
    emb_alias = cfg["embedding_model"]
    result = {
        "schema_version": 1, "experiment_id": eid, "suite_id": suite_id, "group": group, "label": label, "timestamp": ts,
        "duration_seconds": 1.0, "status": status, "pipeline": cfg["name"], "config": cfg,
        "config_hash": chash or hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if k not in ("name", "base_json", "enriched_json", "enrichment_model")}, sort_keys=True).encode()).hexdigest()[:12],
        "embedding_model": {"alias": emb_alias, "model_name": {"emb-a": "BAAI/bge-small-en-v1.5", "emb-b": "BAAI/bge-base-en-v1.5"}.get(emb_alias, emb_alias), "dimension": 8, "device": "cpu"},
        "enrichment_model": cfg["enrichment_model"],
        "enrichment_run": ({"model": enr_model, "quantization": "nf4"} if enr_model else None),
        "enrichment_coverage": ({"semantic_coverage": coverage if coverage is not None else 1.0, "failed_or_incomplete": 0} if enr_model else None),
        "enrichment_cost": ({"total_tokens": 1000.0, "latency_seconds": 3600.0} if enr_model else None),
        "query_set_version": "1.0.0", "ground_truth_version": "1.0.0", "benchmark_id": "toy-bench",
        "dataset": {"project": "Toy", "functions": 10, "base_sha256": base_sha, "base_sha256_mismatch": mismatch},
        "index": {"index_bytes_total": 1000},
        "metrics": ({"n_queries_total": n, "n_queries_evaluated": n, "overall": overall,
                     "confidence_95": {"mrr": {"mean": overall["mrr"], "ci_low": overall["mrr"] - 0.1, "ci_high": overall["mrr"] + 0.1}},
                     "candidate_pool": {}} if status == "ok" else {}),
        "timing": {"build_seconds": 0.1, "build_breakdown_s": {"embed_s": 0.1}, "query_ms_by_stage": {"total": {"mean": query_ms, "median": query_ms, "p95": query_ms * 1.5}}},
        "resources": {"gpu_peak_allocated_mb": 100.0, "rss_peak_mb": 500.0}, "errors": ([] if status == "ok" else ["boom"]),
        "reference": None, "tags": tags or {}, "dataset_name": dataset,
    }
    (run_dir / "result.json").write_text(json.dumps(result))
    if with_per_query and status == "ok":
        with gzip.open(run_dir / "per_query.jsonl.gz", "wt") as h:
            for qid, cat, seed, v in zip(QUERY_IDS, CATEGORIES, SEEDS, rr):
                m = {k: v for k in QUALITY}
                m.update({"first_relevant_rank": None, "num_relevant": 1})
                h.write(json.dumps({"query_id": qid, "category": cat, "seed": seed, "query": f"find {qid}", "relevance": {seed: 1.0},
                                    "metrics": m, "ranking": [{"function_id": seed, "rank": 1}], "timings_ms": {"total": query_ms}}) + "\n")
    return run_dir


def write_suite_json(suite_dir: Path, enriched_dir: str, base_json: str, suite_id: str = "toy-suite") -> None:
    (suite_dir / "suite.json").write_text(json.dumps({"suite_id": suite_id, "spec": {"primary_embedding": "emb-b", "datasets": [
        {"name": "toy", "base_json": base_json, "enriched_dir": enriched_dir, "primary_enrichment": "enr-good"}]}, "environment": {"gpu": "none"}}))


def write_artifact(path: Path, model_id: str, base_sha: str, *, dataset: str = "toy", n: int = 4, partial: bool = False, usable: bool = True,
                   max_new_tokens: int = 768) -> None:
    funcs = []
    for i in range(n):
        enr = {"purpose": "does a thing" if usable else "", "behavior": "behaves" if usable else "", "summary": "short summary" if usable else "", "keywords": ["a", "b"],
               "concepts": ["c"], "status": "completed", "json_parse_status": "ok", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
               "evaluation": {"schema_valid": True, "json_valid": True, "latency_seconds": 1.0}}
        funcs.append({"id": f"fn{i}", "name": f"fn{i}", "enrichment": enr, "raw_llm_response": "{}"})
    data = {"project": {"name": "Toy"}, "files": [{"functions": funcs}],
            "enrichment_metadata": {"model": model_id, "model_slug": "x", "quantization": "nf4", "total_functions": n, "enriched_functions": n, "failed_functions": 0, "partial": partial},
            "experiment": {"dataset": dataset, "model": model_id, "base_sha256": base_sha, "max_new_tokens": max_new_tokens, "config_hash": "abc", "quantization": "nf4"},
            "enrichment_metrics": {"success_rate": 1.0, "json_valid_rate": 1.0, "schema_valid_rate": 1.0, "total_enrichment_seconds": 4.0, "total_input_tokens": 40, "total_output_tokens": 20}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def build_store(root: Path) -> Dict[str, Any]:
    """A store with raw baselines, matched counterparts and two enrichment models (one misplaced)."""
    base_sha = write_base(root / "bases" / "toy" / "base.json", '{"toy": 1}')
    suite = root / "experiments" / "toy-suite"
    suite.mkdir(parents=True)
    enr_dir = root / "enriched"
    write_artifact(enr_dir / "enhanced_enr-good.json", "Qwen/Qwen2.5-7B-Instruct", base_sha)
    write_artifact(enr_dir / "enhanced_enr-wrong.json", "Qwen/Qwen2.5-3B-Instruct", "f" * 64, dataset="other")        # another project's file
    write_suite_json(suite, str(enr_dir), "base.json")
    n = len(BASELINE_RR)
    good = [min(1.0, v + (0.3 if c == "behavioral" else -0.1 if c == "caller" else 0.0)) for v, c in zip(BASELINE_RR, CATEGORIES)]
    good = [max(0.0, v) for v in good]
    good_path = str(enr_dir / "enhanced_enr-good.json")
    wrong_path = str(enr_dir / "enhanced_enr-wrong.json")
    ids = {}

    def run(eid, label, cfg, rr, **kw):
        write_run(suite, eid, label, cfg, rr, base_sha=base_sha, **kw)
        ids[label] = eid

    for emb, shift in (("emb-a", 0.0), ("emb-b", 0.05)):
        base = [min(1.0, v + shift) for v in BASELINE_RR]
        run(f"e-base-{emb}", f"baseline|{emb}", config("baseline_rag", emb), base, query_ms=4.0)
        run(f"e-sem-{emb}", f"semantic|{emb}", config("semantic_enrichment_rag", emb, reps=("semantic",), enriched_json=good_path, enrichment_model="enr-good"),
            [min(1.0, v + 0.04) for v in base], enr_model="Qwen/Qwen2.5-7B-Instruct", ts="2026-01-02T00:00:00+00:00")
        run(f"e-rawh-{emb}", f"raw_hybrid|{emb}", config("ablation", emb, reps=("raw",), bm25=True), [min(1.0, v + 0.1) for v in base], query_ms=6.0)
        run(f"e-rawc-{emb}", f"rawcomp|{emb}", config("ablation", emb, reps=("raw", "compiler"), bm25=True, exact=True, graph=True), [min(1.0, v + 0.1) for v in base], query_ms=8.0)
        run(f"e-comp-{emb}", f"compiler|{emb}", config("compiler_hybrid_rag", emb, reps=("raw", "semantic", "compiler"), bm25=True, exact=True, graph=True, enriched_json=good_path, enrichment_model="enr-good"),
            good if emb == "emb-b" else [min(1.0, v + 0.1) for v in base], enr_model="Qwen/Qwen2.5-7B-Instruct", query_ms=9.0)
    # a repeat of an existing configuration in another group (must be de-duplicated, identical)
    run("e-base-emb-a-dup", "dup_baseline|emb-a", config("baseline_rag", "emb-a"), list(BASELINE_RR), group="pipelines", ts="2026-01-03T00:00:00+00:00")
    # BM25-only raw run (difficulty reference)
    run("e-bm25", "bm25|raw", config("ablation", "emb-a", reps=("raw",), dense=False, bm25=True), [1.0, 0.5, 1.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    # a run that used the misplaced file (joined 0 functions)
    run("e-wrong", "semantic_wrong|emb-b", config("semantic_enrichment_rag", "emb-b", reps=("semantic",), enriched_json=wrong_path, enrichment_model="enr-wrong"),
        list(BASELINE_RR), enr_model="Qwen/Qwen2.5-3B-Instruct", coverage=0.0, mismatch=True)
    # a failed run
    run("e-failed", "failed|emb-b", config("baseline_rag", "emb-b", dense=True, rerank="ms-marco-minilm"), [], status="failed", chash="deadbeef0001")
    return {"root": root, "suite": suite, "base_sha": base_sha, "enr_dir": enr_dir, "ids": ids, "n": n,
            "base_dir": root / "bases", "good_path": good_path}
