"""Tune query-instruction prefixes and routed fusion weights on TUNING data only.

Tuning data: the v2 structural benchmarks (+ the generated literal tuning sets).  The hand-written developer
benchmarks (``developer_v1.json``) are NEVER read here; they are the held-out test.

Everything runs through the real pipeline code: indexes are built once per (dataset, embedding), query vectors are
cached, and only ``_first_stage_lists`` / fusion / graph expansion are re-run for each weight setting.
"""
import argparse, json, logging, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

from optima.rag.research.benchmark import load_benchmark
from optima.rag.research.corpus import load_corpus
from optima.rag.research.dense import EmbeddingCache, get_encoder
from optima.rag.research.fusion import rrf_fuse
from optima.rag.research.graph import GraphIndex
from optima.rag.research.metrics import evaluate_query, relevance_map
from optima.rag.research.pipeline import ResearchPipeline
from optima.rag.research.plan import PlanBuilder
from optima.rag.research.routing import CLASSES, classify_query, profile_for

DATASETS = {
    "chess": dict(base="optima_outputs/base/chess-engine-in-cpp-master/base.json",
                  enr=("qwen-qwen2.5-coder-14b-instruct-chess", "output/chess Engine/enhanced_qwen-qwen2.5-coder-14b-instruct-chess.json"),
                  bench=["benchmarks/chess/structural_v2.json", "benchmarks/chess/tuning_literals.json"]),
    "cjson": dict(base="optima_outputs/base/cJSON-master/base.json",
                  enr=("qwen-qwen2.5-coder-14b-instruct", "output/cJson/enhanced_qwen-qwen2.5-coder-14b-instruct.json"),
                  bench=["benchmarks/cjson/structural_v2.json", "benchmarks/cjson/tuning_literals.json"]),
    "xxhash": dict(base="optima_outputs/base/xxHash-dev/base.json",
                   enr=("qwen-qwen2.5-coder-14b-instruct-xhash", "output/xxhash/enhanced_qwen-qwen2.5-coder-14b-instruct-xhash.json"),
                   bench=["benchmarks/xxhash/structural_v1.json", "benchmarks/xxhash/tuning_literals.json"]),
}
PREFIXES = {
    "bge-small": ["", "Represent this sentence for searching relevant passages: "],
    "bge-base": ["", "Represent this sentence for searching relevant passages: "],
    "qwen3-0.6b": ["", "Instruct: Given a question about source code, retrieve the function that implements or answers it\nQuery: "],
}


class CachingEncoder:
    """Caches query vectors so weight sweeps do not re-encode (documents pass straight through)."""
    def __init__(self, inner):
        self.inner, self.info, self.cache = inner, inner.info, {}
    def encode_documents(self, texts):
        return self.inner.encode_documents(texts)
    def encode_queries(self, texts):
        missing = [t for t in texts if t not in self.cache]
        if missing:
            for t, v in zip(missing, self.inner.encode_queries(missing)):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts])


def build(ds, emb, variant, prefix, corpus_cache, cache):
    spec = DATASETS[ds]
    slug, path = spec["enr"]
    builder = PlanBuilder(spec["base"], {slug: path}, slug, [emb], emb)
    run = next(s for s in builder.build(["pipelines"]) if s.tags.get("variant") == variant)
    key = (spec["base"], run.config.enriched_json)
    if key not in corpus_cache:
        corpus_cache[key] = load_corpus(Path(spec["base"]), Path(run.config.enriched_json) if run.config.enriched_json else None)
    corpus = corpus_cache[key]
    enc = CachingEncoder(get_encoder(emb, prefix, ""))
    pipe = ResearchPipeline(run.config, corpus=corpus, encoder=enc, embedding_cache=cache,
                            graph=GraphIndex(corpus) if run.config.graph.enabled else None)
    pipe.build()
    return pipe


def load_queries(ds):
    out = []
    for f in DATASETS[ds]["bench"]:
        for q in load_benchmark(Path(f)).queries:
            rel = relevance_map(q)
            if rel:
                out.append((q["query"], rel, q.get("category")))
    return out


def run_query(pipe, text, profile):
    """Same code path as ResearchPipeline.search (first stage -> fuse -> optional graph), without timing."""
    cfg = pipe.config
    lists = pipe._first_stage_lists(text, {}, profile)
    fused = rrf_fuse(lists, cfg.retrieval.rrf_k)
    ranking = fused
    if cfg.graph.enabled and pipe.graph is not None and fused:
        ranking, _, _ = pipe._expand(fused, lists, profile)
    return [i.id for i in ranking[:10]]


def mrr_for(pipe, queries, profiles):
    vals = []
    for text, rel, _cat, cls in queries:
        m = evaluate_query(run_query(pipe, text, profiles.get(cls, {})), rel)
        vals.append(m["mrr"] if m else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prefix", "routing"], required=True)
    ap.add_argument("--embeddings", nargs="+", default=["bge-small", "bge-base", "qwen3-0.6b"])
    ap.add_argument("--prefix-json", help="chosen prefixes {embedding: prefix} for --mode routing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cache = EmbeddingCache(Path("experiments/_embedding_cache"))
    corpus_cache, result = {}, {}
    chosen = json.loads(Path(args.prefix_json).read_text()) if args.prefix_json else {}

    if args.mode == "prefix":
        for emb in args.embeddings:
            for variant in ("baseline_rag", "semantic_enrichment_rag", "compiler_hybrid_rag"):
                for prefix in PREFIXES[emb]:
                    scores = {}
                    for ds in DATASETS:
                        pipe = build(ds, emb, variant, prefix, corpus_cache, cache)
                        qs = [(t, r, c, classify_query(t)) for t, r, c in load_queries(ds)]
                        by = defaultdict(list)
                        for t, r, c, cls in qs:
                            m = evaluate_query(run_query(pipe, t, {}), r)
                            by[cls].append(m["mrr"] if m else 0.0)
                        scores[ds] = {"all": float(np.mean([v for x in by.values() for v in x])), **{k: float(np.mean(v)) for k, v in by.items()}}
                    result[f"{emb}|{variant}|{'instruct' if prefix else 'none'}"] = scores
                    print(f"{emb:11s} {variant:24s} {'instruct' if prefix else 'none':8s} " +
                          "  ".join(f"{ds}={scores[ds]['all']:.3f}" for ds in scores), flush=True)
    else:
        grid = {"bm25_weight": [0, 0.25, 0.5, 1, 2, 4], "exact_weight": [0, 0.5, 1, 2, 4],
                "rep.semantic": [0, 0.5, 1, 2], "rep.compiler": [0, 0.5, 1, 2], "rep.raw": [0.5, 1, 2],
                "graph_weight": [0, 0.25, 0.5, 1, 2]}
        pipes, qsets = {}, {}
        t0 = time.time()
        for emb in args.embeddings:
            for ds in DATASETS:
                pipes[(ds, emb)] = build(ds, emb, "compiler_hybrid_rag", chosen.get(emb, ""), corpus_cache, cache)
                qsets[(ds, emb)] = [(t, r, c, classify_query(t)) for t, r, c in load_queries(ds)]
        print(f"built {len(pipes)} pipelines in {time.time() - t0:.0f}s", flush=True)

        def score(cls, profile):
            vals = []
            for key, pipe in pipes.items():
                qs = [q for q in qsets[key] if q[3] == cls]
                if qs:
                    vals.append(mrr_for(pipe, qs, {cls: profile}))
            return float(np.mean(vals)) if vals else float("nan"), len(vals)

        def to_profile(p):
            prof = {k: v for k, v in p.items() if not k.startswith("rep.")}
            reps = {k[4:]: v for k, v in p.items() if k.startswith("rep.")}
            if reps:
                prof["representation_weights"] = reps
            return prof

        counts = {cls: sum(1 for key in pipes for q in qsets[key] if q[3] == cls) // len(args.embeddings) for cls in CLASSES}
        profiles, report = {}, {}
        for cls in CLASSES:
            current = {}
            base, nval = score(cls, to_profile(current))
            best = base
            for _ in range(2):
                for param, values in grid.items():
                    cands = []
                    for v in values:
                        trial = {**current, param: v}
                        cands.append((score(cls, to_profile(trial))[0], v))
                    s, v = max(cands, key=lambda x: x[0])
                    cur_v = current.get(param, {"bm25_weight": 1, "exact_weight": 1, "graph_weight": 0.5}.get(param, 1))
                    if s > best + 0.003 and v != cur_v:
                        current[param], best = v, s
            profiles[cls] = to_profile(current)
            report[cls] = {"queries_per_dataset_embedding_avg": counts[cls], "mrr_neutral": base, "mrr_tuned": best, "profile": profiles[cls]}
            print(f"{cls:11s} n={counts[cls]:4d}  neutral {base:.3f} -> tuned {best:.3f}  {profiles[cls]}", flush=True)
        result = {"profiles": profiles, "report": report, "prefixes": chosen,
                  "tuned_on": {ds: DATASETS[ds]["bench"] for ds in DATASETS}, "embeddings": args.embeddings,
                  "objective": "mean MRR per query class, averaged over (dataset, embedding); coordinate descent, 2 passes, accept gain >= 0.003"}
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
