"""How much does the LLVM compiler-text index help if it is allowed to contribute?

Sweeps the weight of the compiler representation in the routed stack on TUNING data only (v2/v1 structural
benchmarks + generated literal queries; never the developer benchmarks), per query class and per benchmark
category, with every other setting fixed at the frozen routing profiles.  Extra scenario for structural queries:
call-graph expansion switched off, so the compiler text ('Calls: ... / Called by: ...') has to do the work alone.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import tune_routing as T
from optima.rag.research.dense import EmbeddingCache
from optima.rag.research.metrics import evaluate_query
from optima.rag.research.routing import CLASSES, classify_query

WEIGHTS = [0, 0.25, 0.5, 1, 2, 4]
profiles = json.loads(Path("configs/routing_profiles.json").read_text())["profiles"]
prefixes = {"bge-small": T.PREFIXES["bge-small"][1], "bge-base": T.PREFIXES["bge-base"][1], "qwen3-0.6b": T.PREFIXES["qwen3-0.6b"][1]}
cache = EmbeddingCache(Path("experiments/_embedding_cache"))
corpus_cache, pipes, qsets = {}, {}, {}
t0 = time.time()
for emb in prefixes:
    for ds in T.DATASETS:
        pipes[(ds, emb)] = T.build(ds, emb, "compiler_hybrid_rag", prefixes[emb], corpus_cache, cache)
        qsets[(ds, emb)] = [(t, r, c, classify_query(t)) for t, r, c in T.load_queries(ds)]
print(f"built {len(pipes)} pipelines in {time.time() - t0:.0f}s", flush=True)


def evaluate(profile_for_class):
    """rows: (dataset, embedding, class, category, mrr, recall10)"""
    rows = []
    for (ds, emb), pipe in pipes.items():
        for text, rel, cat, cls in qsets[(ds, emb)]:
            m = evaluate_query(T.run_query(pipe, text, profile_for_class(cls)), rel)
            rows.append((ds, emb, cls, cat, m["mrr"] if m else 0.0, m["recall_at_10"] if m else 0.0))
    return rows


def with_compiler(w, graph_off_for=()):
    def f(cls):
        p = json.loads(json.dumps(profiles[cls]))
        p.setdefault("representation_weights", {})["compiler"] = w
        if cls in graph_off_for:
            p["graph_weight"] = 0
        return p
    return f


out = []
for scenario, graph_off in (("graph as tuned", ()), ("call graph OFF for structural queries", ("structural",))):
    for w in WEIGHTS:
        for ds, emb, cls, cat, mrr, r10 in evaluate(with_compiler(w, graph_off)):
            out.append({"scenario": scenario, "compiler_weight": w, "dataset": ds, "embedding": emb, "class": cls,
                        "category": cat, "mrr": mrr, "recall_at_10": r10})
    print("done", scenario, f"{time.time() - t0:.0f}s", flush=True)
import pandas as pd
df = pd.DataFrame(out)
df.to_csv("reports/compiler_text_test/tuning_sweep_raw.csv", index=False)
agg = df.groupby(["scenario", "compiler_weight", "class"])[["mrr", "recall_at_10"]].mean().reset_index()
agg["n_queries"] = df.groupby(["scenario", "compiler_weight", "class"]).size().to_numpy()
agg.to_csv("reports/compiler_text_test/tuning_sweep_by_class.csv", index=False)
cat = df.groupby(["scenario", "compiler_weight", "category"])[["mrr", "recall_at_10"]].mean().reset_index()
cat["n_queries"] = df.groupby(["scenario", "compiler_weight", "category"]).size().to_numpy()
cat.to_csv("reports/compiler_text_test/tuning_sweep_by_category.csv", index=False)
print(agg[agg.scenario == "graph as tuned"].pivot(index="compiler_weight", columns="class", values="mrr").round(3).to_string())
