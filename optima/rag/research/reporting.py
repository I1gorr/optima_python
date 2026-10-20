"""Tables, comparisons, markdown report and plots for a finished suite.

The report is generated mechanically from stored results.  It states numbers,
deltas against the declared reference with paired-bootstrap 95% intervals, and
costs; it makes no causal claims.  A delta is flagged ``CI excludes 0`` only when
its interval does -- that is a statement about this benchmark's query sample.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .experiment import ExperimentStore
from .metrics import HEADLINE_METRICS, METRIC_NAMES, paired_bootstrap

logger = logging.getLogger(__name__)

COMPARE_METRICS = ("mrr", "ndcg_at_10", "recall_at_10", "recall_at_5", "recall_at_1")
# Validated categorical slots (dataviz reference palette, first three: all-pairs safe).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(columns or dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------- load
def load_runs(store: ExperimentStore) -> List[Dict[str, Any]]:
    """Run results with per-query metrics attached as ``_pq`` (query_id -> row)."""
    runs = store.load_runs()
    for run in runs:
        if run.get("status") == "failed" or not run.get("metrics"):
            continue
        run_dir = store.runs_dir / run["experiment_id"]
        try:
            rows = ExperimentStore.load_per_query(run_dir)
        except OSError:
            rows = []
        run["_pq"] = {r["query_id"]: r for r in rows}
    return runs


def _dataset(run: Dict[str, Any]) -> str:
    return (run.get("dataset_name") or (run.get("dataset") or {}).get("project")
            or Path(str(run.get("config", {}).get("base_json", ""))).parent.name or "dataset")


# --------------------------------------------------------------- aggregates
def aggregate_rows(runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        base = {"dataset": _dataset(run), "group": run.get("group"), "label": run.get("label"),
                "pipeline": run.get("pipeline"), "experiment_id": run.get("experiment_id"),
                "config_hash": run.get("config_hash"), "status": run.get("status"),
                "embedding": (run.get("config") or {}).get("embedding_model"),
                "enrichment_model": run.get("enrichment_model"),
                "failure": run.get("error") if run.get("status") == "failed" else ""}
        if run.get("status") == "failed" or not run.get("metrics"):
            rows.append(base)
            continue
        cfg, m = run["config"], run["metrics"]
        timing, res = run["timing"], run["resources"]
        stages = timing["query_ms_by_stage"]
        total = stages.get("total", {})
        pool = m["candidate_pool"]
        cost = run.get("enrichment_cost") or {}
        cov = run.get("enrichment_coverage") or {}
        row = dict(base)
        row.update({
            "representations": "+".join(cfg["retrieval"]["representations"]),
            "dense": cfg["retrieval"]["dense"], "bm25": cfg["retrieval"]["bm25"],
            "exact": cfg["retrieval"]["exact_match"], "graph": cfg["graph"]["enabled"],
            "graph_relations": "+".join(cfg["graph"]["relations"]) if cfg["graph"]["enabled"] else "",
            "graph_depth": cfg["graph"]["max_depth"] if cfg["graph"]["enabled"] else "",
            "rerank": (cfg["rerank"]["model"] if cfg["rerank"]["enabled"] else ""),
            "query_set_version": run.get("query_set_version"),
            "ground_truth_version": run.get("ground_truth_version"),
            "n_queries": m["n_queries_evaluated"],
        })
        for name in METRIC_NAMES:
            row[name] = m["overall"].get(name)
        for name in HEADLINE_METRICS:
            ci = m["confidence_95"].get(name, {})
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = ci.get("ci_low"), ci.get("ci_high")
        row.update({
            "build_s": timing["build_seconds"],
            "embed_s": timing["build_breakdown_s"].get("embed_s"),
            "embed_cache_hit": all(timing["embedding_cache_hits"].values()) if timing["embedding_cache_hits"] else "",
            "index_mb": round((run["index"]["index_bytes_total"] or 0) / 1e6, 3),
            "query_ms_mean": total.get("mean"), "query_ms_median": total.get("median"),
            "query_ms_p95": total.get("p95"),
            "rerank_ms_mean": stages.get("rerank", {}).get("mean"),
            "graph_ms_mean": stages.get("graph", {}).get("mean"),
            "rss_peak_mb": res.get("rss_peak_mb"), "gpu_peak_mb": res.get("gpu_peak_allocated_mb"),
            "candidates_before_graph": pool["mean_candidates_before_graph"],
            "candidates_after_graph": pool["mean_candidates_after_graph"],
            "graph_added_mean": pool["mean_graph_added"],
            "pool_recall_before_graph": pool["recall_before_graph"],
            "pool_recall_after_graph": pool["recall_after_graph"],
            "semantic_coverage": cov.get("semantic_coverage"),
            "enrichment_failed_or_incomplete": cov.get("failed_or_incomplete"),
            "enrichment_total_tokens": cost.get("total_tokens"),
            "enrichment_seconds": cost.get("latency_seconds"),
            "errors": len(run.get("errors", [])),
            "skipped_queries": len(m["skipped"]),
        })
        rows.append(row)
    return rows


def category_rows(runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        if not run.get("metrics"):
            continue
        for category, values in run["metrics"]["by_category"].items():
            row = {"dataset": _dataset(run), "group": run["group"], "label": run["label"],
                   "pipeline": run["pipeline"], "category": category}
            row.update({k: values.get(k) for k in METRIC_NAMES})
            row["n_queries"] = values["n_queries"]
            rows.append(row)
    return rows


def per_query_rows(runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for run in runs:
        for qid, pq in (run.get("_pq") or {}).items():
            ranking = pq["ranking"]
            m = pq.get("metrics") or {}
            rows.append({
                "dataset": _dataset(run), "group": run["group"], "label": run["label"],
                "query_id": qid, "category": pq["category"],
                "first_relevant_rank": m.get("first_relevant_rank"),
                **{k: m.get(k) for k in ("mrr", "ndcg_at_10", "recall_at_10", "recall_at_5", "recall_at_1")},
                "top1": ranking[0]["function_id"] if ranking else "",
                "top1_methods": "+".join(ranking[0]["methods"]) if ranking else "",
                "latency_ms": pq["timings_ms"].get("total"), "error": pq.get("error") or ""})
    return rows


# -------------------------------------------------------------- comparisons
def comparison_rows(runs: Sequence[Dict[str, Any]], n_boot: int = 10000,
                    seed: int = 0) -> List[Dict[str, Any]]:
    by_label = {(_dataset(r), r["label"]): r for r in runs if r.get("_pq")}
    rows = []
    for run in runs:
        ref_label = run.get("reference") or _reference_of(run)
        if not run.get("_pq") or not ref_label or ref_label == run["label"]:
            continue
        ref = by_label.get((_dataset(run), ref_label))
        if ref is None:
            continue
        shared = [q for q in run["_pq"] if q in ref["_pq"]
                  and run["_pq"][q].get("metrics") and ref["_pq"][q].get("metrics")]
        if not shared:
            continue
        groups = [run["_pq"][q].get("seed") or q for q in shared]
        for metric in COMPARE_METRICS:
            cand = [run["_pq"][q]["metrics"][metric] for q in shared]
            base = [ref["_pq"][q]["metrics"][metric] for q in shared]
            result = paired_bootstrap(cand, base, groups=groups, n_boot=n_boot, seed=seed)
            rows.append({
                "dataset": _dataset(run), "group": run["group"], "label": run["label"],
                "reference": ref_label, "metric": metric, "n_queries": len(shared),
                "candidate_mean": float(np.mean(cand)), "reference_mean": float(np.mean(base)),
                "mean_difference": result["mean_difference"], "ci_low": result["ci_low"],
                "ci_high": result["ci_high"], "ci_excludes_zero": result["ci_excludes_zero"],
                "wins": result["wins"], "losses": result["losses"], "ties": result["ties"]})
    return rows


_REFERENCES: Dict[str, str] = {}


def set_references(specs: Iterable[Any]) -> None:
    _REFERENCES.update({s.label: s.reference for s in specs if s.reference})


def _reference_of(run: Dict[str, Any]) -> Optional[str]:
    return _REFERENCES.get(run.get("label", ""))


def category_comparison_rows(runs: Sequence[Dict[str, Any]], n_boot: int = 2000) -> List[Dict[str, Any]]:
    """Per-category paired deltas (MRR, nDCG@10) for runs with a reference."""
    by_label = {(_dataset(r), r["label"]): r for r in runs if r.get("_pq")}
    rows = []
    for run in runs:
        ref_label = run.get("reference") or _reference_of(run)
        ref = by_label.get((_dataset(run), ref_label)) if ref_label else None
        if not run.get("_pq") or ref is None:
            continue
        for category in sorted({p["category"] for p in run["_pq"].values()}):
            shared = [q for q, p in run["_pq"].items() if p["category"] == category
                      and q in ref["_pq"] and p.get("metrics") and ref["_pq"][q].get("metrics")]
            if not shared:
                continue
            for metric in ("mrr", "ndcg_at_10"):
                result = paired_bootstrap([run["_pq"][q]["metrics"][metric] for q in shared],
                                          [ref["_pq"][q]["metrics"][metric] for q in shared],
                                          groups=[run["_pq"][q].get("seed") or q for q in shared],
                                          n_boot=n_boot)
                rows.append({"dataset": _dataset(run), "group": run["group"], "label": run["label"],
                             "reference": ref_label, "category": category, "metric": metric,
                             "n_queries": len(shared), "mean_difference": result["mean_difference"],
                             "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                             "ci_excludes_zero": result["ci_excludes_zero"]})
    return rows


# ------------------------------------------------------------------ markdown
def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _delta(row: Dict[str, Any]) -> str:
    flag = " *" if row["ci_excludes_zero"] else ""
    return (f"{row['mean_difference']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]{flag}")


def render_report(suite_id: str, runs: Sequence[Dict[str, Any]], aggregates: Sequence[Dict[str, Any]],
                  comparisons: Sequence[Dict[str, Any]], suite_info: Dict[str, Any],
                  plots: Sequence[Path] = ()) -> str:
    lines: List[str] = [f"# Optima RAG evaluation report: `{suite_id}`", ""]
    lines += ["Generated mechanically from stored results; numbers only, no causal claims.",
              "Deltas are `candidate − reference` on identical queries with paired-bootstrap 95% "
              "intervals; `*` marks intervals that exclude 0 *for this benchmark's queries*.", ""]
    env = suite_info.get("environment", {})
    lines += ["## Setup", "", f"- Environment: Python {env.get('python')}, torch {env.get('torch')}, "
              f"sentence-transformers {env.get('sentence_transformers')}, GPU {env.get('gpu', 'none')}",
              f"- Git commit: `{env.get('git_commit')}`"]
    for ds in suite_info.get("datasets", []):
        lines.append(f"- Dataset `{ds['name']}`: {ds.get('functions')} functions, benchmark "
                     f"`{ds.get('benchmark_id')}` v{ds.get('query_set_version')} "
                     f"({ds.get('n_queries')} queries; categories {ds.get('counts')}); "
                     f"unsupported categories: {ds.get('unsupported') or 'none'}")
    failed = [a for a in aggregates if a.get("status") == "failed"]
    partial = [a for a in aggregates if a.get("status") == "partial"]
    lines += ["", f"Runs: {len(aggregates)} total, {len(failed)} failed, {len(partial)} partial "
              "(per-query errors).", ""]
    if failed:
        lines += ["### Runs that could not be executed", ""]
        lines += [_table(["label", "reason"], [[a["label"], a["failure"]] for a in failed]), ""]

    ok = [a for a in aggregates if a.get("status") in ("ok", "partial")]
    datasets = sorted({a["dataset"] for a in ok})
    for dataset in datasets:
        rows = [a for a in ok if a["dataset"] == dataset]
        lines += [f"## Dataset: {dataset}", ""]
        pipe = [a for a in rows if a["group"] == "pipelines"]
        if pipe:
            lines += ["### Pipeline comparison", ""]
            cmp_by = {(c["label"], c["metric"]): c for c in comparisons
                      if c["dataset"] == dataset and c["group"] == "pipelines"}
            table = []
            for a in pipe:
                d = cmp_by.get((a["label"], "mrr"))
                table.append([a["label"].split("/", 1)[1], _fmt(a["mrr"]), _fmt(a["ndcg_at_10"]),
                              _fmt(a["recall_at_1"]), _fmt(a["recall_at_5"]), _fmt(a["recall_at_10"]),
                              _fmt(a["precision_at_5"]), _fmt(a["hit_at_5"]),
                              _delta(d) if d else "reference"])
            lines += [_table(["pipeline", "MRR", "nDCG@10", "R@1", "R@5", "R@10", "P@5", "Hit@5",
                              "ΔMRR vs baseline"], table), ""]
        lines += _category_section(dataset, runs)
        for group in ("enrichment", "representations", "hybrid", "graph", "rerank", "embedding"):
            grows = [a for a in rows if a["group"] == group]
            if not grows:
                continue
            lines += [f"### Ablation: {group}", ""]
            cmp_by = {(c["label"], c["metric"]): c for c in comparisons
                      if c["dataset"] == dataset and c["group"] == group}
            table = []
            for a in grows:
                d, dn = cmp_by.get((a["label"], "mrr")), cmp_by.get((a["label"], "ndcg_at_10"))
                table.append([a["label"].split("/", 1)[1], _fmt(a["mrr"]), _fmt(a["ndcg_at_10"]),
                              _fmt(a["recall_at_10"]),
                              _delta(d) if d else "reference", _delta(dn) if dn else "reference",
                              _fmt(a["query_ms_mean"], 1)])
            lines += [_table(["variant", "MRR", "nDCG@10", "R@10", "ΔMRR [95% CI]",
                              "ΔnDCG@10 [95% CI]", "ms/query"], table), ""]
            if group == "graph":
                lines += [_graph_coverage_table(grows), ""]
        lines += _cost_section(rows)
    if plots:
        lines += ["## Plots", ""] + [f"- `{p.name}`" for p in plots] + [""]
    lines += ["## Limitations of this report", "",
              "- The benchmark's structural categories are labelled from the same call graph that "
              "graph expansion traverses; see `benchmarks/README.md`.",
              "- Behavioural labels are single-annotator; no inter-annotator agreement exists.",
              "- Queries about one target function are correlated; intervals are paired-bootstrap over queries, "
              "so treat them as optimistic.",
              "- Latency/RSS are wall-clock on one machine; RSS is process-wide.", ""]
    return "\n".join(lines)


def _category_section(dataset: str, runs: Sequence[Dict[str, Any]]) -> List[str]:
    rows = [r for r in runs if _dataset(r) == dataset and r.get("group") == "pipelines" and r.get("metrics")]
    if not rows:
        return []
    cats = sorted({c for r in rows for c in r["metrics"]["by_category"]})
    table = []
    for r in rows:
        table.append([r["label"].split("/", 1)[1]] + [
            _fmt(r["metrics"]["by_category"].get(c, {}).get("mrr")) for c in cats])
    counts = [str(rows[0]["metrics"]["by_category"].get(c, {}).get("n_queries", "")) for c in cats]
    return ["### Per-category MRR (pipelines)", "",
            _table(["pipeline"] + [f"{c} (n={n})" for c, n in zip(cats, counts)], table), ""]


def _graph_coverage_table(rows: Sequence[Dict[str, Any]]) -> str:
    table = [[a["label"].split("/", 1)[1], _fmt(a["candidates_before_graph"], 1),
              _fmt(a["candidates_after_graph"], 1), _fmt(a["graph_added_mean"], 1),
              _fmt(a["pool_recall_before_graph"]), _fmt(a["pool_recall_after_graph"]),
              _fmt(a["graph_ms_mean"], 2)] for a in rows]
    return ("Candidate-pool coverage (fraction of relevant functions present among candidates):\n\n" +
            _table(["variant", "cand. before", "cand. after", "added", "pool recall before",
                    "pool recall after", "graph ms"], table))


def _cost_section(rows: Sequence[Dict[str, Any]]) -> List[str]:
    sel = [a for a in rows if a["group"] in ("pipelines", "rerank")]
    if not sel:
        return []
    table = [[a["label"], _fmt(a["mrr"]), _fmt(a["ndcg_at_10"]), _fmt(a["build_s"], 2),
              _fmt(a["embed_s"], 2), _fmt(a["index_mb"], 2), _fmt(a["query_ms_mean"], 1),
              _fmt(a["query_ms_p95"], 1), _fmt(a["rerank_ms_mean"], 1), _fmt(a["rss_peak_mb"], 0),
              _fmt(a["gpu_peak_mb"], 0), _fmt(a["enrichment_total_tokens"], 0),
              _fmt(a["enrichment_seconds"], 0), _fmt(a["semantic_coverage"], 3)] for a in sel]
    return ["### Performance vs. cost", "",
            "`build s` is this run's wall time (small when the corpus and embeddings were cached); "
            "`embed s` is the original embedding compute time even on a cache hit. Enrichment tokens/seconds are the one-off LLM cost of producing "
            "the enriched JSON, not paid at query time.", "",
            _table(["run", "MRR", "nDCG@10", "build s", "embed s", "index MB", "ms/q mean",
                    "ms/q p95", "rerank ms", "RSS MB", "GPU MB", "enrich tokens", "enrich s",
                    "semantic coverage"], table), ""]


# --------------------------------------------------------------------- plots
def make_plots(out_dir: Path, aggregates: Sequence[Dict[str, Any]], comparisons: Sequence[Dict[str, Any]],
               runs: Sequence[Dict[str, Any]]) -> List[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not installed; skipping plots")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                         "font.size": 9})
    paths: List[Path] = []
    ok = [a for a in aggregates if a.get("status") in ("ok", "partial")]

    def tidy(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    for dataset in sorted({a["dataset"] for a in ok}):
        pipes = [a for a in ok if a["dataset"] == dataset and a["group"] == "pipelines"
                 and not a["rerank"]]
        # one embedding model only, to keep the chart to one entity per series
        if pipes:
            emb = sorted({a["embedding"] for a in pipes})[-1]
            pipes = [a for a in pipes if a["embedding"] == emb][:3]
            metrics = [("mrr", "MRR"), ("ndcg_at_10", "nDCG@10"), ("recall_at_10", "Recall@10")]
            fig, ax = plt.subplots(figsize=(6.4, 3.4))
            width = 0.26
            for i, a in enumerate(pipes):
                xs = np.arange(len(metrics)) + (i - (len(pipes) - 1) / 2) * (width + 0.02)
                vals = [a[m] or 0 for m, _ in metrics]
                ax.bar(xs, vals, width, color=SERIES[i], label=a["pipeline"], zorder=3)
                for x, v in zip(xs, vals):
                    ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
            ax.set_xticks(range(len(metrics)), [n for _, n in metrics])
            ax.set_ylim(0, 1)
            ax.set_title(f"{dataset}: pipeline comparison ({emb})", loc="left", fontsize=10)
            ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
            tidy(ax)
            path = out_dir / f"pipelines_{_safe(dataset)}.png"
            fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
            paths.append(path)

            cats = sorted({c for r in runs if r.get("metrics") and _dataset(r) == dataset
                           for c in r["metrics"]["by_category"]})
            labs = {a["label"] for a in pipes}
            sel = [r for r in runs if r.get("label") in labs and _dataset(r) == dataset and r.get("metrics")]
            if sel and cats:
                grid = np.array([[r["metrics"]["by_category"].get(c, {}).get("mrr") or 0.0 for c in cats]
                                 for r in sel])
                fig, ax = plt.subplots(figsize=(1.0 + 0.95 * len(cats), 1.2 + 0.5 * len(sel)))
                img = ax.imshow(grid, vmin=0, vmax=1, cmap="Blues", aspect="auto")
                ax.set_xticks(range(len(cats)), cats, rotation=35, ha="right")
                ax.set_yticks(range(len(sel)), [r["pipeline"] for r in sel])
                for (i, j), v in np.ndenumerate(grid):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.6 else INK)
                ax.set_title(f"{dataset}: MRR by query category", loc="left", fontsize=10)
                fig.colorbar(img, ax=ax, fraction=0.03)
                path = out_dir / f"categories_{_safe(dataset)}.png"
                fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
                paths.append(path)

        for group in sorted({c["group"] for c in comparisons if c["dataset"] == dataset}):
            rows = [c for c in comparisons if c["dataset"] == dataset and c["group"] == group
                    and c["metric"] == "mrr"]
            if not rows:
                continue
            rows = rows[:40]
            parts = [r["label"].split("/", 1)[1].split("|") for r in rows]
            keep = [i for i in range(1, max(len(x) for x in parts))
                    if len({x[i] if i < len(x) else "" for x in parts}) > 1]  # only varying fields
            names = ["|".join([x[0]] + [x[i] for i in keep if i < len(x)]) for x in parts]
            fig, ax = plt.subplots(figsize=(8.0, 0.28 * len(rows) + 1.4))
            ys = np.arange(len(rows))[::-1]
            for y, r in zip(ys, rows):
                ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=SERIES[0], linewidth=1.5, zorder=2)
                ax.plot(r["mean_difference"], y, "o", color=SERIES[0],
                        markerfacecolor=SERIES[0] if r["ci_excludes_zero"] else SURFACE,
                        markeredgewidth=1.5, markersize=6, zorder=3)
            ax.axvline(0, color=INK2, linewidth=0.8)
            ax.set_yticks(ys, names, fontsize=7)
            ax.set_xlabel("Δ MRR vs reference, 95% paired-bootstrap CI (filled dot = CI excludes 0)")
            ax.set_title(f"{dataset}: ablation '{group}'",
                         loc="left", fontsize=10)
            ax.grid(axis="x", color=GRID, linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            path = out_dir / f"ablation_{_safe(dataset)}_{group}.png"
            fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
            paths.append(path)

        pts = [a for a in ok if a["dataset"] == dataset and a["query_ms_mean"] and a["mrr"] is not None]
        if len(pts) >= 3:
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.scatter([a["query_ms_mean"] for a in pts], [a["mrr"] for a in pts], s=22,
                       color=SERIES[0], edgecolor=SURFACE, linewidth=1, zorder=3)
            ax.set_xscale("log")
            ax.set_xlabel("mean query latency (ms, log scale)")
            ax.set_ylabel("MRR")
            top = sorted(pts, key=lambda a: -a["mrr"])[:4] + sorted(pts, key=lambda a: -a["query_ms_mean"])[:2]
            for a in {id(x): x for x in top}.values():
                ax.annotate(a["label"].split("/", 1)[1][:34], (a["query_ms_mean"], a["mrr"]),
                            fontsize=6.5, xytext=(4, 3), textcoords="offset points", color=INK2)
            ax.set_title(f"{dataset}: quality vs. query cost", loc="left", fontsize=10)
            tidy(ax)
            path = out_dir / f"cost_vs_quality_{_safe(dataset)}.png"
            fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
            paths.append(path)
    return paths


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


# ------------------------------------------------------------------- driver
def write_suite_reports(store: ExperimentStore, suite_info: Dict[str, Any], specs: Iterable[Any] = (),
                        n_boot: int = 10000) -> Dict[str, Path]:
    set_references(specs)
    runs = load_runs(store)
    for run in runs:
        run["reference"] = run.get("reference") or _reference_of(run)
    aggregates = aggregate_rows(runs)
    comparisons = comparison_rows(runs, n_boot)
    out = store.suite_dir
    _write_csv(out / "aggregate.csv", aggregates)
    _write_csv(out / "per_category.csv", category_rows(runs))
    _write_csv(out / "per_query_metrics.csv", per_query_rows(runs))
    _write_csv(out / "comparisons.csv", comparisons)
    _write_csv(out / "comparisons_by_category.csv", category_comparison_rows(runs))
    plots = make_plots(out / "plots", aggregates, comparisons, runs)
    report = render_report(store.suite_dir.name, runs, aggregates, comparisons, suite_info, plots)
    (out / "report.md").write_text(report, encoding="utf-8")
    return {"report": out / "report.md", "aggregate": out / "aggregate.csv",
            "comparisons": out / "comparisons.csv"}
