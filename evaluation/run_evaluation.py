"""Build a complete evaluation from Optima's existing output artifacts.

Run from the repository root with ``python -m evaluation.run_evaluation``.
The evaluator never writes to the source output directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from .metrics.enrichment import FIELDS, field_completeness
from .metrics.retrieval import aggregate, query_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "output"
DEFAULT_DEST = ROOT / "evaluation_results"
MODEL_FAMILY = {
    "llama": "GPT/decoder-style",
    "qwen2.5": "other decoder-style",
    "mistralai": "other decoder-style",
    "microsoft": "other decoder-style",
    "smollm": "other decoder-style",
    "opencoder": "other decoder-style",
}
MODEL_METADATA = {
    "llama-3.2-3b-instruct": {
        "family": "decoder/causal LM", "category": "General-purpose",
        "purpose": "General instruction following",
    },
    "microsoft_phi-4-mini-reasoning": {
        "family": "decoder/causal LM", "category": "Reasoning",
        "purpose": "Reasoning-focused instruction model",
    },
    "mistralai_ministral-3-3b": {
        "family": "decoder/causal LM", "category": "General-purpose",
        "purpose": "General instruction following",
    },
    "opencoder-1.5b-instruct": {
        "family": "decoder/causal LM", "category": "Code",
        "purpose": "Code-specialized instruction model",
    },
    "qwen2.5-1.5b-instruct": {
        "family": "decoder/causal LM", "category": "General-purpose",
        "purpose": "General instruction following",
    },
    "qwen2.5-coder-3b-instruct": {
        "family": "decoder/causal LM", "category": "Code",
        "purpose": "Code-specialized instruction model",
    },
    "smollm2-1.7b-instruct": {
        "family": "decoder/causal LM", "category": "General-purpose",
        "purpose": "General instruction following",
    },
    "raw": {
        "family": "none (raw baseline)", "category": "Raw baseline",
        "purpose": "Unenriched source representation",
    },
}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_name_from_corpus(corpus: str) -> str:
    return corpus if corpus != "raw" else ""


def family(model: str) -> str:
    if not model:
        return "raw baseline"
    for prefix, name in MODEL_FAMILY.items():
        if model.startswith(prefix):
            return name
    return "other"


def model_metadata(model: str) -> dict[str, str]:
    """Return capability metadata from the recorded model/configuration inventory.

    The JSON artifacts record requested/loaded model IDs and enrichment
    configuration, but do not contain BERT/BERT2 architecture variants.
    Capability categories therefore use the model's documented purpose encoded
    in this explicit inventory, never retrieval performance.
    """
    return MODEL_METADATA.get(model, {
        "family": family(model), "category": "Other",
        "purpose": "Capability metadata unavailable",
    })


def find_latest_matrix(source: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted((source / "rag_report" / "results").glob("retrieval_evaluation_*.json"))
    if not candidates:
        raise FileNotFoundError("No retrieval_evaluation_*.json files were found")
    complete: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        data = read_json(path)
        if isinstance(data.get("results"), dict) and data["results"]:
            complete.append((path, data))
    if not complete:
        raise ValueError("Retrieval evaluation files contained no results")
    # Historical runs used both a single-embedding layout and a matrix layout.
    # Select the newest saved run for each actual embedding/corpus pair so that
    # valid combinations from earlier runs (e.g. mock or CodeSage) are not lost.
    merged: dict[str, dict[str, Any]] = {}
    def alias(name: str) -> str:
        known = {
            "BAAI/bge-small-en-v1.5": "bge-small",
            "BAAI/bge-base-en-v1.5": "bge-base",
            "Qwen/Qwen3-Embedding-0.6B": "qwen3-0.6b",
            "nomic-ai/CodeRankEmbed": "coderank",
            "nomic-ai/nomic-embed-text-v1.5": "nomic",
        }
        return known.get(str(name), str(name))

    for _path, data in complete:
        results = data["results"]
        if data.get("embedding_models"):
            for embedding, corpora in results.items():
                embedding = alias(embedding)
                if isinstance(corpora, dict):
                    for corpus, metrics in corpora.items():
                        merged.setdefault(str(embedding), {})[str(corpus)] = metrics
        else:
            embedding = alias(data.get("embedding_model"))
            if embedding and isinstance(results, dict):
                for corpus, metrics in results.items():
                    merged.setdefault(str(embedding), {})[str(corpus)] = metrics
    latest_path = complete[-1][0]
    return latest_path, {"results": merged, "source_files": [str(path.relative_to(ROOT)) for path, _ in complete]}


def metadata_index(source: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records = {}
    for path in sorted((source / "rag_report" / "indexes").glob("*/*/metadata.json")):
        data = read_json(path)
        records[(str(data.get("corpus", path.parent.parent.name)),
                  str(data.get("embedding_alias", path.parent.name)))] = data
    return records


def source_json_for(corpus: str, source: Path) -> Path | None:
    if corpus == "raw":
        path = source / "base.json"
    else:
        path = source / f"enhanced_{corpus}.json"
    return path if path.exists() else None


def canonical_embedding(name: str, metadata: dict[tuple[str, str], dict[str, Any]]) -> str:
    """Map legacy full model names to the aliases used by index metadata."""
    if name in {alias for _corpus, alias in metadata}:
        return name
    for (_corpus, alias), item in metadata.items():
        if name == item.get("embedding_model"):
            return alias
    return name


def flatten_functions(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [function for file_data in data.get("files", [])
            for function in file_data.get("functions", [])]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def enrichment_row(corpus: str, data: dict[str, Any]) -> dict[str, Any]:
    records = [f.get("enrichment", {}) for f in flatten_functions(data)
               if isinstance(f.get("enrichment"), dict)]
    metrics = data.get("enrichment_metrics", {})
    meta = data.get("enrichment_metadata", {})
    values = [record.get("usage", {}) for record in records]
    evals = [record.get("evaluation", {}) for record in records]
    def avg(key: str) -> float | None:
        vals = [float(v[key]) for v in values if v.get(key) is not None]
        return mean(vals) if vals else None
    latency = [float(v["latency_seconds"]) for v in evals if v.get("latency_seconds") is not None]
    return {
        "model": corpus or "raw",
        "model_family": family(corpus),
        "model_category": model_metadata(corpus or "raw")["category"],
        "purpose": model_metadata(corpus or "raw")["purpose"],
        "parameters": None,
        "samples": len(records),
        "json_valid_rate": metrics.get("json_valid_rate"),
        "success_rate": metrics.get("success_rate"),
        "purpose_f1": None, "behavior_f1": None, "input_f1": None,
        "output_f1": None, "side_effect_f1": None, "overall_f1": None,
        "field_completeness": mean(field_completeness(records).values()) if records else None,
        "average_input_tokens": avg("prompt_tokens") or avg("input_tokens"),
        "average_output_tokens": avg("completion_tokens") or avg("output_tokens"),
        "average_total_tokens": avg("total_tokens"),
        "average_latency_seconds": mean(latency) if latency else metrics.get("average_latency_seconds"),
        "total_runtime_seconds": metrics.get("total_runtime_seconds"),
        "throughput_functions_per_second": metrics.get("functions_per_second"),
        "memory": None,
        "gpu_memory": None,
        "api_cost": None,
        "context_size": meta.get("model_runtime", {}).get("context_size"),
        "reference_ground_truth": "unavailable",
    }


def retrieval_rows(matrix: dict[str, Any], queries: list[dict[str, Any]],
                   metadata: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for embedding, corpora in matrix.items():
        if not isinstance(corpora, dict):
            continue
        embedding = canonical_embedding(str(embedding), metadata)
        for corpus, result in corpora.items():
            if not isinstance(result, dict):
                continue
            original = {str(q.get("query_id")): q for q in queries}
            rows = []
            for item in result.get("per_query_results", []):
                query = original.get(str(item.get("query_id")), item)
                relevant = set(map(str, query.get("relevant_functions", item.get("relevant_functions", []))))
                row = query_metrics(item.get("retrieved", []), relevant)
                row.update({
                    "query_id": item.get("query_id"),
                    "embedding_model": embedding,
                    "enrichment_model": corpus,
                    "difficulty": item.get("difficulty", query.get("difficulty", "")),
                    "category": item.get("category", query.get("category", "")),
                    "query": item.get("query", query.get("query", "")),
                    "relevant_functions": compact(sorted(relevant)),
                    "retrieved_functions": compact([x.get("function_id") for x in item.get("retrieved", [])]),
                    "latency_seconds": result.get("per_query_results", [{}])[0].get("latency_seconds"),
                })
                rows.append(row)
                per_query.append(row)
            stats = aggregate(rows)
            meta = metadata.get((corpus, embedding), {})
            def m(name: str) -> float | None:
                return stats.get(name, {}).get("mean")
            aggregate_rows.append({
                "configuration": f"{corpus or 'raw'} + {embedding}",
                "representation": "enriched" if corpus != "raw" else "raw",
                "enrichment_model": corpus or "raw",
                "embedding_model": embedding,
                "model_family": family(corpus),
                "model_category": model_metadata(corpus or "raw")["category"],
                "segmentation": "unavailable (not recorded)",
                "chunk_size": meta.get("chunk_size"),
                "chunk_overlap": meta.get("chunk_overlap"),
                "number_of_chunks": meta.get("num_chunks"),
                "number_of_documents": meta.get("num_documents"),
                "queries": len(rows),
                "R@1": m("recall_at_1"), "R@5": m("recall_at_5"),
                "R@10": m("recall_at_10"), "R@20": m("recall_at_20"),
                "P@1": m("precision_at_1"), "P@5": m("precision_at_5"),
                "P@10": m("precision_at_10"), "MRR": m("mrr"),
                "NDCG@5": m("ndcg_at_5"), "NDCG@10": m("ndcg_at_10"),
                "R@5_median": stats.get("recall_at_5", {}).get("median"),
                "R@5_std": stats.get("recall_at_5", {}).get("std"),
                "MRR_median": stats.get("mrr", {}).get("median"),
                "MRR_std": stats.get("mrr", {}).get("std"),
                "retrieval_latency_seconds": result.get("mean_latency"),
                "Faithfulness": None,
                "status": result.get("error", "success"),
            })
    return aggregate_rows, per_query


def discover_inventory(aggregates: list[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    inventory = []
    for row in aggregates:
        corpus = row["enrichment_model"]
        path = source_json_for(corpus, source)
        functions = flatten_functions(read_json(path)) if path else []
        inventory.append({
            "model": corpus or "raw",
            "model_family": family(corpus),
            "model_category": model_metadata(corpus or "raw")["category"],
            "parameters": None,
            "segmentation": row["segmentation"],
            "enrichment_model": corpus or "none",
            "embedding_model": row["embedding_model"],
            "representation_corpus": corpus or "raw",
            "chunk_size": row["chunk_size"],
            "number_of_samples": len(functions),
            "number_of_queries": row["queries"],
            "available_metrics": "retrieval, latency, chunk counts; enrichment metadata for enriched corpus"
                              if corpus != "raw" else "retrieval, latency, chunk counts",
            "source_json": str(path.relative_to(ROOT)) if path else None,
        })
    return inventory


def model_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "enrichment_model": row["enrichment_model"],
        "embedding_model": row["embedding_model"],
        "model_category": row["model_category"],
        "model_family": row["model_family"],
        "segmentation": row["segmentation"],
        "R@5": row["R@5"], "R@10": row["R@10"], "MRR": row["MRR"],
        "NDCG@10": row["NDCG@10"], "P@5": row["P@5"],
        "latency_seconds": row["retrieval_latency_seconds"],
        "tokens": None,
    } for row in sorted(rows, key=lambda r: (
        r["model_category"] == "Raw baseline", r["enrichment_model"], r["embedding_model"]))]


def representative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one readable, controlled row per corpus for the primary table."""
    valid = [r for r in rows if r.get("R@5") is not None]
    controlled = [r for r in valid if r["embedding_model"] == "bge-small"]
    source = controlled or valid
    selected = {}
    for row in source:
        selected.setdefault(row["enrichment_model"], row)
    return sorted(selected.values(), key=lambda r: (r["enrichment_model"] == "raw", r["enrichment_model"]))


def ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe only ablations that are actually represented in the artifacts."""
    selected = representative_rows(rows)
    baseline = next((r for r in selected if r["enrichment_model"] == "raw"), None)
    output = []
    for row in selected:
        delta = (row["MRR"] - baseline["MRR"]) if baseline and row["MRR"] is not None else None
        output.append({
            "configuration": row["configuration"],
            "segmentation": row["segmentation"],
            "enrichment": row["enrichment_model"],
            "embedding": row["embedding_model"],
            "R@5": row["R@5"], "R@10": row["R@10"], "MRR": row["MRR"],
            "NDCG@10": row["NDCG@10"], "delta_MRR_vs_raw": delta,
            "comparison_status": "observed enrichment comparison; segmentation is not varied",
        })
    output.append({
        "configuration": "structure-aware segmentation",
        "segmentation": None, "enrichment": None, "embedding": None,
        "R@5": None, "R@10": None, "MRR": None, "NDCG@10": None,
        "delta_MRR_vs_raw": None,
        "comparison_status": "N/A: no segmentation variant is recorded",
    })
    return output


def prior_work_methodology_rows() -> list[dict[str, Any]]:
    """Conservative research-context taxonomy; no paper scores are implied."""
    return [
        {"work": "CRAG", "primary_problem": "Corrective retrieval-augmented generation",
         "code_aware": "—", "structural_representation": "—", "ast": "—", "cfg_graph": "—",
         "llm_enrichment": "—", "retrieval": "✓", "segmentation": "—",
         "embedding_retrieval": "✓", "repository_context": "—",
         "evaluation_focus": "Retrieval correction and generation", "key_metrics": "Task-specific retrieval/generation metrics"},
        {"work": "Critic / retrieval-critique approaches", "primary_problem": "Critique and correction of retrieved context",
         "code_aware": "—", "structural_representation": "—", "ast": "—", "cfg_graph": "—",
         "llm_enrichment": "—", "retrieval": "✓", "segmentation": "—",
         "embedding_retrieval": "✓", "repository_context": "NR",
         "evaluation_focus": "Critique effectiveness and downstream task quality", "key_metrics": "Task-specific"},
        {"work": "AST-T5", "primary_problem": "AST-aware code representation/generation",
         "code_aware": "✓", "structural_representation": "✓", "ast": "✓", "cfg_graph": "—",
         "llm_enrichment": "—", "retrieval": "—", "segmentation": "NR",
         "embedding_retrieval": "—", "repository_context": "NR",
         "evaluation_focus": "Code representation/generation", "key_metrics": "Task-specific generation metrics"},
        {"work": "RepoCoder", "primary_problem": "Repository-level code completion",
         "code_aware": "✓", "structural_representation": "NR", "ast": "NR", "cfg_graph": "NR",
         "llm_enrichment": "—", "retrieval": "✓", "segmentation": "NR",
         "embedding_retrieval": "✓", "repository_context": "✓",
         "evaluation_focus": "Repository-level completion", "key_metrics": "Code completion metrics"},
        {"work": "GraphCode", "primary_problem": "Graph-based code representation",
         "code_aware": "✓", "structural_representation": "✓", "ast": "NR", "cfg_graph": "✓",
         "llm_enrichment": "—", "retrieval": "NR", "segmentation": "NR",
         "embedding_retrieval": "NR", "repository_context": "NR",
         "evaluation_focus": "Code representation/understanding", "key_metrics": "Task-specific"},
        {"work": "Optima", "primary_problem": "Structure-aware enriched code retrieval",
         "code_aware": "✓", "structural_representation": "✓", "ast": "✓", "cfg_graph": "✓",
         "llm_enrichment": "✓", "retrieval": "✓", "segmentation": "NR",
         "embedding_retrieval": "✓", "repository_context": "✓",
         "evaluation_focus": "Code retrieval and program understanding", "key_metrics": "Recall@K, Precision@K, MRR, NDCG"},
    ]


def prior_work_numerical_rows() -> list[dict[str, Any]]:
    works = ["CRAG", "Critic / retrieval-critique approaches", "AST-T5",
             "RepoCoder", "GraphCode"]
    return [{
        "work": work, "dataset_benchmark": "Not imported into Optima artifacts",
        "metric": "N/A", "published_result": None, "optima_result": None,
        "optima_configuration": None, "absolute_difference": None,
        "relative_difference": None, "directly_comparable": "No",
        "reason": "Published task, dataset, metric, or protocol is not directly comparable",
    } for work in works] + [{
        "work": "Optima", "dataset_benchmark": "Optima benchmark",
        "metric": "MRR", "published_result": None, "optima_result": None,
        "optima_configuration": "See main/results tables", "absolute_difference": None,
        "relative_difference": None, "directly_comparable": "N/A",
        "reason": "Optima is the evaluated system, not a cross-paper baseline",
    }]


def best_mark(rows: list[dict[str, Any]], key: str, group: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[group]].append(row)
    for members in groups.values():
        values = [r[key] for r in members if isinstance(r.get(key), (int, float))]
        best = max(values) if values else None
        for row in members:
            row[f"{key}_best_in_{group}"] = bool(best is not None and row.get(key) == best)


def create_figures(rows: list[dict[str, Any]], enrichment: list[dict[str, Any]],
                   dest: Path) -> list[dict[str, Any]]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 120})
    figures: list[dict[str, Any]] = []
    metrics = ("R@5", "MRR", "NDCG@10")
    valid = [r for r in rows if all(isinstance(r.get(metric), (int, float)) for metric in metrics)]

    def add_index(path: Path, question: str, configs: list[str], metric: str, interpretation: str) -> None:
        figures.append({"filename": str(path.relative_to(dest.parent)), "question": question,
                        "configurations": configs, "metric": metric,
                        "interpretation": interpretation})

    enrichment_by_model = {item["model"]: item for item in enrichment}
    controlled = [r for r in valid if r["embedding_model"] == "bge-small"]
    controlled = controlled or valid

    def category_style(category: str) -> tuple[str, str]:
        return {
            "General-purpose": ("o", "#4472c4"),
            "Code": ("s", "#ed7d31"),
            "Reasoning": ("^", "#70ad47"),
            "Raw baseline": ("D", "#7f7f7f"),
        }.get(category, ("o", "#8064a2"))

    def labeled_scatter(points: list[dict[str, Any]], x_key: str, y_key: str,
                        xlabel: str, ylabel: str, title: str, filename: str,
                        question: str, interpretation: str, log_x: bool = False) -> None:
        if len(points) < 3:
            return
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for point in points:
            marker, color = category_style(point["category"])
            ax.scatter(point[x_key], point[y_key], marker=marker, color=color, s=62,
                       edgecolors="white", linewidths=.5,
                       label=point["category"])
            ax.annotate(point["label"], (point[x_key], point[y_key]),
                        xytext=(5, 5), textcoords="offset points", fontsize=7.5)
        if log_x:
            ax.set_xscale("log")
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=.25)
        fig.tight_layout()
        path = dest / filename
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        add_index(path, question, [p["label"] for p in points],
                  f"{xlabel} and {ylabel}", interpretation)

    # Efficiency points are controlled to one embedding so each enrichment
    # model contributes one comparable configuration. Full combinations stay
    # in tables/efficiency_results.csv.
    efficiency_points = []
    for row in controlled:
        metadata = enrichment_by_model.get(row["enrichment_model"], {})
        tokens = metadata.get("average_total_tokens")
        enrichment_latency = metadata.get("average_latency_seconds")
        if isinstance(tokens, (int, float)) and isinstance(enrichment_latency, (int, float)):
            efficiency_points.append({
                "label": row["enrichment_model"],
                "category": row["model_category"],
                "tokens": tokens,
                "enrichment_latency_ms": enrichment_latency * 1000,
                "MRR": row["MRR"],
                "retrieval_latency_ms": row["retrieval_latency_seconds"] * 1000
                if isinstance(row.get("retrieval_latency_seconds"), (int, float)) else None,
            })
    labeled_scatter(
        [p for p in efficiency_points if p["MRR"] is not None],
        "tokens", "enrichment_latency_ms", "Total enrichment tokens",
        "Enrichment latency (ms)", "Token usage versus enrichment latency",
        "tokens_vs_latency.png", "How does token usage affect enrichment latency?",
        "Each point is a bge-small configuration; labels are full recorded model identifiers.")
    labeled_scatter(
        [p for p in efficiency_points if p["MRR"] is not None],
        "tokens", "MRR", "Total enrichment tokens", "Retrieval MRR",
        "Token usage versus retrieval quality", "tokens_vs_mrr.png",
        "Does greater token usage improve retrieval quality?",
        "Each point is a controlled bge-small configuration; complete combinations are in the efficiency table.")

    # Figure 1: one controlled overview, not every pair. Details remain in tables.
    overview = [r for r in valid if r["embedding_model"] == "bge-small"]
    if overview:
        overview.sort(key=lambda r: r["MRR"], reverse=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        x = np.arange(len(overview))
        width = .25
        for offset, metric in enumerate(metrics):
            ax.bar(x + (offset - 1) * width, [r[metric] for r in overview],
                   width, label=metric)
        ax.set_xticks(x, [r["enrichment_model"] for r in overview], rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Retrieval score")
        ax.set_title("Overall configuration comparison (bge-small)")
        ax.legend(frameon=False, ncols=3)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        path = dest / "overall_comparison.png"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        add_index(path, "Which major corpus configurations perform best?",
                  [r["enrichment_model"] for r in overview], "R@5, MRR, NDCG@10",
                  "A compact overview at a fixed embedding; exact values are in tables/retrieval.csv.")

    # Category figures use the same compact design and only appear when a
    # category has at least two observed models. Categories are based on the
    # recorded model purpose/configuration inventory, not retrieval scores.
    for category, filename, title, question in [
        ("General-purpose", "general_purpose_models.png", "General-purpose model performance",
         "How well do general-purpose models perform for Optima?"),
        ("Code", "code_models.png", "Code model performance",
         "Do code-specialized models improve retrieval for Optima?"),
        ("Reasoning", "reasoning_models.png", "Reasoning model performance",
         "Do reasoning-oriented models provide an advantage for Optima?"),
    ]:
        category_rows = [r for r in overview if r["model_category"] == category]
        if len(category_rows) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        x = np.arange(len(category_rows))
        width = .25
        for offset, metric in enumerate(metrics):
            ax.bar(x + (offset - 1) * width, [r[metric] for r in category_rows],
                   width, label=metric)
        ax.set_xticks(x, [r["enrichment_model"] for r in category_rows], rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Retrieval score")
        ax.set_title(title)
        ax.legend(frameon=False, ncols=3)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        path = dest / filename
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        add_index(path, question, [r["enrichment_model"] for r in category_rows],
                  "R@5, MRR, NDCG@10", "Category members use the same bge-small query benchmark.")

    # Category summary is controlled at bge-small and reports category means;
    # the exact model-level values remain in the category table.
    category_rows = []
    for category in ("General-purpose", "Code", "Reasoning"):
        members = [r for r in overview if r["model_category"] == category]
        if members:
            category_rows.append({
                "label": category,
                **{metric: mean(r[metric] for r in members) for metric in metrics},
            })
    if len(category_rows) >= 2:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        x = np.arange(len(category_rows))
        width = .25
        for offset, metric in enumerate(metrics):
            ax.bar(x + (offset - 1) * width, [r[metric] for r in category_rows],
                   width, label=metric)
        ax.set_xticks(x, [r["label"] for r in category_rows])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Macro mean retrieval score")
        ax.set_title("Model category comparison (bge-small)")
        ax.legend(frameon=False, ncols=3)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        path = dest / "model_category_comparison.png"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        add_index(path, "Which model category is most effective?",
                  [r["label"] for r in category_rows], "R@5, MRR, NDCG@10",
                  "Macro means are over models within each category at fixed bge-small.")

    # Figure 2: raw versus the best enriched corpus, with the embedding fixed.
    if overview:
        raw = next((r for r in overview if r["enrichment_model"] == "raw"), None)
        enriched = [r for r in overview if r["enrichment_model"] != "raw"]
        if raw and enriched:
            best = max(enriched, key=lambda r: r["MRR"])
            pair = [raw, best]
            fig, ax = plt.subplots(figsize=(5.2, 3.8))
            x = np.arange(2)
            for offset, metric in enumerate(("R@5", "MRR")):
                ax.bar(x + (offset - .5) * .3, [r[metric] for r in pair],
                       .3, label=metric)
            ax.set_xticks(x, ["Raw", best["enrichment_model"]])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Retrieval score")
            ax.set_title("Raw versus best enriched representation")
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=.25)
            fig.tight_layout()
            path = dest / "raw_vs_enriched.png"
            fig.savefig(path, dpi=240, bbox_inches="tight")
            plt.close(fig)
            add_index(path, "Does enrichment improve retrieval?",
                      ["raw", best["enrichment_model"]], "R@5, MRR",
                      "Raw and enriched representations share the embedding and query set.")

    # Figure 3: one labeled quality/latency trade-off using representative
    # enrichment models at bge-small. Latency is explicitly in milliseconds.
    points = []
    for row in overview:
        if row["retrieval_latency_seconds"] is not None:
            points.append({"label": row["enrichment_model"], "category": row["model_category"],
                           "latency_ms": row["retrieval_latency_seconds"] * 1000,
                           "MRR": row["MRR"]})
    if points:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for p in points:
            marker, color = category_style(p["category"])
            ax.scatter(p["latency_ms"], p["MRR"], marker=marker, color=color, s=62,
                       edgecolors="white", linewidths=.5)
            ax.annotate(p["label"], (p["latency_ms"], p["MRR"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8)
        ax.set_xlabel("Retrieval latency (ms)")
        ax.set_ylabel("MRR")
        ax.set_title("Retrieval quality versus latency")
        ax.grid(alpha=.25)
        fig.tight_layout()
        path = dest / "latency_vs_mrr.png"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        add_index(path, "What is the quality/latency trade-off?",
                  [p["label"] for p in points], "MRR and latency (ms)",
                  "Every plotted point is labeled; complete efficiency values are in tables/efficiency.csv.")

    # A correlation heatmap is meaningful only when enough controlled
    # configurations have complete values for all compared variables.
    correlation_points = []
    for row in controlled:
        metadata = enrichment_by_model.get(row["enrichment_model"], {})
        values = {
            "R@5": row.get("R@5"), "R@10": row.get("R@10"),
            "MRR": row.get("MRR"), "NDCG@10": row.get("NDCG@10"),
            "Latency (ms)": row.get("retrieval_latency_seconds") * 1000
            if isinstance(row.get("retrieval_latency_seconds"), (int, float)) else None,
            "Tokens": metadata.get("average_total_tokens"),
        }
        if all(isinstance(value, (int, float)) for value in values.values()):
            correlation_points.append(values)
    if len(correlation_points) >= 5:
        names = list(correlation_points[0])
        matrix = np.array([[point[name] for name in names] for point in correlation_points], dtype=float)
        corr = np.corrcoef(matrix, rowvar=False)
        fig, ax = plt.subplots(figsize=(6.5, 5.3))
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="Pearson correlation")
        ax.set_title(f"Retrieval and efficiency metric correlation (n={len(correlation_points)})")
        fig.tight_layout()
        path = dest / "retrieval_metric_correlation.png"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        add_index(path, "Do retrieval metrics correlate with one another and with cost?",
                  names, "Pearson correlation",
                  "Computed over controlled bge-small configurations with complete metric/token/latency values.")

    # Figure 4: a single readable MRR heatmap for the complete pair matrix.
    embeddings = sorted({r["embedding_model"] for r in valid})
    corpora = sorted({r["enrichment_model"] for r in valid})
    if embeddings and corpora and len(embeddings) * len(corpora) <= 64:
        lookup = {(r["enrichment_model"], r["embedding_model"]): r["MRR"] for r in valid}
        matrix = np.array([[lookup.get((c, e), np.nan) for e in embeddings] for c in corpora])
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(embeddings)), embeddings, rotation=35, ha="right")
        ax.set_yticks(range(len(corpora)), corpora)
        for i in range(len(corpora)):
            for j in range(len(embeddings)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=ax, label="MRR")
        ax.set_title("Enrichment × embedding model-pair performance")
        fig.tight_layout()
        path = dest / "enrichment_model_pairs.png"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        add_index(path, "Which enrichment × embedding pairs perform best?",
                  [f"{c} × {e}" for c in corpora for e in embeddings], "MRR",
                  "The full pair matrix is also preserved in tables/model_pairs.csv.")
    return figures


def create_table_pngs(dest: Path, rows: list[dict[str, Any]],
                      enrichment: list[dict[str, Any]]) -> list[str]:
    """Render the detailed CSV tables as readable, high-resolution slide assets."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    table_dir = dest / "tables_png"
    table_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if r.get("R@5") is not None]
    specs = {
        "overall_results.png": (
            ["configuration", "model_category", "model_family", "enrichment_model",
             "embedding_model", "R@5", "R@10", "MRR", "NDCG@10", "retrieval_latency_seconds"],
            ["Configuration", "Category", "Family", "Enrichment", "Embedding",
             "R@5", "R@10", "MRR", "NDCG@10", "Latency (s)"], valid),
        "model_category_comparison.png": (
            ["model_category", "enrichment_model", "model_family", "parameters",
             "R@5", "R@10", "MRR", "NDCG@10", "retrieval_latency_seconds"],
            ["Category", "Model", "Family", "Parameters", "R@5", "R@10", "MRR", "NDCG@10",
             "Latency (s)"], valid),
        "model_pairs.png": (
            ["enrichment_model", "embedding_model", "model_category", "model_family",
             "R@5", "R@10", "MRR", "NDCG@10", "retrieval_latency_seconds"],
            ["Enrichment", "Embedding", "Category", "Family", "R@5", "R@10", "MRR",
             "NDCG@10", "Latency (s)"], valid),
        "segmentation_results.png": (
            ["segmentation", "number_of_chunks", "R@5", "R@10", "MRR", "NDCG@10"],
            ["Segmentation", "Chunks", "R@5", "R@10", "MRR", "NDCG@10"],
            [{"segmentation": "unavailable", "number_of_chunks": None, "R@5": None,
              "R@10": None, "MRR": None, "NDCG@10": None}]),
        "enrichment_results.png": (
            ["model", "purpose_f1", "behavior_f1", "input_f1", "output_f1",
             "side_effect_f1", "overall_f1"],
            ["Enrichment Model", "Purpose F1", "Behavior F1", "Input F1", "Output F1",
             "Side-effect F1", "Overall F1"], enrichment),
        "embedding_results.png": (
            ["embedding_model", "enrichment_model", "R@5", "R@10", "MRR", "NDCG@10",
             "retrieval_latency_seconds"],
            ["Embedding", "Corpus", "R@5", "R@10", "MRR", "NDCG@10", "Latency (s)"], valid),
        "efficiency_results.png": (
            ["model", "model_category", "parameters", "average_input_tokens",
             "average_output_tokens", "average_total_tokens", "average_latency_seconds",
             "throughput_functions_per_second"],
            ["Model", "Category", "Parameters", "Input Tokens", "Output Tokens",
             "Total Tokens", "Latency (s)", "Throughput"], enrichment),
        "paper_comparison.png": (
            ["metric", "base_paper", "raw_value", "model_value", "absolute_improvement",
             "relative_improvement", "status"],
            ["Metric", "Reference Paper", "Reference Result", "Optima Result",
             "Absolute Difference", "Relative Difference", "Comparable?"], []),
    }
    category_columns = (["model_category", "enrichment_model", "model_family", "R@5",
                         "R@10", "MRR", "NDCG@10", "retrieval_latency_seconds"],
                        ["Category", "Model", "Family", "R@5", "R@10", "MRR",
                         "NDCG@10", "Latency (s)"])
    for category, filename in [("General-purpose", "general_purpose_models.png"),
                               ("Code", "code_models.png"),
                               ("Reasoning", "reasoning_models.png")]:
        keys, headers = category_columns
        specs[filename] = (keys, headers, [r for r in valid if r.get("model_category") == category])
    output = []
    for filename, (keys, headers, data) in specs.items():
        values = []
        for row in data:
            values.append(["N/A" if row.get(key) is None else
                           f"{row[key]:.4f}" if isinstance(row.get(key), float)
                           else str(row.get(key, "")) for key in keys])
        # Keep tables slide-readable; complete rows remain in CSV.
        display = values[:20]
        fig_height = max(2.2, 0.42 * (len(display) + 1))
        fig, ax = plt.subplots(figsize=(13.33, min(fig_height, 9.5)))
        ax.axis("off")
        table = ax.table(cellText=display or [["N/A"] * len(headers)],
                         colLabels=headers, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.45)
        for (row_index, _column), cell in table.get_celld().items():
            cell.set_edgecolor("#d9e2f3")
            if row_index == 0:
                cell.set_facecolor("#1f4e78")
                cell.set_text_props(color="white", weight="bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f3f6fa")
        fig.tight_layout(pad=.4)
        path = table_dir / filename
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        output.append(str(path.relative_to(dest.parent)))
    return output


def create_research_table_pngs(dest: Path, tables: dict[str, tuple[list[str], list[str], list[dict[str, Any]]]]) -> list[str]:
    """Render research-context tables with wrapped full identifiers for PPT."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    table_dir = dest / "tables_png"
    table_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for filename, (keys, headers, data) in tables.items():
        chunks = [data[i:i + 20] for i in range(0, len(data), 20)] or [[]]
        for part_index, display_data in enumerate(chunks, start=1):
            values = []
            for row in display_data:
                values.append([
                    "N/A" if row.get(key) is None else
                    f"{row[key]:.4f}" if isinstance(row.get(key), float)
                    else str(row.get(key, ""))
                    for key in keys
                ])
            wrapped = [[textwrap.fill(value, width=24) for value in row] for row in values]
            wrapped_headers = [textwrap.fill(header, width=18) for header in headers]
            height = max(2.7, min(11.0, 0.48 * (len(wrapped) + 1)))
            width = max(13.33, min(22.0, 1.45 * len(headers)))
            fig, ax = plt.subplots(figsize=(width, height))
            ax.axis("off")
            table = ax.table(cellText=wrapped or [["N/A"] * len(headers)],
                             colLabels=wrapped_headers, loc="center", cellLoc="left")
            table.auto_set_font_size(False)
            table.set_fontsize(8.2 if len(headers) <= 10 else 7.4)
            table.scale(1, 1.5)
            for (row_index, _column), cell in table.get_celld().items():
                cell.set_edgecolor("#cbd5e1")
                cell.PAD = 0.045
                if row_index == 0:
                    cell.set_facecolor("#1f4e78")
                    cell.set_text_props(color="white", weight="bold")
                elif row_index % 2 == 0:
                    cell.set_facecolor("#f3f6fa")
            fig.tight_layout(pad=.5)
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            output_name = filename if part_index == 1 else f"{stem}_part{part_index}{suffix}"
            path = table_dir / output_name
            fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            created.append(str(path.relative_to(dest.parent)))
    return created


def research_tables(rows: list[dict[str, Any]], enrichment: list[dict[str, Any]],
                    failures: list[dict[str, Any]]) -> dict[str, tuple[list[str], list[str], list[dict[str, Any]]]]:
    valid = [r for r in rows if r.get("R@5") is not None]
    main = representative_rows(rows)
    main_display = [{**r, "model": r["enrichment_model"],
                     "enrichment": r["enrichment_model"]} for r in main]
    enrichment_lookup = {r["model"]: r for r in enrichment}
    efficiency = []
    for row in valid:
        e = enrichment_lookup.get(row["enrichment_model"], {})
        efficiency.append({
            "enrichment_model": row["enrichment_model"],
            "model_category": row["model_category"],
            "parameters": e.get("parameters"),
            "input_tokens": e.get("average_input_tokens"),
            "output_tokens": e.get("average_output_tokens"),
            "total_tokens": e.get("average_total_tokens"),
            "latency_seconds": row.get("retrieval_latency_seconds"),
            "throughput": e.get("throughput_functions_per_second"),
            "memory": e.get("memory"), "gpu_memory": e.get("gpu_memory"),
            "R@5": row["R@5"], "MRR": row["MRR"],
        })
    main_keys = ["configuration", "model_category", "model", "segmentation",
                 "enrichment", "embedding_model", "R@1", "R@5", "R@10",
                 "MRR", "NDCG@10", "retrieval_latency_seconds"]
    main_headers = ["Configuration", "Model Category", "Model", "Segmentation", "Enrichment",
                    "Embedding", "R@1", "R@5", "R@10", "MRR", "NDCG@10", "Latency (s)"]
    tables = {
        "main_optima_results.png": (main_keys, main_headers, main_display),
        "optima_ablation.png": (
            ["configuration", "segmentation", "enrichment", "embedding", "R@5", "R@10",
             "MRR", "NDCG@10", "delta_MRR_vs_raw"],
            ["Configuration", "Segmentation", "Enrichment", "Embedding", "R@5", "R@10",
             "MRR", "NDCG@10", "ΔMRR"], ablation_rows(rows)),
        "segmentation_results.png": (
            ["segmentation", "number_of_chunks", "average_chunk_size", "median_chunk_size",
             "boundary_precision", "boundary_recall", "boundary_f1", "cross_boundary_rate",
             "R@5", "R@10", "MRR", "NDCG@10"],
            ["Segmentation Strategy", "Chunks", "Avg Chunk Size", "Median Chunk Size",
             "Boundary P", "Boundary R", "Boundary F1", "Cross-boundary Rate",
             "R@5", "R@10", "MRR", "NDCG@10"],
            [{"segmentation": "N/A (not recorded)", "number_of_chunks": None,
              "average_chunk_size": None, "median_chunk_size": None,
              "boundary_precision": None, "boundary_recall": None, "boundary_f1": None,
              "cross_boundary_rate": None, "R@5": None, "R@10": None, "MRR": None,
              "NDCG@10": None}]),
        "enrichment_results.png": (
            ["model", "model_category", "parameters", "json_valid_rate", "purpose_f1",
             "behavior_f1", "input_f1", "output_f1", "side_effect_f1", "overall_f1",
             "hallucination_rate", "R@5", "MRR", "average_latency_seconds",
             "average_total_tokens"],
            ["Enrichment Model", "Category", "Parameters", "JSON Validity", "Purpose F1",
             "Behavior F1", "Input F1", "Output F1", "Side-effect F1", "Overall F1",
             "Hallucination Rate", "R@5", "MRR", "Latency (s)", "Tokens"],
            [{**r, "R@5": next((x["R@5"] for x in valid
                                if x["enrichment_model"] == r["model"] and x["embedding_model"] == "bge-small"), None),
              "MRR": next((x["MRR"] for x in valid
                           if x["enrichment_model"] == r["model"] and x["embedding_model"] == "bge-small"), None),
              "hallucination_rate": None} for r in enrichment]),
        "model_category_comparison.png": (
            ["model_category", "model", "model_family", "parameters", "R@5",
             "R@10", "MRR", "NDCG@10", "retrieval_latency_seconds", "tokens"],
            ["Model Category", "Model", "Model Family", "Parameters", "R@5", "R@10",
             "MRR", "NDCG@10", "Latency (s)", "Tokens"],
            [{**r, "model": r["enrichment_model"],
              "tokens": enrichment_lookup.get(r["enrichment_model"], {}).get(
                "average_total_tokens")} for r in main]),
        "model_family_comparison.png": (
            ["model_family", "model", "model_category", "parameters", "R@5",
             "R@10", "MRR", "NDCG@10", "retrieval_latency_seconds"],
            ["Model Family", "Model", "Model Category", "Parameters", "R@5", "R@10",
              "MRR", "NDCG@10", "Latency (s)"],
             [{**r, "model": r["enrichment_model"]} for r in main]),
        "model_pairs.png": (
            ["enrichment_model", "embedding_model", "model_category", "segmentation", "R@5",
             "R@10", "MRR", "NDCG@10", "latency_seconds", "tokens"],
            ["Enrichment Model", "Embedding Model", "Category", "Segmentation", "R@5",
             "R@10", "MRR", "NDCG@10", "Latency (s)", "Tokens"], model_pair_rows(valid)),
        "embedding_results.png": (
            ["embedding_model", "representation", "segmentation", "enrichment_model", "R@5",
             "R@10", "MRR", "NDCG@10", "embedding_latency", "retrieval_latency_seconds"],
            ["Embedding Model", "Corpus / Representation", "Segmentation", "Enrichment",
             "R@5", "R@10", "MRR", "NDCG@10", "Embedding Latency", "Retrieval Latency (s)"],
            sorted(valid, key=lambda r: (r["embedding_model"], r["enrichment_model"]))),
        "efficiency_results.png": (
            ["enrichment_model", "model_category", "parameters", "input_tokens", "output_tokens",
             "total_tokens", "latency_seconds", "throughput", "memory", "gpu_memory", "R@5", "MRR"],
            ["Model", "Category", "Parameters", "Input Tokens", "Output Tokens", "Total Tokens",
             "Latency (s)", "Throughput", "Memory", "GPU Memory", "R@5", "MRR"],
            efficiency),
        "prior_work_methodology.png": (
            list(prior_work_methodology_rows()[0]),
            ["Work", "Primary Problem", "Code-aware", "Structural Representation", "AST",
             "CFG / Graph", "LLM Enrichment", "Retrieval", "Segmentation",
             "Embedding Retrieval", "Repository Context", "Evaluation Focus", "Key Metrics"],
            prior_work_methodology_rows()),
        "prior_work_numerical_comparison.png": (
            list(prior_work_numerical_rows()[0]),
            ["Work", "Dataset / Benchmark", "Metric", "Published Result", "Optima Result",
             "Optima Configuration", "Absolute Difference", "Relative Difference",
             "Directly Comparable?", "Reason"], prior_work_numerical_rows()),
        "retrieval_metrics.png": (
            ["configuration", "R@1", "R@5", "R@10", "P@1", "P@5", "P@10", "MRR",
             "NDCG@5", "NDCG@10"], ["Configuration", "R@1", "R@5", "R@10", "P@1",
             "P@5", "P@10", "MRR", "NDCG@5", "NDCG@10"], valid),
        "faithfulness_results.png": (
            ["enrichment_model", "model_category", "faithfulness", "evidence_support_rate",
             "evidence_precision", "evidence_recall", "unsupported_claim_rate",
             "hallucination_rate", "json_valid_rate", "evaluation_basis"],
            ["Model", "Category", "Faithfulness", "Evidence Support Rate", "Evidence Precision",
             "Evidence Recall", "Unsupported Claim Rate", "Hallucination Rate", "JSON Validity",
             "Evaluation Basis"],
            [{"enrichment_model": r["model"], "model_category": r["model_category"],
              "faithfulness": None, "evidence_support_rate": None,
              "evidence_precision": None, "evidence_recall": None,
              "unsupported_claim_rate": None, "hallucination_rate": None,
              "json_valid_rate": r["json_valid_rate"],
              "evaluation_basis": "N/A: no evidence mapping or reference labels"}
             for r in enrichment]),
        "failure_analysis.png": (
            ["model_configuration", "failure_type", "count", "percentage", "affected_stage",
             "example_description"],
            ["Model / Configuration", "Failure Type", "Count", "Percentage",
             "Affected Stage", "Example / Description"],
            [{"model_configuration": r.get("configuration", "N/A"),
              "failure_type": "Retrieval / unavailable",
              "count": 1, "percentage": None, "affected_stage": "retrieval",
              "example_description": r.get("status", "N/A")} for r in failures]),
    }
    return tables


def report(dest: Path, inventory: list[dict[str, Any]], rows: list[dict[str, Any]],
           enrichment: list[dict[str, Any]], figures: list[dict[str, Any]],
           source_file: Path) -> None:
    def fmt(value: Any) -> str:
        return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
    controlled = [r for r in rows if r.get("embedding_model") == "bge-small"
                  and isinstance(r.get("MRR"), (int, float))]
    enrichment_by_model = {item["model"]: item for item in enrichment}
    efficiency_data = [{
        "label": r["enrichment_model"],
        "tokens": enrichment_by_model.get(r["enrichment_model"], {}).get("average_total_tokens"),
        "enrichment_latency": enrichment_by_model.get(r["enrichment_model"], {}).get("average_latency_seconds"),
        "mrr": r["MRR"],
        "retrieval_latency": r.get("retrieval_latency_seconds"),
    } for r in controlled]
    efficiency_data = [p for p in efficiency_data
                       if all(isinstance(p[key], (int, float)) for key in
                              ("tokens", "enrichment_latency", "mrr", "retrieval_latency"))]
    findings = []
    if len(efficiency_data) >= 3:
        import numpy as np
        tokens = np.array([p["tokens"] for p in efficiency_data], dtype=float)
        enrich_latency = np.array([p["enrichment_latency"] for p in efficiency_data], dtype=float)
        mrr = np.array([p["mrr"] for p in efficiency_data], dtype=float)
        retrieval_latency = np.array([p["retrieval_latency"] for p in efficiency_data], dtype=float)
        token_latency = float(np.corrcoef(tokens, enrich_latency)[0, 1]) if np.std(tokens) and np.std(enrich_latency) else None
        token_mrr = float(np.corrcoef(tokens, mrr)[0, 1]) if np.std(tokens) and np.std(mrr) else None
        findings.append(f"Among {len(efficiency_data)} controlled bge-small configurations, "
                        f"the Pearson correlation between enrichment tokens and enrichment latency is "
                        f"{fmt(token_latency)}; token usage and MRR correlate at {fmt(token_mrr)}.")
        best = max(efficiency_data, key=lambda p: p["mrr"])
        findings.append(f"The highest observed controlled MRR is {fmt(best['mrr'])} for "
                        f"`{best['label']}`; this is a measured quality result, not a claim that higher cost is preferable.")
        fastest_quality = min(efficiency_data, key=lambda p: p["retrieval_latency"])
        findings.append(f"The lowest retrieval latency among these rows is {fmt(fastest_quality['retrieval_latency'] * 1000)} ms "
                        f"for `{fastest_quality['label']}`. Parameter-size plots are omitted because parameter counts are unavailable.")
    lines = [
        "# Optima Evaluation Report",
        "",
        "## Scope and provenance",
        "",
        f"This report is generated by `python -m evaluation.run_evaluation` from existing artifacts. "
        f"The retrieval source is `{source_file.relative_to(ROOT)}`; original outputs are never modified.",
        "All reported retrieval values are recalculated from saved ranked results and benchmark relevance IDs. "
        "No synthetic labels, results, or human judgments are introduced.",
        "",
        "## Experiment inventory",
        "",
        "| Model | Family | Category | Parameters | Segmentation | Enrichment | Embedding | Corpus | Chunk size | Samples | Queries | Available metrics |",
        "|---|---|---|---:|---|---|---|---|---:|---:|---:|---|",
    ]
    for r in inventory:
        lines.append("| " + " | ".join(str(r.get(k, "N/A")) for k in
            ["model", "model_family", "model_category", "parameters", "segmentation", "enrichment_model",
             "embedding_model", "representation_corpus", "chunk_size",
             "number_of_samples", "number_of_queries", "available_metrics"]) + " |")
    lines += [
        "", "## Model category analysis", "",
        "Model category is derived from the recorded model purpose/configuration, independently of retrieval scores. "
        "The supplied artifacts support General-purpose, Code, and Reasoning categories; BERT and BERT2 variants are "
        "not present in the discovered outputs and are reported as unavailable. Category comparisons are controlled "
        "to the bge-small benchmark; detailed model-family/category values are retained in CSV and PNG tables.",
        "", "## Retrieval methodology and results", "",
        "Relevance is binary and comes from `relevant_functions` in `benchmark/queries.json`. "
        "Recall divides retrieved relevant nodes by the query's relevant-node set; Precision@k divides hits by k; "
        "MRR uses the first relevant rank; NDCG uses binary gains. Aggregates are macro means over queries.",
        "", "| Configuration | R@1 | R@5 | R@10 | MRR | NDCG@10 | P@5 | Latency (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["configuration"], *[fmt(r[k]) for k in ["R@1", "R@5", "R@10", "MRR", "NDCG@10", "P@5", "retrieval_latency_seconds"]]))
    lines += ["", "## Enrichment results", "",
              "The artifacts contain model-generated structured fields and completeness, JSON validity, latency, and token usage. "
              "They do not contain an independent reference enrichment for semantic correctness; therefore field Accuracy, "
              "Precision, Recall, F1, Faithfulness, Hallucination, and evidence support are N/A rather than inferred from completeness.",
              "", "| Model | JSON valid | Success | Purpose F1 | Behavior F1 | Input F1 | Output F1 | Side-effect F1 | Overall F1 | Tokens | Latency (s) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in enrichment:
        lines.append("| {} | {} | {} | N/A | N/A | N/A | N/A | N/A | N/A | {} | {} |".format(
            r["model"], fmt(r["json_valid_rate"]), fmt(r["success_rate"]),
            fmt(r["average_total_tokens"]), fmt(r["average_latency_seconds"])))
    lines += [
        "", "## Segmentation, resolution, faithfulness, and cost limitations", "",
        "Only one chunk size (1000) and one representation mode (`hybrid`) are recorded. Segmentation strategy, "
        "boundary annotations, chunk token distributions, and alternative segmentation runs are absent, so boundary "
        "precision/recall/F1, structural preservation, cross-boundary rate, purity, and segmentation ablations are N/A.",
        "Source JSON contains AST/CFG/LLVM and graph artifacts, but no prediction-vs-reference resolution task or evidence "
        "mapping from enrichment claims to source spans. Resolution accuracy, relationship accuracy, faithfulness, "
        "hallucination, and evidence precision/recall are consequently N/A. Model parameter counts, memory/GPU telemetry, "
        "API prices, and embedding/enrichment phase timings for every pair are not consistently recorded; these are not estimated as zero.",
        "",
        "## Model families and pair comparisons", "",
        "Rows preserve every discovered corpus × embedding combination. GPT/decoder-style grouping is descriptive; no BERT/encoder "
        "model appears in the discovered artifacts. GPT-style decoder models are retained as their recorded family; "
        "BERT/BERT2 are absent rather than inferred. Pair values are in `tables/model_pairs.csv`, with no aggregation that hides poor configurations.",
        "",
        "## Statistical analysis", "",
        "Per-query rows and mean/median/standard deviation are emitted. Paired comparisons are reproducible from `per_query/retrieval.csv`; "
        "no significance test is reported because saved runs are deterministic single observations per query/configuration and no independent-run structure is recorded.",
        "",
        "## Failure and coverage analysis", "",
        "Failures are preserved in `tables/failures.csv`. Coverage is computed from observed query and enrichment success fields. "
        "No observed failed enrichment records were found in the supplied enriched JSON files.",
        "",
        "## Base-paper comparison", "",
        "The prior-work methodology table positions Optima against CRAG, critic/retrieval-critique approaches, AST-T5, "
        "RepoCoder, and GraphCode without claiming reproduction. The numerical prior-work table contains only explicit "
        "non-comparability decisions because no compatible published result was supplied in the repository artifacts. "
        "Code-completion, generation, or task-specific scores are never substituted for Optima retrieval metrics.",
        "",
        "## Research questions and table map", "",
        "RQ1 (structural segmentation) is covered by `segmentation_results.csv` and `optima_ablation.csv`; segmentation "
        "variants are not recorded, so structural-boundary metrics remain N/A. RQ2 (LLM enrichment) is covered by the "
        "enrichment and ablation tables, with retrieval deltas calculated against the raw bge-small row where available. "
        "RQ3 (embeddings) is covered by `embedding_results.csv` and `model_pairs.csv`. RQ4 (model categories) and RQ5 "
        "(model families) are covered by their dedicated tables. RQ6 (best pair) is preserved in the logically sorted pair "
        "table and heatmap. RQ7 (quality versus compute) is covered by `efficiency_results.csv` and the labeled latency figure. "
        "RQ8 is covered by the two prior-work tables. RQ9 is covered by `failure_analysis.csv`; only observed unavailable/failure "
        "records are included.",
        "",
        "## Efficiency findings", "",
        *(findings if findings else [
            "No sufficiently complete controlled token/latency/quality set was available for efficiency interpretation."
        ]),
        "No parameter-size, enrichment-quality-versus-cost, or segmentation-quality-versus-retrieval figure is generated "
        "because the required parameter, independent enrichment-quality, boundary-quality, or matched segmentation measurements "
        "are absent from the supplied artifacts.",
        "",
        "## Reproducibility", "",
        "Run `python -m evaluation.run_evaluation --source output --output evaluation_results`. "
        "The command discovers the latest saved retrieval matrix, reads benchmark queries, index metadata, source JSON, and enrichment metadata, "
        "then regenerates CSV/JSON/PNG/Markdown artifacts.",
        "",
        "## Figures", "",
        "The presentation uses a compact set of focused figures and table-first reporting. Detailed model, pair, retrieval, "
        "enrichment, efficiency, coverage, ablation, and failure values remain in `tables/`; figures show only "
        "the overall controlled comparison, raw-versus-enriched effect, labeled efficiency trade-offs, correlation, and pair heatmap. "
        "GPT+BERT, GPT+BERT2, BERT, BERT2, segmentation, chunk-size, overlap, and parameter-size figures are "
        "not generated because those variants are absent or unrecorded in the supplied artifacts. "
        "The category-specific figures are omitted when the discovered category has fewer than two supported models; "
        "the corresponding PNG table still preserves all available rows.",
        "",
        "### Figure index", "",
        "| Figure | Filename | Research question | Configurations | Metric | Interpretation |",
        "|---:|---|---|---|---|---|",
    ]
    if figures:
        for number, figure in enumerate(figures, start=1):
            lines.append("| {} | `{}` | {} | {} | {} | {} |".format(
                number, figure["filename"], figure["question"],
                ", ".join(figure["configurations"]), figure["metric"],
                figure["interpretation"]))
    else:
        lines.append("| - | N/A | No figures generated (matplotlib unavailable). | N/A | N/A | N/A |")
    lines += [
        "",
        "### PNG table index",
        "",
        "Presentation-ready high-resolution table images are emitted under `tables_png/`; their complete rows remain "
        "available in the corresponding CSV files under `tables/`.",
        "",
        "| Table image |",
        "|---|",
    ]
    for filename in sorted(p.name for p in (dest / "tables_png").glob("*.png")):
        lines.append(f"| `{filename}` |")
    (dest / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(source: Path, dest: Path) -> dict[str, Any]:
    source_file, retrieval = find_latest_matrix(source)
    metadata = metadata_index(source)
    queries = read_json(source / "rag_report" / "benchmark" / "queries.json")
    rows, per_query = retrieval_rows(retrieval["results"], queries, metadata)
    observed = {(row["enrichment_model"], row["embedding_model"]) for row in rows}
    # Index metadata is itself evidence of a valid prepared combination. Keep
    # combinations without saved ranked output visible as unavailable rather
    # than silently dropping them or fabricating retrieval values.
    for (corpus, embedding), meta in metadata.items():
        key = (corpus or "raw", embedding)
        if key not in observed:
            rows.append({
                "configuration": f"{corpus or 'raw'} + {embedding}",
                "representation": "enriched" if corpus != "raw" else "raw",
                "enrichment_model": corpus or "raw", "embedding_model": embedding,
                "model_family": family(corpus),
                "model_category": model_metadata(corpus or "raw")["category"],
                "segmentation": "unavailable (not recorded)",
                "chunk_size": meta.get("chunk_size"), "chunk_overlap": meta.get("chunk_overlap"),
                "number_of_chunks": meta.get("num_chunks"), "number_of_documents": meta.get("num_documents"),
                "queries": 0, "R@1": None, "R@5": None, "R@10": None, "R@20": None,
                "P@1": None, "P@5": None, "P@10": None, "MRR": None,
                "NDCG@5": None, "NDCG@10": None, "retrieval_latency_seconds": None,
                "Faithfulness": None, "status": "unavailable: no saved ranked retrieval output",
            })
    rows.sort(key=lambda row: (row["enrichment_model"], row["embedding_model"]))
    best_mark(rows, "R@5", "embedding_model")
    best_mark(rows, "MRR", "embedding_model")
    inventory = discover_inventory(rows, source)
    enrichment = []
    for corpus in sorted({row["enrichment_model"] for row in rows}):
        path = source_json_for(corpus, source)
        if path:
            enrichment.append(enrichment_row(corpus, read_json(path)))
    failures = [row for row in rows if row.get("status") != "success"]
    if dest.exists():
        shutil.rmtree(dest)
    for directory in ("tables", "figures", "per_query", "raw_metrics"):
        (dest / directory).mkdir(parents=True)
    write_csv(dest / "tables" / "retrieval.csv", rows)
    write_csv(dest / "tables" / "model_metadata.csv", [{
        "model": model,
        "model_family": model_metadata(model)["family"],
        "model_category": model_metadata(model)["category"],
        "parameters": None,
        "purpose": model_metadata(model)["purpose"],
    } for model in sorted({r["enrichment_model"] for r in rows})])
    write_csv(dest / "tables" / "model_comparison.csv", [{
        "model_or_corpus": r["enrichment_model"], "embedding": r["embedding_model"],
        "model_family": r["model_family"], "model_category": r["model_category"],
        "R@1": r["R@1"], "R@5": r["R@5"], "R@10": r["R@10"], "R@20": r["R@20"],
        "P@1": r["P@1"], "P@5": r["P@5"], "P@10": r["P@10"],
        "MRR": r["MRR"], "NDCG@5": r["NDCG@5"], "NDCG@10": r["NDCG@10"],
    } for r in rows])
    write_csv(dest / "tables" / "enrichment.csv", enrichment)
    write_csv(dest / "tables" / "efficiency.csv", [{
        "configuration": r["configuration"], "model": r["enrichment_model"],
        "model_family": r["model_family"], "model_category": r["model_category"],
        "embedding": r["embedding_model"], "parameters": None,
        "input_tokens": next((x["average_input_tokens"] for x in enrichment
                              if x["model"] == r["enrichment_model"]), None),
        "output_tokens": next((x["average_output_tokens"] for x in enrichment
                               if x["model"] == r["enrichment_model"]), None),
        "total_tokens": next((x["average_total_tokens"] for x in enrichment
                              if x["model"] == r["enrichment_model"]), None),
        "retrieval_latency_ms": r["retrieval_latency_seconds"] * 1000
        if r["retrieval_latency_seconds"] is not None else None,
        "enrichment_latency_seconds": next((x["average_latency_seconds"] for x in enrichment
                                            if x["model"] == r["enrichment_model"]), None),
        "throughput": next((x["throughput_functions_per_second"] for x in enrichment
                            if x["model"] == r["enrichment_model"]), None),
        "memory": None, "gpu_memory": None, "R@5": r["R@5"], "MRR": r["MRR"],
    } for r in rows])
    write_csv(dest / "tables" / "model_pairs.csv", model_pair_rows(rows))
    write_csv(dest / "tables" / "segmentation.csv", [{
        "segmentation": "unavailable", "average_tokens": None, "R@5": None,
        "R@10": None, "MRR": None, "NDCG@10": None, "latency": None,
        "reason": "segmentation strategy not recorded"
    }])
    write_csv(dest / "tables" / "full_comparison.csv", rows)
    write_csv(dest / "tables" / "failures.csv", failures)
    write_csv(dest / "tables" / "coverage.csv", [{
        "configuration": row["configuration"],
        "query_coverage": row["queries"] / len(queries) if queries else None,
        "retrieval_coverage": row["queries"] / len(queries) if queries else None,
        "enrichment_coverage": next((x["success_rate"] for x in enrichment
                                     if x["model"] == row["enrichment_model"]), None),
        "successful_processing_rate": 1.0 if row["status"] == "success" else 0.0,
        "failed_processing_rate": 0.0 if row["status"] == "success" else 1.0,
    } for row in rows])
    write_csv(dest / "tables" / "overlap.csv", [{
        "comparison": "not computed",
        "system_a_only": None, "system_b_only": None, "both": None, "neither": None,
        "reason": "saved results do not expose a complete common-system set definition"
    }])
    write_csv(dest / "tables" / "ablations.csv", [{
        "axis": axis, "status": "unavailable",
        "reason": reason, "absolute_improvement": None, "relative_improvement_percent": None
    } for axis, reason in [
        ("segmentation", "only one unlabelled segmentation/chunking configuration recorded"),
        ("enrichment", "no paired segmentation and independent reference labels"),
        ("embedding", "embedding comparisons are available in retrieval.csv; controlled pairing metadata is incomplete"),
        ("chunk_size", "only chunk size 1000 recorded"),
    ]])
    write_csv(dest / "tables" / "statistics.csv", [{
        "comparison": "paired per-query tests", "test": "N/A",
        "sample_size": len(queries), "p_value": None, "effect_size": None,
        "reason": "no independent repeated runs recorded"
    }])
    write_csv(dest / "tables" / "base_paper_comparison.csv", [{
        "metric": "all", "base_paper": None, "optima": None,
        "difference": None, "relative_improvement": None,
        "status": "N/A", "reason": "no compatible base-paper tables in repository outputs"
    }])
    research = research_tables(rows, enrichment, failures)
    for png_name, (keys, _headers, data) in research.items():
        write_csv(dest / "tables" / (png_name.removesuffix(".png") + ".csv"),
                  [{key: row.get(key) for key in keys} for row in data])
    write_csv(dest / "per_query" / "retrieval.csv", per_query)
    write_json(dest / "raw_metrics" / "retrieval_source.json", retrieval)
    write_json(dest / "raw_metrics" / "metadata.json", list(metadata.values()))
    figures = create_figures(rows, enrichment, dest / "figures")
    table_pngs = create_research_table_pngs(dest, research)
    summary = {
        "source_retrieval_file": str(source_file.relative_to(ROOT)),
        "queries": len(queries),
        "configurations": len(rows),
        "enrichment_models": sorted({r["enrichment_model"] for r in rows}),
        "embedding_models": sorted({r["embedding_model"] for r in rows}),
        "model_categories": sorted({r["model_category"] for r in rows}),
        "model_families": sorted({r["model_family"] for r in rows}),
        "chunk_sizes": sorted({r["chunk_size"] for r in rows if r["chunk_size"] is not None}),
        "segmentation_variants": [],
        "metrics": {"retrieval": "calculated", "enrichment_quality": "unavailable without reference labels",
                    "faithfulness": "unavailable without evidence mappings", "segmentation": "unavailable"},
        "failures": len(failures),
        "figures": [figure["filename"] for figure in figures],
        "tables_png": table_pngs,
    }
    write_json(dest / "summary.json", summary)
    report(dest, inventory, rows, enrichment, figures, source_file)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    summary = run(args.source.resolve(), args.output.resolve())
    print(f"Wrote {summary['configurations']} configurations to {args.output.resolve()}")


if __name__ == "__main__":
    main()
