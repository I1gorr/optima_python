"""
Evaluation module for Optima RAG system.
Handles benchmark creation, retrieval evaluation, and metric calculation.
"""

import json
import logging
import csv
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
import numpy as np
from collections import defaultdict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


class OptimaRetrievalEvaluator:
    """Evaluates retrieval performance for Optima RAG system."""

    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        Initialize the evaluator.

        Args:
            k_values: List of k values for Recall@k and Precision@k
        """
        self.k_values = k_values

    def create_benchmark_queries(self, functions: List[Dict[str, Any]],
                               num_queries: int = 20) -> List[Dict[str, Any]]:
        """
        Create benchmark queries from function data.

        Args:
            functions: List of function dictionaries from JSON data
            num_queries: Number of queries to generate

        Returns:
            List of query dictionaries with ground truth
        """
        logger.info(f"Creating {num_queries} benchmark queries from {len(functions)} functions")

        # Select functions for queries (avoid very simple ones)
        candidate_functions = []
        for func in functions:
            # Skip functions with very short names or no meaningful content
            name = func.get("name", "")
            if len(name) < 3:
                continue
            # Prefer functions with some complexity indicators
            params = len(func.get("parameters", []))
            deps = len(func.get("dependencies", []))
            if params > 0 or deps > 0 or len(name) > 5:
                candidate_functions.append(func)

        # If not enough complex functions, use all
        if len(candidate_functions) < num_queries // 2:
            candidate_functions = functions

        # Select subset for queries
        selected_functions = candidate_functions[:min(num_queries, len(candidate_functions))]

        queries = []
        for func in selected_functions:
            func_name = func.get("name", "unknown")
            qualified_name = func.get("qualified_name", func_name)
            func_id = func.get("id", "")

            # Generate natural language query based on function
            query = self._generate_natural_language_query(func)
            if not query:
                query = f"What does the function {func_name} do?"

            queries.append({
                "query": query,
                "relevant_functions": [func_id] if func_id else [qualified_name],
                "function_name": func_name,
                "qualified_name": qualified_name,
                "function_id": func_id
            })

        # If we need more queries, generate some generic ones
        while len(queries) < num_queries:
            func = selected_functions[len(queries) % len(selected_functions)]
            func_name = func.get("name", "unknown")
            func_id = func.get("id", "")
            queries.append({
                "query": f"How does the {func_name} function work?",
                "relevant_functions": [func_id] if func_id else [func_name],
                "function_name": func_name,
                "qualified_name": func.get("qualified_name", func_name),
                "function_id": func_id
            })

        logger.info(f"Created {len(queries)} benchmark queries")
        return queries[:num_queries]

    def _generate_natural_language_query(self, func: Dict[str, Any]) -> Optional[str]:
        """Generate a natural language query from function data."""
        name = func.get("name", "")
        return_type = func.get("return_type", "")
        params = func.get("parameters", [])
        purpose = func.get("purpose", "")
        summary = func.get("summary", "")
        algorithm = func.get("algorithm", "")

        # Use purpose or summary if available
        if purpose and len(purpose) > 10:
            # Create a question about the purpose
            return f"What is the purpose of the {name} function?"
        elif summary and len(summary) > 5:
            return f"What does the {name} function do?"
        elif algorithm:
            return f"How does the {name} function implement {algorithm}?"
        elif params:
            param_names = [p.get("name", "") for p in params if p.get("name")]
            if param_names:
                return f"How does the {name} function use {', '.join(param_names[:2])}?"
        elif return_type and return_type != "void":
            return f"What does the {name} function return?"

        # Fallback to name-based query
        if name and len(name) > 2:
            return f"What is the {name} function?"

        return None

    def evaluate_retrieval(self, vector_store: FAISS,
                          queries: List[Dict[str, Any]],
                          k: int = 10,
                          corpus_name: Optional[str] = None,
                          debug: bool = False,
                          debug_limit: int = 20) -> Dict[str, Any]:
        """
        Evaluate retrieval performance using benchmark queries.

        Args:
            vector_store: FAISS vector store
            queries: List of query dictionaries with ground truth
            k: Number of results to retrieve per query

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating retrieval with {len(queries)} queries, k={k}")

        if not queries:
            return self._empty_metrics()

        # Initialize metrics
        metrics = {
            "recall_at_k": {k_val: [] for k_val in self.k_values},
            "precision_at_k": {k_val: [] for k_val in self.k_values},
            "mrr_scores": [],
            "hit_rate_at_k": {k_val: [] for k_val in self.k_values},
            "latencies": [],
            "queries_evaluated": 0,
            "queries_with_results": 0
        }
        per_query_results = []

        retriever = vector_store.as_retriever(search_kwargs={"k": k})

        for query_index, query_data in enumerate(queries):
            query_text = query_data["query"]
            relevant_func_ids = set(query_data["relevant_functions"])

            start_time = time.time()
            try:
                # Preserve scores when FAISS exposes them.  The fallback is
                # intentionally compatible with simple test vector stores.
                if hasattr(vector_store, "similarity_search_with_score"):
                    scored_docs = vector_store.similarity_search_with_score(query_text, k=k)
                    retrieved_docs = [doc for doc, _score in scored_docs]
                    scores = [float(score) for _doc, score in scored_docs]
                else:
                    if hasattr(retriever, 'get_relevant_documents'):
                        retrieved_docs = retriever.get_relevant_documents(query_text)
                    else:
                        retrieved_docs = retriever.invoke(query_text)
                    scores = [None] * len(retrieved_docs)
                elapsed_time = time.time() - start_time

                metrics["latencies"].append(elapsed_time)

                if retrieved_docs:
                    metrics["queries_with_results"] += 1

                ranked_results = []
                for rank, (doc, score) in enumerate(zip(retrieved_docs, scores), start=1):
                    func_id = doc.metadata.get("function_id", "")
                    qualified_name = doc.metadata.get("qualified_name", "")
                    result_id = func_id or qualified_name
                    ranked_results.append({
                        "function_id": result_id,
                        "rank": rank,
                        "score": score,
                        "relevant": bool(
                            result_id in relevant_func_ids
                            or (qualified_name and qualified_name in relevant_func_ids)
                        )
                    })

                # Calculate metrics for each k value
                for k_val in self.k_values:
                    cutoff_results = ranked_results[:min(k_val, len(ranked_results))]
                    retrieved_func_ids = {result["function_id"] for result in cutoff_results}
                    # Recall@k: proportion of relevant functions retrieved
                    if relevant_func_ids:
                        retrieved_relevant = retrieved_func_ids.intersection(relevant_func_ids)
                        recall = len(retrieved_relevant) / len(relevant_func_ids)
                        metrics["recall_at_k"][k_val].append(recall)

                        # Precision@k: proportion of retrieved functions that are relevant
                        if retrieved_func_ids:
                            precision = len(retrieved_relevant) / len(retrieved_func_ids)
                            metrics["precision_at_k"][k_val].append(precision)
                        else:
                            metrics["precision_at_k"][k_val].append(0.0)

                        # Hit Rate@k: binary - did we retrieve at least one relevant function?
                        hit = 1.0 if len(retrieved_relevant) > 0 else 0.0
                        metrics["hit_rate_at_k"][k_val].append(hit)
                    else:
                        # No relevant functions - edge case
                        metrics["recall_at_k"][k_val].append(0.0)
                        metrics["precision_at_k"][k_val].append(0.0)
                        metrics["hit_rate_at_k"][k_val].append(0.0)

                # MRR: Mean Reciprocal Rank
                rr = 0.0
                first_relevant_rank = None
                for result in ranked_results:
                    if result["relevant"]:
                        first_relevant_rank = result["rank"]
                        rr = 1.0 / first_relevant_rank
                        break
                metrics["mrr_scores"].append(rr)

                per_query_results.append({
                    "query_id": query_data.get("query_id", f"q_{query_index + 1:04d}"),
                    "difficulty": query_data.get("difficulty", ""),
                    "category": query_data.get("category", ""),
                    "query": query_text,
                    "relevant_functions": list(relevant_func_ids),
                    "retrieved": ranked_results,
                    "top_1_function": ranked_results[0]["function_id"] if ranked_results else "",
                    "top_5_functions": [r["function_id"] for r in ranked_results[:5]],
                    "top_10_functions": [r["function_id"] for r in ranked_results[:10]],
                    "first_relevant_rank": first_relevant_rank,
                    "reciprocal_rank": rr,
                })
                if debug and query_index < debug_limit:
                    target = ", ".join(sorted(relevant_func_ids))
                    rank_text = str(first_relevant_rank) if first_relevant_rank else "no hit"
                    score_text = (
                        f"{ranked_results[first_relevant_rank - 1]['score']:.4f}"
                        if first_relevant_rank and ranked_results[first_relevant_rank - 1]["score"] is not None
                        else "n/a"
                    )
                    print(f"Query {per_query_results[-1]['query_id']}")
                    print(f"  target: {target}")
                    print(f"  rank: {rank_text}")
                    print(f"  score: {score_text}")

                metrics["queries_evaluated"] += 1

            except Exception as e:
                logger.error(f"Error evaluating query '{query_text}': {e}")
                # Still count the query but with zero scores
                metrics["latencies"].append(time.time() - start_time)
                for k_val in self.k_values:
                    metrics["recall_at_k"][k_val].append(0.0)
                    metrics["precision_at_k"][k_val].append(0.0)
                    metrics["hit_rate_at_k"][k_val].append(0.0)
                metrics["mrr_scores"].append(0.0)
                per_query_results.append({
                    "query_id": query_data.get("query_id", f"q_{query_index + 1:04d}"),
                    "difficulty": query_data.get("difficulty", ""),
                    "category": query_data.get("category", ""),
                    "query": query_text,
                    "relevant_functions": list(relevant_func_ids),
                    "retrieved": [],
                    "top_1_function": "",
                    "top_5_functions": [],
                    "top_10_functions": [],
                    "first_relevant_rank": None,
                    "reciprocal_rank": 0.0,
                })
                metrics["queries_evaluated"] += 1

        # Calculate final metrics
        final_metrics = {
            "num_queries": len(queries),
            "queries_evaluated": metrics["queries_evaluated"],
            "queries_with_results": metrics["queries_with_results"],
            "mean_latency": np.mean(metrics["latencies"]) if metrics["latencies"] else 0.0,
            "median_latency": np.median(metrics["latencies"]) if metrics["latencies"] else 0.0,
            "per_query_results": per_query_results,
        }

        # Add Recall@k, Precision@k, Hit Rate@k
        for k_val in self.k_values:
            recall_scores = metrics["recall_at_k"][k_val]
            precision_scores = metrics["precision_at_k"][k_val]
            hit_scores = metrics["hit_rate_at_k"][k_val]

            final_metrics[f"recall_at_{k_val}"] = np.mean(recall_scores) if recall_scores else 0.0
            final_metrics[f"precision_at_{k_val}"] = np.mean(precision_scores) if precision_scores else 0.0
            final_metrics[f"hit_rate_at_{k_val}"] = np.mean(hit_scores) if hit_scores else 0.0

        # Add MRR
        final_metrics["mrr"] = np.mean(metrics["mrr_scores"]) if metrics["mrr_scores"] else 0.0

        logger.info(f"Evaluation completed: MRR={final_metrics['mrr']:.4f}, "
                   f"Recall@5={final_metrics.get('recall_at_5', 0):.4f}")
        return final_metrics

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        metrics = {
            "num_queries": 0,
            "queries_evaluated": 0,
            "queries_with_results": 0,
            "mean_latency": 0.0,
            "median_latency": 0.0,
            "mrr": 0.0
        }
        for k_val in self.k_values:
            metrics[f"recall_at_{k_val}"] = 0.0
            metrics[f"precision_at_{k_val}"] = 0.0
            metrics[f"hit_rate_at_{k_val}"] = 0.0
        return metrics

    def evaluate_multiple_corpora(self, corpus_indices: Dict[str, FAISS],
                                benchmark_queries: List[Dict[str, Any]],
                                k: int = 10,
                                debug: bool = False,
                                debug_limit: int = 20) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate retrieval performance across multiple corpora with difficulty-level metrics.

        Args:
            corpus_indices: Dictionary mapping corpus names to FAISS indices
            benchmark_queries: List of benchmark queries with difficulty labels
            k: Number of results to retrieve per query

        Returns:
            Dictionary mapping corpus names to their evaluation metrics (overall and by difficulty)
        """
        logger.info(f"Evaluating {len(corpus_indices)} corpora with {len(benchmark_queries)} queries")

        results = {}
        for corpus_name, vector_store in corpus_indices.items():
            logger.info(f"Evaluating corpus: {corpus_name}")
            index_path = getattr(vector_store, "_optima_index_path", "unknown")
            embedding_model = getattr(vector_store, "_optima_embedding_model", "unknown")
            document_count = getattr(vector_store, "_optima_document_count", "unknown")
            print(f"\nCorpus: {corpus_name}")
            print(f"Index: {index_path}")
            print(f"Documents: {document_count}")
            print(f"Query embedding model: {embedding_model}")
            try:
                # Overall metrics
                overall_metrics = self.evaluate_retrieval(
                    vector_store, benchmark_queries, k, corpus_name=corpus_name,
                    debug=debug, debug_limit=debug_limit
                )

                # Difficulty-level metrics
                difficulty_metrics = {}
                difficulties = ["easy", "moderate", "difficult"]
                for difficulty in difficulties:
                    # Filter queries by difficulty
                    diff_queries = [q for q in benchmark_queries if q.get("difficulty") == difficulty]
                    if diff_queries:
                        diff_metrics = self.evaluate_retrieval(vector_store, diff_queries, k)
                        # Prefix metrics with difficulty level
                        for key, value in diff_metrics.items():
                            if key not in ["num_queries", "queries_evaluated", "queries_with_results"]:
                                difficulty_metrics[f"{difficulty}_{key}"] = value
                        difficulty_metrics[f"{difficulty}_num_queries"] = len(diff_queries)
                    else:
                        # No queries for this difficulty level
                        difficulty_metrics[f"{difficulty}_num_queries"] = 0
                        for k_val in self.k_values:
                            difficulty_metrics[f"{difficulty}_recall_at_{k_val}"] = 0.0
                            difficulty_metrics[f"{difficulty}_precision_at_{k_val}"] = 0.0
                            difficulty_metrics[f"{difficulty}_hit_rate_at_{k_val}"] = 0.0
                        difficulty_metrics[f"{difficulty}_mrr"] = 0.0
                        difficulty_metrics[f"{difficulty}_mean_latency"] = 0.0
                        difficulty_metrics[f"{difficulty}_median_latency"] = 0.0

                category_metrics = {}
                categories = sorted({q.get("category", "") for q in benchmark_queries if q.get("category")})
                for category in categories:
                    category_queries = [q for q in benchmark_queries if q.get("category") == category]
                    category_result = self.evaluate_retrieval(vector_store, category_queries, k)
                    for key, value in category_result.items():
                        if key not in ["num_queries", "queries_evaluated", "queries_with_results", "per_query_results"]:
                            category_metrics[f"{category}_{key}"] = value
                    category_metrics[f"{category}_num_queries"] = len(category_queries)

                # Combine overall and difficulty metrics
                corpus_metrics = {**overall_metrics, **difficulty_metrics, **category_metrics}
                for item in corpus_metrics.get("per_query_results", []):
                    item["corpus"] = corpus_name
                    item["embedding_model"] = embedding_model
                results[corpus_name] = corpus_metrics
            except Exception as e:
                logger.error(f"Failed to evaluate corpus {corpus_name}: {e}")
                results[corpus_name] = self._empty_metrics()
                results[corpus_name]["error"] = str(e)

        return results


def load_corpus_indices(output_base_dir: Path, embedding_model_name: str) -> Dict[str, FAISS]:
    """
    Load all vector indices for a given embedding model.

    Args:
        output_base_dir: Base directory containing indices
        embedding_model_name: Name of embedding model

    Returns:
        Dictionary mapping corpus names to FAISS indices
    """
    from .embedding_simple import (
        OptimaEmbedder, embedding_alias, index_directory, resolve_embedding_model
    )

    report_dir = Path(output_base_dir)
    indexes_dir = report_dir / "indexes"
    if not indexes_dir.exists():
        logger.warning("Indices directory not found: %s", indexes_dir)
        return {}

    spec = resolve_embedding_model(embedding_model_name)
    alias = embedding_alias(embedding_model_name)
    # Canonical layout is indexes/<corpus>/<alias>.  Then add the old layout
    # indexes/<model-name>/<corpus> as a compatibility fallback.
    candidates: Dict[str, Path] = {}
    for corpus_dir in indexes_dir.iterdir():
        if not corpus_dir.is_dir():
            continue
        canonical = corpus_dir / alias
        if (canonical / "index.faiss").exists():
            candidates[corpus_dir.name] = canonical

    legacy_root = indexes_dir.joinpath(*spec.model_name.split("/"))
    if legacy_root.exists():
        for corpus_dir in legacy_root.iterdir():
            if corpus_dir.is_dir() and (corpus_dir / "index.faiss").exists():
                candidates.setdefault(corpus_dir.name, corpus_dir)
    # A few early runs used the alias itself as the model directory.
    alias_root = indexes_dir / alias
    if alias_root.exists():
        for corpus_dir in alias_root.iterdir():
            if corpus_dir.is_dir() and (corpus_dir / "index.faiss").exists():
                candidates.setdefault(corpus_dir.name, corpus_dir)

    corpus_indices = {}
    embedder = OptimaEmbedder(embedding_model_name=embedding_model_name)
    for corpus_name, item in candidates.items():
        logger.info("Loading index for corpus %s from %s", corpus_name, item)
        vector_store = embedder.load_vector_index(item)
        if vector_store:
            metadata = {}
            metadata_file = item / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            vector_store._optima_index_path = str(item.resolve())
            vector_store._optima_embedding_model = alias
            vector_store._optima_embedding_model_name = spec.model_name
            vector_store._optima_metadata = metadata
            docstore = getattr(vector_store, "docstore", None)
            values = getattr(docstore, "_dict", {}).values() if docstore else []
            doc_ids = {
                doc.metadata.get("function_id") or doc.metadata.get("qualified_name")
                for doc in values
                if doc.metadata.get("function_id") or doc.metadata.get("qualified_name")
            }
            vector_store._optima_document_count = len(doc_ids)
            vector_store._optima_metadata.setdefault("document_count", len(doc_ids))
            corpus_indices[corpus_name] = vector_store
        else:
            logger.warning("Failed to load index for corpus: %s", corpus_name)

    logger.info(f"Loaded {len(corpus_indices)} corpus indices")
    return corpus_indices


def _result_matrix(results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Normalize old flat results and new embedding x corpus results."""
    if not results:
        return {}
    if all(isinstance(value, dict) and "per_query_results" in value
           for value in results.values()):
        return {"default": results}
    matrix = {}
    for embedding, corpora in results.items():
        if isinstance(corpora, dict):
            matrix[embedding] = corpora
    return matrix


def weighted_overall_score(metrics: Dict[str, Any],
                           weights: Optional[Dict[str, float]] = None) -> float:
    """Calculate a bounded score useful for comparing matrix rows."""
    weights = weights or {
        "mrr": 0.30, "recall_at_5": 0.35,
        "hit_rate_at_5": 0.20, "precision_at_5": 0.15,
    }
    total = sum(float(metrics.get(key, 0.0) or 0.0) * weight
                for key, weight in weights.items())
    return round(total / sum(weights.values()), 6) if weights else 0.0


def evaluate_embedding_matrix(output_base_dir: Path, embedding_models: List[str],
                              benchmark_queries: List[Dict[str, Any]], k: int = 10,
                              corpora: Optional[List[str]] = None,
                              enrichment_models: Optional[List[str]] = None,
                              difficulty: Optional[str] = None,
                              categories: Optional[List[str]] = None,
                              debug: bool = False,
                              debug_limit: int = 20) -> Dict[str, Dict[str, Any]]:
    """Evaluate every selected enrichment corpus for every embedding alias."""
    from .embedding_simple import embedding_alias
    evaluator = OptimaRetrievalEvaluator()
    queries = benchmark_queries
    if difficulty:
        queries = [q for q in queries if q.get("difficulty") == difficulty]
    if categories:
        wanted = set(categories)
        queries = [q for q in queries if q.get("category") in wanted]
    selected_corpora = set(corpora or [])
    selected_enrichment = set(enrichment_models or [])
    matrix: Dict[str, Dict[str, Any]] = {}
    for model in embedding_models:
        indices = load_corpus_indices(output_base_dir, model)
        if selected_corpora:
            indices = {name: store for name, store in indices.items()
                       if name in selected_corpora}
        if selected_enrichment:
            indices = {name: store for name, store in indices.items()
                       if name in selected_enrichment or
                       (name == "raw" and "raw" in selected_enrichment)}
        evaluated = evaluator.evaluate_multiple_corpora(
            indices, queries, k, debug=debug, debug_limit=debug_limit
        )
        for corpus_name, metrics in evaluated.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                metrics["weighted_overall_score"] = weighted_overall_score(metrics)
                metrics["document_count"] = getattr(
                    indices.get(corpus_name), "_optima_document_count", 0
                )
        matrix[embedding_alias(model)] = evaluated
    return matrix


def save_matrix_csv(results: Dict[str, Any], output_path: Path) -> None:
    """Save aggregate metrics for a flat or embedding x corpus result set."""
    matrix = _result_matrix(results)
    metric_names = [
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "precision_at_1", "precision_at_3", "precision_at_5", "precision_at_10",
        "hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5", "hit_rate_at_10",
        "mrr", "mean_latency", "median_latency", "queries_evaluated",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["enrichment_model", "embedding_model",
                                                     "functions", "queries", "R@1", "R@5",
                                                     "R@10", "MRR", "Hit@5", "avg_latency",
                                                     "median_latency", "weighted_overall_score"])
        writer.writeheader()
        for embedding, corpora in matrix.items():
            for corpus, metrics in corpora.items():
                if not isinstance(metrics, dict) or "error" in metrics:
                    continue
                row = {
                    "enrichment_model": corpus,
                    "embedding_model": embedding,
                    "functions": metrics.get("document_count", metrics.get("functions", 0)),
                    "queries": metrics.get("num_queries", 0),
                    "R@1": metrics.get("recall_at_1", 0.0),
                    "R@5": metrics.get("recall_at_5", 0.0),
                    "R@10": metrics.get("recall_at_10", 0.0),
                    "MRR": metrics.get("mrr", 0.0),
                    "Hit@5": metrics.get("hit_rate_at_5", 0.0),
                    "avg_latency": metrics.get("mean_latency", 0.0),
                    "median_latency": metrics.get("median_latency", 0.0),
                    "weighted_overall_score": weighted_overall_score(metrics),
                }
                writer.writerow(row)


def save_best_combinations(results: Dict[str, Any], output_dir: Path) -> None:
    """Write ranked combinations and best choices for each experimental axis."""
    matrix = _result_matrix(results)
    rows = []
    for embedding, corpora in matrix.items():
        for corpus, metrics in corpora.items():
            if not isinstance(metrics, dict) or "error" in metrics:
                continue
            difficult = "difficult_"
            rows.append({
                "enrichment_model": corpus,
                "embedding_model": embedding,
                "R@1": metrics.get("recall_at_1", 0.0),
                "R@5": metrics.get("recall_at_5", 0.0),
                "R@10": metrics.get("recall_at_10", 0.0),
                "MRR": metrics.get("mrr", 0.0),
                "Hit@5": metrics.get("hit_rate_at_5", 0.0),
                "difficult_R@1": metrics.get(difficult + "recall_at_1", 0.0),
                "difficult_MRR": metrics.get(difficult + "mrr", 0.0),
                "overall_score": weighted_overall_score(metrics),
            })
    rows.sort(key=lambda row: row["overall_score"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["rank", "enrichment_model", "embedding_model", "R@1", "R@5",
               "R@10", "MRR", "Hit@5", "difficult_R@1", "difficult_MRR",
               "overall_score"]
    with (output_dir / "best_combinations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    for filename, group_key, group_value in (
        ("best_embedding_per_enrichment.csv", "enrichment_model", "enrichment_model"),
        ("best_enrichment_per_embedding.csv", "embedding_model", "embedding_model"),
    ):
        best = {}
        for row in rows:
            key = row[group_key]
            best.setdefault(key, row)
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(best.values())


def save_group_metrics_csv(results: Dict[str, Any], output_path: Path,
                           group: str) -> None:
    """Save difficulty/category aggregate metrics (e.g. easy_mrr)."""
    matrix = _result_matrix(results)
    rows = []
    for embedding, corpora in matrix.items():
        for corpus, metrics in corpora.items():
            if not isinstance(metrics, dict):
                continue
            prefix = f"{group}_"
            if group == "category":
                prefixes = sorted({
                    key[:-4] for key in metrics
                    if key.endswith("_mrr")
                    and key[:-4] not in {"easy", "moderate", "difficult"}
                })
            else:
                prefixes = ["easy", "moderate", "difficult"]
            for label in prefixes:
                rows.append({
                    "enrichment_model": corpus, "embedding_model": embedding,
                    group: label,
                    "queries": metrics.get(f"{label}_num_queries", 0),
                    "R@1": metrics.get(f"{label}_recall_at_1", 0.0),
                    "R@5": metrics.get(f"{label}_recall_at_5", 0.0),
                    "R@10": metrics.get(f"{label}_recall_at_10", 0.0),
                    "MRR": metrics.get(f"{label}_mrr", 0.0),
                    "Hit@5": metrics.get(f"{label}_hit_rate_at_5", 0.0),
                })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["enrichment_model", "embedding_model", group, "queries",
               "R@1", "R@5", "R@10", "MRR", "Hit@5"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_gain_csv(results: Dict[str, Any], output_path: Path,
                  axis: str = "enrichment") -> None:
    """Save enrichment or embedding gains against a reproducible baseline."""
    matrix = _result_matrix(results)
    rows = []
    if axis == "enrichment":
        for embedding, corpora in matrix.items():
            baseline = corpora.get("raw", {})
            for corpus, metrics in corpora.items():
                if corpus == "raw" or not isinstance(metrics, dict):
                    continue
                for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"):
                    rows.append({
                        "axis": axis, "embedding_model": embedding,
                        "comparison": corpus, "metric": metric,
                        "baseline": baseline.get(metric, 0.0),
                        "value": metrics.get(metric, 0.0),
                        "absolute_gain": metrics.get(metric, 0.0) - baseline.get(metric, 0.0),
                        "relative_gain_percent": (
                            (metrics.get(metric, 0.0) - baseline.get(metric, 0.0))
                            / baseline.get(metric, 1.0) * 100
                            if baseline.get(metric, 0.0) else 0.0
                        ),
                    })
    else:
        models = list(matrix)
        baseline_model = models[0] if models else ""
        baseline = matrix.get(baseline_model, {})
        for embedding, corpora in matrix.items():
            if embedding == baseline_model:
                continue
            for corpus, metrics in corpora.items():
                base_metrics = baseline.get(corpus, {})
                for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"):
                    rows.append({
                        "axis": axis, "embedding_model": embedding,
                        "comparison": corpus, "metric": metric,
                        "baseline": base_metrics.get(metric, 0.0),
                        "value": metrics.get(metric, 0.0),
                        "absolute_gain": metrics.get(metric, 0.0) - base_metrics.get(metric, 0.0),
                        "relative_gain_percent": (
                            (metrics.get(metric, 0.0) - base_metrics.get(metric, 0.0))
                            / base_metrics.get(metric, 1.0) * 100
                            if base_metrics.get(metric, 0.0) else 0.0
                        ),
                    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["axis", "embedding_model", "comparison", "metric",
               "baseline", "value", "absolute_gain", "relative_gain_percent"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_best_tables(results: Dict[str, Any], output_path: Path) -> None:
    """Write the best corpus per embedding and best embedding per corpus."""
    matrix = _result_matrix(results)
    rows = []
    for embedding, corpora in matrix.items():
        valid = [(corpus, metrics) for corpus, metrics in corpora.items()
                 if isinstance(metrics, dict) and "error" not in metrics]
        if valid:
            corpus, metrics = max(valid, key=lambda item: weighted_overall_score(item[1]))
            rows.append({"table": "best_corpus_per_embedding", "embedding_model": embedding,
                         "corpus": corpus, "weighted_overall_score": weighted_overall_score(metrics)})
    corpus_names = sorted({corpus for corpora in matrix.values() for corpus in corpora})
    for corpus in corpus_names:
        valid = [(embedding, corpora[corpus]) for embedding, corpora in matrix.items()
                 if corpus in corpora and isinstance(corpora[corpus], dict)]
        if valid:
            embedding, metrics = max(valid, key=lambda item: weighted_overall_score(item[1]))
            rows.append({"table": "best_embedding_per_corpus", "embedding_model": embedding,
                         "corpus": corpus, "weighted_overall_score": weighted_overall_score(metrics)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["table", "embedding_model", "corpus", "weighted_overall_score"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def generate_visualizations(results: Dict[str, Any], output_dir: Path) -> List[Path]:
    """Generate small comparison plots when matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib is not installed; skipping evaluation plots")
        return []
    matrix = _result_matrix(results)
    if not matrix:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    labels, scores = [], []
    for embedding, corpora in matrix.items():
        for corpus, metrics in corpora.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                labels.append(f"{embedding}\n{corpus}")
                scores.append(weighted_overall_score(metrics))
    if labels:
        figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
        axis.bar(range(len(scores)), scores)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_ylabel("Weighted overall score")
        axis.set_title("Retrieval matrix comparison")
        figure.tight_layout()
        path = output_dir / "weighted_overall_score.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(path)
    combinations = [
        ("R@1", "recall_at_1", "r_at_1_heatmap.png"),
        ("R@5", "recall_at_5", "r_at_5_heatmap.png"),
        ("MRR", "mrr", "mrr_heatmap.png"),
    ]
    corpora = sorted({corpus for values in matrix.values() for corpus in values})
    embeddings = list(matrix)
    if corpora and embeddings:
        for title, metric, filename in combinations:
            values = [[matrix[e].get(c, {}).get(metric, 0.0) for e in embeddings]
                      for c in corpora]
            figure, axis = plt.subplots(figsize=(max(6, len(embeddings) * 1.2), 5))
            image = axis.imshow(values, aspect="auto", vmin=0, vmax=1)
            axis.set_xticks(range(len(embeddings)), embeddings, rotation=45, ha="right")
            axis.set_yticks(range(len(corpora)), corpora)
            axis.set_title(f"{title} by enrichment and embedding model")
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            path = output_dir / filename
            figure.savefig(path, dpi=140)
            plt.close(figure)
            paths.append(path)
    return paths


def save_per_query_results(results: Dict[str, Any], output_path: Path):
    """Save ranked evidence for every query and corpus to CSV."""
    columns = [
        "embedding_model", "corpus", "query_id", "difficulty", "category", "query",
        "relevant_functions", "top_1_function", "top_5_functions",
        "top_10_functions", "first_relevant_rank", "reciprocal_rank"
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        matrix = _result_matrix(results)
        for embedding, corpora in matrix.items():
          for corpus_name, metrics in corpora.items():
            for item in metrics.get("per_query_results", []):
                writer.writerow({
                    "embedding_model": embedding if embedding != "default" else "",
                    "corpus": corpus_name,
                    "query_id": item["query_id"],
                    "difficulty": item["difficulty"],
                    "category": item["category"],
                    "query": item["query"],
                    "relevant_functions": json.dumps(item["relevant_functions"]),
                    "top_1_function": item["top_1_function"],
                    "top_5_functions": json.dumps(item["top_5_functions"]),
                    "top_10_functions": json.dumps(item["top_10_functions"]),
                    "first_relevant_rank": item["first_relevant_rank"] or "",
                    "reciprocal_rank": item["reciprocal_rank"],
                })


def create_benchmark_from_json_files(output_dir: Path,
                                   num_queries: int = 20,
                                   difficulty_distribution: Tuple[float, float, float] = (0.25, 0.35, 0.40),
                                   seed: Optional[int] = 42) -> List[Dict[str, Any]]:
    """
    Create benchmark queries by analyzing JSON files with difficulty levels.

    Args:
        output_dir: Directory containing JSON files
        num_queries: Number of queries to generate
        difficulty_distribution: Tuple of (easy, moderate, difficult) proportions
        seed: Random seed for reproducibility

    Returns:
        List of benchmark query dictionaries with difficulty labels
    """
    logger.info(f"Creating benchmark from JSON files in {output_dir}")

    # Load all functions from all JSON files
    all_functions = []

    # Process base.json
    base_path = output_dir / "base.json"
    if base_path.exists():
        with open(base_path, 'r') as f:
            data = json.load(f)
            for file_data in data.get("files", []):
                all_functions.extend(file_data.get("functions", []))

    # Process enhanced JSONs (optional - could use for variety)
    for json_file in output_dir.glob("enhanced_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                for file_data in data.get("files", []):
                    all_functions.extend(file_data.get("functions", []))
        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")

    logger.info(f"Loaded {len(all_functions)} total functions for benchmark creation")

    if len(all_functions) == 0:
        logger.error("No functions found for benchmark creation!")
        return []

    # Create benchmark generator and generate queries
    from .benchmark_generator import create_benchmark_from_json_files as benchmark_create
    queries = benchmark_create(output_dir, num_queries, difficulty_distribution, seed)

    logger.info(f"Generated {len(queries)} benchmark queries")
    return queries


def save_benchmark_queries(queries: List[Dict[str, Any]], output_path: Path):
    """
    Save benchmark queries to JSON file.

    Args:
        queries: List of query dictionaries
        output_path: Path to save the benchmark
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(queries, f, indent=2)
    logger.info(f"Saved {len(queries)} benchmark queries to {output_path}")


def load_benchmark_queries(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load benchmark queries from JSON file.

    Args:
        input_path: Path to the benchmark file

    Returns:
        List of query dictionaries
    """
    if not input_path.exists():
        logger.warning(f"Benchmark file not found: {input_path}")
        return []

    with open(input_path, 'r') as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} benchmark queries from {input_path}")
    return queries


def calculate_improvement_over_baseline(raw_metrics: Dict[str, Any],
                                      enriched_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate improvement of enriched model over raw baseline.

    Args:
        raw_metrics: Metrics from raw baseline
        enriched_metrics: Metrics from enriched model

    Returns:
        Dictionary with absolute and relative improvements
    """
    improvements = {}

    # Metrics to compare
    metric_keys = [
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "precision_at_1", "precision_at_3", "precision_at_5", "precision_at_10",
        "hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5", "hit_rate_at_10",
        "mrr"
    ]

    for key in metric_keys:
        raw_val = raw_metrics.get(key, 0.0)
        enriched_val = enriched_metrics.get(key, 0.0)

        # Debug: Check types
        if not isinstance(raw_val, (int, float)):
            print(f"WARNING: raw_val for {key} is not a number: {raw_val} (type: {type(raw_val)})")
            raw_val = 0.0
        if not isinstance(enriched_val, (int, float)):
            print(f"WARNING: enriched_val for {key} is not a number: {enriched_val} (type: {type(enriched_val)})")
            enriched_val = 0.0

        absolute_improvement = enriched_val - raw_val
        if raw_val != 0:
            try:
                relative_improvement = (absolute_improvement / raw_val * 100)
            except Exception as e:
                print(f"ERROR calculating relative improvement for {key}: {e}")
                print(f"  absolute_improvement: {absolute_improvement} (type: {type(absolute_improvement)})")
                print(f"  raw_val: {raw_val} (type: {type(raw_val)})")
                relative_improvement = 0.0
        else:
            relative_improvement = 0.0

        improvements[f"{key}_absolute"] = absolute_improvement
        improvements[f"{key}_relative_percent"] = relative_improvement

    return improvements


def rank_enrichment_models(corpus_results: Dict[str, Dict[str, Any]],
                         metric: str = "recall_at_5") -> List[Tuple[str, float]]:
    """
    Rank enrichment models by a specific metric (excluding raw baseline).

    Args:
        corpus_results: Dictionary of corpus evaluation results
        metric: Metric to rank by (e.g., "recall_at_5", "mrr")

    Returns:
        List of (model_name, metric_value) tuples sorted descending
    """
    # Filter out raw baseline
    enrichment_results = {
        name: results for name, results in corpus_results.items()
        if name != "raw" and isinstance(results, dict) and metric in results
    }

    # Sort by metric value descending
    sorted_models = sorted(
        enrichment_results.items(),
        key=lambda x: x[1].get(metric, 0.0),
        reverse=True
    )

    return [(name, results[metric]) for name, results in sorted_models]


if __name__ == "__main__":
    # Test the evaluation module
    logging.basicConfig(level=logging.INFO)

    # Create some dummy data for testing
    dummy_queries = [
        {
            "query": "What does the puzzle solver function do?",
            "relevant_functions": ["benchmark_tests/benchmark_tests.h::Test::BenchMark::Puzzle::Puzzle::25"],
            "function_name": "Puzzle",
            "qualified_name": "Test::BenchMark::Puzzle::Puzzle",
            "function_id": "benchmark_tests/benchmark_tests.h::Test::BenchMark::Puzzle::Puzzle::25"
        }
    ]

    evaluator = OptimaRetrievalEvaluator()
    print("Evaluation module loaded successfully")
    print(f"Created {len(dummy_queries)} dummy queries for testing")