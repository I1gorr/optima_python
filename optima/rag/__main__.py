"""
Main entry point for Optima RAG system.
Provides CLI commands for embedding, evaluation, and retrieval.
"""

import argparse
import logging
import sys
from pathlib import Path

# Use the simple embedding module that has fallback to mock embeddings
from .embedding_simple import embed_all_corpora, get_embedding_registry, embedding_alias
from .evaluation import (
    create_benchmark_from_json_files,
    save_benchmark_queries,
    load_benchmark_queries,
    OptimaRetrievalEvaluator
)
from .rag_pipeline import rag_query

# Try to import LLM for RAG pipeline
try:
    from langchain_community.llms import HuggingFacePipeline
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logging.warning("LLM dependencies not available. RAG pipeline will be limited.")


def _model_list(values):
    """Accept argparse's repeated values and convenient comma-separated values."""
    if not values:
        return []
    return [item.strip() for value in values for item in str(value).split(",") if item.strip()]


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def cmd_embed_all(args):
    """Handle the embed-all command."""
    logging.info(f"Starting embedding process for all corpora")
    requested_models = _model_list(getattr(args, "embedding_models", None))
    if getattr(args, "all_embedding_models", False):
        requested_models = list(get_embedding_registry().keys())
    requested_models = requested_models or [args.embedding_model]
    logging.info(f"Embedding models: {requested_models}")
    logging.info(f"Representation mode: {args.representation_mode}")

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logging.error(f"Output directory does not exist: {output_dir}")
        sys.exit(1)

    try:
        results = embed_all_corpora(
            output_dir=output_dir,
            embedding_model_name=requested_models[0],
            embedding_model_names=requested_models,
            representation_mode=args.representation_mode,
            force=getattr(args, "force", False),
        )

        # Save results
        results_file = output_dir / "rag_report" / "embedding_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Embedding completed. Results saved to {results_file}")

        # Print summary
        summary = results.get("summary", {})
        print(f"\nEmbedding Summary:")
        print(f"  Embedding models: {summary.get('total_embedding_models', len(requested_models))}")
        print(f"  Total corpora: {summary.get('total_corpora', 0)}")
        print(f"  Successful: {summary.get('successful', summary.get('successful_corpora', 0))}")
        print(f"  Failed: {summary.get('failed', summary.get('failed_corpora', 0))}")
        print(f"  Total time: {summary.get('total_elapsed_time_seconds', 0)}s")

        if summary.get('failed_corpora', 0) > 0:
            print("\nFailed corpora:")
            for alias, corpora in results.get("embeddings", {}).items():
                for name, result in corpora.items():
                    if not result.get("success", False):
                        print(f"  {alias}/{name}: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logging.error(f"Embedding process failed: {e}")
        sys.exit(1)


def cmd_evaluate(args):
    """Handle the evaluate command."""
    if getattr(args, "complete_evaluation", False):
        from evaluation.run_evaluation import run as run_complete_evaluation

        source_dir = Path(args.output_dir)
        destination = Path(getattr(args, "evaluation_output", "evaluation_results"))
        try:
            summary = run_complete_evaluation(source_dir.resolve(), destination.resolve())
        except (OSError, ValueError, FileNotFoundError) as exc:
            logging.error("Complete evaluation failed: %s", exc)
            sys.exit(1)
        print(
            f"Complete evaluation written to {destination.resolve()} "
            f"({summary['configurations']} configurations)."
        )
        return

    logging.info(f"Starting evaluation process")
    requested_models = _model_list(getattr(args, "embedding_models", None))
    if getattr(args, "all_embedding_models", False):
        requested_models = list(get_embedding_registry().keys())
    requested_models = requested_models or [args.embedding_model]

    output_dir = Path(args.output_dir)
    rag_report_dir = output_dir / "rag_report"

    if not output_dir.exists():
        logging.error(f"Output directory does not exist: {output_dir}")
        sys.exit(1)

    try:
        # Load or create benchmark
        benchmark_path = Path(args.benchmark) if args.benchmark else rag_report_dir / "benchmark" / "queries.json"

        if args.create_benchmark or not benchmark_path.exists():
            logging.info("Creating benchmark queries...")
            # Handle backward compatibility: if num_queries_per_category is specified, use it
            # Otherwise, use num_queries for backward compatibility
            if hasattr(args, 'num_queries_per_category') and args.num_queries_per_category is not None:
                # Calculate total queries: num_queries_per_category * 8 categories
                num_queries = args.num_queries_per_category * 8
                logging.info(f"Creating benchmark with {args.num_queries_per_category} queries per category (8 categories) = {num_queries} total queries")
            else:
                num_queries = args.num_queries
                logging.info(f"Creating benchmark with {num_queries} total queries (backward compatibility)")

            # Use seed from args if available, otherwise default to 42
            seed = getattr(args, 'seed', 42)

            queries = create_benchmark_from_json_files(
                output_dir,
                num_queries,
                difficulty_distribution=(0.25, 0.35, 0.40),
                seed=seed
            )
            save_benchmark_queries(queries, benchmark_path)
            logging.info(f"Benchmark saved to {benchmark_path}")
        else:
            logging.info(f"Loading benchmark from {benchmark_path}")
            queries = load_benchmark_queries(benchmark_path)

        if not queries:
            logging.error("No benchmark queries available")
            sys.exit(1)

        # Load corpus indices
        print(f"Benchmark queries: {len(queries)}")
        print(f"Query embedding models: {', '.join(requested_models)}")
        logging.info(f"Evaluating {len(queries)} queries with k={args.k}")
        from .evaluation import (
            evaluate_embedding_matrix, save_matrix_csv, save_group_metrics_csv,
            save_gain_csv, save_best_tables, generate_visualizations,
        )
        results_matrix = evaluate_embedding_matrix(
            rag_report_dir, requested_models, queries, args.k,
            corpora=_model_list(getattr(args, "corpora", None)),
            enrichment_models=_model_list(getattr(args, "enrichment_models", None)),
            difficulty=getattr(args, "difficulty", None),
            categories=_model_list(getattr(args, "categories", None)),
            debug=args.debug, debug_limit=args.debug_limit,
        )
        if not any(results_matrix.values()):
            logging.error("No vector indices found for selected embedding model(s)")
            logging.info("Run 'python -m optima.rag embed-all' first to create indices")
            sys.exit(1)
        # Keep the historical flat JSON shape for a single selected model.
        results = (next(iter(results_matrix.values()))
                   if len(results_matrix) == 1 else results_matrix)

        # Save results
        results_dir = rag_report_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_results_file = results_dir / f"retrieval_evaluation_{timestamp}.json"
        detailed_results = {
            "embedding_models": requested_models,
            "benchmark_queries": len(queries),
            "k_value": args.k,
            "corpora_evaluated": sorted({
                corpus for corpora in results_matrix.values() for corpus in corpora
            }),
            "results": results,
            "evaluation_timestamp": timestamp
        }
        with open(detailed_results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # Save CSV summary
        csv_file = results_dir / "retrieval_model_comparison.csv"
        save_matrix_csv(results_matrix, csv_file)
        save_matrix_csv(results_matrix, results_dir / "aggregate_metrics.csv")
        from .evaluation import save_per_query_results
        save_per_query_results(results_matrix, rag_report_dir / "per_query_results.csv")
        save_group_metrics_csv(results_matrix, results_dir / "difficulty_metrics.csv", "difficulty")
        save_group_metrics_csv(results_matrix, results_dir / "category_metrics.csv", "category")
        save_matrix_csv(results_matrix, results_dir / "embedding_comparison.csv")
        save_group_metrics_csv(results_matrix, results_dir / "embedding_by_difficulty.csv", "difficulty")
        save_group_metrics_csv(results_matrix, results_dir / "embedding_by_category.csv", "category")
        save_gain_csv(results_matrix, results_dir / "enrichment_gains.csv", "enrichment")
        if len(results_matrix) > 1:
            save_gain_csv(results_matrix, results_dir / "embedding_gains.csv", "embedding")
        save_best_tables(results_matrix, results_dir / "best_tables.csv")
        from .evaluation import save_best_combinations
        save_best_combinations(results_matrix, results_dir)
        generate_visualizations(results_matrix, results_dir / "plots")

        # Generate improvement over baseline
        if len(results_matrix) == 1 and "raw" in results:
            improvement_file = results_dir / "baseline_improvement.csv"
            try:
                save_improvement_csv(results, "raw", improvement_file)
            except Exception as e:
                logging.error(f"Failed to save improvement CSV: {e}")
                import traceback
                logging.debug(traceback.format_exc())

        # Generate rankings
        ranking_file = results_dir / "model_ranking.csv"
        try:
            save_ranking_csv(results, ranking_file)
        except Exception as e:
            logging.error(f"Failed to save ranking CSV: {e}")
            import traceback
            logging.debug(traceback.format_exc())

        logging.info(f"Evaluation completed. Results saved to {results_dir}")

        # Print summary table
        print_evaluation_summary(results)

    except Exception as e:
        logging.error(f"Evaluation process failed: {e}")
        import traceback
        logging.error(traceback.format_exc())  # Changed from debug to error to ensure it's visible
        sys.exit(1)


def cmd_retrieve(args):
    """Handle the retrieve command."""
    logging.info(f"Starting retrieval query")
    logging.info(f"Embedding model: {args.embedding_model}")
    logging.info(f"Query: {args.query}")

    output_dir = Path(args.output_dir)
    rag_report_dir = output_dir / "rag_report"

    if not output_dir.exists():
        logging.error(f"Output directory does not exist: {output_dir}")
        sys.exit(1)

    try:
        # Determine corpus to query
        if args.corpus == "raw":
            corpus_name = "raw"
            enrichment_model = None
        elif args.enrichment_model:
            corpus_name = args.enrichment_model
            enrichment_model = args.enrichment_model
        else:
            logging.error("Must specify either --corpus raw or --enrichment-model <model>")
            sys.exit(1)

        # Check if index exists
        from .embedding_simple import index_directory
        index_path = index_directory(rag_report_dir, corpus_name, args.embedding_model)
        if not index_path.exists():
            # load_corpus_indices also handles historical indexes; locate the
            # requested corpus before reporting it as missing.
            from .evaluation import load_corpus_indices
            legacy = load_corpus_indices(rag_report_dir, args.embedding_model)
            if corpus_name in legacy:
                index_path = Path(legacy[corpus_name]._optima_index_path)
        if not index_path.exists():
            logging.error(f"Vector index not found: {index_path}")
            logging.info(f"Run 'python -m optima.rag embed-all' first to create indices")
            sys.exit(1)

        # Setup LLM if RAG mode requested
        llm = None
        if args.rag_mode:
            if not LLM_AVAILABLE:
                logging.warning("LLM not available, falling back to retrieval-only mode")
            else:
                # Try to load a simple LLM for demonstration
                try:
                    # Use a small, fast model for testing
                    llm = HuggingFacePipeline.from_model_id(
                        model_id="microsoft/Phi-3-mini-4k-instruct",
                        task="text-generation",
                        model_kwargs={
                            "temperature": 0.7,
                            "max_length": 512,
                        }
                    )
                    logging.info("LLM loaded for RAG pipeline")
                except Exception as e:
                    logging.warning(f"Could not load LLM: {e}. Falling back to retrieval-only mode.")

        # Perform retrieval or RAG
        if args.rag_mode and llm:
            from .rag_pipeline import rag_query
            result = rag_query(
                corpus_name=corpus_name,
                embedding_model_name=args.embedding_model,
                llm=llm,
                question=args.query,
                k=args.k,
                output_base_dir=rag_report_dir,
            )

            print(f"\nQuery: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"Source documents: {result['num_source_documents']}")
            print(f"Response time: {result['response_time_seconds']}s")

            if args.verbose:
                print(f"\nSource documents:")
                for i, doc in enumerate(result['source_documents']):
                    print(f"  {i+1}. {doc.metadata.get('function_name', 'unknown')} "
                          f"(score: N/A)")
                    if args.verbose > 1:  # Show content
                        print(f"     Content: {doc.page_content[:200]}...")
        else:
            # Pure retrieval mode
            from .embedding_simple import OptimaEmbedder
            from langchain_community.vectorstores import FAISS

            embedder = OptimaEmbedder(embedding_model_name=args.embedding_model)
            vector_store = embedder.load_vector_index(index_path)

            if not vector_store:
                logging.error(f"Failed to load vector index from {index_path}")
                sys.exit(1)

            retriever = vector_store.as_retriever(search_kwargs={"k": args.k})
            if hasattr(retriever, 'get_relevant_documents'):
                docs = retriever.get_relevant_documents(args.query)
            else:
                docs = retriever.invoke(args.query)

            print(f"\nQuery: {args.query}")
            print(f"Retrieved {len(docs)} documents:")
            for i, doc in enumerate(docs):
                func_name = doc.metadata.get('function_name', 'unknown')
                qualified_name = doc.metadata.get('qualified_name', '')
                file_name = doc.metadata.get('file_name', '')
                corpus_type = doc.metadata.get('corpus_type', '')
                enrichment_model = doc.metadata.get('enrichment_model', 'none')

                print(f"  {i+1}. {func_name}")
                if qualified_name and qualified_name != func_name:
                    print(f"      Qualified: {qualified_name}")
                print(f"      File: {file_name}")
                print(f"      Corpus: {corpus_type} ({enrichment_model})")
                if args.verbose:
                    print(f"      Content preview: {doc.page_content[:150]}...")
                print()

    except Exception as e:
        logging.error(f"Retrieval process failed: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        sys.exit(1)


def save_results_as_csv(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Save evaluation results as CSV."""
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define columns
    columns = [
        'corpus', 'recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
        'precision_at_1', 'precision_at_3', 'precision_at_5', 'precision_at_10',
        'hit_rate_at_1', 'hit_rate_at_3', 'hit_rate_at_5', 'hit_rate_at_10',
        'mrr', 'mean_latency', 'median_latency', 'queries_evaluated'
    ]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        for corpus_name, metrics in results.items():
            if isinstance(metrics, dict) and 'error' not in metrics:
                row = {'corpus': corpus_name}
                for col in columns[1:]:  # Skip 'corpus'
                    row[col] = metrics.get(col, 0.0)
                writer.writerow(row)


def save_improvement_csv(results: Dict[str, Dict[str, Any]], baseline_name: str, output_path: Path):
    """Save improvement over baseline as CSV."""
    import csv
    from .evaluation import calculate_improvement_over_baseline

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if baseline_name not in results:
        logging.warning(f"Baseline {baseline_name} not found in results")
        return

    baseline_metrics = results[baseline_name]
    logging.info(f"Baseline metrics keys: {list(baseline_metrics.keys())}")

    # Define columns
    columns = [
        'model', 'metric', 'raw_value', 'model_value',
        'absolute_improvement', 'relative_improvement_percent'
    ]

    metrics_to_compare = [
        'recall_at_1', 'recall_at_3', 'recall_at_5', 'recall_at_10',
        'precision_at_1', 'precision_at_3', 'precision_at_5', 'precision_at_10',
        'hit_rate_at_1', 'hit_rate_at_3', 'hit_rate_at_5', 'hit_rate_at_10',
        'mrr'
    ]

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        for model_name, metrics in results.items():
            if model_name == baseline_name or not isinstance(metrics, dict) or 'error' in metrics:
                continue

            logging.info(f"Processing model: {model_name}")
            logging.info(f"Model metrics keys: {list(metrics.keys())}")

            improvements = calculate_improvement_over_baseline(baseline_metrics, metrics)
            logging.info(f"Improvements keys: {list(improvements.keys())}")

            for metric in metrics_to_compare:
                raw_val = baseline_metrics.get(metric, 0.0)
                model_val = metrics.get(metric, 0.0)
                abs_imp = improvements.get(f'{metric}_absolute', 0.0)
                rel_imp = improvements.get(f'{metric}_relative_percent', 0.0)

                logging.info(f"Metric {metric}: raw={raw_val} ({type(raw_val)}), model={model_val} ({type(model_val)}), abs={abs_imp} ({type(abs_imp)}), rel={rel_imp} ({type(rel_imp)})")

                writer.writerow({
                    'model': model_name,
                    'metric': metric,
                    'raw_value': raw_val,
                    'model_value': model_val,
                    'absolute_improvement': abs_imp,
                    'relative_improvement_percent': rel_imp
                })


def save_ranking_csv(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Save model rankings as CSV."""
    import csv
    from .evaluation import rank_enrichment_models

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Rank by different metrics
    metrics_to_rank = ['recall_at_5', 'mrr', 'hit_rate_at_5']

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Rank', 'Model', 'Metric', 'Value'])

        for metric in metrics_to_rank:
            rankings = rank_enrichment_models(results, metric)
            for rank, (model_name, value) in enumerate(rankings, start=1):
                writer.writerow([rank, model_name, metric, f"{value:.4f}"])


def print_evaluation_summary(results: Dict[str, Dict[str, Any]]):
    """Print evaluation results in a formatted table."""
    # Matrix evaluations are printed one row per embedding/corpus pair.
    if results and not any(
        isinstance(value, dict) and "per_query_results" in value
        for value in results.values()
    ):
        print("\n" + "=" * 80)
        print("RETRIEVAL EVALUATION MATRIX")
        print("=" * 80)
        print(f"{'Embedding':<18} {'Corpus':<28} {'R@5':<8} {'MRR':<8} {'Score':<8}")
        for embedding, corpora in results.items():
            for corpus, metrics in corpora.items():
                if not isinstance(metrics, dict) or "error" in metrics:
                    continue
                print(f"{embedding:<18} {corpus:<28} "
                      f"{metrics.get('recall_at_5', 0.0):<8.3f} "
                      f"{metrics.get('mrr', 0.0):<8.3f} "
                      f"{metrics.get('weighted_overall_score', 0.0):<8.3f}")
        print("=" * 80)
        return
    print("\n" + "="*80)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*80)

    # Header
    header = f"{'Model':<25} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'Hit@5':<8} {'Latency(s)':<10}"
    print(header)
    print("-" * len(header))

    # Sort models: raw first, then alphabetically
    model_order = []
    if 'raw' in results:
        model_order.append('raw')
    model_order.extend(sorted([k for k in results.keys() if k != 'raw']))

    for model_name in model_order:
        metrics = results.get(model_name, {})
        if not isinstance(metrics, dict) or 'error' in metrics:
            continue

        r1 = metrics.get('recall_at_1', 0.0)
        r5 = metrics.get('recall_at_5', 0.0)
        r10 = metrics.get('recall_at_10', 0.0)
        mrr = metrics.get('mrr', 0.0)
        hit5 = metrics.get('hit_rate_at_5', 0.0)
        latency = metrics.get('mean_latency', 0.0)

        # Highlight best values
        def format_val(val, is_best=False):
            s = f"{val:.3f}"
            if is_best:
                return f"***{s}***"
            return s

        print(f"{model_name:<25} {format_val(r1):<8} {format_val(r5):<8} "
              f"{format_val(r10):<8} {format_val(mrr):<8} {format_val(hit5):<8} "
              f"{latency:<10.3f}")

    print("="*80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Optima RAG System - Embedding, Retrieval, and Evaluation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # embed-all command
    embed_parser = subparsers.add_parser(
        "embed-all",
        help="Embed all JSON corpora and create vector indices"
    )
    embed_parser.add_argument(
        "--embedding-model",
        default="bge-small",
        help="Embedding alias or HuggingFace model name to use"
    )
    embed_parser.add_argument(
        "--embedding-models", nargs="+",
        help="Embed with several aliases (space or comma separated)"
    )
    embed_parser.add_argument(
        "--all-embedding-models", action="store_true",
        help="Embed with every registered embedding alias"
    )
    embed_parser.add_argument(
        "--force", action="store_true",
        help="Rebuild existing canonical indexes"
    )
    embed_parser.add_argument(
        "--representation-mode",
        choices=["raw", "enriched", "semantic", "compiler", "hybrid"],
        default="hybrid",
        help="Document representation mode"
    )
    embed_parser.add_argument(
        "--output-dir", "-o",
        default="/home/igorr/Projects/op-python/optima/output",
        help="Directory containing Optima JSON output files"
    )
    embed_parser.set_defaults(func=cmd_embed_all)

    # evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate retrieval performance across corpora"
    )
    eval_parser.add_argument(
        "--embedding-model",
        default="bge-small",
        help="Embedding alias or HuggingFace model name used for indices"
    )
    eval_parser.add_argument("--embedding-models", nargs="+",
                             help="Evaluate several embedding aliases")
    eval_parser.add_argument("--all-embedding-models", action="store_true",
                             help="Evaluate every registered embedding alias")
    eval_parser.add_argument("--corpora", nargs="+",
                             help="Only evaluate these corpus names")
    eval_parser.add_argument("--enrichment-models", nargs="+",
                             help="Only evaluate these enrichment corpus names")
    eval_parser.add_argument("--difficulty", choices=["easy", "moderate", "difficult"],
                             help="Filter benchmark queries by difficulty")
    eval_parser.add_argument("--categories", nargs="+",
                             help="Filter benchmark queries by category")
    eval_parser.add_argument(
        "--benchmark",
        help="Path to benchmark queries JSON file (optional)"
    )
    eval_parser.add_argument(
        "--create-benchmark",
        action="store_true",
        help="Create new benchmark queries from JSON files"
    )
    eval_parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of benchmark queries to create (if creating benchmark)"
    )
    eval_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to retrieve per query"
    )
    eval_parser.add_argument(
        "--output-dir", "-o",
        default="/home/igorr/Projects/op-python/optima/output",
        help="Directory containing Optima JSON output files"
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # retrieve command
    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Perform retrieval or RAG query"
    )
    retrieve_parser.add_argument(
        "--embedding-model",
        default="bge-small",
        help="Embedding alias or HuggingFace model name used for indices"
    )
    retrieve_parser.add_argument(
        "--corpus",
        choices=["raw"],
        help="Query the raw baseline corpus"
    )
    retrieve_parser.add_argument(
        "--enrichment-model",
        help="Query an enriched corpus (specify model name)"
    )
    retrieve_parser.add_argument(
        "--query",
        required=True,
        help="Query text"
    )
    retrieve_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to retrieve"
    )
    retrieve_parser.add_argument(
        "--rag-mode",
        action="store_true",
        help="Use RAG pipeline with LLM (retrieval-only if not specified)"
    )
    retrieve_parser.add_argument(
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity for output"
    )
    retrieve_parser.add_argument(
        "--output-dir", "-o",
        default="/home/igorr/Projects/op-python/optima/output",
        help="Directory containing Optima JSON output files"
    )
    retrieve_parser.set_defaults(func=cmd_retrieve)

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose >= 2 else "INFO" if args.verbose >= 1 else "WARNING"
    setup_logging(log_level)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()