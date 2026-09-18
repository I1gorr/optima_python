"""
Optima CLI - Combines enrichment and RAG functionality.
"""
import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Optima: LM Studio LLM enrichment tool with RAG evaluation"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Import the original enrichment CLI functions
    from .cli import main as enrich_main, analyze_command, enrich_command

    # Import RAG functions
    from .rag.__main__ import (
        cmd_embed_all, cmd_evaluate, cmd_retrieve,
        setup_logging, save_results_as_csv, save_improvement_csv,
        save_ranking_csv, print_evaluation_summary
    )

    # ===== ENRICHMENT COMMANDS =====

    # analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze a project and generate base.json'
    )
    analyze_parser.add_argument(
        'project_path',
        help='Path to the project to analyze'
    )
    analyze_parser.add_argument(
        '-o', '--output',
        default=None,
        help=(
            'Output directory (default: optima_outputs/base/<test-suite>, '
            'where <test-suite> is a safe slug of the project directory name)'
        )
    )
    analyze_parser.set_defaults(func=analyze_command)

    # enrich command
    enrich_parser = subparsers.add_parser(
        'enrich',
        help='Enrich base.json with LM Studio to produce enhanced.json'
    )
    enrich_parser.add_argument(
        'base_json',
        help='Path to base.json file to enrich'
    )
    enrich_parser.add_argument(
        '-o', '--output',
        default='output',
        help='Output directory or explicit JSON output path (default: output)'
    )
    enrich_parser.add_argument(
        '--model',
        default=None,
        help='LM Studio model ID; otherwise discover the first available model'
    )
    enrich_parser.add_argument(
        '--base-url',
        default='http://localhost:1234/v1',
        help='LM Studio OpenAI-compatible API base URL'
    )
    enrich_parser.add_argument(
        '--model-load-timeout',
        type=int,
        default=300,
        help='Seconds to wait for the selected model to become ready'
    )
    enrich_parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process at most this many functions (useful for smoke tests)'
    )
    enrich_parser.add_argument(
        '--context-size',
        type=int,
        default=None,
        help='Model context size in tokens (discovered from LM Studio when omitted)'
    )
    enrich_parser.add_argument(
        '--max-input-tokens',
        type=int,
        default=None,
        help='Maximum input tokens; defaults to the model context minus reserved output'
    )
    enrich_parser.add_argument(
        '--reserved-output-tokens',
        type=int,
        default=2048,
        help='Tokens reserved for the enrichment response (default: 2048)'
    )
    enrich_parser.add_argument(
        '--debug',
        action='store_true',
        help='Print context preflight and reduction details'
    )
    enrich_parser.set_defaults(func=enrich_command)

    # ===== RAG COMMANDS =====

    # embed-all command
    embed_parser = subparsers.add_parser(
        'embed-all',
        help='Embed all JSON corpora and create vector indices'
    )
    embed_parser.add_argument(
        "--embedding-model",
        default="bge-small",
        help="Embedding alias or HuggingFace model name to use"
    )
    embed_parser.add_argument("--embedding-models", nargs="+",
                              help="Embed with several embedding aliases")
    embed_parser.add_argument("--all-embedding-models", action="store_true",
                              help="Embed with every registered embedding alias")
    embed_parser.add_argument("--force", action="store_true",
                              help="Rebuild existing canonical indexes")
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
        'evaluate',
        help='Evaluate retrieval performance across corpora'
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
    eval_parser.add_argument("--difficulty", choices=["easy", "moderate", "difficult"])
    eval_parser.add_argument("--categories", nargs="+")
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
        help="Number of benchmark queries to create (if creating benchmark) - for backward compatibility"
    )
    eval_parser.add_argument(
        "--num-queries-per-category",
        type=int,
        default=None,
        help="Number of benchmark queries to create PER CATEGORY (8 categories). Overrides --num-queries if specified."
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible benchmark generation (default: 42)"
    )
    eval_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to retrieve per query"
    )
    eval_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-query retrieval ranks and scores"
    )
    eval_parser.add_argument(
        "--debug-limit",
        type=int,
        default=20,
        help="Maximum number of queries to print per corpus in debug mode"
    )
    eval_parser.add_argument(
        "--output-dir", "-o",
        default="/home/igorr/Projects/op-python/optima/output",
        help="Directory containing Optima JSON output files"
    )
    eval_parser.add_argument(
        "--complete",
        dest="complete_evaluation",
        action="store_true",
        help="Build the publication evaluation from existing outputs without rerunning retrieval"
    )
    eval_parser.add_argument(
        "--evaluation-output",
        default="evaluation_results",
        help="Destination for complete evaluation artifacts (default: evaluation_results)"
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # retrieve command
    retrieve_parser = subparsers.add_parser(
        'retrieve',
        help='Perform retrieval or RAG query'
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

    # Optional: rag command as an alias or grouping
    rag_parser = subparsers.add_parser(
        'rag',
        help='RAG system for embedding, evaluation, and retrieval (alias for subcommands)'
    )
    rag_subparsers = rag_parser.add_subparsers(dest='rag_subcommand', help='RAG subcommands')

    # Rag embed-all
    rag_embed_parser = rag_subparsers.add_parser(
        'embed-all',
        help='Embed all JSON corpora and create vector indices'
    )
    rag_embed_parser.add_argument(
        "--embedding-model",
        default="bge-small",
        help="Embedding alias or HuggingFace model name to use"
    )
    rag_embed_parser.add_argument("--embedding-models", nargs="+")
    rag_embed_parser.add_argument("--all-embedding-models", action="store_true")
    rag_embed_parser.add_argument("--force", action="store_true")
    rag_embed_parser.add_argument(
        "--representation-mode",
        choices=["raw", "enriched", "semantic", "compiler", "hybrid"],
        default="hybrid",
        help="Document representation mode"
    )
    rag_embed_parser.add_argument(
        "--output-dir", "-o",
        default="/home/igorr/Projects/op-python/optima/output",
        help="Directory containing Optima JSON output files"
    )
    rag_embed_parser.set_defaults(func=cmd_embed_all)

    # Rag evaluate
    rag_eval_parser = rag_subparsers.add_parser(
        'evaluate',
        help='Evaluate retrieval performance across corpora'
    )
    rag_eval_parser.add_argument(
        "--embedding-model",
        default="bge-small",
        help="Embedding alias or HuggingFace model name used for indices"
    )
    rag_eval_parser.add_argument("--embedding-models", nargs="+")
    rag_eval_parser.add_argument("--all-embedding-models", action="store_true")
    rag_eval_parser.add_argument("--corpora", nargs="+")
    rag_eval_parser.add_argument("--enrichment-models", nargs="+")
    rag_eval_parser.add_argument("--difficulty", choices=["easy", "moderate", "difficult"])
    rag_eval_parser.add_argument("--categories", nargs="+")
    rag_eval_parser.add_argument(
        "--benchmark",
        help="Path to benchmark queries JSON file (optional)"
    )
    rag_eval_parser.add_argument(
        "--create-benchmark",
        action="store_true",
        help="Create new benchmark queries from JSON files"
    )
    rag_eval_parser.add_argument(
        "--num-queries",
        type=int,
        default=20,
        help="Number of benchmark queries to create (if creating benchmark) - for backward compatibility"
    )
    rag_eval_parser.add_argument(
        "--num-queries-per-category",
        type=int,
        default=None,
        help="Number of benchmark queries to create PER CATEGORY (8 categories). Overrides --num-queries if specified."
    )
    rag_eval_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible benchmark generation (default: 42)"
    )
    rag_eval_parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to retrieve per query"
    )
    rag_eval_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-query retrieval ranks and scores"
    )
    rag_eval_parser.add_argument(
        "--debug-limit",
        type=int,
        default=20,
        help="Maximum number of queries to print per corpus in debug mode"
    )
    rag_eval_parser.add_argument(
        "--output-dir", "-o",
        default="/home/igorr/Projects/op-python/optima/output",
        help="Directory containing Optima JSON output files"
    )
    rag_eval_parser.add_argument(
        "--complete",
        dest="complete_evaluation",
        action="store_true",
        help="Build the publication evaluation from existing outputs without rerunning retrieval"
    )
    rag_eval_parser.add_argument(
        "--evaluation-output",
        default="evaluation_results",
        help="Destination for complete evaluation artifacts (default: evaluation_results)"
    )
    rag_eval_parser.set_defaults(func=cmd_evaluate)

    # Rag retrieve
    rag_retrieve_parser = rag_subparsers.add_parser(
        'retrieve',
        help='Perform retrieval or RAG query'
    )
    rag_retrieve_parser.add_argument(
        "--embedding-model",
        default="bge-small",
        help="Embedding alias or HuggingFace model name used for indices"
    )
    rag_retrieve_parser.add_argument(
        "--corpus",
        choices=["raw"],
        help="Query the raw baseline corpus"
    )
    rag_retrieve_parser.add_argument(
        "--enrichment-model",
        help="Query an enriched corpus (specify model name)"
    )
    rag_retrieve_parser.add_argument(
        "--query",
        required=True,
        help="Query text"
    )
    rag_retrieve_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to retrieve"
    )
    rag_retrieve_parser.add_argument(
        "--rag-mode",
        action="store_true",
        help="Use RAG pipeline with LLM (retrieval-only if not specified)"
    )
    rag_retrieve_parser.add_argument(
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity for output"
    )
    rag_retrieve_parser.add_argument(
        "--output-dir", "-o",
        default="/home/igorr/Projects/op-python/optima/output",
        help="Directory containing Optima JSON output files"
    )
    rag_retrieve_parser.set_defaults(func=cmd_retrieve)

    args = parser.parse_args()

    # Setup logging if we have verbose flag (for RAG commands)
    if hasattr(args, 'verbose'):
        log_level = "DEBUG" if args.verbose >= 2 else "INFO" if args.verbose >= 1 else "WARNING"
        setup_logging(log_level)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the appropriate command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # Handle rag subcommands
        if args.command == 'rag' and hasattr(args, 'rag_subcommand'):
            if args.rag_subcommand == 'embed-all':
                cmd_embed_all(args)
            elif args.rag_subcommand == 'evaluate':
                cmd_evaluate(args)
            elif args.rag_subcommand == 'retrieve':
                cmd_retrieve(args)
            else:
                rag_parser.print_help()
                sys.exit(1)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)

if __name__ == '__main__':
    sys.exit(main())