#!/usr/bin/env python3

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optima.rag.evaluation import OptimaRetrievalEvaluator
from optima.rag.embedding_simple import OptimaEmbedder
from langchain_community.vectorstores import FAISS
from pathlib import Path

# Let's manually run the evaluation steps to see where the error occurs

print("Setting up evaluator...")
evaluator = OptimaRetrievalEvaluator()

print("Loading vector store...")
output_dir = Path("/home/igorr/Projects/op-python/optima/output")
embedding_model = "BAAI/bge-small-en-v1.5"
indices_dir = output_dir / "rag_report" / "indexes" / embedding_model

if not indices_dir.exists():
    print(f"Indices directory does not exist: {indices_dir}")
    sys.exit(1)

embedder = OptimaEmbedder(embedding_model_name=embedding_model)

# Load raw corpus
raw_index_path = indices_dir / "raw"
if not raw_index_path.exists():
    print(f"Raw index path does not exist: {raw_index_path}")
    sys.exit(1)

print("Loading raw vector store...")
vector_store = embedder.load_vector_index(raw_index_path)
if not vector_store:
    print("Failed to load vector store")
    sys.exit(1)

print("Raw vector store loaded successfully")

# Load or create benchmark
print("Loading/creating benchmark...")
from optima.rag.evaluation import create_benchmark_from_json_files, save_benchmark_queries, load_benchmark_queries
from optima.rag.benchmark_generator import create_benchmark_from_json_files as benchmark_create

benchmark_path = output_dir / "rag_report" / "benchmark" / "queries.json"
benchmark_path.parent.mkdir(parents=True, exist_ok=True)

if not benchmark_path.exists():
    print("Creating benchmark queries...")
    queries = create_benchmark_from_json_files(
        output_dir,
        20,  # num_queries
        (0.4, 0.35, 0.25),  # difficulty_distribution
        42  # seed
    )
    save_benchmark_queries(queries, benchmark_path)
    print(f"Benchmark saved to {benchmark_path}")
else:
    print(f"Loading benchmark from {benchmark_path}")
    queries = load_benchmark_queries(benchmark_path)

print(f"Loaded {len(queries)} benchmark queries")

# Let's examine the first few queries to see their structure
print("\nFirst query structure:")
if queries:
    first_query = queries[0]
    for key, value in first_query.items():
        print(f"  {key}: {value} (type: {type(value)})")

# Now run evaluation
print("\nRunning evaluation...")
try:
    results = evaluator.evaluate_retrieval(vector_store, queries, k=10)
    print(f"Evaluation completed successfully!")
    print(f"Results keys: {list(results.keys())}")

    # Check types of all values
    print("\nResult value types:")
    for key, value in results.items():
        print(f"  {key}: {value} (type: {type(value)})")

except Exception as e:
    print(f"Evaluation failed: {e}")
    import traceback
    traceback.print_exc()

    # Let's also try to see what happens if we call evaluate_multiple_corpora
    print("\nTrying evaluate_multiple_corpora...")
    try:
        from optima.rag.evaluation import load_corpus_indices
        corpus_indices = load_corpus_indices(output_dir / "rag_report", embedding_model)
        print(f"Loaded {len(corpus_indices)} corpus indices")

        if corpus_indices:
            multi_results = evaluator.evaluate_multiple_corpora(corpus_indices, queries, k=10)
            print(f"Multi-corpus evaluation completed!")
            print(f"Results keys: {list(multi_results.keys())}")

            # Check first corpus
            first_corpus = list(multi_results.keys())[0]
            first_result = multi_results[first_corpus]
            print(f"\nFirst corpus '{first_corpus}' result types:")
            for key, value in first_result.items():
                print(f"  {key}: {value} (type: {type(value)})")

    except Exception as e2:
        print(f"Multi-corpus evaluation also failed: {e2}")
        traceback.print_exc()