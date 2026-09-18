#!/usr/bin/env python3

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optima.rag.evaluation import OptimaRetrievalEvaluator
from optima.rag.embedding_simple import OptimaEmbedder
from langchain_community.vectorstores import FAISS
from pathlib import Path
import numpy as np

# Create a simple test
print("Creating test evaluator...")
evaluator = OptimaRetrievalEvaluator()

# Create some dummy queries that match the expected format
dummy_queries = [
    {
        "query": "What does the test function do?",
        "relevant_functions": ["test_file.c::test_func"],
        "function_name": "test_func",
        "qualified_name": "test::test_func",
        "function_id": "test_file.c::test_func",
        "difficulty": "easy"
    },
    {
        "query": "How does the main function work?",
        "relevant_functions": ["main.cpp::main"],
        "function_name": "main",
        "qualified_name": "globals::main",
        "function_id": "main.cpp::main",
        "difficulty": "moderate"
    }
]

print(f"Created {len(dummy_queries)} dummy queries")

# Try to load a vector store (if available)
output_dir = Path("/home/igorr/Projects/op-python/optima/output")
embedding_model = "BAAI/bge-small-en-v1.5"
indices_dir = output_dir / "indexes" / embedding_model

print(f"Looking for indices in: {indices_dir}")

if indices_dir.exists():
    print("Found indices directory")
    embedder = OptimaEmbedder(embedding_model_name=embedding_model)

    # Try to load raw corpus
    raw_index_path = indices_dir / "raw"
    if raw_index_path.exists():
        print("Loading raw vector store...")
        vector_store = embedder.load_vector_index(raw_index_path)
        if vector_store:
            print("Raw vector store loaded successfully")

            # Try evaluation
            print("Running evaluation...")
            try:
                results = evaluator.evaluate_retrieval(vector_store, dummy_queries, k=3)
                print(f"Evaluation results: {results}")

                # Check types
                for key, value in results.items():
                    print(f"  {key}: {value} (type: {type(value)})")

            except Exception as e:
                print(f"Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Failed to load vector store")
    else:
        print(f"Raw index path does not exist: {raw_index_path}")
else:
    print(f"Indices directory does not exist: {indices_dir}")

    # Let's see what directories DO exist
    indexes_base = output_dir / "indexes"
    if indexes_base.exists():
        print("Contents of indexes directory:")
        for item in indexes_base.iterdir():
            print(f"  {item.name} (is_dir: {item.is_dir()})")
    else:
        print("Indexes base directory does not exist either")

print("Done.")