#!/usr/bin/env python3

import json
import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optima.rag.evaluation import OptimaRetrievalEvaluator, calculate_improvement_over_baseline
from optima.rag.embedding_simple import OptimaEmbedder
from pathlib import Path

# Monkey patch to add debugging
original_calculate_improvement_over_baseline = calculate_improvement_over_baseline

def debug_calculate_improvement_over_baseline(raw_metrics, enriched_metrics):
    print(f"DEBUG: raw_metrics keys: {list(raw_metrics.keys())}")
    print(f"DEBUG: enriched_metrics keys: {list(enriched_metrics.keys())}")

    # Check a few key values
    for key in ['recall_at_1', 'recall_at_5', 'mrr']:
        raw_val = raw_metrics.get(key, 'MISSING')
        enriched_val = enriched_metrics.get(key, 'MISSING')
        print(f"DEBUG: {key}: raw={raw_val} ({type(raw_val)}), enriched={enriched_val} ({type(enriched_val)})")

        if isinstance(raw_val, list):
            print(f"DEBUG: RAW VAL IS LIST for {key}: {raw_val}")
        if isinstance(enriched_val, list):
            print(f"DEBUG: ENRICHED VAL IS LIST for {key}: {enriched_val}")

    print("DEBUG: Calling original function...")
    try:
        result = original_calculate_improvement_over_baseline(raw_metrics, enriched_metrics)
        print(f"DEBUG: Success! Result keys: {list(result.keys())}")
        return result
    except Exception as e:
        print(f"DEBUG: Error in original function: {e}")
        print(f"DEBUG: Error type: {type(e)}")
        traceback.print_exc()
        raise

# Replace the function
import optima.rag.evaluation
optima.rag.evaluation.calculate_improvement_over_baseline = debug_calculate_improvement_over_baseline

# Now run the evaluation
print("Running evaluation with debugging...")

from optima.rag.__main__ import cmd_evaluate
import argparse

# Create args object
args = argparse.Namespace()
args.embedding_model = "BAAI/bge-small-en-v1.5"
args.benchmark = None
args.create_benchmark = True
args.num_queries = 20
args.k = 10
args.output_dir = "/home/igorr/Projects/op-python/optima/output"
args.verbose = 0

try:
    cmd_evaluate(args)
except Exception as e:
    print(f"Evaluation failed with error: {e}")
    traceback.print_exc()