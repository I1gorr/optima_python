#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from optima.rag.evaluation import create_benchmark_from_json_files
from pathlib import Path
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

print("Testing benchmark creation...")
output_dir = Path("/home/igorr/Projects/op-python/optima/output")

queries = create_benchmark_from_json_files(
    output_dir,
    20,  # num_queries
    (0.4, 0.35, 0.25),  # difficulty_distribution
    42  # seed
)

print(f"Generated {len(queries)} queries")
if queries:
    print("First query:")
    print(queries[0])
else:
    print("No queries generated!")