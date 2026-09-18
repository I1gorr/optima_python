#!/usr/bin/env python3

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'optima', 'rag'))

from benchmark_generator import BenchmarkQueryGenerator

# Load some functions from base.json
base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'base.json')
with open(base_path, 'r') as f:
    data = json.load(f)

functions = []
for file_data in data.get("files", []):
    functions.extend(file_data.get("functions", []))

# Take first 10 functions for testing
test_functions = functions[:10]

print(f"Testing with {len(test_functions)} functions")

# Create generator
generator = BenchmarkQueryGenerator(seed=42)

# Generate queries
queries = generator.generate_benchmark_queries(test_functions, num_queries=10)

print(f"Generated {len(queries)} queries")

for i, query in enumerate(queries):
    print(f"\nQuery {i+1}:")
    print(f"  ID: {query.get('query_id', 'N/A')}")
    print(f"  Difficulty: {query.get('difficulty', 'N/A')}")
    print(f"  Category: {query.get('category', 'N/A')}")
    print(f"  Query: {query.get('query', 'N/A')}")
    print(f"  Relevant Functions: {query.get('relevant_functions', [])}")
    print(f"  Safe: {query.get('is_safe', 'N/A')}")
    if 'validation' in query:
        val = query['validation']
        print(f"  Lexical Overlap: {val.get('lexical_overlap', 0):.3f}")
        print(f"  Contains Function Name: {val.get('contains_function_name', False)}")
        print(f"  Contains Purpose Phrase: {val.get('contains_purpose_phrase', False)}")
        print(f"  Contains Summary Phrase: {val.get('contains_summary_phrase', False)}")
        print(f"  Contains Behavior Phrase: {val.get('contains_behavior_phrase', False)}")