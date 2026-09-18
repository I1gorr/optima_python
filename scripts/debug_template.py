#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optima.rag.benchmark_generator import BenchmarkQueryGenerator
from pathlib import Path
import json

# Load some functions to test with
output_dir = Path("/home/igorr/Projects/op-python/optima/output")
base_path = output_dir / "base.json"

with open(base_path, 'r') as f:
    data = json.load(f)

functions = []
for file_data in data.get("files", []):
    functions.extend(file_data.get("functions", []))

print(f"Loaded {len(functions)} functions")

# Take first 5 functions for testing
test_functions = functions[:5]

for i, func in enumerate(test_functions):
    print(f"\nFunction {i}:")
    print(f"  Name: {func.get('name', 'N/A')}")
    print(f"  Qualified name: {func.get('qualified_name', 'N/A')}")
    print(f"  Return type: {func.get('return_type', 'N/A')}")
    params = func.get('parameters', [])
    print(f"  Parameters: {[p.get('name', 'N/A') for p in params]}")
    print(f"  Dependencies: {func.get('dependencies', [])}")
    enrichment = func.get('enrichment', {})
    print(f"  Has enrichment: {bool(enrichment)}")
    if enrichment:
        print(f"    Purpose: {enrichment.get('purpose', 'N/A')[:50]}...")
        print(f"    Summary: {enrichment.get('summary', 'N/A')[:50]}...")
        print(f"    Behavior: {enrichment.get('behavior', 'N/A')[:50]}...")

# Create generator and test template generation
print("\n" + "="*50)
print("Testing template generation...")

generator = BenchmarkQueryGenerator(seed=42)

# Test with first function
func = test_functions[0]
print(f"\nTesting with function: {func.get('name', 'N/A')}")

# Test a few easy templates
easy_templates = generator.query_templates["easy"]
for category, templates in easy_templates.items():
    print(f"\nCategory: {category}")
    for j, template in enumerate(templates[:3]):  # Test first 3 templates
        print(f"  Template {j}: {template}")
        query = generator._generate_query_from_template(template, func)
        print(f"    Generated: {query}")
        if query:
            is_safe, validation = generator._is_query_safe(query, func)
            print(f"    Safe: {is_safe}")
            if not is_safe:
                print(f"    Validation: {validation}")

print("\n" + "="*50)
print("Testing full generation with debug...")

# Let's monkey patch to add debug prints
original_generate_queries_for_difficulty = generator._generate_queries_for_difficulty

def debug_generate_queries_for_difficulty(functions, count, templates, difficulty, used_function_indices):
    print(f"\nDEBUG: Generating {count} {difficulty} queries")
    print(f"DEBUG: Available functions: {len(functions)}")
    print(f"DEBUG: Used function indices: {len(used_function_indices)}")

    # Call original but wrap with debug
    result = original_generate_queries_for_difficulty(functions, count, templates, difficulty, used_function_indices)
    print(f"DEBUG: Generated {len(result)} {difficulty} queries")
    return result

generator._generate_queries_for_difficulty = debug_generate_queries_for_difficulty

# Test generation
queries = generator.generate_benchmark_queries(test_functions, num_queries=10, difficulty_distribution=(0.4, 0.35, 0.25))
print(f"\nTotal queries generated: {len(queries)}")