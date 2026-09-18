#!/usr/bin/env python3

import json
import sys
from pathlib import Path

# Add the optima package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optima.rag.benchmark_generator import validate_benchmark_queries, load_benchmark_queries

def main():
    output_dir = Path("/home/igorr/Projects/op-python/optima/output")
    benchmark_path = output_dir / "rag_report" / "benchmark" / "queries.json"
    
    print(f"Loading benchmark from {benchmark_path}")
    queries = load_benchmark_queries(benchmark_path)
    
    if not queries:
        print("No benchmark queries found!")
        return 1
    
    # Load all functions for validation
    all_functions = []
    base_path = output_dir / "base.json"
    if base_path.exists():
        with open(base_path, 'r') as f:
            data = json.load(f)
            for file_data in data.get("files", []):
                all_functions.extend(file_data.get("functions", []))
    
    # Process enhanced JSONs for variety
    for json_file in output_dir.glob("enhanced_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                for file_data in data.get("files", []):
                    all_functions.extend(file_data.get("functions", []))
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    print(f"Loaded {len(all_functions)} functions for validation")
    
    # Run validation
    print("Running validation...")
    validation_report = validate_benchmark_queries(queries, all_functions)
    
    # Save validation report
    validation_path = output_dir / "rag_report" / "benchmark" / "validation_report.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(validation_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print(f"Validation report saved to {validation_path}")
    
    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Total queries: {validation_report['total_queries']}")
    print(f"Safe queries: {validation_report['safe_queries']} ({validation_report['safe_queries']/validation_report['total_queries']*100:.1f}%)")
    print(f"Unsafe queries: {validation_report['unsafe_queries']} ({validation_report['unsafe_queries']/validation_report['total_queries']*100:.1f}%)")
    print(f"Average lexical overlap: {validation_report['average_lexical_overlap']:.3f}")
    print(f"Queries with high overlap (>0.3): {validation_report['queries_with_high_overlap']}")
    print(f"Empty relevant functions: {validation_report['empty_relevant_functions']}")
    print(f"Invalid function IDs: {validation_report['invalid_function_ids']}")
    print(f"Duplicate queries: {validation_report.get('duplicate_queries', 0)}")
    print(f"Identifier leaks: {validation_report.get('identifier_leaks', 0)}")
    print(f"Low-quality queries: {validation_report.get('low_quality_queries', 0)}")
    print(f"Average query quality: {validation_report.get('average_query_quality_score', 0.0):.3f}")
    
    print("\nDifficulty distribution:")
    for diff, count in validation_report['difficulty_distribution'].items():
        print(f"  {diff}: {count} ({count/validation_report['total_queries']*100:.1f}%)")
    
    print("\nCategory distribution:")
    for cat, count in sorted(validation_report['category_distribution'].items()):
        print(f"  {cat}: {count} ({count/validation_report['total_queries']*100:.1f}%)")
    
    return 0 if validation_report['unsafe_queries'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
