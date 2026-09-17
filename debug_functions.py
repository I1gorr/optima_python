#!/usr/bin/env python3

import json
from pathlib import Path

# Load base.json and check its structure
output_dir = Path("/home/igorr/Projects/op-python/optima/output")
base_path = output_dir / "base.json"

print(f"Loading {base_path}")
with open(base_path, 'r') as f:
    data = json.load(f)

print(f"Top-level keys: {list(data.keys())}")

if 'project' in data:
    print(f"Project: {data['project']}")

if 'files' in data:
    print(f"Number of files: {len(data['files'])}")

    # Check first few files
    for i, file_data in enumerate(data['files'][:3]):
        print(f"\nFile {i}:")
        print(f"  Path: {file_data.get('path', 'N/A')}")
        print(f"  Language: {file_data.get('language', 'N/A')}")
        functions = file_data.get('functions', [])
        print(f"  Number of functions: {len(functions)}")
        if functions:
            print(f"  First function: {functions[0].get('name', 'N/A')}")
        else:
            print(f"  No functions found")

# Count total functions
all_functions = []
for file_data in data.get("files", []):
    functions = file_data.get("functions", [])
    all_functions.extend(functions)
    print(f"File {file_data.get('path', 'unknown')}: {len(functions)} functions")

print(f"\nTotal functions found: {len(all_functions)}")

# Also check if there are any enhanced JSON files
print("\nChecking for enhanced JSON files:")
for json_file in output_dir.glob("enhanced_*.json"):
    print(f"Found: {json_file.name}")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        file_count = len(data.get("files", []))
        func_count = sum(len(f.get("functions", [])) for f in data.get("files", []))
        print(f"  Files: {file_count}, Functions: {func_count}")
    except Exception as e:
        print(f"  Error loading: {e}")