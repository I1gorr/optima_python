#!/usr/bin/env python3
"""
Optima CLI - LM Studio LLM enrichment tool
"""

import argparse
import sys
import os
from pathlib import Path
from .analyzer import analyze_project
from .enricher import enrich_analysis

def main():
    parser = argparse.ArgumentParser(
        description="Optima: LM Studio LLM enrichment tool for code analysis"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
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
        default='output',
        help='Output directory (default: output)'
    )
    
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
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        return analyze_command(args)
    elif args.command == 'enrich':
        return enrich_command(args)
    else:
        parser.print_help()
        return 1

def analyze_command(args):
    """Handle the analyze subcommand"""
    project_path = Path(args.project_path).resolve()
    requested_output = Path(args.output).resolve()
    output_dir = requested_output if requested_output.suffix.lower() != '.json' else requested_output.parent
    
    if not project_path.exists():
        print(f"Error: Project path '{project_path}' does not exist", file=sys.stderr)
        return 1
    
    if not project_path.is_dir():
        print(f"Error: Project path '{project_path}' is not a directory", file=sys.stderr)
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing project: {project_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        base_json_path = analyze_project(project_path, output_dir)
        print(f"Analysis complete. Base JSON saved to: {base_json_path}")
        print(f"To enrich with LM Studio, run:")
        print(f"  optima enrich {base_json_path}")
        return 0
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        return 1

def enrich_command(args):
    """Handle the enrich subcommand"""
    base_json_path = Path(args.base_json).resolve()
    requested_output = Path(args.output).resolve()
    output_dir = requested_output if requested_output.suffix.lower() != '.json' else requested_output.parent
    
    if not base_json_path.exists():
        print(f"Error: Base JSON file '{base_json_path}' does not exist", file=sys.stderr)
        return 1
    
    if not base_json_path.is_file():
        print(f"Error: Base JSON path '{base_json_path}' is not a file", file=sys.stderr)
        return 1
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Enriching: {base_json_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        enhanced_json_path = enrich_analysis(
            base_json_path,
            output_dir,
            model=args.model,
            output_path=requested_output if requested_output.suffix.lower() == '.json' else None,
            base_url=args.base_url,
            model_load_timeout=args.model_load_timeout,
            limit=args.limit,
            context_size=args.context_size,
            max_input_tokens=args.max_input_tokens,
            reserved_output_tokens=args.reserved_output_tokens,
            debug=args.debug,
        )
        print(f"Enrichment complete. Enhanced JSON saved to: {enhanced_json_path}")
        return 0
    except Exception as e:
        print(f"Error during enrichment: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
