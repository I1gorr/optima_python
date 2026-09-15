# Optima RAG Evaluation Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/I1gorr/optima_python/blob/main/colab/optima_colab.ipynb)

This project implements a Retrieval-Augmented Generation (RAG) pipeline to evaluate whether LLM enrichment improves code retrieval performance. It compares raw baseline JSON against LLM-enriched JSON files using the same embedding model, queries, chunking, and retrieval configuration.

## Google Colab

Optima includes a Google Colab workflow for GPU-based LLM enrichment and
evaluation. It accepts existing `base.json` or enriched JSON artifacts, then
continues through the controlled embedding, FAISS retrieval, evaluation, and
plotting pipeline without requiring local Ollama, LM Studio, Docker, or a web
server.

[Open Optima in Google Colab](https://colab.research.google.com/github/I1gorr/optima_python/blob/main/colab/optima_colab.ipynb)

The notebook and its resumable workflow are documented in
[`colab/README.md`](colab/README.md).

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running the Full Embedding Matrix Experiment](#running-the-full-embedding-matrix-experiment)
- [Detailed Usage](#detailed-usage)
- [Understanding Results](#understanding-results)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8+
- Git
- Approximately 2GB free disk space
- (Optional for embeddings) ~1GB for sentence-transformers models

## Installation

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd optima

# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install the package in development mode
pip install -e .

# For real embeddings (recommended for meaningful comparisons)
pip install sentence-transformers torch

# Verify installation
python -c "import sentence_transformers; import torch; print('Dependencies OK')"
```

## Quick Start

Once the virtual environment is activated and the package is installed, you can use the `optima` command directly.

### Option 1: Test Pipeline Structure (Mock Embeddings)
```bash
# Process all JSON corpora
optima embed-all --output-dir output --representation-mode hybrid

# Create benchmark and evaluate
optima evaluate --output-dir output --create-benchmark --num-queries 15 --k 10

# Test retrieval
optima retrieve --output-dir output --corpus raw --query "What does the Puzzle function do?" --k 5
```

## Running the Full Embedding Matrix Experiment

Run these commands from the repository root after activating the virtual
environment:

```bash
source .venv/bin/activate
```

### 1. Analyze a project

```bash
optima analyze /path/to/project -o ./output
```

This creates `output/base.json`.

### 2. Generate enriched corpora

The enrichment command uses an LM Studio OpenAI-compatible server. Start LM
Studio with the requested model loaded, then run one command per enrichment
model:

```bash
optima enrich ./output/base.json \
  --model llama-3.2-3b-instruct \
  -o ./output

optima enrich ./output/base.json \
  --model qwen2.5-coder-3b-instruct \
  -o ./output
```

This creates files such as:

```text
output/enhanced_llama-3.2-3b-instruct.json
output/enhanced_qwen2.5-coder-3b-instruct.json
```

### 3. Build embedding indexes

For an offline smoke test:

```bash
optima embed-all \
  --output-dir ./output \
  --embedding-models mock bge-small
```

For the full experiment:

```bash
optima embed-all \
  --output-dir ./output \
  --embedding-models bge-small bge-base nomic qwen3-0.6b coderank
```

Indexes are stored independently under:

```text
output/rag_report/indexes/<enrichment-model>/<embedding-model>/
```

Existing indexes are reused. Rebuild them with:

```bash
optima embed-all \
  --output-dir ./output \
  --embedding-models bge-small bge-base nomic qwen3-0.6b coderank \
  --force
```

### 4. Evaluate every enrichment/embedding combination

Use the existing benchmark:

```bash
optima evaluate \
  --output-dir ./output \
  --benchmark ./output/rag_report/benchmark/queries.json \
  --embedding-models bge-small bge-base nomic qwen3-0.6b coderank \
  --k 10
```

To create a benchmark during evaluation instead:

```bash
optima evaluate \
  --output-dir ./output \
  --create-benchmark \
  --num-queries 1600 \
  --embedding-models bge-small bge-base nomic qwen3-0.6b coderank \
  --k 10
```

Evaluate only difficult queries or selected corpora with:

```bash
optima evaluate \
  --output-dir ./output \
  --benchmark ./output/rag_report/benchmark/queries.json \
  --embedding-model bge-small \
  --enrichment-models raw llama-3.2-3b-instruct \
  --difficulty difficult \
  --k 10
```

For per-query debugging:

```bash
optima evaluate \
  --output-dir ./output \
  --benchmark ./output/rag_report/benchmark/queries.json \
  --embedding-model bge-small \
  --enrichment-models raw llama-3.2-3b-instruct \
  --k 10 \
  --debug \
  --debug-limit 20
```

### 5. Inspect the results

Reports are written to `output/rag_report/results/`:

```text
embedding_comparison.csv
embedding_by_difficulty.csv
embedding_by_category.csv
enrichment_gains.csv
embedding_gains.csv
best_combinations.csv
best_embedding_per_enrichment.csv
best_enrichment_per_embedding.csv
```

Detailed per-query rankings are saved to:

```text
output/rag_report/per_query_results.csv
```

Visualizations are saved to:

```text
output/rag_report/results/plots/
```

Useful commands for viewing CSV results:

```bash
column -s, -t < output/rag_report/results/embedding_comparison.csv | less -S
column -s, -t < output/rag_report/results/best_combinations.csv | less -S
```

### 6. Run a manual retrieval query

Query the raw baseline:

```bash
optima retrieve \
  --output-dir ./output \
  --embedding-model bge-small \
  --corpus raw \
  --query "Where should I investigate the logic responsible for determining legal movement?" \
  --k 10
```

Query an enriched corpus:

```bash
optima retrieve \
  --output-dir ./output \
  --embedding-model bge-small \
  --enrichment-model llama-3.2-3b-instruct \
  --query "Where should I investigate the logic responsible for determining legal movement?" \
  --k 10
```

### Option 2: Real Embeddings (Meaningful Comparisons)
```bash
# Install required packages for real embeddings
pip install sentence-transformers torch

# Run embedding with real model
optima embed-all --embedding-model BAAI/bge-small-en-v1.5 --output-dir output --representation-mode hybrid

# Create benchmark and evaluate
optima evaluate --embedding-model BAAI/bge-small-en-v1.5 --output-dir output --create-benchmark --num-queries 15 --k 10

# Compare results across enrichment models
```

## Detailed Usage

### Embedding Corpora
```bash
optima embed-all \
  --embedding-model BAAI/bge-small-en-v1.5 \  # Default: BAAI/bge-small-en-v1.5
  --output-dir output \                       # Directory containing JSON files
  --representation-mode hybrid                # Choices: raw, enriched, semantic, compiler, hybrid
```

### Evaluation
```bash
optima evaluate \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --output-dir output \
  --create-benchmark \          # Create new benchmark (omit to load existing)
  --num-queries 20 \            # Number of benchmark queries to create
  --k 10                        # Number of results to retrieve per query
```

### Retrieval Queries
```bash
# Raw baseline corpus
optima retrieve \
  --output-dir output \
  --corpus raw \
  --query "How does the static evaluation function work?" \
  --k 5

# Specific enrichment model
optima retrieve \
  --output-dir output \
  --enrichment-model llama-3.2-3b-instruct \
  --query "How does the static evaluation function work?" \
  --k 5

# With RAG pipeline (requires LLM dependencies)
optima retrieve \
  --output-dir output \
  --corpus raw \
  --query "Explain the perft algorithm" \
  --k 5 \
  --rag-mode
```

## Understanding Results

After running evaluation, examine the `rag_report/` directory:

### Key Output Files
1. `embedding_results.json` - Summary of embedding process for each corpus
2. `benchmark/queries.json` - Generated benchmark queries with ground truth
3. `indexes/` - FAISS vector indices for each corpus (organized by embedding model)
4. `results/` - Evaluation outputs:
   - `retrieval_model_comparison.csv` - Side-by-side model metrics
   - `baseline_improvement.csv` - Improvement of each model over raw baseline
   - `model_ranking.csv` - Model rankings by Recall@5, MRR, Hit Rate@5
   - `retrieval_evaluation_TIMESTAMP.json` - Detailed results per run

### Metrics Explained
- **Recall@k**: Proportion of relevant functions retrieved in top k results
- **Precision@k**: Proportion of retrieved functions that are relevant
- **Hit Rate@k**: Binary - did we retrieve at least one relevant function in top k?
- **MRR**: Mean Reciprocal Rank - average of reciprocal ranks of first relevant result
- **Latency**: Average query processing time

### Comparing Enrichment Models
1. Check `baseline_improvement.csv` for absolute/relative improvements over raw
2. Review `model_ranking.csv` to see top performers by metric
3. Examine `retrieval_model_comparison.csv` for full metric breakdown

## Customization

### Changing Embedding Models
```bash
# Registry aliases (indexes are stored under rag_report/indexes/<corpus>/<alias>)
optima embed-all --embedding-model bge-small
optima embed-all --embedding-models bge-small bge-base --force
optima embed-all --all-embedding-models

# The built-in aliases are bge-small, bge-base, nomic, qwen3-0.6b, and coderank.
# A HuggingFace model identifier is still accepted for compatibility.
optima embed-all --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Existing indexes are skipped unless `--force` is supplied.  Each index includes
`metadata.json`.  Evaluation can compare the full enrichment x embedding
matrix and optionally filter it:

```bash
optima evaluate --embedding-models bge-small bge-base \
  --corpora raw llama-3.2-3b-instruct --difficulty difficult
```

The evaluation report includes aggregate, difficulty/category, enrichment
gain, embedding gain, weighted-score, best-table, and per-query CSV files.

### Adjusting Representation Modes
- `raw`: Original JSON structure
- `enriched`: LLM-added fields only
- `semantic`: Natural language summaries
- `compiler`: AST-like structural representation
- `hybrid`: Combination of above (default)

### Modifying Benchmark Generation
Edit `optima/rag/evaluation.py` to adjust:
- Query generation logic in `create_benchmark_queries()`
- Function selection criteria
- Natural language templates

## Troubleshooting

### Common Issues
1. **"Could not import sentence_transformers"**
   - Solution: `pip install sentence-transformers torch`

2. **FAISS loading errors**
   - Solution: Ensure indices exist in `rag_report/indexes/` and run embedding first

3. **CUDA/GPU issues**
   - Solution: The code defaults to CPU. For GPU, ensure proper CUDA setup and modify embedding initialization

4. **Memory constraints**
   - Solution: Reduce `--num-queries` or increase chunk size in `OptimaEmbedder`

### Logging
Increase verbosity with `-v` flags:
- `-v`: INFO level
- `-vv`: DEBUG level
- `-vvv`: More detailed DEBUG

Example:
```bash
optima evaluate -vvv --output-dir output --create-benchmark
```

## Next Steps

1. **Install real embeddings** for meaningful model comparisons
2. **Experiment with different embedding models** (BGE variants, MiniLM, etc.)
3. **Tune representation modes** for specific code understanding tasks
4. **Add more sophisticated evaluation metrics** (nDCG, MAP, etc.)
5. **Create visualizations** of results using the generated CSV files
6. **Extend to other programming languages** by adapting the document constructor

## Example Commands for Full Experiment

```bash
# Install real embedding dependencies
pip install sentence-transformers torch

# Process all corpora with hybrid representation
optima embed-all \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --output-dir output \
  --representation-mode hybrid

# Create benchmark with 25 queries
optima evaluate \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --output-dir output \
  --create-benchmark \
  --num-queries 25 \
  --k 10

# Retrieve examples from different models
optima retrieve --output-dir output --corpus raw --query "Explain the transposition table" --k 3
optima retrieve --output-dir output --enrichment-model qwen2.5-coder-3b-instruct --query "Explain the transposition table" --k 3

# Review results
cat rag_report/final_summary.txt
head -20 rag_report/results/retrieval_model_comparison.csv
```

## License

MIT License - see LICENSE file for details.

---

**Note**: The framework is designed to be extensible. For research purposes, consider:
1. Adding ablation studies for different enrichment techniques
2. Implementing cross-validation for more robust comparisons
3. Adding human evaluation components for qualitative assessment
4. Integrating with code execution benchmarks for functional correctness

For questions or contributions, please open an issue or submit a pull request. Happy researching!