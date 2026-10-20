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
- [Research RAG Pipelines and Ablation Benchmark](#research-rag-pipelines-and-ablation-benchmark)

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

## Research RAG Pipelines and Ablation Benchmark

`optima/rag/research/` adds three named pipelines and a reproducible evaluation framework
that measures whether LLM enrichment, lexical retrieval, compiler-graph expansion and
cross-encoder reranking each improve function retrieval, and what they cost. It sits next to
the legacy FAISS flow (`embed-all` / `evaluate` / `retrieve`), which is unchanged, and reuses
its embedding registry (`optima/rag/embedding_simple.py`) and the shared metric code in
`evaluation/metrics/retrieval.py`.

Install the extras once: `pip install -e ".[research]"` (sentence-transformers, torch,
matplotlib, psutil). Everything except real embedding/reranker models also runs without them
(`--embedding-model hashing` is an offline, non-semantic encoder for smoke tests).

### The three pipelines

| Pipeline | Representations | Retrieval | Graph | Rerank |
|---|---|---|---|---|
| `baseline_rag` | raw (signature + source from `base.json`) | dense only | off | off |
| `semantic_enrichment_rag` | semantic (purpose, summary, behavior, inputs, outputs, side effects, concepts, keywords, algorithm from the enriched JSON) | dense (BM25 optional) | off | off |
| `compiler_hybrid_rag` | raw + semantic + compiler (calls, callers, scope, CFG summary), one index each | dense + BM25 + exact symbol, fused with RRF | bounded call-graph expansion | optional cross-encoder |

Every stage is an independent switch in `PipelineConfig` (`configs/pipelines/*.json`);
turning one off never changes another. Functions with missing or failed enrichment are kept
(identity header only by default, `representation.missing_semantic`) and counted in the
coverage report. Unlike the legacy loader, a model that fails to load is an error, never a
silent fall-back to mock embeddings.

### Run one pipeline

```bash
python -m optima.rag run-pipeline --pipeline baseline_rag \
  --base-json optima_outputs/base/chess-engine-in-cpp-master/base.json \
  --embedding-model bge-base --benchmark benchmarks/chess/structural_v1.json

python -m optima.rag run-pipeline --pipeline compiler_hybrid_rag \
  --base-json optima_outputs/base/chess-engine-in-cpp-master/base.json \
  --enriched-json "output/chess Engine/enhanced_qwen-qwen2.5-coder-14b-instruct-chess.json" \
  --embedding-model bge-base --benchmark benchmarks/chess/structural_v1.json \
  --set graph.max_depth=2 --set rerank.enabled=true --set rerank.model=ms-marco-minilm
```

`--set section.key=value` overrides any configuration field; `--config file.json` loads a
full configuration. Configuration keys and defaults are in
`optima/rag/research/config.py`; a stored result contains the exact configuration and its
`config_hash`.

### Run the complete benchmark

```bash
python -m optima.rag run-suite --spec configs/suites/chess.json --dry-run   # list planned runs
python -m optima.rag -v run-suite --spec configs/suites/chess.json          # run everything
python -m optima.rag run-suite --spec configs/suites/chess.json --resume    # skip finished runs
python -m optima.rag report --suite-dir experiments/chess-v1                # rebuild tables/plots
```

A suite spec names datasets (a `base.json`, a benchmark, enriched JSON files), embedding
models, rerankers and ablation groups (`pipelines`, `enrichment`, `representations`,
`hybrid`, `graph`, `rerank`, `embedding`). Each group varies one factor against a declared
reference and reports paired-bootstrap deltas. Results go to `experiments/<suite_id>/`:

```
suite.json  aggregate.csv  per_category.csv  per_query_metrics.csv  comparisons.csv
comparisons_by_category.csv  report.md  plots/*.png
runs/<experiment_id>/result.json        # config, models, dataset/index metadata, metrics, timings, errors
runs/<experiment_id>/per_query.jsonl.gz # ranked function IDs, scores, representation/method provenance
```

Experiment IDs are unique and existing runs are never overwritten.

### Metrics

Computed per query, then averaged over queries that have at least one relevant function
(queries without judgments are listed under `skipped`, not counted as zero):
Recall@{1,3,5,10} (relevant found in top-k / all relevant), Precision@k (relevant in top-k / k),
Hit@k, MRR (reciprocal rank of the first relevant result within the returned top-10),
nDCG@k (graded gains, ideal ranking built from the judgments). Rankings are de-duplicated
by function ID. Also recorded: per-stage query latency, embedding/index build time, index
size, process RSS and GPU peak, candidate counts before/after graph expansion, candidate-pool
recall, rerank latency, enrichment tokens/time and failed or incomplete enrichment records.

### Cross-encoder rerankers

`python -m optima.rag rerankers` lists them. Models are downloaded from the Hugging Face hub on
first use (internet required); check the licence before redistributing.

| Alias | Model | Licence | Size | Hardware |
|---|---|---|---|---|
| `ms-marco-minilm` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache-2.0 | 22.7 M | CPU fine, ~90 MB |
| `bge-reranker-base` | `BAAI/bge-reranker-base` | MIT | 278 M | GPU preferred (<2 GB VRAM), ~1.1 GB |
| `bge-reranker-v2-m3` | `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 568 M | GPU recommended, ~2.3 GB |

None is code-specific. If a reranker cannot load, the run is recorded as failed, never
replaced by another scorer.

### Benchmarks and limitations

See `benchmarks/README.md` for how ground truth was built and its caveats (structural
categories share the call graph used by graph expansion; behavioural labels are single-annotator;
no dependency ground truth exists because `dependencies` is empty in the bundled data).
Embedding models that need `trust_remote_code` (nomic, CodeRankEmbed) are not loaded unless
`OPTIMA_TRUST_REMOTE_CODE=1` is set. Tests: `python -m unittest discover -s tests`.

## Experiment Analytics (`optima_analytics`)

Read-only analysis of everything in the experiment store (`experiments/*/runs/*/result.json`) and the
enrichment artifacts (`output/**/enhanced_*.json`): unified validated tables, paired statistics with
multiple-comparison correction, rankings, plots, Markdown reports and a self-contained dashboard.
Install the extra with `pip install -e ".[analytics]"` (pandas, scipy, matplotlib).

```bash
# check inputs only: manifest + data-quality report (exit 1 with --strict when errors exist)
python -m optima_analytics validate --input-dir experiments --output-dir reports/optima_analytics

# everything: CSVs, statistical_tests.csv, plots/, plot_data/, markdown reports, index.html
python -m optima_analytics generate --input-dir experiments --output-dir reports/optima_analytics

# figures only (optionally a subset by name substring)
python -m optima_analytics plots --output-dir reports/optima_analytics --only A01 F36
```

Useful options: `--input-dir` / `--enriched-dir` (repeatable), `--base-dir`, `--datasets`, `--n-boot`,
`--n-perm`, `--seed`, `--no-legacy`, `--force`. `generate` refuses to overwrite an existing report and
refuses an output directory inside an input directory. Open `reports/optima_analytics/index.html`
directly in a browser (no server, no external resources).

What it does and does not do:

* Runs repeated with the same dataset/configuration/corpus are counted once (and checked to be identical);
  enrichment files whose recorded `base_sha256` does not match their project's `base.json`, and runs that
  used them, are flagged and excluded from enrichment analyses (never modified or deleted).
* Enrichment effects are measured against a *matched no-enrichment counterpart* of the same retrieval stack
  (raw dense, raw dense+BM25, raw+compiler stack), so retrieval-stack effects are not credited to enrichment.
  The crossed grid behind this comes from the `grid` ablation group (`configs/suites/*-grid.json`).
* MRR is the only confirmatory metric; p-values are cluster-level (target function) sign-flip permutation
  tests, intervals are cluster bootstraps, Holm and Benjamini-Hochberg corrections are applied within named
  families. Model-level correlations (n = number of models) are descriptive only.
* Model sizes/families come from `optima_analytics/model_catalog.json` (model-card values, `unknown` when not
  documented); nothing is inferred from a model name.

Tests: `python -m unittest discover -s tests -p "test_analytics_*.py"`.

### Query routing and instruction prefixes (research pipelines)

`routing` (config section, off by default; hash of existing configs unchanged) classifies each query from its text
(`optima/rag/research/routing.py`: `literal` / `structural` / `symbol` / `natural`) and applies a per-class fusion
profile (dense / BM25 / exact / graph / per-representation weights). Profiles live in `configs/routing_profiles.json`
(tuned with `scripts/tune_routing.py` on tuning data only); suites enable them with the `improved` ablation group and
`routing_profiles_file`. Query instruction prefixes per embedding model go through the suite's `embedding_prefixes`.

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