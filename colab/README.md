# Optima on Google Colab

`optima_colab.ipynb` is the primary interface for running the existing Optima
experiment on a Colab GPU. It keeps the local Optima JSON schema, enrichment
objective, embedding aliases, FAISS index layout, retrieval implementation,
evaluation metrics, CSV reports, and plotting code. The only provider-specific
adapter is Hugging Face Transformers for GPU-based enrichment.

## Exact workflow

1. Open `colab/optima_colab.ipynb` in Google Colab.
2. In **Runtime > Change runtime type**, select a GPU.
3. Run the setup cells in order.
4. Upload `base.json` or an already enriched JSON artifact.
5. Inspect the uploaded schema and choose `MODEL_ID`, `INPUT_JSON`, and
   `EMBEDDING_MODEL` in the configuration cell.
6. Run enrichment when the uploaded artifact has no enrichment, or skip it when
   an enriched JSON is already available.
7. Save enriched JSON under `colab_outputs/enrichment/<model>/`.
8. Build the raw and enriched embedding indexes using the existing Optima
   embedding registry (`bge-small`, `bge-base`, `nomic`, `qwen3-0.6b`,
   `coderank`, or `mock`).
9. Run retrieval and evaluation with the existing benchmark and metrics.
10. Inspect Recall@5, MRR, latency, CSV files, and plots.
11. Download individual artifacts or a ZIP of `colab_outputs/`.

The notebook can be resumed at any stage. Uploading an existing enriched JSON
skips completed nodes, and existing FAISS indexes are reused unless
`FORCE_REBUILD = True`. Existing benchmark queries can also be supplied instead
of generating a new benchmark. Outputs are written to `colab_outputs/`; source
uploads are copied and never overwritten.

## Notes

- `MODEL_ID` accepts any compatible Hugging Face causal language model. The
  default is `ibm-granite/granite-3.3-8b-instruct`.
- 4-bit BitsAndBytes quantization is enabled by default. Disable it only when
  the selected model and runtime have enough GPU memory.
- The notebook stops before model loading if CUDA is unavailable rather than
  silently running a large model on CPU.
- Embeddings, document construction, FAISS indexing, retrieval, evaluation, and
  plots are delegated to the existing `optima.rag` modules. This is deliberate:
  enrichment-model comparisons keep the downstream configuration controlled.
- To use the notebook from a cloned repository, the setup cell adds the
  repository root to `sys.path`; no manual path edits are required.
