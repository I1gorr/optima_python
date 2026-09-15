"""Google Colab adapter for the existing Optima enrichment and RAG pipeline.

This module intentionally delegates embeddings, indexing, retrieval, benchmark
generation, evaluation, and plotting to ``optima.rag``.  Only the LLM provider
is adapted from the local LM Studio client to a Hugging Face Transformers model.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm

from optima.enricher import (
    ENRICHMENT_JSON_SCHEMA,
    _dataset_metrics,
    _evaluation,
    _parse_json_object,
)
from optima.rag.embedding_simple import (
    EMBEDDING_REGISTRY,
    embed_corpus,
    embedding_alias,
)
from optima.rag.evaluation import (
    create_benchmark_from_json_files,
    evaluate_embedding_matrix,
    generate_visualizations,
    load_benchmark_queries,
    load_corpus_indices,
    save_benchmark_queries,
    save_best_combinations,
    save_best_tables,
    save_gain_csv,
    save_group_metrics_csv,
    save_matrix_csv,
    save_per_query_results,
)

logger = logging.getLogger(__name__)

ENRICHMENT_FIELDS = (
    "purpose", "behavior", "summary", "inputs", "outputs", "side_effects",
    "dependencies", "concepts", "keywords", "algorithm", "complexity",
)
STRING_FIELDS = ("purpose", "behavior", "summary", "algorithm")
ARRAY_FIELDS = (
    "inputs", "outputs", "side_effects", "dependencies", "concepts", "keywords",
)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load one Optima JSON artifact without changing its structure."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an Optima JSON object, got {type(value).__name__}")
    return value


def save_json(value: dict[str, Any], path: str | Path) -> Path:
    """Write an artifact, creating parent directories and never mutating input."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


def flatten_functions(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        function
        for file_data in data.get("files", [])
        for function in file_data.get("functions", [])
    ]


def inspect_json(path: str | Path) -> dict[str, Any]:
    """Print and return useful schema information for an uploaded artifact."""
    data = load_json(path)
    functions = flatten_functions(data)
    fields = sorted({key for function in functions for key in function})
    enrichment_fields = sorted({
        key for function in functions
        for key in function.get("enrichment", {})
        if key not in {"evaluation", "usage"}
    })
    info = {
        "path": str(path),
        "files": len(data.get("files", [])),
        "nodes": len(functions),
        "top_level_fields": sorted(data),
        "function_fields": fields,
        "enrichment_fields": enrichment_fields,
        "nested_structure": {
            "project": isinstance(data.get("project"), dict),
            "files": "list",
            "functions": "files[].functions[]",
            "enrichment": "functions[].enrichment",
        },
    }
    print(f"Number of nodes: {info['nodes']}")
    print(f"Available fields: {', '.join(fields) or '(none)'}")
    print(f"Nested structure: project -> files[] -> functions[]")
    print(f"Enrichment fields already present: {', '.join(enrichment_fields) or 'none'}")
    return info


def _compact(value: Any, limit: int = 4000) -> str:
    if not value:
        return "unavailable"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value)[:limit]


def build_enrichment_messages(function: dict[str, Any]) -> list[dict[str, str]]:
    """Use the local Optima v2 semantic objective for a Transformers model."""
    identity = {
        "id": function.get("id"),
        "name": function.get("name"),
        "qualified_name": function.get("qualified_name"),
        "source_range": function.get("source_location", {}),
        "signature": {
            "parameters": function.get("parameters", []),
            "return_type": function.get("return_type"),
        },
    }
    user = f"""Analyze this ONE function.

Generate a JSON object with these fields:
purpose, behavior, summary, inputs (array), outputs (array), side_effects (array),
dependencies (array), concepts (array), keywords (array), algorithm, and
complexity (object with time and space).
Use "unknown" when complexity cannot be determined reliably. Use empty arrays only
when no items are supported by the evidence.

Rules:
- Base the answer on the supplied implementation; source is the primary source of truth.
- Do not infer behavior solely from the function name or invent functionality.
- Do not describe unrelated functions, CFG edges, or call graph relationships.
- Keep behavior and purpose to 1-2 sentences.
- If evidence is insufficient, say "Insufficient implementation context."
- Return ONLY valid JSON, with no markdown fences or explanation.

FUNCTION:
{json.dumps(identity, indent=2, ensure_ascii=False)}

SOURCE:
{_compact(function.get("source_code"), 12000)}

LLVM IR:
{_compact(function.get("llvm_ir"), 5000)}

AST:
{_compact(function.get("ast"), 2000)}

CFG:
{_compact(function.get("cfg"), 2000)}

CALLS:
{_compact(function.get("calls"), 2000)}
"""
    system = (
        "You are a software code-analysis assistant. Analyze individual functions "
        "using actual source code and compiler-derived information. Produce concise, "
        "factual semantic descriptions. Never invent behavior, relationships, or "
        "compiler facts."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _valid_enrichment(value: Any) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "json_root_not_object"
    missing = [field for field in ENRICHMENT_FIELDS if field not in value]
    if missing:
        return False, f"missing_fields:{','.join(missing)}"
    if any(not isinstance(value[field], str) for field in STRING_FIELDS):
        return False, "wrong_field_type:string"
    if any(
        not isinstance(value[field], list)
        or any(not isinstance(item, str) for item in value[field])
        for field in ARRAY_FIELDS
    ):
        return False, "wrong_field_type:string_array"
    complexity = value["complexity"]
    if not isinstance(complexity, dict) or not all(
        isinstance(complexity.get(key), str) for key in ("time", "space")
    ):
        return False, "wrong_field_type:complexity"
    if not value["purpose"].strip() or not value["behavior"].strip():
        return False, "empty_required_string"
    return True, "valid"


def _fallback(reason: str, model_id: str, retries: int) -> dict[str, Any]:
    return {
        "purpose": "Insufficient implementation context.",
        "behavior": "Insufficient implementation context.",
        "summary": "Insufficient implementation context.",
        "inputs": [], "outputs": [], "side_effects": [], "dependencies": [],
        "concepts": [], "keywords": [], "algorithm": "unknown",
        "complexity": {"time": "unknown", "space": "unknown"},
        "model": model_id, "status": "failed",
        "evaluation": {"failure_reason": reason, "retry_count": retries},
    }


def load_model(model_id: str, load_in_4bit: bool = True):
    """Load a causal LM with Colab-friendly automatic device placement."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Select a Colab GPU runtime before loading "
            f"the requested model: {model_id}"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=kwargs["torch_dtype"],
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


def _generate(model: Any, tokenizer: Any, messages: list[dict[str, str]],
              max_new_tokens: int) -> tuple[str, int | None]:
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = "\n\n".join(f"{item['role']}: {item['content']}" for item in messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), int(generated.numel())


def enrich_nodes(
    input_json: str | Path,
    output_json: str | Path,
    model_id: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 768,
    retries: int = 2,
    limit: int | None = None,
) -> dict[str, Any]:
    """Enrich nodes with resumable writes; one failed node does not abort a run."""
    base = load_json(input_json)
    result = copy.deepcopy(base)
    existing = load_json(output_json) if Path(output_json).exists() else {}
    prior = {
        function.get("id"): function.get("enrichment")
        for function in flatten_functions(existing)
        if function.get("id") and function.get("enrichment", {}).get("status") == "completed"
    }
    functions = flatten_functions(result)
    selected = functions[:limit] if limit is not None else functions
    start = time.perf_counter()
    failures: list[dict[str, str]] = []
    latencies: list[float] = []
    token_counts: list[int] = []
    for function in tqdm(selected, desc=f"Enriching with {model_id}"):
        node_start = time.perf_counter()
        if function.get("id") in prior:
            function["enrichment"] = prior[function["id"]]
            continue
        enrichment = None
        reason = "unknown"
        used_retries = 0
        for attempt in range(retries + 1):
            used_retries = attempt
            try:
                text, generated_tokens = _generate(
                    model, tokenizer, build_enrichment_messages(function), max_new_tokens
                )
                parsed, parse_reason = _parse_json_object(text)
                valid, schema_reason = _valid_enrichment(parsed)
                if valid:
                    enrichment = parsed
                    token_counts.append(generated_tokens or 0)
                    reason = "valid"
                    break
                reason = schema_reason if parsed is not None else parse_reason
            except Exception as exc:  # node-level isolation is intentional
                reason = f"{type(exc).__name__}:{exc}"
            if attempt < retries:
                time.sleep(1)
        elapsed = time.perf_counter() - node_start
        latencies.append(elapsed)
        generated = token_counts[-1] if token_counts else None
        if enrichment is None:
            enrichment = _fallback(reason, model_id, used_retries)
            failures.append({"id": str(function.get("id", "")), "reason": reason})
        else:
            enrichment.update({"model": model_id, "status": "completed"})
        enrichment["usage"] = {
            "prompt_tokens": None,
            "completion_tokens": generated,
            "total_tokens": generated,
        }
        metrics = {
            "request_success": enrichment.get("status") == "completed",
            "json_valid": enrichment.get("status") == "completed",
            "retry_count": used_retries,
            "latency_seconds": elapsed,
            "generated_tokens": generated,
        }
        enrichment["evaluation"] = _evaluation(function, enrichment, metrics, {})
        function["enrichment"] = enrichment
        save_json(result, output_json)
    elapsed = time.perf_counter() - start
    all_selected = selected
    result["enrichment_metadata"] = {
        "provider": "Hugging Face Transformers",
        "model": model_id,
        "total_functions": len(all_selected),
        "enriched_functions": sum(
            f.get("enrichment", {}).get("status") == "completed" for f in all_selected
        ),
        "failed_functions": len(failures),
        "resumed_functions": len(all_selected) - len(latencies),
    }
    result["experiment"] = {
        "model": model_id, "temperature": 0.0, "batch_size": 1,
        "prompt_version": "v2", "schema_version": "v1", "base_json": str(input_json),
    }
    result["enrichment_metrics"] = {
        **_dataset_metrics(all_selected, elapsed),
        "total_runtime_seconds": elapsed,
        "total_inference_seconds": sum(latencies),
        "average_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "generated_tokens": sum(token_counts),
        "failed_nodes": failures,
    }
    save_json(result, output_json)
    return result["enrichment_metrics"]


def prepare_workspace(base_json: str | Path, enriched_json: str | Path,
                      workspace: str | Path, corpus_name: str) -> Path:
    """Create the standard filenames expected by Optima's discovery helpers."""
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    if base_json is not None and Path(base_json).exists():
        shutil.copy2(base_json, root / "base.json")
    shutil.copy2(enriched_json, root / f"enhanced_{corpus_name}.json")
    return root


def build_index(enriched_json: str | Path, corpus_name: str, embedding_model: str,
                report_dir: str | Path, representation_mode: str = "hybrid",
                force: bool = False) -> dict[str, Any]:
    """Call the existing Optima embedding/index implementation."""
    return embed_corpus(
        Path(enriched_json), corpus_name, embedding_model, Path(report_dir),
        representation_mode, force=force,
    )


def build_raw_index(base_json: str | Path, embedding_model: str,
                    report_dir: str | Path, representation_mode: str = "hybrid",
                    force: bool = False) -> dict[str, Any]:
    return embed_corpus(
        Path(base_json), None, embedding_model, Path(report_dir),
        representation_mode, force=force,
    )


def run_evaluation(report_dir: str | Path, embedding_models: Iterable[str],
                   benchmark_path: str | Path | None = None,
                   benchmark_source: str | Path | None = None, num_queries: int = 20,
                   k: int = 10, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the existing matrix evaluation and write its normal CSV/plot artifacts."""
    report = Path(report_dir)
    if benchmark_path and Path(benchmark_path).exists():
        queries = load_benchmark_queries(Path(benchmark_path))
        benchmark = Path(benchmark_path)
    else:
        if benchmark_source is None:
            raise ValueError("Provide benchmark_path or benchmark_source")
        queries = create_benchmark_from_json_files(Path(benchmark_source), num_queries, seed=42)
        benchmark = report / "benchmark" / "queries.json"
        save_benchmark_queries(queries, benchmark)
    aliases = [embedding_alias(model) for model in embedding_models]
    matrix = evaluate_embedding_matrix(report, aliases, queries, k)
    destination = Path(output_dir) if output_dir else report / "results"
    destination.mkdir(parents=True, exist_ok=True)
    save_matrix_csv(matrix, destination / "aggregate_metrics.csv")
    save_matrix_csv(matrix, destination / "retrieval_model_comparison.csv")
    save_per_query_results(matrix, report / "per_query_results.csv")
    save_group_metrics_csv(matrix, destination / "difficulty_metrics.csv", "difficulty")
    save_group_metrics_csv(matrix, destination / "category_metrics.csv", "category")
    save_gain_csv(matrix, destination / "enrichment_gains.csv", "enrichment")
    if len(aliases) > 1:
        save_gain_csv(matrix, destination / "embedding_gains.csv", "embedding")
    save_best_tables(matrix, destination / "best_tables.csv")
    save_best_combinations(matrix, destination)
    generate_visualizations(matrix, destination / "plots")
    save_json({"embedding_models": aliases, "benchmark": str(benchmark), "k": k,
               "results": matrix}, destination / "results.json")
    return matrix


def retrieve(report_dir: str | Path, corpus_name: str, embedding_model: str,
             query: str, k: int = 5) -> list[dict[str, Any]]:
    """Run the existing FAISS retrieval path and return serializable results."""
    indices = load_corpus_indices(Path(report_dir), embedding_model)
    if corpus_name not in indices:
        raise FileNotFoundError(f"No index for corpus={corpus_name}, embedding={embedding_model}")
    store = indices[corpus_name]
    docs = store.similarity_search(query, k=k) if hasattr(store, "similarity_search") else (
        store.as_retriever(search_kwargs={"k": k}).invoke(query)
    )
    return [{"rank": rank, "function_id": doc.metadata.get("function_id", ""),
             "function_name": doc.metadata.get("function_name", ""),
             "file_name": doc.metadata.get("file_name", ""),
             "content": doc.page_content}
            for rank, doc in enumerate(docs, 1)]


def zip_outputs(output_root: str | Path, zip_path: str | Path) -> Path:
    """Package all Colab artifacts without deleting prior experiments."""
    root, destination = Path(output_root), Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in root.rglob("*"):
            if item.is_file() and item != destination:
                archive.write(item, item.relative_to(root))
    return destination
