"""Embedding, retrieval, and evaluation guards on top of the existing
``optima.rag`` pipeline.

Nothing here reimplements FAISS indexing, retrieval, or metrics: it wraps
``optima.rag.embedding_simple.embed_corpus``, ``colab_pipeline.retrieve``,
and ``colab_pipeline.run_evaluation`` with the fairness/silent-failure
guards described in the architecture plan (no mock-embedding fallback, no
stale index reuse, no benchmark contamination, no error swallowed as
zeroed metrics).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from .errors import ComparisonMismatchError, EmbeddingError, EvaluationError, NoCorporaError


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def discover_passed_corpora(ctx: Any) -> list[str]:
    """Return ['raw', <passed model slugs...>]. Raises if nothing has passed."""
    manifest = ctx.load_manifest()
    passed = sorted(
        slug for slug, info in manifest.get("models", {}).items()
        if info.get("status") == "enrichment_passed"
    )
    if not passed:
        raise NoCorporaError(
            "No enrichment model has reached status='enrichment_passed'; there "
            "is nothing to embed or evaluate. Run the model cells first."
        )
    return ["raw", *passed]


def make_embedding_view(ctx: Any, slug: str) -> Path:
    """Write an embedding-ready copy of enhanced_<slug>.json with failed
    enrichments' ``enrichment`` key removed, so a failed function is embedded
    as raw representation rather than as the fallback placeholder text.
    """
    from colab.colab_pipeline import flatten_functions, load_json, save_json

    enhanced_path = ctx.enriched_dir / f"enhanced_{slug}.json"
    if not enhanced_path.exists():
        raise EmbeddingError(
            f"No materialized enrichment file found for {slug!r} at "
            f"{enhanced_path}. Run that model's full enrichment first."
        )
    data = load_json(enhanced_path)
    view = copy.deepcopy(data)
    functions = flatten_functions(view)
    stripped_ids = []
    for fn in functions:
        enrichment = fn.get("enrichment")
        if isinstance(enrichment, dict) and enrichment.get("status") != "completed":
            del fn["enrichment"]
            stripped_ids.append(fn.get("id"))

    view_path = ctx.rag_inputs_dir / f"enhanced_{slug}.json"
    save_json(view, view_path)
    manifest = {
        "slug": slug, "stripped_failed_ids": stripped_ids,
        "stripped_count": len(stripped_ids), "source": str(enhanced_path),
    }
    (ctx.rag_inputs_dir / f"{slug}.view.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if stripped_ids:
        print(f"EMBEDDING VIEW [{slug}]: stripped {len(stripped_ids)} failed "
              f"enrichment(s) -> raw representation for those functions")
    return view_path


def _corpus_input_path(ctx: Any, snap: Any, slug: str) -> Path:
    return snap.path if slug == "raw" else make_embedding_view(ctx, slug)


def _fingerprint_changed(report_dir: Path, corpus_name: str, alias: str, input_path: Path) -> bool:
    from optima.rag.embedding_simple import index_directory

    fp_path = index_directory(report_dir, corpus_name, alias) / "source_fingerprint.json"
    current = _sha256_file(input_path)
    if not fp_path.exists():
        return True
    try:
        stored = json.loads(fp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return stored.get("sha256") != current


def _write_fingerprint(report_dir: Path, corpus_name: str, alias: str, input_path: Path) -> None:
    from optima.rag.embedding_simple import index_directory

    index_dir = index_directory(report_dir, corpus_name, alias)
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "source_fingerprint.json").write_text(
        json.dumps({"sha256": _sha256_file(input_path), "source": str(input_path)}, indent=2) + "\n",
        encoding="utf-8",
    )


def _expected_function_count(ctx: Any) -> Optional[int]:
    report_path = ctx.base_dir / "validation_report.json"
    if report_path.exists():
        try:
            return json.loads(report_path.read_text(encoding="utf-8")).get("functions")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _assert_index_ok(result: dict[str, Any], mode: str, expected_documents: Optional[int]) -> None:
    if not result.get("success"):
        raise EmbeddingError(f"embed_corpus failed: {result}")
    metadata = result.get("metadata", {})
    if metadata.get("using_mock_embeddings"):
        raise EmbeddingError(
            f"Index for corpus={result.get('corpus_name')} silently used mock "
            f"embeddings instead of a real model; refusing to use it."
        )
    index_path = Path(result.get("index_metrics", {}).get("index_path", ""))
    if not (index_path / "index.faiss").exists() or not (index_path / "index.pkl").exists():
        raise EmbeddingError(
            f"Index files are missing at {index_path} for corpus="
            f"{result.get('corpus_name')}; embed_corpus likely fell back to an "
            f"in-memory index."
        )
    if metadata.get("representation_mode") != mode:
        raise EmbeddingError(
            f"Index representation_mode mismatch for corpus="
            f"{result.get('corpus_name')}: expected {mode!r}, got "
            f"{metadata.get('representation_mode')!r}."
        )
    if expected_documents is not None and metadata.get("num_documents") not in (None, expected_documents):
        raise EmbeddingError(
            f"Index for corpus={result.get('corpus_name')} has "
            f"{metadata.get('num_documents')} documents, expected "
            f"{expected_documents} (one per function in base.json)."
        )


def build_indexes(ctx: Any, corpora: list[str], embedding_models: list[str],
                   modes: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """Build (or reuse, if the source fingerprint is unchanged) a FAISS index
    for every (mode, embedding alias, corpus) combination.
    """
    from optima.rag.embedding_simple import OptimaEmbedder, embed_corpus, embedding_alias

    expected_documents = _expected_function_count(ctx)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in modes:
        report_dir = ctx.rag_dir / mode
        for alias_or_name in embedding_models:
            alias = embedding_alias(alias_or_name)
            embedder = OptimaEmbedder(alias)
            if embedder.using_mock_embeddings:
                raise EmbeddingError(
                    f"Embedding alias {alias!r} resolved to mock embeddings "
                    f"before any index was built; check network access and "
                    f"the model name/alias."
                )
            for slug in corpora:
                enrichment_model = None if slug == "raw" else slug
                input_path = _corpus_input_path(ctx, ctx.base, slug)
                changed = _fingerprint_changed(report_dir, slug, alias, input_path)
                result = embed_corpus(input_path, enrichment_model, alias, report_dir, mode, force=changed)
                _assert_index_ok(result, mode, expected_documents)
                if changed:
                    _write_fingerprint(report_dir, slug, alias, input_path)
                results.setdefault(mode, {}).setdefault(alias, {})[slug] = result
                print(f"INDEX [{mode}/{alias}/{slug}]: "
                      f"{'built' if not result.get('skipped') else 'reused'}, "
                      f"documents={result.get('metadata', {}).get('num_documents')}")
    return results


def retrieval_smoke(ctx: Any, corpora: list[str], alias: str, mode: str,
                     bench: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Sanity-check the retrieval pipeline itself (not quality): every corpus
    returns non-empty, well-formed results for one benchmark query.
    """
    from colab.colab_pipeline import retrieve
    from optima.rag.embedding_simple import embedding_alias

    report_dir = ctx.rag_dir / mode
    resolved_alias = embedding_alias(alias)
    query = bench["queries"][0]
    query_text = query["query"]
    relevant = set(query.get("relevant_functions", []))

    results: dict[str, list[dict[str, Any]]] = {}
    for slug in corpora:
        try:
            retrieved = retrieve(report_dir, slug, resolved_alias, query_text, k=10)
        except Exception as exc:  # noqa: BLE001 - reported as a pipeline failure
            raise EmbeddingError(f"Retrieval smoke test failed for corpus={slug}: {exc}") from exc
        if not retrieved or any(not item.get("function_id") for item in retrieved):
            raise EmbeddingError(
                f"Retrieval smoke test returned empty/invalid results for "
                f"corpus={slug} (mode={mode}, alias={resolved_alias})."
            )
        for item in retrieved:
            item["is_relevant"] = item["function_id"] in relevant
        results[slug] = retrieved
        print(f"[{mode}/{resolved_alias}] {slug}:")
        for item in retrieved[:5]:
            marker = "*" if item["is_relevant"] else " "
            print(f"  {marker} {item['rank']}. {item['function_name']} ({item['function_id']})")

    out_path = report_dir / "results" / "retrieval_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"query": query_text, "results": results}, indent=2, default=str) + "\n",
                        encoding="utf-8")
    return results


def _validate_matrix(matrix: dict[str, Any], aliases: list[str], corpora: list[str], num_queries: int) -> None:
    for alias in aliases:
        corpus_results = matrix.get(alias, {})
        for slug in corpora:
            metrics = corpus_results.get(slug)
            if metrics is None:
                raise EvaluationError(f"Missing evaluation result for alias={alias}, corpus={slug}.")
            if "error" in metrics:
                raise EvaluationError(f"Evaluation error for alias={alias}, corpus={slug}: {metrics['error']}")
            if metrics.get("num_queries") != num_queries:
                raise EvaluationError(
                    f"alias={alias}, corpus={slug}: num_queries={metrics.get('num_queries')} "
                    f"!= expected {num_queries}."
                )
            if metrics.get("queries_with_results") != num_queries:
                raise EvaluationError(
                    f"alias={alias}, corpus={slug}: only "
                    f"{metrics.get('queries_with_results')}/{num_queries} queries "
                    f"returned any results."
                )


def evaluate_all(ctx: Any, embedding_models: list[str], modes: list[str],
                  bench: dict[str, Any], k: int, corpora: list[str]) -> dict[str, Any]:
    from colab.colab_pipeline import run_evaluation
    from optima.rag.embedding_simple import OptimaEmbedder, embedding_alias

    matrices: dict[str, Any] = {}
    aliases = [embedding_alias(name) for name in embedding_models]
    for alias in aliases:
        embedder = OptimaEmbedder(alias)
        if embedder.using_mock_embeddings:
            raise EmbeddingError(
                f"Embedding alias {alias!r} is using mock embeddings; refusing "
                f"to run evaluation with it."
            )
    for mode in modes:
        report_dir = ctx.rag_dir / mode
        matrix = run_evaluation(
            report_dir, aliases, benchmark_path=bench["path"],
            num_queries=len(bench["queries"]), k=k, output_dir=report_dir / "results",
        )
        _validate_matrix(matrix, aliases, corpora, len(bench["queries"]))
        matrices[mode] = matrix
        print(f"EVALUATION [{mode}]: validated {len(aliases)} embedding alias(es) x "
              f"{len(corpora)} corpus/corpora")
    return matrices


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(value: Any, baseline: Any) -> Optional[float]:
    v, b = _to_float(value), _to_float(baseline)
    return None if v is None or b is None else round(v - b, 6)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_experiment_settings(ctx: Any, corpora: list[str]) -> dict[str, dict[str, Any]]:
    settings: dict[str, dict[str, Any]] = {}
    for slug in corpora:
        if slug == "raw":
            continue
        enhanced_path = ctx.enriched_dir / f"enhanced_{slug}.json"
        if not enhanced_path.exists():
            continue
        data = json.loads(enhanced_path.read_text(encoding="utf-8"))
        experiment = data.get("experiment", {})
        metadata = data.get("enrichment_metadata", {})
        metrics = data.get("enrichment_metrics", {})
        settings[slug] = {
            "prompt_variant": experiment.get("prompt_variant"),
            "do_sample": experiment.get("do_sample"),
            "max_new_tokens": experiment.get("max_new_tokens"),
            "retries": experiment.get("retries"),
            "base_sha256": experiment.get("base_sha256"),
            "model_id": experiment.get("model"),
            "quantization": metadata.get("quantization"),
            "resolved_commit_hash": metadata.get("resolved_commit_hash"),
            "config_hash": experiment.get("config_hash"),
            "success_rate": metrics.get("success_rate"),
            "failed_functions": metrics.get("functions_failed"),
        }
    return settings


def build_comparison(ctx: Any, matrices: dict[str, Any], corpora: list[str],
                      bench: dict[str, Any]) -> dict[str, Any]:
    """Build the cross-model comparison table. Refuses to compare corpora
    whose generation/prompt settings differ (model and quantization may
    differ; everything that would make the comparison unfair may not).
    """
    from optima.rag.embedding_simple import get_embedding_model_name

    settings = _load_experiment_settings(ctx, corpora)
    if settings:
        keys_to_match = ("prompt_variant", "do_sample", "max_new_tokens", "retries", "base_sha256")
        reference_slug, reference = next(iter(settings.items()))
        for slug, this_settings in settings.items():
            for key in keys_to_match:
                if this_settings.get(key) != reference.get(key):
                    raise ComparisonMismatchError(
                        f"Corpus {slug!r} has {key}={this_settings.get(key)!r} but "
                        f"{reference_slug!r} has {key}={reference.get(key)!r}; these "
                        f"runs cannot be compared fairly. (Model and quantization "
                        f"may differ; generation/prompt settings and the base.json "
                        f"snapshot must match.)"
                    )

    raw_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for mode, matrix in matrices.items():
        for alias, corpus_results in matrix.items():
            raw_lookup[(mode, alias)] = corpus_results.get("raw", {})

    rows: list[dict[str, Any]] = []
    for mode, matrix in matrices.items():
        for alias, corpus_results in matrix.items():
            for slug in corpora:
                metrics = corpus_results.get(slug)
                if metrics is None:
                    continue
                raw_metrics = raw_lookup.get((mode, alias), {})
                model_settings = settings.get(slug, {})
                rows.append({
                    "base_sha12": ctx.base.sha256[:12], "base_url": ctx.base.url,
                    "optima_commit": ctx.optima_commit,
                    "benchmark_sha256": bench["manifest"].get("sha256"),
                    "num_queries": metrics.get("num_queries"),
                    "representation_mode": mode, "embedding_alias": alias,
                    "embedding_model": get_embedding_model_name(alias),
                    "corpus": slug, "enrichment_model_id": model_settings.get("model_id"),
                    "model_slug": slug, "quantization": model_settings.get("quantization"),
                    "resolved_commit_hash": model_settings.get("resolved_commit_hash"),
                    "prompt_variant": model_settings.get("prompt_variant"),
                    "config_hash": model_settings.get("config_hash"),
                    "enrichment_success_rate": model_settings.get("success_rate"),
                    "failed_functions_stripped": model_settings.get("failed_functions"),
                    "recall_at_1": _to_float(metrics.get("recall_at_1")),
                    "recall_at_5": _to_float(metrics.get("recall_at_5")),
                    "recall_at_10": _to_float(metrics.get("recall_at_10")),
                    "mrr": _to_float(metrics.get("mrr")),
                    "precision_at_5": _to_float(metrics.get("precision_at_5")),
                    "hit_rate_at_5": _to_float(metrics.get("hit_rate_at_5")),
                    "mean_latency": _to_float(metrics.get("mean_latency")),
                    "median_latency": _to_float(metrics.get("median_latency")),
                    "document_count": metrics.get("document_count"),
                    "delta_recall_at_5_vs_raw": _delta(metrics.get("recall_at_5"), raw_metrics.get("recall_at_5")),
                    "delta_mrr_vs_raw": _delta(metrics.get("mrr"), raw_metrics.get("mrr")),
                })

    rows.sort(key=lambda row: (row["representation_mode"], row["embedding_alias"], -(row["mrr"] or 0)))

    out_json = ctx.summaries_dir / "comparison.json"
    out_csv = ctx.summaries_dir / "comparison.csv"
    out_json.write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")
    _write_csv(out_csv, rows)
    print(f"COMPARISON: wrote {len(rows)} row(s) to {out_csv}")
    return {"rows": rows, "csv_path": out_csv, "json_path": out_json}
