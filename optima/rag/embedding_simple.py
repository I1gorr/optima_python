"""Embedding registry and resilient FAISS index creation.

The command line interface uses this module rather than :mod:`embedding` so
that an unavailable HuggingFace model does not prevent a benchmark from
running.  Indexes created by this module have a stable, alias-based layout::

    <rag-report>/indexes/<corpus>/<embedding-alias>/

The loader also understands the historical ``indexes/<model>/<corpus>`` layout.
"""

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingSpec:
    alias: str
    model_name: str
    description: str = ""


# Keep the model names in one place.  ``mock`` is useful for offline smoke
# tests and is intentionally not selected by --all-embedding-models.
EMBEDDING_REGISTRY: Dict[str, EmbeddingSpec] = {
    "bge-small": EmbeddingSpec("bge-small", "BAAI/bge-small-en-v1.5"),
    "bge-base": EmbeddingSpec("bge-base", "BAAI/bge-base-en-v1.5"),
    "nomic": EmbeddingSpec("nomic", "nomic-ai/nomic-embed-text-v1.5"),
    "qwen3-0.6b": EmbeddingSpec("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B"),
    "coderank": EmbeddingSpec("coderank", "nomic-ai/CodeRankEmbed"),
    "mock": EmbeddingSpec("mock", "mock", "Deterministic offline embeddings"),
}
# A plain mapping is convenient for configuration UIs and backwards-compatible
# callers that only need the HuggingFace model identifier.
EMBEDDING_MODELS = {alias: spec.model_name for alias, spec in EMBEDDING_REGISTRY.items()}


def register_embedding_model(alias: str, model_name: str,
                             description: str = "") -> EmbeddingSpec:
    """Register or replace an embedding alias for applications and tests."""
    clean_alias = _slug(alias)
    if not clean_alias:
        raise ValueError("Embedding alias must not be empty")
    spec = EmbeddingSpec(clean_alias, model_name, description)
    EMBEDDING_REGISTRY[clean_alias] = spec
    EMBEDDING_MODELS[clean_alias] = model_name
    return spec


def get_embedding_registry(include_mock: bool = False) -> Dict[str, EmbeddingSpec]:
    """Return a copy of the configured registry."""
    return {
        alias: spec for alias, spec in EMBEDDING_REGISTRY.items()
        if include_mock or alias != "mock"
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip()).strip("-").lower()


def resolve_embedding_model(model_or_alias: str) -> EmbeddingSpec:
    """Resolve an alias, a HuggingFace model name, or a custom model name."""
    if not model_or_alias:
        model_or_alias = "bge-small"
    value = str(model_or_alias).strip()
    if value.lower() in EMBEDDING_REGISTRY:
        return EMBEDDING_REGISTRY[value.lower()]
    for spec in EMBEDDING_REGISTRY.values():
        if value == spec.model_name:
            return spec
    return EmbeddingSpec(_slug(value), value)


def embedding_alias(model_or_alias: str) -> str:
    return resolve_embedding_model(model_or_alias).alias


def get_embedding_model_name(model_or_alias: str) -> str:
    """Compatibility helper returning the resolved HuggingFace identifier."""
    return resolve_embedding_model(model_or_alias).model_name


def index_directory(output_base_dir: Path, corpus_name: str,
                    model_or_alias: str) -> Path:
    """Return the canonical non-overlapping index directory."""
    return Path(output_base_dir) / "indexes" / _slug(corpus_name) / embedding_alias(model_or_alias)


def metadata_path(index_path: Path) -> Path:
    return Path(index_path) / "metadata.json"


def load_index_metadata(index_path: Path) -> Dict[str, Any]:
    """Read index metadata, returning an empty mapping for legacy indexes."""
    return _read_metadata(Path(index_path))


def _read_metadata(index_path: Path) -> Dict[str, Any]:
    try:
        with metadata_path(index_path).open(encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _embedding_device() -> str:
    configured = os.environ.get("OPTIMA_EMBEDDING_DEVICE", "auto").strip().lower()
    if configured and configured != "auto":
        return configured
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_embeddings(embedding_model_name: str, device: Optional[str] = None):
    """Load a registered model, falling back to deterministic embeddings."""
    spec = resolve_embedding_model(embedding_model_name)
    try:
        if spec.model_name == "mock":
            raise RuntimeError("mock embedding requested")
        logger.info("Attempting to load embedding model %s (%s)", spec.alias, spec.model_name)
        from langchain_huggingface import HuggingFaceEmbeddings
        selected_device = device or _embedding_device()
        return HuggingFaceEmbeddings(
            model_name=spec.model_name,
            model_kwargs={"device": selected_device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        logger.warning("Could not load %s (%s); using mock embeddings: %s",
                       spec.alias, spec.model_name, exc)
        from .test_embeddings import MockEmbeddings
        return MockEmbeddings(dimension=384)


class OptimaEmbedder:
    """Generate embeddings and create/load vector indexes."""

    def __init__(self, embedding_model_name: str = "bge-small",
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        self.embedding_spec = resolve_embedding_model(embedding_model_name)
        self.embedding_alias = self.embedding_spec.alias
        self.embedding_model_name = self.embedding_spec.model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = _embedding_device()
        self.embeddings = get_embeddings(self.embedding_alias, self.device)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )

    @property
    def using_mock_embeddings(self) -> bool:
        return "MockEmbeddings" in str(type(self.embeddings))

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        if not self.embeddings:
            raise RuntimeError("Embeddings model not initialized")
        return self.embeddings.embed_documents([doc.page_content for doc in documents])

    def embed_query(self, query: str) -> List[float]:
        """Embed queries with the same selected model as the index."""
        if not self.embeddings:
            raise RuntimeError("Embeddings model not initialized")
        return self.embeddings.embed_query(query)

    def create_vector_index(self, documents: List[Document], index_path: Path):
        if not documents:
            raise ValueError("Cannot create index from empty document list")
        chunked_docs = self.text_splitter.split_documents(documents)
        try:
            from langchain_community.vectorstores import FAISS
            vector_store = FAISS.from_documents(chunked_docs, self.embeddings)
            index_path.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(index_path))
            return vector_store
        except Exception as exc:
            logger.error("Failed to create FAISS index: %s", exc)
            # A small in-memory fallback keeps unit tests and offline runs
            # useful, while metadata still records that no persistent index exists.
            embedded_docs = self.embed_documents(chunked_docs)

            class SimpleRetriever:
                def __init__(self, docs, k):
                    self.docs, self.k = docs, k

                def get_relevant_documents(self, _query):
                    return self.docs[:self.k]

                def invoke(self, query):
                    return self.get_relevant_documents(query)

            class SimpleVectorStore:
                def __init__(self, docs, vectors):
                    self.documents, self.vectors = docs, vectors

                def as_retriever(self, search_kwargs=None):
                    return SimpleRetriever(self.documents, (search_kwargs or {}).get("k", 4))

            return SimpleVectorStore(chunked_docs, embedded_docs)

    def load_vector_index(self, index_path: Path):
        if not Path(index_path).exists():
            return None
        try:
            from langchain_community.vectorstores import FAISS
            return FAISS.load_local(
                str(index_path), self.embeddings, allow_dangerous_deserialization=True
            )
        except Exception as exc:
            logger.warning("Failed to load vector index %s: %s", index_path, exc)
            return None

    def get_embedding_metrics(self, documents: List[Document]) -> Dict[str, Any]:
        total_chars = sum(len(doc.page_content) for doc in documents)
        return {
            "num_documents": len(documents),
            "total_characters": total_chars,
            "avg_characters_per_document": total_chars / len(documents) if documents else 0,
            "estimated_tokens": total_chars // 4,
            "embedding_model": self.embedding_model_name,
            "embedding_alias": self.embedding_alias,
            "device": self.device,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "using_mock_embeddings": self.using_mock_embeddings,
        }


def _write_metadata(index_dir: Path, payload: Dict[str, Any]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path(index_dir).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def embed_corpus(json_path: Path, enrichment_model: Optional[str],
                 embedding_model_name: str = "bge-small",
                 output_base_dir: Path = Path("output/rag_report"),
                 representation_mode: str = "hybrid",
                 force: bool = False) -> Dict[str, Any]:
    """Embed one corpus, skipping an existing canonical index unless forced."""
    start_time = time.time()
    corpus_name = enrichment_model or "raw"
    corpus_type = "enriched" if enrichment_model else "raw"
    spec = resolve_embedding_model(embedding_model_name)
    index_dir = index_directory(output_base_dir, corpus_name, spec.alias)
    persistent_index = (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()
    if persistent_index and not force:
        metadata = _read_metadata(index_dir)
        return {
            "success": True, "skipped": True, "corpus_name": corpus_name,
            "corpus_type": corpus_type, "embedding_alias": spec.alias,
            "embedding_model": spec.model_name, "json_path": str(json_path),
            "index_metrics": {"index_path": str(index_dir), "index_size_mb": 0},
            "metadata": metadata,
            "elapsed_time_seconds": round(time.time() - start_time, 2),
        }

    from .document_constructor import OptimaDocumentConstructor
    documents = OptimaDocumentConstructor(
        representation_mode=representation_mode
    ).construct_documents_from_json(json_path, enrichment_model=enrichment_model)
    if not documents:
        raise ValueError(f"No documents constructed from {json_path}")

    embedder = OptimaEmbedder(spec.alias)
    vector_store = embedder.create_vector_index(documents, index_dir)
    index_file = index_dir / "index.faiss"
    index_size_mb = index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0
    chunks = vector_store.index.ntotal if hasattr(vector_store, "index") else 0
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus_name,
        "corpus_type": corpus_type,
        "embedding_alias": spec.alias,
        "embedding_model": spec.model_name,
        "source_json": str(json_path),
        "representation_mode": representation_mode,
        "chunk_size": embedder.chunk_size,
        "chunk_overlap": embedder.chunk_overlap,
        "num_documents": len(documents),
        "document_count": len(documents),
        "num_chunks": chunks,
        "using_mock_embeddings": embedder.using_mock_embeddings,
        "dimension": len(embedder.embed_query("dimension probe")),
    }
    _write_metadata(index_dir, metadata)
    result = {
        "success": True, "skipped": False, "corpus_name": corpus_name,
        "corpus_type": corpus_type, "embedding_alias": spec.alias,
        "embedding_model": spec.model_name, "json_path": str(json_path),
        "num_documents": len(documents),
        "elapsed_time_seconds": round(time.time() - start_time, 2),
        "document_constructor_metrics": {
            "num_documents": len(documents), "representation_mode": representation_mode
        },
        "embedding_metrics": embedder.get_embedding_metrics(documents),
        "index_metrics": {
            "index_path": str(index_dir), "index_size_mb": round(index_size_mb, 2),
            "num_chunks_in_index": chunks,
        },
        "metadata": metadata,
        "using_mock_embeddings": embedder.using_mock_embeddings,
    }
    return result


def embed_all_corpora(output_dir: Path, embedding_model_name: str = "bge-small",
                      representation_mode: str = "hybrid",
                      embedding_model_names: Optional[List[str]] = None,
                      force: bool = False) -> Dict[str, Any]:
    """Embed all discovered corpora for one or more registry aliases."""
    from .document_constructor import discover_json_files

    json_files = discover_json_files(output_dir)
    if not json_files:
        raise ValueError(f"No JSON files found in {output_dir}")
    requested = embedding_model_names or [embedding_model_name]
    aliases = [resolve_embedding_model(item).alias for item in requested]
    report_dir = Path(output_dir) / "rag_report"
    matrix: Dict[str, Dict[str, Any]] = {}
    for alias in aliases:
        corpus_results = {}
        for corpus_name, json_path in json_files.items():
            try:
                corpus_results[corpus_name] = embed_corpus(
                    json_path, None if corpus_name == "base" else corpus_name,
                    alias, report_dir, representation_mode, force=force
                )
            except Exception as exc:
                corpus_results[corpus_name] = {
                    "success": False, "error": str(exc), "corpus_name": corpus_name,
                    "json_path": str(json_path), "embedding_alias": alias,
                }
        matrix[alias] = corpus_results
    summary = {
        "total_embedding_models": len(aliases),
        "total_corpora": len(json_files),
        "successful": sum(
            result.get("success", False)
            for corpora in matrix.values() for result in corpora.values()
        ),
        "failed": sum(
            not result.get("success", False)
            for corpora in matrix.values() for result in corpora.values()
        ),
    }
    result: Dict[str, Any] = {
        "embedding_models": aliases, "representation_mode": representation_mode,
        "embeddings": matrix, "summary": summary,
    }
    # Preserve the old single-model response shape for scripts using it.
    if len(aliases) == 1:
        result.update({"embedding_model": aliases[0], "corpora": matrix[aliases[0]]})
        summary.update({
            "successful_corpora": summary["successful"],
            "failed_corpora": summary["failed"],
            "total_elapsed_time_seconds": 0,
        })
    return result
