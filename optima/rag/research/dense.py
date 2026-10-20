"""Dense (embedding) retrieval at function granularity.

Model names are resolved through the existing registry in
``optima.rag.embedding_simple`` (aliases, HF ids, ``OPTIMA_EMBEDDING_DEVICE``);
nothing here hardcodes a model.  Unlike the legacy FAISS path, which silently
falls back to hash-based mock vectors when a model cannot load, a model that
fails to load raises :class:`EmbeddingUnavailable` -- a result must never be
reported under a real model's name when it was produced by a stand-in.

Each function is embedded once (no chunking), so one function is one vector and
the ranked list is already de-duplicated by function ID.  Exact cosine search
over a normalised matrix is deterministic and fast at this corpus size.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from optima.rag.embedding_simple import _embedding_device, resolve_embedding_model

from .lexical import normalize_query, tokenize

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEQ_LENGTH = 512


class EmbeddingUnavailable(RuntimeError):
    """The requested embedding model could not be loaded or run."""


class Encoder(Protocol):
    info: Dict[str, Any]

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEncoder:
    """Offline, deterministic bag-of-tokens encoder (feature hashing).

    Not a semantic model: it exists for CI, unit tests and offline smoke runs and
    is always labelled ``hashing`` in results.
    """

    def __init__(self, dimension: int = 512, prefix_doc: str = "", prefix_query: str = ""):
        self.dimension = dimension
        self.prefix_doc, self.prefix_query = prefix_doc, prefix_query
        self.info = {"alias": "hashing", "model_name": "hashing", "device": "cpu",
                     "dimension": dimension, "mock": True, "max_seq_length": None,
                     "note": "offline non-semantic feature-hashing encoder"}

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(prefix + (text or "")):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                out[row, int.from_bytes(digest[:4], "little") % self.dimension] += (
                    1.0 if digest[4] & 1 else -1.0)
        return _l2_normalise(out)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.prefix_doc)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([normalize_query(t) for t in texts], self.prefix_query)


class SentenceTransformerEncoder:
    """SentenceTransformers-backed encoder for any registry alias / HF id."""

    def __init__(self, model_or_alias: str, device: Optional[str] = None, batch_size: int = 32,
                 max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, query_prefix: str = "",
                 document_prefix: str = "", trust_remote_code: Optional[bool] = None):
        spec = resolve_embedding_model(model_or_alias)
        self.spec = spec
        self.batch_size = batch_size
        self.query_prefix, self.document_prefix = query_prefix, document_prefix
        self.device = device or _embedding_device()
        if trust_remote_code is None:
            trust_remote_code = os.environ.get("OPTIMA_TRUST_REMOTE_CODE", "") == "1"
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingUnavailable(
                "sentence-transformers is not installed; "
                "run `pip install sentence-transformers torch`") from exc
        try:
            self.model = SentenceTransformer(
                spec.model_name, device=self.device, trust_remote_code=bool(trust_remote_code))
        except Exception as exc:
            hint = ""
            if "trust_remote_code" in str(exc) or "remote code" in str(exc).lower():
                hint = (" This model ships custom code; set OPTIMA_TRUST_REMOTE_CODE=1 only if "
                        "you have reviewed it.")
            raise EmbeddingUnavailable(
                f"could not load embedding model {spec.alias!r} ({spec.model_name}): "
                f"{type(exc).__name__}: {exc}.{hint}") from exc
        native = getattr(self.model, "max_seq_length", None) or max_seq_length
        self.model.max_seq_length = int(min(native, max_seq_length))
        self.info = {
            "alias": spec.alias, "model_name": spec.model_name, "device": self.device,
            "dimension": int((getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension)() or 0),
            "mock": False, "max_seq_length": int(self.model.max_seq_length),
            "query_prefix": query_prefix, "document_prefix": document_prefix,
            "trust_remote_code": bool(trust_remote_code),
        }

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        try:
            vectors = self.model.encode(
                [prefix + t for t in texts], batch_size=self.batch_size,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:
            raise EmbeddingUnavailable(
                f"embedding inference failed for {self.spec.alias}: "
                f"{type(exc).__name__}: {exc}") from exc
        return _l2_normalise(np.asarray(vectors, dtype=np.float32))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(list(texts), self.document_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([normalize_query(t) for t in texts], self.query_prefix)


def get_encoder(model_or_alias: str, query_prefix: str = "", document_prefix: str = "",
                device: Optional[str] = None, batch_size: int = 32) -> Encoder:
    """Resolve an encoder.  ``hashing`` (and the registry's ``mock``) are offline."""
    value = (model_or_alias or "").strip().lower()
    if value in ("hashing", "mock"):
        return HashingEncoder(prefix_doc=document_prefix, prefix_query=query_prefix)
    return SentenceTransformerEncoder(
        model_or_alias, device=device, batch_size=batch_size,
        query_prefix=query_prefix, document_prefix=document_prefix)


# ------------------------------------------------------------------- caching
class EmbeddingCache:
    """On-disk document-embedding cache keyed by model + prefixes + text hashes."""

    def __init__(self, directory: Optional[Path]):
        self.directory = Path(directory) if directory else None

    @staticmethod
    def key(encoder_info: Dict[str, Any], texts: Sequence[str]) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(
            {k: encoder_info.get(k) for k in
             ("alias", "model_name", "max_seq_length", "document_prefix", "dimension")},
            sort_keys=True).encode())
        for text in texts:
            digest.update(hashlib.sha256(text.encode("utf-8")).digest())
        return digest.hexdigest()[:32]

    def _path(self, info: Dict[str, Any], key: str) -> Optional[Path]:
        if self.directory is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(info.get("alias") or "model"))
        return self.directory / safe / f"{key}.npz"

    def load(self, info: Dict[str, Any], key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        path = self._path(info, key)
        if path is None or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return data["vectors"], json.loads(str(data["meta"]))
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("ignoring unreadable embedding cache %s: %s", path, exc)
            return None

    def store(self, info: Dict[str, Any], key: str, vectors: np.ndarray,
              meta: Dict[str, Any]) -> None:
        path = self._path(info, key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, vectors=vectors, meta=json.dumps(meta))
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("could not write embedding cache %s: %s", path, exc)


@dataclass
class DenseIndex:
    """Exact cosine index over one representation."""

    ids: List[str]
    matrix: np.ndarray
    build_info: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def build(cls, ids: Sequence[str], texts: Sequence[str], encoder: Encoder,
              cache: Optional[EmbeddingCache] = None) -> "DenseIndex":
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        info = dict(getattr(encoder, "info", {}))
        info["document_prefix"] = info.get("document_prefix", "")
        start = time.perf_counter()
        key = EmbeddingCache.key(info, texts)
        cached = cache.load(info, key) if cache is not None and texts else None
        if cached is not None and cached[0].shape[0] == len(ids):
            matrix, meta = cached
            build = {"cache_hit": True,
                     "embed_seconds": float(meta.get("embed_seconds", 0.0)),
                     "load_seconds": time.perf_counter() - start}
        else:
            if texts:
                t0 = time.perf_counter()
                matrix = np.asarray(encoder.encode_documents(list(texts)), dtype=np.float32)
                embed_seconds = time.perf_counter() - t0
            else:
                matrix, embed_seconds = np.zeros((0, int(info.get("dimension") or 0)),
                                                 dtype=np.float32), 0.0
            build = {"cache_hit": False, "embed_seconds": embed_seconds, "load_seconds": 0.0}
            if cache is not None and texts:
                cache.store(info, key, matrix, {"embed_seconds": embed_seconds,
                                                "count": len(ids), "encoder": info})
        build.update({"documents": len(ids), "dimension": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
                      "index_bytes": int(matrix.nbytes), "cache_key": key})
        return cls(list(ids), matrix, build)

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if not self.ids or k < 1:
            return []
        scores = self.matrix @ np.asarray(query_vector, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order]
