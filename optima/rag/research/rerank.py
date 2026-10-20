"""Optional cross-encoder reranking.

A cross-encoder scores each ``(query, function text)`` pair jointly.  The model
is loaded lazily, exactly once per ``(model, device)`` in the process, and shared
by every pipeline that uses it.

There is deliberately no heuristic fallback: if the model cannot be loaded,
:class:`RerankerUnavailable` is raised so the run is recorded as *not reranked*
rather than reported as a cross-encoder result.  An inference error on a single
query raises :class:`RerankerError`; the pipeline keeps that query's prior
ranking and flags it.

Licences were read from the Hugging Face model cards (2026-10-11); confirm them
again before redistributing a model.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class RerankerUnavailable(RuntimeError):
    """The cross-encoder model could not be loaded."""


class RerankerError(RuntimeError):
    """Inference failed for a query."""


@dataclass(frozen=True)
class RerankerSpec:
    alias: str
    model_name: str
    license: str
    parameters_m: float
    hardware: str


RERANKER_REGISTRY: Dict[str, RerankerSpec] = {
    "ms-marco-minilm": RerankerSpec(
        "ms-marco-minilm", "cross-encoder/ms-marco-MiniLM-L6-v2", "Apache-2.0", 22.7,
        "~90 MB download; runs comfortably on CPU, trivial on GPU. Trained on MS MARCO "
        "web passages, not on code."),
    "bge-reranker-base": RerankerSpec(
        "bge-reranker-base", "BAAI/bge-reranker-base", "MIT", 278.0,
        "~1.1 GB download; CPU is slow (seconds per query pool), <2 GB VRAM on GPU. "
        "Multilingual relevance model, not code-specific."),
    "bge-reranker-v2-m3": RerankerSpec(
        "bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3", "Apache-2.0", 567.8,
        "~2.3 GB download; GPU recommended (~3 GB VRAM). Multilingual, not code-specific."),
}


def resolve_reranker(model_or_alias: str) -> RerankerSpec:
    value = (model_or_alias or "").strip()
    spec = RERANKER_REGISTRY.get(value.lower())
    if spec:
        return spec
    for item in RERANKER_REGISTRY.values():
        if value == item.model_name:
            return item
    return RerankerSpec(value, value, "unknown (check the model card)", 0.0, "unknown")


_LOAD_LOCK = threading.Lock()
_LOADED: Dict[Tuple[str, str, int], Any] = {}


def _resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class CrossEncoderReranker:
    """Lazily-loaded cross-encoder.  ``predict_fn`` injects a scorer for tests."""

    def __init__(self, model_or_alias: str, device: str = "auto", batch_size: int = 16,
                 max_length: int = 512,
                 predict_fn: Optional[Callable[[List[Tuple[str, str]]], Sequence[float]]] = None):
        self.spec = resolve_reranker(model_or_alias)
        self.device = _resolve_device(device)
        self.batch_size, self.max_length = batch_size, max_length
        self._predict_fn = predict_fn
        self._model: Any = None
        self.load_seconds: Optional[float] = None
        self.error: Optional[str] = None

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        """Load once; raises :class:`RerankerUnavailable` with an actionable message."""
        if self._predict_fn is not None or self._model is not None:
            return
        key = (self.spec.model_name, self.device, self.max_length)
        with _LOAD_LOCK:
            if key in _LOADED:
                self._model = _LOADED[key]
                self.load_seconds = 0.0
                return
            start = time.perf_counter()
            try:
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(self.spec.model_name, device=self.device,
                                     max_length=self.max_length)
            except ImportError as exc:
                self.error = "sentence-transformers is not installed"
                raise RerankerUnavailable(
                    "cross-encoder reranking needs `pip install sentence-transformers torch`"
                ) from exc
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                raise RerankerUnavailable(
                    f"could not load cross-encoder {self.spec.alias!r} "
                    f"({self.spec.model_name}) on {self.device}: {self.error}. "
                    "Check network access / the model id, or disable rerank.enabled."
                ) from exc
            self.load_seconds = time.perf_counter() - start
            _LOADED[key] = self._model = model

    @property
    def info(self) -> Dict[str, Any]:
        return {"alias": self.spec.alias, "model_name": self.spec.model_name,
                "license": self.spec.license, "parameters_m": self.spec.parameters_m,
                "hardware": self.spec.hardware, "device": self.device,
                "batch_size": self.batch_size, "max_length": self.max_length,
                "load_seconds": self.load_seconds}

    # ------------------------------------------------------------------ scoring
    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []
        self.load()
        pairs = [(query, doc) for doc in documents]
        try:
            if self._predict_fn is not None:
                scores = list(self._predict_fn(pairs))
            else:
                scores = list(self._model.predict(
                    pairs, batch_size=self.batch_size, show_progress_bar=False))
        except Exception as exc:
            raise RerankerError(f"{type(exc).__name__}: {exc}") from exc
        if len(scores) != len(pairs):
            raise RerankerError(f"scorer returned {len(scores)} scores for {len(pairs)} pairs")
        return [float(s) for s in scores]

    def rerank(self, query: str, candidates: Sequence[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """Return ``(id, score)`` best first.  Equal scores keep their input order."""
        scores = self.score(query, [text for _, text in candidates])
        ranked = sorted(range(len(candidates)), key=lambda i: (-round(scores[i], 9), i))
        return [(candidates[i][0], scores[i]) for i in ranked]

    def gpu_peak_mb(self) -> Optional[float]:
        if self.device.startswith("cuda"):
            try:
                import torch
                return torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception:  # pragma: no cover - torch optional
                return None
        return None
