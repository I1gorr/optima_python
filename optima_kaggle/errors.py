"""Typed exception hierarchy for the Kaggle heavy-LLM enrichment experiment.

Every exception here is meant to stop the notebook (or one model's stage of
it) with a message that states what failed, the key values involved, and one
concrete next action. Nothing in this module imports torch/transformers/etc:
it must be importable before any heavy dependency is installed.
"""

from __future__ import annotations

from typing import Any


class OptimaKaggleError(Exception):
    """Base class for every error raised by optima_kaggle."""


class EnvironmentError_(OptimaKaggleError):
    """The Kaggle/Python environment is not usable as configured."""


class QuantizationUnavailableError(OptimaKaggleError):
    """4-bit (bitsandbytes) quantization was required but is not usable."""


class BaseJsonError(OptimaKaggleError):
    """base.json could not be downloaded, parsed, or validated."""


class BenchmarkError(OptimaKaggleError):
    """The benchmark query set could not be created, loaded, or validated."""


class ModelNotFeasibleError(OptimaKaggleError):
    """The requested model is classified as infeasible on this GPU (Tier C)."""


class ModelDoesNotFitError(OptimaKaggleError):
    """The pre-load VRAM estimate says the model will not fit."""


class InsufficientDiskError(OptimaKaggleError):
    """Not enough free disk space to download the requested model."""


class ModelLoadError(OptimaKaggleError):
    """Tokenizer or model loading failed."""

    def __init__(self, message: str, category: str = "load_failed", stage: str | None = None):
        super().__init__(message)
        self.category = category
        self.stage = stage


class InsufficientHeadroomError(OptimaKaggleError):
    """The model loaded, but not enough VRAM headroom remains for generation."""


class GateFailedError(OptimaKaggleError):
    """A required smoke-test gate did not pass."""

    def __init__(self, gate: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"{gate} failed: {message}")
        self.gate = gate
        self.details = details or {}


class GenerationOOMError(OptimaKaggleError):
    """CUDA ran out of memory during model.generate()."""


class PromptTooLongError(OptimaKaggleError):
    """A function's prompt could not be reduced under max_input_tokens."""


class SystemicEnrichmentFailure(OptimaKaggleError):
    """Too many consecutive enrichment failures; something is systemically broken."""


class EnrichmentQualityError(OptimaKaggleError):
    """The full enrichment run finished but did not meet the success-rate bar."""


class ResumeMismatchError(OptimaKaggleError):
    """A checkpoint does not match the current base.json / config identity."""


class CheckpointCorruptError(OptimaKaggleError):
    """A checkpoint JSONL file has a malformed line that is not the last one."""


class GpuNotCleanError(OptimaKaggleError):
    """GPU memory was not released after unloading a model."""


class EmbeddingError(OptimaKaggleError):
    """Embedding/index construction failed or produced an unusable index."""


class EvaluationError(OptimaKaggleError):
    """Retrieval evaluation failed or produced incomplete/invalid results."""


class NoCorporaError(OptimaKaggleError):
    """No enrichment corpus has passed; there is nothing to embed/evaluate."""


class ComparisonMismatchError(OptimaKaggleError):
    """Corpora with incompatible experiment settings cannot be compared fairly."""


class ExperimentIncompleteError(OptimaKaggleError):
    """One or more configured models did not reach a passing state."""
