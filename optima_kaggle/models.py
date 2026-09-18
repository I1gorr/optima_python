"""Model registry, VRAM fit estimation, and memory-conscious Hugging Face
model loading/generation/unloading across one or more GPUs (e.g. a single
T4, or 2xT4 with model-parallel sharding for ~30B-class models).

Nothing here imports torch/transformers/accelerate at module load time:
those imports happen inside functions so that ``MODEL_REGISTRY`` and
``resolve_spec`` can be inspected (and unit tested) without a GPU or those
packages installed.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from . import environment
from .errors import (
    InsufficientDiskError,
    InsufficientHeadroomError,
    ModelDoesNotFitError,
    ModelLoadError,
    ModelNotFeasibleError,
)

_VALID_SINGLE_GPU_TIERS = ("A", "B", "C")
_VALID_QUANTS = ("fp16", "nf4", "nf4-prequantized")


def _device_index(value: Any) -> Optional[int]:
    """Normalize a hf_device_map value ("", 0, "cuda:1", torch.device(...), ...)
    to a plain GPU index, or None if it is not a GPU (e.g. "cpu"/"disk")."""
    if isinstance(value, int):
        return value
    text = str(value)
    if text.startswith("cuda:"):
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return None
    if text.isdigit():
        return int(text)
    return None


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_id: str
    quantization: str  # "fp16" | "nf4" | "nf4-prequantized"
    single_gpu_tier: str  # "A" (fits 1xT4) | "B" (fits, tight) | "C" (does not fit 1xT4)
    max_input_tokens: int
    revision: Optional[str] = None
    max_new_tokens: Optional[int] = None  # None = no artificial cap; bounded only by the
                                           # model's own context window (see max_new_tokens_for_context)
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    strip_think: bool = False
    allow_cpu_offload: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.single_gpu_tier not in _VALID_SINGLE_GPU_TIERS:
            raise ValueError(f"Invalid single_gpu_tier {self.single_gpu_tier!r} for {self.slug}")
        if self.quantization not in _VALID_QUANTS:
            raise ValueError(f"Invalid quantization {self.quantization!r} for {self.slug}")
        if self.slug in ("raw", "base"):
            raise ValueError(f"Model slug {self.slug!r} collides with a reserved corpus name")
        from optima.rag.embedding_simple import _slug  # local import: no heavy deps
        if _slug(self.slug) != self.slug:
            raise ValueError(f"Model slug {self.slug!r} must already be a valid corpus slug")


# Sizes are INFERRED estimates; the authoritative gate is check_fit(), which
# runs a meta-device parameter count against the config actually resolved
# from Hugging Face at call time, and is aware of every visible GPU.
MODEL_REGISTRY: dict[str, ModelSpec] = {

    # ============================================================
    # QWEN 2.5
    # ============================================================

    "qwen25-1.5b-instruct-fp16": ModelSpec(
        slug="qwen25-1.5b-instruct-fp16",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="Small Qwen2.5 instruction baseline.",
    ),

    "qwen25-3b-instruct-fp16": ModelSpec(
        slug="qwen25-3b-instruct-fp16",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="Control model; works without bitsandbytes.",
    ),

    "qwen25-7b-instruct-nf4": ModelSpec(
        slug="qwen25-7b-instruct-nf4",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quantization="nf4",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="General-purpose 7B instruction model.",
    ),

    "qwen25-coder-7b-instruct-nf4": ModelSpec(
        slug="qwen25-coder-7b-instruct-nf4",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        quantization="nf4",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="Code-specialized 7B model.",
    ),

    "qwen25-14b-instruct-nf4": ModelSpec(
        slug="qwen25-14b-instruct-nf4",
        model_id="Qwen/Qwen2.5-14B-Instruct",
        quantization="nf4",
        single_gpu_tier="B",
        max_input_tokens=4096,
        notes="General-purpose 14B model.",
    ),

    "qwen25-coder-14b-instruct-nf4": ModelSpec(
        slug="qwen25-coder-14b-instruct-nf4",
        model_id="Qwen/Qwen2.5-Coder-14B-Instruct",
        quantization="nf4",
        single_gpu_tier="B",
        max_input_tokens=4096,
        notes="Code-specialized 14B model.",
    ),

    "qwen25-32b-instruct-nf4": ModelSpec(
        slug="qwen25-32b-instruct-nf4",
        model_id="Qwen/Qwen2.5-32B-Instruct",
        quantization="nf4",
        single_gpu_tier="C",
        max_input_tokens=4096,
        notes="Large 32B model; requires multi-GPU sharding.",
    ),

    # ============================================================
    # QWEN 3
    # ============================================================

    "qwen3-0.6b-fp16": ModelSpec(
        slug="qwen3-0.6b-fp16",
        model_id="Qwen/Qwen3-0.6B",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        chat_template_kwargs={"enable_thinking": False},
        strip_think=True,
        notes="Small Qwen3 baseline.",
    ),

    "qwen3-1.7b-fp16": ModelSpec(
        slug="qwen3-1.7b-fp16",
        model_id="Qwen/Qwen3-1.7B",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        chat_template_kwargs={"enable_thinking": False},
        strip_think=True,
        notes="Small Qwen3 model.",
    ),

    "qwen3-4b-fp16": ModelSpec(
        slug="qwen3-4b-fp16",
        model_id="Qwen/Qwen3-4B",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        chat_template_kwargs={"enable_thinking": False},
        strip_think=True,
        notes="4B Qwen3 model.",
    ),

    "qwen3-8b-nf4": ModelSpec(
        slug="qwen3-8b-nf4",
        model_id="Qwen/Qwen3-8B",
        quantization="nf4",
        single_gpu_tier="A",
        max_input_tokens=6144,
        chat_template_kwargs={"enable_thinking": False},
        strip_think=True,
        notes="8B Qwen3 model.",
    ),

    "qwen3-14b-nf4": ModelSpec(
        slug="qwen3-14b-nf4",
        model_id="Qwen/Qwen3-14B",
        quantization="nf4",
        single_gpu_tier="B",
        max_input_tokens=4096,
        chat_template_kwargs={"enable_thinking": False},
        strip_think=True,
        notes="14B Qwen3 model.",
    ),

    "qwen3-30b-a3b-nf4": ModelSpec(
        slug="qwen3-30b-a3b-nf4",
        model_id="Qwen/Qwen3-30B-A3B",
        quantization="nf4",
        single_gpu_tier="C",
        max_input_tokens=4096,
        chat_template_kwargs={"enable_thinking": False},
        strip_think=True,
        notes=(
            "30B total-parameter MoE with approximately 3B active "
            "parameters per token. Verify NF4 Linear4bit coverage at runtime."
        ),
    ),

    # ============================================================
    # DEEPSEEK
    # ============================================================

    "deepseek-r1-distill-qwen-1.5b-fp16": ModelSpec(
        slug="deepseek-r1-distill-qwen-1.5b-fp16",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        strip_think=True,
        notes="Small reasoning-distilled model.",
    ),

    "deepseek-r1-distill-qwen-7b-nf4": ModelSpec(
        slug="deepseek-r1-distill-qwen-7b-nf4",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        quantization="nf4",
        single_gpu_tier="A",
        max_input_tokens=6144,
        strip_think=True,
        notes="7B reasoning-distilled model.",
    ),

    "deepseek-r1-distill-qwen-14b-nf4": ModelSpec(
        slug="deepseek-r1-distill-qwen-14b-nf4",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        quantization="nf4",
        single_gpu_tier="B",
        max_input_tokens=4096,
        strip_think=True,
        notes="14B reasoning-distilled model.",
    ),

    "deepseek-r1-distill-qwen-32b-nf4": ModelSpec(
        slug="deepseek-r1-distill-qwen-32b-nf4",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        quantization="nf4",
        single_gpu_tier="C",
        max_input_tokens=4096,
        strip_think=True,
        notes=(
            "32B reasoning-distilled model. Long reasoning output may "
            "increase retries and output truncation."
        ),
    ),

    "deepseek-r1-distill-llama-8b-nf4": ModelSpec(
        slug="deepseek-r1-distill-llama-8b-nf4",
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        quantization="nf4",
        single_gpu_tier="A",
        max_input_tokens=6144,
        strip_think=True,
        notes="8B reasoning-distilled Llama model.",
    ),

    # ============================================================
    # IBM GRANITE
    # ============================================================

    "granite-3.3-2b-instruct-fp16": ModelSpec(
        slug="granite-3.3-2b-instruct-fp16",
        model_id="ibm-granite/granite-3.3-2b-instruct",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="Small Granite instruction model.",
    ),

    "granite-3.3-8b-instruct-nf4": ModelSpec(
        slug="granite-3.3-8b-instruct-nf4",
        model_id="ibm-granite/granite-3.3-8b-instruct",
        quantization="nf4",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="8B Granite instruction model used in Optima.",
    ),

    # ============================================================
    # MISTRAL / MINISTRAL
    # ============================================================

    "ministral-3-3b-fp16": ModelSpec(
        slug="ministral-3-3b-fp16",
        model_id="mistralai/Ministral-3-3B",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="3B Ministral model used in Optima experiments.",
    ),

    "ministral-8b-instruct-nf4": ModelSpec(
        slug="ministral-8b-instruct-nf4",
        model_id="mistralai/Ministral-8B-Instruct-2410",
        quantization="nf4",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="8B Ministral instruction model.",
    ),

    "mistral-7b-instruct-nf4": ModelSpec(
        slug="mistral-7b-instruct-nf4",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        quantization="nf4",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="Classic 7B Mistral instruction baseline.",
    ),

    "mistral-small-24b-instruct-nf4": ModelSpec(
        slug="mistral-small-24b-instruct-nf4",
        model_id="mistralai/Mistral-Small-24B-Instruct-2501",
        quantization="nf4",
        single_gpu_tier="C",
        max_input_tokens=4096,
        notes="24B Mistral instruction model.",
    ),

    # ============================================================
    # MICROSOFT PHI
    # ============================================================

    "phi3-mini-4k-instruct-fp16": ModelSpec(
        slug="phi3-mini-4k-instruct-fp16",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=4096,
        notes="3.8B Phi-3 instruction baseline.",
    ),

    "phi4-mini-reasoning-fp16": ModelSpec(
        slug="phi4-mini-reasoning-fp16",
        model_id="microsoft/Phi-4-mini-reasoning",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        strip_think=True,
        notes="Reasoning model previously used in Optima.",
    ),

    "phi4-nf4": ModelSpec(
        slug="phi4-nf4",
        model_id="microsoft/phi-4",
        quantization="nf4",
        single_gpu_tier="B",
        max_input_tokens=4096,
        notes="14B Phi-4 instruction/reasoning model.",
    ),

    # ============================================================
    # META LLAMA
    # ============================================================

    "llama32-1b-instruct-fp16": ModelSpec(
        slug="llama32-1b-instruct-fp16",
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="Small Llama 3.2 baseline.",
    ),

    "llama32-3b-instruct-fp16": ModelSpec(
        slug="llama32-3b-instruct-fp16",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="Llama 3.2 3B model used in Optima experiments.",
    ),

    "llama31-8b-instruct-nf4": ModelSpec(
        slug="llama31-8b-instruct-nf4",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        quantization="nf4",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="Llama 3.1 8B instruction model.",
    ),

    "llama33-70b-instruct-nf4": ModelSpec(
        slug="llama33-70b-instruct-nf4",
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        quantization="nf4",
        single_gpu_tier="C",
        max_input_tokens=4096,
        notes="Large 70B Llama model; multi-GPU required.",
    ),

    # ============================================================
    # GOOGLE GEMMA
    # ============================================================

    "gemma3-1b-it-fp16": ModelSpec(
        slug="gemma3-1b-it-fp16",
        model_id="google/gemma-3-1b-it",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="Small Gemma 3 instruction model.",
    ),

    "gemma3-4b-it-fp16": ModelSpec(
        slug="gemma3-4b-it-fp16",
        model_id="google/gemma-3-4b-it",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="4B Gemma 3 instruction model.",
    ),

    "gemma3-12b-it-nf4": ModelSpec(
        slug="gemma3-12b-it-nf4",
        model_id="google/gemma-3-12b-it",
        quantization="nf4",
        single_gpu_tier="B",
        max_input_tokens=4096,
        notes="12B Gemma 3 instruction model.",
    ),

    "gemma3-27b-it-nf4": ModelSpec(
        slug="gemma3-27b-it-nf4",
        model_id="google/gemma-3-27b-it",
        quantization="nf4",
        single_gpu_tier="C",
        max_input_tokens=4096,
        notes="27B Gemma 3 instruction model.",
    ),

    # ============================================================
    # HUGGING FACE SMOLLM
    # ============================================================

    "smollm2-360m-instruct-fp16": ModelSpec(
        slug="smollm2-360m-instruct-fp16",
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=4096,
        notes="Very small instruction baseline.",
    ),

    "smollm2-1.7b-instruct-fp16": ModelSpec(
        slug="smollm2-1.7b-instruct-fp16",
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=6144,
        notes="1.7B instruction model used in Optima experiments.",
    ),

    # ============================================================
    # CODE-SPECIALIZED MODELS
    # ============================================================

    "starcoder2-3b-fp16": ModelSpec(
        slug="starcoder2-3b-fp16",
        model_id="bigcode/starcoder2-3b",
        quantization="fp16",
        single_gpu_tier="A",
        max_input_tokens=4096,
        notes="3B code-specialized model.",
    ),

    "starcoder2-7b-nf4": ModelSpec(
        slug="starcoder2-7b-nf4",
        model_id="bigcode/starcoder2-7b",
        quantization="nf4",
        single_gpu_tier="A",
        max_input_tokens=4096,
        notes="7B code-specialized model.",
    ),

    "starcoder2-15b-nf4": ModelSpec(
        slug="starcoder2-15b-nf4",
        model_id="bigcode/starcoder2-15b",
        quantization="nf4",
        single_gpu_tier="B",
        max_input_tokens=4096,
        notes="15B code-specialized model.",
    ),

    "codestral-22b-v0.1-nf4": ModelSpec(
        slug="codestral-22b-v0.1-nf4",
        model_id="mistralai/Codestral-22B-v0.1",
        quantization="nf4",
        single_gpu_tier="C",
        max_input_tokens=4096,
        notes="22B code-specialized model.",
    ),
}

def resolve_spec(name_or_slug: str, overrides: Optional[dict[str, Any]] = None,
                  num_gpus: int = 1, allow_infeasible: bool = False) -> ModelSpec:
    """Resolve a registry slug to a ModelSpec.

    ``single_gpu_tier`` is advisory metadata, not the feasibility gate: a
    Tier C spec is refused here only when ``num_gpus < 2`` (a genuinely
    single-GPU session). On a 2+ GPU session it resolves normally and is
    left to ``check_fit()`` -- the sole authority on whether it actually
    fits, sharded or otherwise.
    """
    if name_or_slug not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model slug {name_or_slug!r}. Known slugs: "
            f"{sorted(MODEL_REGISTRY)}. Add a new ModelSpec to MODEL_REGISTRY "
            f"instead of passing an arbitrary model_id."
        )
    spec = MODEL_REGISTRY[name_or_slug]
    if overrides:
        spec = replace(spec, **overrides)
    if spec.single_gpu_tier == "C" and num_gpus < 2 and not allow_infeasible:
        raise ModelNotFeasibleError(
            f"{spec.slug} ({spec.model_id}) is classified single_gpu_tier=C: "
            f"{spec.notes} It is refused on a {num_gpus}-GPU session unless "
            f"allow_infeasible=True or a second GPU is attached (num_gpus>=2), "
            f"in which case check_fit() performs the real, GPU-count-aware "
            f"feasibility check instead of this early refusal."
        )
    return spec


def spec_from_model_id(model_id: str, quantization: str = "nf4",
                        max_input_tokens: int = 1500, max_new_tokens: Optional[int] = None,
                        revision: Optional[str] = None,
                        chat_template_kwargs: Optional[dict[str, Any]] = None,
                        strip_think: bool = False,
                        allow_cpu_offload: bool = False) -> ModelSpec:
    """Build a ModelSpec directly from a Hugging Face model ID, bypassing
    MODEL_REGISTRY entirely -- for the simple "I type one model name and run
    it" workflow, where nothing consults or refuses on a registry tier.
    ``check_fit()``/``load_model_safe()`` are the same functions either way;
    this only skips the registry lookup that ``resolve_spec()`` does.

    ``max_new_tokens=None`` (the default) means no artificial output cap: the
    model generates until it emits EOS or exhausts its own context window
    (``max_new_tokens_for_context``/``generate_text``), which is what a fair
    model-comparison run wants -- a fixed round number we picked is not.
    """
    from optima.rag.embedding_simple import _slug  # local import: no heavy deps

    slug = _slug(model_id)
    return ModelSpec(
        slug=slug, model_id=model_id, quantization=quantization,
        single_gpu_tier="B",  # unused for ad hoc specs: check_fit() is the only real gate
        max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
        revision=revision, chat_template_kwargs=chat_template_kwargs or {},
        strip_think=strip_think, allow_cpu_offload=allow_cpu_offload,
        notes=f"Ad hoc spec for {model_id} (built from MODEL_NAME, not MODEL_REGISTRY).",
    )


def estimate_weights_gib(spec: ModelSpec) -> dict[str, Any]:
    """Estimate weight memory using a meta-device model (no weight download)."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(spec.model_id, revision=spec.revision, trust_remote_code=False)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    quantizable_params = 0
    other_params = 0
    tied = bool(getattr(config, "tie_word_embeddings", False))
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            is_lm_head = name.endswith("lm_head")
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if is_lm_head:
                if not tied:
                    other_params += params
            else:
                quantizable_params += params
    counted = {id(p) for m in model.modules() if isinstance(m, torch.nn.Linear) for p in m.parameters(recurse=False)}
    for _name, param in model.named_parameters():
        if id(param) not in counted:
            other_params += param.numel()

    del model
    gc.collect()

    if spec.quantization in ("nf4", "nf4-prequantized"):
        weights_gib = (quantizable_params * 0.53 + other_params * 2) / (1024 ** 3)
    else:
        weights_gib = (quantizable_params + other_params) * 2 / (1024 ** 3)
    return {
        "config": config, "quantizable_params": quantizable_params,
        "other_params": other_params, "weights_gib": round(weights_gib, 3),
    }


def _model_context_window(config: Any, default: int = 32768) -> int:
    return (
        getattr(config, "max_position_embeddings", None)
        or getattr(config, "n_positions", None)
        or getattr(config, "max_sequence_length", None)
        or getattr(config, "seq_length", None)
        or default
    )


def max_new_tokens_for_context(handle: Any, input_tokens: int, min_new_tokens: int = 256,
                                safety_margin: int = 64) -> int:
    """No artificial output cap: let the model generate until it hits its own
    context window, not a fixed number picked ahead of time. Used by
    ``enrich_one`` whenever ``GenerationSettings.max_new_tokens`` is ``None``
    (the default for a pure model-comparison run).
    """
    context_window = _model_context_window(handle.model.config)
    remaining = context_window - input_tokens - safety_margin
    return max(min_new_tokens, remaining)


def required_headroom_gib(spec: ModelSpec, config: Any) -> float:
    num_layers = getattr(config, "num_hidden_layers", None)
    num_kv_heads = getattr(config, "num_key_value_heads", None) or getattr(config, "num_attention_heads", None)
    hidden_size = getattr(config, "hidden_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "head_dim", None) or (
        hidden_size // num_heads if hidden_size and num_heads else None
    )
    if not all([num_layers, num_kv_heads, head_dim]):
        # Conservative fallback when config fields are missing.
        return 3.0
    max_new_tokens = spec.max_new_tokens
    if max_new_tokens is None:
        # No artificial cap: reserve for the worst case, where the model uses
        # its entire remaining context window for the response, so check_fit()
        # still reserves real headroom instead of under-counting it.
        max_new_tokens = max(256, _model_context_window(config) - spec.max_input_tokens)
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2
    total_tokens = spec.max_input_tokens + max_new_tokens
    kv_gib = kv_bytes_per_token * total_tokens / (1024 ** 3)
    return round(kv_gib + 1.0 + 0.3, 3)


@dataclass
class FitReport:
    num_gpus: int
    free_gib: list[float]
    per_gpu_reserved_gib: list[float]
    budget_gib: list[float]
    weights_gib: float
    total_headroom_gib: float
    placement: str  # "single_gpu" | "sharded" | "sharded_cpu_offload"
    fits: bool
    chosen_gpu: Optional[int] = None
    max_memory: Optional[dict[Any, str]] = None
    cpu_budget_gib: Optional[float] = None


def check_fit(spec: ModelSpec, num_gpus: Optional[int] = None,
              allow_cpu_offload: Optional[bool] = None, safety_factor: float = 0.95) -> FitReport:
    """Multi-GPU-aware pre-download VRAM fit check using a meta-device
    parameter count. This is the sole feasibility gate for a model -- the
    registry's ``single_gpu_tier`` (§6.1) is advisory only.

    This is a *benchmarking* pipeline: whenever more than one GPU is
    visible, every visible GPU is used to hold this one model instance --
    never just GPU 0 with a second GPU left idle, even when the model would
    comfortably fit alone on GPU 0. Decision order:

    1. single_gpu  - only reachable with exactly one visible GPU: it alone
                     holds weights + full headroom.
    2. sharded     - two or more GPUs are visible and the weights fit across
                     them combined. ``max_memory`` here is each GPU's real
                     safe budget (free VRAM, minus a safety factor, minus
                     this model's apportioned generation headroom) -- NOT an
                     artificially tightened "fair share". Balancing layers
                     across GPUs is ``device_map="balanced"``'s job (set in
                     ``load_model_safe``); it is restricted to GPU devices
                     only and never spills to CPU/disk the way
                     ``device_map="auto"`` does when a per-GPU cap is too
                     tight -- which is exactly what previously produced
                     "Some modules are dispatched on the CPU or the disk"
                     for a 4-bit model (4-bit modules cannot be CPU/disk
                     offloaded without ``llm_int8_enable_fp32_cpu_offload``,
                     which this pipeline deliberately does not set).
    3. sharded_cpu_offload - only if allow_cpu_offload; GPUs + a bounded
                     slice of free CPU RAM hold the weights. Orders of
                     magnitude slower; a documented last resort, not a
                     default path.
    Raises ModelDoesNotFitError if none of the above fit.
    """
    import torch

    num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
    if num_gpus == 0:
        raise ModelDoesNotFitError(
            f"No CUDA device is visible; cannot fit {spec.slug}. Enable a GPU runtime."
        )
    allow_cpu_offload = spec.allow_cpu_offload if allow_cpu_offload is None else allow_cpu_offload

    estimate = estimate_weights_gib(spec)
    weights_gib = estimate["weights_gib"]
    total_headroom_gib = required_headroom_gib(spec, estimate["config"])

    free_gib = [torch.cuda.mem_get_info(i)[0] / (1024 ** 3) for i in range(num_gpus)]
    per_gpu_reserved_gib = [max(1.0, total_headroom_gib / num_gpus) for _ in range(num_gpus)]
    budget_gib = [
        max(0.0, free_gib[i] * safety_factor - per_gpu_reserved_gib[i]) for i in range(num_gpus)
    ]

    def _print_table(placement: str, fits: bool) -> None:
        print(f"FIT ESTIMATE for {spec.slug} ({num_gpus} GPU(s)): "
              f"weights={weights_gib} GiB, total_headroom={total_headroom_gib} GiB")
        for i in range(num_gpus):
            print(f"  gpu[{i}]: free={round(free_gib[i], 3)} GiB, "
                  f"reserved={round(per_gpu_reserved_gib[i], 3)} GiB, "
                  f"budget={round(budget_gib[i], 3)} GiB")
        print(f"  decision: placement={placement}, fits={fits}")

    # 1. Single-GPU fit: only considered when exactly one GPU is visible.
    #    With two or more visible GPUs this branch is skipped entirely --
    #    see the module-level note above: a second idle GPU is not an
    #    acceptable placement for this benchmarking pipeline.
    if num_gpus == 1:
        single_gpu_margin = free_gib[0] * safety_factor - total_headroom_gib - weights_gib
        if single_gpu_margin >= 0:
            report = FitReport(
                num_gpus=num_gpus, free_gib=free_gib, per_gpu_reserved_gib=per_gpu_reserved_gib,
                budget_gib=budget_gib, weights_gib=weights_gib, total_headroom_gib=total_headroom_gib,
                placement="single_gpu", fits=True, chosen_gpu=0, max_memory=None,
            )
            _print_table("single_gpu", True)
            return report

    # 2. Sharded fit (num_gpus >= 2): do the GPUs combined (each with its own
    #    apportioned headroom reservation) hold the weights? Every visible
    #    GPU's real safe budget is offered to the planner; device_map=
    #    "balanced" (load_model_safe) is what actually spreads layers across
    #    all of them instead of filling GPU 0 alone -- max_memory here only
    #    bounds each GPU so generation headroom is never eaten by weights.
    elif sum(budget_gib) >= weights_gib:
        max_memory = {i: f"{budget_gib[i]:.2f}GiB" for i in range(num_gpus) if budget_gib[i] > 0}
        report = FitReport(
            num_gpus=num_gpus, free_gib=free_gib, per_gpu_reserved_gib=per_gpu_reserved_gib,
            budget_gib=budget_gib, weights_gib=weights_gib, total_headroom_gib=total_headroom_gib,
            placement="sharded", fits=True, max_memory=max_memory,
        )
        _print_table("sharded", True)
        return report

    # 3. Sharded + CPU offload: opt-in only, and only as a last resort.
    if allow_cpu_offload:
        cpu_budget_gib = (environment.cpu_free_ram_gib() or 0.0) * 0.7
        if sum(budget_gib) + cpu_budget_gib >= weights_gib:
            max_memory = {i: f"{budget_gib[i]:.2f}GiB" for i in range(num_gpus) if budget_gib[i] > 0}
            max_memory["cpu"] = f"{cpu_budget_gib:.2f}GiB"
            print(f"WARNING: {spec.slug} requires CPU offload to fit "
                  f"({cpu_budget_gib:.2f} GiB of CPU RAM will hold some layers). "
                  f"This is an order of magnitude slower than pure-GPU generation "
                  f"and is NOT comparable to other models' latency; "
                  f"used_cpu_offload=True will be recorded.")
            report = FitReport(
                num_gpus=num_gpus, free_gib=free_gib, per_gpu_reserved_gib=per_gpu_reserved_gib,
                budget_gib=budget_gib, weights_gib=weights_gib, total_headroom_gib=total_headroom_gib,
                placement="sharded_cpu_offload", fits=True, max_memory=max_memory,
                cpu_budget_gib=cpu_budget_gib,
            )
            _print_table("sharded_cpu_offload", True)
            return report

    _print_table("none", False)
    cpu_note = (
        "CPU offload was considered but insufficient." if allow_cpu_offload
        else "CPU offload was not enabled for this model (allow_cpu_offload=False)."
    )
    raise ModelDoesNotFitError(
        f"{spec.slug} does not fit: weights={weights_gib} GiB, "
        f"total_headroom={total_headroom_gib} GiB, per-GPU free={[round(g, 2) for g in free_gib]} GiB "
        f"across {num_gpus} GPU(s). {cpu_note} "
        f"Try: lower max_input_tokens, confirm nf4 quantization is selected, "
        f"attach a second GPU, or (if appropriate for this model) enable allow_cpu_offload."
    )


def check_disk_for_download(spec: ModelSpec, hf_home: Optional[str] = None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(spec.model_id, revision=spec.revision, files_metadata=True)
    total_bytes = sum(
        (sibling.size or 0) for sibling in (info.siblings or [])
        if sibling.rfilename.endswith((".safetensors", ".bin"))
    )
    required_gib = (total_bytes * 1.1) / (1024 ** 3)
    import os as _os
    cache_root = hf_home or _os.environ.get("HF_HOME") or "~/.cache/huggingface"
    free_gib = environment.disk_free_gib(_os.path.expanduser(cache_root))
    report = {"required_gib": round(required_gib, 2), "free_gib": free_gib, "ok": free_gib >= required_gib}
    print(f"DISK CHECK for {spec.slug}: needs ~{report['required_gib']} GiB, "
          f"{report['free_gib']} GiB free")
    if not report["ok"]:
        raise InsufficientDiskError(
            f"{spec.slug} needs an estimated {report['required_gib']} GiB of disk "
            f"to download but only {report['free_gib']} GiB is free. Delete a "
            f"previous model's cache (DELETE_MODEL_CACHE_AFTER_UNLOAD=True) or "
            f"choose a smaller model."
        )
    return report


@dataclass
class ModelHandle:
    spec: ModelSpec
    model: Any
    tokenizer: Any
    input_device: Any
    load_report: dict[str, Any]


def load_model_safe(spec: ModelSpec, bnb_status: Optional["environment.BnbStatus"] = None) -> ModelHandle:
    """Load a single causal LM instance with explicit device placement -- one
    GPU when the model fits alone (``num_gpus == 1``), or ``device_map=
    "balanced"`` across every visible GPU (with an explicit, pre-computed,
    GPU-only ``max_memory``) otherwise -- plus full post-load verification.
    "balanced" never spills a module to CPU/disk, which quantized (nf4)
    weights cannot tolerate; there is no CPU/disk fallback path here except
    the separate, opt-in ``sharded_cpu_offload`` placement. Never falls back
    from nf4 to fp16.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ModelLoadError(
            f"CUDA is unavailable; cannot load {spec.model_id}.", category="load_failed",
        )

    fit = check_fit(spec, num_gpus=num_gpus)
    check_disk_for_download(spec)

    if spec.quantization == "nf4":
        if bnb_status is None:
            from .errors import QuantizationUnavailableError
            raise QuantizationUnavailableError(
                f"{spec.slug} requires nf4 quantization but no BnbStatus was "
                f"provided to load_model_safe(); run environment.probe_bitsandbytes() first."
            )
        environment.require_bitsandbytes(bnb_status, spec.slug)

    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision, use_fast=True)
    except (OSError, ValueError) as exc:
        raise ModelLoadError(
            f"Could not load tokenizer for {spec.model_id!r}: {exc}",
            category="load_failed", stage="tokenizer",
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ModelLoadError(
            f"{spec.model_id} has no chat_template; cannot build enrichment "
            f"prompts consistently.", category="load_failed", stage="tokenizer",
        )

    # Both T4s share the same compute capability, so a single dtype decision
    # (read from GPU 0) is correct even when sharding across both; a
    # heterogeneous GPU pair would need a per-GPU dtype decision, out of scope.
    major, _minor = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16

    if fit.placement == "single_gpu":
        device_map: Any = {"": fit.chosen_gpu}
    elif fit.placement == "sharded":
        # GPU-only multi-GPU placement: "balanced" is restricted to the GPU
        # devices in max_memory and never spills a module to CPU/disk when a
        # per-GPU budget is tight -- unlike device_map="auto", which treats
        # CPU/disk as ordinary fallback targets and is what previously
        # produced "Some modules are dispatched on the CPU or the disk" for
        # this 4-bit model (4-bit modules cannot be CPU/disk offloaded
        # without llm_int8_enable_fp32_cpu_offload, which is deliberately
        # never set here -- see check_fit()'s docstring).
        device_map = "balanced"
    else:
        # "sharded_cpu_offload": the one placement that deliberately wants
        # CPU included, via the explicit "cpu" entry in fit.max_memory --
        # "balanced" would refuse that entirely, so "auto" is still correct
        # here, and only reached when allow_cpu_offload was explicitly set.
        device_map = "auto"

    load_kwargs: dict[str, Any] = {
        "revision": spec.revision, "device_map": device_map, "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if fit.placement != "single_gpu":
        load_kwargs["max_memory"] = fit.max_memory
    try:
        from transformers import __version__ as _tf_version
        if tuple(int(p) for p in _tf_version.split(".")[:2]) >= (4, 56):
            load_kwargs["dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = dtype
    except Exception:  # noqa: BLE001 - best-effort version detection
        load_kwargs["torch_dtype"] = dtype

    if spec.quantization == "nf4":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )

    start = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        _cleanup_gpu()
        raise ModelLoadError(
            f"CUDA OOM while loading {spec.model_id} ({spec.quantization}): {exc}",
            category="cuda_oom_on_load",
        ) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise ModelLoadError(
            f"Failed to load {spec.model_id} ({spec.quantization}): {exc}",
            category="load_failed",
        ) from exc
    load_seconds = time.perf_counter() - start

    model.eval()
    resolved_device_map = getattr(model, "hf_device_map", None) or {"": fit.chosen_gpu or 0}

    # Print the full device map unconditionally: this is the artifact that
    # lets the user visually confirm a sharded 30B-class load actually landed
    # on both cuda:0 and cuda:1. A single-GPU load legitimately shows only
    # one device -- that is expected, not a failure.
    print(f"DEVICE MAP for {spec.slug}:")
    print(json.dumps({str(k): str(v) for k, v in resolved_device_map.items()}, indent=2))

    gpu_indices_used = {
        idx for idx in (_device_index(v) for v in resolved_device_map.values()) if idx is not None
    }
    allowed = {i for i in range(num_gpus)} | {f"cuda:{i}" for i in range(num_gpus)}
    if fit.placement == "sharded_cpu_offload":
        allowed = allowed | {"cpu"}
    bad_devices = {v for v in resolved_device_map.values() if v not in allowed}
    if bad_devices:
        del model
        _cleanup_gpu()
        raise ModelLoadError(
            f"{spec.model_id} was placed on unsupported devices {bad_devices} "
            f"(disk offload is never supported; CPU offload is only allowed "
            f"when placement=='sharded_cpu_offload', currently {fit.placement!r}).",
            category="load_failed",
        )

    if fit.placement == "sharded" and len(gpu_indices_used) < 2:
        print(f"WARNING: {spec.slug} was given a balanced sharded device_map "
              f"across {num_gpus} GPU(s) but Accelerate placed it entirely on "
              f"GPU {gpu_indices_used}; the requested multi-GPU placement was "
              f"NOT achieved. gate1_load() below checks for this and fails "
              f"the gate rather than silently proceeding on one GPU.")

    if spec.quantization == "nf4":
        has_4bit = any(type(m).__name__ == "Linear4bit" for m in model.modules())
        if not has_4bit:
            del model
            _cleanup_gpu()
            raise ModelLoadError(
                f"{spec.model_id} was requested in nf4 but no Linear4bit "
                f"modules were found after loading; quantization silently "
                f"did not apply.", category="load_failed",
            )

    footprint_gib = model.get_memory_footprint() / (1024 ** 3)
    allocated_gib = sum(torch.cuda.memory_allocated(i) for i in range(num_gpus)) / (1024 ** 3)

    # Per-GPU headroom check: every GPU actually holding part of the model
    # must independently satisfy the reservation check_fit() used to build
    # max_memory. A single under-provisioned GPU (often GPU 0, which the
    # device map typically also loads with embedding/LM-head/first-layer
    # overhead) is named specifically rather than averaged away.
    required_headroom = required_headroom_gib(spec, model.config)
    insufficient = []
    for i in sorted(gpu_indices_used) or [0]:
        free_i_gib = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)
        reserved_i_gib = fit.per_gpu_reserved_gib[i] if i < len(fit.per_gpu_reserved_gib) else required_headroom
        if free_i_gib < reserved_i_gib:
            insufficient.append((i, round(free_i_gib, 3), round(reserved_i_gib, 3)))
    if insufficient:
        del model
        _cleanup_gpu()
        detail = ", ".join(f"GPU {i}: {free} GiB free < {reserved} GiB reserved"
                           for i, free, reserved in insufficient)
        raise InsufficientHeadroomError(
            f"{spec.model_id} loaded (footprint {footprint_gib:.2f} GiB) but "
            f"{len(insufficient)} GPU(s) are below their reserved headroom for "
            f"max_input_tokens={spec.max_input_tokens}+max_new_tokens="
            f"{spec.max_new_tokens}: {detail}. Lower max_input_tokens or "
            f"choose a smaller/more-quantized model."
        )
    free_gib_after = min(torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)) / (1024 ** 3)

    resolved_commit = getattr(model.config, "_commit_hash", None)
    input_device = model.get_input_embeddings().weight.device

    # Per-GPU allocated memory, explicitly, so a sharded load's actual
    # footprint on EVERY visible GPU (not just a combined total) is visible
    # in the checkpointed manifest and printed below -- the direct evidence
    # that GPU 1 (etc.) actually holds parameters, not just GPU 0.
    per_gpu_allocated_gib = {
        i: round(torch.cuda.memory_allocated(i) / (1024 ** 3), 3) for i in range(num_gpus)
    }

    load_report = {
        "load_seconds": round(load_seconds, 2), "footprint_gib": round(footprint_gib, 3),
        "allocated_gib": round(allocated_gib, 3), "free_gib": round(free_gib_after, 3),
        "per_gpu_allocated_gib": per_gpu_allocated_gib,
        "required_headroom_gib": required_headroom, "resolved_commit_hash": resolved_commit,
        "dtype": str(dtype),
        "device_map_summary": {str(k): str(v) for k, v in resolved_device_map.items()},
        "has_linear4bit": spec.quantization == "nf4",
        "gpu_placement": fit.placement, "gpu_count_used": len(gpu_indices_used) or 1,
        "used_cpu_offload": fit.placement == "sharded_cpu_offload",
        "fit_report": vars(fit),
    }
    print(f"MODEL LOADED: {spec.slug} in {load_report['load_seconds']}s, "
          f"placement={fit.placement}, gpus_used={sorted(gpu_indices_used) or [0]}, "
          f"footprint={load_report['footprint_gib']} GiB, free_after={load_report['free_gib']} GiB")
    print(f"PER-GPU ALLOCATED for {spec.slug}: {per_gpu_allocated_gib}")
    return ModelHandle(spec=spec, model=model, tokenizer=tokenizer,
                        input_device=input_device, load_report=load_report)


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    finish_reason: str  # "eos" | "length"
    peak_allocated_gib: float
    free_after_gib: float
    think_stripped: bool = False


def _strip_think(text: str) -> tuple[str, bool]:
    marker_open, marker_close = "<think>", "</think>"
    if marker_open in text and marker_close in text:
        start = text.find(marker_open)
        end = text.find(marker_close, start) + len(marker_close)
        return (text[:start] + text[end:]).strip(), True
    return text, False


def generate_text(handle: ModelHandle, messages: list[dict[str, str]],
                   max_new_tokens: int, do_sample: bool = False,
                   temperature: Optional[float] = None,
                   top_p: Optional[float] = None) -> GenerationResult:
    """Run one bounded generation with explicit, model-agnostic decoding
    settings. Does not catch torch.cuda.OutOfMemoryError: callers decide how
    to react (an OOM here aborts the enrichment run rather than being
    silently swallowed).
    """
    import torch

    tokenizer = handle.tokenizer
    model = handle.model
    spec = handle.spec

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **spec.chat_template_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - reported, not retried here
        raise RuntimeError(f"prompt_build_error: {type(exc).__name__}: {exc}") from exc

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_tokens = int(encoded["input_ids"].shape[1])
    encoded = {key: value.to(handle.input_device) for key, value in encoded.items()}

    # Reset stats on every visible GPU, not just torch.cuda.current_device()
    # (the bare no-argument form), which would silently miss a second GPU
    # holding part of a sharded model.
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens, "do_sample": do_sample,
        "repetition_penalty": 1.0, "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None
        gen_kwargs["top_k"] = None
    eos_id = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
    gen_kwargs["eos_token_id"] = eos_id

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, **gen_kwargs)
    latency_s = time.perf_counter() - start

    generated_ids = output[0][encoded["input_ids"].shape[1]:]
    output_tokens = int(generated_ids.numel())
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    last_token_is_eos = bool(
        output_tokens > 0 and eos_id is not None
        and int(generated_ids[-1].item()) in (eos_id if isinstance(eos_id, (list, tuple)) else (eos_id,))
    )
    finish_reason = "eos" if last_token_is_eos or output_tokens < max_new_tokens else "length"

    think_stripped = False
    if spec.strip_think:
        text, think_stripped = _strip_think(text)

    # Aggregate across every visible GPU: peak is the SUM (a sharded model's
    # total footprint is what determines whether the next model fits, not
    # any single device's number); free is the MIN (the tightest GPU is what
    # will OOM first on the next call, not the average or the sum).
    num_gpus = torch.cuda.device_count()
    peak_allocated_gib = sum(torch.cuda.max_memory_allocated(i) for i in range(num_gpus)) / (1024 ** 3)
    free_after_gib = min(torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)) / (1024 ** 3)

    del encoded, output, generated_ids

    return GenerationResult(
        text=text, input_tokens=input_tokens, output_tokens=output_tokens,
        latency_s=round(latency_s, 4), finish_reason=finish_reason,
        peak_allocated_gib=round(peak_allocated_gib, 4),
        free_after_gib=round(free_after_gib, 4), think_stripped=think_stripped,
    )


def _cleanup_gpu() -> None:
    """Release cached allocator memory on EVERY visible GPU.

    torch.cuda.empty_cache() and torch.cuda.reset_peak_memory_stats() with no
    explicit device act on torch.cuda.current_device() only -- calling them
    once, unscoped, after unloading a model sharded across GPU 0 and GPU 1
    would silently leave GPU 1's cached allocator memory unreleased.
    """
    import torch
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        torch.cuda.ipc_collect()  # process-global, not per-device


def unload_model(handle: Optional[ModelHandle], delete_cache: bool = False) -> None:
    """Release a model's GPU memory and, optionally, its on-disk HF cache."""
    import sys

    if handle is not None:
        model_id = handle.spec.model_id
        handle.model = None
        handle.tokenizer = None
    else:
        model_id = None

    sys.last_type = sys.last_value = sys.last_traceback = None  # type: ignore[attr-defined]
    _cleanup_gpu()
    environment.assert_gpu_clean(threshold_gib=0.3)

    if delete_cache and model_id:
        import shutil as _shutil
        from huggingface_hub import scan_cache_dir
        try:
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == model_id:
                    _shutil.rmtree(repo.repo_path, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001 - cache cleanup is best-effort
            print(f"WARNING: could not delete HF cache for {model_id}: {exc}")
    print(f"MODEL UNLOADED: {model_id or '(none)'}, delete_cache={delete_cache}")


def emergency_unload(namespace: dict[str, Any]) -> None:
    """Best-effort unload used from an except/finally block to guarantee GPU
    memory is released even when a gate or generation call raised.
    """
    handle = namespace.get("HANDLE")
    if handle is not None:
        try:
            unload_model(handle, delete_cache=False)
        except Exception as exc:  # noqa: BLE001 - this runs during error handling
            print(f"WARNING: emergency_unload could not fully clean up: {exc}")
        namespace["HANDLE"] = None
