"""Model registry, VRAM fit estimation, and memory-conscious Hugging Face
model loading/generation/unloading for a single T4 (or similar) GPU.

Nothing here imports torch/transformers/accelerate at module load time:
those imports happen inside functions so that ``MODEL_REGISTRY`` and
``resolve_spec`` can be inspected (and unit tested) without a GPU or those
packages installed.
"""

from __future__ import annotations

import gc
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

_VALID_TIERS = ("A", "B", "C")
_VALID_QUANTS = ("fp16", "nf4", "nf4-prequantized")


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_id: str
    quantization: str  # "fp16" | "nf4" | "nf4-prequantized"
    tier: str  # "A" (fits comfortably) | "B" (fits, tight) | "C" (does not fit on 1xT4)
    max_input_tokens: int
    revision: Optional[str] = None
    max_new_tokens: int = 768
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    strip_think: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.tier not in _VALID_TIERS:
            raise ValueError(f"Invalid tier {self.tier!r} for {self.slug}")
        if self.quantization not in _VALID_QUANTS:
            raise ValueError(f"Invalid quantization {self.quantization!r} for {self.slug}")
        if self.slug in ("raw", "base"):
            raise ValueError(f"Model slug {self.slug!r} collides with a reserved corpus name")
        from optima.rag.embedding_simple import _slug  # local import: no heavy deps
        if _slug(self.slug) != self.slug:
            raise ValueError(f"Model slug {self.slug!r} must already be a valid corpus slug")


# Sizes are INFERRED estimates; the authoritative gate is check_fit(), which
# runs a meta-device parameter count against the config actually resolved
# from Hugging Face at call time.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "qwen25-3b-instruct-fp16": ModelSpec(
        slug="qwen25-3b-instruct-fp16", model_id="Qwen/Qwen2.5-3B-Instruct",
        quantization="fp16", tier="A", max_input_tokens=6144,
        notes="Control model; works without bitsandbytes (~6.2 GiB weights).",
    ),
    "qwen25-7b-instruct-nf4": ModelSpec(
        slug="qwen25-7b-instruct-nf4", model_id="Qwen/Qwen2.5-7B-Instruct",
        quantization="nf4", tier="A", max_input_tokens=6144,
        notes="~5.5 GiB in nf4. Recommended default first model.",
    ),
    "qwen25-coder-7b-instruct-nf4": ModelSpec(
        slug="qwen25-coder-7b-instruct-nf4", model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        quantization="nf4", tier="A", max_input_tokens=6144,
        notes="Code-specialized 7B.",
    ),
    "qwen25-14b-instruct-nf4": ModelSpec(
        slug="qwen25-14b-instruct-nf4", model_id="Qwen/Qwen2.5-14B-Instruct",
        quantization="nf4", tier="B", max_input_tokens=4096,
        notes="~9.5-10 GiB in nf4; tight headroom on a 14.56 GiB T4.",
    ),
    "qwen25-coder-14b-instruct-nf4": ModelSpec(
        slug="qwen25-coder-14b-instruct-nf4", model_id="Qwen/Qwen2.5-Coder-14B-Instruct",
        quantization="nf4", tier="B", max_input_tokens=4096,
        notes="Code-specialized 14B; same headroom profile as qwen25-14b.",
    ),
    "qwen3-14b-nf4": ModelSpec(
        slug="qwen3-14b-nf4", model_id="Qwen/Qwen3-14B",
        quantization="nf4", tier="B", max_input_tokens=4096,
        chat_template_kwargs={"enable_thinking": False}, strip_think=True,
        notes="Needs transformers>=4.51 (NOT CONFIRMED on Kaggle's preinstalled version).",
    ),
    "qwen25-32b-instruct-nf4": ModelSpec(
        slug="qwen25-32b-instruct-nf4", model_id="Qwen/Qwen2.5-32B-Instruct",
        quantization="nf4", tier="C", max_input_tokens=4096,
        notes="~18-19 GiB in nf4. Does not fit on 1xT4; refused unless ALLOW_INFEASIBLE.",
    ),
    "qwen3-30b-a3b-nf4": ModelSpec(
        slug="qwen3-30b-a3b-nf4", model_id="Qwen/Qwen3-30B-A3B",
        quantization="nf4", tier="C", max_input_tokens=4096,
        notes="MoE stores all experts regardless of active params; refused.",
    ),
    "deepseek-r1-distill-qwen-32b-nf4": ModelSpec(
        slug="deepseek-r1-distill-qwen-32b-nf4", model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        quantization="nf4", tier="C", max_input_tokens=4096,
        strip_think=True,
        notes="Does not fit; also emits long <think> output that conflicts with bounded JSON. Refused.",
    ),
}


def resolve_spec(name_or_slug: str, overrides: Optional[dict[str, Any]] = None,
                  allow_infeasible: bool = False) -> ModelSpec:
    if name_or_slug not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model slug {name_or_slug!r}. Known slugs: "
            f"{sorted(MODEL_REGISTRY)}. Add a new ModelSpec to MODEL_REGISTRY "
            f"instead of passing an arbitrary model_id."
        )
    spec = MODEL_REGISTRY[name_or_slug]
    if overrides:
        spec = replace(spec, **overrides)
    if spec.tier == "C" and not allow_infeasible:
        raise ModelNotFeasibleError(
            f"{spec.slug} ({spec.model_id}) is classified Tier C: "
            f"{spec.notes} It is refused on a single T4 unless "
            f"allow_infeasible=True (and even then check_fit() will still "
            f"block it if it genuinely does not fit)."
        )
    return spec


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
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2
    total_tokens = spec.max_input_tokens + spec.max_new_tokens
    kv_gib = kv_bytes_per_token * total_tokens / (1024 ** 3)
    return round(kv_gib + 1.0 + 0.3, 3)


def check_fit(spec: ModelSpec, safety_factor: float = 0.95) -> dict[str, Any]:
    """Pre-download VRAM fit check using a meta-device parameter count.
    Raises ModelDoesNotFitError before any weight download if it will not fit.
    """
    import torch

    estimate = estimate_weights_gib(spec)
    headroom_gib = required_headroom_gib(spec, estimate["config"])
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024 ** 3)
    required_gib = estimate["weights_gib"] + headroom_gib
    margin_gib = free_gib * safety_factor - required_gib

    report = {
        "slug": spec.slug, "model_id": spec.model_id, "quantization": spec.quantization,
        "weights_gib": estimate["weights_gib"], "headroom_gib": headroom_gib,
        "required_gib": round(required_gib, 3), "free_gib": round(free_gib, 3),
        "margin_gib": round(margin_gib, 3), "fits": margin_gib >= 0,
    }
    print(f"FIT ESTIMATE for {spec.slug}: weights={report['weights_gib']} GiB, "
          f"headroom={report['headroom_gib']} GiB, required={report['required_gib']} GiB, "
          f"free={report['free_gib']} GiB, margin={report['margin_gib']} GiB")
    if not report["fits"]:
        raise ModelDoesNotFitError(
            f"{spec.slug} is estimated to need {report['required_gib']} GiB "
            f"(weights {report['weights_gib']} + headroom {report['headroom_gib']}) "
            f"but only {report['free_gib']} GiB is free (margin {report['margin_gib']} GiB). "
            f"Try: lower max_input_tokens, switch to nf4 if not already, or "
            f"choose a Tier A model."
        )
    return report


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
    """Load a causal LM with explicit, single-GPU placement and full
    post-load verification. Never falls back from nf4 to fp16.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise ModelLoadError(
            f"CUDA is unavailable; cannot load {spec.model_id}.", category="load_failed",
        )

    check_fit(spec)
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

    major, _minor = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16

    load_kwargs: dict[str, Any] = {
        "revision": spec.revision, "device_map": {"": 0}, "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
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
    device_map = getattr(model, "hf_device_map", None) or {"": 0}
    allowed_devices = {0, "cuda:0"}
    bad_devices = {v for v in device_map.values() if v not in allowed_devices}
    if bad_devices:
        del model
        _cleanup_gpu()
        raise ModelLoadError(
            f"{spec.model_id} was placed on non-GPU-0 devices {bad_devices} "
            f"(CPU/disk offload is not supported by this pipeline).",
            category="load_failed",
        )

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
    allocated_gib = torch.cuda.memory_allocated() / (1024 ** 3)
    free_bytes, _total = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024 ** 3)
    required_headroom = required_headroom_gib(spec, model.config)
    if free_gib < required_headroom:
        del model
        _cleanup_gpu()
        raise InsufficientHeadroomError(
            f"{spec.model_id} loaded (footprint {footprint_gib:.2f} GiB) but "
            f"only {free_gib:.2f} GiB free VRAM remains, below the required "
            f"headroom of {required_headroom:.2f} GiB for "
            f"max_input_tokens={spec.max_input_tokens}+max_new_tokens="
            f"{spec.max_new_tokens}. Lower max_input_tokens or choose a "
            f"smaller/more-quantized model."
        )

    resolved_commit = getattr(model.config, "_commit_hash", None)
    input_device = model.get_input_embeddings().weight.device

    load_report = {
        "load_seconds": round(load_seconds, 2), "footprint_gib": round(footprint_gib, 3),
        "allocated_gib": round(allocated_gib, 3), "free_gib": round(free_gib, 3),
        "required_headroom_gib": required_headroom, "resolved_commit_hash": resolved_commit,
        "dtype": str(dtype), "device_map_summary": {str(k): str(v) for k, v in device_map.items()},
        "has_linear4bit": spec.quantization == "nf4",
    }
    print(f"MODEL LOADED: {spec.slug} in {load_report['load_seconds']}s, "
          f"footprint={load_report['footprint_gib']} GiB, free_after={load_report['free_gib']} GiB")
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

    torch.cuda.reset_peak_memory_stats()
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

    peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    free_bytes, _total = torch.cuda.mem_get_info()
    free_after_gib = free_bytes / (1024 ** 3)

    del encoded, output, generated_ids

    return GenerationResult(
        text=text, input_tokens=input_tokens, output_tokens=output_tokens,
        latency_s=round(latency_s, 4), finish_reason=finish_reason,
        peak_allocated_gib=round(peak_allocated_gib, 4),
        free_after_gib=round(free_after_gib, 4), think_stripped=think_stripped,
    )


def _cleanup_gpu() -> None:
    import torch
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


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
