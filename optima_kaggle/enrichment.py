"""Bounded prompting, per-function enrichment with a full failure taxonomy,
a checkpointed/resumable full run, and the four smoke-test gates.

Reuses (never reimplements): ``colab.colab_pipeline.build_enrichment_messages``,
``_valid_enrichment``, ``_fallback``, ``_parse_json_object``, ``_trim_source``,
and ``optima.enricher._evaluation`` / ``_dataset_metrics``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .errors import (
    CheckpointCorruptError,
    EnrichmentQualityError,
    GateFailedError,
    GenerationOOMError,
    PromptTooLongError,
    ResumeMismatchError,
    SystemicEnrichmentFailure,
)

RETRYABLE_CATEGORIES = {"generation_error", "invalid_json", "schema_invalid", "output_truncated"}
REPAIRABLE_CATEGORIES = {"invalid_json", "schema_invalid", "output_truncated"}
REQUIRED_GATES = ("gate1_load", "gate2_trivial", "gate3_one_function", "gate4_three_functions")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GenerationSettings:
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: int = 768
    retries: int = 2
    prompt_variant: str = "colab_v2_no_module_ir"  # or "colab_v2" (exact Colab behavior)
    repetition_penalty: float = 1.0


def config_hash(spec: Any, gen: GenerationSettings) -> str:
    """Identity of (model + quantization + generation settings). Two runs with
    the same hash are directly comparable; a different hash requires a new
    checkpoint (new slug suffix) rather than silently mixing results.
    """
    payload = {
        "model_id": spec.model_id, "revision": spec.revision,
        "quantization": spec.quantization, "max_input_tokens": spec.max_input_tokens,
        "max_new_tokens": gen.max_new_tokens, "do_sample": gen.do_sample,
        "temperature": gen.temperature, "top_p": gen.top_p,
        "repetition_penalty": gen.repetition_penalty, "prompt_variant": gen.prompt_variant,
        "retries": gen.retries, "chat_template_kwargs": spec.chat_template_kwargs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _count_tokens(tokenizer: Any, messages: list[dict[str, str]], spec: Any) -> int:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **spec.chat_template_kwargs
    )
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def build_bounded_messages(function: dict[str, Any], tokenizer: Any, spec: Any,
                            prompt_variant: str) -> tuple[list[dict[str, str]], int, list[str]]:
    """Build the enrichment prompt and reduce it, front-to-back in a fixed
    order, until it fits spec.max_input_tokens. Never truncates from the
    right (which would cut the chat template's closing/assistant turn).
    """
    from colab.colab_pipeline import build_enrichment_messages
    from optima.enricher import _trim_source

    view: dict[str, Any] = dict(function)
    if prompt_variant == "colab_v2_no_module_ir":
        # llvm_ir on a function record is the WHOLE translation unit's IR, not
        # a per-function slice; its first ~5000 chars are just the module
        # preamble (ModuleID/datalayout/type declarations), identical for
        # every function in a file. Per-function structure is already
        # available via function["llvm"]["basic_blocks"]/["cfg"].
        view["llvm_ir"] = ""
        # AST/CFG are large compiler structures that are not part of the
        # "useful local information" a function-level enrichment prompt
        # needs (name, qualified name, parameters, return type, source,
        # short calls list); excluding them by default -- rather than only
        # as a fallback once over budget -- is what keeps typical prompts in
        # the few-hundred-to-~1500-token range instead of the 3000+ tokens
        # they reached when AST/CFG were included by default.
        view["ast"] = ""
        view["cfg"] = ""

    messages = build_enrichment_messages(view)
    input_tokens = _count_tokens(tokenizer, messages, spec)
    reductions: list[str] = []
    if input_tokens <= spec.max_input_tokens:
        return messages, input_tokens, reductions

    steps: list[tuple[str, str, Any]] = [
        ("ast_removed", "ast", ""),
        ("cfg_removed", "cfg", ""),
        ("calls_removed", "calls", ""),
    ]
    if prompt_variant == "colab_v2":
        steps.append(("llvm_ir_removed", "llvm_ir", ""))
    for limit in (6000, 3000, 1500):
        steps.append((f"source_trimmed_{limit}", "source_code", limit))

    for name, field_name, value in steps:
        if field_name == "source_code":
            view = {**view, "source_code": _trim_source(view.get("source_code") or "", value)}
        else:
            view = {**view, field_name: value}
        reductions.append(name)
        messages = build_enrichment_messages(view)
        input_tokens = _count_tokens(tokenizer, messages, spec)
        if input_tokens <= spec.max_input_tokens:
            return messages, input_tokens, reductions

    raise PromptTooLongError(
        f"Function {function.get('id')} still needs {input_tokens} tokens "
        f"(limit {spec.max_input_tokens}) after applying every reduction: {reductions}."
    )


def _fully_reduced_view(function: dict[str, Any]) -> dict[str, Any]:
    """The most aggressive per-function reduction: every compiler-structural
    field dropped and source code cut to 1500 characters. Used only for the
    one-time OOM-recovery retry in ``enrich_one`` -- not a normal step of the
    ``build_bounded_messages`` reduction ladder, and not applied by default.
    """
    from optima.enricher import _trim_source

    view = dict(function)
    view["llvm_ir"] = ""
    view["ast"] = ""
    view["cfg"] = ""
    view["calls"] = ""
    view["source_code"] = _trim_source(view.get("source_code") or "", 1500)
    return view


def _log_attempt(path: Path, *, function_id: str, attempt: int, category: str, reason: str,
                  input_tokens: Optional[int], output_tokens: Optional[int],
                  latency_s: Optional[float], finish_reason: Optional[str],
                  peak_allocated_gib: Optional[float], prompt_reductions: list[str],
                  raw_response: Optional[str]) -> None:
    record = {
        "function_id": function_id, "attempt": attempt, "category": category, "reason": reason,
        "input_tokens": input_tokens, "output_tokens": output_tokens, "latency_s": latency_s,
        "finish_reason": finish_reason, "peak_allocated_gib": peak_allocated_gib,
        "prompt_reductions": prompt_reductions, "raw_response": raw_response,
        "timestamp": _now_iso(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def enrich_one(handle: Any, function: dict[str, Any], gen: GenerationSettings,
                attempts_log_path: Path, max_new_tokens_override: Optional[int] = None) -> dict[str, Any]:
    """Enrich one function with the taxonomy A-H (module docstring in the
    architecture plan). Raises GenerationOOMError immediately on CUDA OOM
    (no retry, no swallowing); every other failure is recorded on the
    function record and the raw model output is preserved.

    ``max_new_tokens_override``, when given, replaces ``gen.max_new_tokens``
    for the actual generation call only -- ``gen`` itself (and therefore
    ``config_hash``/checkpoint identity) is untouched. This lets a smoke-test
    gate use a smaller, conservative generation budget than the full run
    without the gate being recorded under a different config_hash than the
    full run it is meant to unlock (``require_all_gates_passed`` compares
    config_hash across gates and the full run).
    """
    import torch

    from colab.colab_pipeline import _fallback, _parse_json_object, _valid_enrichment
    from optima.enricher import _evaluation

    from . import models

    category = "unknown"
    reason = "no_attempts_made"
    total_latency = 0.0
    total_output_tokens = 0
    last_input_tokens = 0
    last_finish_reason: Optional[str] = None
    last_raw = ""
    last_peak_gib: Optional[float] = None
    last_free_gib: Optional[float] = None
    repair_used = False
    prompt_reductions: list[str] = []
    generation_success_any = False
    attempts_made = 0
    parsed_enrichment: Optional[dict[str, Any]] = None

    for attempt_index in range(gen.retries + 1):
        attempts_made = attempt_index + 1
        try:
            messages, input_tokens, reductions = build_bounded_messages(
                function, handle.tokenizer, handle.spec, gen.prompt_variant
            )
        except PromptTooLongError as exc:
            category, reason = "prompt_too_long", str(exc)
            _log_attempt(attempts_log_path, function_id=function.get("id", ""), attempt=attempt_index,
                         category=category, reason=reason, input_tokens=None, output_tokens=None,
                         latency_s=None, finish_reason=None, peak_allocated_gib=None,
                         prompt_reductions=[], raw_response=None)
            break
        except Exception as exc:  # noqa: BLE001 - a deterministic build failure, not retried
            category, reason = "prompt_build_error", f"{type(exc).__name__}: {exc}"
            _log_attempt(attempts_log_path, function_id=function.get("id", ""), attempt=attempt_index,
                         category=category, reason=reason, input_tokens=None, output_tokens=None,
                         latency_s=None, finish_reason=None, peak_allocated_gib=None,
                         prompt_reductions=[], raw_response=None)
            break

        prompt_reductions = reductions
        max_new_tokens = max_new_tokens_override if max_new_tokens_override is not None else gen.max_new_tokens
        if attempt_index > 0 and category in REPAIRABLE_CATEGORIES:
            repair_note = (
                f"Your previous reply was not valid JSON matching the required "
                f"fields ({category}: {reason}). Return ONLY the JSON object, "
                f"with no markdown fences or explanation."
            )
            messages = [*messages, {"role": "assistant", "content": last_raw},
                        {"role": "user", "content": repair_note}]
            repair_used = True
            if category == "output_truncated":
                max_new_tokens = min(max_new_tokens * 2, 1536)

        try:
            result = models.generate_text(
                handle, messages, max_new_tokens=max_new_tokens, do_sample=gen.do_sample,
                temperature=gen.temperature, top_p=gen.top_p,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            _log_attempt(attempts_log_path, function_id=function.get("id", ""), attempt=attempt_index,
                         category="cuda_oom", reason=str(exc), input_tokens=input_tokens,
                         output_tokens=None, latency_s=None, finish_reason=None,
                         peak_allocated_gib=None, prompt_reductions=reductions, raw_response=None)
            # OOM is a resource issue, not a content-quality issue: give it one
            # dedicated recovery retry with the most-reduced prompt and a
            # halved output budget, independent of the JSON-repair retry loop
            # above. Only if THAT also OOMs do we decide, from how much VRAM
            # actually came back after cleanup, whether to fail just this one
            # function (GPU is healthy) or abort the whole run (it is not).
            retry_max_new_tokens = max(128, max_new_tokens // 2)
            retry_input_tokens = input_tokens
            try:
                retry_view = _fully_reduced_view(function)
                retry_messages, retry_input_tokens, retry_reductions = build_bounded_messages(
                    retry_view, handle.tokenizer, handle.spec, gen.prompt_variant
                )
                result = models.generate_text(
                    handle, retry_messages, max_new_tokens=retry_max_new_tokens,
                    do_sample=gen.do_sample, temperature=gen.temperature, top_p=gen.top_p,
                )
            except torch.cuda.OutOfMemoryError as retry_exc:
                torch.cuda.empty_cache()
                free_gib_after = min(
                    torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())
                ) / (1024 ** 3)
                _log_attempt(
                    attempts_log_path, function_id=function.get("id", ""), attempt=attempt_index,
                    category="cuda_oom",
                    reason=f"OOM persisted after reduced-budget retry (free_after_gib="
                           f"{free_gib_after:.2f}): {retry_exc}",
                    input_tokens=retry_input_tokens, output_tokens=None, latency_s=None,
                    finish_reason=None, peak_allocated_gib=None,
                    prompt_reductions=reductions + ["oom_retry_fully_reduced"], raw_response=None,
                )
                if free_gib_after < 1.0:
                    raise GenerationOOMError(
                        f"CUDA OOM persisted for function {function.get('id')} even after a "
                        f"reduced-budget retry, and only {free_gib_after:.2f} GiB is free "
                        f"afterward -- the GPU/model state looks unhealthy. Aborting this "
                        f"model's run rather than continuing on a possibly corrupted state."
                    ) from retry_exc
                # The GPU recovered real headroom after cleanup, so this looks
                # like one unusually expensive function, not a broken model --
                # fail only this function and let the caller move on.
                category, reason = (
                    "cuda_oom",
                    f"CUDA OOM persisted after a reduced-budget retry: {retry_exc}",
                )
                break
            else:
                # The reduced retry produced a real result; fall through to the
                # normal parse/validate logic below exactly as if the first
                # attempt had succeeded.
                input_tokens = retry_input_tokens
                reductions = reductions + ["oom_retry_fully_reduced"] + retry_reductions
                prompt_reductions = reductions
        except Exception as exc:  # noqa: BLE001 - isolated per-attempt, retried
            category, reason = "generation_error", f"{type(exc).__name__}: {exc}"
            _log_attempt(attempts_log_path, function_id=function.get("id", ""), attempt=attempt_index,
                         category=category, reason=reason, input_tokens=input_tokens,
                         output_tokens=None, latency_s=None, finish_reason=None,
                         peak_allocated_gib=None, prompt_reductions=reductions, raw_response=None)
            continue

        generation_success_any = True
        last_input_tokens = result.input_tokens
        last_finish_reason = result.finish_reason
        last_raw = result.text
        last_peak_gib = result.peak_allocated_gib
        last_free_gib = result.free_after_gib
        total_latency += result.latency_s
        total_output_tokens += result.output_tokens

        parsed, parse_reason = _parse_json_object(result.text)
        if parsed is None:
            category = "output_truncated" if result.finish_reason == "length" else "invalid_json"
            reason = parse_reason
            _log_attempt(attempts_log_path, function_id=function.get("id", ""), attempt=attempt_index,
                         category=category, reason=reason, input_tokens=result.input_tokens,
                         output_tokens=result.output_tokens, latency_s=result.latency_s,
                         finish_reason=result.finish_reason, peak_allocated_gib=result.peak_allocated_gib,
                         prompt_reductions=reductions, raw_response=result.text)
            continue

        valid, schema_reason = _valid_enrichment(parsed)
        if not valid:
            category = "output_truncated" if result.finish_reason == "length" else "schema_invalid"
            reason = schema_reason
            _log_attempt(attempts_log_path, function_id=function.get("id", ""), attempt=attempt_index,
                         category=category, reason=reason, input_tokens=result.input_tokens,
                         output_tokens=result.output_tokens, latency_s=result.latency_s,
                         finish_reason=result.finish_reason, peak_allocated_gib=result.peak_allocated_gib,
                         prompt_reductions=reductions, raw_response=result.text)
            continue

        category, reason = "completed", "valid"
        parsed_enrichment = parsed
        _log_attempt(attempts_log_path, function_id=function.get("id", ""), attempt=attempt_index,
                     category=category, reason=reason, input_tokens=result.input_tokens,
                     output_tokens=result.output_tokens, latency_s=result.latency_s,
                     finish_reason=result.finish_reason, peak_allocated_gib=result.peak_allocated_gib,
                     prompt_reductions=reductions, raw_response=result.text)
        break

    retry_count = max(0, attempts_made - 1)
    if category == "completed" and parsed_enrichment is not None:
        result_enrichment = {**parsed_enrichment, "model": handle.spec.model_id, "status": "completed"}
    else:
        result_enrichment = _fallback(reason, handle.spec.model_id, retry_count)

    total_tokens = (last_input_tokens or 0) + total_output_tokens if (last_input_tokens or total_output_tokens) else None
    result_enrichment["usage"] = {
        "prompt_tokens": last_input_tokens or None,
        "completion_tokens": total_output_tokens or None,
        "total_tokens": total_tokens,
    }

    metrics = {
        "request_success": category == "completed",
        "json_valid": category == "completed",
        "generation_success": generation_success_any,
        "retry_count": retry_count,
        "latency_seconds": total_latency,
        "generated_tokens": total_output_tokens,
    }
    cfg = (function.get("cfg") or {})
    llvm_cfg = ((function.get("llvm") or {}).get("cfg") or {})
    context = {
        "source_code": bool(function.get("source_code")),
        "ast": bool(function.get("ast")),
        "call_graph": bool(function.get("calls") or function.get("called_by")),
        "llvm": bool(function.get("llvm_ir")),
        "cfg": bool(cfg.get("nodes") or llvm_cfg.get("nodes")),
        "source_characters": len(function.get("source_code") or ""),
        "llvm_characters": len(function.get("llvm_ir") or ""),
        "cfg_nodes": len(cfg.get("nodes", [])),
        "cfg_edges": len(cfg.get("edges", [])),
        "call_count": len(function.get("calls") or []),
        "prompt_variant": gen.prompt_variant,
        "prompt_reductions": prompt_reductions,
        "input_tokens": last_input_tokens,
        "max_input_tokens": handle.spec.max_input_tokens,
        "max_new_tokens": max_new_tokens_override if max_new_tokens_override is not None else gen.max_new_tokens,
    }

    load_report = getattr(handle, "load_report", None) or {}
    result_enrichment["evaluation"] = _evaluation(function, result_enrichment, metrics, context)
    result_enrichment["evaluation"].update({
        "failure_category": None if category == "completed" else category,
        "failure_reason": None if category == "completed" else reason,
        "finish_reason": last_finish_reason,
        "raw_response": None if category == "completed" else (last_raw or None),
        "repair_used": repair_used,
        "peak_allocated_gib": last_peak_gib,
        "free_after_gib": last_free_gib,
        # GPU placement is constant across every function in one model's run
        # (it is decided once, at load time), but recorded per function here
        # too so per-function metrics rows are self-contained for comparison
        # tooling that reads function-level records in isolation.
        "gpu_placement": load_report.get("gpu_placement"),
        "gpu_count_used": load_report.get("gpu_count_used"),
        "used_cpu_offload": load_report.get("used_cpu_offload", False),
    })
    return result_enrichment


class Checkpoint:
    """Append-only, fsynced JSONL checkpoint. The source of truth for what
    has been enriched; the materialized enhanced_<slug>.json is derived from
    it and can always be rebuilt.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.header: Optional[dict[str, Any]] = None
        self._records: dict[str, dict[str, Any]] = {}

    def open_or_create(self, header: dict[str, Any]) -> None:
        if self.path.exists() and self.path.stat().st_size > 0:
            self._load()
            for key in ("base_sha256", "config_hash"):
                if key in header and self.header.get(key) != header.get(key):
                    raise ResumeMismatchError(
                        f"Checkpoint {self.path} was created with {key}="
                        f"{self.header.get(key)!r}, but the current run has "
                        f"{key}={header.get(key)!r}. Change the model slug "
                        f"suffix (e.g. add '-v2') or delete {self.path} if "
                        f"restarting intentionally."
                    )
            return
        self.header = {"type": "header", "schema": 1, **header}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.header, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _load(self) -> None:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.header = None
        self._records = {}
        for index, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    continue  # tolerate a torn last write (session interrupted mid-flush)
                raise CheckpointCorruptError(
                    f"{self.path}: malformed JSON on line {index + 1} (not the "
                    f"last line); this checkpoint cannot be trusted."
                )
            if obj.get("type") == "header":
                self.header = obj
            elif obj.get("type") == "function":
                self._records[obj["function_id"]] = obj  # last write wins
        if self.header is None:
            raise CheckpointCorruptError(f"{self.path}: no header record found.")

    def is_completed(self, function_id: str) -> bool:
        record = self._records.get(function_id)
        if record is None:
            return False
        enrichment = record.get("enrichment", {})
        return record.get("status") == "completed" and enrichment.get("evaluation", {}).get("json_valid") is True

    def get(self, function_id: str) -> Optional[dict[str, Any]]:
        return self._records.get(function_id)

    def all_ids(self) -> set[str]:
        return set(self._records)

    def recent_records(self, count: int) -> list[dict[str, Any]]:
        return list(self._records.values())[-count:]

    def append(self, function_id: str, enrichment: dict[str, Any]) -> None:
        prior = self._records.get(function_id)
        round_num = (prior.get("round", 0) + 1) if prior else 1
        record = {
            "type": "function", "function_id": function_id,
            "status": enrichment.get("status"), "enrichment": enrichment,
            "round": round_num, "written_at": _now_iso(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._records[function_id] = record


def materialize(ctx: Any, snap: Any, spec: Any, gen: GenerationSettings, ckpt: Checkpoint,
                 elapsed: float, partial: bool, load_report: Optional[dict[str, Any]] = None) -> Path:
    """Rebuild enhanced_<slug>.json from base.json + the checkpoint. Always
    safe to call (even mid-run): a torn enhanced_*.json can never lose work
    because the checkpoint, not this file, is authoritative.

    ``load_report`` is ``ModelHandle.load_report`` (models.py) -- its
    ``resolved_commit_hash``/``gpu_placement``/``gpu_count_used``/
    ``used_cpu_offload`` fields are threaded into ``enrichment_metadata``
    below so the multi-GPU placement a model actually used is visible in the
    comparison table (§20), not just its success rate.
    """
    load_report = load_report or {}
    from colab.colab_pipeline import flatten_functions, load_json, save_json
    from optima.enricher import _dataset_metrics

    base = load_json(snap.path)
    result = copy.deepcopy(base)
    functions = flatten_functions(result)

    enriched_count = 0
    failed_count = 0
    failure_categories: dict[str, int] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_inference_seconds = 0.0
    for fn in functions:
        record = ckpt.get(fn["id"])
        if record is None:
            continue
        enrichment = record["enrichment"]
        fn["enrichment"] = enrichment
        evaluation = enrichment.get("evaluation", {})
        usage = enrichment.get("usage", {})
        total_input_tokens += usage.get("prompt_tokens") or 0
        total_output_tokens += usage.get("completion_tokens") or 0
        total_inference_seconds += evaluation.get("latency_seconds") or 0
        if enrichment.get("status") == "completed":
            enriched_count += 1
        else:
            failed_count += 1
            category = evaluation.get("failure_category") or "unknown"
            failure_categories[category] = failure_categories.get(category, 0) + 1

    result["enrichment_metadata"] = {
        "provider": "Hugging Face Transformers", "model": spec.model_id,
        "model_slug": spec.slug, "resolved_commit_hash": load_report.get("resolved_commit_hash"),
        "quantization": spec.quantization, "total_functions": len(functions),
        "enriched_functions": enriched_count, "failed_functions": failed_count,
        "partial": partial, "started_at": (ckpt.header or {}).get("created_at"),
        "completed_at": _now_iso(),
        "gpu_placement": load_report.get("gpu_placement"),
        "gpu_count_used": load_report.get("gpu_count_used"),
        "used_cpu_offload": load_report.get("used_cpu_offload", False),
    }
    result["experiment"] = {
        "model": spec.model_id, "temperature": gen.temperature if gen.do_sample else None,
        "do_sample": gen.do_sample, "batch_size": 1, "prompt_version": "v2",
        "prompt_variant": gen.prompt_variant, "schema_version": "v1",
        "base_json_url": snap.url, "base_sha256": snap.sha256,
        "config_hash": config_hash(spec, gen), "optima_commit": getattr(ctx, "optima_commit", None),
        "max_new_tokens": gen.max_new_tokens, "retries": gen.retries,
        "max_input_tokens": spec.max_input_tokens,
    }
    metrics = _dataset_metrics(functions, elapsed)
    metrics["failure_categories"] = failure_categories
    metrics["total_input_tokens"] = total_input_tokens
    metrics["total_output_tokens"] = total_output_tokens
    metrics["total_inference_seconds"] = total_inference_seconds
    result["enrichment_metrics"] = metrics

    out_path = ctx.enriched_dir / f"enhanced_{spec.slug}.json"
    save_json(result, out_path)
    return out_path


# --------------------------------------------------------------------------
# Smoke-test gates
# --------------------------------------------------------------------------

def _select_gate3_function(functions: list[dict[str, Any]],
                            smoke_function_ids: Optional[list[str]]) -> dict[str, Any]:
    if smoke_function_ids:
        by_id = {f["id"]: f for f in functions}
        if smoke_function_ids[0] not in by_id:
            raise KeyError(f"SMOKE_FUNCTION_IDS[0]={smoke_function_ids[0]!r} not found in base.json.")
        return by_id[smoke_function_ids[0]]
    candidates = [f for f in functions
                  if f.get("analysis_status") == "success" and len(f.get("source_code") or "") >= 200]
    if not candidates:
        candidates = [f for f in functions if f.get("source_code")]
    if not candidates:
        raise ValueError("No function with usable source_code found for GATE 3.")
    candidates = sorted(candidates, key=lambda f: (len(f.get("source_code") or ""), f["id"]))
    return candidates[len(candidates) // 2]


def _select_gate4_functions(functions: list[dict[str, Any]], tokenizer: Any, spec: Any,
                             gen: GenerationSettings, gate3_function: dict[str, Any],
                             smoke_function_ids: Optional[list[str]]) -> list[dict[str, Any]]:
    if smoke_function_ids and len(smoke_function_ids) >= 3:
        by_id = {f["id"]: f for f in functions}
        missing = [fid for fid in smoke_function_ids[:3] if fid not in by_id]
        if missing:
            raise KeyError(f"SMOKE_FUNCTION_IDS not found in base.json: {missing}")
        return [by_id[fid] for fid in smoke_function_ids[:3]]

    selected = [gate3_function]
    seen_ids = {gate3_function["id"]}

    other_status = sorted(
        (f for f in functions
         if f.get("analysis_status") in ("source_only", "llvm_function_not_found") and f["id"] not in seen_ids),
        key=lambda f: f["id"],
    )
    if other_status:
        selected.append(other_status[0])
        seen_ids.add(other_status[0]["id"])

    largest, largest_tokens = None, -1
    for fn in functions:
        if fn["id"] in seen_ids:
            continue
        try:
            _messages, tokens, _reductions = build_bounded_messages(fn, tokenizer, spec, gen.prompt_variant)
        except Exception:  # noqa: BLE001 - selection helper only, real failures surface during the gate
            continue
        if tokens > largest_tokens:
            largest, largest_tokens = fn, tokens
    if largest is not None:
        selected.append(largest)
        seen_ids.add(largest["id"])

    if len(selected) < 3:
        for fn in sorted(functions, key=lambda f: f["id"]):
            if fn["id"] not in seen_ids:
                selected.append(fn)
                seen_ids.add(fn["id"])
            if len(selected) >= 3:
                break
    return selected[:3]


def record_gate(ctx: Any, spec: Any, gen: GenerationSettings, name: str, passed: bool,
                 details: dict[str, Any]) -> dict[str, Any]:
    gate_dir = ctx.smoke_dir / spec.slug
    gate_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name, "passed": passed, "details": details,
        "spec": {"slug": spec.slug, "model_id": spec.model_id, "quantization": spec.quantization},
        "config_hash": config_hash(spec, gen), "base_sha256": ctx.base.sha256,
        "session_id": ctx.session_id, "timestamp": _now_iso(),
    }
    (gate_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    all_path = gate_dir / "gates.json"
    all_gates = json.loads(all_path.read_text(encoding="utf-8")) if all_path.exists() else {}
    all_gates[name] = payload
    all_path.write_text(json.dumps(all_gates, indent=2, default=str) + "\n", encoding="utf-8")

    if not passed:
        raise GateFailedError(name, json.dumps(details, default=str)[:800], details)
    print(f"GATE PASSED: {name} for {spec.slug}")
    return payload


def require_all_gates_passed(ctx: Any, spec: Any, gen: GenerationSettings) -> None:
    gate_dir = ctx.smoke_dir / spec.slug
    all_path = gate_dir / "gates.json"
    if not all_path.exists():
        raise GateFailedError("gates", f"No gate records found for {spec.slug}; run gates 1-4 first.")
    all_gates = json.loads(all_path.read_text(encoding="utf-8"))
    expected_hash = config_hash(spec, gen)
    for name in REQUIRED_GATES:
        record = all_gates.get(name)
        if record is None or not record.get("passed"):
            raise GateFailedError("gates", f"{name} has not passed for {spec.slug}.")
        if record.get("config_hash") != expected_hash:
            raise GateFailedError(
                "gates", f"{name} was recorded with config_hash={record.get('config_hash')} "
                f"but the current config_hash={expected_hash}; re-run the gates."
            )
        if record.get("base_sha256") != ctx.base.sha256:
            raise GateFailedError("gates", f"{name} was recorded for a different base.json snapshot.")
        if name == "gate1_load" and record.get("session_id") != ctx.session_id:
            raise GateFailedError(
                "gates", "gate1_load was recorded in a different session; re-run GATE 1 in "
                "this session before starting a full run."
            )
    print(f"All gates passed for {spec.slug} (config_hash={expected_hash}).")


def _require_handle(handle: Any, gate_name: str) -> None:
    """Fail loud and clear when a gate is called without a loaded model,
    instead of letting a later ``handle.something`` raise AttributeError
    (and, worse, letting an except-block's own ``handle.spec`` access mask
    that AttributeError with a second, more confusing one). This is a
    defense-in-depth check: the notebook itself also guards HANDLE before
    calling into each gate, but library code must not depend on that.
    """
    if handle is None:
        raise GateFailedError(
            gate_name,
            f"{gate_name} requires a successfully loaded model HANDLE, but "
            f"HANDLE is None. GATE 1 (models.load_model_safe) must complete "
            f"successfully first.",
        )


def gate1_load(ctx: Any, handle: Any, gen: GenerationSettings) -> dict[str, Any]:
    _require_handle(handle, "gate1_load")
    return record_gate(ctx, handle.spec, gen, "gate1_load", True, handle.load_report)


def gate2_trivial(ctx: Any, handle: Any, gen: GenerationSettings) -> dict[str, Any]:
    from . import models

    _require_handle(handle, "gate2_trivial")
    messages = [{"role": "user", "content": "In one short sentence, what is a binary search tree?"}]
    try:
        result = models.generate_text(handle, messages, max_new_tokens=64, do_sample=False)
    except Exception as exc:  # noqa: BLE001 - a smoke-test gate reports, then stops the notebook
        return record_gate(ctx, handle.spec, gen, "gate2_trivial", False,
                            {"error": f"{type(exc).__name__}: {exc}"})
    passed = bool(result.text.strip()) and any(ch.isalpha() for ch in result.text)
    details = {
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "latency_s": result.latency_s,
        "tokens_per_second": round(result.output_tokens / result.latency_s, 2) if result.latency_s else None,
        "peak_allocated_gib": result.peak_allocated_gib, "free_after_gib": result.free_after_gib,
        "finish_reason": result.finish_reason, "text": result.text,
    }
    return record_gate(ctx, handle.spec, gen, "gate2_trivial", passed, details)


def gate3_one_function(ctx: Any, handle: Any, gen: GenerationSettings, snap: Any,
                        smoke_function_ids: Optional[list[str]] = None,
                        max_new_tokens_override: Optional[int] = None) -> dict[str, Any]:
    from colab.colab_pipeline import _valid_enrichment, flatten_functions, load_json

    _require_handle(handle, "gate3_one_function")
    base = load_json(snap.path)
    functions = flatten_functions(base)
    fn = _select_gate3_function(functions, smoke_function_ids)

    ckpt = Checkpoint(ctx.smoke_dir / handle.spec.slug / "gate3.checkpoint.jsonl")
    ckpt.open_or_create({"base_sha256": ctx.base.sha256, "config_hash": config_hash(handle.spec, gen),
                         "model_slug": handle.spec.slug, "created_at": _now_iso()})

    attempts_log = ctx.logs_dir / handle.spec.slug / "attempts.jsonl"
    enrichment = enrich_one(handle, fn, gen, attempts_log, max_new_tokens_override=max_new_tokens_override)
    ckpt.append(fn["id"], enrichment)

    evaluation = enrichment.get("evaluation", {})
    schema_valid, _reason = _valid_enrichment(enrichment)
    passed = (
        enrichment.get("status") == "completed"
        and evaluation.get("json_valid") is True
        and schema_valid
        and (evaluation.get("input_tokens") or 0) > 0
        and (evaluation.get("output_tokens") or 0) > 0
        and evaluation.get("finish_reason") == "eos"
    )
    details = {
        "function_id": fn["id"], "function_name": fn.get("name"),
        "analysis_status": fn.get("analysis_status"),
        "input_tokens": evaluation.get("input_tokens"), "output_tokens": evaluation.get("output_tokens"),
        "latency_seconds": evaluation.get("latency_seconds"),
        "prompt_reductions": evaluation.get("context_metrics", {}).get("prompt_reductions"),
        "raw_response": evaluation.get("raw_response"),
        "parsed_enrichment": {k: v for k, v in enrichment.items() if k != "evaluation"},
        "json_valid": evaluation.get("json_valid"), "request_success": evaluation.get("request_success"),
        "generation_success": evaluation.get("generation_success"), "retry_count": evaluation.get("retry_count"),
        "failure_category": evaluation.get("failure_category"), "failure_reason": evaluation.get("failure_reason"),
        "finish_reason": evaluation.get("finish_reason"),
        "peak_allocated_gib": evaluation.get("peak_allocated_gib"), "free_after_gib": evaluation.get("free_after_gib"),
    }
    print(json.dumps(details, indent=2, default=str)[:4000])
    return record_gate(ctx, handle.spec, gen, "gate3_one_function", passed, details)


def gate4_three_functions(ctx: Any, handle: Any, gen: GenerationSettings, snap: Any,
                           smoke_function_ids: Optional[list[str]] = None,
                           max_new_tokens_override: Optional[int] = None) -> dict[str, Any]:
    """``max_new_tokens_override`` bounds the actual generation budget for
    this smoke test (e.g. a conservative 256 instead of the full run's 768)
    without changing ``gen``/``config_hash`` -- see ``enrich_one`` docstring.
    """
    from optima.enricher import _dataset_metrics
    from colab.colab_pipeline import flatten_functions, load_json

    _require_handle(handle, "gate4_three_functions")
    base = load_json(snap.path)
    functions = flatten_functions(base)
    gate3_fn = _select_gate3_function(functions, smoke_function_ids[:1] if smoke_function_ids else None)
    selected = _select_gate4_functions(functions, handle.tokenizer, handle.spec, gen, gate3_fn, smoke_function_ids)

    ckpt = Checkpoint(ctx.smoke_dir / handle.spec.slug / "gate4.checkpoint.jsonl")
    ckpt.open_or_create({"base_sha256": ctx.base.sha256, "config_hash": config_hash(handle.spec, gen),
                         "model_slug": handle.spec.slug, "created_at": _now_iso()})

    attempts_log = ctx.logs_dir / handle.spec.slug / "attempts.jsonl"
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for fn in selected:
        enrichment = enrich_one(handle, fn, gen, attempts_log, max_new_tokens_override=max_new_tokens_override)
        ckpt.append(fn["id"], enrichment)
        records.append((fn, enrichment))

    def _ok(e: dict[str, Any]) -> bool:
        return e.get("status") == "completed" and e.get("evaluation", {}).get("json_valid") is True

    completed = [(fn, e) for fn, e in records if _ok(e)]
    all_three_completed = len(completed) == len(records) == 3

    purposes = [e.get("purpose") for _fn, e in completed]
    purposes_not_all_same = len(set(purposes)) > 1 if len(purposes) > 1 else bool(purposes)
    insufficient_count = sum(1 for p in purposes if p == "Insufficient implementation context.")

    try:
        _dataset_metrics([{"enrichment": e} for _fn, e in records], 1.0)
        dataset_metrics_ok = True
        dataset_metrics_error = None
    except Exception as exc:  # noqa: BLE001 - captured as a check result, not raised
        dataset_metrics_ok = False
        dataset_metrics_error = f"{type(exc).__name__}: {exc}"

    # GATE 4 tests that enrichment actually works end to end: three real
    # functions, processed sequentially, each producing a distinct,
    # schema-valid, correctly-attributed result. It intentionally does not
    # gate on any GPU/headroom metric -- a theoretical "not enough spare
    # VRAM" estimate is not evidence that enrichment failed, and three
    # functions that actually completed must not be reported as a gate
    # failure over it. (GPU telemetry for the model overall is still
    # available from GATE 1's load report and per-function evaluation
    # records; it just is not a GATE 4 pass/fail criterion.)
    checks = {
        "all_three_completed": all_three_completed,
        "purposes_not_all_same": purposes_not_all_same,
        "insufficient_count_below_two": insufficient_count < 2,
        "model_field_correct": all(e.get("model") == handle.spec.model_id for _fn, e in records),
        "metadata_present": all(
            isinstance(e.get("evaluation", {}).get("retry_count"), int)
            and isinstance(e.get("evaluation", {}).get("request_success"), bool)
            and isinstance(e.get("evaluation", {}).get("json_valid"), bool)
            and (e.get("evaluation", {}).get("input_tokens") or 0) > 0
            and (e.get("evaluation", {}).get("output_tokens") or 0) > 0
            for _fn, e in records
        ),
        "dataset_metrics_roundtrip_ok": dataset_metrics_ok,
    }
    passed = all(checks.values())
    status = "PASSED" if passed else "FAILED"

    details = {
        "status": status, "checks": checks, "dataset_metrics_error": dataset_metrics_error,
        "functions": [
            {"function_id": fn["id"], "status": e.get("status"), "purpose": e.get("purpose"),
             "failure_category": e.get("evaluation", {}).get("failure_category"),
             "input_tokens": e.get("evaluation", {}).get("input_tokens"),
             # Informational only (not a check, not a warning, not part of
             # `passed`): real GPU telemetry per function, for visibility.
             "free_after_gib": e.get("evaluation", {}).get("free_after_gib")}
            for fn, e in records
        ],
    }
    print(f"GATE 4 status: {status}")
    print(json.dumps(details, indent=2, default=str)[:4000])
    return record_gate(ctx, handle.spec, gen, "gate4_three_functions", passed, details)


# --------------------------------------------------------------------------
# Full enrichment
# --------------------------------------------------------------------------

def run_full_enrichment(ctx: Any, handle: Any, gen: GenerationSettings, snap: Any,
                         max_consecutive_failures: int = 5, materialize_every: int = 10,
                         min_success_rate: float = 0.95) -> dict[str, Any]:
    _require_handle(handle, "run_full_enrichment")
    require_all_gates_passed(ctx, handle.spec, gen)

    from colab.colab_pipeline import flatten_functions, load_json
    from . import environment

    slug = handle.spec.slug
    ckpt = Checkpoint(ctx.enriched_dir / f"{slug}.checkpoint.jsonl")
    ckpt.open_or_create({
        "base_sha256": ctx.base.sha256, "config_hash": config_hash(handle.spec, gen),
        "model_id": handle.spec.model_id, "model_slug": slug,
        "prompt_variant": gen.prompt_variant, "created_at": _now_iso(),
    })

    base = load_json(snap.path)
    functions = flatten_functions(base)
    function_ids = {fn["id"] for fn in functions}
    unknown_ids = ckpt.all_ids() - function_ids
    if unknown_ids:
        raise ResumeMismatchError(
            f"Checkpoint has {len(unknown_ids)} function ID(s) not present in "
            f"the current base.json: {sorted(unknown_ids)[:5]}"
        )

    todo = [fn for fn in functions if not ckpt.is_completed(fn["id"])]
    already_completed = len(functions) - len(todo)
    print(f"FULL ENRICHMENT [{slug}]: {len(functions)} total, "
          f"{already_completed} already completed, {len(todo)} to do")

    attempts_log = ctx.logs_dir / slug / "attempts.jsonl"
    consecutive_failures = 0
    partial = True
    start = time.perf_counter()
    out_path: Optional[Path] = None
    try:
        from tqdm.auto import tqdm
        for index, fn in enumerate(tqdm(todo, desc=f"Enriching with {slug}")):
            enrichment = enrich_one(handle, fn, gen, attempts_log)
            ckpt.append(fn["id"], enrichment)
            if enrichment.get("status") == "completed":
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    recent = [
                        {"failure_category": r.get("enrichment", {}).get("evaluation", {}).get("failure_category"),
                         "failure_reason": r.get("enrichment", {}).get("evaluation", {}).get("failure_reason")}
                        for r in ckpt.recent_records(5)
                    ]
                    raise SystemicEnrichmentFailure(
                        f"{consecutive_failures} consecutive enrichment failures for "
                        f"{slug}; aborting rather than burning the rest of the GPU "
                        f"budget. Recent failures: {recent}"
                    )
            if (index + 1) % materialize_every == 0:
                materialize(ctx, snap, handle.spec, gen, ckpt, time.perf_counter() - start, partial=True,
                           load_report=handle.load_report)
            if (index + 1) % 10 == 0:
                environment.gpu_report(f"{slug}: after {index + 1}/{len(todo)}", ctx.gpu_log_path)
        partial = False
    finally:
        elapsed = time.perf_counter() - start
        out_path = materialize(ctx, snap, handle.spec, gen, ckpt, elapsed, partial=partial,
                              load_report=handle.load_report)
        if partial:
            ctx.set_model_status(slug, "enrichment_incomplete", {"enriched_json": str(out_path)})

    from colab.colab_pipeline import load_json as _load_json
    materialized = _load_json(out_path)
    metrics = materialized["enrichment_metrics"]
    success_rate = metrics.get("success_rate", 0.0)
    categories = metrics.get("failure_categories", {})
    print(f"FULL ENRICHMENT [{slug}] done: success_rate={success_rate:.3f}, "
          f"failure_categories={categories}")

    if success_rate >= min_success_rate:
        ctx.set_model_status(slug, "enrichment_passed",
                             {"success_rate": success_rate, "enriched_json": str(out_path)})
    else:
        ctx.set_model_status(slug, "enrichment_failed",
                             {"success_rate": success_rate, "enriched_json": str(out_path)})
        raise EnrichmentQualityError(
            f"{slug} finished with success_rate={success_rate:.3f} < "
            f"{min_success_rate}. failure_categories={categories}. "
            f"See {out_path} and {attempts_log}."
        )

    return {
        "enriched_json": out_path, "metrics": metrics, "partial": partial,
        "resumed_functions": already_completed, "processed_this_run": len(todo),
    }


def run_model_queue(ctx: Any, model_queue: list[str], gen_settings_factory: Any, bnb_status: Any,
                     num_gpus: Optional[int] = None, stop_on_failure: bool = True,
                     allow_infeasible: bool = False, delete_cache_after_unload: bool = True,
                     max_consecutive_failures: int = 5, materialize_every: int = 10,
                     min_success_rate: float = 0.95) -> dict[str, Any]:
    """Run check_fit -> load -> gates 1-4 -> full enrichment -> unload for
    each queued model slug, using the exact same functions the single-model
    notebook cells use. ``num_gpus`` defaults to ``torch.cuda.device_count()``
    when not given -- resolve_spec/check_fit both re-derive it internally
    too, so a stale caller-supplied value can only affect the early,
    advisory single_gpu_tier refusal in resolve_spec, never the real fit
    decision.
    """
    from . import environment, models

    if num_gpus is None:
        import torch
        num_gpus = torch.cuda.device_count()

    results: dict[str, Any] = {}
    for slug in model_queue:
        handle = None
        try:
            spec = models.resolve_spec(slug, num_gpus=num_gpus, allow_infeasible=allow_infeasible)
            gen = gen_settings_factory(spec)
            models.check_fit(spec, num_gpus=num_gpus)
            environment.gpu_report(f"{slug}: before load", ctx.gpu_log_path)
            handle = models.load_model_safe(spec, bnb_status)
            environment.gpu_report(f"{slug}: after load", ctx.gpu_log_path)
            gate1_load(ctx, handle, gen)
            gate2_trivial(ctx, handle, gen)
            gate3_one_function(ctx, handle, gen, ctx.base)
            gate4_three_functions(ctx, handle, gen, ctx.base)
            full = run_full_enrichment(ctx, handle, gen, ctx.base, max_consecutive_failures,
                                       materialize_every, min_success_rate)
            results[slug] = {"status": "enrichment_passed", **full}
        except Exception as exc:  # noqa: BLE001 - per-model isolation is the point of this loop
            results[slug] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            ctx.set_model_status(slug, "failed", {"error": str(exc)})
            if stop_on_failure:
                raise
        finally:
            if handle is not None:
                models.unload_model(handle, delete_cache=delete_cache_after_unload)
    return results
