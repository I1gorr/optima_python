"""
Optima enricher: uses LM Studio to enrich base.json with semantic information.
"""
import json
import os
import sys
import time
import datetime
import re
import statistics
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default LM Studio configuration
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_RESERVED_OUTPUT_TOKENS = 2048
CONTEXT_SAFETY_MARGIN = 128
ENRICHMENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "purpose": {"type": "string"},
        "behavior": {"type": "string"},
        "summary": {"type": "string"},
        "inputs": {"type": "array", "items": {"type": "string"}},
        "outputs": {"type": "array", "items": {"type": "string"}},
        "side_effects": {"type": "array", "items": {"type": "string"}},
        "dependencies": {"type": "array", "items": {"type": "string"}},
        "concepts": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "algorithm": {"type": "string"},
        "complexity": {
            "type": "object",
            "properties": {"time": {"type": "string"}, "space": {"type": "string"}},
            "required": ["time", "space"],
            "additionalProperties": False,
        },
    },
    "required": [
        "purpose", "behavior", "summary", "inputs", "outputs", "side_effects",
        "dependencies", "concepts", "keywords", "algorithm", "complexity",
    ],
    "additionalProperties": False,
}


def _estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens without adding a tokenizer dependency."""
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except (ImportError, Exception):
        # Source and IR contain many symbols that tokenize more densely than prose.
        return math.ceil(len(text) / 3)


def _context_size_from_record(record: Dict[str, Any]) -> Optional[int]:
    preferred_keys = ("context_length", "n_ctx", "context_window")
    fallback_keys = ("max_context_length", "max_context_tokens", "max_position_embeddings")
    queue = [record]
    fallback = None
    while queue:
        current = queue.pop()
        if isinstance(current, dict):
            for key in preferred_keys:
                value = current.get(key)
                if isinstance(value, int) and value > 0:
                    return value
            for key in fallback_keys:
                value = current.get(key)
                if isinstance(value, int) and value > 0:
                    fallback = fallback or value
            queue.extend(value for value in current.values() if isinstance(value, (dict, list)))
        elif isinstance(current, list):
            queue.extend(current)
    return fallback


def _compact_calls(calls: Any) -> str:
    names = []
    for call in calls if isinstance(calls, list) else []:
        if isinstance(call, dict):
            name = call.get("qualified_name") or call.get("name")
        else:
            name = str(call)
        if name and name not in names:
            names.append(name)
    return "\n".join(names) or "none"


def _compact_cfg(cfg: Any) -> str:
    if not isinstance(cfg, dict):
        return "unavailable"
    nodes = cfg.get("nodes", [])
    edges = cfg.get("edges", [])
    lines = [f"Blocks: {len(nodes)}", f"Edges: {len(edges)}"]
    lines.extend(
        f"{edge.get('from', '?')} -> {edge.get('to', '?')}"
        for edge in edges
        if isinstance(edge, dict)
    )
    return "\n".join(lines)


def _compact_ast(ast: Any, limit: int = 1200) -> str:
    if not ast:
        return "unavailable"
    if isinstance(ast, dict):
        nodes = []
        def visit(node: Any, depth: int = 0) -> None:
            if not isinstance(node, dict) or len(nodes) >= 80:
                return
            kind = node.get("kind", "")
            spelling = node.get("spelling") or node.get("displayname") or ""
            if kind or spelling:
                nodes.append(f"{'  ' * min(depth, 3)}{kind}: {spelling}".strip())
            for child in node.get("children", []):
                visit(child, depth + 1)
        visit(ast)
        return "\n".join(nodes)[:limit] or "unavailable"
    return str(ast)[:limit]


def _compact_llvm(llvm: Any, limit: int = 5000) -> str:
    if not llvm:
        return "unavailable"
    lines = str(llvm).splitlines()
    important = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith(("define ", "}", "br ", "switch ", "ret ", "call ",
                                  "load ", "store ", "atomicrmw ", "cmpxchg ",
                                  "add ", "sub ", "mul ", "icmp ", "fcmp "))
            or (stripped.endswith(":") and not stripped.startswith(";"))
        ):
            important.append(line)
    compact = "\n".join(important)
    return (compact or str(llvm))[:limit] or "unavailable"


def _trim_source(source: str, limit: int) -> str:
    if len(source) <= limit:
        return source
    if limit < 80:
        return source[:limit]
    head = int(limit * 0.75)
    return source[:head] + "\n... [source reduced] ...\n" + source[-(limit - head - 25):]


def _response_preview(content: str, limit: int = 500) -> str:
    """Return a bounded, single-line response preview for diagnostics."""
    preview = " ".join(content.split())
    return preview if len(preview) <= limit else preview[:limit] + "... [truncated]"


def _parse_json_object(content: str) -> tuple[Optional[Dict[str, Any]], str]:
    """Recover a JSON object from a raw LLM response as permissively as
    possible, in a fixed order, so a model is never marked invalid_json for
    formatting alone:

    1. Parse the raw response directly as JSON.
    2. Extract a Markdown-fenced ```json``` block (anywhere in the text, not
       just when the fence is the entire response).
    3. Extract any other fenced ``` block (unlabeled or a different tag).
    4. Scan the text for every '{' (or '[') and try decoding a JSON value
       starting there, in order, until one succeeds -- this recovers JSON
       that is preceded/followed by prose without relying on fences at all.
    5. Only report "invalid_json" once none of the above recovers anything.

    A recovered value that parses but is not a JSON object (e.g. a bare
    array) is reported as "json_root_not_object" rather than treated as a
    successful parse -- the enrichment schema is always an object.
    """
    stripped = content.strip()
    if not stripped:
        return None, "empty_response"

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, dict):
            return parsed, "json"
        return None, "json_root_not_object"

    fence_pattern = re.compile(r"```(\w*)[ \t]*\r?\n?(.*?)```", re.DOTALL)
    fence_matches = list(fence_pattern.finditer(stripped))
    ordered_bodies = (
        [m.group(2) for m in fence_matches if m.group(1).lower() == "json"]
        + [m.group(2) for m in fence_matches if m.group(1).lower() != "json"]
    )
    saw_non_object = False
    for body in ordered_bodies:
        try:
            parsed = json.loads(body.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, "markdown_wrapped_json"
        saw_non_object = True

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{\[]", stripped):
        start = match.start()
        try:
            candidate, _end = decoder.raw_decode(stripped, start)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate, "surrounding_text"
        saw_non_object = True

    return None, "json_root_not_object" if saw_non_object else "invalid_json"


class LMStudioClient:
    def __init__(
        self, base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY,
        model: Optional[str] = None, timeout: int = 120,
        context_size: Optional[int] = None, max_input_tokens: Optional[int] = None,
        reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
        debug: bool = False,
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.model = model
        self.model_id = None
        self.context_size = context_size or DEFAULT_CONTEXT_SIZE
        self.max_input_tokens = max_input_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.debug = debug
        if not self.model:
            self._discover_model()

    def _discover_model(self):
        """Discover the first available model from LM Studio."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.load(response)
                models = data.get("data", [])
                if not models:
                    raise RuntimeError("No models available in LM Studio")
                # Use the first model
                self.model = models[0]["id"]
                self.model_id = models[0].get("id")
                logger.info(f"Discovered model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to discover LM Studio model: {e}")
            raise

    def check_connection(self) -> bool:
        """Check if LM Studio server is reachable and the model exists."""
        try:
            # Check models endpoint
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.load(response)
                model_records = data.get("data", [])
                models = {m["id"] for m in model_records}
                if self.model not in models:
                    logger.error(f"Model '{self.model}' not found. Available models: {list(models)}")
                    return False
                self.model_id = next(
                    (m.get("id") for m in model_records if m.get("id") == self.model),
                    None
                )
            return True
        except Exception as e:
            logger.error(f"LM Studio server unavailable at {self.base_url}: {e}")
            return False

    def _get_model_records(self) -> List[Dict[str, Any]]:
        req = urllib.request.Request(f"{self.base_url}/models")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.load(response).get("data", [])

    def _get_native_model_records(self) -> Optional[List[Dict[str, Any]]]:
        """Return LM Studio runtime records, including loaded_instances."""
        api_root = self.base_url.rsplit("/v1", 1)[0]
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"{api_root}/api/v1/models"),
                timeout=self.timeout,
            ) as response:
                return json.load(response).get("models", [])
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
            return None

    def prepare_model(self, timeout_seconds: int = 300, poll_seconds: int = 2) -> Dict[str, Any]:
        """Load and verify the selected model once before processing functions."""
        started = time.time()
        records = self._get_model_records()
        native_records = self._get_native_model_records()
        selected = next((record for record in records if record.get("id") == self.model), None)
        if selected is None:
            raise RuntimeError(f"Requested model '{self.model}' is not available in LM Studio.")
        native_selected = next(
            (record for record in native_records or []
             if record.get("key") == self.model), None
        )
        already_loaded = (
            bool(native_selected.get("loaded_instances"))
            if native_selected is not None
            else selected.get("loaded", selected.get("state", "loaded") == "loaded")
        )
        if not already_loaded:
            api_root = self.base_url.rsplit("/v1", 1)[0]
            payload = json.dumps({"model": self.model}).encode("utf-8")
            request = urllib.request.Request(
                f"{api_root}/api/v1/models/load",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout):
                pass
            print("Loading model...")
        else:
            print("Model already loaded.")
        while time.time() - started <= timeout_seconds:
            selected = next((record for record in self._get_model_records()
                             if record.get("id") == self.model), None)
            native_records = self._get_native_model_records()
            native_selected = next(
                (record for record in native_records or []
                 if record.get("key") == self.model), None
            )
            if selected:
                discovered_context = (
                    _context_size_from_record(native_selected or {})
                    or _context_size_from_record(selected)
                )
                if self.max_input_tokens is None and self.context_size == DEFAULT_CONTEXT_SIZE and discovered_context:
                    self.context_size = discovered_context
                state = selected.get("state", "loaded")
                loaded = (
                    bool(native_selected.get("loaded_instances"))
                    if native_selected is not None
                    else selected.get("loaded", state == "loaded")
                )
                if loaded and state not in {"loading", "unloading", "error"}:
                    waited = 0.0 if already_loaded else time.time() - started
                    print(f"Requested model:\n{self.model}\n\nLoaded model:\n{selected.get('id')}\n\nStatus:\nREADY")
                    return {
                        "requested_model": self.model,
                        "loaded_model": selected.get("id"),
                        "load_wait_seconds": waited,
                        "load_verified": True,
                        "context_size": self.context_size,
                    }
            print(f"Waiting for model... {int(time.time() - started)}s")
            time.sleep(poll_seconds)
        raise TimeoutError(f"Timed out waiting for model '{self.model}'.")

    def check_connection(self) -> bool:
        """Check if LM Studio is reachable and exposes the requested model."""
        try:
            records = self._get_model_records()
            if self.model not in {record.get("id") for record in records}:
                logger.error("Model '%s' is not available.", self.model)
                return False
            self.model_id = self.model
            return True
        except Exception as error:
            logger.error("LM Studio unavailable at %s: %s", self.base_url, error)
            return False

    def _prompt_for_stage(self, function_data: Dict[str, Any], stage: str) -> tuple[str, str, Dict[str, Any]]:
        """Build a priority-ordered prompt without cutting the final prompt blindly."""
        system_prompt = ("You are a software code-analysis assistant.\n"
                         "Analyze one function using its actual source and compiler evidence.\n"
                         "Produce concise, factual semantic descriptions. The source is primary.\n"
                         "Never invent behaviour, relationships, or compiler facts.")
        source_location = function_data.get("source_location", function_data.get("source", {}))
        source = function_data.get("source_code", "")
        calls = _compact_calls(function_data.get("calls", []))
        called_by = _compact_calls(function_data.get("called_by", []))
        cfg = _compact_cfg(function_data.get("cfg", {}))
        llvm = _compact_llvm(function_data.get("llvm_ir", ""))
        ast = _compact_ast(function_data.get("ast", {}))
        include = {"source": True, "calls": True, "cfg": True, "ast": True, "llvm": True}
        if stage == "compact_ast":
            ast = _compact_ast(function_data.get("ast", {}), 400)
        elif stage == "compact_cfg":
            ast = _compact_ast(function_data.get("ast", {}), 400)
            cfg = "\n".join(_compact_cfg(function_data.get("cfg", {})).splitlines()[:10])
        elif stage == "compact_llvm":
            ast = _compact_ast(function_data.get("ast", {}), 250)
            cfg = "\n".join(_compact_cfg(function_data.get("cfg", {})).splitlines()[:8])
            llvm = _compact_llvm(function_data.get("llvm_ir", ""), 2200)
        elif stage == "remove_llvm":
            ast = _compact_ast(function_data.get("ast", {}), 150)
            cfg = "\n".join(_compact_cfg(function_data.get("cfg", {})).splitlines()[:6])
            llvm = "removed to fit context"
            include["llvm"] = False
        elif stage == "minimal_source_context":
            ast = "removed to fit context"
            cfg = "removed to fit context"
            llvm = "removed to fit context"
            include.update(ast=False, cfg=False, llvm=False)

        identity = {
            "id": function_data.get("id"),
            "name": function_data.get("name"),
            "qualified_name": function_data.get("qualified_name"),
            "file": source_location.get("file") if isinstance(source_location, dict) else "",
            "source_range": source_location,
            "signature": {
                "parameters": function_data.get("parameters", []),
                "return_type": function_data.get("return_type"),
            },
        }
        system_prompt = ("You are a software code-analysis assistant.\n"
                         "Analyze individual functions using their actual source code and\n"
                         "compiler-derived information.\n"
                         "Produce concise, factual semantic descriptions.\n"
                         "The source implementation is the primary source of truth.\n"
                         "Never invent behaviour.\n"
                         "Never invent relationships.\n"
                         "Never modify compiler facts.")
        
        user_prompt = f"""Analyze this ONE function.

Generate a JSON object with these fields:

1. purpose
2. behavior
3. summary
4. inputs (array)
5. outputs (array)
6. side_effects (array)
7. dependencies (array)
8. concepts (array)
9. keywords (array)
10. algorithm
11. complexity (object with time and space)
Use "unknown" when complexity cannot be determined reliably. Use empty arrays
only when no items are supported by the evidence.

Rules:
- Base the answer on the supplied implementation.
- Do not infer behaviour solely from the function name.
- Do not invent functionality.
- Do not describe unrelated functions.
- Do not generate CFG edges.
- Do not generate call graph relationships.
- Do not repeat these instructions.
- Do not use generic/template descriptions.
- Keep behaviour to 1-2 sentences.
- Keep purpose to 1-2 sentences.
- Never invent calls, dependencies, source lines, or compiler facts.
- If there is insufficient evidence, say:
  "Insufficient implementation context."

Return ONLY valid JSON:

{{
  "purpose": "...", "behavior": "...", "summary": "...",
  "inputs": [], "outputs": [], "side_effects": [], "dependencies": [],
  "concepts": [], "keywords": [], "algorithm": "...",
  "complexity": {{"time": "unknown", "space": "unknown"}}
}}

FUNCTION:
{json.dumps(identity, indent=2)}

SOURCE:
{source}

LLVM IR:
{llvm}

CALLS:
{calls}

CALLED BY:
{json.dumps(called_by)}

CFG SUMMARY:
{cfg}

AST:
{ast}
"""
        return system_prompt, user_prompt, include

    def _build_context(self, function_data: Dict[str, Any]) -> tuple[list[dict], Dict[str, Any], List[str]]:
        stages = ["full", "compact_ast", "compact_cfg", "compact_llvm", "remove_llvm", "minimal_source_context"]
        input_budget = self.max_input_tokens or (
            self.context_size - self.reserved_output_tokens - CONTEXT_SAFETY_MARGIN
        )
        input_budget = max(256, min(input_budget, self.context_size - self.reserved_output_tokens - CONTEXT_SAFETY_MARGIN))
        attempts = []
        for stage in stages:
            system_prompt, user_prompt, included = self._prompt_for_stage(function_data, stage)
            estimated = _estimate_tokens(system_prompt + user_prompt)
            attempts.append((stage, system_prompt, user_prompt, included, estimated))
            if estimated <= input_budget:
                metrics = {
                    "model_context_size": self.context_size,
                    "estimated_input_tokens": estimated,
                    "reserved_output_tokens": self.reserved_output_tokens,
                    "input_budget": input_budget,
                    "total_budget": estimated + self.reserved_output_tokens,
                    "context_reduction_level": stage,
                    "source_included": included["source"],
                    "ast_included": included["ast"],
                    "cfg_included": included["cfg"],
                    "llvm_included": included["llvm"],
                }
                if self.debug:
                    print(
                        "Context:\n"
                        f"  Model context:       {self.context_size}\n"
                        f"  Input tokens:        {estimated}\n"
                        f"  Reserved output:     {self.reserved_output_tokens}\n"
                        f"  Total budget:        {estimated + self.reserved_output_tokens}\n"
                        f"  Status:              OK"
                    )
                    if stage != "full":
                        print(f"Reducing optional context: {stage}")
                return [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ], metrics, stages[stages.index(stage):]
        # The source is high priority, but a very large source still needs a bounded final representation.
        stage, system_prompt, user_prompt, included, _ = attempts[-1]
        available_chars = max(200, input_budget * 3 - _estimate_tokens(system_prompt) * 3)
        source_marker = "\nSOURCE:\n"
        source_start = user_prompt.find(source_marker) + len(source_marker)
        source_end = user_prompt.find("\n\nLLVM IR:", source_start)
        if source_start >= len(source_marker) and source_end >= source_start:
            source = _trim_source(user_prompt[source_start:source_end], available_chars)
            user_prompt = user_prompt[:source_start] + source + user_prompt[source_end:]
        estimated = _estimate_tokens(system_prompt + user_prompt)
        while estimated > input_budget and source_start >= len(source_marker) and source_end >= source_start:
            current_source = user_prompt[source_start:source_end]
            reduced = _trim_source(current_source, max(80, int(len(current_source) * 0.85)))
            if reduced == current_source:
                break
            user_prompt = user_prompt[:source_start] + reduced + user_prompt[source_end:]
            estimated = _estimate_tokens(system_prompt + user_prompt)
        metrics = {
            "model_context_size": self.context_size,
            "estimated_input_tokens": estimated,
            "reserved_output_tokens": self.reserved_output_tokens,
            "input_budget": input_budget,
            "total_budget": estimated + self.reserved_output_tokens,
            "context_reduction_level": stage,
            "source_included": True,
            "ast_included": False,
            "cfg_included": False,
            "llvm_included": False,
        }
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], metrics, []

    def enrich_function(self, function_data: Dict[str, Any], on_model_failure=None):
        """Send one budgeted function prompt to LM Studio."""
        messages, context_metrics, remaining_stages = self._build_context(function_data)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": self.reserved_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "optima_enrichment",
                    "strict": True,
                    "schema": ENRICHMENT_JSON_SCHEMA,
                },
            },
        }
        max_retries = 2
        attempt = 0
        reduction_retry_used = False
        while attempt <= max_retries:
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "******"},
                method="POST",
            )
            start_time = time.time()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_data = json.loads(response.read().decode("utf-8", errors="replace"))
                latency_ms = int((time.time() - start_time) * 1000)
                content = response_data["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError("LM Studio message content is not a string")
                result, parse_reason = _parse_json_object(content)
                valid, schema_reason = self._validate_enhancement(result)
                if valid and result is not None:
                    result["model"] = self.model
                    result["status"] = "completed"
                    usage = response_data.get("usage", {})
                    result["usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens", context_metrics["estimated_input_tokens"]),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    }
                    return result, {
                        "request_success": True, "json_valid": True, "retry_count": attempt,
                        "latency_seconds": latency_ms / 1000,
                        "context_metrics": context_metrics,
                    }
                reason = schema_reason if result is not None else parse_reason
                logger.warning(
                    "Enrichment failed: node_id=%s attempt=%d reason=%s response_preview=%s",
                    function_data.get("id", "<unknown>"), attempt + 1, reason,
                    _response_preview(content),
                )
                if attempt >= max_retries:
                    return self._get_fallback_enhancement(attempt, reason), {
                        "request_success": False, "json_valid": False,
                        "retry_count": attempt, "latency_seconds": latency_ms / 1000,
                        "failure_reason": reason,
                        "response_preview": _response_preview(content),
                        "context_metrics": context_metrics,
                    }
            except urllib.error.HTTPError as error:
                error_body = error.read().decode("utf-8", errors="replace")
                if "exceed_context_size" in error_body or "context size" in error_body.lower():
                    if not reduction_retry_used and remaining_stages and len(remaining_stages) > 1:
                        reduction_retry_used = True
                        next_stage = remaining_stages[1]
                        system_prompt, user_prompt, included = self._prompt_for_stage(function_data, next_stage)
                        estimated = _estimate_tokens(system_prompt + user_prompt)
                        context_metrics = {
                            **context_metrics,
                            "estimated_input_tokens": estimated,
                            "total_budget": estimated + self.reserved_output_tokens,
                            "context_reduction_level": next_stage,
                            "source_included": included["source"],
                            "ast_included": included["ast"],
                            "cfg_included": included["cfg"],
                            "llvm_included": included["llvm"],
                        }
                        payload["messages"] = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]
                        remaining_stages = remaining_stages[1:]
                        logger.warning("Context limit reported by LM Studio; reduced context to %s", next_stage)
                        continue
                    logger.error("LM Studio rejected the prompt for context size; not retrying unchanged request")
                    return self._get_fallback_enhancement(attempt, "context_limit"), {
                        "request_success": False, "json_valid": False,
                        "retry_count": attempt, "latency_seconds": time.time() - start_time,
                        "context_metrics": context_metrics,
                    }
                reason = "api_request_failure"
                logger.error(
                    "Enrichment failed: node_id=%s attempt=%d reason=%s http_status=%s response_preview=%s",
                    function_data.get("id", "<unknown>"), attempt + 1, reason,
                    error.code, _response_preview(error_body),
                )
                if on_model_failure is not None and attempt < max_retries:
                    on_model_failure()
                if attempt >= max_retries:
                    return self._get_fallback_enhancement(attempt, reason), {
                        "request_success": False, "json_valid": False,
                        "retry_count": attempt, "latency_seconds": time.time() - start_time,
                        "failure_reason": reason,
                        "context_metrics": context_metrics,
                    }
            except urllib.error.URLError as error:
                reason = "api_request_failure"
                logger.error(
                    "Enrichment failed: node_id=%s attempt=%d reason=%s error=%s",
                    function_data.get("id", "<unknown>"), attempt + 1, reason, error.reason,
                )
                if on_model_failure is not None and attempt < max_retries:
                    on_model_failure()
                if attempt >= max_retries:
                    return self._get_fallback_enhancement(attempt, reason), {
                        "request_success": False, "json_valid": False,
                        "retry_count": attempt, "latency_seconds": time.time() - start_time,
                        "failure_reason": reason,
                        "context_metrics": context_metrics,
                    }
            except Exception as error:
                reason = "unexpected_error"
                logger.exception(
                    "Enrichment failed: node_id=%s attempt=%d reason=%s",
                    function_data.get("id", "<unknown>"), attempt + 1, reason,
                )
                if attempt >= max_retries:
                    return self._get_fallback_enhancement(attempt, reason), {
                        "request_success": False, "json_valid": False,
                        "retry_count": attempt, "latency_seconds": time.time() - start_time,
                        "failure_reason": reason,
                        "context_metrics": context_metrics,
                    }
            attempt += 1
            time.sleep(1)

        # Prepare the request
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 500  # We don't need too many tokens
        }
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )

        # Retry logic
        max_retries = 2
        for attempt in range(max_retries + 1):
            start_time = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    response_data = json.load(response)
                    latency_ms = int((time.time() - start_time) * 1000)
                    content = response_data["choices"][0]["message"]["content"].strip()
                    # Parse the JSON response
                    try:
                        result = json.loads(content)
                        # Validate the result
                        if self._validate_enhancement(result):
                            # Add metadata
                            result["model"] = self.model
                            result["status"] = "completed"
                            # Usage information is not always provided; we'll try to get it if available
                            usage = response_data.get("usage", {})
                            if usage:
                                result["usage"] = {
                                    "prompt_tokens": usage.get("prompt_tokens"),
                                    "completion_tokens": usage.get("completion_tokens"),
                                    "total_tokens": usage.get("total_tokens")
                                }
                            else:
                                result["usage"] = {
                                    "prompt_tokens": None,
                                    "completion_tokens": None,
                                    "total_tokens": None
                                }
                            return result, {
                                "request_success": True,
                                "json_valid": True,
                                "retry_count": attempt,
                                "latency_seconds": latency_ms / 1000,
                            }
                        else:
                            logger.warning(f"Invalid enrichment received (attempt {attempt+1}): {result}")
                            if attempt == max_retries:
                                return self._get_fallback_enhancement(attempt), {
                                    "request_success": False,
                                    "json_valid": False,
                                    "retry_count": attempt,
                                    "latency_seconds": latency_ms / 1000,
                                }
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from LM Studio (attempt {attempt+1}): {content}")
                        if attempt == max_retries:
                            return self._get_fallback_enhancement(attempt), {
                                "request_success": False,
                                "json_valid": False,
                                "retry_count": attempt,
                                "latency_seconds": time.time() - start_time,
                            }
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")
                logger.error(
                    f"HTTP error from LM Studio (attempt {attempt+1}): "
                    f"{e.code} {e.reason}: {error_body[:500]}"
                )
                if on_model_failure is not None and attempt < max_retries:
                    on_model_failure()
                if attempt == max_retries:
                    return self._get_fallback_enhancement(attempt), {
                        "request_success": False, "json_valid": False,
                        "retry_count": attempt, "latency_seconds": time.time() - start_time,
                    }
            except urllib.error.URLError as e:
                logger.error(f"URL error from LM Studio (attempt {attempt+1}): {e.reason}")
                if on_model_failure is not None and attempt < max_retries:
                    on_model_failure()
                if attempt == max_retries:
                    return self._get_fallback_enhancement(attempt), {
                        "request_success": False, "json_valid": False,
                        "retry_count": attempt, "latency_seconds": time.time() - start_time,
                    }
            except Exception as e:
                logger.error(f"Unexpected error enriching function (attempt {attempt+1}): {e}")
                if on_model_failure is not None and attempt < max_retries:
                    on_model_failure()
                if attempt == max_retries:
                            return self._get_fallback_enhancement(attempt), {
                                "request_success": False, "json_valid": False,
                                "retry_count": attempt, "latency_seconds": time.time() - start_time,
                            }
            # Wait a bit before retrying
            time.sleep(1)

    def _validate_enhancement(self, enhancement: Optional[Dict[str, Any]]) -> tuple[bool, str]:
        """Validate the enrichment result."""
        if not isinstance(enhancement, dict):
            return False, "json_root_not_object"
        required_strings = ("purpose", "behavior", "summary", "algorithm")
        required_arrays = ("inputs", "outputs", "side_effects", "dependencies", "concepts", "keywords")
        missing = [field for field in (*required_strings, *required_arrays, "complexity")
                   if field not in enhancement]
        if missing:
            return False, f"missing_fields:{','.join(missing)}"
        if any(not isinstance(enhancement[field], str) for field in required_strings):
            return False, "wrong_field_type:string"
        if any(
            not isinstance(enhancement[field], list)
            or any(not isinstance(item, str) for item in enhancement[field])
            for field in required_arrays
        ):
            return False, "wrong_field_type:string_array"
        complexity = enhancement["complexity"]
        if (
            not isinstance(complexity, dict)
            or not isinstance(complexity.get("time"), str)
            or not isinstance(complexity.get("space"), str)
        ):
            return False, "wrong_field_type:complexity"
        if not enhancement["purpose"].strip() or not enhancement["behavior"].strip():
            return False, "empty_required_string"
        # Reject known placeholder responses
        placeholders = [
            "What the function actually does",
            "What this function is intended to accomplish",
            "What this function is intended to do"
        ]
        if enhancement["behavior"].strip() in placeholders or enhancement["purpose"].strip() in placeholders:
            return False, "placeholder_response"
        return True, "valid"

    def _get_fallback_enhancement(
        self, retry_count: int = 0, failure_reason: str = "enrichment_failed"
    ) -> Dict[str, Any]:
        """Return a fallback enhancement when LM Studio fails."""
        return {
            "behavior": "Insufficient implementation context.",
            "purpose": "Insufficient implementation context.",
            "summary": "Insufficient implementation context.",
            "inputs": [],
            "outputs": [],
            "side_effects": [],
            "dependencies": [],
            "concepts": [],
            "keywords": [],
            "algorithm": "unknown",
            "complexity": {"time": "unknown", "space": "unknown"},
            "confidence": {},
            "model": self.model,
            "status": "failed",
            "failure_reason": failure_reason,
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
        }


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._") or "unknown-model"


def _usable(value: Any) -> bool:
    return bool(value) if isinstance(value, (str, list, dict)) else value is not None


def _unique_string_count(value: Any) -> int:
    """Count distinct schema-valid values without hashing arbitrary model data."""
    if not isinstance(value, list):
        return 0
    return len({item for item in value if isinstance(item, str)})


def _field_completeness(enrichment: Dict[str, Any]) -> Dict[str, bool]:
    fields = ["purpose", "behavior", "summary", "inputs", "outputs",
              "side_effects", "dependencies", "concepts", "keywords",
              "algorithm", "complexity"]
    return {field: _usable(enrichment.get(field)) for field in fields}


def _evaluation(function_data: Dict[str, Any], enrichment: Dict[str, Any],
                request_metrics: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    completeness = _field_completeness(enrichment)
    usage = enrichment.get("usage", {})
    total_tokens = usage.get("total_tokens")
    return {
        **request_metrics,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": total_tokens,
        "response_characters": len(json.dumps(enrichment, ensure_ascii=False)),
        "field_completeness": completeness,
        "keyword_count": len(enrichment.get("keywords", [])) if isinstance(enrichment.get("keywords"), list) else 0,
        "concept_count": len(enrichment.get("concepts", [])) if isinstance(enrichment.get("concepts"), list) else 0,
        "unique_keyword_count": _unique_string_count(enrichment.get("keywords")),
        "unique_concept_count": _unique_string_count(enrichment.get("concepts")),
        "summary_length": len(str(enrichment.get("summary", ""))),
        "purpose_length": len(str(enrichment.get("purpose", ""))),
        "behavior_length": len(str(enrichment.get("behavior", enrichment.get("behaviour", "")))),
        "context_metrics": context,
        "automatic": {},
        "human": {"accuracy": None, "relevance": None, "faithfulness": None, "usefulness": None},
        "model_confidence": enrichment.get("confidence", {}),
    }


def _dataset_metrics(functions: List[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
    evaluations = [f["enrichment"]["evaluation"] for f in functions if "enrichment" in f]
    total = len(functions)
    successes = sum(e["request_success"] for e in evaluations)
    # json_valid is defined identically to request_success (JSON recovered
    # by any means -- raw, Markdown-fenced, or extracted from surrounding
    # text), never full schema conformance -- see enrich_one()'s comment.
    # schema_valid (below) is the separate, stricter, purely observational
    # metric that used to be mislabeled json_valid.
    valid = sum(e["json_valid"] for e in evaluations)
    schema_valid_count = sum(1 for e in evaluations if e.get("schema_valid"))
    latencies = [e["latency_seconds"] for e in evaluations]
    fields = ["purpose", "behavior", "summary", "inputs", "outputs", "side_effects",
              "dependencies", "concepts", "keywords", "algorithm", "complexity"]
    def rate(field):
        return sum(e["field_completeness"].get(field, False) for e in evaluations) / len(evaluations) if evaluations else 0

    # JSON-recovery method breakdown -- only meaningful for successful
    # (json_valid) requests; json_recovery_method is None/absent otherwise.
    # These are strictly a partition of `successes`, plus the explicit
    # unrecoverable count (identical to failed_requests, named for clarity
    # in the JSON-recovery context specifically).
    raw_json_valid_count = sum(1 for e in evaluations if e.get("json_recovery_method") == "json")
    markdown_json_recovered_count = sum(
        1 for e in evaluations if e.get("json_recovery_method") == "markdown_wrapped_json"
    )
    surrounding_text_recovered_count = sum(
        1 for e in evaluations if e.get("json_recovery_method") == "surrounding_text"
    )
    json_repair_count = sum(1 for e in evaluations if e.get("request_success") and e.get("repair_used"))
    unrecoverable_json_count = sum(1 for e in evaluations if not e.get("request_success"))

    return {
        "functions_total": total,
        "functions_enriched": successes,
        "functions_failed": total - successes,
        "total_requests": len(evaluations),
        "successful_requests": successes,
        "failed_requests": total - successes,
        "success_rate": successes / total if total else 0,
        "json_valid_rate": valid / total if total else 0,
        "raw_json_valid_count": raw_json_valid_count,
        "markdown_json_recovered_count": markdown_json_recovered_count,
        "surrounding_text_recovered_count": surrounding_text_recovered_count,
        "json_repair_count": json_repair_count,
        "unrecoverable_json_count": unrecoverable_json_count,
        "schema_valid_count": schema_valid_count,
        "schema_valid_rate": schema_valid_count / total if total else 0,
        "retry_rate": sum(e["retry_count"] > 0 for e in evaluations) / total if total else 0,
        "failure_rate": (total - successes) / total if total else 0,
        "average_latency_seconds": statistics.mean(latencies) if latencies else 0,
        "median_latency_seconds": statistics.median(latencies) if latencies else 0,
        "p95_latency_seconds": sorted(latencies)[min(len(latencies) - 1, max(0, int(len(latencies) * .95) - 1))] if latencies else 0,
        "total_enrichment_seconds": elapsed,
        "functions_per_second": total / elapsed if elapsed else 0,
        "average_keyword_count": statistics.mean([e["keyword_count"] for e in evaluations]) if evaluations else 0,
        "average_concept_count": statistics.mean([e["concept_count"] for e in evaluations]) if evaluations else 0,
        "average_summary_characters": statistics.mean([e["summary_length"] for e in evaluations]) if evaluations else 0,
        "field_completeness": {field: rate(field) for field in fields},
        "overall_field_completeness": statistics.mean([sum(e["field_completeness"].values()) / len(fields) for e in evaluations]) if evaluations else 0,
    }


def enrich_analysis(base_json_path: Path, output_dir: Path, model: Optional[str] = None,
                    output_path: Optional[Path] = None, base_url: str = DEFAULT_BASE_URL,
                    model_load_timeout: int = 300, limit: Optional[int] = None,
                    context_size: Optional[int] = None, max_input_tokens: Optional[int] = None,
                    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
                    debug: bool = False) -> Path:
    """Enrich base.json with LM Studio to produce enhanced.json."""
    base_json_path = base_json_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(base_json_path, 'r') as f:
        base_data = json.load(f)

    client = LMStudioClient(
        base_url=base_url, model=model, context_size=context_size,
        max_input_tokens=max_input_tokens,
        reserved_output_tokens=reserved_output_tokens, debug=debug,
    )
    if not client.check_connection():
        logger.error("LM Studio connection check failed. Please ensure LM Studio is running and a model is loaded.")
        sys.exit(1)
    print("Optima Enrichment\n=================\n")
    print(f"Base: {base_json_path}\nModel: {client.model}\n")
    print("Checking LM Studio...\nModel available: YES")
    model_runtime = client.prepare_model(timeout_seconds=model_load_timeout)

    if output_path is None:
        output_file = output_dir / f"enhanced_{_safe_model_name(client.model)}.json"
    else:
        output_file = output_path.resolve()
        if output_file.is_dir():
            output_file = output_file / f"enhanced_{_safe_model_name(client.model)}.json"
    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
    result_data = json.loads(json.dumps(base_data))
    prior_functions = {
        f["id"]: f for file_info in existing.get("files", [])
        for f in file_info.get("functions", [])
        if f.get("enrichment", {}).get("evaluation", {}).get("request_success") is True
    }
    started_at = _timestamp()
    experiment_start = time.time()
    all_functions = []
    for file_info in result_data.get("files", []):
        all_functions.extend(file_info.get("functions", []))
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        all_functions = all_functions[:limit]
    total_functions = len(all_functions)
    functions_by_id = {func["id"]: func for func in all_functions}
    context = lambda func: {
        "source_code": bool(func.get("source_code")),
        "ast": bool(func.get("ast")),
        "call_graph": bool(func.get("calls") or func.get("called_by")),
        "llvm": bool(func.get("llvm_ir")),
        "cfg": bool(func.get("cfg", {}).get("nodes") or func.get("llvm", {}).get("cfg", {}).get("nodes")),
        "source_characters": len(func.get("source_code", "")),
        "llvm_characters": len(func.get("llvm_ir", "")),
        "cfg_nodes": len(func.get("cfg", {}).get("nodes", [])),
        "cfg_edges": len(func.get("cfg", {}).get("edges", [])),
        "call_count": len(func.get("calls", [])),
    }
    for file_info in base_data.get("files", []):
        for func in file_info.get("functions", []):
            if func["id"] not in functions_by_id:
                continue
            target = next(f for f in result_data["files"] if f["relative_path"] == file_info["relative_path"])["functions"]
            target_func = next(f for f in target if f["id"] == func["id"])
            if func["id"] in prior_functions:
                target_func["enrichment"] = prior_functions[func["id"]]["enrichment"]
            else:
                enrichment, metrics = client.enrich_function(
                    func,
                    on_model_failure=lambda: client.prepare_model(
                        timeout_seconds=model_load_timeout
                    ),
                )
                evaluation_context = {
                    **context(func),
                    **metrics.get("context_metrics", {}),
                }
                enrichment["evaluation"] = _evaluation(func, enrichment, metrics, evaluation_context)
                target_func["enrichment"] = enrichment
                result_data["enrichment_metadata"] = {
                    "provider": "LM Studio", "base_url": client.base_url,
                    "model": client.model, "model_id": client.model_id,
                    "started_at": started_at, "completed_at": _timestamp(),
                    "total_functions": total_functions,
                    "model_runtime": model_runtime,
                }
                result_data["enrichment_metrics"] = _dataset_metrics(all_functions, time.time() - experiment_start)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile("w", dir=output_file.parent, delete=False) as tmp:
                    json.dump(result_data, tmp, indent=2)
                    temp_name = tmp.name
                os.replace(temp_name, output_file)

    result_data["enrichment_metadata"] = {
        "provider": "LM Studio", "base_url": client.base_url,
        "model": client.model, "model_id": client.model_id,
        "started_at": started_at, "completed_at": _timestamp(),
        "total_functions": total_functions,
        "enriched_functions": sum(
            f.get("enrichment", {}).get("evaluation", {}).get("request_success", False)
            for f in all_functions
        ),
        "model_runtime": model_runtime,
        "failed_functions": sum(
            not f.get("enrichment", {}).get("evaluation", {}).get("request_success", False)
            for f in all_functions
        ),
    }
    result_data["experiment"] = {
        "model": client.model, "temperature": 0.1, "batch_size": 1,
        "prompt_version": "v2", "schema_version": "v1",
        "base_json": str(base_json_path),
    }
    result_data["enrichment_metrics"] = _dataset_metrics(all_functions, time.time() - experiment_start)
    result_data["enrichment_metrics"]["model_load_seconds"] = model_runtime["load_wait_seconds"]
    result_data["enrichment_metrics"]["total_inference_seconds"] = sum(
        f["enrichment"]["evaluation"]["latency_seconds"]
        for f in all_functions if "enrichment" in f
    )
    result_data["enrichment_metrics"]["total_runtime_seconds"] = time.time() - experiment_start
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    return output_file


if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) != 2:
        print("Usage: python enricher.py <base_json_path>")
        sys.exit(1)
    base_json_path = Path(sys.argv[1])
    output_dir = Path("output")
    enrich_analysis(base_json_path, output_dir)
    print(f"Enrichment complete. Output in {output_dir}")
