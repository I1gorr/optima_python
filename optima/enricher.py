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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default LM Studio configuration
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"


class LMStudioClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY, model: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.model = model
        self.model_id = None
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

    def enrich_function(self, function_data: Dict[str, Any], on_model_failure=None):
        """Send function data to LM Studio for enrichment."""
        prompt_function_data = {
            "id": function_data.get("id"),
            "name": function_data.get("name"),
            "qualified_name": function_data.get("qualified_name"),
            "source": function_data.get("source", function_data.get("source_location", {})),
            "parameters": function_data.get("parameters", []),
            "return_type": function_data.get("return_type"),
            "calls": function_data.get("calls", []),
            "llvm": {
                "matched": function_data.get("llvm", {}).get("matched"),
                "function_name": function_data.get("llvm", {}).get("function_name"),
                "mangled_name": function_data.get("llvm", {}).get("mangled_name"),
                "cfg": function_data.get("llvm", {}).get("cfg", {}),
            },
            "cfg": function_data.get("cfg", {}),
            "ast": json.dumps(function_data.get("ast", {}))[:1000],
        }
        # Prepare the prompt
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
12. confidence (object with optional field-level confidence values)

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
  "complexity": {{"time": "unknown", "space": "unknown"}},
  "confidence": {{}}
}}

FUNCTION:
{json.dumps(prompt_function_data, indent=2)}

SOURCE:
{function_data.get('source_code', '')[:6000]}

LLVM IR:
{function_data.get('llvm_ir', '')[:4000]}

CALLS:
{json.dumps(function_data.get('calls', []))}

CALLED BY:
{json.dumps(function_data.get('called_by', []))}

CFG SUMMARY:
{json.dumps(function_data.get('cfg', {}))}

AST:
{json.dumps(function_data.get('ast', {}))[:500]}
"""

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

    def _validate_enhancement(self, enhancement: Dict[str, Any]) -> bool:
        """Validate the enrichment result."""
        if not isinstance(enhancement, dict):
            return False
        behaviour = enhancement.get("behavior", enhancement.get("behaviour", ""))
        purpose = enhancement.get("purpose", "")
        if not isinstance(behaviour, str) or not isinstance(purpose, str):
            return False
        if not behaviour.strip() or not purpose.strip():
            return False
        # Reject known placeholder responses
        placeholders = [
            "What the function actually does",
            "What this function is intended to accomplish",
            "What this function is intended to do"
        ]
        if behaviour.strip() in placeholders or purpose.strip() in placeholders:
            return False
        return True

    def _get_fallback_enhancement(self, retry_count: int = 0) -> Dict[str, Any]:
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
        "unique_keyword_count": len(set(enrichment.get("keywords", []))) if isinstance(enrichment.get("keywords"), list) else 0,
        "unique_concept_count": len(set(enrichment.get("concepts", []))) if isinstance(enrichment.get("concepts"), list) else 0,
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
    valid = sum(e["json_valid"] for e in evaluations)
    latencies = [e["latency_seconds"] for e in evaluations]
    fields = ["purpose", "behavior", "summary", "inputs", "outputs", "side_effects",
              "dependencies", "concepts", "keywords", "algorithm", "complexity"]
    def rate(field):
        return sum(e["field_completeness"].get(field, False) for e in evaluations) / len(evaluations) if evaluations else 0
    return {
        "functions_total": total,
        "functions_enriched": successes,
        "functions_failed": total - successes,
        "total_requests": len(evaluations),
        "successful_requests": successes,
        "failed_requests": total - successes,
        "success_rate": successes / total if total else 0,
        "json_valid_rate": valid / total if total else 0,
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
                    model_load_timeout: int = 300, limit: Optional[int] = None) -> Path:
    """Enrich base.json with LM Studio to produce enhanced.json."""
    base_json_path = base_json_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(base_json_path, 'r') as f:
        base_data = json.load(f)

    client = LMStudioClient(base_url=base_url, model=model)
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
                enrichment["evaluation"] = _evaluation(func, enrichment, metrics, context(func))
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
