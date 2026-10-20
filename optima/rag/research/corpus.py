"""Load Optima base / enriched JSON into compact, ID-stable function records.

Only the fields the research pipelines need are kept; the (often 100 MB) parsed
JSON is released as soon as the records are extracted.  Function IDs are never
rewritten: they are exactly the ``id`` values of ``base.json``, which is the
authoritative universe of functions.  Enrichment is joined to it by ID.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SEMANTIC_FIELDS: Tuple[str, ...] = (
    "purpose", "behavior", "summary", "inputs", "outputs", "side_effects",
    "concepts", "keywords", "algorithm",
)
# Fields that must carry text for enrichment to count as usable.  ``inputs``
# and friends are legitimately empty for many functions.
_CORE_SEMANTIC_FIELDS = ("purpose", "behavior", "summary")
_PLACEHOLDERS = {"", "unknown", "none", "n/a", "null"}


class CorpusError(ValueError):
    """Raised for unreadable or structurally invalid Optima JSON."""


@dataclass
class FunctionRecord:
    id: str
    name: str
    qualified_name: str
    file: str
    language: str = ""
    return_type: str = ""
    parameters: Tuple[Tuple[str, str], ...] = ()
    source_code: str = ""
    analysis_status: str = ""
    is_method: bool = False
    class_name: str = ""
    # Resolved call edges, by function ID, and unresolved (external) callee names.
    calls: Tuple[str, ...] = ()
    called_by: Tuple[str, ...] = ()
    external_calls: Tuple[str, ...] = ()
    dependencies: Tuple[Any, ...] = ()
    cfg: Optional[Dict[str, Any]] = None
    # Semantic enrichment (None when there is no enrichment record at all).
    semantic: Optional[Dict[str, Any]] = None
    enrichment_status: str = "missing"   # ok | empty | failed | missing
    enriched_dependencies: Tuple[str, ...] = ()
    enrichment_usage: Dict[str, float] = field(default_factory=dict)

    @property
    def has_semantic(self) -> bool:
        return self.enrichment_status == "ok"


@dataclass
class Corpus:
    records: Dict[str, FunctionRecord]
    project: Dict[str, Any]
    extra_edges: List[Tuple[str, str]]
    meta: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def ids(self) -> List[str]:
        return list(self.records)

    def enrichment_summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {"ok": 0, "empty": 0, "failed": 0, "missing": 0}
        for record in self.records.values():
            counts[record.enrichment_status] = counts.get(record.enrichment_status, 0) + 1
        total = len(self.records) or 1
        usage = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0,
                 "latency_seconds": 0.0}
        for record in self.records.values():
            for key in usage:
                usage[key] += float(record.enrichment_usage.get(key, 0.0) or 0.0)
        return {
            "functions": len(self.records),
            "status_counts": counts,
            "semantic_coverage": counts["ok"] / total,
            "failed_or_incomplete": total - counts["ok"],
            "usage_totals": usage,
        }


# --------------------------------------------------------------------- helpers
def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CorpusError(f"JSON file not found: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path} is not valid JSON ({exc.msg} at line {exc.lineno})") from exc
    except OSError as exc:
        raise CorpusError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("files"), list):
        raise CorpusError(
            f"{path} does not look like Optima output: expected an object with a 'files' list"
        )
    return data


def _iter_functions(data: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for file_data in data.get("files", []):
        if not isinstance(file_data, dict):
            continue
        for func in file_data.get("functions", []) or []:
            if isinstance(func, dict):
                yield file_data, func


def _usable_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _PLACEHOLDERS
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def cfg_summary(cfg: Any) -> Optional[Dict[str, Any]]:
    """Summarise a CFG dict as blocks / edges / branching / loop presence."""
    if not isinstance(cfg, dict):
        return None
    nodes = [str(n) for n in (cfg.get("nodes") or [])]
    if not nodes:
        return None
    edges = []
    for edge in cfg.get("edges") or []:
        if isinstance(edge, dict) and "from" in edge and "to" in edge:
            edges.append((str(edge["from"]), str(edge["to"])))
    successors: Dict[str, List[str]] = {n: [] for n in nodes}
    for src, dst in edges:
        successors.setdefault(src, []).append(dst)
    entry = str(cfg.get("entry_node") or nodes[0])
    return {
        "blocks": len(nodes),
        "edges": len(edges),
        "branch_blocks": sum(1 for n in nodes if len(set(successors.get(n, []))) >= 2),
        "cyclomatic": len(edges) - len(nodes) + 2,
        "has_loop": _has_cycle(entry, successors),
    }


def _has_cycle(entry: str, successors: Dict[str, List[str]]) -> bool:
    """Iterative DFS back-edge detection from the entry block."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour: Dict[str, int] = {}
    stack: List[Tuple[str, int]] = [(entry, 0)]
    colour[entry] = GREY
    while stack:
        node, index = stack[-1]
        succ = successors.get(node, [])
        if index < len(succ):
            stack[-1] = (node, index + 1)
            nxt = succ[index]
            state = colour.get(nxt, WHITE)
            if state == GREY:
                return True
            if state == WHITE:
                colour[nxt] = GREY
                stack.append((nxt, 0))
        else:
            colour[node] = BLACK
            stack.pop()
    return False


def _edge_ids(entries: Any, resolved_only: bool) -> Tuple[List[str], List[str]]:
    """Split call entries into resolved IDs and unresolved callee names."""
    ids: List[str] = []
    names: List[str] = []
    for entry in entries or []:
        if isinstance(entry, dict):
            ident = entry.get("id")
            if ident and (entry.get("resolved", True) or not resolved_only):
                ids.append(str(ident))
            else:
                label = entry.get("qualified_name") or entry.get("name")
                if label:
                    names.append(str(label))
        elif isinstance(entry, str) and entry:
            names.append(entry)
    return list(dict.fromkeys(ids)), list(dict.fromkeys(names))


def _extract_enrichment(func: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str,
                                                        Tuple[str, ...], Dict[str, float]]:
    enrichment = func.get("enrichment")
    if not isinstance(enrichment, dict) or not enrichment:
        return None, "missing", (), {}
    semantic = {key: enrichment[key] for key in SEMANTIC_FIELDS if key in enrichment}
    usage: Dict[str, float] = {}
    raw_usage = enrichment.get("usage")
    if isinstance(raw_usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(raw_usage.get(key), (int, float)):
                usage[key] = float(raw_usage[key])
    evaluation = enrichment.get("evaluation")
    if isinstance(evaluation, dict) and isinstance(evaluation.get("latency_seconds"), (int, float)):
        usage["latency_seconds"] = float(evaluation["latency_seconds"])
    deps = enrichment.get("dependencies")
    enriched_deps: Tuple[str, ...] = ()
    if isinstance(deps, list):
        enriched_deps = tuple(str(d) for d in deps if isinstance(d, str) and d.strip())
    status = str(enrichment.get("status") or "").lower()
    if status == "failed":
        return None, "failed", (), usage
    if not any(_usable_text(semantic.get(key)) for key in _CORE_SEMANTIC_FIELDS):
        return None, "empty", enriched_deps, usage
    return semantic, "ok", enriched_deps, usage


def _record_from(file_data: Dict[str, Any], func: Dict[str, Any]) -> Optional[FunctionRecord]:
    ident = func.get("id")
    if not ident or not isinstance(ident, str):
        return None
    class_info = func.get("class_info") if isinstance(func.get("class_info"), dict) else {}
    qualified = str(func.get("qualified_name") or func.get("name") or "")
    name = str(func.get("name") or qualified.split("::")[-1])
    calls, external = _edge_ids(func.get("calls"), resolved_only=True)
    called_by, _ = _edge_ids(func.get("called_by"), resolved_only=False)
    params: List[Tuple[str, str]] = []
    for param in func.get("parameters") or []:
        if isinstance(param, dict):
            params.append((str(param.get("type") or ""), str(param.get("name") or "")))
    semantic, status, enriched_deps, usage = _extract_enrichment(func)
    deps = func.get("dependencies")
    return FunctionRecord(
        id=ident, name=name, qualified_name=qualified,
        file=str(file_data.get("relative_path") or file_data.get("path") or ""),
        language=str(file_data.get("language") or ""),
        return_type=str(func.get("return_type") or ""),
        parameters=tuple(params),
        source_code=str(func.get("source_code") or ""),
        analysis_status=str(func.get("analysis_status") or ""),
        is_method=bool(class_info.get("is_method")),
        class_name="::".join(qualified.split("::")[:-1]),
        calls=tuple(calls), called_by=tuple(called_by), external_calls=tuple(external),
        dependencies=tuple(deps) if isinstance(deps, list) else (),
        cfg=cfg_summary(func.get("cfg")),
        semantic=semantic, enrichment_status=status,
        enriched_dependencies=enriched_deps, enrichment_usage=usage,
    )


# ------------------------------------------------------------------- loading
def load_corpus(base_json: Path, enriched_json: Optional[Path] = None) -> Corpus:
    """Build a :class:`Corpus` from ``base.json`` and an optional enriched file.

    The set of functions is exactly the set of IDs in ``base.json``.  Duplicate
    IDs keep the first occurrence and are counted in ``meta``.  Functions with no
    enrichment record keep ``enrichment_status == "missing"``; they are never
    dropped.
    """
    base_json = Path(base_json)
    data = _read_json(base_json)
    project = data.get("project") if isinstance(data.get("project"), dict) else {}
    records: Dict[str, FunctionRecord] = {}
    duplicates: List[str] = []
    skipped_no_id = 0
    for file_data, func in _iter_functions(data):
        record = _record_from(file_data, func)
        if record is None:
            skipped_no_id += 1
            continue
        if record.id in records:
            duplicates.append(record.id)
            continue
        records[record.id] = record
    graph = data.get("graph") if isinstance(data.get("graph"), dict) else {}
    extra_edges: List[Tuple[str, str]] = []
    for edge in graph.get("call_edges") or []:
        if isinstance(edge, dict) and edge.get("from") and edge.get("to"):
            extra_edges.append((str(edge["from"]), str(edge["to"])))
    del data
    gc.collect()
    if not records:
        raise CorpusError(f"{base_json} contains no functions with an 'id'")

    meta: Dict[str, Any] = {
        "base_json": str(base_json),
        "base_sha256": file_sha256(base_json),
        "project": project.get("name", ""),
        "language": project.get("language", ""),
        "functions": len(records),
        "duplicate_ids": len(duplicates),
        "duplicate_id_examples": duplicates[:5],
        "functions_without_id": skipped_no_id,
        "enriched_json": None,
    }
    if duplicates:
        logger.warning("%s: %d duplicate function IDs ignored (first occurrence kept)",
                       base_json.name, len(duplicates))

    if enriched_json is not None:
        _join_enrichment(records, Path(enriched_json), meta)
    corpus = Corpus(records=records, project=dict(project), extra_edges=extra_edges, meta=meta)
    meta["enrichment"] = corpus.enrichment_summary()
    return corpus


def _join_enrichment(records: Dict[str, FunctionRecord], enriched_json: Path,
                     meta: Dict[str, Any]) -> None:
    data = _read_json(enriched_json)
    enrichment_meta = data.get("enrichment_metadata") if isinstance(
        data.get("enrichment_metadata"), dict) else {}
    experiment = data.get("experiment") if isinstance(data.get("experiment"), dict) else {}
    seen: set = set()
    enriched_only = 0
    duplicates = 0
    for _file_data, func in _iter_functions(data):
        ident = func.get("id")
        if not isinstance(ident, str):
            continue
        if ident in seen:
            duplicates += 1
            continue
        seen.add(ident)
        record = records.get(ident)
        if record is None:
            enriched_only += 1
            continue
        semantic, status, enriched_deps, usage = _extract_enrichment(func)
        record.semantic, record.enrichment_status = semantic, status
        record.enriched_dependencies, record.enrichment_usage = enriched_deps, usage
    del data
    gc.collect()
    meta["enriched_json"] = str(enriched_json)
    meta["enriched_sha256"] = file_sha256(enriched_json)
    meta["enriched_only_ids"] = enriched_only
    meta["enriched_duplicate_ids"] = duplicates
    meta["enrichment_model_reported"] = (
        enrichment_meta.get("model") or experiment.get("model") or "")
    meta["enrichment_run"] = {
        key: enrichment_meta.get(key)
        for key in ("provider", "model", "quantization", "total_functions",
                    "enriched_functions", "failed_functions", "started_at", "completed_at")
        if key in enrichment_meta
    }
    if experiment.get("base_sha256") and experiment["base_sha256"] != meta["base_sha256"]:
        meta["base_sha256_mismatch"] = True
        logger.warning(
            "%s was enriched from a different base.json (sha %s != %s); "
            "IDs are joined by value, but results may not be comparable",
            enriched_json.name, str(experiment["base_sha256"])[:12], meta["base_sha256"][:12])
    if enriched_only:
        logger.warning("%d enriched function IDs are not in base.json and were ignored",
                       enriched_only)
