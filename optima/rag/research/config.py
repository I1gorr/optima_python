"""Validated, hashable configuration for the research RAG pipelines.

Every advanced stage (BM25, exact matching, graph expansion, reranking) has its
own section with an independent ``enabled``-style switch, so switching one
stage off never changes the behaviour of another.  ``PipelineConfig`` round-trips
through plain JSON (``to_dict`` / ``from_dict``) and exposes ``config_hash`` so a
stored result can be matched to the exact configuration that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

REPRESENTATIONS: Tuple[str, ...] = ("raw", "semantic", "compiler", "coupled")
GRAPH_RELATIONS: Tuple[str, ...] = (
    "callers", "callees", "dependencies", "enriched_dependencies",
)
MISSING_SEMANTIC_POLICIES: Tuple[str, ...] = ("identity", "raw", "exclude")
CONFIG_SCHEMA_VERSION = 1


class ConfigError(ValueError):
    """Raised for an invalid or inconsistent pipeline configuration."""


def _tuple_of_str(value: Any, what: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    try:
        items = tuple(str(item) for item in value)
    except TypeError as exc:
        raise ConfigError(f"{what} must be a list of strings, got {value!r}") from exc
    return items


@dataclass(frozen=True)
class RepresentationConfig:
    """How each searchable representation is rendered to text."""

    raw_max_chars: int = 2000
    semantic_max_chars: int = 2000
    compiler_max_chars: int = 1500
    # Prefix every representation with "name / qualified name / signature" so
    # raw, semantic and compiler text share the same identity header.
    identity_header: bool = True
    # What to index for a function whose enrichment is missing or failed:
    #   identity -> identity header only (still retrievable by name)
    #   raw      -> fall back to the raw text (mixes representations; use knowingly)
    #   exclude  -> leave the function out of the semantic index entirely
    missing_semantic: str = "identity"
    # "coupled" representation: ONE chunk per function holding enrichment and source together, so a single
    # vector / BM25 document scores both.  Order matters for encoders with a token limit (the part that comes
    # first is never truncated away): semantic_first = header, description, source; source_first = header, source,
    # description.  A function without usable enrichment is represented by its raw text only.
    coupled_order: str = "semantic_first"
    coupled_max_chars: int = 4000

    def validate(self) -> None:
        if self.coupled_order not in ("semantic_first", "source_first"):
            raise ConfigError("representation.coupled_order must be 'semantic_first' or 'source_first'")
        if int(self.coupled_max_chars) < 100:
            raise ConfigError("representation.coupled_max_chars must be >= 100")
        for name in ("raw_max_chars", "semantic_max_chars", "compiler_max_chars"):
            if int(getattr(self, name)) < 50:
                raise ConfigError(f"representation.{name} must be >= 50")
        if self.missing_semantic not in MISSING_SEMANTIC_POLICIES:
            raise ConfigError(
                f"representation.missing_semantic must be one of {MISSING_SEMANTIC_POLICIES}"
            )


@dataclass(frozen=True)
class RetrievalConfig:
    """First-stage retrieval: which indexes, how they are queried and fused."""

    representations: Tuple[str, ...] = ("raw",)
    dense: bool = True
    bm25: bool = False
    exact_match: bool = False
    candidate_k: int = 50            # candidates drawn from every index/list
    rrf_k: int = 60                  # RRF damping constant
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    exact_weight: float = 1.0
    # Optional per-representation RRF weights, e.g. {"semantic": 1.5}.
    representation_weights: Dict[str, float] = field(default_factory=dict)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # Model-specific instruction prefixes (e.g. nomic's "search_query: ").
    query_prefix: str = ""
    document_prefix: str = ""

    def validate(self) -> None:
        if not self.representations:
            raise ConfigError("retrieval.representations must not be empty")
        for rep in self.representations:
            if rep not in REPRESENTATIONS:
                raise ConfigError(f"unknown representation {rep!r}; expected {REPRESENTATIONS}")
        if len(set(self.representations)) != len(self.representations):
            raise ConfigError("retrieval.representations contains duplicates")
        if not (self.dense or self.bm25 or self.exact_match):
            raise ConfigError("enable at least one of dense / bm25 / exact_match")
        if self.candidate_k < 1 or self.rrf_k < 1:
            raise ConfigError("candidate_k and rrf_k must be >= 1")
        for name in ("dense_weight", "bm25_weight", "exact_weight"):
            if float(getattr(self, name)) < 0:
                raise ConfigError(f"retrieval.{name} must be >= 0")
        for rep, weight in self.representation_weights.items():
            if rep not in REPRESENTATIONS:
                raise ConfigError(f"representation_weights has unknown key {rep!r}")
            if float(weight) < 0:
                raise ConfigError("representation weights must be >= 0")
        if self.bm25_k1 <= 0 or not 0 <= self.bm25_b <= 1:
            raise ConfigError("bm25_k1 must be > 0 and bm25_b in [0, 1]")


@dataclass(frozen=True)
class GraphConfig:
    """Bounded compiler-graph expansion of the first-stage ranking."""

    enabled: bool = False
    relations: Tuple[str, ...] = ("callers", "callees")
    max_depth: int = 1
    seed_k: int = 10                 # top-ranked candidates used as seeds
    max_added: int = 20              # cap on graph-only candidates
    max_neighbors_per_node: int = 20  # fan-out cap per visited node
    graph_weight: float = 0.5        # RRF weight of the graph ordering

    def validate(self) -> None:
        for rel in self.relations:
            if rel not in GRAPH_RELATIONS:
                raise ConfigError(f"unknown graph relation {rel!r}; expected {GRAPH_RELATIONS}")
        if self.enabled and not self.relations:
            raise ConfigError("graph.enabled requires at least one relation")
        if not 1 <= self.max_depth <= 4:
            raise ConfigError("graph.max_depth must be in 1..4 (traversal must stay bounded)")
        if self.seed_k < 1 or self.max_added < 0 or self.max_neighbors_per_node < 1:
            raise ConfigError("graph.seed_k >= 1, max_added >= 0, max_neighbors_per_node >= 1")
        if self.graph_weight < 0:
            raise ConfigError("graph.graph_weight must be >= 0")


@dataclass(frozen=True)
class RerankConfig:
    """Optional cross-encoder reranking of a candidate pool."""

    enabled: bool = False
    model: str = "ms-marco-minilm"   # alias from rerank.RERANKER_REGISTRY or an HF id
    pool_size: int = 30
    batch_size: int = 16
    device: str = "auto"
    max_length: int = 512
    text_representations: Tuple[str, ...] = ("raw",)
    max_chars: int = 1500

    def validate(self) -> None:
        if self.pool_size < 1 or self.batch_size < 1 or self.max_length < 16:
            raise ConfigError("rerank.pool_size/batch_size >= 1 and max_length >= 16")
        for rep in self.text_representations:
            if rep not in REPRESENTATIONS:
                raise ConfigError(f"rerank.text_representations has unknown entry {rep!r}")
        if not self.text_representations:
            raise ConfigError("rerank.text_representations must not be empty")


@dataclass(frozen=True)
class RoutingConfig:
    """Rule-based query routing (see ``routing.py``): per-class fusion-weight profiles."""

    enabled: bool = False
    # class -> {dense_weight, bm25_weight, exact_weight, graph_weight, representation_weights}
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        from .routing import CLASSES, PROFILE_KEYS
        for cls_name, profile in self.profiles.items():
            if cls_name not in CLASSES:
                raise ConfigError(f"routing.profiles has unknown class {cls_name!r}; expected {CLASSES}")
            for key, value in dict(profile).items():
                if key not in PROFILE_KEYS:
                    raise ConfigError(f"routing.profiles[{cls_name!r}] has unknown key {key!r}; expected {PROFILE_KEYS}")
                values = dict(value).values() if key == "representation_weights" else [value]
                if any(float(v) < 0 for v in values):
                    raise ConfigError("routing weights must be >= 0")


_SECTIONS: Dict[str, type] = {
    "representation": RepresentationConfig,
    "retrieval": RetrievalConfig,
    "graph": GraphConfig,
    "rerank": RerankConfig,
    "routing": RoutingConfig,
}
_TUPLE_FIELDS = {
    ("retrieval", "representations"), ("graph", "relations"),
    ("rerank", "text_representations"),
}
# Keys that identify/describe a run but do not change what it computes.
_NON_FUNCTIONAL = ("name", "description", "tags", "base_json", "enriched_json",
                   "enrichment_model")


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    description: str = ""
    base_json: Optional[str] = None
    enriched_json: Optional[str] = None
    enrichment_model: str = "none"
    embedding_model: str = "bge-small"
    top_k: int = 10
    seed: int = 0
    tags: Tuple[str, ...] = ()
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    schema_version: int = CONFIG_SCHEMA_VERSION

    # ------------------------------------------------------------------ checks
    def validate(self) -> "PipelineConfig":
        if not self.name or not str(self.name).strip():
            raise ConfigError("pipeline name must not be empty")
        if self.top_k < 1:
            raise ConfigError("top_k must be >= 1")
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigError(
                f"unsupported config schema_version {self.schema_version}; "
                f"this code reads {CONFIG_SCHEMA_VERSION}"
            )
        self.representation.validate()
        self.retrieval.validate()
        self.graph.validate()
        self.rerank.validate()
        self.routing.validate()
        needs_enrichment = ("semantic" in self.retrieval.representations or "coupled" in self.retrieval.representations) or (
            self.rerank.enabled and "semantic" in self.rerank.text_representations
        )
        if needs_enrichment and not self.enriched_json:
            raise ConfigError(
                f"pipeline {self.name!r} uses the semantic representation but "
                "enriched_json is not set"
            )
        if self.rerank.enabled and self.rerank.pool_size < self.top_k:
            raise ConfigError("rerank.pool_size must be >= top_k")
        return self

    # ----------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for section, name in _TUPLE_FIELDS:
            data[section][name] = list(data[section][name])
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PipelineConfig":
        if not isinstance(data, Mapping):
            raise ConfigError("pipeline configuration must be a JSON object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown pipeline config key(s): {unknown}")
        if "name" not in data:
            raise ConfigError("pipeline configuration requires a 'name'")
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key in _SECTIONS:
                kwargs[key] = _build_section(key, value)
            elif key == "tags":
                kwargs[key] = _tuple_of_str(value, "tags")
            else:
                kwargs[key] = value
        try:
            return cls(**kwargs).validate()
        except TypeError as exc:
            raise ConfigError(f"invalid pipeline configuration: {exc}") from exc

    def config_hash(self) -> str:
        """Stable 12-hex digest of everything that changes what is computed."""
        payload = self.to_dict()
        for key in _NON_FUNCTIONAL:
            payload.pop(key, None)
        if "coupled" not in payload.get("retrieval", {}).get("representations", []):
            for key in ("coupled_order", "coupled_max_chars"):    # irrelevant unless the coupled chunk is used
                payload.get("representation", {}).pop(key, None)
        if not payload.get("routing", {}).get("enabled"):
            payload.pop("routing", None)       # keeps the hash of every pre-routing configuration unchanged
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def replaced(self, overrides: Mapping[str, Any]) -> "PipelineConfig":
        """Return a validated copy with dotted-key overrides, e.g. ``retrieval.bm25``."""
        return PipelineConfig.from_dict(apply_overrides(self.to_dict(), overrides))


def _build_section(key: str, value: Any):
    section_cls = _SECTIONS[key]
    if isinstance(value, section_cls):
        return value
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{key}' must be a JSON object")
    valid = {f.name: f for f in fields(section_cls)}
    unknown = sorted(set(value) - set(valid))
    if unknown:
        raise ConfigError(f"unknown key(s) in '{key}': {unknown}")
    kwargs = {}
    for name, item in value.items():
        if (key, name) in _TUPLE_FIELDS:
            item = _tuple_of_str(item, f"{key}.{name}")
        elif name == "representation_weights":
            item = {str(k): float(v) for k, v in dict(item).items()}
        kwargs[name] = item
    try:
        return section_cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"invalid '{key}' section: {exc}") from exc


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply ``{"retrieval.bm25": True, "top_k": 5}`` to a config dictionary."""
    result = copy.deepcopy(dict(config))
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        target = result
        for part in parts[:-1]:
            nxt = target.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                target[part] = nxt
            target = nxt
        target[parts[-1]] = copy.deepcopy(value)
    return result


# --------------------------------------------------------------- named pipelines
def baseline_rag(base_json: str, **overrides: Any) -> PipelineConfig:
    """Control group: raw representation, dense similarity only."""
    config = PipelineConfig(
        name="baseline_rag",
        description="Raw compiler/source representation, dense retrieval only.",
        base_json=str(base_json),
        retrieval=RetrievalConfig(representations=("raw",), dense=True),
        tags=("pipeline",),
    )
    return config.replaced(overrides) if overrides else config.validate()


def semantic_enrichment_rag(base_json: str, enriched_json: str, enrichment_model: str = "",
                            **overrides: Any) -> PipelineConfig:
    """LLM-enriched semantic representation, dense retrieval (BM25 optional)."""
    config = PipelineConfig(
        name="semantic_enrichment_rag",
        description="LLM-enriched semantic representation, dense retrieval.",
        base_json=str(base_json),
        enriched_json=str(enriched_json),
        enrichment_model=enrichment_model or Path(str(enriched_json)).stem,
        retrieval=RetrievalConfig(representations=("semantic",), dense=True),
        tags=("pipeline",),
    )
    return config.replaced(overrides) if overrides else config.validate()


def compiler_hybrid_rag(base_json: str, enriched_json: str, enrichment_model: str = "",
                        **overrides: Any) -> PipelineConfig:
    """Raw+semantic+compiler indexes, dense+BM25+RRF, graph expansion, optional rerank."""
    config = PipelineConfig(
        name="compiler_hybrid_rag",
        description=("Multi-representation dense+BM25 retrieval fused with RRF, "
                     "bounded call-graph expansion and optional cross-encoder reranking."),
        base_json=str(base_json),
        enriched_json=str(enriched_json),
        enrichment_model=enrichment_model or Path(str(enriched_json)).stem,
        retrieval=RetrievalConfig(
            representations=("raw", "semantic", "compiler"), dense=True, bm25=True,
            exact_match=True,
        ),
        graph=GraphConfig(enabled=True, relations=("callers", "callees"), max_depth=1),
        rerank=RerankConfig(enabled=False),
        tags=("pipeline",),
    )
    return config.replaced(overrides) if overrides else config.validate()


NAMED_PIPELINES = {
    "baseline_rag": baseline_rag,
    "semantic_enrichment_rag": semantic_enrichment_rag,
    "compiler_hybrid_rag": compiler_hybrid_rag,
}


def load_config_file(path: Path) -> PipelineConfig:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"configuration {path} is not valid JSON: {exc}") from exc
    return PipelineConfig.from_dict(data)
