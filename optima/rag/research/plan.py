"""Declarative ablation plan: which configurations to run, and against which reference.

Every :class:`RunSpec` carries the ``reference`` label it should be compared to
(paired, on identical queries).  Groups isolate one factor at a time, holding the
rest of the configuration constant:

``pipelines``       the three named pipelines (+ reranked hybrid)
``enrichment``      one semantic/hybrid run per enrichment model vs the raw baseline
``representations`` all 7 non-empty subsets of raw/semantic/compiler x {dense, bm25, dense+bm25}
``hybrid``          dense vs BM25 vs RRF vs RRF+exact per representation, plus rrf_k sensitivity
``graph``           none / callees / callers / calls / deeper / dependencies / graph-weight sweep
``rerank``          initial / rerank / graph / graph+rerank for each reranker
``embedding``       the hybrid pipeline across embedding models
``improved``        compiler_hybrid_rag with rule-based query routing (``routing_profiles`` of the suite
                    spec) and its matched no-enrichment counterpart; combine with ``embedding_prefixes``
``attribution``     2x2x2 factorial inside the routed stack: LLVM compiler representation x LLVM call-graph
                    expansion x LLM enrichment text (raw source is always present), for contribution analysis
``coupled``         one chunk per function holding enrichment AND source (enrichment-first and source-first) vs the
                    same information in separate raw / semantic indexes, dense-only and hybrid
``grid``            fully crossed enrichment model x embedding model x {semantic dense,
                    semantic hybrid, compiler hybrid}, each with matched no-enrichment
                    counterparts (raw baseline, raw hybrid, raw+compiler hybrid) per embedding
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .config import (GraphConfig, PipelineConfig, RepresentationConfig, RerankConfig,
                     RetrievalConfig)

ALL_GROUPS = ("pipelines", "enrichment", "representations", "hybrid", "graph", "rerank",
              "embedding", "grid", "improved", "attribution", "coupled")
# Groups run by a suite that does not name its groups explicitly (grid is opt-in: it is large).
DEFAULT_GROUPS = tuple(group for group in ALL_GROUPS if group not in ("grid", "improved", "attribution", "coupled"))
REPS = ("raw", "semantic", "compiler")


@dataclass
class RunSpec:
    group: str
    label: str
    config: PipelineConfig
    reference: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._+-]+", "-", str(value)).strip("-").lower()


def discover_enriched(directory: Path) -> Dict[str, Path]:
    """``enhanced_<slug>.json`` files in a directory, keyed by slug."""
    return {p.stem[len("enhanced_"):]: p for p in sorted(Path(directory).glob("enhanced_*.json"))}


def enriched_base_mismatch(path: Path, base_sha256: str) -> Optional[str]:
    """Why an enriched file does not belong to this ``base.json`` (None when it does).

    The enrichment run records the SHA-256 of the ``base.json`` it was produced from.  A file
    for another project (e.g. an xxHash enrichment saved under the cJSON directory) would join
    by function ID to nothing and silently yield "no enrichment" runs, so it is rejected here.
    Files that do not record a hash are accepted (nothing to check).
    """
    import json
    try:
        with open(path, "r", encoding="utf-8") as handle:
            experiment = (json.load(handle) or {}).get("experiment") or {}
    except (OSError, ValueError) as exc:
        return f"unreadable: {type(exc).__name__}: {exc}"
    recorded = experiment.get("base_sha256")
    if recorded and recorded != base_sha256:
        return (f"enriched from a different base.json (recorded sha {str(recorded)[:12]}, "
                f"dataset {experiment.get('dataset')!r}; expected {base_sha256[:12]})")
    return None


class PlanBuilder:
    def __init__(self, base_json: str, enriched: Mapping[str, str], primary_enrichment: str,
                 embedding_models: Sequence[str], primary_embedding: str,
                 rerankers: Sequence[str] = (), prefixes: Optional[Mapping[str, Mapping[str, str]]] = None,
                 top_k: int = 10, rerank_pool: int = 30,
                 routing_profiles: Optional[Mapping[str, Mapping[str, Any]]] = None):
        if primary_enrichment not in enriched:
            raise ValueError(f"primary enrichment {primary_enrichment!r} not among {sorted(enriched)}")
        if primary_embedding not in embedding_models:
            raise ValueError("primary_embedding must be one of embedding_models")
        self.base_json, self.enriched = str(base_json), {k: str(v) for k, v in enriched.items()}
        self.primary_enr, self.embeddings, self.primary_emb = primary_enrichment, list(embedding_models), primary_embedding
        self.rerankers, self.prefixes = list(rerankers), dict(prefixes or {})
        self.top_k, self.rerank_pool = top_k, rerank_pool
        self.routing_profiles = {k: dict(v) for k, v in (routing_profiles or {}).items()}
        self.specs: List[RunSpec] = []
        self._labels: set = set()

    # ------------------------------------------------------------- helpers
    def _config(self, name: str, enr: Optional[str], emb: str, *, reps: Sequence[str],
                dense: bool = True, bm25: bool = False, exact: bool = False,
                graph: Optional[Mapping[str, Any]] = None, rerank: Optional[Mapping[str, Any]] = None,
                retrieval_extra: Optional[Mapping[str, Any]] = None, tags: Sequence[str] = (),
                routing: bool = False, representation_extra: Optional[Mapping[str, Any]] = None) -> PipelineConfig:
        prefix = self.prefixes.get(emb, {})
        retrieval = dict(representations=tuple(reps), dense=dense, bm25=bm25, exact_match=exact,
                         query_prefix=prefix.get("query", ""), document_prefix=prefix.get("document", ""))
        retrieval.update(retrieval_extra or {})
        relations = (graph or {}).get("relations", ("callers", "callees")) if (graph or {}).get("enabled") else ()
        needs_enr = bool(enr) and ("semantic" in reps or "coupled" in reps
                                   or (rerank and "semantic" in rerank.get("text_representations", ()))
                                   or "enriched_dependencies" in relations)
        return PipelineConfig.from_dict({
            "name": name, "base_json": self.base_json,
            "enriched_json": self.enriched[enr] if needs_enr else None,
            "enrichment_model": enr if needs_enr else "none", "embedding_model": emb,
            "top_k": self.top_k,
            "retrieval": retrieval,
            **({"representation": dict(representation_extra)} if representation_extra else {}),
            "graph": dict(graph or {"enabled": False}),
            "rerank": dict(rerank or {"enabled": False}),
            "routing": ({"enabled": True, "profiles": self.routing_profiles} if routing else {"enabled": False}),
            "tags": list(tags),
        })

    def _add(self, group: str, variant: str, config: PipelineConfig, reference: Optional[str],
             **tags: Any) -> str:
        suffix = ""
        if config.retrieval.dense or config.rerank.enabled:
            suffix += f"|emb={slug(config.embedding_model)}"
        if config.enriched_json:
            suffix += f"|enr={slug(config.enrichment_model)}"
        label = f"{group}/{variant}{suffix}"
        if label in self._labels:
            return label
        self._labels.add(label)
        self.specs.append(RunSpec(group, label, config, reference, dict(tags, variant=variant)))
        return label

    # --------------------------------------------------------------- groups
    def pipelines(self) -> None:
        for emb in self.embeddings:
            enr = self.primary_enr
            ref = self._add("pipelines", f"baseline_rag",
                            self._config("baseline_rag", None, emb, reps=("raw",)), None)
            self._add("pipelines", f"semantic_enrichment_rag",
                      self._config("semantic_enrichment_rag", enr, emb, reps=("semantic",)), ref)
            self._add("pipelines", f"compiler_hybrid_rag",
                      self._config("compiler_hybrid_rag", enr, emb, reps=REPS, bm25=True, exact=True,
                                   graph={"enabled": True}), ref)
            for rr in self.rerankers:
                self._add("pipelines", f"compiler_hybrid_rag+rerank({rr})",
                          self._config("compiler_hybrid_rag", enr, emb, reps=REPS, bm25=True, exact=True,
                                       graph={"enabled": True},
                                       rerank={"enabled": True, "model": rr, "pool_size": self.rerank_pool}), ref)

    def enrichment(self) -> None:
        emb = self.primary_emb
        ref = self._add("enrichment", f"raw_baseline",
                        self._config("baseline_rag", None, emb, reps=("raw",)), None)
        for enr in sorted(self.enriched):
            self._add("enrichment", f"semantic_dense",
                      self._config("semantic_enrichment_rag", enr, emb, reps=("semantic",)), ref)
            self._add("enrichment", f"semantic_hybrid",
                      self._config("semantic_enrichment_rag", enr, emb, reps=("semantic",), bm25=True), ref)
            self._add("enrichment", f"raw+semantic_dense",
                      self._config("ablation", enr, emb, reps=("raw", "semantic")), ref)
            self._add("enrichment", f"compiler_hybrid_rag",
                      self._config("compiler_hybrid_rag", enr, emb, reps=REPS, bm25=True, exact=True,
                                   graph={"enabled": True}), ref)

    def representations(self) -> None:
        emb, enr = self.primary_emb, self.primary_enr
        modes = {"dense": dict(dense=True, bm25=False), "bm25": dict(dense=False, bm25=True),
                 "dense+bm25": dict(dense=True, bm25=True)}
        for mode, flags in modes.items():
            ref = None
            for size in (1, 2, 3):
                for combo in itertools.combinations(REPS, size):
                    label = self._add("representations", f"{mode}:{'+'.join(combo)}",
                                      self._config("ablation", enr, emb, reps=combo, **flags),
                                      ref, mode=mode, representations="+".join(combo))
                    if combo == ("raw",):
                        ref = label

    def hybrid(self) -> None:
        emb, enr = self.primary_emb, self.primary_enr
        for rep in ("raw", "semantic", "compiler"):
            ref = self._add("hybrid", f"{rep}:dense", self._config("ablation", enr, emb, reps=(rep,)), None)
            self._add("hybrid", f"{rep}:bm25", self._config("ablation", enr, emb, reps=(rep,), dense=False, bm25=True), ref)
            self._add("hybrid", f"{rep}:dense+bm25(rrf)", self._config("ablation", enr, emb, reps=(rep,), bm25=True), ref)
            self._add("hybrid", f"{rep}:dense+bm25+exact",
                      self._config("ablation", enr, emb, reps=(rep,), bm25=True, exact=True), ref)
            for k in (10, 100):
                self._add("hybrid", f"{rep}:dense+bm25(rrf_k={k})",
                          self._config("ablation", enr, emb, reps=(rep,), bm25=True,
                                       retrieval_extra={"rrf_k": k}), ref)

    def graph(self) -> None:
        emb, enr = self.primary_emb, self.primary_enr

        def hyb(graph, variant, ref, **kw):
            return self._add("graph", f"{variant}",
                             self._config("compiler_hybrid_rag", enr, emb, reps=REPS, bm25=True, exact=True,
                                          graph=graph, **kw), ref)
        ref = hyb({"enabled": False}, "no_graph", None)
        hyb({"enabled": True, "relations": ["callees"]}, "callees_d1", ref)
        hyb({"enabled": True, "relations": ["callers"]}, "callers_d1", ref)
        hyb({"enabled": True, "relations": ["callers", "callees"]}, "calls_d1", ref)
        hyb({"enabled": True, "relations": ["callers", "callees"], "max_depth": 2}, "calls_d2", ref)
        hyb({"enabled": True, "relations": ["dependencies"]}, "dependencies_d1", ref)
        hyb({"enabled": True, "relations": ["enriched_dependencies"]}, "enriched_dependencies_d1", ref)
        hyb({"enabled": True, "relations": ["callers", "callees", "dependencies", "enriched_dependencies"]},
            "combined_d1", ref)
        hyb({"enabled": True, "relations": ["callers", "callees", "dependencies", "enriched_dependencies"],
             "max_depth": 2}, "combined_d2", ref)
        for weight in (0.1, 0.25, 1.0):
            hyb({"enabled": True, "relations": ["callers", "callees"], "graph_weight": weight},
                f"calls_d1_weight{weight}", ref)
        # Graph expansion on the simplest retriever, isolating it from multi-index fusion.
        simple_ref = self._add("graph", f"raw_dense:no_graph",
                               self._config("ablation", None, emb, reps=("raw",)), None)
        self._add("graph", f"raw_dense:calls_d1",
                  self._config("ablation", None, emb, reps=("raw",), graph={"enabled": True}), simple_ref)

    def rerank(self) -> None:
        emb, enr = self.primary_emb, self.primary_enr
        base_kwargs = dict(reps=REPS, bm25=True, exact=True)
        for rr in self.rerankers:
            ref = self._add("rerank", f"initial",
                            self._config("compiler_hybrid_rag", enr, emb, **base_kwargs), None)
            spec = {"enabled": True, "model": rr, "pool_size": self.rerank_pool}
            self._add("rerank", f"graph_only", self._config(
                "compiler_hybrid_rag", enr, emb, graph={"enabled": True}, **base_kwargs), ref)
            self._add("rerank", f"rerank({rr})", self._config(
                "compiler_hybrid_rag", enr, emb, rerank=spec, **base_kwargs), ref)
            self._add("rerank", f"graph+rerank({rr})", self._config(
                "compiler_hybrid_rag", enr, emb, graph={"enabled": True}, rerank=spec, **base_kwargs), ref)
            self._add("rerank", f"rerank({rr},raw+semantic)", self._config(
                "compiler_hybrid_rag", enr, emb,
                rerank=dict(spec, text_representations=["raw", "semantic"]), **base_kwargs), ref)
            # Reranking the weakest retriever shows whether rerank helps when recall@pool is low.
            r0 = self._add("rerank", f"raw_dense:initial",
                           self._config("ablation", None, emb, reps=("raw",)), None)
            self._add("rerank", f"raw_dense:rerank({rr})", self._config(
                "ablation", None, emb, reps=("raw",), rerank=spec), r0)

    def embedding(self) -> None:
        enr = self.primary_enr
        ref = None
        for emb in self.embeddings:
            label = self._add("embedding", f"compiler_hybrid_rag",
                              self._config("compiler_hybrid_rag", enr, emb, reps=REPS, bm25=True,
                                           exact=True, graph={"enabled": True}), ref)
            ref = ref or label
            self._add("embedding", f"baseline_rag",
                      self._config("baseline_rag", None, emb, reps=("raw",)), None)

    def grid(self) -> None:
        """Fully crossed enrichment x embedding grid with matched no-enrichment counterparts.

        Per embedding: the raw baseline, ``raw_hybrid`` (counterpart of the semantic hybrid:
        same fusion, raw text) and ``raw+compiler_hybrid_rag`` (counterpart of the compiler
        hybrid pipeline with the semantic index removed).  The counterpart is recorded in
        ``tags['counterpart']`` so the enrichment effect inside a pipeline is a matched ablation.
        """
        for emb in self.embeddings:
            ref = self._add("grid", "raw_baseline",
                            self._config("baseline_rag", None, emb, reps=("raw",)), None)
            self._add("grid", "raw_hybrid",
                      self._config("ablation", None, emb, reps=("raw",), bm25=True), ref)
            self._add("grid", "raw+compiler_hybrid_rag",
                      self._config("ablation", None, emb, reps=("raw", "compiler"), bm25=True,
                                   exact=True, graph={"enabled": True}), ref)
            for enr in sorted(self.enriched):
                self._add("grid", "semantic_dense",
                          self._config("semantic_enrichment_rag", enr, emb, reps=("semantic",)),
                          ref, counterpart="raw_baseline")
                self._add("grid", "semantic_hybrid",
                          self._config("semantic_enrichment_rag", enr, emb, reps=("semantic",),
                                       bm25=True), ref, counterpart="raw_hybrid")
                self._add("grid", "compiler_hybrid_rag",
                          self._config("compiler_hybrid_rag", enr, emb, reps=REPS, bm25=True,
                                       exact=True, graph={"enabled": True}), ref,
                          counterpart="raw+compiler_hybrid_rag")

    def improved(self) -> None:
        """Routed compiler stack + matched no-enrichment counterpart, per embedding (and enrichment) model."""
        if not self.routing_profiles:
            raise ValueError("group 'improved' needs routing_profiles in the suite spec")
        for emb in self.embeddings:
            ref = self._add("improved", "raw_baseline", self._config("baseline_rag", None, emb, reps=("raw",)), None)
            self._add("improved", "raw+compiler_hybrid_routed",
                      self._config("ablation", None, emb, reps=("raw", "compiler"), bm25=True, exact=True,
                                   graph={"enabled": True}, routing=True), ref)
            for enr in sorted(self.enriched):
                self._add("improved", "compiler_hybrid_routed",
                          self._config("compiler_hybrid_rag", enr, emb, reps=REPS, bm25=True, exact=True,
                                       graph={"enabled": True}, routing=True), ref,
                          counterpart="raw+compiler_hybrid_routed")

    def coupled(self) -> None:
        """Coupled chunk (enrichment + source in ONE document) vs separate raw / semantic indexes."""
        for emb in self.embeddings:
            ref = self._add("coupled", "raw_baseline", self._config("baseline_rag", None, emb, reps=("raw",)), None)
            for enr in sorted(self.enriched):
                dense_kw, hybrid_kw = dict(), dict(bm25=True, exact=True)
                self._add("coupled", "separate_dense", self._config("ablation", enr, emb, reps=("raw", "semantic"), **dense_kw), ref)
                self._add("coupled", "coupled_dense", self._config("ablation", enr, emb, reps=("coupled",), **dense_kw), ref,
                          counterpart="separate_dense")
                self._add("coupled", "coupled_dense_src", self._config("ablation", enr, emb, reps=("coupled",),
                                                                        representation_extra={"coupled_order": "source_first"}, **dense_kw), ref,
                          counterpart="coupled_dense")
                self._add("coupled", "separate_hybrid", self._config("ablation", enr, emb, reps=("raw", "semantic"), **hybrid_kw), ref)
                self._add("coupled", "coupled_hybrid", self._config("ablation", enr, emb, reps=("coupled",), **hybrid_kw), ref,
                          counterpart="separate_hybrid")
                self._add("coupled", "coupled_hybrid_src", self._config("ablation", enr, emb, reps=("coupled",),
                                                                         representation_extra={"coupled_order": "source_first"}, **hybrid_kw), ref,
                          counterpart="coupled_hybrid")

    def attribution(self) -> None:
        """All 8 combinations of {compiler text, call-graph expansion, enrichment text} on top of raw source, all routed.

        Raw source (dense + BM25 + exact match) is always present; routing and the query instruction come from the
        suite spec.  Without enrichment the configuration does not depend on an enrichment model (built once per
        embedding); with enrichment it is built per enrichment model.
        """
        if not self.routing_profiles:
            raise ValueError("group 'attribution' needs routing_profiles in the suite spec")
        for emb in self.embeddings:
            ref = self._add("attribution", "raw_baseline", self._config("baseline_rag", None, emb, reps=("raw",)), None)
            for compiler in (False, True):
                for graph in (False, True):
                    reps = ("raw", "compiler") if compiler else ("raw",)
                    self._add("attribution", f"routed:{'+'.join(reps)}{'+graph' if graph else ''}",
                              self._config("ablation", None, emb, reps=reps, bm25=True, exact=True,
                                           graph={"enabled": True} if graph else None, routing=True),
                              ref, factors=f"E0C{int(compiler)}G{int(graph)}")
            for enr in sorted(self.enriched):
                for compiler in (False, True):
                    for graph in (False, True):
                        reps = ("raw", "semantic", "compiler") if compiler else ("raw", "semantic")
                        self._add("attribution", f"routed:{'+'.join(reps)}{'+graph' if graph else ''}",
                                  self._config("ablation", enr, emb, reps=reps, bm25=True, exact=True,
                                               graph={"enabled": True} if graph else None, routing=True),
                                  ref, factors=f"E1C{int(compiler)}G{int(graph)}")

    def build(self, groups: Sequence[str]) -> List[RunSpec]:
        for group in groups:
            if group not in ALL_GROUPS:
                raise ValueError(f"unknown ablation group {group!r}; choose from {ALL_GROUPS}")
            getattr(self, group)()
        return self.specs
