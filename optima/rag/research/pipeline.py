"""The configurable research retrieval pipeline.

Stages (each independently switchable through :class:`PipelineConfig`)::

    query -> normalise
          -> per representation: dense list and/or BM25 list   (retrieval.*)
          -> exact symbol list                                   (retrieval.exact_match)
          -> RRF fusion of all lists                             (first-stage ranking)
          -> graph expansion + RRF with the graph ordering       (graph.enabled)
          -> cross-encoder rerank of the top pool                (rerank.enabled)
          -> top-k with provenance

Disabling a stage removes only that stage; the ranking it would have received is
passed through unchanged, so e.g. ``graph.enabled=false`` yields exactly the
first-stage ranking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import REPRESENTATIONS, PipelineConfig
from .corpus import Corpus, load_corpus
from .dense import DenseIndex, EmbeddingCache, Encoder, get_encoder
from .fusion import FusedItem, RankedList, rrf_fuse
from .graph import GraphCandidate, GraphIndex
from .lexical import BM25Index, ExactSymbolMatcher, normalize_query
from .representations import build_texts
from .rerank import CrossEncoderReranker, RerankerError

logger = logging.getLogger(__name__)


@dataclass
class ResultItem:
    function_id: str
    rank: int
    score: float
    score_type: str                          # "rrf" or "cross_encoder"
    first_stage_rank: Optional[int] = None   # rank before graph/rerank (None: graph-only)
    first_stage_score: Optional[float] = None
    pre_rerank_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)   # retriever provenance
    representations: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)              # dense / bm25 / exact / graph
    graph_evidence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class QueryResult:
    query: str
    results: List[ResultItem] = field(default_factory=list)
    timings_ms: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    rerank_applied: bool = False
    rerank_error: Optional[str] = None
    # Candidate-pool IDs kept for pool-recall analysis; not serialised per query.
    pool_before_graph: List[str] = field(default_factory=list, repr=False)
    pool_after_graph: List[str] = field(default_factory=list, repr=False)


def _contribution_dict(c) -> Dict[str, Any]:
    return {"source": c.source, "rank": c.rank, "score": round(c.score, 6),
            "rrf": round(c.rrf, 8)}


class ResearchPipeline:
    def __init__(self, config: PipelineConfig, corpus: Optional[Corpus] = None,
                 encoder: Optional[Encoder] = None,
                 reranker: Optional[CrossEncoderReranker] = None,
                 embedding_cache: Optional[EmbeddingCache] = None,
                 graph: Optional[GraphIndex] = None):
        self.config = config.validate()
        self.corpus = corpus
        self.encoder = encoder
        self.reranker = reranker
        self.embedding_cache = embedding_cache
        self.graph = graph
        self.dense: Dict[str, DenseIndex] = {}
        self.bm25: Dict[str, BM25Index] = {}
        self.exact: Optional[ExactSymbolMatcher] = None
        self.texts: Dict[str, Dict[str, str]] = {}
        self.build_info: Dict[str, Any] = {}
        self._built = False

    # ------------------------------------------------------------------ build
    def build(self) -> Dict[str, Any]:
        cfg = self.config
        timings: Dict[str, float] = {}
        t_total = time.perf_counter()
        if self.corpus is None:
            t0 = time.perf_counter()
            if not cfg.base_json:
                raise ValueError(f"pipeline {cfg.name!r} has no base_json to load")
            self.corpus = load_corpus(Path(cfg.base_json),
                                      Path(cfg.enriched_json) if cfg.enriched_json else None)
            timings["load_corpus_s"] = time.perf_counter() - t0
        corpus = self.corpus

        needed_texts = set(cfg.retrieval.representations)
        if cfg.rerank.enabled:
            needed_texts |= set(cfg.rerank.text_representations)
        t0 = time.perf_counter()
        for rep in sorted(needed_texts, key=lambda r: REPRESENTATIONS.index(r)):
            self.texts[rep] = build_texts(corpus, rep, cfg.representation)
        timings["build_texts_s"] = time.perf_counter() - t0

        index_info: Dict[str, Any] = {}
        if cfg.retrieval.dense:
            if self.encoder is None:
                self.encoder = get_encoder(
                    cfg.embedding_model, query_prefix=cfg.retrieval.query_prefix,
                    document_prefix=cfg.retrieval.document_prefix)
            for rep in cfg.retrieval.representations:
                ids = list(self.texts[rep])
                index = DenseIndex.build(ids, [self.texts[rep][i] for i in ids],
                                         self.encoder, self.embedding_cache)
                self.dense[rep] = index
                index_info[f"dense:{rep}"] = index.build_info
            timings["embed_s"] = sum(i.build_info["embed_seconds"] for i in self.dense.values())
        if cfg.retrieval.bm25:
            t0 = time.perf_counter()
            for rep in cfg.retrieval.representations:
                ids = list(self.texts[rep])
                self.bm25[rep] = BM25Index(cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b).build(
                    ids, [self.texts[rep][i] for i in ids])
                index_info[f"bm25:{rep}"] = {
                    "documents": len(ids), "index_bytes": self.bm25[rep].approx_size_bytes()}
            timings["bm25_build_s"] = time.perf_counter() - t0
        if cfg.retrieval.exact_match:
            self.exact = ExactSymbolMatcher(
                (r.id, r.name, r.qualified_name) for r in corpus.records.values())
        if cfg.graph.enabled and self.graph is None:
            t0 = time.perf_counter()
            self.graph = GraphIndex(corpus)
            timings["graph_build_s"] = time.perf_counter() - t0
        if cfg.rerank.enabled and self.reranker is None:
            self.reranker = CrossEncoderReranker(
                cfg.rerank.model, device=cfg.rerank.device, batch_size=cfg.rerank.batch_size,
                max_length=cfg.rerank.max_length)
        if cfg.rerank.enabled:
            self.reranker.load()  # fail fast, before any query is evaluated
            timings["reranker_load_s"] = self.reranker.load_seconds or 0.0
        timings["build_total_s"] = time.perf_counter() - t_total
        self.build_info = {
            "timings": timings,
            "indexes": index_info,
            "index_bytes_total": sum(i.get("index_bytes", 0) for i in index_info.values()),
            "documents": len(corpus),
            "graph": dict(self.graph.stats) if self.graph else None,
            "encoder": dict(getattr(self.encoder, "info", {})) if self.encoder else None,
            "reranker": self.reranker.info if self.reranker else None,
        }
        self._built = True
        return self.build_info

    # ----------------------------------------------------------------- search
    def search(self, query: str) -> QueryResult:
        if not self._built:
            self.build()
        cfg, ret = self.config, self.config.retrieval
        timings: Dict[str, float] = {}
        t_start = time.perf_counter()
        text = normalize_query(query)
        if not text:
            return QueryResult(query=str(query or ""), error="empty_query")
        profile: Dict[str, Any] = {}
        route = None
        if cfg.routing.enabled:
            from .routing import classify_query, profile_for
            route = classify_query(text)
            profile = profile_for(route, cfg.routing.profiles)
        try:
            lists = self._first_stage_lists(text, timings, profile)
        except Exception as exc:  # an embedding/runtime failure must not hide the query
            logger.error("retrieval failed for %r: %s", text[:80], exc)
            return QueryResult(query=text, error=f"{type(exc).__name__}: {exc}",
                               timings_ms={"total": (time.perf_counter() - t_start) * 1000})

        t0 = time.perf_counter()
        fused = rrf_fuse(lists, ret.rrf_k)
        timings["fuse"] = (time.perf_counter() - t0) * 1000
        first_rank = {item.id: rank for rank, item in enumerate(fused, 1)}
        first_score = {item.id: item.score for item in fused}
        pool_before = [item.id for item in fused]
        counts = {"first_stage_candidates": len(fused), "graph_seeds": 0, "graph_added": 0}
        if route is not None:
            counts["route"] = route

        graph_evidence: Dict[str, List[Dict[str, Any]]] = {}
        ranking: List[FusedItem] = fused
        if cfg.graph.enabled and self.graph is not None and fused:
            t0 = time.perf_counter()
            ranking, graph_evidence, added = self._expand(fused, lists, profile)
            timings["graph"] = (time.perf_counter() - t0) * 1000
            counts["graph_seeds"] = min(cfg.graph.seed_k, len(fused))
            counts["graph_added"] = added
        counts["candidates_after_graph"] = len(ranking)

        provenance = {item.id: item.contributions for item in ranking}
        final_scores: Dict[str, Tuple[float, str]] = {i.id: (i.score, "rrf") for i in ranking}
        order = [i.id for i in ranking]
        pre_rerank_rank = {ident: rank for rank, ident in enumerate(order, 1)}
        rerank_scores: Dict[str, float] = {}
        rerank_applied, rerank_error = False, None
        if cfg.rerank.enabled and order:
            t0 = time.perf_counter()
            pool = order[:cfg.rerank.pool_size]
            try:
                scored = self.reranker.rerank(text, [(i, self._rerank_text(i)) for i in pool])
                rerank_scores = dict(scored)
                order = [i for i, _ in scored] + order[len(pool):]
                final_scores.update({i: (s, "cross_encoder") for i, s in scored})
                rerank_applied = True
            except RerankerError as exc:
                rerank_error = str(exc)
                logger.error("rerank failed for %r: %s", text[:80], exc)
            timings["rerank"] = (time.perf_counter() - t0) * 1000
            counts["rerank_pool"] = len(pool)

        items: List[ResultItem] = []
        for rank, ident in enumerate(order[:cfg.top_k], 1):
            contributions = [_contribution_dict(c) for c in provenance.get(ident, [])]
            sources = sorted({c["source"] for c in contributions})
            methods = {s.split(":")[0] for s in sources}
            if ident in graph_evidence:
                methods.add("graph")
            score, score_type = final_scores[ident]
            items.append(ResultItem(
                function_id=ident, rank=rank, score=round(float(score), 8),
                score_type=score_type if rerank_applied and ident in rerank_scores else "rrf",
                first_stage_rank=first_rank.get(ident),
                first_stage_score=(round(first_score[ident], 8) if ident in first_score else None),
                pre_rerank_rank=pre_rerank_rank.get(ident) if rerank_applied else None,
                rerank_score=(round(rerank_scores[ident], 6) if ident in rerank_scores else None),
                sources=contributions,
                representations=sorted({s.split(":", 1)[1] for s in sources if ":" in s}),
                methods=sorted(methods), graph_evidence=graph_evidence.get(ident, [])))
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return QueryResult(
            query=text, results=items, timings_ms={k: round(v, 3) for k, v in timings.items()},
            counts=counts, rerank_applied=rerank_applied, rerank_error=rerank_error,
            pool_before_graph=pool_before, pool_after_graph=[i.id for i in ranking])

    # ---------------------------------------------------------------- helpers
    def _first_stage_lists(self, text: str, timings: Dict[str, float],
                           profile: Optional[Dict[str, Any]] = None) -> List[RankedList]:
        ret = self.config.retrieval
        profile = profile or {}
        w_dense = float(profile.get("dense_weight", ret.dense_weight))
        w_bm25 = float(profile.get("bm25_weight", ret.bm25_weight))
        w_exact = float(profile.get("exact_weight", ret.exact_weight))
        rep_weights = {**ret.representation_weights, **{k: float(v) for k, v in dict(profile.get("representation_weights", {})).items()}}
        lists: List[RankedList] = []
        qvec: Optional[np.ndarray] = None
        if ret.dense:
            t0 = time.perf_counter()
            qvec = self.encoder.encode_queries([text])[0]
            timings["encode_query"] = (time.perf_counter() - t0) * 1000
        for rep in ret.representations:
            rep_weight = float(rep_weights.get(rep, 1.0))
            if ret.dense:
                t0 = time.perf_counter()
                items = self.dense[rep].search(qvec, ret.candidate_k)
                timings["dense"] = timings.get("dense", 0.0) + (time.perf_counter() - t0) * 1000
                lists.append(RankedList(f"dense:{rep}", items, w_dense * rep_weight))
            if ret.bm25:
                t0 = time.perf_counter()
                items = self.bm25[rep].search(text, ret.candidate_k)
                timings["bm25"] = timings.get("bm25", 0.0) + (time.perf_counter() - t0) * 1000
                lists.append(RankedList(f"bm25:{rep}", items, w_bm25 * rep_weight))
        if ret.exact_match and self.exact is not None:
            t0 = time.perf_counter()
            items = self.exact.search(text, ret.candidate_k)
            timings["exact"] = (time.perf_counter() - t0) * 1000
            if items:
                lists.append(RankedList("exact", items, w_exact))
        return lists

    def _expand(self, fused: Sequence[FusedItem], lists: Sequence[RankedList],
                profile: Optional[Dict[str, Any]] = None):
        """Graph-expand the top seeds, then re-fuse the *original* lists plus the graph list.

        final(d) = sum over first-stage lists of w / (rrf_k + rank)   (unchanged)
                   + graph_weight / (rrf_k + graph_rank(d))            (graph ordering)

        Adding the graph as one more RRF list -- instead of fusing the already-fused
        ranking with it -- keeps the first-stage score margins intact, so graph
        evidence corroborates or adds candidates without flattening the ranking.
        """
        gcfg, ret = self.config.graph, self.config.retrieval
        seeds = [(item.id, rank) for rank, item in enumerate(fused[:gcfg.seed_k], 1)]
        candidates = self.graph.expand(
            seeds, gcfg.relations, gcfg.max_depth, gcfg.max_neighbors_per_node, gcfg.max_added)
        ordered = self.graph.order_for_fusion(candidates)
        graph_list = RankedList(
            "graph", [(c.id, c.support / c.hop) for c in ordered],
            float((profile or {}).get("graph_weight", gcfg.graph_weight)))
        merged = rrf_fuse([*lists, graph_list], ret.rrf_k)
        evidence = {
            c.id: [{"seed": e.seed, "seed_rank": e.seed_rank, "via": e.parent,
                    "relation": e.relation, "hop": e.hop, "edge_sources": list(e.edge_sources)}
                   for e in c.evidence[:8]]
            for c in ordered}
        added = sum(1 for c in candidates if not c.is_seed)
        return merged, evidence, added

    def _rerank_text(self, ident: str) -> str:
        rcfg = self.config.rerank
        parts = [self.texts[rep][ident] for rep in rcfg.text_representations
                 if ident in self.texts.get(rep, {})]
        if not parts:  # e.g. semantic excluded by policy: fall back to the identity line
            record = self.corpus.records[ident]
            parts = [record.qualified_name or record.name]
        return "\n\n".join(parts)[:rcfg.max_chars]
