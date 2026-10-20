"""Compiler-graph expansion with bounded traversal and per-candidate provenance.

Only relationships that actually exist in the Optima JSON become edges:

* ``callees`` / ``callers`` -- resolved entries of each function's ``calls`` and
  ``called_by`` plus the top-level ``graph.call_edges`` (union, de-duplicated).
  Unresolved callees (external functions) have no function ID and are not nodes.
* ``dependencies`` -- entries of the base ``dependencies`` field that are an
  existing function ID.  (This field is empty in all bundled datasets.)
* ``enriched_dependencies`` -- LLM-generated dependency names that exactly match
  exactly one function's qualified name or name.  These are *unvalidated model
  output* and are labelled as such in the evidence; this relation is off unless
  requested.

CFG metadata describes control flow *inside* one function; it is not turned into
interprocedural edges here.

Traversal is breadth-first from the seed set with a hop limit, a per-node
neighbour cap and a cap on the number of added candidates.  Graph evidence stays
separate from semantic similarity: ``order_for_fusion`` produces a plain ranking
that the pipeline fuses with RRF (see ``pipeline.py``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .corpus import Corpus

RELATION_TO_DIRECTION = {
    "callees": "caller->callee",
    "callers": "callee->caller",
    "dependencies": "function->dependency",
    "enriched_dependencies": "function->llm_dependency",
}


@dataclass(frozen=True)
class GraphEvidence:
    seed: str                  # seed function the path started from
    seed_rank: int             # that seed's 1-based rank in the first-stage ranking
    parent: str                # node the edge was followed from
    relation: str              # callees | callers | dependencies | enriched_dependencies
    hop: int
    edge_sources: Tuple[str, ...]   # where the edge was observed in the JSON


@dataclass
class GraphCandidate:
    id: str
    hop: int
    is_seed: bool
    evidence: List[GraphEvidence] = field(default_factory=list)

    @property
    def support(self) -> int:
        return len({e.seed for e in self.evidence})

    @property
    def best_seed_rank(self) -> int:
        return min(e.seed_rank for e in self.evidence)

    @property
    def relations(self) -> Tuple[str, ...]:
        return tuple(sorted({e.relation for e in self.evidence}))


class GraphIndex:
    def __init__(self, corpus: Corpus):
        self._order = {ident: i for i, ident in enumerate(corpus.records)}
        self._adj: Dict[str, Dict[str, List[str]]] = {
            rel: defaultdict(list) for rel in RELATION_TO_DIRECTION}
        self._edge_sources: Dict[Tuple[str, str, str], set] = defaultdict(set)
        self.stats: Dict[str, Any] = {}
        self._build(corpus)

    # ------------------------------------------------------------------ build
    def _add(self, relation: str, src: str, dst: str, source: str) -> str:
        """Record an edge; returns ``new``, ``duplicate``, ``self`` or ``dangling``."""
        if src == dst:
            return "self"
        if src not in self._order or dst not in self._order:
            return "dangling"
        if relation in ("callees", "callers"):
            call = (src, dst) if relation == "callees" else (dst, src)
            key = ("calls", call[0], call[1])
        else:
            key = (relation, src, dst)
        is_new = key not in self._edge_sources
        self._edge_sources[key].add(source)
        if is_new:
            if relation in ("callees", "callers"):
                self._adj["callees"][key[1]].append(key[2])
                self._adj["callers"][key[2]].append(key[1])
            else:
                self._adj[relation][src].append(dst)
        return "new" if is_new else "duplicate"

    def _build(self, corpus: Corpus) -> None:
        dangling = 0
        for record in corpus.records.values():
            for callee in record.calls:
                dangling += self._add("callees", record.id, callee, "calls") == "dangling"
            for caller in record.called_by:
                dangling += self._add("callers", record.id, caller, "called_by") == "dangling"
        for src, dst in corpus.extra_edges:
            dangling += self._add("callees", src, dst, "graph.call_edges") == "dangling"

        by_qualified: Dict[str, List[str]] = defaultdict(list)
        by_name: Dict[str, List[str]] = defaultdict(list)
        for record in corpus.records.values():
            by_qualified[record.qualified_name].append(record.id)
            by_name[record.name].append(record.id)
        unresolved_deps = ambiguous = unmatched = 0
        for record in corpus.records.values():
            for dep in record.dependencies:
                ident = dep.get("id") if isinstance(dep, dict) else dep
                outcome = self._add("dependencies", record.id, ident,
                                    "dependencies") if isinstance(ident, str) else "dangling"
                unresolved_deps += outcome == "dangling"
            for dep_name in record.enriched_dependencies:
                matches = by_qualified.get(dep_name) or by_name.get(dep_name) or []
                if len(matches) == 1:
                    self._add("enriched_dependencies", record.id, matches[0],
                              "llm_enrichment.dependencies")
                elif len(matches) > 1:
                    ambiguous += 1
                else:
                    unmatched += 1
        for adjacency in self._adj.values():
            for neighbours in adjacency.values():
                neighbours.sort(key=self._order.__getitem__)
        counts = defaultdict(int)
        for key in self._edge_sources:
            counts[key[0]] += 1
        self.stats = {
            "nodes": len(self._order),
            "call_edges": counts["calls"],
            "dependency_edges": counts["dependencies"],
            "enriched_dependency_edges": counts["enriched_dependencies"],
            "dangling_edges_ignored": dangling,
            "base_dependency_entries_unresolved": unresolved_deps,
            "enriched_dependency_names_ambiguous": ambiguous,
            "enriched_dependency_names_unmatched": unmatched,
        }

    # -------------------------------------------------------------- inspection
    def neighbours(self, ident: str, relation: str) -> List[str]:
        return list(self._adj[relation].get(ident, ()))

    def edge_sources(self, src: str, dst: str, relation: str) -> Tuple[str, ...]:
        if relation == "callees":
            key = ("calls", src, dst)
        elif relation == "callers":
            key = ("calls", dst, src)
        else:
            key = (relation, src, dst)
        return tuple(sorted(self._edge_sources.get(key, ())))

    def relation_edge_count(self, relation: str) -> int:
        if relation in ("callers", "callees"):
            return self.stats["call_edges"]
        if relation == "dependencies":
            return self.stats["dependency_edges"]
        return self.stats["enriched_dependency_edges"]

    # --------------------------------------------------------------- traversal
    def expand(self, seeds: Sequence[Tuple[str, int]], relations: Sequence[str],
               max_depth: int = 1, max_neighbors: int = 20,
               max_added: int = 20) -> List[GraphCandidate]:
        """Breadth-first expansion from ``seeds`` = ``[(function_id, rank), ...]``.

        Returns graph candidates in discovery order.  New non-seed candidates are
        capped by ``max_added``; seeds reached from *other* seeds are returned too
        (as structural corroboration) and do not count against the cap.
        """
        if max_depth < 1:
            return []
        for relation in relations:
            if relation not in RELATION_TO_DIRECTION:
                raise ValueError(f"unknown graph relation {relation!r}")
        seed_rank = {ident: rank for ident, rank in seeds if ident in self._order}
        candidates: Dict[str, GraphCandidate] = {}
        hop_of: Dict[str, int] = {ident: 0 for ident in seed_rank}
        frontier: Dict[str, List[str]] = {ident: [ident] for ident in seed_rank}
        added = 0
        for depth in range(1, max_depth + 1):
            next_frontier: Dict[str, List[str]] = {}
            for node, origins in frontier.items():
                for relation in relations:
                    for neighbour in self._adj[relation].get(node, ())[:max_neighbors]:
                        known_hop = hop_of.get(neighbour)
                        if known_hop is None:
                            if added >= max_added:
                                continue
                            hop_of[neighbour] = depth
                            added += 1
                            candidates[neighbour] = GraphCandidate(neighbour, depth, False)
                        elif known_hop == 0:
                            # A seed reached from another seed.
                            if all(o == neighbour for o in origins):
                                continue
                            candidates.setdefault(
                                neighbour, GraphCandidate(neighbour, depth, True))
                        elif known_hop != depth:
                            continue
                        candidate = candidates[neighbour]
                        sources = self.edge_sources(node, neighbour, relation)
                        for origin in origins:
                            if origin == neighbour:
                                continue
                            candidate.evidence.append(GraphEvidence(
                                origin, seed_rank[origin], node, relation, depth, sources))
                        if known_hop != 0:
                            bucket = next_frontier.setdefault(neighbour, [])
                            bucket.extend(o for o in origins if o not in bucket)
            frontier = next_frontier
            if not frontier:
                break
        return [c for c in candidates.values() if c.evidence]

    def order_for_fusion(self, candidates: Iterable[GraphCandidate]) -> List[GraphCandidate]:
        """Deterministic graph-only ordering.

        Nearer hops first; within a hop, neighbours of better-ranked seeds first
        (graph relevance is inherited from the seed's first-stage confidence, so a
        hub adjacent to many weak seeds does not outrank the neighbours of the top
        seed); then more distinct supporting seeds; then corpus order.
        """
        return sorted(
            candidates,
            key=lambda c: (c.hop, c.best_seed_rank, -c.support, self._order[c.id]))
