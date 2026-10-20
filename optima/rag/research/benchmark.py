"""Ground-truth benchmark for function retrieval: builder, loader and validator.

Independence of labels
----------------------
Labels are derived *only* from the compiler-produced ``base.json`` (names, source
text, resolved call edges, CFG) or written by hand.  They never use LLM
enrichment, embeddings or any retrieval output, so the benchmark cannot favour a
system by construction.  Every relevant function is identified by its stable
function ID, never by name alone.

Categories and how they are labelled
------------------------------------
``exact_name``       Where is ``f`` defined?             all functions whose name is ``f``
``identifier_api``   Which functions use identifier ``t``?   functions whose source contains ``t``
``caller``           Which functions call ``f``?          resolved callers of ``f``
``callee``           Which functions does ``f`` call?     resolved callees of ``f``
``control_flow``     callers of ``f`` that contain a loop / most complex function in a file
                     (CFG back-edge / cyclomatic complexity from the LLVM CFG)
``cross_function``   functions calling both ``a`` and ``b`` / called by both
``multi_hop``        functions reaching ``f`` within two calls (graded: 1 hop = 2, 2 hops = 1)
``dependency``       only if ``dependencies`` are present in the data (they are empty in
                     all bundled datasets, so no dependency queries are generated)
``behavioral``       hand-authored natural-language queries (see ``behavioral_authored.json``)

The structural categories (caller/callee/control_flow/cross_function/multi_hop) are
labelled from the same call graph that graph expansion traverses; results on them
measure how useful that graph is for *structural questions*, not general retrieval.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .corpus import Corpus

logger = logging.getLogger(__name__)

BENCHMARK_SCHEMA_VERSION = 1
CATEGORY_ORDER = ("exact_name", "identifier_api", "behavioral", "control_flow", "caller",
                  "callee", "dependency", "cross_function", "multi_hop")
CATEGORY_DESCRIPTIONS = {
    "exact_name": "Exact function-name lookup",
    "identifier_api": "Identifier / API usage lookup",
    "behavioral": "Natural-language behavioural description (hand-authored)",
    "control_flow": "Control-flow questions answered from the LLVM CFG",
    "caller": "Who calls function X (direct callers)",
    "callee": "What does function X call (direct callees)",
    "dependency": "Dependency questions (needs `dependencies` data)",
    "cross_function": "Cross-function tracing (common callers / callees)",
    "multi_hop": "Multi-hop structural reachability (graded by hop count)",
}
_KEYWORDS = {
    "static", "const", "return", "unsigned", "signed", "struct", "class", "template", "typename",
    "namespace", "include", "define", "inline", "virtual", "override", "nullptr", "sizeof",
    "typedef", "volatile", "register", "extern", "default", "continue", "explicit", "operator",
    "constexpr", "noexcept", "decltype", "static_cast", "reinterpret_cast", "const_cast",
    "dynamic_cast", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int64_t", "int32_t", "size_t",
}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{5,}")


class BenchmarkError(ValueError):
    """Raised for an invalid benchmark file or unresolvable authored labels."""


@dataclass
class Benchmark:
    benchmark_id: str
    query_set_version: str
    ground_truth_version: str
    project: str
    base_sha256: str
    queries: List[Dict[str, Any]]
    meta: Dict[str, Any] = field(default_factory=dict)

    def categories(self) -> List[str]:
        return [q.get("category", "") for q in self.queries]

    def subset(self, categories: Optional[Sequence[str]] = None,
               limit: Optional[int] = None) -> "Benchmark":
        queries = [q for q in self.queries if not categories or q.get("category") in categories]
        if limit:
            queries = queries[:limit]
        return Benchmark(self.benchmark_id, self.query_set_version, self.ground_truth_version,
                         self.project, self.base_sha256, queries, dict(self.meta))


# ------------------------------------------------------------------- loading
def load_benchmark(path: Path) -> Benchmark:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"cannot read benchmark {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"benchmark {path} is not valid JSON: {exc}") from exc
    if isinstance(data, list):  # legacy `queries.json`: a bare list of query objects
        queries, meta = data, {"legacy_format": True}
        header = {"benchmark_id": path.stem, "query_set_version": "legacy",
                  "ground_truth_version": "legacy", "project": "", "base_sha256": ""}
    elif isinstance(data, dict) and isinstance(data.get("queries"), list):
        queries = data["queries"]
        header = {k: data.get(k, "") for k in
                  ("benchmark_id", "query_set_version", "ground_truth_version", "project",
                   "base_sha256")}
        meta = {k: v for k, v in data.items() if k not in header and k != "queries"}
    else:
        raise BenchmarkError(f"benchmark {path} must be a list or an object with 'queries'")
    cleaned = []
    for index, query in enumerate(queries, 1):
        if not isinstance(query, dict) or "query" not in query:
            raise BenchmarkError(f"{path}: query #{index} is not an object with a 'query' field")
        item = dict(query)
        item.setdefault("query_id", f"q_{index:04d}")
        item.setdefault("category", "uncategorised")
        cleaned.append(item)
    return Benchmark(queries=cleaned, meta=meta, **header)


def validate_benchmark(bench: Benchmark, corpus: Corpus) -> Dict[str, Any]:
    """Check IDs against the corpus and report problems; never mutates the benchmark."""
    ids = set(corpus.records)
    invalid: Dict[str, List[str]] = {}
    empty, duplicates = [], []
    seen_ids, seen_text = set(), set()
    for query in bench.queries:
        qid = query["query_id"]
        if qid in seen_ids:
            duplicates.append(qid)
        seen_ids.add(qid)
        text = " ".join(str(query.get("query", "")).lower().split())
        if text in seen_text:
            duplicates.append(f"{qid} (duplicate text)")
        seen_text.add(text)
        relevant = list(query.get("relevance") or {}) or list(query.get("relevant_functions") or [])
        if not relevant:
            empty.append(qid)
        bad = [i for i in relevant if i not in ids]
        if bad:
            invalid[qid] = bad
    sha_mismatch = bool(bench.base_sha256) and bench.base_sha256 != corpus.meta.get("base_sha256")
    return {
        "queries": len(bench.queries),
        "queries_with_invalid_ids": len(invalid), "invalid_ids": invalid,
        "queries_without_judgments": empty, "duplicates": duplicates,
        "base_sha256_matches": not sha_mismatch if bench.base_sha256 else None,
        "ok": not invalid and not duplicates and not sha_mismatch,
    }


# ------------------------------------------------------------------- building
def _edges(corpus: Corpus) -> Set[Tuple[str, str]]:
    """Union of resolved call edges (self-loops dropped).  Deliberately independent
    of ``graph.GraphIndex`` so labels and the system under test share no code."""
    ids = set(corpus.records)
    edges: Set[Tuple[str, str]] = set()
    for record in corpus.records.values():
        for callee in record.calls:
            edges.add((record.id, callee))
        for caller in record.called_by:
            edges.add((caller, record.id))
    edges.update(corpus.extra_edges)
    return {(a, b) for a, b in edges if a != b and a in ids and b in ids}


def _sample(items: Sequence[Any], count: int, seed: int, salt: str) -> List[Any]:
    rng = random.Random(f"{seed}:{salt}")
    pool = list(items)
    rng.shuffle(pool)
    return pool[:count]


def _resolve_authored(corpus: Corpus, authored: Mapping[str, Any]) -> List[Dict[str, Any]]:
    by_qualified: Dict[str, List[str]] = defaultdict(list)
    for record in corpus.records.values():
        by_qualified[record.qualified_name].append(record.id)
    out, problems = [], []
    for index, entry in enumerate(authored.get("queries", []), 1):
        relevance: Dict[str, float] = {}
        for target in entry.get("targets", []):
            matches = [i for i in by_qualified.get(target["qualified_name"], [])
                       if i.split("::", 1)[0].endswith(target.get("file", ""))
                       and (not target.get("line") or i.endswith(f"::{target['line']}"))]
            if len(matches) != 1:
                problems.append(f"authored query #{index} target {target} matched "
                                f"{len(matches)} functions: {matches[:3]}")
                continue
            relevance[matches[0]] = max(relevance.get(matches[0], 0.0), float(target.get("grade", 1)))
        if relevance:
            out.append({"query": entry["query"], "relevance": relevance,
                        "notes": entry.get("notes", "")})
    if problems:
        raise BenchmarkError("cannot resolve authored labels:\n  " + "\n  ".join(problems))
    return out


DEVELOPER_INTENTS = {
    "vague_memory": "Half-remembered function ('the thing that ...')",
    "task": "Task-oriented question ('how do I ...')",
    "error_message": "Error text / log message / output string to code",
    "concept": "Concept or mechanism to the code that implements it",
    "impact": "Impact analysis: what uses / what does a function depend on",
}


def _resolve_developer(corpus: Corpus, authored: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Resolve hand-authored developer queries.  Three label kinds, none of which looks at any system
    output: explicit ``targets`` (hand-labelled), ``string_literal`` (every function whose source
    contains the literal) and ``callers_of`` / ``callees_of`` (resolved call graph)."""
    by_qualified: Dict[str, List[str]] = defaultdict(list)
    for record in corpus.records.values():
        by_qualified[record.qualified_name].append(record.id)
    edges = _edges(corpus)
    callers: Dict[str, Set[str]] = defaultdict(set)
    callees: Dict[str, Set[str]] = defaultdict(set)
    for a, b in edges:
        callers[b].add(a)
        callees[a].add(b)

    def find(ref: Mapping[str, Any]) -> List[str]:
        return [i for i in by_qualified.get(ref["qualified_name"], [])
                if i.split("::", 1)[0].endswith(ref.get("file", ""))
                and (not ref.get("line") or i.endswith(f"::{ref['line']}"))]

    out, problems = [], []
    for index, entry in enumerate(authored.get("queries", []), 1):
        intent = entry.get("intent")
        if intent not in DEVELOPER_INTENTS:
            problems.append(f"query #{index}: unknown intent {intent!r}")
            continue
        relevance: Dict[str, float] = {}
        source = "hand_authored"
        if "string_literal" in entry:
            literal, limit = entry["string_literal"], int(entry.get("max_matches", 3))
            matches = [r.id for r in corpus.records.values() if literal in r.source_code]
            if not 1 <= len(matches) <= limit:
                problems.append(f"query #{index}: literal {literal!r} matches {len(matches)} functions (allowed 1..{limit})")
                continue
            relevance = {i: 2.0 for i in matches}
            source = "source_string_literal"
        elif "callers_of" in entry or "callees_of" in entry:
            kind = "callers_of" if "callers_of" in entry else "callees_of"
            found = find(entry[kind])
            if len(found) != 1:
                problems.append(f"query #{index}: {kind} {entry[kind]} matched {len(found)} functions")
                continue
            related = (callers if kind == "callers_of" else callees)[found[0]]
            if not related:
                problems.append(f"query #{index}: {kind} {entry[kind]} has no resolved edges")
                continue
            relevance = {i: 1.0 for i in related}
            source = "resolved_call_graph"
        else:
            for target in entry.get("targets", []):
                found = find(target)
                if len(found) != 1:
                    problems.append(f"query #{index}: target {target} matched {len(found)} functions: {found[:3]}")
                    continue
                relevance[found[0]] = max(relevance.get(found[0], 0.0), float(target.get("grade", 1)))
        if relevance:
            out.append({"intent": intent, "query": entry["query"], "relevance": relevance,
                        "label_source": source, "notes": entry.get("notes", "")})
        elif not any(p.startswith(f"query #{index}:") for p in problems):
            problems.append(f"query #{index}: no labels")
    if problems:
        raise BenchmarkError("cannot resolve developer queries:\n  " + "\n  ".join(problems))
    return out


def build_developer_benchmark(corpus: Corpus, authored: Mapping[str, Any], benchmark_id: str = "",
                              version: str = "1.0.0") -> Dict[str, Any]:
    """Benchmark of developer-style queries grouped by intent (see ``DEVELOPER_INTENTS``)."""
    queries: List[Dict[str, Any]] = []
    for item in _resolve_developer(corpus, authored):
        queries.append({
            "category": item["intent"], "query": item["query"], "relevance": item["relevance"],
            "relevant_functions": list(item["relevance"]),
            "seed": max(item["relevance"], key=item["relevance"].get),
            "label_source": item["label_source"], "notes": item["notes"]})
    counters: Dict[str, int] = defaultdict(int)
    for query in queries:
        counters[query["category"]] += 1
        query["query_id"] = f"{query['category']}-{counters[query['category']]:03d}"
    project = corpus.meta.get("project", "")
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id or f"{project}-developer",
        "query_set_version": version, "ground_truth_version": version,
        "project": project, "base_sha256": corpus.meta.get("base_sha256", ""),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {"module": "optima.rag.research.benchmark.build_developer_benchmark"},
        "labeling": {
            "independence": ("queries were written by hand as a developer would ask them; labels come from hand-resolved "
                             "targets, source string literals or the resolved call graph; no LLM enrichment, embedding or "
                             "retrieval output was used, and no system was run on these queries before they were fixed"),
            "review": "single annotator (AI-assisted authoring); not independently reviewed by a second person",
        },
        "categories": {c: DEVELOPER_INTENTS[c] for c in DEVELOPER_INTENTS if any(q["category"] == c for q in queries)},
        "unsupported_categories": {},
        "counts": {c: sum(1 for q in queries if q["category"] == c) for c in DEVELOPER_INTENTS
                   if any(q["category"] == c for q in queries)},
        "queries": queries,
    }


def build_structural_benchmark(corpus: Corpus, per_category: int = 20, seed: int = 13,
                               authored: Optional[Mapping[str, Any]] = None,
                               benchmark_id: str = "", version: str = "1.0.0") -> Dict[str, Any]:
    """Generate a benchmark dictionary (see module docstring for the labelling rules)."""
    records = corpus.records
    edges = _edges(corpus)
    callees: Dict[str, List[str]] = defaultdict(list)
    callers: Dict[str, List[str]] = defaultdict(list)
    for a, b in sorted(edges):
        callees[a].append(b)
        callers[b].append(a)
    qualified_counts: Dict[str, int] = defaultdict(int)
    name_groups: Dict[str, List[str]] = defaultdict(list)
    for record in records.values():
        qualified_counts[record.qualified_name] += 1
        name_groups[record.name].append(record.id)

    def unique_ref(ident: str) -> Optional[str]:
        record = records[ident]
        if qualified_counts[record.qualified_name] == 1 and len(record.qualified_name) >= 3:
            return record.qualified_name
        return None

    queries: List[Dict[str, Any]] = []
    unsupported: Dict[str, str] = {}

    def add(category: str, text: str, relevance: Mapping[str, float], seed_id: str,
            label_source: str, notes: str = "") -> None:
        queries.append({
            "category": category, "query": text, "relevance": dict(relevance),
            "relevant_functions": list(relevance), "seed": seed_id,
            "label_source": label_source, "notes": notes})

    # exact_name ------------------------------------------------------------
    templates = ["Where is the function {t} defined?", "Show me the implementation of {t}",
                 "Find the definition of the {t} function"]
    names = sorted(n for n, ids in name_groups.items()
                   if len(n) >= 4 and len(ids) <= 3 and "operator" not in n and not n.startswith("~"))
    for n in _sample(names, per_category, seed, "exact_name"):
        text = templates[len(queries) % len(templates)].format(t=n)
        add("exact_name", text, {i: 1.0 for i in name_groups[n]}, name_groups[n][0],
            "name_equality")

    # identifier_api --------------------------------------------------------
    defined = {r.name for r in records.values()} | {p for r in records.values()
                                                    for p in r.qualified_name.split("::")}
    doc_freq: Dict[str, Set[str]] = defaultdict(set)
    for record in records.values():
        for token in set(_IDENT.findall(record.source_code)):
            doc_freq[token].add(record.id)

    def symbol_shaped(token: str) -> bool:
        return ("_" in token.strip("_") or bool(re.search(r"[a-z][A-Z]", token))
                or (token.isupper() and len(token) >= 6))
    tokens = sorted(t for t, hits in doc_freq.items()
                    if 1 <= len(hits) <= 4 and symbol_shaped(t) and t not in defined
                    and t.lower() not in _KEYWORDS)
    templates = ["Which functions use the identifier {t}?", "Find code that references {t}",
                 "Where is {t} used?"]
    for n, t in enumerate(_sample(tokens, per_category, seed, "identifier_api")):
        add("identifier_api", templates[n % len(templates)].format(t=t),
            {i: 1.0 for i in sorted(doc_freq[t])}, sorted(doc_freq[t])[0], "source_token_presence")

    # caller / callee --------------------------------------------------------
    callee_targets = sorted(i for i in records if unique_ref(i) and 1 <= len(callers.get(i, [])) <= 6)
    templates = ["Which functions call {q}?", "What are the direct callers of {q}?",
                 "Find every function that invokes {q}"]
    for n, ident in enumerate(_sample(callee_targets, per_category, seed, "caller")):
        add("caller", templates[n % len(templates)].format(q=unique_ref(ident)),
            {c: 1.0 for c in callers[ident]}, ident, "resolved_call_graph")
    caller_targets = sorted(i for i in records if unique_ref(i) and 1 <= len(callees.get(i, [])) <= 6)
    templates = ["Which functions does {q} call?", "What functions are called directly by {q}?",
                 "List the callees of {q}"]
    for n, ident in enumerate(_sample(caller_targets, per_category, seed, "callee")):
        add("callee", templates[n % len(templates)].format(q=unique_ref(ident)),
            {c: 1.0 for c in callees[ident]}, ident, "resolved_call_graph")

    # control_flow -----------------------------------------------------------
    loop_targets = []
    for ident in sorted(records):
        cs = callers.get(ident, [])
        if unique_ref(ident) and len(cs) >= 2 and all(records[c].cfg for c in cs):
            looping = [c for c in cs if records[c].cfg["has_loop"]]
            if 0 < len(looping) < len(cs):
                loop_targets.append((ident, looping))
    half = max(1, per_category // 2)
    for ident, looping in _sample(loop_targets, half, seed, "cf_loop"):
        add("control_flow", f"Which of the functions that call {unique_ref(ident)} contain a loop?",
            {c: 1.0 for c in looping}, ident, "cfg_back_edge+call_graph")
    by_file: Dict[str, List[str]] = defaultdict(list)
    for record in records.values():
        by_file[record.file].append(record.id)
    complex_targets = []
    for file, ids in sorted(by_file.items()):
        if len(ids) >= 3 and all(records[i].cfg for i in ids):
            ranked = sorted(ids, key=lambda i: -records[i].cfg["cyclomatic"])
            if records[ranked[0]].cfg["cyclomatic"] >= records[ranked[1]].cfg["cyclomatic"] + 2:
                complex_targets.append((file, ranked[0]))
    for file, ident in _sample(complex_targets, per_category - half, seed, "cf_complex"):
        add("control_flow",
            f"Which function in {file} has the most complex control flow (highest cyclomatic complexity)?",
            {ident: 1.0}, ident, "cfg_cyclomatic")

    # dependency -------------------------------------------------------------
    dep_targets = []
    for record in records.values():
        resolved = [d.get("id") if isinstance(d, dict) else d for d in record.dependencies]
        resolved = [d for d in resolved if isinstance(d, str) and d in records and d != record.id]
        if resolved and unique_ref(record.id):
            dep_targets.append((record.id, resolved))
    if dep_targets:
        for ident, deps in _sample(sorted(dep_targets), per_category, seed, "dependency"):
            add("dependency", f"Which functions does {unique_ref(ident)} depend on?",
                {d: 1.0 for d in deps}, ident, "base_dependencies_field")
    else:
        unsupported["dependency"] = ("the `dependencies` field is empty for every function in "
                                     "base.json, so no dependency ground truth exists")

    # cross_function ---------------------------------------------------------
    both_callers: List[Tuple[str, str, List[str]]] = []
    both_callees: List[Tuple[str, str, List[str]]] = []
    for ident in sorted(records):
        outs = [o for o in callees.get(ident, []) if unique_ref(o)]
        for x in range(len(outs)):
            for y in range(x + 1, len(outs)):
                common = sorted(set(callers[outs[x]]) & set(callers[outs[y]]))
                if 1 <= len(common) <= 5:
                    both_callers.append((outs[x], outs[y], common))
        ins = [c for c in callers.get(ident, []) if unique_ref(c)]
        for x in range(len(ins)):
            for y in range(x + 1, len(ins)):
                common = sorted(set(callees[ins[x]]) & set(callees[ins[y]]))
                if 1 <= len(common) <= 5:
                    both_callees.append((ins[x], ins[y], common))
    half = max(1, per_category // 2)
    seen_pairs: Set[Tuple[str, str]] = set()
    for a, b, common in _sample(sorted(set((a, b, tuple(c)) for a, b, c in both_callers)),
                                half * 3, seed, "cross_callers"):
        if (a, b) in seen_pairs or len([q for q in queries if q.get("notes") == "common_callers"]) >= half:
            continue
        seen_pairs.add((a, b))
        add("cross_function", f"Which functions call both {unique_ref(a)} and {unique_ref(b)}?",
            {c: 1.0 for c in common}, a, "resolved_call_graph", "common_callers")
    for a, b, common in _sample(sorted(set((a, b, tuple(c)) for a, b, c in both_callees)),
                                per_category * 3, seed, "cross_callees"):
        if (a, b) in seen_pairs or len([q for q in queries if q.get("notes") == "common_callees"]) >= per_category - half:
            continue
        seen_pairs.add((a, b))
        add("cross_function", f"Which functions are called by both {unique_ref(a)} and {unique_ref(b)}?",
            {c: 1.0 for c in common}, a, "resolved_call_graph", "common_callees")

    # multi_hop (graded) -----------------------------------------------------
    hop_targets = []
    for ident in sorted(records):
        if not unique_ref(ident):
            continue
        for direction, adj in (("callers", callers), ("callees", callees)):
            one = set(adj.get(ident, []))
            two = {n for m in one for n in adj.get(m, [])} - one - {ident}
            if one and two and len(one) + len(two) <= 20:
                hop_targets.append((ident, direction, sorted(one), sorted(two)))
    for ident, direction, one, two in _sample(hop_targets, per_category, seed, "multi_hop"):
        verb = ("can reach {q} through at most two calls" if direction == "callers"
                else "can be reached from {q} through at most two calls")
        add("multi_hop", f"Which functions {verb.format(q=unique_ref(ident))}?",
            {**{i: 1.0 for i in two}, **{i: 2.0 for i in one}}, ident,
            "resolved_call_graph_2hop", "graded: 1 hop = 2, 2 hops = 1")

    # behavioural (hand-authored) -----------------------------------------------
    if authored:
        for item in _resolve_authored(corpus, authored):
            add("behavioral", item["query"], item["relevance"], max(item["relevance"],
                key=item["relevance"].get), "hand_authored", item.get("notes", ""))

    counters: Dict[str, int] = defaultdict(int)
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    queries.sort(key=lambda q: order.get(q["category"], 99))
    for query in queries:
        counters[query["category"]] += 1
        query["query_id"] = f"{query['category']}-{counters[query['category']]:03d}"
    project = corpus.meta.get("project", "")
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_id": benchmark_id or f"{project}-structural",
        "query_set_version": version,
        "ground_truth_version": version,
        "project": project,
        "base_sha256": corpus.meta.get("base_sha256", ""),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {"module": "optima.rag.research.benchmark", "seed": seed,
                      "per_category": per_category},
        "labeling": {
            "independence": ("labels come from base.json (names, source text, resolved call "
                             "edges, LLVM CFG) or were hand-authored; no LLM enrichment, "
                             "embedding or retrieval output was used"),
            "behavioral_review": ("single annotator (AI-assisted authoring); not independently "
                                  "reviewed by a second person"),
            "caveat": ("structural categories are labelled from the same call graph used by "
                       "graph expansion"),
        },
        "categories": {c: CATEGORY_DESCRIPTIONS[c] for c in CATEGORY_ORDER
                       if any(q["category"] == c for q in queries)},
        "unsupported_categories": unsupported,
        "counts": {c: sum(1 for q in queries if q["category"] == c) for c in CATEGORY_ORDER
                   if any(q["category"] == c for q in queries)},
        "queries": queries,
    }
