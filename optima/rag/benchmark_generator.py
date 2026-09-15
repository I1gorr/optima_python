"""Evidence-backed benchmark query generation for Optima's RAG evaluator.

The benchmark deliberately describes *behaviour* rather than naming the
function being tested.  This keeps retrieval evaluation meaningful: a query
must be matched to code/enrichment/graph evidence, not to an identifier copied
from the ground truth.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


DEFAULT_DIFFICULTY_DISTRIBUTION = (0.25, 0.35, 0.40)
DIFFICULTIES = ("easy", "moderate", "difficult")
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
_STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "with", "from",
    "into", "that", "this", "when", "then", "is", "are", "be", "by", "on",
    "in", "as", "it", "its", "their", "they", "does", "do", "used", "use",
    "function", "method", "code", "component", "value", "values", "data",
    "none", "unknown", "true", "false",
}


class BenchmarkQueryGenerator:
    """Generate indirect, evidence-grounded benchmark queries."""

    # Kept public for callers which inspect the old generator.  These templates
    # are intentionally semantic; no ``{function_name}``/identifier template is
    # offered anymore.
    query_templates = {
        "easy": {
            "functionality": [
                "Which part of the code is responsible for {subject}?",
                "Where is the behaviour for {subject} implemented?",
            ],
            "input_output": [
                "What processing turns {input_subject} into {output_subject}?",
                "Where is {output_subject} produced from {input_subject}?",
            ],
        },
        "moderate": {
            "data_flow": [
                "How does {source_subject} become {target_subject} across the code?",
                "What path carries {source_subject} toward {target_subject}?",
            ],
            "dependencies": [
                "What supporting work must happen before {subject} can be completed?",
                "Which lower-level operation enables {subject}?",
            ],
            "control_flow": [
                "Where does execution decide whether {subject} proceeds?",
                "What condition changes the path used for {subject}?",
            ],
        },
        "difficult": {
            "multi_hop": [
                "Trace the two-stage path from {source_subject} through an intermediate step to {target_subject}.",
                "Which intermediary connects the preparation of {source_subject} with the resulting {target_subject}?",
            ],
            "cross_function": [
                "How do the components responsible for {left_subject} and {right_subject} cooperate?",
                "Where is the boundary between {left_subject} and {right_subject} coordinated?",
            ],
            "reasoning": [
                "What change would preserve the relationship between {left_subject} and {right_subject}?",
                "Why does the implementation need both {left_subject} and {right_subject} to achieve its result?",
            ],
            "state_transition": [
                "How does the code move from {source_subject} to {target_subject}, including the state it preserves?",
            ],
            "error_handling": [
                "What happens when the path for {subject} cannot complete, and where is that handled?",
            ],
        },
    }

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        self.random = random.Random(seed)
        # Preserve the historical attribute while avoiding global RNG changes.
        self.generic_templates = {
            level: [template for group in groups.values() for template in group]
            for level, groups in self.query_templates.items()
        }

    @staticmethod
    def _clean_text(text: Any) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9_ -]", " ", str(text))).strip()

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        return {token.lower() for token in _WORD_RE.findall(text or "")}

    def _calculate_lexical_overlap(self, text1: str, text2: str) -> float:
        left, right = self._tokens(text1), self._tokens(text2)
        return len(left & right) / len(left | right) if left and right else 0.0

    @staticmethod
    def _function_id(func: Dict[str, Any]) -> str:
        return str(func.get("id") or func.get("qualified_name") or func.get("name") or "")

    @staticmethod
    def _relationship_name(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("name") or value.get("qualified_name") or "")
        return str(value or "")

    def _get_relationships(self, func: Dict[str, Any]) -> Dict[str, List[str]]:
        """Return graph evidence while accepting old string and new dict shapes."""
        result: Dict[str, List[str]] = defaultdict(list)
        for key in ("calls", "called_by", "dependencies"):
            values = func.get(key, []) or []
            for value in values:
                if isinstance(value, dict):
                    value = value.get("id") or value.get("qualified_name") or value.get("name")
                if value:
                    result[key].append(str(value))
        return result

    def _all_identifiers(self, func: Dict[str, Any]) -> Set[str]:
        """Identifiers which must never appear in a generated query."""
        identifiers: Set[str] = set()
        for key in ("id", "name", "qualified_name", "mangled_name"):
            value = str(func.get(key) or "")
            if value:
                identifiers.add(value.lower())
                # Hide class/namespace/path components, but keep camelCase
                # function names intact.  Banning "get" or "handle" would
                # reject valid semantic descriptions without hiding a symbol.
                identifiers.update(
                    part.lower() for part in re.split(r"[:/\\<>(),\s]+", value)
                    if len(part) > 2
                )
        for parameter in func.get("parameters", []) or []:
            if isinstance(parameter, dict):
                for key in ("name", "type"):
                    value = str(parameter.get(key) or "")
                    identifiers.add(value.lower())
                    identifiers.update(part.lower() for part in re.split(r"[:./\\<>(),\s]+", value) if len(part) > 2)
        for key in ("calls", "called_by", "dependencies"):
            for value in func.get(key, []) or []:
                value = self._relationship_name(value)
                identifiers.add(value.lower())
                identifiers.update(
                    part.lower() for part in re.split(r"[:./\\<>(),\s]+", value)
                    if len(part) > 2
                )
        # Enrichment often mentions symbols in prose or dependency lists.
        # Remove identifier-shaped tokens before using that prose as evidence.
        for key in ("enrichment",):
            value = func.get(key) or {}
            if isinstance(value, dict):
                for field in ("dependencies", "side_effects", "inputs", "outputs"):
                    values = value.get(field, [])
                    if isinstance(values, str):
                        values = [values]
                    for item in values or []:
                        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", str(item)):
                            if "_" in token or re.search(r"[A-Z]", token):
                                identifiers.add(token.lower())
        return {item for item in identifiers if item and item not in _STOP_WORDS}

    def _evidence_text(self, func: Dict[str, Any]) -> str:
        """Collect semantic evidence from enrichment and source, not identifiers."""
        enrichment = func.get("enrichment") or {}
        parts: List[str] = []
        for key in (
            "purpose", "summary", "behavior", "algorithm", "inputs", "outputs",
            "side_effects", "concepts", "keywords",
        ):
            value = enrichment.get(key, "")
            if isinstance(value, (list, tuple)):
                parts.extend(str(item) for item in value)
            elif isinstance(value, str):
                parts.append(value)
        source = func.get("source_code") or func.get("source") or ""
        if source:
            # A short source sample supplies evidence when enrichment is absent.
            parts.append(re.sub(r"//.*|/\*.*?\*/", " ", str(source), flags=re.S)[:900])
        return self._clean_text(" ".join(parts))

    def _semantic_subject(
        self,
        func: Dict[str, Any],
        fallback: str = "the observed operation",
        banned_identifiers: Optional[Set[str]] = None,
    ) -> str:
        """Make a short, indirect description and remove target identifiers."""
        enrichment = func.get("enrichment") or {}
        # Prefer one concise explanatory field. Combining every enrichment
        # field creates keyword soup and leaks implementation vocabulary.
        preferred: List[str] = []
        for key in ("summary", "purpose", "behavior", "algorithm"):
            value = enrichment.get(key, "")
            if isinstance(value, str) and len(value.split()) >= 4:
                preferred.append(value)
        text = self._clean_text(preferred[0] if preferred else "")
        text = re.sub(r"\b(this|the)\s+function\b", "", text, flags=re.I)
        if not text:
            source = str(func.get("source_code") or func.get("source") or "")
            control_words = re.findall(
                r"\b(?:parse|calculate|compute|convert|create|construct|"
                r"initialize|update|validate|check|compare|iterate|sort|"
                r"search|store|load|open|close|print|return|insert|remove|"
                r"generate|evaluate|apply|set|clear|copy|restore)\w*\b",
                source,
                flags=re.I,
            )
            structural = []
            if re.search(r"\b(if|switch|case)\b", source):
                structural.append("conditional branch")
            if re.search(r"\b(for|while|do)\b", source):
                structural.append("iterative traversal")
            if re.search(r"\b(throw|try|catch)\b", source):
                structural.append("failure handling")
            if re.search(r"\b(new|make_unique|make_shared)\b", source):
                structural.append("object construction")
            if re.search(r"\breturn\b", source):
                structural.append("computed result")
            if re.search(r"(?<![=!<>])=(?!=)", source):
                structural.append("state update")
            # String literals often contain protocol/function names; structural
            # and action evidence is safer and more portable than copying them.
            text = self._clean_text(" ".join(control_words + structural))
        identifiers = self._all_identifiers(func) | set(banned_identifiers or ())
        metadata_words = {
            "purpose", "behavior", "summary", "algorithm", "inputs", "outputs",
            "side", "effects", "dependencies", "unknown", "object", "function",
            "implementation", "result", "code", "named", "called", "returns",
            "given", "various", "necessary", "components", "including",
            "first", "then", "finally", "based",
        }
        source_noise = {
            "static", "const", "auto", "return", "true", "false", "std", "cout",
            "cerr", "endl", "size", "string", "vector", "int", "void", "bool",
            "char", "long", "unsigned", "signed", "include", "public", "private",
        }
        words = []
        for word in _WORD_RE.findall(text):
            lower = word.lower()
            if (
                lower in _STOP_WORDS
                or lower in source_noise
                or lower in metadata_words
                or lower in identifiers
                or len(lower) < 3
            ):
                continue
            # Avoid copying implementation spelling (camel case and symbols).
            if "_" in word or re.search(r"[A-Z].*[A-Z]", word):
                continue
            if lower not in {item.lower() for item in words}:
                words.append(lower)
        if not words:
            return ""
        # Prefer a phrase rich enough to distinguish the evidence, but avoid
        # long enrichment quotations (which would make the query too direct).
        return " ".join(words[:10])

    def _get_function_signature(self, func: Dict[str, Any]) -> str:
        """Compatibility helper; signatures are not used in generated queries."""
        params = [p.get("name", "") for p in func.get("parameters", []) if isinstance(p, dict)]
        return f"{func.get('name', '')}({', '.join(p for p in params if p)})".rstrip("()")

    def _is_query_safe(
        self, query: str, func: Dict[str, Any], evidence_functions: Optional[Sequence[Dict[str, Any]]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Reject identifier leakage and copied enrichment phrases."""
        targets = list(evidence_functions or [func])
        query_tokens = self._tokens(query)
        identifiers = set().union(*(self._all_identifiers(item) for item in targets))
        leaked = sorted(token for token in identifiers if token in query_tokens)
        overlap = max(
            (self._calculate_lexical_overlap(query, self._evidence_text(item)) for item in targets),
            default=0.0,
        )
        details = {
            "lexical_overlap": round(overlap, 4),
            "contains_function_name": bool(leaked),
            "contains_target_identifier": bool(leaked),
            "target_identifiers": leaked,
            "contains_summary_phrase": overlap > 0.42,
            "contains_purpose_phrase": overlap > 0.42,
            "contains_behavior_phrase": overlap > 0.42,
        }
        # Queries with a copied phrase are not useful semantic tests even if
        # they contain no identifier.
        safe = not leaked and overlap < 0.42 and len(self._tokens(query)) >= 5
        return safe, details

    def _generate_query_from_template(self, template: str, func: Dict[str, Any]) -> Optional[str]:
        """Compatibility wrapper that substitutes semantic evidence only."""
        subject = self._semantic_subject(func)
        values = {
            "subject": subject,
            "input_subject": subject,
            "output_subject": self._semantic_subject(func, "the resulting state"),
            "source_subject": subject,
            "target_subject": "the resulting state",
            "left_subject": subject,
            "right_subject": "the downstream result",
        }
        try:
            if "{" in template:
                return template.format(**values)
            return template
        except (KeyError, IndexError, ValueError):
            # Support old positional templates without reintroducing names.
            try:
                return template.format(subject)
            except (KeyError, IndexError, ValueError):
                return None

    def _index_functions(self, functions: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for func in functions:
            fid = self._function_id(func)
            if fid:
                index[fid] = func
            for key in ("name", "qualified_name"):
                value = str(func.get(key) or "")
                if value and value not in index:
                    index[value] = func
        return index

    def _related_functions(
        self, func: Dict[str, Any], functions: Sequence[Dict[str, Any]], max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Resolve a bounded call/dependency graph for multi-hop evidence."""
        index = self._index_functions(functions)
        result, seen = [], {self._function_id(func)}
        queue = deque([(func, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            rel = self._get_relationships(current)
            neighbors = rel["calls"] + rel["called_by"] + rel["dependencies"]
            for ref in neighbors:
                related = index.get(ref) or index.get(ref.split("::")[-1])
                if not related:
                    continue
                rid = self._function_id(related)
                if rid in seen:
                    continue
                seen.add(rid)
                result.append(related)
                queue.append((related, depth + 1))
        return result

    def _category_candidates(
        self, func: Dict[str, Any], functions: Sequence[Dict[str, Any]], difficulty: str
    ) -> List[Tuple[str, str, List[Dict[str, Any]], int]]:
        """Return (category, query, evidence, graph depth) candidates."""
        banned = set().union(*(self._all_identifiers(item) for item in functions))
        primary = self._semantic_subject(func, banned_identifiers=banned)
        related = self._related_functions(func, functions, 2)
        candidates: List[Tuple[str, str, List[Dict[str, Any]], int]] = []
        source = str(func.get("source_code") or func.get("source") or "")
        enrichment = func.get("enrichment") or {}
        if not primary:
            return candidates

        def add(category: str, text: str, evidence: Sequence[Dict[str, Any]], depth: int = 0):
            if text and not text.endswith("{"):
                candidates.append((category, text, list(evidence), depth))

        if difficulty == "easy":
            add("functionality", f"Which part of the code is responsible for {primary}?", [func])
            add("input_output", f"What processing produces the observed result after {primary}?", [func])
            add("functionality", f"Where should I start looking to understand {primary}?", [func])
            add("functionality", f"What implementation area explains the behaviour described as {primary}?", [func])
            add("input_output", f"How does the system turn {primary} into an outcome used elsewhere?", [func])
            add("input_output", f"What transformation is associated with {primary} before a result is returned?", [func])
        elif difficulty == "moderate":
            add("data_flow", f"How does {primary} become the resulting state across the code?", [func])
            add("data_flow", f"Where does information from {primary} go before it is consumed?", [func])
            add("data_flow", f"What intermediate representation connects {primary} with the eventual result?", [func])
            add("data_flow", f"What should I trace to see where the information from {primary} is consumed?", [func])
            add("data_flow", f"How is the outcome of {primary} carried into later processing?", [func])
            if related:
                add("dependencies", f"What supporting operation enables {primary} to complete?", [func, related[0]])
                add("dependencies", f"What lower-level capability does {primary} rely on?", [func, related[0]])
                add("dependencies", f"Which prerequisite relationship should I inspect around {primary}?", [func, related[0]])
                add("dependencies", f"What does {primary} need from the surrounding implementation?", [func, related[0]])
            if re.search(r"\b(if|switch|case|else|while|for)\b", source):
                add("control_flow", f"What condition changes the execution path for {primary}?", [func])
                add("control_flow", f"Where should I inspect the decision that changes what happens after {primary}?", [func])
                add("control_flow", f"What branch determines the next step after {primary}?", [func])
        else:
            if related:
                other = related[0]
                other_subject = self._semantic_subject(other, "the downstream operation", banned)
                add("cross_function", f"How do the components responsible for {primary} and {other_subject} cooperate?", [func, other], 1)
                add("cross_function", f"Where should I trace the handoff between {primary} and {other_subject}?", [func, other], 1)
                add("cross_function", f"What interaction connects {primary} with {other_subject} during execution?", [func, other], 1)
                add("cross_function", f"Which implementation boundary should I follow between {primary} and {other_subject}?", [func, other], 1)
                add("reasoning", f"Why does the implementation need both {primary} and {other_subject} to achieve its result?", [func, other], 1)
                add("reasoning", f"If {primary} changes, what relationship with {other_subject} would need review?", [func, other], 1)
                add("reasoning", f"What design responsibility is shared by {primary} and {other_subject}?", [func, other], 1)
                add("reasoning", f"What would break if the relationship between {primary} and {other_subject} changed?", [func, other], 1)
                if len(related) > 1:
                    final = self._semantic_subject(related[1], "the final result", banned)
                    add("multi_hop", f"How does {primary} reach {final} through an intermediate processing step?", [func, related[0], related[1]], 2)
                    add("multi_hop", f"Which intermediate transformation links {primary} with {final} across the call path?", [func, related[0], related[1]], 2)
                    add("multi_hop", f"What should I trace from {primary} through an intermediate step before it reaches {final}?", [func, related[0], related[1]], 2)
                    add("multi_hop", f"Which sequence carries {primary} through an intermediate operation toward {final}?", [func, related[0], related[1]], 2)
            if re.search(r"(?<![=!<>])=(?!=)|\b(update|reset|restore|initialize|clear|set)\w*\b", source, re.I):
                add("state_transition", f"How does the code move from {primary} to a resulting state while preserving context?", [func])
                add("state_transition", f"Where should I investigate how {primary} prepares or changes state before the next stage?", [func])
                add("state_transition", f"What state is established while the system performs {primary}?", [func])
                add("state_transition", f"If state from an earlier operation persists, what should I trace around {primary}?", [func])
                add("state_transition", f"What part of the implementation controls the transition associated with {primary}?", [func])
                add("state_transition", f"How is the result of {primary} reflected in the next state?", [func])
            if (
                re.search(r"\b(throw|try|catch|error|fail|invalid|assert)\w*\b", source, re.I)
                or any(
                    re.search(r"\b(error|fail|invalid|exception|reject)\b", str(enrichment.get(key, "")), re.I)
                    for key in ("behavior", "purpose", "summary", "side_effects")
                )
            ):
                add("error_handling", f"What happens when the path for {primary} cannot complete, and where is that handled?", [func])
                add("error_handling", f"Where should I look if {primary} produces an unexpected or unusable result?", [func])
                add("error_handling", f"What protects the system when the processing described by {primary} goes wrong?", [func])
                add("error_handling", f"If this operation fails under unusual input, what implementation area should I inspect?", [func])
                add("error_handling", f"Where would I investigate an incomplete path or invalid result related to {primary}?", [func])
        return candidates

    @staticmethod
    def _allocate_counts(total: int, distribution: Sequence[float]) -> List[int]:
        if total <= 0:
            return [0, 0, 0]
        weights = list(distribution[:3]) + [0.0] * (3 - len(distribution))
        if sum(weights) <= 0:
            weights = list(DEFAULT_DIFFICULTY_DISTRIBUTION)
        raw = [total * max(0.0, weight) / sum(weights) for weight in weights[:3]]
        counts = [int(value) for value in raw]
        for index in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[: total - sum(counts)]:
            counts[index] += 1
        return counts

    def generate_benchmark_queries(
        self,
        functions: List[Dict[str, Any]],
        num_queries: int = 20,
        difficulty_distribution: Tuple[float, float, float] = DEFAULT_DIFFICULTY_DISTRIBUTION,
    ) -> List[Dict[str, Any]]:
        """Generate quality-filtered queries with a 25/35/40 default split."""
        if not functions or num_queries <= 0:
            return []
        counts = self._allocate_counts(num_queries, difficulty_distribution)
        candidates = [f for f in functions if self._function_id(f)]
        if not candidates:
            return []
        generated: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        # Cycling functions makes small fixtures useful while preserving variety.
        order = list(range(len(candidates)))
        self.random.shuffle(order)
        attempts = max(num_queries * 30, 100)
        for difficulty, target in zip(DIFFICULTIES, counts):
            made = 0
            offset = 0
            while made < target and offset < attempts:
                func = candidates[order[offset % len(order)]]
                offset += 1
                options = self._category_candidates(func, candidates, difficulty)
                self.random.shuffle(options)
                for category, query, evidence, depth in options:
                    normalized = " ".join(query.lower().split())
                    if normalized in seen:
                        continue
                    safe, validation = self._is_query_safe(query, func, evidence)
                    quality = self._score_query(query, evidence, difficulty, depth, validation)
                    if not safe or quality < 0.45:
                        continue
                    ids = [self._function_id(item) for item in evidence if self._function_id(item)]
                    generated.append({
                        "query": query,
                        "relevant_functions": list(dict.fromkeys(ids)),
                        "function_name": func.get("name", ""),  # legacy metadata only
                        "qualified_name": func.get("qualified_name", ""),
                        "function_id": self._function_id(func),
                        "difficulty": difficulty,
                        "category": category,
                        "query_type": category,
                        "reasoning_depth": max(1, depth + (2 if difficulty == "difficult" else 1)),
                        "directness_score": round(
                            self._directness_score(query, validation, difficulty), 4
                        ),
                        "query_quality_score": round(quality, 4),
                        "reasoning_required": "high" if difficulty == "difficult" else ("medium" if difficulty == "moderate" else "low"),
                        "evidence": {
                            "source": bool(func.get("source_code") or func.get("source")),
                            "enrichment": bool(func.get("enrichment")),
                            "graph_depth": depth,
                            "relationship_count": len(self._related_functions(func, candidates, 2)),
                        },
                        "validation": validation,
                        "is_safe": True,
                    })
                    seen.add(normalized)
                    made += 1
                    if made >= target:
                        break
            if made < target:
                logger.warning("Only generated %d/%d %s queries", made, target, difficulty)
        self.random.shuffle(generated)
        for index, item in enumerate(generated[:num_queries], 1):
            item["query_id"] = f"q_{index:04d}"
        return generated[:num_queries]

    def _score_query(
        self, query: str, evidence: Sequence[Dict[str, Any]], difficulty: str,
        graph_depth: int, validation: Dict[str, Any],
    ) -> float:
        words = self._tokens(query)
        evidence_words = set().union(*(self._tokens(self._evidence_text(item)) for item in evidence))
        grounding = min(1.0, len(words & evidence_words) / 4.0)
        indirectness = 1.0 if not validation.get("contains_target_identifier") else 0.0
        specificity = min(1.0, len(words) / 12.0)
        graph = min(1.0, graph_depth / 2.0) if difficulty == "difficult" else 0.5
        return max(0.0, min(1.0, 0.35 * grounding + 0.25 * indirectness + 0.2 * specificity + 0.2 * graph))

    @staticmethod
    def _directness_score(
        query: str, validation: Dict[str, Any], difficulty: str = "moderate"
    ) -> float:
        target = {"easy": 0.68, "moderate": 0.46, "difficult": 0.24}.get(difficulty, 0.46)
        return max(
            0.0,
            min(1.0, target - validation.get("lexical_overlap", 0.0) * 0.25),
        )

    def _get_function_by_id(self, func_id: str, functions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return next((f for f in functions if self._function_id(f) == func_id), None)

    def _get_function_by_name(self, name: str, functions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return next((f for f in functions if f.get("name") == name or f.get("qualified_name") == name), None)

    def _generate_queries_for_difficulty(
        self, functions: List[Dict[str, Any]], count: int, templates: Dict[str, List[str]],
        difficulty: str, used_function_indices: set,
    ) -> List[Dict[str, Any]]:
        """Compatibility entry point; generation now uses evidence-aware candidates."""
        distribution = {
            "easy": (1.0, 0.0, 0.0),
            "moderate": (0.0, 1.0, 0.0),
            "difficult": (0.0, 0.0, 1.0),
        }.get(difficulty, DEFAULT_DIFFICULTY_DISTRIBUTION)
        result = self.generate_benchmark_queries(functions, count, distribution)
        return result

    def _generate_generic_queries(self, functions: List[Dict[str, Any]], count: int, used_function_indices: set) -> List[Dict[str, Any]]:
        return self.generate_benchmark_queries(functions, count, (1.0, 0.0, 0.0))

    def _infer_category(self, query: str) -> str:
        text = query.lower()
        for category, terms in {
            "multi_hop": ("intermediate", "two-stage", "through"),
            "cross_function": ("cooperate", "boundary", "components"),
            "control_flow": ("condition", "execution path"),
            "dependencies": ("enables", "supporting"),
            "data_flow": ("becomes", "path carries"),
            "input_output": ("produces", "result"),
            "error_handling": ("cannot complete", "handled"),
            "state_transition": ("state", "preserving"),
            "reasoning": ("why", "both"),
            "functionality": ("responsible", "implemented"),
        }.items():
            if any(term in text for term in terms):
                return category
        return "general"

    def _estimate_reasoning(self, query: str) -> str:
        words = len(self._tokens(query))
        return "high" if words >= 18 else ("medium" if words >= 11 else "low")


def create_benchmark_from_json_files(
    output_dir: Path, num_queries: int = 20,
    difficulty_distribution: Tuple[float, float, float] = DEFAULT_DIFFICULTY_DISTRIBUTION,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """Load base/enriched analysis files and generate an evidence-backed set."""
    merged: Dict[str, Dict[str, Any]] = {}
    paths = [output_dir / "base.json", *sorted(output_dir.glob("enhanced_*.json"))]
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            for file_data in data.get("files", []):
                for function in file_data.get("functions", []):
                    fid = str(function.get("id") or function.get("qualified_name") or function.get("name") or "")
                    if not fid:
                        continue
                    current = merged.get(fid, {})
                    # Later enhanced records contain enrichment/graph evidence;
                    # merge rather than letting base.json shadow that evidence.
                    for key, value in function.items():
                        if value not in (None, "", [], {}):
                            current[key] = value
                    merged[fid] = current
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load %s: %s", path, exc)
    return BenchmarkQueryGenerator(seed=seed).generate_benchmark_queries(
        list(merged.values()), num_queries, difficulty_distribution
    )


def save_benchmark_queries(queries: List[Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(queries, indent=2))
    logger.info("Saved %d benchmark queries to %s", len(queries), output_path)


def load_benchmark_queries(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        logger.warning("Benchmark file not found: %s", input_path)
        return []
    return json.loads(input_path.read_text())


def validate_benchmark_queries(queries: List[Dict[str, Any]], functions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate schema, safety, uniqueness, grounding, and difficulty balance."""
    generator = BenchmarkQueryGenerator(seed=0)
    valid_ids = {generator._function_id(func) for func in functions}
    report: Dict[str, Any] = {
        "total_queries": len(queries),
        "safe_queries": 0,
        "unsafe_queries": 0,
        "difficulty_distribution": {level: 0 for level in DIFFICULTIES},
        "category_distribution": defaultdict(int),
        "average_lexical_overlap": 0.0,
        "queries_with_high_overlap": 0,
        "empty_relevant_functions": 0,
        "invalid_function_ids": 0,
        "duplicate_queries": 0,
        "identifier_leaks": 0,
        "low_quality_queries": 0,
        "invalid_schema": 0,
        "average_reasoning_depth": 0.0,
        "average_directness_score": 0.0,
        "average_query_quality_score": 0.0,
        "validation_details": [],
    }
    seen: Set[str] = set()
    overlap_total = depth_total = directness_total = quality_total = 0.0
    for item in queries:
        text = str(item.get("query", "")).strip()
        ids = item.get("relevant_functions", [])
        required = bool(text) and isinstance(ids, list) and bool(ids)
        if not required:
            report["invalid_schema"] += 1
        if not ids:
            report["empty_relevant_functions"] += 1
        for fid in ids:
            if fid not in valid_ids:
                report["invalid_function_ids"] += 1
        key = " ".join(text.lower().split())
        if key in seen:
            report["duplicate_queries"] += 1
        seen.add(key)
        level = item.get("difficulty", "unknown")
        category = item.get("category", "unknown")
        if level in report["difficulty_distribution"]:
            report["difficulty_distribution"][level] += 1
        report["category_distribution"][category] += 1
        evidence = [func for func in functions if generator._function_id(func) in ids]
        target = evidence[0] if evidence else {}
        safe, details = generator._is_query_safe(text, target, evidence)
        overlap = details["lexical_overlap"]
        quality = float(item.get("query_quality_score", generator._score_query(text, evidence, level, int(item.get("reasoning_depth", 1)), details)))
        if safe:
            report["safe_queries"] += 1
        else:
            report["unsafe_queries"] += 1
        if details.get("contains_target_identifier"):
            report["identifier_leaks"] += 1
        if overlap > 0.3:
            report["queries_with_high_overlap"] += 1
        if quality < 0.45:
            report["low_quality_queries"] += 1
        overlap_total += overlap
        depth_total += float(item.get("reasoning_depth", 0) or 0)
        directness_total += float(item.get("directness_score", generator._directness_score(text, details)) or 0)
        quality_total += quality
        report["validation_details"].append({
            "query_id": item.get("query_id", "unknown"),
            "query": text[:160],
            "is_safe": safe,
            "lexical_overlap": overlap,
            "identifier_leak": details.get("contains_target_identifier", False),
            "quality_score": quality,
            "relevant_functions_count": len(ids),
        })
    if queries:
        count = len(queries)
        report["average_lexical_overlap"] = overlap_total / count
        report["average_reasoning_depth"] = depth_total / count
        report["average_directness_score"] = directness_total / count
        report["average_query_quality_score"] = quality_total / count
    report["category_distribution"] = dict(report["category_distribution"])
    return report
