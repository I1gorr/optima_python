"""Lexical retrieval: code-aware tokenisation, Okapi BM25, exact symbol matching.

BM25 is implemented here (about 80 lines) instead of adding ``rank_bm25``: the
project avoids new hard dependencies, and an owned implementation makes
tie-breaking and tokenisation deterministic and testable.  The same
``tokenize`` and ``normalize_query`` are used when indexing and querying.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_IDENT = re.compile(r"[A-Za-z_~][A-Za-z0-9_]*(?:::~?[A-Za-z_][A-Za-z0-9_]*)*")
_QUOTED = re.compile(r"[`'\"]([A-Za-z_~][A-Za-z0-9_:~]*)[`'\"]")
_CALL_FORM = re.compile(r"\b([A-Za-z_][A-Za-z0-9_:]*)\s*\(")


def normalize_query(query: object) -> str:
    """Shared query normalisation: NFKC, collapse whitespace, strip."""
    if query is None:
        return ""
    text = unicodedata.normalize("NFKC", str(query))
    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    """Lower-cased tokens; identifiers yield the whole token plus sub-words.

    ``cJSON_ParseWithOpts`` -> ``cjson_parsewithopts, cjson, parse, with, opts``.
    Sub-words make natural-language queries match identifiers while the whole
    token keeps exact-identifier matches precise.
    """
    tokens: List[str] = []
    for word in _WORD.findall(text or ""):
        whole = word.lower().strip("_")
        if len(whole) < 2 and not whole.isdigit():
            continue
        tokens.append(whole)
        chunks = [chunk for chunk in word.split("_") if chunk]
        pieces = []
        for chunk in chunks:
            if len(chunks) > 1:
                pieces.append(chunk.lower())      # snake_case part, e.g. "cjson"
            pieces.extend(p.lower() for p in _CAMEL.findall(chunk))
        pieces = [p for p in dict.fromkeys(pieces) if p != whole]
        tokens.extend(p for p in pieces if len(p) >= 2 or p.isdigit())
    return tokens


class BM25Index:
    """Okapi BM25 with the non-negative Lucene idf; deterministic ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: List[str] = []
        self._postings: Dict[str, List[Tuple[int, int]]] = {}
        self._doc_len: List[int] = []
        self._avg_len = 0.0
        self._idf: Dict[str, float] = {}

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> "BM25Index":
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        doc_len: List[int] = []
        for position, text in enumerate(texts):
            counts = Counter(tokenize(text))
            doc_len.append(sum(counts.values()))
            for term, freq in counts.items():
                postings[term].append((position, freq))
        self.ids = list(ids)
        self._postings = dict(postings)
        self._doc_len = doc_len
        n = len(ids)
        self._avg_len = (sum(doc_len) / n) if n else 0.0
        self._idf = {
            term: math.log(1.0 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self._postings.items()
        }
        return self

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Top-``k`` ``(id, score)`` with score > 0, ties broken by index order."""
        terms = list(dict.fromkeys(tokenize(normalize_query(query))))
        if not terms or not self.ids or k < 1:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for term in terms:
            plist = self._postings.get(term)
            if not plist:
                continue
            idf = self._idf[term]
            for position, freq in plist:
                norm = 1.0 - self.b + self.b * (self._doc_len[position] / self._avg_len
                                                if self._avg_len else 1.0)
                scores[position] += idf * freq * (self.k1 + 1.0) / (freq + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda item: (-round(item[1], 12), item[0]))
        return [(self.ids[position], score) for position, score in ranked[:k]]

    def approx_size_bytes(self) -> int:
        """Rough in-memory footprint of the postings (16 bytes per posting)."""
        return sum(len(p) for p in self._postings.values()) * 16 + sum(
            len(t) for t in self._postings) + 8 * len(self._doc_len)


class ExactSymbolMatcher:
    """Exact function-name / qualified-name lookup for symbol-shaped query tokens.

    A token counts as a symbol when it is quoted, written as a call (``name(``),
    qualified (``A::b``) or identifier-shaped (underscore, digit, or an internal
    capital).  Plain lower-case words are not treated as symbols, otherwise
    ordinary words such as "add" or "get" would match every function of that name.
    Match strength: qualified-name (3) > name (2) > case-insensitive name (1).
    """

    def __init__(self, entries: Iterable[Tuple[str, str, str]]):
        """``entries`` are ``(id, name, qualified_name)`` in corpus order."""
        self._order: Dict[str, int] = {}
        self._by_qualified: Dict[str, List[str]] = defaultdict(list)
        self._by_name: Dict[str, List[str]] = defaultdict(list)
        self._by_name_ci: Dict[str, List[str]] = defaultdict(list)
        for position, (ident, name, qualified) in enumerate(entries):
            self._order[ident] = position
            if qualified:
                self._by_qualified[qualified].append(ident)
            if name:
                self._by_name[name].append(ident)
                self._by_name_ci[name.lower()].append(ident)

    @staticmethod
    def symbols_in(query: str) -> List[str]:
        text = normalize_query(query)
        found: List[str] = []
        found.extend(match.group(1) for match in _QUOTED.finditer(text))
        found.extend(match.group(1) for match in _CALL_FORM.finditer(text))
        for match in _IDENT.finditer(text):
            token = match.group(0)
            if (("_" in token.strip("_")) or "::" in token or any(c.isdigit() for c in token)
                    or re.search(r"[a-z][A-Z]", token) or re.search(r"[A-Z]{2,}", token)):
                found.append(token)
        return list(dict.fromkeys(found))

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        best: Dict[str, float] = {}
        for symbol in self.symbols_in(query):
            for ident in self._by_qualified.get(symbol, ()):
                best[ident] = max(best.get(ident, 0.0), 3.0)
            for ident in self._by_name.get(symbol, ()):
                best[ident] = max(best.get(ident, 0.0), 2.0)
            for ident in self._by_name_ci.get(symbol.lower(), ()):
                best[ident] = max(best.get(ident, 0.0), 1.0)
        ranked = sorted(best.items(), key=lambda item: (-item[1], self._order[item[0]]))
        return ranked[:k]
