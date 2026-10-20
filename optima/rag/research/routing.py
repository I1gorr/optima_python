"""Rule-based query routing: pick fusion weights from the *text* of the query.

The classifier looks only at the query string (never at benchmark categories or labels), so a routed
pipeline can be used on real queries.  Four classes:

``literal``     a quoted message / log line / string ("where is 'Failed to open the file.' printed")
``structural``  asks about callers/callees/dependencies/impact and names a symbol
``symbol``      names a code identifier (camelCase, snake_case, ``A::b``, digits, ``f()``) but is not structural
``natural``     everything else (natural-language description, 'how do I ...')

Each class maps to a *profile*: fusion weights for dense / BM25 / exact-symbol retrieval, per-representation
weights and the graph weight.  Profiles are data (``RoutingConfig.profiles`` overrides ``DEFAULT_PROFILES``),
so they can be tuned without touching code.  With routing disabled nothing here is used.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

from .lexical import ExactSymbolMatcher, normalize_query

CLASSES = ("literal", "structural", "symbol", "natural")
PROFILE_KEYS = ("dense_weight", "bm25_weight", "exact_weight", "graph_weight", "representation_weights")

_QUOTED = re.compile(r"""(['"`])(?P<body>[^'"`]{6,}?)\1""")
_STRUCTURAL = re.compile(
    r"\b(calls?|called|callers?|callees?|invokes?|invoked|depends?|depend(?:s|ed)?\s+on|relies|rely|reach(?:es)?|"
    r"breaks?|uses?|used|impact)\b", re.IGNORECASE)

# "which functions use the identifier X" is a text search for a token, not an impact question.
_TEXT_SEARCH = re.compile(r"\b(identifier|macro|constant|token|references?|mentions?)\b", re.IGNORECASE)

# Neutral by default: identical to the un-routed pipeline until tuned profiles are supplied.
DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {c: {} for c in CLASSES}


def classify_query(query: str) -> str:
    text = normalize_query(query)
    for match in _QUOTED.finditer(text):
        if " " in match.group("body").strip():
            return "literal"
    symbols = ExactSymbolMatcher.symbols_in(text)
    if symbols and _STRUCTURAL.search(text) and not _TEXT_SEARCH.search(text):
        return "structural"
    if symbols:
        return "symbol"
    return "natural"


def profile_for(query_class: str, overrides: Optional[Mapping[str, Mapping[str, Any]]]) -> Dict[str, Any]:
    profile = dict(DEFAULT_PROFILES.get(query_class, {}))
    profile.update(dict((overrides or {}).get(query_class, {})))
    return profile
