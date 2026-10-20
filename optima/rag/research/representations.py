"""Searchable text representations of a function: raw, semantic, compiler.

Each builder is a pure function of a :class:`FunctionRecord` (and, for the
compiler view, the corpus to resolve callee names).  Only fields that carry
retrieval signal are rendered: large blobs such as the AST, LLVM IR text and the
raw LLM response are deliberately not embedded.
"""

from __future__ import annotations

from typing import Dict, Optional

from optima.rag.document_constructor import _stringify

from .config import RepresentationConfig
from .corpus import Corpus, FunctionRecord, _usable_text

_SEMANTIC_LABELS = (
    ("purpose", "Purpose"), ("summary", "Summary"), ("behavior", "Behavior"),
    ("inputs", "Inputs"), ("outputs", "Outputs"), ("side_effects", "Side effects"),
    ("concepts", "Concepts"), ("keywords", "Keywords"), ("algorithm", "Algorithm"),
)


def signature(record: FunctionRecord) -> str:
    params = ", ".join(" ".join(part for part in (ptype, pname) if part)
                       for ptype, pname in record.parameters)
    prefix = f"{record.return_type} " if record.return_type else ""
    return f"{prefix}{record.qualified_name or record.name}({params})"


def identity_header(record: FunctionRecord) -> str:
    lines = [f"Function: {record.qualified_name or record.name}",
             f"Signature: {signature(record)}"]
    if record.file:
        lines.append(f"File: {record.file}")
    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n... (truncated)"


def raw_text(record: FunctionRecord, cfg: RepresentationConfig) -> str:
    parts = [identity_header(record)] if cfg.identity_header else [
        f"Function: {record.qualified_name or record.name}"]
    if record.source_code:
        parts.append("Source:\n" + record.source_code)
    return _truncate("\n".join(parts), cfg.raw_max_chars)


def semantic_text(record: FunctionRecord, cfg: RepresentationConfig) -> Optional[str]:
    """Rendered enrichment, or ``None`` when the policy excludes the function."""
    header = identity_header(record) if cfg.identity_header else (
        f"Function: {record.qualified_name or record.name}")
    if not record.has_semantic or not record.semantic:
        if cfg.missing_semantic == "exclude":
            return None
        if cfg.missing_semantic == "raw":
            return raw_text(record, cfg)
        return header  # "identity": retrievable by name/signature only
    parts = [header]
    for key, label in _SEMANTIC_LABELS:
        value = record.semantic.get(key)
        if not _usable_text(value):
            continue
        text = _stringify(value).strip()
        if text and text.lower() not in ("unknown", "none"):
            parts.append(f"{label}: {text}")
    return _truncate("\n".join(parts), cfg.semantic_max_chars)


def _semantic_lines(record: FunctionRecord) -> list:
    """Usable enrichment fields rendered as ``Label: text`` lines (no identity header)."""
    lines = []
    if record.has_semantic and record.semantic:
        for key, label in _SEMANTIC_LABELS:
            value = record.semantic.get(key)
            if not _usable_text(value):
                continue
            text = _stringify(value).strip()
            if text and text.lower() not in ("unknown", "none"):
                lines.append(f"{label}: {text}")
    return lines


def coupled_text(record: FunctionRecord, cfg: RepresentationConfig) -> str:
    """One chunk = identity header + enrichment description + source code, scored together.

    The identity header appears once.  Enrichment and source are each capped (``semantic_max_chars`` /
    ``raw_max_chars``) before joining, then the whole chunk is capped at ``coupled_max_chars``.  Without usable
    enrichment the chunk is the raw text (so every function stays retrievable; the number of such functions is
    reported by the corpus coverage statistics).
    """
    lines = _semantic_lines(record)
    if not lines:
        return _truncate(raw_text(record, cfg), cfg.coupled_max_chars)
    header = identity_header(record) if cfg.identity_header else (
        f"Function: {record.qualified_name or record.name}")
    description = _truncate("Description:\n" + "\n".join(lines), cfg.semantic_max_chars)
    source = _truncate("Source:\n" + record.source_code, cfg.raw_max_chars) if record.source_code else ""
    parts = [header, description, source] if cfg.coupled_order == "semantic_first" else [header, source, description]
    return _truncate("\n".join(p for p in parts if p), cfg.coupled_max_chars)


def compiler_text(record: FunctionRecord, corpus: Corpus, cfg: RepresentationConfig) -> str:
    """Structural facts only: call relationships, class membership, CFG summary."""
    header = identity_header(record) if cfg.identity_header else (
        f"Function: {record.qualified_name or record.name}")
    parts = [header]
    if record.class_name:
        kind = "method of" if record.is_method else "member of"
        parts.append(f"Scope: {kind} {record.class_name}")

    def names(ids):
        out = []
        for ident in ids:
            target = corpus.records.get(ident)
            out.append(target.qualified_name or target.name if target else ident)
        return out

    callees = names(record.calls) + list(record.external_calls)
    if callees:
        parts.append("Calls: " + ", ".join(dict.fromkeys(callees)))
    callers = names(record.called_by)
    if callers:
        parts.append("Called by: " + ", ".join(dict.fromkeys(callers)))
    if record.cfg:
        cfg_info = record.cfg
        parts.append(
            f"Control flow: {cfg_info['blocks']} basic blocks, {cfg_info['edges']} edges, "
            f"{cfg_info['branch_blocks']} branching blocks, cyclomatic complexity "
            f"{cfg_info['cyclomatic']}, {'contains a loop' if cfg_info['has_loop'] else 'no loop'}"
        )
    return _truncate("\n".join(parts), cfg.compiler_max_chars)


def build_texts(corpus: Corpus, representation: str,
                cfg: RepresentationConfig) -> Dict[str, str]:
    """Texts for one representation, keyed by function ID, in corpus order.

    A function missing from the result was excluded by policy (only possible for
    ``semantic`` with ``missing_semantic="exclude"``); it is never silently lost
    otherwise.
    """
    texts: Dict[str, str] = {}
    for ident, record in corpus.records.items():
        if representation == "raw":
            text: Optional[str] = raw_text(record, cfg)
        elif representation == "semantic":
            text = semantic_text(record, cfg)
        elif representation == "compiler":
            text = compiler_text(record, corpus, cfg)
        elif representation == "coupled":
            text = coupled_text(record, cfg)
        else:
            raise ValueError(f"unknown representation {representation!r}")
        if text is not None:
            texts[ident] = text
    return texts
