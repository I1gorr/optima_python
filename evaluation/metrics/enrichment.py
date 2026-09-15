"""Structured enrichment metrics and field availability checks."""

from __future__ import annotations

from typing import Any


FIELDS = ("purpose", "behavior", "inputs", "outputs", "side_effects")


def normalized_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    return {str(value).strip().lower()} if str(value).strip() else set()


def field_completeness(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Return observed non-empty rates; this is not semantic accuracy."""
    if not records:
        return {field: None for field in FIELDS}
    return {
        field: sum(bool(normalized_set(record.get(field))) for record in records) / len(records)
        for field in FIELDS
    }


def reference_metrics(*_args: Any, **_kwargs: Any) -> dict[str, None]:
    """Reference-grounded scores are unavailable without independent labels."""
    return {f"{field}_f1": None for field in FIELDS} | {"overall_f1": None}
