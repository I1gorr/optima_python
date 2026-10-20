"""Small synthetic Optima base/enriched JSON for the research-pipeline tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# (key, name, qualified_name, source, calls, has_loop)
_FUNCS = [
    ("parse", "parse_json", "parse_json", "int parse_json(const char *text) { alloc_node(); read_token(text); return 0; }",
     ["read_token", "alloc_node"], False),
    ("read", "read_token", "read_token", "int read_token(const char *s) { while (is_digit(*s)) s++; return 1; }",
     ["is_digit"], True),
    ("alloc", "alloc_node", "alloc_node", "void *alloc_node(void) { return malloc(32); }", [], False),
    ("digit", "is_digit", "is_digit", "int is_digit(int c) { return c >= '0' && c <= '9'; }", [], False),
    ("print", "print_value", "print_value", "void print_value(double d) { format_number(d); }",
     ["format_number"], False),
    ("format", "format_number", "format_number", "char *format_number(double d) { return sprintf_buffer(d); }",
     [], False),
    ("hash", "hash_string", "hash_string", "unsigned hash_string(const char *s) { unsigned h = 5381; return h; }",
     [], False),
    ("read2", "read_token", "other::read_token", "int read_token(int fd) { return read(fd); }", [], False),
]
ENRICHMENT = {
    "parse": {"purpose": "Parse a JSON document into a tree", "behavior": "Allocates nodes and reads tokens",
              "summary": "JSON parser entry point", "keywords": ["json", "parse"], "concepts": ["tokenizer"],
              "dependencies": ["read_token", "std::string"]},
    "read": {"purpose": "Read a numeric token from the input", "behavior": "Advances over digit characters",
             "summary": "Scan digits", "keywords": ["digits", "scan"], "concepts": []},
    "alloc": {"purpose": "Allocate memory for a tree node", "behavior": "Calls malloc", "summary": "Node allocator"},
    "digit": {"purpose": "Check whether a character is a decimal digit", "behavior": "Compares to 0 and 9",
              "summary": "Digit predicate"},
    "print": {"purpose": "Print a numeric value", "behavior": "Formats then prints", "summary": "Printer"},
    "format": {"purpose": "Format a double as text", "behavior": "Uses a buffer", "summary": "Number formatter"},
}


def make_id(key: str) -> str:
    return f"src/{key}.c::{key}::1"


def build_base(include_duplicate: bool = False) -> Dict[str, Any]:
    ids = {name_key: make_id(name_key) for name_key, *_ in _FUNCS}
    by_name: Dict[str, List[str]] = {}
    for key, name, *_ in _FUNCS:
        by_name.setdefault(name, []).append(ids[key])
    functions = []
    edges = []
    called_by: Dict[str, List[str]] = {k: [] for k in ids}
    for key, name, qualified, source, calls, loop in _FUNCS:
        entries = []
        for callee in calls:
            target = next(k for k, n, *_ in _FUNCS if n == callee and k != "read2")
            entries.append({"name": callee, "qualified_name": callee, "resolved": True, "id": ids[target]})
            called_by[target].append(key)
            edges.append({"from": ids[key], "to": ids[target]})
        if "malloc" in source:
            entries.append({"name": "malloc", "qualified_name": "malloc", "resolved": False})
        cfg = ({"entry_node": "entry", "nodes": ["entry", "a", "b"],
                "edges": [{"from": "entry", "to": "a"}, {"from": "a", "to": "b"}, {"from": "b", "to": "a"}]}
               if loop else {"entry_node": "entry", "nodes": ["entry", "a"],
                             "edges": [{"from": "entry", "to": "a"}]})
        functions.append({
            "id": ids[key], "name": name, "qualified_name": qualified, "return_type": "int",
            "parameters": [{"name": "s", "type": "const char *"}], "source_code": source,
            "analysis_status": "success", "class_info": {"is_method": False},
            "calls": entries, "called_by": [], "dependencies": [], "cfg": cfg,
        })
    for fn in functions:
        key = next(k for k, i in ids.items() if i == fn["id"])
        fn["called_by"] = [{"id": ids[c], "name": c, "qualified_name": c} for c in called_by[key]]
    if include_duplicate:
        functions.append(dict(functions[0]))  # same id again
    return {"project": {"name": "synthetic", "language": "c"},
            "files": [{"path": "src/all.c", "relative_path": "src/all.c", "language": "c",
                       "functions": functions}],
            "graph": {"call_edges": edges}}


def build_enriched(base: Dict[str, Any], failed: Optional[List[str]] = None,
                   include_duplicate: bool = False, extra_id: bool = False) -> Dict[str, Any]:
    data = json.loads(json.dumps(base))
    functions = data["files"][0]["functions"]
    key_by_id = {make_id(k): k for k, *_ in _FUNCS}
    for fn in functions:
        key = key_by_id[fn["id"]]
        if key in ENRICHMENT:
            fn["enrichment"] = dict(ENRICHMENT[key], status="completed", usage={
                "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                evaluation={"latency_seconds": 1.5})
        elif key == "hash":
            fn["enrichment"] = {"status": "failed", "error": "bad json"}
        # "read2" gets no enrichment object at all (missing)
    if include_duplicate:
        functions.append(json.loads(json.dumps(functions[0])))
    if extra_id:
        stray = json.loads(json.dumps(functions[0]))
        stray["id"] = "src/stray.c::stray::1"
        functions.append(stray)
    data["enrichment_metadata"] = {"model": "fake/model", "total_functions": len(functions)}
    return data


def write_fixture(directory: Path, **kwargs) -> Dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = build_base(include_duplicate=kwargs.pop("duplicate_base", False))
    base_path = directory / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    enriched_path = directory / "enhanced_fake.json"
    enriched_path.write_text(json.dumps(build_enriched(base, **kwargs)), encoding="utf-8")
    return {"base": base_path, "enriched": enriched_path}
