"""Generate a TUNING-ONLY set of 'where is this message printed' queries from string literals.

Labels are mechanical (functions whose source contains the literal) and the sentences are templated, so this set
shares nothing with the hand-written developer benchmark: literals that appear in ``developer_authored.json`` are
excluded.  It exists only so the ``literal`` routing profile can be tuned without touching the held-out set.
"""
import json, random, re, sys
from pathlib import Path
from optima.rag.research.corpus import load_corpus

PROJECTS = {"chess": "chess-engine-in-cpp-master", "cjson": "cJSON-master", "xxhash": "xxHash-dev"}
TEMPLATES = ["where does it print '{lit}'", "which function outputs '{lit}'", "find the code that logs '{lit}'",
             "who is responsible for the message '{lit}'"]
LIT = re.compile(r'"((?:[^"\\\n]|\\.){14,70})"')

for ds, base in PROJECTS.items():
    corpus = load_corpus(Path(f"optima_outputs/base/{base}/base.json"))
    dev = Path(f"benchmarks/{ds}/developer_authored.json").read_text()
    seen = {}
    for r in corpus.records.values():
        for m in LIT.finditer(r.source_code):
            lit = m.group(1)
            if "\\" in lit or "%" in lit or len(re.findall(r"[A-Za-z]{3,}", lit)) < 2 or " " not in lit or lit in dev:
                continue
            seen.setdefault(lit, set()).add(r.id)
    cands = sorted((l, sorted(ids)) for l, ids in seen.items() if len(ids) <= 3)
    random.Random(7).shuffle(cands)
    queries = []
    for i, (lit, ids) in enumerate(cands[:30]):
        queries.append({"category": "literal", "query": TEMPLATES[i % len(TEMPLATES)].format(lit=lit),
                        "relevance": {j: 2.0 for j in ids}, "relevant_functions": ids, "seed": ids[0],
                        "label_source": "source_string_literal", "query_id": f"literal-{i + 1:03d}"})
    doc = {"schema_version": 1, "benchmark_id": f"{ds}-literal-tuning", "query_set_version": "tuning-1", "ground_truth_version": "tuning-1",
           "project": corpus.meta.get("project", ""), "base_sha256": corpus.meta.get("base_sha256", ""),
           "labeling": {"independence": "templated sentences + mechanical literal labels; TUNING ONLY, never reported as a result"},
           "categories": {"literal": "message / string literal to code"}, "unsupported_categories": {},
           "counts": {"literal": len(queries)}, "queries": queries}
    Path(f"benchmarks/{ds}/tuning_literals.json").write_text(json.dumps(doc, indent=1))
    print(ds, len(queries), "queries from", len(cands), "candidate literals")
