# Retrieval benchmarks

Ground truth for function retrieval. One directory per project:

| File | Contents |
|---|---|
| `chess/structural_v1.json` | 174 queries for `Chess-Engine-in-cpp-master` (164 functions) |
| `cjson/structural_v1.json` | 182 queries for `cJSON-master` (1201 functions, including the vendored Unity test framework) |
| `chess/structural_v2.json` | v1 + 49 more hand-authored behavioural queries (223 queries) |
| `cjson/structural_v2.json` | v1 + 58 more hand-authored behavioural queries (240 queries) |
| `xxhash/structural_v1.json` | 225 queries for `xxHash-dev` (339 functions): 89 hand-authored behavioural + generated categories |
| `*/behavioral_authored.json`, `*_v2.json` | the hand-authored natural-language queries and their targets (input to the builder) |

**Versions.** `structural_v2` keeps every v1 query unchanged (same seed, same generated structural queries) and
only adds behavioural queries; the version string differs, so v1 and v2 results are never pooled. Suite specs:
`configs/suites/{chess,cjson}-v2.json`, `configs/suites/xxhash{,-grid}.json` (planned, not yet run). The added
queries mostly target previously unused helpers/utilities, so they are harder to find by name and cover more
of each corpus. xxHash is behaviour-heavy (89 of 225) because it has few natural identifier-style queries.

## Schema (`schema_version` 1)

```jsonc
{
  "benchmark_id": "chess-engine-structural",
  "query_set_version": "1.0.0",       // recorded in every experiment result
  "ground_truth_version": "1.0.0",
  "project": "...", "base_sha256": "...",  // the base.json the IDs refer to
  "labeling": { ... }, "categories": { ... }, "unsupported_categories": { ... },
  "queries": [{
    "query_id": "caller-007", "category": "caller", "query": "Which functions call X?",
    "relevance": { "<function id>": 1.0 },   // graded: > 0 relevant; nDCG uses the grades
    "relevant_functions": ["<function id>"], // same IDs, legacy field
    "seed": "<function id>",                  // target the query is about (bootstrap cluster key)
    "label_source": "resolved_call_graph"
  }]
}
```

Every relevant function is identified by its stable function ID from `base.json`
(`file::qualified_name::line`), never by name. `validate-benchmark` checks the IDs
against a `base.json` and the stored `base_sha256`.

## How labels were created

Independent of every system under test: labels come only from the compiler-produced
`base.json` (names, source text, resolved call edges, LLVM CFG) or were written by hand.
No LLM enrichment, embedding or retrieval output was used. They are regenerated
deterministically (`--seed 13`) by `optima.rag.research.benchmark`.

| Category | Query shape | Label rule |
|---|---|---|
| `exact_name` | "Where is the function `f` defined?" | every function whose `name == f` |
| `identifier_api` | "Which functions use the identifier `t`?" | functions whose source contains `t` (1–4 hits, `t` symbol-shaped and not a defined function) |
| `behavioral` | natural-language description of behaviour | hand-authored (below) |
| `control_flow` | callers of `f` that contain a loop; most complex function in a file | CFG back-edge / cyclomatic complexity from the LLVM CFG |
| `caller` | "Which functions call `f`?" | resolved callers (union of `called_by`, `calls`, `graph.call_edges`) |
| `callee` | "Which functions does `f` call?" | resolved callees |
| `cross_function` | callers of both `a` and `b` / callees shared by `a` and `b` | set intersection of the above |
| `multi_hop` | functions reaching `f` (or reachable from it) within two calls | graded: 1 hop = 2, 2 hops = 1 |
| `dependency` | **not generated** | the `dependencies` field is empty in every bundled `base.json` |

Hand-authored `behavioral` queries (45 chess, 42 cJSON): written by reading the function
source / API in `base.json` before any retrieval run; grade 2 = implements the behaviour,
grade 1 = helper or close sibling. **Single annotator (AI-assisted); not independently
reviewed.** Treat the behavioural labels as unreviewed until a second person checks them.

## Known limitations

* The structural categories (`caller`, `callee`, `control_flow`, `cross_function`, `multi_hop`)
  are labelled from the same call graph that graph expansion traverses. Gains from graph
  expansion there show the graph is useful for *structural questions*; they say little about
  general retrieval.
* `exact_name` and `identifier_api` queries contain the identifier verbatim; they test lexical
  matching, so BM25/exact-symbol retrieval is expected to excel there.
* Several queries share a target function; analyses use a cluster bootstrap on `seed`.
* The older `output/rag_report/benchmark/queries.json` (1600 queries) is **not** used: it was
  generated from the enrichment text of the models being scored (circular), many queries are
  ungrammatical, and its IDs refer to an older 96-function base. The loader still accepts that
  legacy list format.

## Rebuilding

```bash
python -m optima.rag build-benchmark \
  --base-json optima_outputs/base/chess-engine-in-cpp-master/base.json \
  --authored benchmarks/chess/behavioral_authored.json \
  --out benchmarks/chess/structural_v2.json --version 2.0.0 --benchmark-id chess-engine-structural
python -m optima.rag validate-benchmark --base-json <base.json> --benchmark <file>
```

`build-benchmark` refuses to overwrite an existing file. Change the version when labels change:
results record `query_set_version` / `ground_truth_version` so runs on different label sets are
never silently compared.


## Developer-style benchmarks (`developer_v1.json`) and tuning data

| File | Contents |
|---|---|
| `{chess,cjson,xxhash}/developer_v1.json` | 38 / 47 / 40 queries written the way a developer asks (see intents below) |
| `*/developer_authored.json` | the authored input (query, intent, targets / string literal / call-graph reference) |
| `*/tuning_literals.json` | 30 generated "where is this message printed" queries per project, **tuning only** |

Intents: `vague_memory` ("the function that ... I forgot the name"), `task` ("how do I ..."), `error_message`
(a log/error string -> the function that contains it), `concept` ("how does X work"), `impact` ("what calls / what
does X depend on"). Labels: hand-resolved targets (grade 2 = implements, 1 = helper), `string_literal` (every function
whose source contains the literal -- mechanical), or the resolved call graph (`impact`; the same graph graph-expansion
uses, so impact gains show graph usefulness, not independent retrieval ability).

**Held-out protocol.** Query-instruction prefixes and routing profiles (`configs/routing_profiles.json`) were
chosen on `structural_v2` / `structural_v1` + `tuning_literals.json` only (`scripts/tune_routing.py`); the developer
benchmarks were run once afterwards (`configs/suites/dev-*.json`, results in `experiments_dev/`, report in
`reports/optima_analytics_dev/`). The router's regular expressions were written by the same author as the developer
queries, so they may be tuned to that phrasing; validate on queries from other people before relying on the numbers.
Single annotator (AI-assisted), not independently reviewed.
