# Optima: does compiler (LLVM) data and LLM enrichment improve code retrieval?

This folder is a self-contained summary of the experiments: what was measured, the graphs (each explained below, with the
focus on **recall, precision and latency**), the underlying tables, the benchmark queries and the configuration. The tables
in this README are generated from the result files in `tables/`, and the ranges quoted in the prose were checked against them.

## 1. The question and the system

Optima analyses C/C++ projects with the LLVM/clang compiler (AST, control-flow graph, call graph) and enriches every
function with LLM-generated descriptions. We ask: **given a developer's question, how well do different retrieval pipelines
find the right function, and does the LLVM-derived data and the LLM enrichment make them better or faster?**

* **Retrieval task.** A query (for example "how do I print json into a buffer I already allocated") must return the
  right function ID. One function = one retrievable unit. A query usually has 1-2 relevant functions.
* **Projects.** chess engine (C++, 164 functions), cJSON (C, 1,201 functions incl. its test framework), xxHash (C, 339 functions).
* **Embedding models (3).** BGE-small (0.033B parameters), BGE-base (0.109B), Qwen3-Embedding-0.6B (0.596B).
* **Enrichment models (10 LLMs, 1.5B-14.8B parameters; 9 on cJSON).** Qwen2.5 (1.5B/3B/7B/14B and Coder 7B/14B), Qwen3-14B,
  Mistral-7B, Ministral-8B, SmolLM2-1.7B; each produced its own JSON description of every function. The held-out developer
  runs of the improved pipeline use 4 of them (Qwen2.5-Coder-14B, Qwen2.5-14B, Qwen2.5-7B, Ministral-8B).
* **Pipelines.**
  * `baseline_rag`: dense retrieval over the raw function source.
  * `semantic_enrichment_rag`: dense retrieval over the LLM enrichment text.
  * `compiler_hybrid_rag`: raw + enrichment + compiler-text indexes, dense + BM25 + exact-symbol match fused with reciprocal-rank
    fusion, plus bounded call-graph expansion of the top results.
  * **IMPROVED pipeline (this work):** `compiler_hybrid_rag` + **rule-based query routing** (the system reads the query text and
    chooses the fusion weights: quoted messages and identifiers use BM25 on raw source, plain-language questions use dense
    retrieval on the enrichment text, "what calls/uses X" questions use dense retrieval + the call graph) + the embedding
    model's **query instruction prefix**.

## 2. How it was measured

* **Metrics (per query, then averaged).** *Recall@K* = relevant functions found in the top K / all relevant functions.
  *Precision@K* = relevant functions in the top K / K, so with 1-2 relevant functions per query it is capped near 1/K
  (P@5 can never exceed 0.2-0.4 here). *MRR* = mean of 1/rank of the first relevant function. *Latency* = end-to-end
  time per query (query encoding + retrieval + fusion + graph + optional rerank), one query at a time on one RTX 4070 laptop
  GPU; **median per query** is reported because the first run of each experiment includes one-off GPU warm-up spikes.
* **Benchmarks.**
  * **Developer benchmark (held-out): 125 queries** (chess 38, cJSON 47, xxHash 40) written the way a developer asks
    (vague memory, task "how do I...", error message to code, concept, impact "what calls X"); labels from hand-resolved
    targets, string literals in the source, or the call graph. **Nothing was tuned on it**: the routing weights and
    prefixes were chosen on a separate tuning benchmark first, then these queries were run once. Files: `benchmarks/`.
  * **Structural benchmark (223 / 240 / 225 queries):** templated + hand-written queries (exact name, identifier, behavioural,
    caller/callee, multi-hop...), used for the enrichment-model comparison (figures 14-17) and for tuning.
* **Statistics.** Differences are paired on identical queries, with 95% cluster-bootstrap intervals (queries about the same
  function are resampled together), cluster permutation tests, and Holm correction across many comparisons. A `*` in a table
  below means the 95% interval excludes zero.

## 3. Headline result (held-out developer queries)

Mean over the 3 embedding models and (for enrichment pipelines) the enrichment models tested. Full tables: `tables/developer_benchmark/`.

| project | pipeline | Recall@10 | Precision@5 | MRR | median latency (ms) |
|---|---|---|---|---|---|
{headline_table}

**Improvement of the IMPROVED pipeline over `baseline_rag`:**

| project | Recall@10 | Precision@5 | MRR | median latency | vs default compiler_hybrid_rag: R@10 / P@5 / latency |
|---|---|---|---|---|---|
{imp_lines}

* **Recall and precision go up on all three projects.** Latency of the improved pipeline relative to `baseline_rag` is {lat_range}
  (the largest increase is on cJSON, the biggest corpus), and it is lower than the default compiler pipeline on all three projects
  (column "vs default"). Retrieval takes roughly 8-18 ms per query in every case (median over the table above).
* Significance: improved vs `baseline_rag` was significantly better on MRR in {sg_base} of {n_base} (embedding x enrichment-model)
  pairs after Holm correction, and vs the default compiler pipeline in {sg_def} of {n_def}. Each project has only 38-47 queries,
  so many individual comparisons are inside the noise; the direction is consistent across projects.

## 4. Figure guide

All figures are in `figures/` (300 dpi PNG). Tags show the focus: **[RECALL] [PRECISION] [LATENCY]**.

### 4.1 Headline

**`00_headline_before_after.png` [RECALL][PRECISION][LATENCY].** Four panels (Recall@10, Precision@5, MRR, median latency) with
one group of bars per project and four pipelines per group: baseline, default compiler pipeline, semantic pipeline with query
instruction, and the improved pipeline. *Read it as:* higher bars are better in the first three panels, lower in the last.
*Takeaway:* the improved pipeline (pink) has the tallest accuracy bars on every project; its latency is slightly above the baseline and
below the default compiler pipeline.

### 4.2 Recall and precision

**`01_precision_recall_at_k_by_pipeline.png` [PRECISION][RECALL].** Precision@K (top row) and Recall@K (bottom row) for K = 1, 3, 5, 10,
one column per project. *Read it as:* a curve above the others is better at that K; precision falls with K and recall rises
by construction. *Takeaway:* the improved pipeline (pink) is highest in recall at every K and highest in precision at every K,
including P@1 (the first result being right).

**`02_precision_recall_by_enrichment_model.png` [PRECISION][RECALL].** Precision@5 (top) and Recall@5 (bottom) of the semantic
pipeline for each LLM used to write the enrichment, with the raw-source baseline as a dashed line. *Read it as:* a bar past the
dashed line means that enrichment model beats searching raw code. *Takeaway:* {enr_extremes} Enrichment models differ by a few
hundredths, so only the large gaps (very small models) should be read as real.

**`03_precision_recall_cutoffs_by_embedding.png` [PRECISION][RECALL].** Precision against recall at the evaluated cutoffs
(K = 1, 3, 5, 10; the number labels K) per embedding model, for the baseline (dashed) and the default compiler pipeline (solid).
*Read it as:* up and to the right is better. It is a four-point precision-recall-at-cutoff plot, not a full PR curve (only the
top 10 are evaluated).

**`04_precision_recall_latency_by_model_category.png` [PRECISION][RECALL][LATENCY].** Precision@5, Recall@5 and latency grouped by
model size bucket: enrichment models (<4B / 4-10B / >=10B parameters) in the semantic pipeline (top row) and embedding models
(<0.1B / 0.1-0.5B / >=0.5B) in the compiler pipeline (bottom row). *Takeaway (Recall@5 of the semantic pipeline by enrichment-model size,
small / medium / large):* {cat_enr}. *Latency by embedding-model size (compiler pipeline, median ms, small / medium / large):* {cat_emb}.
Buckets with one embedding model carry no spread information.

**`05_improvement_by_developer_intent.png` [RECALL][PRECISION] (as MRR).** Paired MRR gain of the improved pipeline over the default
compiler pipeline (orange) and over the baseline (blue), per developer intent, with 95% intervals. *Takeaway:* the largest and most consistent gains are on "impact" (what calls / uses X) questions; error-message and concept questions
also gain in most cases; "task" and "vague memory" gains are inconsistent (relative to the baseline they range from slightly negative to
clearly positive depending on the project). Each intent has only 6-18 queries, so intervals are wide and most are not individually significant.

### 4.3 Latency

**`06_latency_by_embedding_and_pipeline.png` [LATENCY].** Median per-query latency (bars, log scale) and p95 (tick) per embedding model
and pipeline. *Read it as:* shorter is faster. *Takeaway:* latency is set mainly by the **embedding model**; the improved pipeline is
not slower than the default compiler pipeline, and compared with the plain baseline it adds roughly 1.5-2.5 ms on the BGE models (less in relative
terms on Qwen3-Embedding). Per embedding model (mean over the three projects), baseline -> improved:

| embedding model | Recall@10 | Precision@5 | median latency |
|---|---|---|---|
{emb_table}

**`07_latency_by_enrichment_model.png` [LATENCY].** Top: retrieval latency of the compiler pipeline for each enrichment model.
Bottom: the one-off cost of *generating* the enrichment (seconds per function, from the enrichment logs). *Takeaway:* which LLM wrote the
enrichment hardly changes retrieval latency, but generating it costs seconds per function on a 14B model (hours per project); this is
the real cost of enrichment (paid once, offline).

**`08_accuracy_vs_latency_by_data_source.png` [RECALL][LATENCY].** Recall@10 against latency for every on/off combination of the three
data sources inside the routed stack (E = LLM enrichment, C = LLVM compiler text, G = LLVM call graph; "raw" = raw source only) plus the
baseline (squares). *Read it as:* up and to the left is better. *Takeaway:* combinations containing G (and, on cJSON/xxHash, E) reach the highest recall; adding C does not move a point up (recall unchanged) and
changes latency by only a fraction of a millisecond to about 1.5 ms. Note the narrow axes: the horizontal differences between combinations are
about 1-2 ms.

### 4.4 Where the improvement comes from (ablation, held-out developer queries)

A controlled 2x2x2 experiment switched the LLVM compiler text (C), the LLVM call graph (G) and the LLM enrichment (E) on and off
independently (all 8 combinations; `tables/ablations/attribution_*.csv`). Contributions are exact Shapley values (average marginal
gain over all orders of adding them), so they do not depend on the order and add up to the total gain. The first two components are
sequential steps from the baseline.

| contribution to Recall@10 | chess | cJSON | xxHash |
|---|---|---|---|
{comp_r10}

| contribution to Precision@5 | chess | cJSON | xxHash |
|---|---|---|---|
{comp_p5}

| contribution to MRR | chess | cJSON | xxHash |
|---|---|---|---|
{comp_mrr}

| contribution to median latency (ms) | chess | cJSON | xxHash |
|---|---|---|---|
{comp_lat}

(`*` = 95% interval excludes zero.)

**`09_contribution_of_each_data_source.png` [RECALL][PRECISION][LATENCY].** The table above as stacked bars (one panel per metric): grey =
query instruction, blue = hybrid retrieval + routing on raw source, orange = LLVM compiler text, green = LLVM call graph, pink = LLM
enrichment, diamond = net total. *Takeaway:* the **LLVM call graph is the largest contributor to Recall@10 and Precision@5 on all three projects** (and to MRR on chess and xxHash;
on cJSON the LLM enrichment is the largest MRR contributor). The LLM enrichment adds a smaller share that is significant on cJSON and xxHash,
the **LLVM compiler text contributes nothing**, and the median latency changes by a few milliseconds at most (the largest total change is on cJSON).

**`10_contribution_with_confidence_intervals.png` [RECALL].** The same Shapley contributions to Recall@10 (top) and MRR (bottom) with 95%
intervals. *Read it as:* an interval that crosses the vertical zero line is not distinguishable from no effect. The call graph's interval
excludes zero on all three projects.

**`11_contribution_by_developer_intent.png` [RECALL].** Recall@10 contribution of each data source for each developer intent.
*Takeaway:* the call graph helps almost only the "impact" questions (+0.38 to +0.53); the enrichment helps plain-language intents
(task, concept, vague memory) on cJSON and xxHash, with no consistent effect on chess. **Caveat:** impact queries are labelled from the same call graph, so this shows the graph is
useful for call-graph questions, not that it improves retrieval in general.

### 4.5 Does the LLVM compiler text help if it is allowed to? (tuned weight is 0)

The routing tuning gave the compiler-text index weight 0 in every query class. To check this was not an artefact, its weight was swept on
the tuning data and then switched on (weight 1, chosen in advance) on the held-out queries.

**`12_compiler_text_weight_sweep_tuning_data.png` [RECALL (as MRR)].** Left: MRR of each query class as the weight rises (MRR by class, call
graph on): 

| compiler-text weight | literal | symbol | natural | structural |
|---|---|---|---|---|
{sweep_rows}

Accuracy falls in every class. Middle/right: for call-graph questions with the **graph switched off**, the compiler text does help (structural MRR
{sw_off0:.3f} at weight 0 -> {sw_off4:.3f} at weight 4) because it contains the callers and callees, but the call graph itself reaches
{sw_on0:.3f}, so with the graph on the text is redundant.

**`13_compiler_text_switched_on_heldout.png` [RECALL][LATENCY].** Top: gain of the full improved stack over the baseline with the compiler text
off (green) and on (orange). Bottom: the compiler text's own contribution. Recall@10 gain of the full stack, text OFF: {off_r10} (chess / cJSON /
xxHash); text ON: {on_r10}. The compiler text's own Recall@10 contribution at weight 1: {ct_r10}; MRR contribution: {ct_mrr}. It also adds
latency. **Conclusion: leave the compiler text out of the routed stack; the call graph provides the structural signal.**

### 4.6 Context: enrichment on the larger structural benchmark (223 / 240 / 225 queries)

On this benchmark, mean Recall@10 / Precision@5 (default settings, mean over embedding and enrichment models):

| project | baseline_rag | semantic_enrichment_rag | compiler_hybrid_rag |
|---|---|---|---|
{v2_rows}

**`14_enrichment_vs_raw_by_enrichment_model.png` [(MRR)].** MRR with each enrichment model's text (three pipelines) against the raw-code baseline (dashed).
*Takeaway:* the best models are above the baseline, the smallest models (SmolLM2-1.7B, Qwen2.5-1.5B) fall below it.

**`15_enrichment_effect_gain_heatmap.png`.** MRR gain of the enriched corpus over the raw corpus for every enrichment model x embedding model
(blue = better, red = worse); gains shrink or become negative with the strongest embedding model (Qwen3-Embedding).

**`16_enrichment_effect_forest_with_ci.png`.** The same comparisons with 95% intervals and Holm-corrected significance (blue dots), against the same
retrieval stack without enrichment text. Most individual comparisons are not significant; the signal is in the overall pattern.

**`17_pipelines_by_query_family.png`.** MRR per query family (lexical / semantic / structural). *Takeaway:* enrichment helps natural-language
(semantic) questions and not lexical or structural ones, and structural questions are weak for every pipeline: why routing by query type works.

### 4.7 One chunk per function: enrichment and source coupled in a single document

Every function is already one chunk. The earlier design kept the LLM enrichment in a *separate* index and fused its results with the
raw-code index (rank fusion). The **coupled chunk** puts the identity header, the enrichment description and the source into ONE document that is
embedded once, so both are matched together. Two orders were pre-specified: enrichment first (main variant) and source first (the encoder
truncates long inputs, so the order decides what survives; about a quarter of coupled chunks exceed the 512-token limit). Runs: 4 enrichment models x
3 embedding models per project, both benchmarks. Figures 18-19 use the structural benchmark (more queries), figures 20-21 the held-out developer queries.

Paired differences, coupled minus separate indexes (mean over the 12 embedding x enrichment-model pairs per project; "n/12" = pairs with a Holm-significant MRR change):

| retrieval mode (structural benchmark) | project | dRecall@10 | dPrecision@5 | dMRR | MRR significant |
|---|---|---|---|---|---|
{cp_struct}

| retrieval mode (held-out developer queries) | project | dRecall@10 | dPrecision@5 | dMRR | MRR significant |
|---|---|---|---|---|---|
{cp_dev}

Mean levels on the structural benchmark (all runs of each variant; median latency and index size per variant):

| project | variant | Recall@10 | Precision@5 | MRR | median latency (ms) | index size (MB) |
|---|---|---|---|---|---|---|
{cp_levels}

**`18_coupled_chunk_overview_structural.png` [RECALL][PRECISION][LATENCY].** Recall@10, Precision@5, MRR and median latency of raw-only,
enrichment-only, separate raw+enrichment indexes, and the coupled chunk (both orders); top row dense retrieval, bottom row dense + BM25 + exact match.
*Read it as:* higher is better except the latency panel. *Takeaway:* the coupled chunk (green/purple) has the highest recall in every project and
retrieval mode. Median latency is within 1-2 ms of the separate indexes in either direction (slightly higher for dense, slightly lower for hybrid), which is of the order of the variation between separately run suites, so no latency conclusion is drawn; the dense index is half the size.

**`19_coupled_chunk_paired_differences_structural.png` [RECALL (as MRR)].** One dot per (embedding x enrichment model): the paired MRR difference
coupled minus separate with a 95% interval; blue dots are Holm-significant. *Takeaway:* with the two BGE embedding models the dense gain is
consistent (chess, cJSON); with Qwen3-Embedding-0.6B it is zero or slightly negative. For hybrid search the gain is positive for all three embedding
models and significant for all 12 pairs on cJSON.

**`20_coupled_chunk_overview_developer.png` [RECALL][PRECISION][LATENCY].** The overview plot on the 125 held-out developer queries.

**`21_coupled_chunk_paired_differences_developer.png` [RECALL (as MRR)].** The paired-difference plot on the same queries.
*Takeaway (both):* the direction is the same (coupled is mostly better than separate indexes), but there are only 38-47 queries per project, so few
pairs are individually significant.

**Reading the results.** The coupled chunk is better than the separate-index design while using half the index (dense) and the same query time;
against an enrichment-only chunk it is about equal (dMRR near 0): most of the gain over raw source comes from adding the enrichment text, and
coupling mainly avoids the loss that the separate-index fusion incurs. The by-category table in `reports/final_results_structural_benchmark.md` (section 9)
shows the dense gain comes from behavioural (natural-language) queries, while exact-name and caller/callee queries are flat or slightly lower.
Source-first versus enrichment-first has no clear winner on MRR (differences up to 0.02); source-first gives slightly higher Recall@10.
The coupled chunk was **not** tested inside the routed improved stack (section 3); that would need re-tuning the routing profiles on the tuning data.

## 5. What improved, what did not

| change | recall / precision | latency | verdict |
|---|---|---|---|
| Query-type routing (+ call-graph expansion for "what uses X") | clear gain (sections 3-4.4) | no extra cost vs the default compiler pipeline (lower on all 3 projects); a few ms above plain baseline | **keep** |
| Query instruction prefix for embedding models | small, mostly not significant | -0.9 to +2.5 ms | keep, minor |
| LLM enrichment text (natural-language queries) | modest gain, significant on cJSON/xxHash | +0.3 to +2.7 ms | keep for natural-language queries |
| Enrichment from small LLMs (SmolLM2-1.7B, Qwen2.5-1.5B) | **hurts** (structural benchmark) | n/a | avoid |
| LLVM compiler-text index | no gain, hurts when on | +0.1 to +1.4 ms | **remove** |
| Coupled chunk (enrichment + source in one document) instead of separate indexes | recall up on all 3 projects (structural benchmark); mostly up on developer queries | within 1-2 ms either way (no conclusion); index half the size for dense | **keep** (section 4.7) |
| MiniLM cross-encoder reranker (earlier experiment) | no gain on the hybrid stack | 3.5-9x slower | do not use |
| bge-reranker-base | not tested (download failed twice) | n/a | open |

## 6. Limitations and honest caveats

* **Single annotator.** All developer-style and hand-written queries were written by one (AI-assisted) author, not independently reviewed.
  The router's rules were written by the same author as the queries, so they may be tuned to that phrasing. The next step is queries from
  real developers.
* **Small query counts.** 38-47 developer queries per project and 6-18 per intent: many differences are inside the noise (intervals in
  figures 05, 10, 16).
* **"Impact" labels come from the call graph** that the improved pipeline exploits; those gains measure graph usefulness.
* **Enrichment artifacts differ in settings** (quantization, max tokens, one partial run); model-to-model comparisons are confounded
  (`reports/data_quality_report.md`). One enrichment file stored under cJSON was actually an xxHash enrichment; it was detected by its
  recorded hash and excluded from the cJSON analysis.
* **Three small C/C++ projects, one GPU, one run per configuration** (retrieval is deterministic; enrichment was generated once per model).
* **Why earlier numbers looked higher.** An earlier benchmark (1,600 queries on a 96-function version of the chess project) generated its
  queries from the enrichment text itself, which favours enrichment. Re-running those queries through the current pipeline reproduced the old
  numbers, so the pipeline did not get worse; that benchmark was dropped as circular.
* **Coupled chunk:** about a quarter of coupled documents exceed the 512-token encoder limit, so the gain depends partly on truncation order; the gain is
  clear for BGE embeddings but not for Qwen3-Embedding-0.6B with dense retrieval; developer-query differences are mostly not individually significant.
* The contribution analysis is for this tuned stack (routing weights were tuned with all data sources present).

## 7. Folder contents

```
Optima_report.pdf / .tex       the same content as a formatted report (figure numbers = file numbers 00-21)
figures/                       the 22 figures above (00-21)
tables/developer_benchmark/    precision/recall/latency by pipeline, embedding model, enrichment model, model category, query intent, stage
tables/structural_benchmark/   the same for the larger structural benchmark, plus statistical_tests.csv
tables/ablations/              contribution (Shapley) tables, improvement tests, compiler-text sweep and held-out comparison
tables/coupled_chunk/          coupled-vs-separate paired tests and by-category differences (structural and developer benchmark)
reports/                       generated reports (text only; figures are in figures/) and the data-quality report
benchmarks/                    the 125 developer queries (json + authored source) and how labels were made
configs/                       the frozen routing profiles (and the compiler-on test profile)
```

## 8. Reproducing (from the repository root)

```
python -m optima.rag build-benchmark --developer --base-json <base.json> --authored benchmarks/<project>/developer_authored.json --out <out.json>
python -m optima.rag run-suite --spec configs/suites/dev-<project>-grid.json         # default configuration
python -m optima.rag run-suite --spec configs/suites/dev-<project>-improved.json      # routing + instruction
python -m optima.rag run-suite --spec configs/suites/dev-<project>-attribution.json   # 2x2x2 ablation
python -m optima_analytics generate --input-dir experiments_dev --output-dir reports/optima_analytics_dev
python scripts/build_professor_package.py                                              # rebuilds this folder
```

Code: `optima/rag/research/` (pipelines, routing), `optima_analytics/` (statistics and plots), `scripts/` (tuning and packaging).
