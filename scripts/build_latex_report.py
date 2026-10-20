"""Write final_for_professor/Optima_report.tex (numbers read from the result CSVs) and, if a TeX engine is given, compile it.

Usage: python scripts/build_latex_report.py [--engine /path/to/tectonic]
Run scripts/build_professor_package.py first (figures and tables are taken from final_for_professor/).
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(".")
PKG = ROOT / "final_for_professor"
DEV, V2, CT = ROOT / "reports/optima_analytics_dev", ROOT / "reports/optima_analytics_final", ROOT / "reports/compiler_text_test"
DS = ["chess", "cjson", "xxhash"]
DSN = {"chess": "chess engine (C++)", "cjson": "cJSON (C)", "xxhash": "xxHash (C)"}
B, C, S_, I = "raw_baseline", "compiler_hybrid", "semantic_dense+instr", "compiler_hybrid+instr+routed"
PIPE = {B: "baseline\\_rag", C: "compiler\\_hybrid\\_rag (default)", S_: "semantic + query instruction", I: "\\textbf{IMPROVED (routing + instruction)}"}


def esc(s):
    return str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def fig(name, caption, width="\\textwidth", tags=""):
    with Image.open(PKG / "figures" / name) as im:
        ratio = im.width / im.height
    body = ("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=%s]{figures/%s}\n\\caption{%s%s}\n\\end{figure}\n"
            % ("0.98\\linewidth" if ratio >= 1.8 else width, name, ("\\textbf{%s}: " % tags) if tags else "", caption))
    return ("\\begin{landscape}\n" + body + "\\end{landscape}\n") if ratio >= 1.8 else body


def table(head, rows, spec, caption=None, small=True):
    out = ["\\begin{table}[H]\n\\centering" + ("\\small" if small else "")]
    if caption:
        out.append("\\caption{%s}" % caption)
    out.append("\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%%\n\\begin{tabular}{%s}\n\\toprule" % spec)
    out.append(" & ".join("\\textbf{%s}" % h for h in head) + " \\\\\n\\midrule")
    for r in rows:
        out.append("\\addlinespace" if r is None else " & ".join(str(c) for c in r) + " \\\\")
    out.append("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    return "\n".join(out)


def build():
    bp = pd.read_csv(DEV / "final/by_pipeline.csv")
    g = lambda ds, cell, m: float(bp[(bp.dataset == ds) & (bp.cell == cell)][m].iloc[0])
    # --- headline
    rows = []
    for ds in DS:
        for k, cell in enumerate((B, C, S_, I)):
            rows.append([DSN[ds] if k == 0 else "", PIPE[cell], "%.3f" % g(ds, cell, "recall_at_10"), "%.3f" % g(ds, cell, "precision_at_5"),
                         "%.3f" % g(ds, cell, "mrr"), "%.1f" % g(ds, cell, "query_ms_median")])
        if ds != DS[-1]:
            rows.append(None)
    t_head = table(["Project", "Pipeline", "Recall@10", "Precision@5", "MRR", "Median latency (ms)"], rows, "llrrrr",
                   "Held-out developer queries, mean over the 3 embedding models and (for enrichment pipelines) the enrichment models tested.")
    rows = []
    lats = []
    for ds in DS:
        r10 = g(ds, I, "recall_at_10") - g(ds, B, "recall_at_10"); p5 = g(ds, I, "precision_at_5") - g(ds, B, "precision_at_5")
        lat = g(ds, I, "query_ms_median") - g(ds, B, "query_ms_median"); lats.append(lat)
        rows.append([DSN[ds], "%+.3f (%+.0f\\%%)" % (r10, 100 * r10 / g(ds, B, "recall_at_10")), "%+.3f (%+.0f\\%%)" % (p5, 100 * p5 / g(ds, B, "precision_at_5")),
                     "%+.3f" % (g(ds, I, "mrr") - g(ds, B, "mrr")), "%+.1f ms" % lat,
                     "%+.1f ms" % (g(ds, I, "query_ms_median") - g(ds, C, "query_ms_median"))])
    t_imp = table(["Project", "Recall@10", "Precision@5", "MRR", "Latency vs baseline", "Latency vs default compiler pipeline"], rows, "lrrrrr",
                  "Improvement of the IMPROVED pipeline over baseline\\_rag (held-out developer queries).")
    lat_range = "%+.1f to %+.1f ms" % (min(lats), max(lats))
    # --- per embedding
    be = pd.read_csv(DEV / "final/by_embedding_model.csv")
    rows = []
    for e in ("bge-small", "bge-base", "qwen3-0.6b"):
        x = be[be.embedding_alias == e]
        mm = lambda cell, m: x[x.cell == cell][m].mean()
        rows.append([e, "%.3f $\\to$ %.3f" % (mm(B, "recall_at_10"), mm(I, "recall_at_10")), "%.3f $\\to$ %.3f" % (mm(B, "precision_at_5"), mm(I, "precision_at_5")),
                     "%.1f $\\to$ %.1f ms" % (mm(B, "query_ms_median"), mm(I, "query_ms_median"))])
    t_emb = table(["Embedding model", "Recall@10", "Precision@5", "Median latency"], rows, "lrrr",
                  "Baseline $\\to$ improved, per embedding model (mean over the three projects).")
    # --- Shapley
    ac = pd.read_csv(DEV / "tables/attribution_components.csv")
    COMP = ["query instruction", "hybrid retrieval + routing (raw source)", "LLVM compiler representation", "LLVM call graph", "LLM enrichment"]
    TOT = "TOTAL (full stack - baseline_rag)"

    def comp_table(m, fmt, caption):
        rows = []
        for c in COMP + [TOT]:
            cells = []
            for ds in DS:
                x = ac[(ac.dataset == ds) & (ac.metric == m) & (ac.component == c)].iloc[0]
                star = "$^*$" if (x.ci_low > 0 or x.ci_high < 0) and not (x.ci_low == x.ci_high == 0) else ""
                cells.append(fmt % x["mean"] + star)
            label = c.replace(" (full stack - baseline_rag)", " (full stack vs baseline)")
            rows.append([("\\textbf{%s}" % esc(label)) if c == TOT else esc(label)] + cells)
        return table(["Component"] + [DSN[d].split(" (")[0] for d in DS], rows, "lrrr", caption)
    t_r10 = comp_table("recall_at_10", "%+.3f", "Contribution to Recall@10 (exact Shapley values; $^*$ = 95\\% interval excludes zero).")
    t_p5 = comp_table("precision_at_5", "%+.3f", "Contribution to Precision@5.")
    t_lat = comp_table("query_ms_median", "%+.1f", "Contribution to median query latency (ms).")
    # --- compiler text on/off
    con = pd.read_csv(CT / "heldout_compiler_on_vs_off.csv")
    cv = lambda ds, m, prof: float(con[(con.dataset == ds) & (con.metric == m) & (con.profile == prof)]["full_stack_gain_vs_baseline"].iloc[0])
    rows = []
    for lab, prof, met in (("full-stack gain in Recall@10, compiler text OFF (tuned weight 0)", "compiler text OFF (tuned weight 0)", "recall_at_10"),
                           ("full-stack gain in Recall@10, compiler text ON (weight 1)", "compiler text ON (weight 1)", "recall_at_10"),
                           ("compiler text's own contribution to Recall@10 at weight 1", "compiler text contribution at weight 1", "recall_at_10"),
                           ("compiler text's own contribution to MRR at weight 1", "compiler text contribution at weight 1", "mrr")):
        rows.append([lab] + ["%+.3f" % cv(d, met, prof) for d in DS])
    t_ct = table(["Held-out developer queries"] + [DSN[d].split(" (")[0] for d in DS], rows, "lrrr",
                 "Switching the LLVM compiler text on (held-out queries).")
    # --- structural levels
    v2 = pd.read_csv(V2 / "final/by_pipeline.csv")
    g2 = lambda ds, cell, m: float(v2[(v2.dataset == ds) & (v2.cell == cell)][m].iloc[0])
    rows = [[DSN[ds]] + ["%.3f / %.3f" % (g2(ds, c, "recall_at_10"), g2(ds, c, "precision_at_5")) for c in ("raw_baseline", "semantic_dense", "compiler_hybrid")] for ds in DS]
    t_v2 = table(["Project", "baseline\\_rag", "semantic\\_enrichment\\_rag", "compiler\\_hybrid\\_rag"], rows, "llll",
                 "Structural benchmark, Recall@10 / Precision@5 (default settings, mean over embedding and enrichment models).")
    # --- coupled
    def coupled_rows(rep):
        t = pd.read_csv(rep / "tables/coupled_tests.csv")
        out = []
        for comp, name in (("coupled chunk vs separate raw+semantic indexes (dense)", "dense"), ("coupled chunk vs separate indexes (dense + BM25 + exact)", "dense + BM25 + exact")):
            for k, ds in enumerate(DS):
                x = t[(t.comparison == comp) & (t.dataset == ds)]
                m = x[x.metric == "mrr"]
                d = lambda metric: x[x.metric == metric]["mean_diff"].mean()
                sig = int(((m.p_holm < 0.05) & (m.mean_diff > 0)).sum())
                out.append([name if k == 0 else "", DSN[ds].split(" (")[0], "%+.3f" % d("recall_at_10"), "%+.3f" % d("precision_at_5"), "%+.3f" % m.mean_diff.mean(), "%d/%d" % (sig, len(m))])
        return out
    t_cs = table(["Retrieval mode", "Project", "$\\Delta$Recall@10", "$\\Delta$Precision@5", "$\\Delta$MRR", "Significant"], coupled_rows(V2), "llrrrr",
                 "Coupled chunk minus separate indexes, structural benchmark (mean over 12 embedding $\\times$ enrichment-model pairs; ``Significant'' = pairs with a Holm-significant MRR gain).")
    t_cd = table(["Retrieval mode", "Project", "$\\Delta$Recall@10", "$\\Delta$Precision@5", "$\\Delta$MRR", "Significant"], coupled_rows(DEV), "llrrrr",
                 "Coupled chunk minus separate indexes, held-out developer queries.")
    CPN = [("raw_baseline", "raw-only chunk (dense)"), ("semantic_dense", "enrichment-only chunk (dense)"), ("separate_dense", "separate raw + enrichment indexes (dense)"),
           ("coupled_dense", "coupled chunk (dense)"), ("separate_hybrid", "separate indexes (hybrid)"), ("coupled_hybrid", "coupled chunk (hybrid)"),
           ("coupled_hybrid_src", "coupled chunk, source first (hybrid)")]
    rows = []
    for ds in DS:
        for k, (c, lab) in enumerate(CPN):
            rows.append([DSN[ds].split(" (")[0] if k == 0 else "", lab, "%.3f" % g2(ds, c, "recall_at_10"), "%.3f" % g2(ds, c, "precision_at_5"), "%.3f" % g2(ds, c, "mrr"),
                         "%.1f" % g2(ds, c, "query_ms_median"), "%.2f" % g2(ds, c, "index_mb")])
    t_cl = table(["Project", "Variant", "R@10", "P@5", "MRR", "Median ms", "Index MB"], rows, "llrrrrr",
                 "Structural benchmark: mean level of each variant (all runs of the variant; latency differences of 1--2 ms between separately run suites are within run-to-run variation).")

    F = lambda n, cap, tags="": fig(n, cap, tags=tags)
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{pdflscape,graphicx,booktabs,array,xcolor,hyperref,caption,float,microtype,enumitem,amsmath,amssymb}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\captionsetup{font=small,labelfont=bf}
\usepackage{ragged2e}
\setcounter{figure}{-1}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.55em}
\title{Does Compiler (LLVM) Data and LLM Enrichment Improve Code Retrieval?\\[0.3em]\large Results of the Optima retrieval experiments}
\author{}
\date{\today}
\begin{document}
\vspace{-3.5em}\maketitle
\begin{abstract}
Optima analyses C/C++ projects with the LLVM compiler and enriches every function with LLM-generated descriptions. We ask whether
retrieval pipelines that use this data find the function a developer is looking for more often (\emph{recall}, \emph{precision}) and
whether they do so at acceptable cost (\emph{latency}). On 125 held-out developer-style queries over three projects, a pipeline that
routes each query to a suitable retrieval strategy and uses the embedding model's query instruction improves Recall@10 on all three
projects, with a median query-latency change of """ + lat_range + r""" relative to plain dense retrieval over the source code. An ablation shows the LLVM
\emph{call graph} is the largest single contributor, LLM enrichment contributes a smaller significant share on two projects, and the LLVM \emph{compiler text} contributes nothing.
Putting enrichment and source code into one chunk per function (``coupled chunk'') further improves recall over indexing them separately.
This document describes every graph and table in the accompanying folder \texttt{final\_for\_professor/}.
\end{abstract}

\setcounter{tocdepth}{1}
\tableofcontents
\bigskip


\section{Question, system and setup}
\textbf{Task.} A developer query (e.g.\ ``how do I print JSON into a buffer I already allocated'') must return the right function. One function is one
retrievable unit; a query usually has one or two relevant functions.

\textbf{Projects.} chess engine (C++, 164 functions), cJSON (C, 1,201 functions including its test framework), xxHash (C, 339 functions).

\textbf{Embedding models (3).} BGE-small (0.033B parameters), BGE-base (0.109B), Qwen3-Embedding-0.6B (0.596B).

\textbf{Enrichment models (10 LLMs, 1.5B--14.8B parameters).} Qwen2.5 (1.5B, 3B, 7B, 14B, Coder 7B/14B), Qwen3-14B, Mistral-7B, Ministral-8B,
SmolLM2-1.7B; each produced its own description of every function. The held-out runs of the improved pipeline use four of them
(Qwen2.5-Coder-14B, Qwen2.5-14B, Qwen2.5-7B, Ministral-8B).

\textbf{Pipelines.}
\begin{itemize}[nosep]
\item \texttt{baseline\_rag}: dense retrieval over the raw function source.
\item \texttt{semantic\_enrichment\_rag}: dense retrieval over the LLM enrichment text.
\item \texttt{compiler\_hybrid\_rag} (default): raw + enrichment + compiler-text indexes, dense + BM25 + exact-symbol match fused with reciprocal-rank fusion, plus bounded call-graph expansion.
\item \textbf{IMPROVED} (this work): the default pipeline plus \emph{rule-based query routing} (quoted messages and identifiers use BM25 on the source; plain-language questions use dense retrieval on the enrichment text; ``what calls/uses X'' questions use dense retrieval and the call graph) and the embedding model's \emph{query instruction prefix}.
\item \textbf{Coupled chunk} (Section~\ref{sec:coupled}): one document per function that contains the enrichment description and the source code together.
\end{itemize}

\section{How it was measured}
\textbf{Metrics} (per query, then averaged). \emph{Recall@K}: relevant functions found in the top K divided by all relevant functions.
\emph{Precision@K}: relevant functions in the top K divided by K; with 1--2 relevant functions per query it is capped near $1/K$, so Precision@5 cannot exceed
roughly 0.2--0.4 here. \emph{MRR}: mean reciprocal rank of the first relevant function. \emph{Latency}: end-to-end time per query (encoding, retrieval, fusion, graph, optional rerank), one query at a time on one RTX 4070 laptop GPU;
the \emph{median} is reported because the first run of each experiment contains one-off GPU warm-up spikes.

\textbf{Benchmarks.}
\begin{itemize}[nosep]
\item \emph{Developer benchmark (held-out), 125 queries} (chess 38, cJSON 47, xxHash 40), written the way a developer asks (vague memory, ``how do I\ldots'', error message to code, concept, impact ``what calls X''). Labels come from hand-resolved targets, string literals in the source, or the call graph. \textbf{Nothing was tuned on it}: routing weights and prefixes were chosen on a separate tuning benchmark, then these queries were run once.
\item \emph{Structural benchmark (223 / 240 / 225 queries):} templated and hand-written queries (exact name, identifier, behavioural, caller/callee, multi-hop), used for the enrichment-model comparison and for tuning.
\end{itemize}

\textbf{Statistics.} Differences are paired on identical queries with 95\% cluster-bootstrap intervals (queries about the same function are resampled together), cluster permutation tests, and Holm correction across many comparisons. A star in a table means the interval excludes zero.

\section{Headline result}
""" + t_head + t_imp + r"""
Recall and precision increase on all three projects. Latency of the improved pipeline relative to \texttt{baseline\_rag} is """ + lat_range + r""" (median), the largest increase on cJSON, the biggest corpus; it is lower than the default compiler pipeline on all three projects. Each project has only 38--47 queries, so many individual comparisons are inside the noise; the direction is consistent across projects.
""" + F("00_headline_before_after.png", "Recall@10, Precision@5, MRR (higher is better) and median latency (lower is better) for four pipelines on the held-out developer queries. The improved pipeline (pink) has the tallest accuracy bars on every project; its latency is slightly above the baseline and below the default compiler pipeline.", "[Recall][Precision][Latency]") + r"""
\newpage
\section{Recall and precision}
""" + F("01_precision_recall_at_k_by_pipeline.png", "Precision@K (top) and Recall@K (bottom) for K = 1, 3, 5, 10 per project. Precision falls with K and recall rises by construction; the improved pipeline is highest in recall at every K and in precision at every K, including P@1.", "[Precision][Recall]") + F("02_precision_recall_by_enrichment_model.png", "Precision@5 (top) and Recall@5 (bottom) of the semantic pipeline for each LLM that wrote the enrichment; the dashed line is the raw-source baseline. Models differ by a few hundredths, so only the large gaps (very small models) should be read as real.", "[Precision][Recall]") + F("03_precision_recall_cutoffs_by_embedding.png", "Precision against recall at the evaluated cutoffs (K = 1, 3, 5, 10) for each embedding model; up and to the right is better. This is a four-point precision--recall-at-cutoff plot, not a full PR curve, because only the top 10 are evaluated.", "[Precision][Recall]") + F("04_precision_recall_latency_by_model_category.png", "Precision@5, Recall@5 and latency grouped by model-size bucket: enrichment models (<4B / 4--10B / $\\geq$10B) in the semantic pipeline (top) and embedding models in the compiler pipeline (bottom). Buckets with one model carry no spread information.", "[Precision][Recall][Latency]") + F("05_improvement_by_developer_intent.png", "Paired MRR gain of the improved pipeline over the default compiler pipeline (orange) and over the baseline (blue) per developer intent, with 95\\% intervals. The largest and most consistent gains are on ``impact'' questions; others are inconsistent. Each intent has only 6--18 queries, so most intervals are wide.", "[Recall][Precision] (as MRR)") + r"""
\newpage
\section{Latency}
""" + t_emb + F("06_latency_by_embedding_and_pipeline.png", "Median per-query latency (bars, log scale) and p95 (tick) per embedding model and pipeline. Latency is set mainly by the embedding model; the improved pipeline is not slower than the default compiler pipeline.", "[Latency]") + F("07_latency_by_enrichment_model.png", "Top: retrieval latency of the compiler pipeline for each enrichment model. Bottom: the one-off cost of generating the enrichment (seconds per function, from the enrichment logs; different GPUs and settings). Which LLM wrote the enrichment hardly changes retrieval latency, but generating it costs seconds per function on a 14B model (hours per project), paid once offline.", "[Latency]") + F("08_accuracy_vs_latency_by_data_source.png", "Recall@10 against latency for every on/off combination of the three data sources inside the routed stack (E = LLM enrichment, C = LLVM compiler text, G = LLVM call graph) plus the baseline (squares); up and to the left is better. Combinations containing G reach the highest recall; adding C does not move a point up. Note the narrow axes: horizontal differences are about 1--2 ms.", "[Recall][Latency]") + r"""
\newpage
\section{Where the improvement comes from}
A controlled $2\times2\times2$ experiment switched the LLVM compiler text (C), the LLVM call graph (G) and the LLM enrichment (E) on and off independently
(all 8 combinations). Contributions are exact Shapley values (average marginal gain over all orders of adding the data sources), so they do not depend on order and add up to the total gain.
The first two components are sequential steps from the baseline.
""" + t_r10 + t_p5 + t_lat + F("09_contribution_of_each_data_source.png", "The contribution tables as stacked bars (grey = query instruction, blue = hybrid retrieval and routing on the raw source, orange = LLVM compiler text, green = LLVM call graph, pink = LLM enrichment, diamond = net total). The LLVM call graph is the largest contributor to Recall@10 and Precision@5 on all three projects; LLM enrichment adds a smaller share that is significant on cJSON and xxHash; the LLVM compiler text contributes nothing.", "[Recall][Precision][Latency]") + F("10_contribution_with_confidence_intervals.png", "Shapley contributions to Recall@10 (top) and MRR (bottom) with 95\\% intervals; an interval that crosses zero is indistinguishable from no effect. The call graph's interval excludes zero on all three projects.", "[Recall]") + F("11_contribution_by_developer_intent.png", "Recall@10 contribution of each data source for each developer intent. The call graph helps almost only the ``impact'' questions; enrichment helps plain-language intents on cJSON and xxHash. Caveat: impact queries are labelled from the same call graph, so this shows that the graph is useful for call-graph questions, not that it improves retrieval in general.", "[Recall]") + r"""
\section{Does the LLVM compiler text help if it is allowed to?}
The routing tuning gave the compiler-text index weight 0 in every query class. To check this was not an artefact, its weight was swept on the tuning data and then
switched on (weight 1, fixed in advance) on the held-out queries.
""" + t_ct + F("12_compiler_text_weight_sweep_tuning_data.png", "MRR by query class as the compiler-text weight rises (call graph on): accuracy falls in every class. With the call graph switched off the compiler text does help structural questions, because it contains callers and callees, but the call graph itself does better, so with the graph on the text is redundant.", "[Recall] (as MRR)") + F("13_compiler_text_switched_on_heldout.png", "Top: gain of the full improved stack over the baseline with the compiler text off (green) and on (orange). Bottom: the compiler text's own contribution; it also adds latency. Conclusion: leave the compiler text out; the call graph provides the structural signal.", "[Recall][Latency]") + r"""
\section{Enrichment on the larger structural benchmark}
""" + t_v2 + F("14_enrichment_vs_raw_by_enrichment_model.png", "MRR with each enrichment model's text (three pipelines) against the raw-code baseline (dashed). The best models are above the baseline; the smallest (SmolLM2-1.7B, Qwen2.5-1.5B) fall below it.", "(MRR)") + F("15_enrichment_effect_gain_heatmap.png", "MRR gain of the enriched corpus over the raw corpus for every enrichment model $\\times$ embedding model (blue = better, red = worse); gains shrink or turn negative with the strongest embedding model.") + F("16_enrichment_effect_forest_with_ci.png", "The same comparisons with 95\\% intervals and Holm-corrected significance (blue dots). Most individual comparisons are not significant; the signal is in the overall pattern.") + F("17_pipelines_by_query_family.png", "MRR per query family (lexical / semantic / structural). Enrichment helps natural-language questions and not lexical or structural ones, and structural questions are weak for every pipeline, which is why routing by query type works.") + r"""
\section{One chunk per function: enrichment and source coupled}\label{sec:coupled}
Every function is already one chunk. The earlier design kept the LLM enrichment in a \emph{separate} index and fused its results with the raw-code index.
The \textbf{coupled chunk} puts the identity header, the enrichment description and the source into one document that is embedded once, so both are matched together.
Two orders were pre-specified: enrichment first (main variant) and source first (the encoder truncates long inputs, and about a quarter of coupled chunks exceed its 512-token limit).
Each project was run with 4 enrichment models $\times$ 3 embedding models on both benchmarks.
""" + t_cs + t_cd + t_cl + F("18_coupled_chunk_overview_structural.png", "Structural benchmark. Recall@10, Precision@5, MRR and median latency of raw-only, enrichment-only, separate raw+enrichment indexes and the coupled chunk (both orders); top row dense retrieval, bottom row dense + BM25 + exact match. The coupled chunk has the highest recall in every project and retrieval mode.", "[Recall][Precision][Latency]") + F("19_coupled_chunk_paired_differences_structural.png", "One dot per (embedding $\\times$ enrichment model): paired MRR difference coupled minus separate with a 95\\% interval; blue = Holm-significant. With the two BGE embedding models the dense gain is consistent on chess and cJSON; with Qwen3-Embedding-0.6B it is zero or slightly negative. For hybrid search the gain is positive for all three embedding models and significant for all 12 pairs on cJSON.", "[Recall] (as MRR)") + F("20_coupled_chunk_overview_developer.png", "The overview plot on the 125 held-out developer queries.", "[Recall][Precision][Latency]") + F("21_coupled_chunk_paired_differences_developer.png", "The paired-difference plot on the developer queries. The direction agrees with the structural benchmark, but with 38--47 queries per project few pairs are individually significant.", "[Recall] (as MRR)") + r"""
\textbf{Reading.} The coupled chunk beats the separate-index design while using half the index for dense retrieval. Against an enrichment-only chunk it is about equal
($\Delta$MRR near 0): most of the gain over raw source comes from adding the enrichment text, and coupling mainly avoids the loss that fusing two separate indexes incurs.
The dense gain comes from behavioural (natural-language) queries; exact-name and caller/callee queries are flat or slightly lower. Source-first versus enrichment-first has no clear winner on MRR
(differences up to 0.02); source-first gives slightly higher Recall@10. Median query latency is within 1--2 ms of the separate indexes in either direction, which is of the order of
the variation between separately run suites, so no latency conclusion is drawn. The coupled chunk has \emph{not} been tested inside the routed improved stack; that would require re-tuning the routing profiles on the tuning data.

\section{What improved and what did not}
\begin{table}[H]\centering\small
\begin{tabular}{>{\raggedright\arraybackslash}p{4.6cm}>{\raggedright\arraybackslash}p{4.3cm}>{\raggedright\arraybackslash}p{3.2cm}>{\raggedright\arraybackslash}p{2.2cm}}\toprule
\textbf{Change} & \textbf{Recall / precision} & \textbf{Latency} & \textbf{Verdict}\\\midrule
Query-type routing (+ call-graph expansion for ``what uses X'') & clear gain & no extra cost vs.\ default compiler pipeline; a few ms above plain baseline & keep\\
Query instruction prefix & small, mostly not significant & $-0.9$ to $+2.5$ ms & keep, minor\\
LLM enrichment text (natural-language queries) & modest gain, significant on cJSON/xxHash & $+0.3$ to $+2.7$ ms & keep\\
Coupled chunk instead of separate indexes & recall up on all projects (structural); mostly up on developer queries & same order; index half the size (dense) & keep\\
Enrichment from small LLMs (SmolLM2-1.7B, Qwen2.5-1.5B) & hurts (structural benchmark) & n/a & avoid\\
LLVM compiler-text index & no gain; hurts when on & $+0.1$ to $+1.4$ ms & remove\\
MiniLM cross-encoder reranker & no gain on hybrid stack & 3.5--9$\times$ slower & do not use\\
bge-reranker-base & not tested (download failed twice) & n/a & open\\
\bottomrule\end{tabular}\caption{Summary of the changes evaluated.}\end{table}

\section{Limitations}
\begin{itemize}[nosep]
\item \textbf{Single annotator.} All developer-style and hand-written queries were written by one (AI-assisted) author and not independently reviewed. The router's rules were written by the same author as the queries, so they may be tuned to that phrasing. Queries from real developers are the next step.
\item \textbf{Small query counts.} 38--47 developer queries per project and 6--18 per intent: many differences are inside the noise.
\item \textbf{``Impact'' labels come from the call graph} that the improved pipeline exploits; those gains measure graph usefulness.
\item \textbf{Enrichment artifacts differ in settings} (quantization, maximum tokens, one partial run), so model-to-model comparisons are confounded. One enrichment file stored under cJSON was actually an xxHash enrichment; it was detected by its recorded hash and excluded.
\item \textbf{Three small C/C++ projects, one GPU, one run per configuration} (retrieval is deterministic; enrichment was generated once per model).
\item \textbf{Earlier higher numbers.} A previous benchmark (1,600 queries on a 96-function version of the chess project) generated its queries from the enrichment text itself, which favours enrichment. Re-running those queries through the current pipeline reproduced the old numbers, so the pipeline did not get worse; that benchmark was dropped as circular.
\item \textbf{Coupled chunk:} a quarter of coupled documents exceed the 512-token encoder limit; the dense gain is clear for BGE embeddings but not for Qwen3-Embedding-0.6B; developer-query differences are mostly not individually significant.
\item The contribution analysis is for this tuned stack (routing weights were tuned with all data sources present).
\end{itemize}

\section{Contents of the accompanying folder}
\begin{description}[nosep,style=nextline]
\item[\texttt{Optima\_report.pdf / .tex}] this report.
\item[\texttt{README.md}] figure-by-figure guide, tagged recall / precision / latency, with the tables of this report.
\item[\texttt{figures/}] the 22 figures shown here, 300 dpi PNG; figure numbers in this report equal the file-name prefixes (00--21).
\item[\texttt{tables/developer\_benchmark/}] precision / recall / latency by pipeline, embedding model, enrichment model, model category, query intent and stage; statistical tests.
\item[\texttt{tables/structural\_benchmark/}] the same for the larger structural benchmark.
\item[\texttt{tables/ablations/}] contribution (Shapley) tables, improvement tests, compiler-text sweep and held-out comparison.
\item[\texttt{tables/coupled\_chunk/}] coupled-vs-separate paired tests and by-category differences for both benchmarks.
\item[\texttt{reports/}] generated text reports and the data-quality report.
\item[\texttt{benchmarks/}] the 125 developer queries (JSON and authored source) and how labels were made.
\item[\texttt{configs/}] the frozen routing profiles.
\end{description}

\end{document}
"""
    return tex


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None)
    args = ap.parse_args()
    out = PKG / "Optima_report.tex"
    out.write_text(build())
    print("wrote", out)
    if args.engine:
        subprocess.run([args.engine, out.name], cwd=PKG, check=True)
        import shutil
        shutil.make_archive(str(PKG), "zip", ROOT, PKG.name)
        print("compiled", PKG / "Optima_report.pdf", "and re-zipped", PKG.name + ".zip")
