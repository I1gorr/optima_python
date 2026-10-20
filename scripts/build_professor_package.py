"""Assemble final_for_professor/: curated figures, tables, benchmarks and a README with numbers read from the results."""
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "final_for_professor"
DEV, V2, CT = ROOT / "reports/optima_analytics_dev", ROOT / "reports/optima_analytics_final", ROOT / "reports/compiler_text_test"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dddcd7"
COLORS = {"raw_baseline": "#2a78d6", "compiler_hybrid": "#eb6834", "semantic_dense+instr": "#1baf7a", "compiler_hybrid+instr+routed": "#e87ba4"}
LABELS = {"raw_baseline": "baseline_rag (dense on raw source)", "compiler_hybrid": "compiler_hybrid_rag (default)",
          "semantic_dense+instr": "semantic_enrichment_rag + query instruction", "compiler_hybrid+instr+routed": "IMPROVED: routing + instruction"}
DS = ["chess", "cjson", "xxhash"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "font.size": 9.5, "savefig.dpi": 300, "legend.frameon": False})

if OUT.exists():
    shutil.rmtree(OUT)
for d in ("figures", "tables/developer_benchmark", "tables/structural_benchmark", "tables/ablations", "tables/coupled_chunk", "reports", "benchmarks", "configs"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ headline figure
bp = pd.read_csv(DEV / "final/by_pipeline.csv")
cells = list(COLORS)
metrics = [("recall_at_10", "Recall@10 (higher is better)"), ("precision_at_5", "Precision@5 (higher is better)"),
           ("mrr", "MRR (higher is better)"), ("query_ms_median", "median query latency, ms (lower is better)")]
fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
for ax, (m, title) in zip(axes, metrics):
    w = 0.2
    for i, c in enumerate(cells):
        vals = [float(bp[(bp.dataset == d) & (bp.cell == c)][m].iloc[0]) for d in DS]
        xs = np.arange(len(DS)) + (i - 1.5) * w
        ax.bar(xs, vals, w * 0.92, color=COLORS[c], label=LABELS[c])
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.2f}" if m != "query_ms_median" else f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels(["chess (C++)", "cJSON (C)", "xxHash (C)"]); ax.set_title(title, fontsize=10)
    if m != "query_ms_median":
        ax.set_ylim(0, max(float(bp[bp.cell.isin(cells)][m].max()) * 1.15, 0.1))
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Before vs after on held-out developer queries (125 queries, 3 projects; mean over 3 embedding models and the enrichment models)", fontweight="bold", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "figures/00_headline_before_after.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ copy figures
FIG = [
    ("01_precision_recall_at_k_by_pipeline.png", DEV / "plots/G01_precision_recall_at_k_by_pipeline.png"),
    ("02_precision_recall_by_enrichment_model.png", DEV / "plots/G03_precision_recall_by_enrichment_model.png"),
    ("03_precision_recall_cutoffs_by_embedding.png", DEV / "plots/G02_precision_recall_cutoffs_by_embedding.png"),
    ("04_precision_recall_latency_by_model_category.png", DEV / "plots/G06_precision_recall_latency_by_model_category.png"),
    ("05_improvement_by_developer_intent.png", DEV / "plots/G07_improvement_by_intent.png"),
    ("06_latency_by_embedding_and_pipeline.png", DEV / "plots/G04_latency_by_embedding_and_pipeline.png"),
    ("07_latency_by_enrichment_model.png", DEV / "plots/G05_latency_by_enrichment_model.png"),
    ("08_accuracy_vs_latency_by_data_source.png", DEV / "plots/H04_accuracy_latency_by_data_source.png"),
    ("09_contribution_of_each_data_source.png", DEV / "plots/H01_contribution_stacked.png"),
    ("10_contribution_with_confidence_intervals.png", DEV / "plots/H02_contribution_with_intervals.png"),
    ("11_contribution_by_developer_intent.png", DEV / "plots/H03_contribution_by_intent.png"),
    ("12_compiler_text_weight_sweep_tuning_data.png", CT / "compiler_text_tuning_sweep.png"),
    ("13_compiler_text_switched_on_heldout.png", CT / "compiler_text_heldout.png"),
    ("14_enrichment_vs_raw_by_enrichment_model.png", V2 / "plots/A02_raw_vs_enriched.png"),
    ("15_enrichment_effect_gain_heatmap.png", V2 / "plots/A03_heatmap_enrichment_x_embedding_gain.png"),
    ("16_enrichment_effect_forest_with_ci.png", V2 / "plots/F36_paired_difference_forest.png"),
    ("17_pipelines_by_query_family.png", V2 / "plots/C18_pipeline_by_query_family.png"),
    ("18_coupled_chunk_overview_structural.png", V2 / "plots/I01_coupled_chunk_overview.png"),
    ("19_coupled_chunk_paired_differences_structural.png", V2 / "plots/I02_coupled_chunk_paired_differences.png"),
    ("20_coupled_chunk_overview_developer.png", DEV / "plots/I01_coupled_chunk_overview.png"),
    ("21_coupled_chunk_paired_differences_developer.png", DEV / "plots/I02_coupled_chunk_paired_differences.png"),
]
for name, src in FIG:
    shutil.copy(src, OUT / "figures" / name)

# ------------------------------------------------------------------ tables / reports / benchmarks / configs
for f in ("by_pipeline", "by_embedding_model", "by_enrichment_model", "by_model_category", "by_intent", "precision_recall_at_k", "latency_by_stage", "all_configurations"):
    shutil.copy(DEV / f"final/{f}.csv", OUT / "tables/developer_benchmark" / f"{f}.csv")
for f in ("improvement_tests", "improvement_by_category", "attribution_components", "attribution_by_intent", "attribution_configs"):
    shutil.copy(DEV / f"tables/{f}.csv", OUT / "tables/ablations" / f"{f}.csv")
shutil.copy(DEV / "statistical_tests.csv", OUT / "tables/developer_benchmark/statistical_tests.csv")
for f in ("by_pipeline", "by_embedding_model", "by_enrichment_model", "by_model_category"):
    shutil.copy(V2 / f"final/{f}.csv", OUT / "tables/structural_benchmark" / f"{f}.csv")
shutil.copy(V2 / "statistical_tests.csv", OUT / "tables/structural_benchmark/statistical_tests.csv")
for f in ("tuning_sweep_by_class.csv", "tuning_sweep_by_category.csv", "heldout_compiler_on_vs_off.csv"):
    shutil.copy(CT / f, OUT / "tables/ablations" / f)
for tag, rep in (("structural", V2), ("developer", DEV)):
    for f in ("coupled_tests", "coupled_by_category"):
        shutil.copy(rep / f"tables/{f}.csv", OUT / "tables/coupled_chunk" / f"{f}_{tag}_benchmark.csv")
strip = lambda s: re.sub(r"!\[[^\]]*\]\([^)]*\)\n*", "", s)
(OUT / "reports/final_results_developer_benchmark.md").write_text(strip((DEV / "final_results.md").read_text()))
(OUT / "reports/final_results_structural_benchmark.md").write_text(strip((V2 / "final_results.md").read_text()))
(OUT / "reports/executive_summary_structural_benchmark.md").write_text((V2 / "executive_summary.md").read_text())
(OUT / "reports/data_quality_report.md").write_text((V2 / "data_quality_report.md").read_text())
for ds in DS:
    shutil.copy(ROOT / f"benchmarks/{ds}/developer_v1.json", OUT / "benchmarks" / f"{ds}_developer_v1.json")
    shutil.copy(ROOT / f"benchmarks/{ds}/developer_authored.json", OUT / "benchmarks" / f"{ds}_developer_authored.json")
shutil.copy(ROOT / "benchmarks/README.md", OUT / "benchmarks/README.md")
shutil.copy(ROOT / "configs/routing_profiles.json", OUT / "configs/routing_profiles.json")
shutil.copy(ROOT / "configs/routing_profiles_compiler_on.json", OUT / "configs/routing_profiles_compiler_on.json")

# ------------------------------------------------------------------ numbers for the README
g = lambda ds, cell, m: float(bp[(bp.dataset == ds) & (bp.cell == cell)][m].iloc[0])
B, C, S, I = "raw_baseline", "compiler_hybrid", "semantic_dense+instr", "compiler_hybrid+instr+routed"
rows = []
for ds in DS:
    for cell in (B, C, S, I):
        rows.append(f"| {ds} | {LABELS[cell]} | {g(ds, cell, 'recall_at_10'):.3f} | {g(ds, cell, 'precision_at_5'):.3f} | {g(ds, cell, 'mrr'):.3f} | {g(ds, cell, 'query_ms_median'):.1f} |")
headline_table = "\n".join(rows)
imp = {}
for ds in DS:
    imp[ds] = dict(r10=g(ds, I, "recall_at_10") - g(ds, B, "recall_at_10"), r10p=100 * (g(ds, I, "recall_at_10") / g(ds, B, "recall_at_10") - 1),
                   p5=g(ds, I, "precision_at_5") - g(ds, B, "precision_at_5"), p5p=100 * (g(ds, I, "precision_at_5") / g(ds, B, "precision_at_5") - 1),
                   mrr=g(ds, I, "mrr") - g(ds, B, "mrr"), lat=g(ds, I, "query_ms_median") - g(ds, B, "query_ms_median"),
                   lat_vs_default=g(ds, I, "query_ms_median") - g(ds, C, "query_ms_median"),
                   r10_vs_default=g(ds, I, "recall_at_10") - g(ds, C, "recall_at_10"), p5_vs_default=g(ds, I, "precision_at_5") - g(ds, C, "precision_at_5"))
lats = [v["lat"] for v in imp.values()]
lat_range = f"{min(lats):+.1f} to {max(lats):+.1f} ms (median)"
imp_lines = "\n".join(f"| {ds} | {v['r10']:+.3f} ({v['r10p']:+.0f}%) | {v['p5']:+.3f} ({v['p5p']:+.0f}%) | {v['mrr']:+.3f} | {v['lat']:+.1f} ms | {v['r10_vs_default']:+.3f} / {v['p5_vs_default']:+.3f} / {v['lat_vs_default']:+.1f} ms |" for ds, v in imp.items())

ac = pd.read_csv(DEV / "tables/attribution_components.csv")
def acv(ds, m, c, col="mean"):
    return float(ac[(ac.dataset == ds) & (ac.metric == m) & (ac.component == c)][col].iloc[0])
COMP = ["query instruction", "hybrid retrieval + routing (raw source)", "LLVM compiler representation", "LLVM call graph", "LLM enrichment"]
TOT = "TOTAL (full stack - baseline_rag)"
def comp_rows(m, fmt="{:+.3f}"):
    out = []
    for c in COMP + [TOT]:
        cells = []
        for ds in DS:
            v = acv(ds, m, c); lo, hi = acv(ds, m, c, "ci_low"), acv(ds, m, c, "ci_high")
            star = "*" if (lo > 0 or hi < 0) and not (lo == hi == 0) else ""
            cells.append(fmt.format(v) + star)
        out.append(f"| {c.replace(' (full stack - baseline_rag)', ' (full stack vs baseline_rag)')} | " + " | ".join(cells) + " |")
    return "\n".join(out)

con = pd.read_csv(CT / "heldout_compiler_on_vs_off.csv")
def cv(ds, m, prof):
    return float(con[(con.dataset == ds) & (con.metric == m) & (con.profile == prof)]["full_stack_gain_vs_baseline"].iloc[0])
sw = pd.read_csv(CT / "tuning_sweep_by_class.csv")
swg = sw[sw.scenario == "graph as tuned"].pivot(index="compiler_weight", columns="class", values="mrr")
sweep_rows = "\n".join(f"| {w:g} | {r['literal']:.3f} | {r['symbol']:.3f} | {r['natural']:.3f} | {r['structural']:.3f} |" for w, r in swg.loc[[0, 0.5, 1, 4]].iterrows())
swoff = sw[(sw.scenario != "graph as tuned") & (sw["class"] == "structural")].set_index("compiler_weight")["mrr"]
swon = sw[(sw.scenario == "graph as tuned") & (sw["class"] == "structural") & (sw.compiler_weight == 0)]["mrr"].iloc[0]

be = pd.read_csv(DEV / "final/by_embedding_model.csv")
emb_rows = []
for e in ("bge-small", "bge-base", "qwen3-0.6b"):
    x = be[(be.embedding_alias == e)]
    def mm(cell, m): return x[x.cell == cell][m].mean()
    emb_rows.append(f"| {e} | {mm(B, 'recall_at_10'):.3f} -> {mm(I, 'recall_at_10'):.3f} | {mm(B, 'precision_at_5'):.3f} -> {mm(I, 'precision_at_5'):.3f} | {mm(B, 'query_ms_median'):.1f} -> {mm(I, 'query_ms_median'):.1f} ms |")
emb_table = "\n".join(emb_rows)

bn = pd.read_csv(DEV / "final/by_enrichment_model.csv")
im = bn[bn.cell == I]
enr_rows = "\n".join(f"| {m} | {r.precision_at_5:.3f} | {r.recall_at_5:.3f} | {r.recall_at_10:.3f} | {r.mrr:.3f} | {r.query_ms_median:.1f} |" for m, r in im.groupby("enrichment_model")[["precision_at_5", "recall_at_5", "recall_at_10", "mrr", "query_ms_median"]].mean().iterrows())

it = pd.read_csv(DEV / "tables/improvement_tests.csv")
def sigcount(comp):
    x = it[(it.comparison == comp) & (it.metric == "mrr")]
    return int(((x.p_holm < 0.05) & (x.mean_diff > 0)).sum()), len(x)
sg_base, n_base = sigcount("improved vs baseline_rag")
sg_def, n_def = sigcount("improved (instruction + routing) vs default compiler_hybrid_rag")

v2 = pd.read_csv(V2 / "final/by_pipeline.csv")
def g2(ds, cell, m): return float(v2[(v2.dataset == ds) & (v2.cell == cell)][m].iloc[0])
v2_rows = "\n".join(f"| {ds} | {g2(ds, 'raw_baseline', 'recall_at_10'):.3f} / {g2(ds, 'raw_baseline', 'precision_at_5'):.3f} | {g2(ds, 'semantic_dense', 'recall_at_10'):.3f} / {g2(ds, 'semantic_dense', 'precision_at_5'):.3f} | {g2(ds, 'compiler_hybrid', 'recall_at_10'):.3f} / {g2(ds, 'compiler_hybrid', 'precision_at_5'):.3f} |" for ds in DS)

# coupled chunk tables
def coupled_pairs(rep, label):
    t = pd.read_csv(rep / "tables/coupled_tests.csv")
    out = []
    for comp, name in (("coupled chunk vs separate raw+semantic indexes (dense)", "dense"),
                       ("coupled chunk vs separate indexes (dense + BM25 + exact)", "dense + BM25 + exact")):
        for ds in DS:
            x = t[(t.comparison == comp) & (t.dataset == ds)]
            m = x[x.metric == "mrr"]
            d = lambda metric: x[x.metric == metric]["mean_diff"].mean()
            sig = int(((m.p_holm < 0.05) & (m.mean_diff > 0)).sum()); neg = int(((m.p_holm < 0.05) & (m.mean_diff < 0)).sum())
            out.append(f"| {name} | {ds} | {d('recall_at_10'):+.3f} | {d('precision_at_5'):+.3f} | {m.mean_diff.mean():+.3f} | {sig}/{len(m)} (worse: {neg}) |")
    return "\n".join(out)
cp_struct, cp_dev = coupled_pairs(V2, "structural"), coupled_pairs(DEV, "developer")
CPN = [("raw_baseline", "raw-only chunk (dense)"), ("semantic_dense", "enrichment-only chunk (dense)"), ("separate_dense", "separate raw + enrichment indexes (dense)"),
       ("coupled_dense", "coupled chunk (dense)"), ("separate_hybrid", "separate indexes (dense + BM25 + exact)"),
       ("coupled_hybrid", "coupled chunk (dense + BM25 + exact)"), ("coupled_hybrid_src", "coupled chunk, source first (dense + BM25 + exact)")]
cp_levels = "\n".join(f"| {ds} | {lab} | {g2(ds, c, 'recall_at_10'):.3f} | {g2(ds, c, 'precision_at_5'):.3f} | {g2(ds, c, 'mrr'):.3f} | {g2(ds, c, 'query_ms_median'):.1f} | {g2(ds, c, 'index_mb'):.2f} |"
                       for ds in DS for c, lab in CPN)

# extra computed text for the README
x = bn[bn.cell == "semantic_dense"]
ext = []
for ds in DS:
    y = x[x.dataset == ds].groupby("enrichment_model")["recall_at_5"].mean().sort_values(ascending=False)
    base = float(bp[(bp.dataset == ds) & (bp.cell == B)]["recall_at_5"].iloc[0])
    ext.append(f"{ds}: best {y.index[0]} (R@5 {y.iloc[0]:.3f}), worst {y.index[-1]} ({y.iloc[-1]:.3f}) vs raw baseline {base:.3f}")
enr_extremes = "; ".join(ext) + "."
bc = pd.read_csv(DEV / "final/by_model_category.csv")
ce = []
for ds in DS:
    y = bc[(bc.dataset == ds) & (bc.axis == "enrichment model size") & (bc.cell == "semantic_dense")].set_index("level")["recall_at_5"]
    ce.append(f"{ds} " + " / ".join(f"{y[l]:.3f}" for l in ("small", "medium", "large") if l in y.index))
cat_enr = "; ".join(ce)
cm = []
for ds in DS:
    y = bc[(bc.dataset == ds) & (bc.axis == "embedding model size") & (bc.cell == "compiler_hybrid")].set_index("level")["query_ms_median"]
    cm.append(f"{ds} " + " / ".join(f"{y[l]:.1f}" for l in ("small", "medium", "large") if l in y.index))
cat_emb = "; ".join(cm)
(OUT / "README.md").write_text(open("scripts/professor_readme_template.md").read().format(
    headline_table=headline_table, imp_lines=imp_lines, emb_table=emb_table, enr_rows=enr_rows,
    comp_r10=comp_rows("recall_at_10"), comp_p5=comp_rows("precision_at_5"), comp_mrr=comp_rows("mrr"), comp_lat=comp_rows("query_ms_median", "{:+.1f}"),
    sweep_rows=sweep_rows, sw_off0=swoff[0], sw_off4=swoff[4], sw_on0=swon,
    on_r10=" / ".join(f"{cv(d, 'recall_at_10', 'compiler text ON (weight 1)'):+.3f}" for d in DS), off_r10=" / ".join(f"{cv(d, 'recall_at_10', 'compiler text OFF (tuned weight 0)'):+.3f}" for d in DS),
    ct_r10=" / ".join(f"{cv(d, 'recall_at_10', 'compiler text contribution at weight 1'):+.3f}" for d in DS),
    ct_mrr=" / ".join(f"{cv(d, 'mrr', 'compiler text contribution at weight 1'):+.3f}" for d in DS),
    lat_range=lat_range, enr_extremes=enr_extremes, cat_enr=cat_enr, cat_emb=cat_emb, sg_base=sg_base, n_base=n_base, sg_def=sg_def, n_def=n_def, v2_rows=v2_rows, cp_struct=cp_struct, cp_dev=cp_dev, cp_levels=cp_levels))
txt = (OUT / "README.md").read_text()
for name, _ in [("00_headline_before_after.png", None)] + FIG:
    txt = txt.replace(f"**`{name}`", f"![{name}](figures/{name})\n\n**`{name}`", 1)
(OUT / "README.md").write_text(txt)
shutil.make_archive(str(OUT), "zip", ROOT, OUT.name)
print("package built:", OUT, "and", OUT.name + ".zip")
