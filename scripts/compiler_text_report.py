"""Figures + summary for the 'what if the compiler text is allowed to contribute' test."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("reports/compiler_text_test")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dddcd7"
BLUE, ORANGE, AQUA, MAG = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
                     "font.size": 9, "savefig.dpi": 300, "legend.frameon": False})

by_class = pd.read_csv(OUT / "tuning_sweep_by_class.csv")
by_cat = pd.read_csv(OUT / "tuning_sweep_by_category.csv")

# ---- Figure 1: tuning sweep
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axes[0]
on = by_class[by_class.scenario == "graph as tuned"]
for cls, col, mk in (("literal", BLUE, "o"), ("symbol", ORANGE, "s"), ("natural", MAG, "^"), ("structural", AQUA, "D")):
    g = on[on["class"] == cls].sort_values("compiler_weight")
    ax.plot(g.compiler_weight.astype(str), g.mrr, marker=mk, color=col, lw=1.8, label=f"{cls} (n={int(g.n_queries.iloc[0])})")
ax.set_xlabel("weight of the LLVM compiler-text index"); ax.set_ylabel("MRR (tuning queries)")
ax.set_title("Compiler text lowers accuracy in every query class\n(call graph on, as tuned)", fontsize=10); ax.legend(fontsize=8)
ax = axes[1]
st = by_class[by_class["class"] == "structural"]
for sc, col, lab in (("graph as tuned", AQUA, "call graph ON"), ("call graph OFF for structural queries", ORANGE, "call graph OFF")):
    g = st[st.scenario == sc].sort_values("compiler_weight")
    ax.plot(g.compiler_weight.astype(str), g.mrr, marker="o", color=col, lw=1.8, label=lab)
ax.set_xlabel("weight of the LLVM compiler-text index"); ax.set_ylabel("MRR on call-graph questions")
ax.set_title("Structural questions: compiler text is a weak\nsubstitute for the call graph", fontsize=10); ax.legend(fontsize=8)
ax = axes[2]
cats = ["caller", "callee", "control_flow", "cross_function", "multi_hop"]
off = by_cat[by_cat.scenario != "graph as tuned"]
x = np.arange(len(cats)); w = 0.27
for i, (wt, col) in enumerate(((0, "#b9b8b2"), (1, ORANGE), (4, "#6250d6"))):
    v = [off[(off.compiler_weight == wt) & (off.category == c)].mrr.iloc[0] for c in cats]
    ax.bar(x + (i - 1) * w, v, w * 0.92, color=col, label=f"compiler weight {wt}")
tuned = [by_cat[(by_cat.scenario == "graph as tuned") & (by_cat.compiler_weight == 0) & (by_cat.category == c)].mrr.iloc[0] for c in cats]
ax.scatter(x, tuned, marker="_", s=500, color=AQUA, linewidths=3, zorder=5, label="call graph ON (tuned)")
ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20); ax.set_ylabel("MRR (call graph OFF unless marked)")
ax.set_title("By question type: compiler text helps a little\nwithout the graph, far below the graph", fontsize=10); ax.legend(fontsize=7.5)
fig.suptitle("Tuning data: effect of the LLVM compiler-text index weight", fontweight="bold", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "compiler_text_tuning_sweep.png", bbox_inches="tight"); plt.close(fig)

# ---- Figure 2: held-out comparison (needs both attribution tables)
a0 = pd.read_csv("reports/optima_analytics_dev/tables/attribution_components.csv")
a1_path = Path("reports/optima_analytics_compiler_on/tables/attribution_components.csv")
if a1_path.exists():
    a1 = pd.read_csv(a1_path)
    TOT, C = "TOTAL (full stack - baseline_rag)", "LLVM compiler representation"
    dsets = sorted(set(a0.dataset) & set(a1.dataset))
    metrics = [("recall_at_10", "Recall@10"), ("mrr", "MRR"), ("query_ms_median", "median latency (ms)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    rows = []
    for j, (m, title) in enumerate(metrics):
        ax = axes[0][j]
        x = np.arange(len(dsets)); w = 0.36
        for k, (tab, lab, col) in enumerate(((a0, "compiler text OFF (tuned weight 0)", AQUA), (a1, "compiler text ON (weight 1)", ORANGE))):
            v = [tab[(tab.dataset == d) & (tab.metric == m) & (tab.component == TOT)].iloc[0] for d in dsets]
            ax.bar(x + (k - 0.5) * w, [r["mean"] for r in v], w * 0.92, color=col, label=lab)
            ax.vlines(x + (k - 0.5) * w, [r["ci_low"] for r in v], [r["ci_high"] for r in v], color=INK2, lw=1)
            for d, r in zip(dsets, v):
                rows.append({"dataset": d, "metric": m, "profile": lab, "full_stack_gain_vs_baseline": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
        ax.set_xticks(x); ax.set_xticklabels(dsets); ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"Full improved stack: gain in {title} vs baseline_rag", fontsize=9.5)
        if j == 0: ax.legend(fontsize=8)
        ax = axes[1][j]
        v = [a1[(a1.dataset == d) & (a1.metric == m) & (a1.component == C)].iloc[0] for d in dsets]
        ax.bar(x, [r["mean"] for r in v], 0.5, color=ORANGE)
        ax.vlines(x, [r["ci_low"] for r in v], [r["ci_high"] for r in v], color=INK2, lw=1.2)
        ax.axhline(0, color=INK, lw=0.9); ax.set_xticks(x); ax.set_xticklabels(dsets)
        ax.set_title(f"Shapley contribution of the compiler text (weight 1) to {title}", fontsize=9.5)
        for d, r in zip(dsets, v):
            rows.append({"dataset": d, "metric": m, "profile": "compiler text contribution at weight 1", "full_stack_gain_vs_baseline": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    fig.suptitle("Held-out developer queries: switching the LLVM compiler text on", fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(OUT / "compiler_text_heldout.png", bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "heldout_compiler_on_vs_off.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
