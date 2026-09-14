from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH
# ============================================================

OUTPUT_DIR = Path("/home/igorr/Projects/op-python/optima/output")
REPORT_DIR = OUTPUT_DIR / "model_report"
GRAPH_DIR = REPORT_DIR / "graphs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("OPTIMA METRICS ANALYZER")
print("=" * 70)
print(f"Output directory: {OUTPUT_DIR}")
print()


# ============================================================
# CHECK DIRECTORY
# ============================================================

if not OUTPUT_DIR.exists():
    print("ERROR: Output directory does not exist!")
    print(OUTPUT_DIR)
    raise SystemExit(1)


# ============================================================
# FIND JSON FILES
# ============================================================

files = sorted(
    f for f in OUTPUT_DIR.glob("*.json")
    if f.name.lower() != "base.json"
)

print(f"Found {len(files)} model JSON files")
for f in files:
    print(f"  {f.name}")
print()

if not files:
    print("No model JSON files found.")
    raise SystemExit(1)


# ============================================================
# FIND ENRICHMENTS
# ============================================================

def find_enrichments(obj):
    """Recursively find nodes containing enrichment.evaluation."""

    results = []

    if isinstance(obj, dict):
        enrichment = obj.get("enrichment")

        if (
            isinstance(enrichment, dict)
            and isinstance(enrichment.get("evaluation"), dict)
        ):
            results.append(obj)

        for value in obj.values():
            results.extend(find_enrichments(value))

    elif isinstance(obj, list):
        for item in obj:
            results.extend(find_enrichments(item))

    return results


# ============================================================
# FUNCTION TYPE
# ============================================================

def classify_function_type(node):
    """
    Infer a broad function category from the Optima node identifier.

    This is intentionally conservative. It does not claim to know
    the exact C++ AST declaration kind unless the identifier gives
    enough information.
    """

    raw = str(
        node.get("node_id")
        or node.get("id")
        or node.get("name")
        or ""
    )

    lower = raw.lower()

    # Tests
    if (
        "::test::" in lower
        or "test::" in lower
        or lower.startswith("test::")
        or "benchmark" in lower
    ):
        return "Test"

    # Constructor / destructor patterns
    parts = [p for p in raw.split("::") if p]

    if len(parts) >= 2:
        function_name = parts[-1]
        parent_name = parts[-2]

        if function_name == parent_name:
            return "Constructor"

        if function_name == f"~{parent_name}":
            return "Destructor"

        return "Method"

    return "Function"


# ============================================================
# DEPENDENCY / GRAPH COMPLEXITY
# ============================================================

def get_dependency_count(node, enrichment):
    """
    Prefer the original node dependency list. Fall back to the
    enrichment dependency list.
    """

    dependencies = node.get("dependencies")

    if not isinstance(dependencies, list):
        dependencies = enrichment.get("dependencies", [])

    if isinstance(dependencies, list):
        return len(dependencies)

    return 0


def classify_graph_complexity(dependency_count):
    """
    Simple structural proxy until CFG/call-graph metrics are
    populated reliably.

        0       -> Leaf
        1-2     -> Low
        3-5     -> Medium
        >5      -> High
    """

    if dependency_count == 0:
        return "Leaf"

    if dependency_count <= 2:
        return "Low"

    if dependency_count <= 5:
        return "Medium"

    return "High"


# ============================================================
# PROCESS FILES
# ============================================================

records = []

for file in files:

    print("-" * 70)
    print(f"Processing: {file.name}")

    try:
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

    except Exception as e:
        print(f"ERROR loading JSON: {e}")
        continue

    nodes = find_enrichments(data)

    print(f"Found {len(nodes)} enriched functions")

    for node in nodes:

        enrichment = node.get("enrichment", {})
        evaluation = enrichment.get("evaluation", {})
        usage = enrichment.get("usage", {})

        context = evaluation.get("context_metrics", {})
        completeness = evaluation.get("field_completeness", {})

        node_id = (
            node.get("node_id")
            or node.get("id")
            or node.get("name")
            or "unknown"
        )

        model = enrichment.get("model", "unknown")

        dependency_count = get_dependency_count(
            node,
            enrichment,
        )

        source_chars = context.get("source_characters", 0) or 0
        llvm_chars = context.get("llvm_characters", 0) or 0

        records.append({

            # Identity
            "model": model,
            "function": node_id,
            "file": file.name,

            # Structural information
            "source_characters": source_chars,
            "llvm_characters": llvm_chars,
            "representation_size": source_chars + llvm_chars,
            "dependency_count": dependency_count,

            # Evaluation
            "json_valid": evaluation.get("json_valid"),
            "request_success": evaluation.get("request_success"),
            "latency_seconds": evaluation.get("latency_seconds"),
            "retry_count": evaluation.get("retry_count"),

            # Tokens
            "prompt_tokens": evaluation.get(
                "input_tokens",
                usage.get("prompt_tokens"),
            ),
            "output_tokens": evaluation.get(
                "output_tokens",
                usage.get("completion_tokens"),
            ),
            "total_tokens": evaluation.get(
                "total_tokens",
                usage.get("total_tokens"),
            ),

            # Generated information
            "keyword_count": evaluation.get("keyword_count"),
            "concept_count": evaluation.get("concept_count"),
            "summary_length": evaluation.get("summary_length"),
            "purpose_length": evaluation.get("purpose_length"),
            "behavior_length": evaluation.get("behavior_length"),

            # Context availability
            "source_code": context.get("source_code"),
            "ast": context.get("ast"),
            "llvm": context.get("llvm"),
            "cfg": context.get("cfg"),
            "call_graph": context.get("call_graph"),

            # Completeness
            "purpose_complete": completeness.get("purpose"),
            "behavior_complete": completeness.get("behavior"),
            "summary_complete": completeness.get("summary"),
            "inputs_complete": completeness.get("inputs"),
            "outputs_complete": completeness.get("outputs"),
            "side_effects_complete": completeness.get("side_effects"),
            "dependencies_complete": completeness.get("dependencies"),
            "concepts_complete": completeness.get("concepts"),
            "keywords_complete": completeness.get("keywords"),
            "algorithm_complete": completeness.get("algorithm"),
            "complexity_complete": completeness.get("complexity"),

            # Human evaluation
            "human_accuracy": evaluation.get("human", {}).get("accuracy"),
            "human_relevance": evaluation.get("human", {}).get("relevance"),
            "human_faithfulness": evaluation.get("human", {}).get("faithfulness"),
            "human_usefulness": evaluation.get("human", {}).get("usefulness"),

            # Function category
            "function_type": classify_function_type(node),

        })


# ============================================================
# CHECK RESULTS
# ============================================================

print()
print("=" * 70)
print(f"TOTAL FUNCTIONS FOUND: {len(records)}")
print("=" * 70)
print()

if not records:
    print("No function metrics were extracted.")
    raise SystemExit(1)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)

numeric_columns = [
    "source_characters",
    "llvm_characters",
    "representation_size",
    "dependency_count",
    "latency_seconds",
    "retry_count",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "keyword_count",
    "concept_count",
    "summary_length",
    "purpose_length",
    "behavior_length",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# COMPLEXITY CATEGORY
# ============================================================
#
# Use the distribution of representation size rather than
# arbitrary fixed thresholds.
#
# Bottom third  -> Simple
# Middle third  -> Moderate
# Top third     -> Complex
#
# This is based on source + LLVM representation size because
# CFG/call_count are currently not reliably populated.
# ============================================================

sizes = df["representation_size"]

q1 = sizes.quantile(1 / 3)
q2 = sizes.quantile(2 / 3)

def classify_complexity(value):
    if pd.isna(value):
        return "Unknown"

    if value <= q1:
        return "Simple"

    if value <= q2:
        return "Moderate"

    return "Complex"


df["complexity_category"] = sizes.apply(
    classify_complexity
)

df["graph_complexity"] = df["dependency_count"].apply(
    lambda x: classify_graph_complexity(
        int(x) if pd.notna(x) else 0
    )
)


# ============================================================
# FUNCTION-LEVEL CSV
# ============================================================

function_csv = REPORT_DIR / "function_metrics.csv"

df.to_csv(
    function_csv,
    index=False,
)

print(f"Function metrics saved: {function_csv}")
print()


# ============================================================
# COMMON AGGREGATION
# ============================================================

def aggregate(group):

    result = {
        "functions": group["function"].count(),

        "json_valid_rate":
            group["json_valid"].mean() * 100,

        "success_rate":
            group["request_success"].mean() * 100,

        "avg_latency":
            group["latency_seconds"].mean(),

        "median_latency":
            group["latency_seconds"].median(),

        "avg_input_tokens":
            group["prompt_tokens"].mean(),

        "avg_output_tokens":
            group["output_tokens"].mean(),

        "avg_total_tokens":
            group["total_tokens"].mean(),

        "avg_keywords":
            group["keyword_count"].mean(),

        "avg_concepts":
            group["concept_count"].mean(),

    }

    completeness_columns = [
        "purpose_complete",
        "behavior_complete",
        "summary_complete",
        "inputs_complete",
        "outputs_complete",
        "side_effects_complete",
        "dependencies_complete",
        "concepts_complete",
        "keywords_complete",
        "algorithm_complete",
        "complexity_complete",
    ]

    available = [
        c for c in completeness_columns
        if c in group.columns
    ]

    if available:
        values = group[available].apply(
            pd.to_numeric,
            errors="coerce",
        )

        result["field_completeness_rate"] = (
            values.mean(axis=1).mean() * 100
        )
    else:
        result["field_completeness_rate"] = None

    return pd.Series(result)


# ============================================================
# 1. OVERALL MODEL COMPARISON
# ============================================================

summary = (
    df.groupby("model", dropna=False)
      .apply(aggregate, include_groups=False)
      .reset_index()
)

summary = summary.sort_values(
    "field_completeness_rate",
    ascending=False,
)

summary_csv = REPORT_DIR / "model_comparison.csv"

summary.to_csv(
    summary_csv,
    index=False,
)


# ============================================================
# 2. MODEL × COMPLEXITY
# ============================================================

complexity_summary = (
    df.groupby(
        ["model", "complexity_category"],
        dropna=False,
    )
    .apply(aggregate, include_groups=False)
    .reset_index()
)

complexity_order = {
    "Simple": 0,
    "Moderate": 1,
    "Complex": 2,
    "Unknown": 3,
}

complexity_summary["_order"] = (
    complexity_summary["complexity_category"]
    .map(complexity_order)
)

complexity_summary = (
    complexity_summary
    .sort_values(["_order", "model"])
    .drop(columns="_order")
)

complexity_csv = REPORT_DIR / "model_by_complexity.csv"

complexity_summary.to_csv(
    complexity_csv,
    index=False,
)


# ============================================================
# 3. MODEL × FUNCTION TYPE
# ============================================================

function_type_summary = (
    df.groupby(
        ["model", "function_type"],
        dropna=False,
    )
    .apply(aggregate, include_groups=False)
    .reset_index()
)

function_type_csv = REPORT_DIR / "model_by_function_type.csv"

function_type_summary.to_csv(
    function_type_csv,
    index=False,
)


# ============================================================
# 4. MODEL × GRAPH COMPLEXITY
# ============================================================

graph_summary = (
    df.groupby(
        ["model", "graph_complexity"],
        dropna=False,
    )
    .apply(aggregate, include_groups=False)
    .reset_index()
)

graph_order = {
    "Leaf": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

graph_summary["_order"] = (
    graph_summary["graph_complexity"]
    .map(graph_order)
)

graph_summary = (
    graph_summary
    .sort_values(["_order", "model"])
    .drop(columns="_order")
)

graph_csv = REPORT_DIR / "model_by_graph_complexity.csv"

graph_summary.to_csv(
    graph_csv,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("OVERALL MODEL COMPARISON")
print("=" * 70)
print(
    summary.to_string(index=False)
)
print()

print("MODEL × COMPLEXITY")
print("=" * 70)
print(
    complexity_summary.to_string(index=False)
)
print()

print("MODEL × FUNCTION TYPE")
print("=" * 70)
print(
    function_type_summary.to_string(index=False)
)
print()

print("MODEL × GRAPH COMPLEXITY")
print("=" * 70)
print(
    graph_summary.to_string(index=False)
)
print()


# ============================================================
# GRAPH HELPERS
# ============================================================

def grouped_bar(
    data,
    category_column,
    metric,
    title,
    ylabel,
    filename,
    order=None,
):

    if metric not in data.columns:
        return

    if category_column not in data.columns:
        return

    plot = data[
        [category_column, "model", metric]
    ].dropna()

    if plot.empty:
        return

    if order:
        plot[category_column] = pd.Categorical(
            plot[category_column],
            categories=order,
            ordered=True,
        )

    pivot = plot.pivot(
        index=category_column,
        columns="model",
        values=metric,
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 7),
    )

    ax.set_title(title)
    ax.set_xlabel(category_column.replace("_", " ").title())
    ax.set_ylabel(ylabel)

    plt.xticks(
        rotation=0,
    )

    plt.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()

    path = GRAPH_DIR / filename

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Graph: {path}")


# ============================================================
# COMPLEXITY GRAPHS
# ============================================================

complexity_categories = [
    "Simple",
    "Moderate",
    "Complex",
]

grouped_bar(
    complexity_summary,
    "complexity_category",
    "field_completeness_rate",
    "Field Completeness by Function Complexity",
    "Field Completeness (%)",
    "complexity_completeness.png",
    complexity_categories,
)

grouped_bar(
    complexity_summary,
    "complexity_category",
    "avg_latency",
    "Latency by Function Complexity",
    "Average Latency (seconds)",
    "complexity_latency.png",
    complexity_categories,
)

grouped_bar(
    complexity_summary,
    "complexity_category",
    "avg_total_tokens",
    "Token Usage by Function Complexity",
    "Average Total Tokens",
    "complexity_tokens.png",
    complexity_categories,
)

grouped_bar(
    complexity_summary,
    "complexity_category",
    "avg_keywords",
    "Keywords by Function Complexity",
    "Average Keywords",
    "complexity_keywords.png",
    complexity_categories,
)

grouped_bar(
    complexity_summary,
    "complexity_category",
    "avg_concepts",
    "Concepts by Function Complexity",
    "Average Concepts",
    "complexity_concepts.png",
)


# ============================================================
# FUNCTION TYPE GRAPHS
# ============================================================

function_types = sorted(
    df["function_type"].dropna().unique()
)

grouped_bar(
    function_type_summary,
    "function_type",
    "field_completeness_rate",
    "Field Completeness by Function Type",
    "Field Completeness (%)",
    "function_type_completeness.png",
    function_types,
)

grouped_bar(
    function_type_summary,
    "function_type",
    "avg_latency",
    "Latency by Function Type",
    "Average Latency (seconds)",
    "function_type_latency.png",
    function_types,
)


# ============================================================
# GRAPH COMPLEXITY GRAPHS
# ============================================================

graph_categories = [
    "Leaf",
    "Low",
    "Medium",
    "High",
]

grouped_bar(
    graph_summary,
    "graph_complexity",
    "field_completeness_rate",
    "Field Completeness by Graph Complexity",
    "Field Completeness (%)",
    "graph_complexity_completeness.png",
    graph_categories,
)

grouped_bar(
    graph_summary,
    "graph_complexity",
    "avg_latency",
    "Latency by Graph Complexity",
    "Average Latency (seconds)",
    "graph_complexity_latency.png",
    graph_categories,
)


# ============================================================
# EXCEL REPORT
# ============================================================

excel_path = REPORT_DIR / "model_report.xlsx"

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl",
) as writer:

    summary.to_excel(
        writer,
        sheet_name="Model Comparison",
        index=False,
    )

    complexity_summary.to_excel(
        writer,
        sheet_name="By Complexity",
        index=False,
    )

    function_type_summary.to_excel(
        writer,
        sheet_name="By Function Type",
        index=False,
    )

    graph_summary.to_excel(
        writer,
        sheet_name="By Graph Complexity",
        index=False,
    )

    df.to_excel(
        writer,
        sheet_name="Function Metrics",
        index=False,
    )


# ============================================================
# MARKDOWN REPORT
# ============================================================

report_path = REPORT_DIR / "report.md"

with report_path.open(
    "w",
    encoding="utf-8",
) as f:

    f.write("# Optima Model Enrichment Report\n\n")

    f.write(
        f"- Functions analyzed: **{len(df)}**\n"
    )

    f.write(
        f"- Models analyzed: **{df['model'].nunique()}**\n"
    )

    f.write(
        f"- Models: {', '.join(df['model'].dropna().unique())}\n\n"
    )

    f.write("## Overall Model Comparison\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Model × Function Complexity\n\n")
    f.write(complexity_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Model × Function Type\n\n")
    f.write(function_type_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Model × Graph Complexity\n\n")
    f.write(graph_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Complexity Definition\n\n")
    f.write(
        "Function complexity is calculated from the combined "
        "source-code and LLVM representation size. Functions are "
        "divided into Simple, Moderate, and Complex using the "
        "lower and upper tertiles of the complete benchmark.\n\n"
    )

    f.write("## Graph Complexity Definition\n\n")
    f.write(
        "Graph complexity currently uses dependency count as a "
        "structural proxy: Leaf = 0 dependencies, Low = 1-2, "
        "Medium = 3-5, High = more than 5. CFG/call-graph metrics "
        "should replace this proxy once those metrics are populated "
        "reliably.\n\n"
    )

    f.write("## Generated Graphs\n\n")

    for graph in sorted(GRAPH_DIR.glob("*.png")):
        f.write(f"- `{graph.name}`\n")


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print(f"Report              : {report_path}")
print(f"Excel               : {excel_path}")
print(f"Overall CSV         : {summary_csv}")
print(f"Function CSV        : {function_csv}")
print(f"Complexity CSV      : {complexity_csv}")
print(f"Function Type CSV   : {function_type_csv}")
print(f"Graph Complexity CSV: {graph_csv}")
print(f"Graphs              : {GRAPH_DIR}")
print()
