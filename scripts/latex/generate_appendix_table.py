#!/usr/bin/env python3
r"""
Generate appendix \longtable snippets from all_results.csv.
"""

import os

import pandas as pd

# Paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
CSV_PATH = os.path.join(_PROJECT_ROOT, "all_results.csv")
OUT_PATH = os.path.join(_SCRIPT_DIR, "generated", "all_results_tables.tex")

# Metric columns (mean only; *_std columns are intentionally omitted)
METRIC_COLS = [
    "mae_mean",
    "spearmanr_mean",
    "pearsonr_mean",
    "kendalltau_mean",
    "prop_efficient_mean",
]

# Order of methods within each table
METHOD_ORDER = ["DEA", "UMAP-DEA", "PCA-DEA"]

# Display names for the reduction levels (LaTeX-safe)
REDUCTION_DISPLAY = {
    "original": "--",
    "half": "half",
    "sqrt": "sqrt",
    "ten_percent": r"10\%",
    "log": "log",
}


def fmt_number(value) -> str:
    """Round to exactly three decimals."""
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.3f}"


def reduction_display(level, dims) -> str:
    """Return the combined ``Reduction (d)`` cell text."""
    if level == "original":
        return "--"
    return f"{REDUCTION_DISPLAY[level]} ({int(dims)})"


def build_rows(df: pd.DataFrame) -> list[dict]:
    """Return a list of row dicts in final display order."""
    rows: list[dict] = []

    # DEA baseline: one row per (N, n); metrics are identical across
    # duplicates, so dedupe on (N, n).
    original = df[df["dim_reduction_level"] == "original"].copy()
    original = original.drop_duplicates(subset=["N", "n"])
    for _, r in original.sort_values(["N", "n"]).iterrows():
        rows.append({
            "method": "DEA",
            "N": int(r["N"]),
            "n": int(r["n"]),
            "k": "--",
            "reduction": reduction_display("original", r["dims"]),
            "d": int(r["dims"]),
            **{c: r[c] for c in METRIC_COLS},
        })

    # Reduced embeddings: UMAP vs PCA
    reduced = df[df["dim_reduction_level"] != "original"].copy()
    reduced["method"] = reduced["pca"].map({True: "PCA-DEA", False: "UMAP-DEA"})

    for method in ["UMAP-DEA", "PCA-DEA"]:
        sub = reduced[reduced["method"] == method]
        for _, r in sub.sort_values(["n", "dim_reduction_level", "umap_n_neighbors"]).iterrows():
            k = int(r["umap_n_neighbors"]) if method == "UMAP-DEA" else "--"
            rows.append({
                "method": method,
                "N": int(r["N"]),
                "n": int(r["n"]),
                "k": k,
                "reduction": reduction_display(r["dim_reduction_level"], r["dims"]),
                "d": int(r["dims"]),
                **{c: r[c] for c in METRIC_COLS},
            })

    # Final ordering: method order, then n, then descending d (with reduction
    # name as a stable tiebreak), then ascending k.
    def sort_key(row):
        k_key = -1 if row["k"] == "--" else int(row["k"])
        return (
            METHOD_ORDER.index(row["method"]),
            row["n"],
            -row["d"],
            row["reduction"],
            k_key,
        )

    rows.sort(key=sort_key)
    return rows


def render_longtable(N: int, rows: list[dict]) -> str:
    """Render one longtable for a given value of N."""
    subset = [r for r in rows if r["N"] == N]

    header = (
        r"Method & \(n\) & \(k\) & Reduction (\(d\)) & MAE & "
        r"\makecell{Spearman\\\(\rho\)} & \makecell{Pearson\\\(r\)} & "
        r"\makecell{Kendall\\\(\tau\)} & \makecell{Prop.\\eff.} \\"
    )

    lines = []
    lines.append(r"{\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{longtable}{lrrlrrrrr}")
    lines.append(
        rf"\caption{{Complete Monte Carlo results for \(N={N}\) inputs. The "
        rf"reduction column reports the reduction rule and the resulting "
        rf"embedding dimension \(d\) in parentheses; \enquote{{--}} denotes no "
        rf"reduction. Metrics are rounded to three decimals.}}"
        rf"\label{{tab:full_results_N{N}}}\\"
    )
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(
        r"\multicolumn{9}{c}{{\tablename\ \thetable{} -- continued from previous page}} \\"
    )
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{9}{r}{{Continued on next page}} \\"
    )
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for r in subset:
        k_cell = "" if r["k"] == "--" else r["k"]
        cells = [
            r["method"],
            r["n"],
            k_cell if k_cell != "" else "--",
            r["reduction"],
            fmt_number(r["mae_mean"]),
            fmt_number(r["spearmanr_mean"]),
            fmt_number(r["pearsonr_mean"]),
            fmt_number(r["kendalltau_mean"]),
            fmt_number(r["prop_efficient_mean"]),
        ]
        lines.append(" & ".join(str(c) for c in cells) + r" \\")

    lines.append(r"\end{longtable}")
    lines.append(r"}")
    return "\n".join(lines)


def main():
    df = pd.read_csv(CSV_PATH)
    rows = build_rows(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    blocks = []
    for N in sorted(df["N"].unique()):
        blocks.append(render_longtable(int(N), rows))

    content = "\n\n".join(blocks) + "\n"

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Wrote {len(blocks)} longtable block(s) to {OUT_PATH}")
    print(f"Total rows written: {sum(1 for r in rows)}")


if __name__ == "__main__":
    main()