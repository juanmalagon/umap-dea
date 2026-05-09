#!/usr/bin/env python3
"""
generate_comparison_umap_vs_conventional.py

Reads summary_df_*.csv and params_dict_*.csv from results/.
For each experiment, picks the sqrt dimensionality reduction level
and compares it vs conventional DEA (original),
separated by RTS (CRS/VRS) and method (UMAP/PCA).

Output: tex/comparison/umap_sqrt_vs_conventional_{crs,vrs}.tex
        tex/comparison/pca_sqrt_vs_conventional_{crs,vrs}.tex
"""

import glob
import os
import re

import pandas as pd
import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "tex", "comparison")


def extract_uuid(filename):
    match = re.search(r"(?:summary_df|params_dict)_([0-9a-fA-F-]+)\.csv", filename)
    if match:
        return match.group(1)
    return ""


def fmt_val(val, ndigits=4):
    """Format a single numeric value."""
    if pd.isna(val):
        return "—"
    return f"{val:.{ndigits}f}"


def fmt_pct(val):
    """Format as percentage with one decimal."""
    if pd.isna(val):
        return "—"
    return f"{val * 100:.1f}\\%"


def bold_if_better(val_best, val_orig, best_direction, ndigits=4):
    """Return (best_str, orig_str) with the better value bolded.
    best_direction: 'min' means lower is better, 'max' means higher is better."""
    if pd.isna(val_best) or pd.isna(val_orig):
        return fmt_val(val_best, ndigits), fmt_val(val_orig, ndigits)

    best_str = f"{val_best:.{ndigits}f}"
    orig_str = f"{val_orig:.{ndigits}f}"

    if best_direction == "min":
        better_best = val_best <= val_orig
    else:
        better_best = val_best >= val_orig

    if better_best:
        best_str = f"\\textbf{{{best_str}}}"
    else:
        orig_str = f"\\textbf{{{orig_str}}}"

    return best_str, orig_str


def bold_pct_if_better(val_best, val_orig, best_direction):
    """Same for percentage values (lower is better for non-discriminating)."""
    if pd.isna(val_best) or pd.isna(val_orig):
        return fmt_pct(val_best), fmt_pct(val_orig)

    pct_best = val_best * 100
    pct_orig = val_orig * 100
    best_str = f"{pct_best:.1f}\\%"
    orig_str = f"{pct_orig:.1f}\\%"

    if best_direction == "min":
        better_best = pct_best <= pct_orig
    else:
        better_best = pct_best >= pct_orig

    if better_best:
        best_str = f"\\textbf{{{best_str}}}"
    else:
        orig_str = f"\\textbf{{{orig_str}}}"

    return best_str, orig_str


def generate_table(rows, rts_label, method_label):
    """Generate a single comparison LaTeX table for a given (rts, method)."""

    caption = (
        f"{method_label}-DEA ($d=\\sqrt{{N}}$) vs Conventional DEA "
        f"(no reduction). {rts_label.upper()}"
    )
    label = f"tab:{method_label.lower()}_sqrt_vs_conventional_{rts_label}"

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append(f"\\caption{{{caption}\\label{{{label}}}}}")
    lines.append("\\begin{tabular}{@{}c c c c c c c c c c c c@{}}")
    lines.append("\\toprule")
    lines.append(
        " & & \\multicolumn{2}{c}{MAE} & "
        "\\multicolumn{2}{c}{Spearman $\\rho$} & "
        "\\multicolumn{2}{c}{Pearson $r$} & "
        "\\multicolumn{2}{c}{Kendall $\\tau$} & "
        "\\multicolumn{2}{c}{\\% Non-discrim.} \\\\"
    )
    lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10} \\cmidrule(lr){11-12}")
    lines.append(
        f"$N$ & $n$ & {method_label} & Orig. & {method_label} & Orig. & "
        f"{method_label} & Orig. & {method_label} & Orig. & {method_label} & Orig. \\\\"
    )
    lines.append("\\midrule")

    prev_N = None
    for i, r in enumerate(rows):
        if prev_N is not None and r["N"] != prev_N:
            lines.append("\\addlinespace")
        prev_N = r["N"]

        # MAE
        mae_best, mae_orig = bold_if_better(
            r["sqrt_mae_mean"], r["orig_mae_mean"], "min"
        )

        # Spearman
        spe_best, spe_orig = bold_if_better(
            r["sqrt_spearman"], r["orig_spearman"], "max"
        )

        # Pearson
        pea_best, pea_orig = bold_if_better(
            r["sqrt_pearson"], r["orig_pearson"], "max"
        )

        # Kendall
        ken_best, ken_orig = bold_if_better(
            r["sqrt_kendall"], r["orig_kendall"], "max"
        )

        # % Non-discriminating
        nd_best, nd_orig = bold_pct_if_better(
            r["sqrt_non_discrim"], r["orig_non_discrim"], "min"
        )

        lines.append(
            f" {r['N']} & {r['n']} & "
            f"{mae_best} & {mae_orig} & "
            f"{spe_best} & {spe_orig} & "
            f"{pea_best} & {pea_orig} & "
            f"{ken_best} & {ken_orig} & "
            f"{nd_best} & {nd_orig} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    summary_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "summary_df_*.csv")))

    experiments = []

    for summary_path in summary_files:
        uuid = extract_uuid(os.path.basename(summary_path))
        if not uuid:
            continue

        params_path = os.path.join(RESULTS_DIR, f"params_dict_{uuid}.csv")
        if not os.path.exists(params_path):
            continue

        summary_df = pd.read_csv(summary_path)
        params_df = pd.read_csv(params_path)

        if summary_df.empty or params_df.empty:
            continue

        params = params_df.iloc[0].to_dict()

        # Pick sqrt level
        sqrt_rows = summary_df[summary_df["dim_reduction_level"] == "sqrt"]
        if sqrt_rows.empty:
            print(f"Warning: no sqrt level found for {uuid}, skipping.")
            continue
        sqrt_row = sqrt_rows.iloc[0]

        # Get original row
        orig_rows = summary_df[summary_df["dim_reduction_level"] == "original"]
        if orig_rows.empty:
            continue
        orig_row = orig_rows.iloc[0]

        nr_sim = int(params.get("nr_simulations", 1000))

        experiments.append({
            "N": int(params["N"]),
            "n": int(params["n"]),
            "rts": params["rts"],
            "pca": params["pca"],
            # Sqrt metrics
            "sqrt_mae_mean": sqrt_row["mae_mean"],
            "sqrt_mae_std": sqrt_row["mae_std"],
            "sqrt_spearman": sqrt_row.get("spearmanr_mean", np.nan),
            "sqrt_pearson": sqrt_row.get("pearsonr_mean", np.nan),
            "sqrt_kendall": sqrt_row.get("kendalltau_mean", np.nan),
            "sqrt_non_discrim": sqrt_row.get("spearmanr_warning_count", 0) / nr_sim,
            # Original metrics
            "orig_mae_mean": orig_row["mae_mean"],
            "orig_mae_std": orig_row["mae_std"],
            "orig_spearman": orig_row.get("spearmanr_mean", np.nan),
            "orig_pearson": orig_row.get("pearsonr_mean", np.nan),
            "orig_kendall": orig_row.get("kendalltau_mean", np.nan),
            "orig_non_discrim": orig_row.get("spearmanr_warning_count", 0) / nr_sim,
        })

    df = pd.DataFrame(experiments)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate 4 tables: {umap,pca} x {crs,vrs}
    for method, method_label in [(True, "PCA"), (False, "UMAP")]:
        method_df = df[df["pca"] == method]
        for rts in ["crs", "vrs"]:
            rts_df = method_df[method_df["rts"] == rts].sort_values(["N", "n"])
            if rts_df.empty:
                print(f"No data for {method_label} {rts.upper()}, skipping.")
                continue

            rows = rts_df.to_dict("records")
            table = generate_table(rows, rts, method_label)

            filename = f"{method_label.lower()}_sqrt_vs_conventional_{rts}.tex"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w") as f:
                f.write(table)
            print(f"Wrote {filepath}")


if __name__ == "__main__":
    main()