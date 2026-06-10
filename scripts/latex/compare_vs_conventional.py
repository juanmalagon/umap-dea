#!/usr/bin/env python3
"""
compare_vs_conventional.py

Reads summary_df_*.csv and params_dict_*.csv from results/.
For each experiment, picks the sqrt dimensionality reduction level
and compares it vs conventional DEA (original),
separated by RTS (CRS/VRS) and method (UMAP/PCA).

Output: tex/comparison/umap_sqrt_vs_conventional_{crs,vrs}.tex
        tex/comparison/pca_sqrt_vs_conventional_{crs,vrs}.tex
"""

import glob
import os

import pandas as pd
import numpy as np

from _utils import gamma_to_dirname, extract_uuid, fmt_val, fmt_pct, bold_if_better, bold_pct_if_better, format_hyperparams_suffix


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
OUTPUT_BASE_DIR = os.path.join(_PROJECT_ROOT, "tex")


def generate_table(rows, rts_label, method_label, k_val=None, hyperparams_suffix=""):
    """Generate a single comparison LaTeX table for a given (rts, method, optional k)."""

    if k_val is not None:
        caption = (
            f"{method_label}-DEA ($k={k_val}$, $d=\\sqrt{{N}}$) vs Conventional DEA "
            f"(no reduction). {rts_label.upper()}{hyperparams_suffix}"
        )
        label = f"tab:{method_label.lower()}_k{k_val}_sqrt_vs_conventional_{rts_label}"
    else:
        caption = (
            f"{method_label}-DEA ($d=\\sqrt{{N}}$) vs Conventional DEA "
            f"(no reduction). {rts_label.upper()}{hyperparams_suffix}"
        )
        label = f"tab:{method_label.lower()}_sqrt_vs_conventional_{rts_label}"

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
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
    summary_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "**", "summary_df_*.csv"), recursive=True))

    experiments = []

    for summary_path in summary_files:
        uuid = extract_uuid(os.path.basename(summary_path))
        if not uuid:
            continue

        params_path = os.path.join(os.path.dirname(summary_path), f"params_dict_{uuid}.csv")
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
        is_pca = bool(params.get("pca", False))
        k_val = None if is_pca else int(float(params.get("umap_n_neighbors", 15)))

        experiments.append({
            "N": int(params["N"]),
            "n": int(params["n"]),
            "rts": params["rts"],
            "pca": params["pca"],
            "k": k_val,
            "gamma": params.get("gamma", "unknown"),
            "sigma_u": params.get("sigma_u", None),
            "alpha_1": params.get("alpha_1", None),
            "M": params.get("M", None),
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

    # Group by gamma
    for gamma_val, gamma_df in df.groupby("gamma"):
        # Format gamma for directory name
        gamma_str = gamma_to_dirname(gamma_val)
        comparison_dir = os.path.join(OUTPUT_BASE_DIR, f"gamma_{gamma_str}", "comparison")
        os.makedirs(comparison_dir, exist_ok=True)

        # Generate tables: {umap,pca} x {crs,vrs}, with k subgroups for UMAP
        for method, method_label in [(True, "PCA"), (False, "UMAP")]:
            method_df = gamma_df[gamma_df["pca"] == method]
            if method_df.empty:
                continue

            # For UMAP, further group by k; for PCA, use a single dummy group
            if method:
                # PCA: no k parameter
                k_groups = [(None, method_df)]
            else:
                # UMAP: split by k (cast to int since NaN -> float column)
                k_groups = [(int(k), grp) for k, grp in method_df.groupby("k")]

            for k_val, k_df in k_groups:
                for rts in ["crs", "vrs"]:
                    rts_df = k_df[k_df["rts"] == rts].sort_values(["N", "n"])
                    if rts_df.empty:
                        print(f"No data for gamma={gamma_val} {method_label} k={k_val} {rts.upper()}, skipping.")
                        continue

                    rows = rts_df.to_dict("records")
                    # Extract shared hyperparams from the first row
                    first_row = rows[0]
                    shared_params = {
                        "gamma": first_row.get("gamma"),
                        "sigma_u": first_row.get("sigma_u"),
                        "alpha_1": first_row.get("alpha_1"),
                        "M": first_row.get("M"),
                        "pca": method,  # needed to skip UMAP-specific params
                    }
                    hyperparams_suffix = format_hyperparams_suffix(shared_params, include_method_specific=False)
                    table = generate_table(rows, rts, method_label, k_val=k_val, hyperparams_suffix=hyperparams_suffix)

                    if k_val is not None:
                        filename = f"{method_label.lower()}_k{k_val}_sqrt_vs_conventional_{rts}.tex"
                    else:
                        filename = f"{method_label.lower()}_sqrt_vs_conventional_{rts}.tex"
                    filepath = os.path.join(comparison_dir, filename)
                    with open(filepath, "w") as f:
                        f.write(table)
                    print(f"Wrote {filepath}")


if __name__ == "__main__":
    main()