#!/usr/bin/env python3
"""
compare_vs_pca.py

Reads summary_df_*.csv and params_dict_*.csv from results/.
For each (N,n,rts) combination, picks the sqrt dimensionality reduction level
for UMAP and for PCA, then compares them head-to-head.

Output: tex/comparison/umap_sqrt_vs_pca_sqrt_{crs,vrs}.tex
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


def generate_table(rows, rts_label, k_val=None, hyperparams_suffix=""):
    """Generate a UMAP-DEA vs PCA-DEA comparison LaTeX table (both at sqrt level).

    Note: The ``% Non-discriminating`` metric is intentionally omitted from this
    table because the goal is to compare dimensionality-reduction methods (UMAP
    vs PCA), not to evaluate discrimination quality.  The metric is still computed
    and stored in the ``non_discrim`` field of each experiment record for possible
    use in other analyses.
    """
    if k_val is not None:
        caption = (
            f"UMAP-DEA ($k={k_val}$, $d=\\sqrt{{N}}$) vs PCA-DEA "
            f"($d=\\sqrt{{N}}$). {rts_label.upper()}{hyperparams_suffix}"
        )
        label = f"tab:umap_k{k_val}_sqrt_vs_pca_sqrt_{rts_label}"
    else:
        caption = (
            f"UMAP-DEA vs PCA-DEA "
            f"(both $d=\\sqrt{{N}}$). {rts_label.upper()}{hyperparams_suffix}"
        )
        label = f"tab:umap_sqrt_vs_pca_sqrt_{rts_label}"

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(f"\\caption{{{caption}\\label{{{label}}}}}")
    lines.append("\\begin{tabular}{@{}c c c c c c c c c c@{}}")
    lines.append("\\toprule")
    lines.append(
        " & & \\multicolumn{2}{c}{MAE} & "
        "\\multicolumn{2}{c}{Spearman $\\rho$} & "
        "\\multicolumn{2}{c}{Pearson $r$} & "
        "\\multicolumn{2}{c}{Kendall $\\tau$} \\\\"
    )
    lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}")
    lines.append(
        "$N$ & $n$ & UMAP & PCA & UMAP & PCA & "
        "UMAP & PCA & UMAP & PCA \\\\"
    )
    lines.append("\\midrule")

    prev_N = None
    for i, r in enumerate(rows):
        if prev_N is not None and r["N"] != prev_N:
            lines.append("\\addlinespace")
        prev_N = r["N"]

        mae_u, mae_p = bold_if_better(r["umap_mae"], r["pca_mae"], "min")
        spe_u, spe_p = bold_if_better(r["umap_spearman"], r["pca_spearman"], "max")
        pea_u, pea_p = bold_if_better(r["umap_pearson"], r["pca_pearson"], "max")
        ken_u, ken_p = bold_if_better(r["umap_kendall"], r["pca_kendall"], "max")
        lines.append(
            f" {r['N']} & {r['n']} & "
            f"{mae_u} & {mae_p} & "
            f"{spe_u} & {spe_p} & "
            f"{pea_u} & {pea_p} & "
            f"{ken_u} & {ken_p} \\\\"
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
            "mae_mean": sqrt_row["mae_mean"],
            "spearman": sqrt_row.get("spearmanr_mean", np.nan),
            "pearson": sqrt_row.get("pearsonr_mean", np.nan),
            "kendall": sqrt_row.get("kendalltau_mean", np.nan),
            "non_discrim": sqrt_row.get("spearmanr_warning_count", 0) / nr_sim,
        })

    df = pd.DataFrame(experiments)

    umap = df[df["pca"] == False].copy()
    pca = df[df["pca"] == True].copy()

    umap = umap.rename(columns={
        "mae_mean": "umap_mae",
        "spearman": "umap_spearman",
        "pearson": "umap_pearson",
        "kendall": "umap_kendall",
        "non_discrim": "umap_non_discrim",
    })
    pca = pca.rename(columns={
        "mae_mean": "pca_mae",
        "spearman": "pca_spearman",
        "pearson": "pca_pearson",
        "kendall": "pca_kendall",
        "non_discrim": "pca_non_discrim",
    })

    umap = umap.drop(columns=["pca"])
    pca = pca.drop(columns=["pca"])

    # Group UMAP by k first, then merge each k-group with PCA separately
    for k_val, umap_k in umap.groupby("k"):
        k_val = int(float(k_val)) if k_val is not None and not pd.isna(k_val) else None
        merged = pd.merge(umap_k, pca, on=["N", "n", "rts", "gamma", "sigma_u", "alpha_1", "M"], how="inner")

        # Group by gamma
        for gamma_val, gamma_merged in merged.groupby("gamma"):
            # Format gamma for directory name
            gamma_str = gamma_to_dirname(gamma_val)
            comparison_dir = os.path.join(OUTPUT_BASE_DIR, f"gamma_{gamma_str}", "comparison")
            os.makedirs(comparison_dir, exist_ok=True)

            for rts in ["crs", "vrs"]:
                rts_df = gamma_merged[gamma_merged["rts"] == rts].sort_values(["N", "n"])
                if rts_df.empty:
                    print(f"No matched UMAP/PCA pairs for gamma={gamma_val} k={k_val} {rts.upper()}, skipping.")
                    continue

                rows = rts_df.to_dict("records")
                # Extract shared hyperparams from the first row (use pca=False so UMAP params are not appended)
                first_row = rows[0]
                shared_params = {
                    "gamma": first_row.get("gamma"),
                    "sigma_u": first_row.get("sigma_u"),
                    "alpha_1": first_row.get("alpha_1"),
                    "M": first_row.get("M"),
                    "pca": False,  # suppress UMAP-specific in suffix for method-agnostic comparison
                }
                hyperparams_suffix = format_hyperparams_suffix(shared_params, include_method_specific=False)
                table = generate_table(rows, rts, k_val=k_val, hyperparams_suffix=hyperparams_suffix)

                if k_val is not None:
                    filename = f"umap_k{k_val}_sqrt_vs_pca_sqrt_{rts}.tex"
                else:
                    filename = f"umap_sqrt_vs_pca_sqrt_{rts}.tex"
                filepath = os.path.join(comparison_dir, filename)
                with open(filepath, "w") as f:
                    f.write(table)
                print(f"Wrote {filepath} (matched {len(rows)} UMAP/PCA pairs, k={k_val})")


if __name__ == "__main__":
    main()