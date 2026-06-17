#!/usr/bin/env python3
"""
Generate LaTeX tables from simulation results.

Reads all params_dict_*.csv and summary_df_*.csv pairs from the results folder,
groups runs by common hyperparameters (excluding n, which becomes the row variable),
and outputs one .tex file per group containing:

1. A sidewaystable (landscape) with accuracy, correlation, and discrimination
   metrics (mean and standard deviation).

2. A standard table (portrait) with accuracy, correlation, and discrimination
   metrics (mean only, no standard deviations, \\scriptsize font).
"""

import os
import re
import glob
import hashlib
import pandas as pd
import numpy as np

from _utils import gamma_to_dirname, get_output_subpath_from_params

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
OUTPUT_BASE_DIR = os.path.join(_PROJECT_ROOT, "tex")


def load_all_runs(results_dir: str) -> list[dict]:
    """
    Scan results_dir for params_dict_*.csv and summary_df_*.csv pairs,
    match them by run_serial (UUID), and return a list of combined records.
    """
    params_files = glob.glob(os.path.join(results_dir, "**", "params_dict_*.csv"), recursive=True)

    runs = []
    for params_path in params_files:
        basename = os.path.basename(params_path)
        match = re.match(r"params_dict_(.+)\.csv", basename)
        if not match:
            continue
        run_serial = match.group(1)

        params_df = pd.read_csv(params_path)
        if params_df.empty:
            print(f"Warning: empty params file {params_path}, skipping")
            continue
        params_dict = params_df.iloc[0].to_dict()

        summary_path = os.path.join(
            os.path.dirname(params_path), f"summary_df_{run_serial}.csv"
        )
        if not os.path.exists(summary_path):
            print(f"Warning: missing summary file {summary_path}, skipping")
            continue
        summary_df = pd.read_csv(summary_path)

        runs.append(
            {
                "run_serial": run_serial,
                "params": params_dict,
                "summary_df": summary_df,
            }
        )

    return runs


def _format_param_value(key: str, value) -> str:
    """Format a parameter value for LaTeX display."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str):
        return value.replace("_", r"\_")
    return str(value)


def _format_number(value) -> str:
    """Format a numeric value for LaTeX; show '---' for NaN (e.g. VRS degeneracy)."""
    try:
        if pd.isna(value):
            return "---"
    except (TypeError, ValueError):
        pass
    return f"{value:.4f}"


def _dim_reduction_label(level: str) -> str:
    """Map internal dim_reduction_level names to LaTeX display labels."""
    mapping = {
        "log": "log",
        "sqrt": "sqrt",
        "ten_percent": r"10\%",
        "half": "half",
        "original": "original",
    }
    return mapping.get(level, level)


def _is_pca(params: dict) -> bool:
    """Return True if this parameter set uses PCA (not UMAP) for dim reduction."""
    return bool(params.get("pca", False))


def _generate_group_label(params: dict) -> str:
    """
    Generate a unique identifying label for a group of runs (for file naming).
    Encodes all hyperparameters that define the group.
    """
    parts = []

    parts.append(f"N_{params.get('N', 'missing')}")

    rts = params.get("rts", "")
    orient = params.get("orientation", "")
    parts.append(f"{rts}_{orient}")

    parts.append(f"gamma_{_format_param_value_for_filename(params.get('gamma', '?'))}")
    parts.append(f"sigma_{_format_param_value_for_filename(params.get('sigma_u', '?'))}")
    parts.append(f"alpha_{_format_param_value_for_filename(params.get('alpha_1', '?'))}")

    if not _is_pca(params):
        nn = params.get("umap_n_neighbors", 15)
        parts.append(f"nn_{nn}")
        md = params.get("umap_min_dist", 0.1)
        parts.append(f"md_{_format_param_value_for_filename(md)}")
        metric = params.get("umap_metric", "euclidean")
        parts.append(f"metric_{metric}")

    m_val = params.get("M", 1)
    parts.append(f"M_{m_val}")

    parts.append("PCA" if _is_pca(params) else "UMAP")

    return "_".join(parts)


def _format_param_value_for_filename(value) -> str:
    """Format a param value for safe use in a filename.

    Integer-valued floats (e.g. 1.0) are formatted without a decimal point
    so that gamma_to_dirname and _generate_group_label produce consistent
    directory and file names.  e.g. '1.0' -> '1', '0.5' -> '0p5'.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:g}".replace(".", "p")
    return str(value).replace(".", "p").replace("_", "")


def _format_header_params(params: dict) -> str:
    """
    Build a compact LaTeX header description from parameters.
    Excludes n (rows) and parameters that are not interesting to display.
    """
    parts = []

    rts_display = params.get("rts", "").upper()
    parts.append(rts_display)

    parts.append(r"\(N=" + str(params.get("N", "?")) + r"\)")

    parts.append(r"\(\gamma=" + _format_param_value("gamma", params["gamma"]) + r"\)")

    parts.append(
        r"\(\sigma_u=" + _format_param_value("sigma_u", params["sigma_u"]) + r"\)"
    )

    alpha_val = params.get("alpha_1", "?")
    parts.append(r"\(\alpha_1=" + _format_param_value("alpha_1", alpha_val) + r"\)")

    if not _is_pca(params):
        nn = params.get("umap_n_neighbors", 15)
        parts.append(r"\(k=" + str(nn) + r"\)")

        md = params.get("umap_min_dist", 0.1)
        parts.append(r"\(\text{min\_dist}=" + _format_param_value("min_dist", md) + r"\)")

        metric = params.get("umap_metric", "euclidean")
        parts.append(r"\(\text{metric}=" + metric + r"\)")

    m_val = params.get("M", 1)
    parts.append(r"\(M=" + str(m_val) + r"\)")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
#  Merged landscape table – Accuracy, Correlation & Discrimination
# ---------------------------------------------------------------------------

def _generate_table_merged(combined: pd.DataFrame, ref_params: dict) -> str:
    """
    Generate a single LaTeX sidewaystable (landscape) merging accuracy/correlation
    and discrimination metrics into one wide table.

    Columns (14 total):
      n | Method | d |
      MAE μ | MAE σ | Spearman ρ μ | Spearman ρ σ |
      Pearson r μ | Pearson r σ | Kendall τ μ | Kendall τ σ |
      Efficient DMUs avg | Efficient DMUs sd |
      % Non-discriminating
    """
    lines = []
    lines.append(r"\begin{sidewaystable}")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")

    header_params_str = _format_header_params(ref_params)
    rts_lower = ref_params.get("rts", "unknown").lower()
    pca_suffix = "_pca" if ref_params.get("pca", False) else "_umap"
    label_safe = f"tab:N_{ref_params['N']}_{rts_lower}{pca_suffix}"

    dea_type = "PCA-DEA" if ref_params.get("pca", False) else "UMAP-DEA"
    lines.append(
        r"\caption{Accuracy, correlation, and discrimination metrics " + dea_type + r".\\ "
        + header_params_str
        + r"\label{" + label_safe + r"}}"
    )

    # 14 columns: l l c + 11 c
    lines.append(
        r"\begin{tabular}{@{}l l c c c c c c c c c c c c@{}}"
    )
    lines.append(r"\toprule")

    # --- Three-level header with \cmidrule grouping ---

    # Row 1: Super-group labels
    lines.append(
        r"& & & "
        r"\multicolumn{8}{c}{Accuracy and correlation} & "
        r"\multicolumn{3}{c}{Discrimination} \\"
    )
    lines.append(r"\cmidrule(lr){4-11} \cmidrule(lr){12-14}")

    # Row 2: Individual metric names
    lines.append(
        r"\(n\) & Method & \(d\) & "
        r"\multicolumn{2}{c}{MAE} & "
        r"\multicolumn{2}{c}{Spearman \(\rho\)} & "
        r"\multicolumn{2}{c}{Pearson \(r\)} & "
        r"\multicolumn{2}{c}{Kendall \(\tau\)} & "
        r"\multicolumn{2}{c}{Efficient DMUs} & "
        r"\% Non-discrim. \\"
    )
    lines.append(
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} "
        r"\cmidrule(lr){10-11} \cmidrule(lr){12-13}"
    )

    # Row 3: μ/σ row
    lines.append(
        r"& & & "
        r"mean & std & "
        r"mean & std & "
        r"mean & std & "
        r"mean & std & "
        r"avg & sd & "
        r" \\"
    )
    lines.append(r"\midrule")

    nr_simulations = ref_params.get("nr_simulations", 1000)

    # Data rows
    prev_n = None
    for _, row in combined.iterrows():
        current_n = row["n"]
        if prev_n is not None and current_n != prev_n:
            lines.append(r"\addlinespace")

        n_str = str(int(current_n))
        method = _dim_reduction_label(row["dim_reduction_level"])
        # Cast to float first to handle both int and float-like columns
        # (some parquet sources may store dims as float).
        d_str = str(int(float(row["dims"])))

        # Accuracy / correlation metrics
        mae_mean = _format_number(row["mae_mean"])
        mae_std = _format_number(row["mae_std"])
        spr_mean = _format_number(row["spearmanr_mean"])
        spr_std = _format_number(row["spearmanr_std"])
        ppr_mean = _format_number(row["pearsonr_mean"])
        ppr_std = _format_number(row["pearsonr_std"])
        kt_mean = _format_number(row["kendalltau_mean"])
        kt_std = _format_number(row["kendalltau_std"])

        # Discrimination metrics
        nr_eff_mean = _format_number(row["nr_efficient_mean"])
        nr_eff_std = _format_number(row["nr_efficient_std"])
        warning_count = row["spearmanr_warning_count"]
        non_discrim_pct = (
            float(warning_count) / nr_simulations * 100
            if pd.notna(warning_count) and nr_simulations > 0
            else float("nan")
        )
        non_discrim_str = f"{non_discrim_pct:.1f}\\%"

        lines.append(
            f"{n_str} & {method} & {d_str} & "
            f"{mae_mean} & {mae_std} & "
            f"{spr_mean} & {spr_std} & "
            f"{ppr_mean} & {ppr_std} & "
            f"{kt_mean} & {kt_std} & "
            f"{nr_eff_mean} & {nr_eff_std} & "
            f"{non_discrim_str} \\\\"
        )
        prev_n = current_n

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{sidewaystable}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Standard (portrait) table – Means only, no standard deviations
# ---------------------------------------------------------------------------

def _generate_table_standard(combined: pd.DataFrame, ref_params: dict) -> str:
    """
    Generate a standard LaTeX table (portrait, not rotated) with means only.
    Uses \\scriptsize font and compact column spacing to fit the page width.

    Columns (9 total):
      n | Method | d |
      MAE | Spearman ρ | Pearson r | Kendall τ |
      Efficient DMUs | % Non-discrim.
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")

    header_params_str = _format_header_params(ref_params)
    rts_lower = ref_params.get("rts", "unknown").lower()
    pca_suffix = "_pca" if ref_params.get("pca", False) else "_umap"
    label_safe = f"tab:std_N_{ref_params['N']}_{rts_lower}{pca_suffix}"

    dea_type = "PCA-DEA" if ref_params.get("pca", False) else "UMAP-DEA"
    lines.append(
        r"\caption{Accuracy, correlation, and discrimination metrics (mean values) "
        + dea_type + r".\\ "
        + header_params_str
        + r"\label{" + label_safe + r"}}"
    )

    # 9 columns: l l c + 6 c
    lines.append(
        r"\begin{tabular}{@{}l l c c c c c c c@{}}"
    )
    lines.append(r"\toprule")

    # --- Two-level header with \cmidrule grouping ---

    # Row 1: Super-group labels
    lines.append(
        r"& & & "
        r"\multicolumn{4}{c}{Accuracy and correlation} & "
        r"\multicolumn{2}{c}{Discrimination} \\"
    )
    lines.append(r"\cmidrule(lr){4-7} \cmidrule(lr){8-9}")

    # Row 2: Individual metric names
    lines.append(
        r"\(n\) & Method & \(d\) & "
        r"MAE & "
        r"Spearman \(\rho\) & "
        r"Pearson \(r\) & "
        r"Kendall \(\tau\) & "
        r"Efficient DMUs & "
        r"\% Non-discrim. \\"
    )
    lines.append(r"\midrule")

    nr_simulations = ref_params.get("nr_simulations", 1000)

    # Data rows
    prev_n = None
    for _, row in combined.iterrows():
        current_n = row["n"]
        if prev_n is not None and current_n != prev_n:
            lines.append(r"\addlinespace")

        n_str = str(int(current_n))
        method = _dim_reduction_label(row["dim_reduction_level"])
        d_str = str(int(float(row["dims"])))

        # Accuracy / correlation metrics (means only, no std devs)
        mae_mean = _format_number(row["mae_mean"])
        spr_mean = _format_number(row["spearmanr_mean"])
        ppr_mean = _format_number(row["pearsonr_mean"])
        kt_mean = _format_number(row["kendalltau_mean"])

        # Discrimination metrics
        nr_eff_mean = _format_number(row["nr_efficient_mean"])
        warning_count = row["spearmanr_warning_count"]
        non_discrim_pct = (
            float(warning_count) / nr_simulations * 100
            if pd.notna(warning_count) and nr_simulations > 0
            else float("nan")
        )
        non_discrim_str = f"{non_discrim_pct:.1f}\\%"

        lines.append(
            f"{n_str} & {method} & {d_str} & "
            f"{mae_mean} & "
            f"{spr_mean} & "
            f"{ppr_mean} & "
            f"{kt_mean} & "
            f"{nr_eff_mean} & "
            f"{non_discrim_str} \\\\"
        )
        prev_n = current_n

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Combined generator
# ---------------------------------------------------------------------------

def generate_latex_tables(runs_in_group: list[dict], group_index: int) -> str:
    """
    Generate LaTeX tables (sidewaystable + standard) for a group of runs,
    merging accuracy, correlation, and discrimination metrics.

    Runs in the group share the same hyperparameters (except n).

    Produces two tables in the same output:
      1. A sidewaystable (landscape) with mean and standard deviation.
      2. A standard table (portrait) with mean values only (\\scriptsize).
    """
    if not runs_in_group:
        return ""

    ref_params = runs_in_group[0]["params"]

    # Build combined DataFrame
    all_rows = []
    for run in runs_in_group:
        n_val = run["params"]["n"]
        df = run["summary_df"].copy()
        df["n"] = n_val
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values(by=["n", "dims"], ascending=[True, True])

    # Header comment with traceability info
    serials = sorted(run["run_serial"] for run in runs_in_group)
    serials_comment = "% Generated from runs: " + ", ".join(serials)

    header_params_str = _format_header_params(ref_params)
    separator = (
        f"\n\n% {'=' * 60}\n"
        f"% {header_params_str}\n"
        f"% {serials_comment}\n"
        f"% {'=' * 60}\n\n"
    )

    table_sideways = _generate_table_merged(combined, ref_params)
    table_standard = _generate_table_standard(combined, ref_params)

    return separator + table_sideways + "\n\n\\newpage\n\n" + table_standard + "\n"


# ---------------------------------------------------------------------------
#  Grouping
# ---------------------------------------------------------------------------

def group_key_func(params: dict) -> tuple:
    """
    Group runs by all parameters EXCEPT n (which varies within a table).
    UMAP hyperparameters are only included for non-PCA runs.
    """
    group_params = {
        "N",
        "M",
        "alpha_1",
        "gamma",
        "sigma_u",
        "rts",
        "orientation",
        "pca",
    }
    if not _is_pca(params):
        group_params.update({"umap_n_neighbors", "umap_min_dist", "umap_metric"})

    return tuple(sorted((k, params.get(k)) for k in group_params))


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    runs = load_all_runs(RESULTS_DIR)
    print(f"Loaded {len(runs)} runs from {RESULTS_DIR}")

    if not runs:
        print("No runs found. Exiting.")
        return

    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        key = group_key_func(run["params"])
        if key not in groups:
            groups[key] = []
        groups[key].append(run)

    print(f"Found {len(groups)} distinct groups of hyperparameter configurations")

    sorted_groups = sorted(
        groups.items(),
        key=lambda item: item[1][0]["params"].get("N", 0),
    )

    for group_index, (key, group_runs) in enumerate(sorted_groups):
        ref_params = group_runs[0]["params"]
        label = _generate_group_label(ref_params)

        # Determine gamma subdirectory and method-specific subpath
        gamma_val = ref_params.get("gamma", "unknown")
        gamma_dir = f"gamma_{gamma_to_dirname(gamma_val)}"
        method_subpath = get_output_subpath_from_params(ref_params)
        output_dir = os.path.join(OUTPUT_BASE_DIR, gamma_dir, method_subpath)
        os.makedirs(output_dir, exist_ok=True)

        # Short hash of all run serials for traceability
        serials = sorted(run["run_serial"] for run in group_runs)
        serials_hash = hashlib.sha256(",".join(serials).encode()).hexdigest()[:8]

        output_path = os.path.join(
            output_dir, f"latex_table_group_{group_index}_{label}_h{serials_hash}.tex"
        )

        latex_content = generate_latex_tables(group_runs, group_index)

        with open(output_path, "w") as f:
            f.write(latex_content)

        print(
            f"  -> Group {group_index}: {len(group_runs)} runs "
            f"-> {output_path}"
        )

    print("Done!")


if __name__ == "__main__":
    main()