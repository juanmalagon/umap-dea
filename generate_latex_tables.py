#!/usr/bin/env python3
"""
Generate LaTeX tables from simulation results.

Reads all params_dict_*.csv and summary_df_*.csv pairs from the results folder,
groups runs by common hyperparameters (excluding n, which becomes the row variable),
and outputs one .tex file per group.
"""

import os
import re
import glob
import pandas as pd
import numpy as np

RESULTS_DIR = "results"
OUTPUT_DIR = "results"


def load_all_runs(results_dir: str) -> list[dict]:
    """
    Scan results_dir for params_dict_*.csv and summary_df_*.csv pairs,
    match them by run_serial (UUID), and return a list of combined records.
    """
    # Collect all params_dict files
    params_files = glob.glob(os.path.join(results_dir, "params_dict_*.csv"))

    runs = []
    for params_path in params_files:
        # Extract run_serial from filename: params_dict_{run_serial}.csv
        basename = os.path.basename(params_path)
        match = re.match(r"params_dict_(.+)\.csv", basename)
        if not match:
            continue
        run_serial = match.group(1)

        # Load params
        params_df = pd.read_csv(params_path)
        if params_df.empty:
            print(f"Warning: empty params file {params_path}, skipping")
            continue
        params_dict = params_df.iloc[0].to_dict()

        # Load corresponding summary_df
        summary_path = os.path.join(
            results_dir, f"summary_df_{run_serial}.csv"
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
        # Use nice decimal representation
        return f"{value:g}"
    if isinstance(value, str):
        # Escape underscores for LaTeX
        return value.replace("_", r"\_")
    return str(value)


def _latex_escape_text(text: str) -> str:
    """Escape text for LaTeX."""
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_header_params(params: dict) -> str:
    """
    Build a compact LaTeX header description from parameters.
    Excludes n (rows) and parameters that are not interesting to display.
    Returns a string like:
        VRS, \\(N=200\\), \\(\\gamma=1.0\\), \\(\\sigma_u=0.1\\), ...
    """
    parts = []

    # RTS and orientation
    rts_display = params.get("rts", "").upper()
    parts.append(rts_display)

    # N
    parts.append(r"\(N=" + str(params["N"]) + r"\)")

    # gamma
    parts.append(r"\(\gamma=" + _format_param_value("gamma", params["gamma"]) + r"\)")

    # sigma_u
    parts.append(
        r"\(\sigma_u=" + _format_param_value("sigma_u", params["sigma_u"]) + r"\)"
    )

    # alpha_1
    alpha_val = params["alpha_1"]
    parts.append(r"\(\alpha_1=" + _format_param_value("alpha_1", alpha_val) + r"\)")

    # UMAP hyperparameters
    nn = params.get("umap_n_neighbors", 15)
    parts.append(r"\(n_{\text{neighbors}}=" + str(nn) + r"\)")

    md = params.get("umap_min_dist", 0.1)
    parts.append(r"\(\text{min\_dist}=" + _format_param_value("min_dist", md) + r"\)")

    metric = params.get("umap_metric", "euclidean")
    parts.append(r"\(\text{metric}=" + metric + r"\)")

    # sigma_u and others
    m_val = params.get("M", 1)
    parts.append(r"\(M=" + str(m_val) + r"\)")

    pca_val = params.get("pca", False)
    if pca_val:
        parts.append("PCA")

    return ", ".join(parts)


def _format_number(value) -> str:
    """Format a numeric value for LaTeX, handling NaN."""
    if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
        return "NaN"
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


def _generate_group_label(params: dict) -> str:
    """
    Generate a unique identifying label for a group of runs (for file naming).
    Encodes all hyperparameters that define the group, so different
    configurations produce distinct filenames.
    """
    parts = []

    # Core DEA params
    parts.append(f"N_{params.get('N', '?')}")

    # RTS and orientation
    rts = params.get("rts", "")
    orient = params.get("orientation", "")
    parts.append(f"{rts}_{orient}")

    # DGP params
    parts.append(f"gamma_{_format_param_value_for_filename(params.get('gamma', '?'))}")
    parts.append(f"sigma_{_format_param_value_for_filename(params.get('sigma_u', '?'))}")
    parts.append(f"alpha_{_format_param_value_for_filename(params.get('alpha_1', '?'))}")

    # UMAP params
    nn = params.get("umap_n_neighbors", 15)
    parts.append(f"nn_{nn}")
    md = params.get("umap_min_dist", 0.1)
    parts.append(f"md_{_format_param_value_for_filename(md)}")
    metric = params.get("umap_metric", "euclidean")
    parts.append(f"metric_{metric}")

    # Other params
    m_val = params.get("M", 1)
    parts.append(f"M_{m_val}")

    pca_val = params.get("pca", False)
    if pca_val:
        parts.append("PCA")

    return "_".join(parts)


def _format_param_value_for_filename(value) -> str:
    """Format a param value for safe use in a filename."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        # Use clean decimal, replace dot with 'p' for filename safety
        return f"{value:g}".replace(".", "p")
    return str(value).replace(".", "p").replace("_", "")


def generate_latex_table(
    runs_in_group: list[dict], group_index: int
) -> str:
    """
    Generate a complete LaTeX table* environment for a group of runs.
    runs_in_group: list of run dicts that share the same hyperparameters (except n).
    """
    if not runs_in_group:
        return ""

    # All runs in the group share the same params (except n)
    # Use the first run's params for the header
    ref_params = runs_in_group[0]["params"]

    # Build a combined DataFrame with n from params
    all_rows = []
    for run in runs_in_group:
        n_val = run["params"]["n"]
        df = run["summary_df"].copy()
        df["n"] = n_val
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)

    # Sort by n ascending, then by dims ascending
    combined = combined.sort_values(by=["n", "dims"], ascending=[True, True])

    # Build LaTeX
    lines = []

    # Header comment with identifying info
    header_params_str = _format_header_params(ref_params)
    lines.append(f"% ==================== {header_params_str} ====================")

    # Begin table*
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\tiny")

    # Caption / label
    # Create a safe label from the group identifier
    label_safe = f"tab:group_{group_index}"
    lines.append(
        "{"
        + header_params_str
        + r"\label{" + label_safe + r"}}"
    )

    # Begin tabular*
    lines.append(
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l c c c c c c c c c c@{}}"
    )
    lines.append(r"\toprule")

    # Column headers
    lines.append(
        r"\(n\) & Method & \(d\) & "
        r"\shortstack{MAE\\mean} & \shortstack{MAE\\std} & "
        r"\shortstack{Spearman\\\(r\) mean} & \shortstack{Spearman\\\(r\) std} & "
        r"\shortstack{Pearson\\\(r\) mean} & \shortstack{Pearson\\\(r\) std} & "
        r"\shortstack{Kendall\\\(\tau\) mean} & \shortstack{Kendall\\\(\tau\) std} \\"
    )
    lines.append(r"\midrule")

    # Rows
    prev_n = None
    for _, row in combined.iterrows():
        current_n = row["n"]

        # Add \addlinespace when n changes (except before the first group)
        if prev_n is not None and current_n != prev_n:
            lines.append(r"\addlinespace")

        n_str = str(int(current_n))
        method = _dim_reduction_label(row["dim_reduction_level"])
        d_str = str(int(row["dims"]))

        mae_mean = _format_number(row["mae_mean"])
        mae_std = _format_number(row["mae_std"])
        spr_mean = _format_number(row["spearmanr_mean"])
        spr_std = _format_number(row["spearmanr_std"])
        ppr_mean = _format_number(row["pearsonr_mean"])
        ppr_std = _format_number(row["pearsonr_std"])
        kt_mean = _format_number(row["kendalltau_mean"])
        kt_std = _format_number(row["kendalltau_std"])

        lines.append(
            f"{n_str} & {method} & {d_str} & "
            f"{mae_mean} & {mae_std} & "
            f"{spr_mean} & {spr_std} & "
            f"{ppr_mean} & {ppr_std} & "
            f"{kt_mean} & {kt_std} \\\\"
        )

        prev_n = current_n

    # End table
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"


def group_key_func(params: dict) -> tuple:
    """
    Define the grouping key for runs.
    We group by all parameters EXCEPT n (which becomes the row variable).
    Returns a tuple of (param_name, value) pairs for all non-n params.
    
    The idea: two runs belong to the same table if they share all hyperparameters
    except for n.
    """
    # Parameters that define the group (everything except n)
    group_params = {
        "N",
        "M",
        "alpha_1",
        "gamma",
        "sigma_u",
        "rts",
        "orientation",
        "pca",
        "umap_n_neighbors",
        "umap_min_dist",
        "umap_metric",
    }
    return tuple(sorted((k, params.get(k)) for k in group_params))


def main():
    # Load all runs
    runs = load_all_runs(RESULTS_DIR)
    print(f"Loaded {len(runs)} runs from {RESULTS_DIR}")

    if not runs:
        print("No runs found. Exiting.")
        return

    # Group runs by shared hyperparameters (excluding n)
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        key = group_key_func(run["params"])
        if key not in groups:
            groups[key] = []
        groups[key].append(run)

    print(f"Found {len(groups)} distinct groups of hyperparameter configurations")

    # Generate one .tex file per group
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sort groups by N for consistent output order
    sorted_groups = sorted(
        groups.items(),
        key=lambda item: item[1][0]["params"].get("N", 0),
    )

    for group_index, (key, group_runs) in enumerate(sorted_groups):
        # Create a descriptive filename
        ref_params = group_runs[0]["params"]
        label = _generate_group_label(ref_params)
        output_path = os.path.join(
            OUTPUT_DIR, f"latex_table_group_{group_index}_{label}.tex"
        )

        latex_content = generate_latex_table(group_runs, group_index)

        with open(output_path, "w") as f:
            f.write(latex_content)

        print(
            f"  -> Group {group_index}: {len(group_runs)} runs "
            f"-> {output_path}"
        )

    print("Done!")


if __name__ == "__main__":
    main()
