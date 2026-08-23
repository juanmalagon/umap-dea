#!/usr/bin/env python3
"""
find_best_params.py

Reads all_results.csv. For each combination of (rts, gamma, nr_simulations, N, n),
finds the best parameters (pca, dims, dim_reduction_level, umap_n_neighbors) for a given
metric. Outputs the results as a CSV file in the analysis/ folder.

Usage as a script:
    python scripts/find_best_params.py <metric_name> [--input all_results.csv] [--output best_params.csv]

Usage as a module:
    from scripts.find_best_params import find_best_params, METRICS_CONFIG
    import pandas as pd
    df = pd.read_csv("all_results.csv")
    result = find_best_params(df, "MAE")
    result.to_csv("analysis/best_params_mae.csv", index=False)

Metrics that are "better" when lower:
    - MAE

Metrics that are "better" when higher:
    - Spearman, Pearson, Kendall, Proportion Efficient, Number Efficient
"""

import argparse
import os
import sys
from typing import Optional

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "analysis")


# Metric configuration: (column_name, lower_is_better)
METRICS_CONFIG = {
    "MAE": ("mae_mean", True),
    "Spearman": ("spearmanr_mean", False),
    "Pearson": ("pearsonr_mean", False),
    "Kendall": ("kendalltau_mean", False),
    "Proportion Efficient": ("prop_efficient_mean", True),
    "Number Efficient": ("nr_efficient_mean", True),
}

# Columns to group by
GROUP_COLS = ["rts", "gamma", "nr_simulations", "N", "n"]

# Columns that define the "best" parameters to output
PARAM_COLS = ["pca", "dims", "dim_reduction_level", "umap_n_neighbors"]


def find_best_params(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    For each group defined by (rts, gamma, nr_simulations, N, n), find the row
    with the best value for `metric` and return the corresponding parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame read from all_results.csv (or equivalent).
    metric : str
        One of: "MAE", "Spearman", "Pearson", "Kendall",
        "Proportion Efficient", "Number Efficient".

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for each group key, the best parameters
        (pca, dim_reduction_level, umap_n_neighbors), and the best metric
        value (named best_<metric_column>).
    """
    if metric not in METRICS_CONFIG:
        valid = ", ".join(METRICS_CONFIG.keys())
        raise ValueError(f"Unknown metric: '{metric}'. Must be one of: {valid}")

    metric_col, lower_is_better = METRICS_CONFIG[metric]

    results = []

    grouped = df.groupby(GROUP_COLS, dropna=False)

    for group_keys, group_df in grouped:
        # Drop rows where the metric is NaN
        valid = group_df.dropna(subset=[metric_col])
        if valid.empty:
            continue

        if lower_is_better:
            best_idx = valid[metric_col].idxmin()
        else:
            best_idx = valid[metric_col].idxmax()

        best_row = valid.loc[best_idx]

        row_result: dict = {}

        # Add the group key values
        if isinstance(group_keys, tuple):
            for i, col in enumerate(GROUP_COLS):
                row_result[col] = group_keys[i]
        else:
            # Single group column (shouldn't happen with 5 cols, but safe)
            row_result[GROUP_COLS[0]] = group_keys

        # Add best parameters
        for col in PARAM_COLS:
            row_result[col] = best_row[col]

        # Add the best metric value
        row_result[f"best_{metric_col}"] = best_row[metric_col]

        results.append(row_result)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find best parameters (pca, dims, dim_reduction_level, umap_n_neighbors) "
            "per group for a given metric from all_results.csv."
        )
    )
    parser.add_argument(
        "metric",
        choices=list(METRICS_CONFIG.keys()),
        help="Metric to optimize for.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default="all_results.csv",
        help="Input CSV file path (default: all_results.csv).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output CSV file name (default: best_params_<metric>.csv). "
            "The file is written to the analysis/ folder."
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)

    result_df = find_best_params(df, args.metric)

    if args.output:
        output_path = args.output
    else:
        metric_slug = args.metric.lower().replace(" ", "_")
        output_path = f"best_params_{metric_slug}.csv"

    if not os.path.isabs(output_path):
        output_path = os.path.join(ANALYSIS_DIR, output_path)

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Results written to {output_path}")
    print(f"Found best parameters for {len(result_df)} groups.")


if __name__ == "__main__":
    main()
