#!/usr/bin/env python3
"""
find_best_dim_reduction.py

Reads all_results.csv from the project root.
For each experiment (identified by unique combinations of parameter values),
finds the best dim_reduction_level with respect to each metric
(mae, spearman, pearson, kendall).
Outputs a single CSV table in analysis/best_dim_reduction.csv with the best
level, dims, and value for each metric, along with all other parameters.
"""

import os

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
ALL_RESULTS_FILE = os.path.join(_PROJECT_ROOT, "all_results.csv")
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "analysis")
OUTPUT_FILE = os.path.join(ANALYSIS_DIR, "best_dim_reduction.csv")

# Columns that identify an experiment (grouping key).
# These are the parameters that are fixed within a single experiment.
PARAM_COLS = [
    "seed",
    "alpha_1",
    "sigma_u",
    "gamma",
    "M",
    "rts",
    "orientation",
    "nr_simulations",
    "umap_min_dist",
    "umap_metric",
    "pca",
    "N",
    "n",
    "umap_n_neighbors",
]

# Metrics to find the best for
# For mae: lower is better -> use idxmin
# For spearman, pearson, kendall: higher is better -> use idxmax
METRICS = {
    "mae": {"column": "mae_mean", "best": "min"},
    "spearman": {"column": "spearmanr_mean", "best": "max"},
    "pearson": {"column": "pearsonr_mean", "best": "max"},
    "kendall": {"column": "kendalltau_mean", "best": "max"},
}


def find_best_row(df: pd.DataFrame, metric_col: str, best: str) -> pd.Series:
    """Return the row that is best for the given metric column."""
    # Drop rows where the metric is NaN for correlation-based metrics
    valid = df.dropna(subset=[metric_col])
    if valid.empty:
        # Return a row of NaN if no valid data
        result = df.iloc[0].copy()
        result[:] = float("nan")
        return result
    if best == "min":
        idx = valid[metric_col].idxmin()
    else:
        idx = valid[metric_col].idxmax()
    return valid.loc[idx]


def find_overall_best_row(df: pd.DataFrame, metrics: dict) -> tuple[pd.Series, float]:
    """
    Find the row with the best overall dim_reduction_level across all metrics.

    Uses average rank across all metrics to determine the best overall level.
    Excludes 'original' from consideration.

    Returns
    -------
    best_row : pd.Series
        The row corresponding to the best overall level.
    mean_rank : float
        The mean rank of that level (lower is better).
    """
    # Filter out "original"
    valid = df[df["dim_reduction_level"] != "original"].copy()
    if valid.empty:
        result = df.iloc[0].copy()
        result[:] = float("nan")
        return result, float("nan")

    # Build a DataFrame of ranks for each metric
    rank_df = pd.DataFrame(index=valid.index)

    for metric_name, metric_info in metrics.items():
        col = metric_info["column"]
        sub = valid[[col]].dropna()
        if sub.empty:
            continue
        if metric_info["best"] == "min":
            rank_df[metric_name] = sub[col].rank(method="average", ascending=True)
        else:
            rank_df[metric_name] = sub[col].rank(method="average", ascending=False)

    if rank_df.empty:
        result = df.iloc[0].copy()
        result[:] = float("nan")
        return result, float("nan")

    # Mean rank across all metrics
    rank_df["mean_rank"] = rank_df.mean(axis=1)

    # Pick the level with the lowest mean rank
    best_idx = rank_df["mean_rank"].idxmin()
    return valid.loc[best_idx], rank_df.loc[best_idx, "mean_rank"]


def main():
    # Read the single all_results.csv file
    all_df = pd.read_csv(ALL_RESULTS_FILE)

    # Verify all PARAM_COLS exist in the CSV
    missing_params = [c for c in PARAM_COLS if c not in all_df.columns]
    if missing_params:
        raise KeyError(f"Columns {missing_params} not found in {ALL_RESULTS_FILE}")

    # Group by the parameter columns
    groups = all_df.groupby(PARAM_COLS, sort=False, dropna=False)

    rows = []
    exp_id = 0

    for group_key, group_df in groups:
        exp_id += 1

        # Build the output row
        row = {"experiment_id": f"exp_{exp_id:04d}"}

        for metric_name, metric_info in METRICS.items():
            col = metric_info["column"]
            best = metric_info["best"]
            best_row = find_best_row(group_df, col, best)

            row[f"best_{metric_name}_level"] = best_row.get("dim_reduction_level", float("nan"))
            row[f"best_{metric_name}_dims"] = best_row.get("dims", float("nan"))
            row[f"best_{metric_name}_value"] = best_row.get(col, float("nan"))

        # Find the overall best level (across all metrics, excluding "original")
        overall_row, overall_meanrank = find_overall_best_row(group_df, METRICS)
        row["best_overall_level"] = overall_row.get("dim_reduction_level", float("nan"))
        row["best_overall_dims"] = overall_row.get("dims", float("nan"))
        row["best_overall_meanrank"] = overall_meanrank

        # Add all parameter values from the group key
        for i, param_name in enumerate(PARAM_COLS):
            row[param_name] = group_key[i]

        rows.append(row)

    # Build output DataFrame
    output_df = pd.DataFrame(rows)

    # Reorder columns: experiment_id, best_* columns, then params
    id_cols = ["experiment_id"]
    best_cols = [f"best_{m}_{suffix}" for m in METRICS for suffix in ["level", "dims", "value"]] + [
        "best_overall_level",
        "best_overall_dims",
        "best_overall_meanrank",
    ]

    output_df = output_df[id_cols + best_cols + PARAM_COLS]

    # Save
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Output written to {OUTPUT_FILE}")
    print(f"Processed {len(rows)} experiments.")


if __name__ == "__main__":
    main()
