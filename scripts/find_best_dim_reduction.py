#!/usr/bin/env python3
"""
find_best_dim_reduction.py

Reads summary_df and params_dict CSV files from the results/ folder.
For each experiment (matched by UUID), finds the best dim_reduction_level
with respect to each metric (mae, spearman, pearson, kendall).
Outputs a single CSV table in analysis/best_dim_reduction.csv with the best
level, dims, and value for each metric, along with all other parameters.
"""

import glob
import os
import re

import pandas as pd


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
ANALYSIS_DIR = os.path.join(_PROJECT_ROOT, "analysis")
OUTPUT_FILE = os.path.join(ANALYSIS_DIR, "best_dim_reduction.csv")

# Metrics to find the best for
# For mae: lower is better -> use idxmin
# For spearman, pearson, kendall: higher is better -> use idxmax
METRICS = {
    "mae": {"column": "mae_mean", "best": "min"},
    "spearman": {"column": "spearmanr_mean", "best": "max"},
    "pearson": {"column": "pearsonr_mean", "best": "max"},
    "kendall": {"column": "kendalltau_mean", "best": "max"},
}


def extract_uuid(filename: str) -> str:
    """Extract the UUID from a filename like summary_df_<uuid>.csv."""
    match = re.search(
        r"(?:summary_df|params_dict)_([0-9a-fA-F-]+)\.csv", filename
    )
    if match:
        return match.group(1)
    return ""


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


def find_overall_best_row(
    df: pd.DataFrame, metrics: dict
) -> tuple[pd.Series, float]:
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
    # Find all summary_df files
    summary_files = glob.glob(os.path.join(RESULTS_DIR, "summary_df_*.csv"))

    rows = []

    for summary_path in sorted(summary_files):
        uuid = extract_uuid(os.path.basename(summary_path))
        if not uuid:
            print(f"Warning: could not extract UUID from {summary_path}")
            continue

        params_path = os.path.join(RESULTS_DIR, f"params_dict_{uuid}.csv")
        if not os.path.exists(params_path):
            print(f"Warning: no params_dict found for UUID {uuid}, skipping")
            continue

        # Read files
        summary_df = pd.read_csv(summary_path)
        params_df = pd.read_csv(params_path)

        if params_df.empty:
            print(f"Warning: params_dict is empty for UUID {uuid}, skipping")
            continue

        params = params_df.iloc[0].to_dict()

        # Build the output row
        row = {"uuid": uuid}

        for metric_name, metric_info in METRICS.items():
            col = metric_info["column"]
            best = metric_info["best"]
            best_row = find_best_row(summary_df, col, best)

            row[f"best_{metric_name}_level"] = best_row.get(
                "dim_reduction_level", float("nan")
            )
            row[f"best_{metric_name}_dims"] = best_row.get(
                "dims", float("nan")
            )
            row[f"best_{metric_name}_value"] = best_row.get(
                col, float("nan")
            )

        # Find the overall best level (across all metrics, excluding "original")
        overall_row, overall_meanrank = find_overall_best_row(summary_df, METRICS)
        row["best_overall_level"] = overall_row.get(
            "dim_reduction_level", float("nan")
        )
        row["best_overall_dims"] = overall_row.get(
            "dims", float("nan")
        )
        row["best_overall_meanrank"] = overall_meanrank

        # Add all params_dict columns
        row.update(params)

        rows.append(row)

    # Build output DataFrame
    output_df = pd.DataFrame(rows)

    # Reorder columns: uuid, best_* columns, then params
    id_cols = ["uuid"]
    best_cols = (
        [
            f"best_{m}_{suffix}"
            for m in METRICS
            for suffix in ["level", "dims", "value"]
        ]
        + ["best_overall_level", "best_overall_dims", "best_overall_meanrank"]
    )
    # Get all unique param keys in order of first appearance
    param_cols = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in id_cols and k not in best_cols and k not in seen:
                param_cols.append(k)
                seen.add(k)

    output_df = output_df[id_cols + best_cols + param_cols]

    # Save
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Output written to {OUTPUT_FILE}")
    print(f"Processed {len(rows)} experiments.")


if __name__ == "__main__":
    main()