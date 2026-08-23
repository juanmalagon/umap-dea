"""Module to load all_results.csv for the compare_four_runs notebook."""

import pandas as pd
import numpy as np
import os


def add_algorithm_column(df):
    """Add the human-readable algorithm label from the PCA flag."""
    pca = df["pca"]
    if pca.dtype == object:
        pca = pca.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    df = df.copy()
    df["algorithm"] = pca.map({True: "PCA-DEA", False: "UMAP-DEA"})
    return df


def load_data(path=None):
    """Find and load all_results.csv. Returns the DataFrame."""
    if path is None:
        candidates = [
            "all_results.csv",
            os.path.join("..", "all_results.csv"),
        ]
        csv_path = None
        for p in candidates:
            if os.path.exists(p):
                csv_path = p
                break
    else:
        csv_path = path if os.path.exists(path) else None

    if csv_path is None:
        raise FileNotFoundError(
            "all_results.csv not found. Place it next to this notebook or one level up."
        )

    df = pd.read_csv(csv_path)
    df = add_algorithm_column(df)

    print(f"Loaded {len(df)} rows from {csv_path}")
    print("Available parameter values:")
    print(f"  Algorithm: {sorted(df['algorithm'].unique())}")
    print(f"  N:         {sorted(df['N'].unique())}")
    print(f"  n:         {sorted(df['n'].unique())}")
    print(f"  RTS:       {sorted(df['rts'].unique())}")
    print(f"  Gamma:     {sorted(df['gamma'].unique())}")
    print(f"  Metrics:   MAE, Spearman, Pearson, Kendall, Proportion Efficient, Number Efficient")
    print(f"  umap_n_neighbors & nr_simulations depend on your (algo, N, n) choices —")
    print(f"  run the Plot cell to see valid options for your parameters")

    return df
