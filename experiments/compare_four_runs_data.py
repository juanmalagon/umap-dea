"""Module to load all_results.csv for the compare_four_runs notebook."""

import pandas as pd
import numpy as np
import os


def load_data():
    """Find and load all_results.csv. Returns the DataFrame."""
    candidates = [
        "all_results.csv",
        os.path.join("..", "all_results.csv"),
    ]
    csv_path = None
    for p in candidates:
        if os.path.exists(p):
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError(
            "all_results.csv not found. Place it next to this notebook or one level up."
        )

    df = pd.read_csv(csv_path)
    df["algorithm"] = df["pca"].map({True: "PCA-DEA", False: "UMAP-DEA"})

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