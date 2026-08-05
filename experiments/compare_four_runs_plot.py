import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

# Metric mapping
METRICS = {
    "MAE": ("mae_mean", "mae_std"),
    "Spearman": ("spearmanr_mean", "spearmanr_std"),
    "Pearson": ("pearsonr_mean", "pearsonr_std"),
    "Kendall": ("kendalltau_mean", "kendalltau_std"),
    "Proportion Efficient": ("prop_efficient_mean", "prop_efficient_std"),
    "Number Efficient": ("nr_efficient_mean", "nr_efficient_std"),
}

# Dimension order
DIM_ORDER = ["original", "half", "ten_percent", "sqrt", "log"]
DIM_LABELS = ["N (original)", "N/2", "10%N", "√N", "ln(N)"]

# Colors for the four runs
RUN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
RUN_LETTERS = ["A", "B", "C", "D"]


def show_valid_options(label, df_sub):
    """Print valid umap_n_neighbors and nr_simulations for a given subset."""
    if len(df_sub) == 0:
        print(f"  {label} → No matching data for these base parameters!")
        return
    valid_k = sorted(df_sub["umap_n_neighbors"].unique())
    valid_nr = sorted(df_sub["nr_simulations"].unique())
    print(f"  Valid umap_n_neighbors for {label}: {valid_k}")
    print(f"  Valid nr_simulations    for {label}: {valid_nr}")


def _check_and_report(mask_base, mask_full, run_label, umap_k, nr_sim, df):
    """Check if a run's parameters are valid; return True if invalid."""
    if mask_full.sum() == 0:
        valid_k = sorted(df[mask_base]["umap_n_neighbors"].unique())
        valid_nr = sorted(df[mask_base]["nr_simulations"].unique())
        print(
            f"❌ No data for {run_label} with umap_n_neighbors={umap_k}, "
            f"nr_simulations={nr_sim}"
        )
        print(f"   Valid umap_n_neighbors: {valid_k}")
        print(f"   Valid nr_simulations:    {valid_nr}")
        return True
    return False


def _aggregate(df_sub, levels, mean_col, std_col):
    """Group by dim_reduction_level, reindex to given levels, and return aggregated DataFrame."""
    agg = (
        df_sub.groupby("dim_reduction_level")
        .agg(
            metric_mean=(mean_col, "mean"),
            metric_std=(std_col, "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reindex(levels)
        .reset_index()
    )
    return agg


def _plot_bar_chart(
    ax, x, width, aggs, show_std, run_labels, labels_ordered, metric_name
):
    """Draw the grouped bar chart."""

    clean_labels = [re.sub(r"^Run [A-D]: ", "", lbl) for lbl in run_labels]

    offsets = [-1.5, -0.5, +0.5, +1.5]
    all_bars = []

    for i, (agg, label) in enumerate(zip(aggs, clean_labels)):
        yerr = agg["metric_std"].values if show_std else None
        bars = ax.bar(
            x + offsets[i] * width,
            agg["metric_mean"].values,
            width,
            yerr=yerr,
            capsize=6,
            color=RUN_COLORS[i],
            alpha=0.85,
            edgecolor="white",
            label=label,
        )
        all_bars.append((bars, RUN_COLORS[i]))

    # Large, bold numbers above the bars
    for bars, color in all_bars:
        for rect in bars:
            h = rect.get_height()
            if not np.isnan(h):
                if h >= 0:
                    offset = (0, 6)
                    va = "bottom"
                else:
                    offset = (0, -6)
                    va = "top"

                ax.annotate(
                    f"{h:.3f}",
                    (rect.get_x() + rect.get_width() / 2.0, h),
                    textcoords="offset points",
                    xytext=offset,
                    ha="center",
                    va=va,
                    fontsize=16,
                    fontweight="bold",
                    color=color,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_ordered, fontsize=22)
    ax.set_xlabel("Dimension Reduction Level", fontsize=22)
    ax.tick_params(axis="y", labelsize=22)
    ax.set_ylabel(metric_name, fontsize=22)
    title = (
        f"{clean_labels[0]}  vs  {clean_labels[1]}  vs  {clean_labels[2]}  vs  {clean_labels[3]}"
    )
    ax.set_title(title, fontsize=24, fontweight="bold")
    ax.legend(fontsize=22, loc="best")
    ax.grid(True, alpha=0.3, axis="y")


def plot_comparison(df, runs, metric_names=None, show_std=True):
    """Generate the four‑run comparison bar chart (one per metric).

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe.
    runs : list of dict
        List of 4 dicts, one per run, each with keys:
        algo, N, n, rts, gamma, umap_n_neighbors, nr_simulations.
    metric_names : str or list of str, optional
        Metric name(s) from METRICS keys. Defaults to ["Kendall"].
    show_std : bool
        Whether to show error bars (std).
    """
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    elif metric_names is None:
        metric_names = ["Kendall"]

    if len(runs) != 4:
        raise ValueError(f"Expected 4 runs, got {len(runs)}")

    for single_metric in metric_names:
        _plot_single_metric(df, runs, single_metric, show_std)


def _plot_single_metric(df, runs, metric_name, show_std):
    """Generate the bar chart for a single metric."""
    mean_col, std_col = METRICS[metric_name]

    # --- Temporary boost of all default font sizes ---
    original_rc = plt.rcParams.copy()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "figure.titlesize": 22,
        }
    )

    try:
        # --- Build and validate masks for all runs ---
        masks_base = []
        masks_full = []
        run_labels = []

        for i, run in enumerate(runs):
            algo = run["algo"]
            N = run["N"]
            n = run["n"]
            rts = run["rts"]
            gamma = run["gamma"]
            umap_k = run["umap_n_neighbors"]
            nr_sim = run["nr_simulations"]

            letter = RUN_LETTERS[i]
            label_desc = f"Run {letter} ({algo}, N={N}, n={n}, {rts}, γ={gamma})"
            if algo != "PCA-DEA":
                run_label = f"Run {letter}: {algo} (k={umap_k})"
            else:
                run_label = f"Run {letter}: {algo}"
            run_labels.append(run_label)

            mask_base = (
                (df["algorithm"] == algo)
                & (df["N"] == N)
                & (df["n"] == n)
                & (df["rts"] == rts)
                & (df["gamma"] == gamma)
            )
            masks_base.append(mask_base)

            mask_full = (
                mask_base
                & (df["umap_n_neighbors"] == umap_k)
                & (df["nr_simulations"] == nr_sim)
            )
            masks_full.append(mask_full)

            show_valid_options(label_desc, df[mask_base])

        print("─" * 50)

        # --- Validate all runs ---
        invalid = False
        for i, run in enumerate(runs):
            letter = RUN_LETTERS[i]
            invalid |= _check_and_report(
                masks_base[i],
                masks_full[i],
                f"Run {letter}",
                run["umap_n_neighbors"],
                run["nr_simulations"],
                df,
            )
        if invalid:
            raise SystemExit("One or more runs have invalid parameter combinations.")

        # --- Filter dataframes for each run ---
        dfs = [df[mask].copy() for mask in masks_full]

        # --- Align levels across runs ---
        levels_present = set.intersection(
            *[set(d["dim_reduction_level"].unique()) for d in dfs]
        )
        dims_map = dfs[0].groupby("dim_reduction_level")["dims"].first().to_dict()
        levels_ordered = sorted(
            levels_present, key=lambda l: dims_map.get(l, 0), reverse=True
        )
        labels_ordered = []
        for l in levels_ordered:
            base_label = DIM_LABELS[DIM_ORDER.index(l)]
            actual_dims = dims_map.get(l, "?")
            labels_ordered.append(f"{base_label} ({actual_dims})")

        # Merge ten_percent and sqrt if they have the same dims value
        if "ten_percent" in levels_ordered and "sqrt" in levels_ordered:
            idx_tp = levels_ordered.index("ten_percent")
            idx_sq = levels_ordered.index("sqrt")
            if dims_map["ten_percent"] == dims_map["sqrt"]:
                labels_ordered[idx_tp] = f"10%N and √N ({dims_map['ten_percent']})"
                labels_ordered.pop(idx_sq)
                levels_ordered.remove("sqrt")

        if len(levels_ordered) == 0:
            print("⚠️ No common dim_reduction_level values between the four runs.")
            raise SystemExit

        # --- Aggregate each run ---
        aggs = [_aggregate(d, levels_ordered, mean_col, std_col) for d in dfs]

        # --- Build short labels for filename ---
        def _short_label(run):
            if run["algo"] == "PCA-DEA":
                return "PCA-DEA"
            return f"{run['algo']}_k{run['umap_n_neighbors']}"

        shorts = [_short_label(run) for run in runs]

        N_values = {run["N"] for run in runs}
        n_values = {run["n"] for run in runs}
        dim_tag_parts = []
        if len(N_values) > 1:
            dim_tag_parts.append(
                f"N_A{runs[0]['N']}_B{runs[1]['N']}_C{runs[2]['N']}_D{runs[3]['N']}"
            )
        else:
            dim_tag_parts.append(f"N{runs[0]['N']}")
        if len(n_values) > 1:
            dim_tag_parts.append(
                f"n_A{runs[0]['n']}_B{runs[1]['n']}_C{runs[2]['n']}_D{runs[3]['n']}"
            )
        else:
            dim_tag_parts.append(f"n{runs[0]['n']}")
        dim_tag = "_".join(dim_tag_parts)

        output_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(output_dir, exist_ok=True)

        # --- Create the bar chart ---
        x = np.arange(len(levels_ordered))
        width = 0.20
        fig_bar, ax_bar = plt.subplots(figsize=(18, 12))
        fig_bar.suptitle(
            f"N = {runs[0]['N']}, n = {runs[0]['n']}, {metric_name}",
            fontsize=22,
            fontweight="bold",
            y=0.98,
        )

        _plot_bar_chart(
            ax_bar,
            x,
            width,
            aggs,
            show_std,
            run_labels,
            labels_ordered,
            metric_name,
        )

        bar_filename = (
            f"{metric_name}__bar__"
            f"A_{shorts[0]}_B_{shorts[1]}_C_{shorts[2]}_D_{shorts[3]}__"
            f"{dim_tag}.png"
        )
        path = os.path.join(output_dir, bar_filename)
        fig_bar.tight_layout()
        fig_bar.savefig(path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved: {path}")
        plt.show()
        plt.close(fig_bar)

    finally:
        plt.rcParams.update(original_rc)