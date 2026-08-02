"""Module for generating the four-run comparison plot."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

# Colors
COLOR_A = "#1f77b4"
COLOR_B = "#ff7f0e"
COLOR_C = "#2ca02c"
COLOR_D = "#d62728"


def show_valid_options(label, df_sub):
    """Print valid umap_n_neighbors and nr_simulations for a given subset."""
    if len(df_sub) == 0:
        print(f"  {label} → No matching data for these base parameters!")
        return
    valid_k = sorted(df_sub["umap_n_neighbors"].unique())
    valid_nr = sorted(df_sub["nr_simulations"].unique())
    print(f"  Valid umap_n_neighbors for {label}: {valid_k}")
    print(f"  Valid nr_simulations    for {label}: {valid_nr}")


def _build_base_mask(df, algo, N, n, rts, gamma):
    """Build a boolean Series filtering df by base parameters."""
    return (
        (df["algorithm"] == algo)
        & (df["N"] == N)
        & (df["n"] == n)
        & (df["rts"] == rts)
        & (df["gamma"] == gamma)
    )


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


def _plot_bar_chart(ax, x, width, agg_a, agg_b, agg_c, agg_d, show_std,
                    run_labels, labels_ordered, metric_name):
    """Draw the grouped bar chart on the given axes."""
    def _bar(x_offset, agg, color, label, show_std):
        yerr = agg["metric_std"].values if show_std else None
        return ax.bar(
            x + x_offset,
            agg["metric_mean"].values,
            width,
            yerr=yerr,
            capsize=5,
            color=color,
            alpha=0.85,
            edgecolor="white",
            label=label,
        )

    bars_a = _bar(-1.5 * width, agg_a, COLOR_A, run_labels[0], show_std)
    bars_b = _bar(-0.5 * width, agg_b, COLOR_B, run_labels[1], show_std)
    bars_c = _bar(+0.5 * width, agg_c, COLOR_C, run_labels[2], show_std)
    bars_d = _bar(+1.5 * width, agg_d, COLOR_D, run_labels[3], show_std)

    # Annotate bars
    for bar, color in [(bars_a, COLOR_A), (bars_b, COLOR_B), (bars_c, COLOR_C), (bars_d, COLOR_D)]:
        for rect in bar:
            h = rect.get_height()
            if not np.isnan(h):
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    h + 0.002,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=color,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_ordered, fontsize=11)
    ax.set_xlabel("Dimension Reduction Level", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    title = (
        f"{run_labels[0]}  vs  {run_labels[1]}  vs  {run_labels[2]}  vs  {run_labels[3]}"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_diff_panel(ax, x, width, diff, col_a, col_opponent, label, labels_ordered):
    """Draw a single difference panel (A − opponent)."""
    colors = [col_a if d >= 0 else col_opponent for d in diff]
    ax.bar(x, diff, width * 1.8, color=colors, alpha=0.75, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.8)
    for i, d in enumerate(diff):
        if not np.isnan(d):
            va = "bottom" if d >= 0 else "top"
            offset = 0.002 if d >= 0 else -0.002
            ax.text(
                i, d + offset, f"{d:+.3f}",
                ha="center", va=va, fontsize=8, color="black", fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_ordered, fontsize=11)
    ax.set_xlabel("Dimension Reduction Level", fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=11, fontweight="bold", color="#555555")
    ax.grid(True, alpha=0.3, axis="y")


def plot_comparison(
    df,
    run_a_algo, run_a_N, run_a_n, run_a_rts, run_a_gamma, run_a_umap_n_neighbors, run_a_nr_simulations,
    run_b_algo, run_b_N, run_b_n, run_b_rts, run_b_gamma, run_b_umap_n_neighbors, run_b_nr_simulations,
    run_c_algo, run_c_N, run_c_n, run_c_rts, run_c_gamma, run_c_umap_n_neighbors, run_c_nr_simulations,
    run_d_algo, run_d_N, run_d_n, run_d_rts, run_d_gamma, run_d_umap_n_neighbors, run_d_nr_simulations,
    metric_name="Kendall",
    show_std=True,
):
    """Generate the four-run comparison plot with difference panels and summary table.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded all_results.csv DataFrame.
    run_*_algo : str
        Algorithm name ("PCA-DEA" or "UMAP-DEA").
    run_*_N, run_*_n : int
        Dimensions (N, n).
    run_*_rts : str
        Returns to scale ("vrs" or "crs").
    run_*_gamma : float
        Gamma value.
    run_*_umap_n_neighbors : int
        UMAP n_neighbors filter value.
    run_*_nr_simulations : int
        Number of simulations filter value.
    metric_name : str
        One of "MAE", "Spearman", "Pearson", "Kendall",
        "Proportion Efficient", "Number Efficient".
    show_std : bool
        Whether to show error bars (standard deviation).
    """
    mean_col, std_col = METRICS[metric_name]

    # Build base masks
    mask_a_base = _build_base_mask(df, run_a_algo, run_a_N, run_a_n, run_a_rts, run_a_gamma)
    mask_b_base = _build_base_mask(df, run_b_algo, run_b_N, run_b_n, run_b_rts, run_b_gamma)
    mask_c_base = _build_base_mask(df, run_c_algo, run_c_N, run_c_n, run_c_rts, run_c_gamma)
    mask_d_base = _build_base_mask(df, run_d_algo, run_d_N, run_d_n, run_d_rts, run_d_gamma)

    # Show valid sub-options
    print("─" * 50)
    show_valid_options(
        f"Run A ({run_a_algo}, N={run_a_N}, n={run_a_n}, {run_a_rts}, γ={run_a_gamma})",
        df[mask_a_base],
    )
    show_valid_options(
        f"Run B ({run_b_algo}, N={run_b_N}, n={run_b_n}, {run_b_rts}, γ={run_b_gamma})",
        df[mask_b_base],
    )
    show_valid_options(
        f"Run C ({run_c_algo}, N={run_c_N}, n={run_c_n}, {run_c_rts}, γ={run_c_gamma})",
        df[mask_c_base],
    )
    show_valid_options(
        f"Run D ({run_d_algo}, N={run_d_N}, n={run_d_n}, {run_d_rts}, γ={run_d_gamma})",
        df[mask_d_base],
    )
    print("─" * 50)

    # Build full masks (base + sub-params)
    mask_a = mask_a_base & (df["umap_n_neighbors"] == run_a_umap_n_neighbors) & (df["nr_simulations"] == run_a_nr_simulations)
    mask_b = mask_b_base & (df["umap_n_neighbors"] == run_b_umap_n_neighbors) & (df["nr_simulations"] == run_b_nr_simulations)
    mask_c = mask_c_base & (df["umap_n_neighbors"] == run_c_umap_n_neighbors) & (df["nr_simulations"] == run_c_nr_simulations)
    mask_d = mask_d_base & (df["umap_n_neighbors"] == run_d_umap_n_neighbors) & (df["nr_simulations"] == run_d_nr_simulations)

    # Validate
    invalid = False
    invalid |= _check_and_report(mask_a_base, mask_a, "Run A", run_a_umap_n_neighbors, run_a_nr_simulations, df)
    invalid |= _check_and_report(mask_b_base, mask_b, "Run B", run_b_umap_n_neighbors, run_b_nr_simulations, df)
    invalid |= _check_and_report(mask_c_base, mask_c, "Run C", run_c_umap_n_neighbors, run_c_nr_simulations, df)
    invalid |= _check_and_report(mask_d_base, mask_d, "Run D", run_d_umap_n_neighbors, run_d_nr_simulations, df)
    if invalid:
        raise SystemExit("One or more runs have invalid parameter combinations. See messages above.")

    # Filter
    df_a = df[mask_a].copy()
    df_b = df[mask_b].copy()
    df_c = df[mask_c].copy()
    df_d = df[mask_d].copy()

    # Align levels
    levels_present = (
        set(df_a["dim_reduction_level"].unique())
        & set(df_b["dim_reduction_level"].unique())
        & set(df_c["dim_reduction_level"].unique())
        & set(df_d["dim_reduction_level"].unique())
    )
    # Sort by actual dimensionality (dims) in descending order
    dims_map = df_a.groupby("dim_reduction_level")["dims"].first().to_dict()
    levels_ordered = sorted(levels_present, key=lambda l: dims_map.get(l, 0), reverse=True)
    # Build labels with actual dims from the data
    labels_ordered = []
    for l in levels_ordered:
        base_label = DIM_LABELS[DIM_ORDER.index(l)]
        actual_dims = dims_map.get(l, "?")
        labels_ordered.append(f"{base_label} ({actual_dims})")

    if len(levels_ordered) == 0:
        print("⚠️ No common dim_reduction_level values between the four runs.")
        raise SystemExit

    # Aggregate
    agg_a = _aggregate(df_a, levels_ordered, mean_col, std_col)
    agg_b = _aggregate(df_b, levels_ordered, mean_col, std_col)
    agg_c = _aggregate(df_c, levels_ordered, mean_col, std_col)
    agg_d = _aggregate(df_d, levels_ordered, mean_col, std_col)

    # Build run labels for the plot legend / title
    run_label_a = f"Run A: {run_a_algo}" + (f" (k={run_a_umap_n_neighbors})" if run_a_algo != "PCA-DEA" else "")
    run_label_b = f"Run B: {run_b_algo}" + (f" (k={run_b_umap_n_neighbors})" if run_b_algo != "PCA-DEA" else "")
    run_label_c = f"Run C: {run_c_algo}" + (f" (k={run_c_umap_n_neighbors})" if run_c_algo != "PCA-DEA" else "")
    run_label_d = f"Run D: {run_d_algo}" + (f" (k={run_d_umap_n_neighbors})" if run_d_algo != "PCA-DEA" else "")
    run_labels = [run_label_a, run_label_b, run_label_c, run_label_d]

    # Build short labels and output directory
    def _short_label(algo, umap_k):
        """Compact run label for filenames."""
        if algo == "PCA-DEA":
            return "PCA-DEA"
        return f"{algo}_k{umap_k}"

    short_a = _short_label(run_a_algo, run_a_umap_n_neighbors)
    short_b = _short_label(run_b_algo, run_b_umap_n_neighbors)
    short_c = _short_label(run_c_algo, run_c_umap_n_neighbors)
    short_d = _short_label(run_d_algo, run_d_umap_n_neighbors)

    # Include N/n in filename only if they differ between runs
    N_values = set([run_a_N, run_b_N, run_c_N, run_d_N])
    n_values = set([run_a_n, run_b_n, run_c_n, run_d_n])
    dim_tag_parts = []
    if len(N_values) > 1:
        dim_tag_parts.append(f"N_A{run_a_N}_B{run_b_N}_C{run_c_N}_D{run_d_N}")
    else:
        dim_tag_parts.append(f"N{run_a_N}")
    if len(n_values) > 1:
        dim_tag_parts.append(f"n_A{run_a_n}_B{run_b_n}_C{run_c_n}_D{run_d_n}")
    else:
        dim_tag_parts.append(f"n{run_a_n}")
    dim_tag = "_".join(dim_tag_parts)

    output_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(output_dir, exist_ok=True)

    def _save_and_show(fig, filename):
        """Save figure to plots/ and display it."""
        path = os.path.join(output_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved: {path}")
        plt.show()
        plt.close(fig)

    x = np.arange(len(levels_ordered))
    width = 0.20

    # --- 1. Bar chart ---
    fig_bar, ax_bar = plt.subplots(figsize=(12, 8))
    fig_bar.suptitle(f"N = {run_a_N}, n = {run_a_n}, {metric_name}", fontsize=12, fontweight="bold", y=0.98)
    _plot_bar_chart(ax_bar, x, width, agg_a, agg_b, agg_c, agg_d,
                    show_std, run_labels, labels_ordered, metric_name)
    bar_filename = (
        f"{metric_name}__bar__"
        f"A_{short_a}_B_{short_b}_C_{short_c}_D_{short_d}__"
        f"{dim_tag}.png"
    )
    _save_and_show(fig_bar, bar_filename)

    # --- 2. Difference A − B ---
    diff_ab = agg_a["metric_mean"].values - agg_b["metric_mean"].values
    fig_diff_ab, ax_diff_ab = plt.subplots(figsize=(12, 4))
    _plot_diff_panel(ax_diff_ab, x, width, diff_ab, COLOR_A, COLOR_B, "Δ (A − B)", labels_ordered)
    diff_ab_filename = (
        f"{metric_name}__diff_A-B__"
        f"A_{short_a}_B_{short_b}__"
        f"{dim_tag}.png"
    )
    _save_and_show(fig_diff_ab, diff_ab_filename)

    # --- 3. Difference A − C ---
    diff_ac = agg_a["metric_mean"].values - agg_c["metric_mean"].values
    fig_diff_ac, ax_diff_ac = plt.subplots(figsize=(12, 4))
    _plot_diff_panel(ax_diff_ac, x, width, diff_ac, COLOR_A, COLOR_C, "Δ (A − C)", labels_ordered)
    diff_ac_filename = (
        f"{metric_name}__diff_A-C__"
        f"A_{short_a}_C_{short_c}__"
        f"{dim_tag}.png"
    )
    _save_and_show(fig_diff_ac, diff_ac_filename)

    # --- 4. Difference A − D ---
    diff_ad = agg_a["metric_mean"].values - agg_d["metric_mean"].values
    fig_diff_ad, ax_diff_ad = plt.subplots(figsize=(12, 4))
    _plot_diff_panel(ax_diff_ad, x, width, diff_ad, COLOR_A, COLOR_D, "Δ (A − D)", labels_ordered)
    diff_ad_filename = (
        f"{metric_name}__diff_A-D__"
        f"A_{short_a}_D_{short_d}__"
        f"{dim_tag}.png"
    )
    _save_and_show(fig_diff_ad, diff_ad_filename)

    # Summary table
    table_data = {
        "Dim Reduction": labels_ordered,
        f"Run A ({metric_name})": [
            f"{v:.5f}" if not np.isnan(v) else "NaN" for v in agg_a["metric_mean"].values
        ],
        f"Run B ({metric_name})": [
            f"{v:.5f}" if not np.isnan(v) else "NaN" for v in agg_b["metric_mean"].values
        ],
        f"Run C ({metric_name})": [
            f"{v:.5f}" if not np.isnan(v) else "NaN" for v in agg_c["metric_mean"].values
        ],
        f"Run D ({metric_name})": [
            f"{v:.5f}" if not np.isnan(v) else "NaN" for v in agg_d["metric_mean"].values
        ],
        "Δ (A−B)": [f"{d:+.5f}" if not np.isnan(d) else "NaN" for d in diff_ab],
        "Δ (A−C)": [f"{d:+.5f}" if not np.isnan(d) else "NaN" for d in diff_ac],
        "Δ (A−D)": [f"{d:+.5f}" if not np.isnan(d) else "NaN" for d in diff_ad],
    }
    table_df = pd.DataFrame(table_data)
    caption = (
        f"Comparison Table — {metric_name}   "
        f"(Run A: ns={run_a_nr_simulations}, seeds={int(agg_a['n_seeds'].iloc[0])}   "
        f"Run B: ns={run_b_nr_simulations}, seeds={int(agg_b['n_seeds'].iloc[0])}   "
        f"Run C: ns={run_c_nr_simulations}, seeds={int(agg_c['n_seeds'].iloc[0])}   "
        f"Run D: ns={run_d_nr_simulations}, seeds={int(agg_d['n_seeds'].iloc[0])})"
    )
    print(f"\n{caption}")
    print("=" * len(caption))

    from IPython.display import display as ipy_display
    ipy_display(table_df)