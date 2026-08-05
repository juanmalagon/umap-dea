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


def _plot_bar_chart(
    ax, x, width, agg_a, agg_b, agg_c, agg_d, show_std, run_labels, labels_ordered, metric_name
):
    """Draw the grouped bar chart (readable, large fonts, no run‑prefix in legend)."""

    # Strip "Run X: " from legend/title labels
    clean_labels = [re.sub(r"^Run [A-D]: ", "", lbl) for lbl in run_labels]

    def _bar(x_offset, agg, color, label, show_std):
        yerr = agg["metric_std"].values if show_std else None
        return ax.bar(
            x + x_offset,
            agg["metric_mean"].values,
            width,
            yerr=yerr,
            capsize=6,  # slightly larger caps for error bars
            color=color,
            alpha=0.85,
            edgecolor="white",
            label=label,  # cleaned label for the legend
        )

    bars_a = _bar(-1.5 * width, agg_a, COLOR_A, clean_labels[0], show_std)
    bars_b = _bar(-0.5 * width, agg_b, COLOR_B, clean_labels[1], show_std)
    bars_c = _bar(+0.5 * width, agg_c, COLOR_C, clean_labels[2], show_std)
    bars_d = _bar(+1.5 * width, agg_d, COLOR_D, clean_labels[3], show_std)

    # Large, bold numbers above the bars (offset in points for consistent spacing)
    for bar, color in [(bars_a, COLOR_A), (bars_b, COLOR_B), (bars_c, COLOR_C), (bars_d, COLOR_D)]:
        for rect in bar:
            h = rect.get_height()
            if not np.isnan(h):
                # Choose offset and vertical alignment based on sign
                if h >= 0:
                    offset = (0, 6)  # 6 points above the bar
                    va = "bottom"
                else:
                    offset = (0, -6)  # 6 points below the bar
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


def plot_comparison(
    df,
    run_a_algo,
    run_a_N,
    run_a_n,
    run_a_rts,
    run_a_gamma,
    run_a_umap_n_neighbors,
    run_a_nr_simulations,
    run_b_algo,
    run_b_N,
    run_b_n,
    run_b_rts,
    run_b_gamma,
    run_b_umap_n_neighbors,
    run_b_nr_simulations,
    run_c_algo,
    run_c_N,
    run_c_n,
    run_c_rts,
    run_c_gamma,
    run_c_umap_n_neighbors,
    run_c_nr_simulations,
    run_d_algo,
    run_d_N,
    run_d_n,
    run_d_rts,
    run_d_gamma,
    run_d_umap_n_neighbors,
    run_d_nr_simulations,
    metric_name="Kendall",
    show_std=True,
):
    """Generate only the four‑run comparison bar chart (one per metric)."""
    if isinstance(metric_name, str):
        metric_names = [metric_name]
    else:
        metric_names = list(metric_name)

    for single_metric in metric_names:
        _plot_single_metric(
            df,
            run_a_algo,
            run_a_N,
            run_a_n,
            run_a_rts,
            run_a_gamma,
            run_a_umap_n_neighbors,
            run_a_nr_simulations,
            run_b_algo,
            run_b_N,
            run_b_n,
            run_b_rts,
            run_b_gamma,
            run_b_umap_n_neighbors,
            run_b_nr_simulations,
            run_c_algo,
            run_c_N,
            run_c_n,
            run_c_rts,
            run_c_gamma,
            run_c_umap_n_neighbors,
            run_c_nr_simulations,
            run_d_algo,
            run_d_N,
            run_d_n,
            run_d_rts,
            run_d_gamma,
            run_d_umap_n_neighbors,
            run_d_nr_simulations,
            metric_name=single_metric,
            show_std=show_std,
        )


def _plot_single_metric(
    df,
    run_a_algo,
    run_a_N,
    run_a_n,
    run_a_rts,
    run_a_gamma,
    run_a_umap_n_neighbors,
    run_a_nr_simulations,
    run_b_algo,
    run_b_N,
    run_b_n,
    run_b_rts,
    run_b_gamma,
    run_b_umap_n_neighbors,
    run_b_nr_simulations,
    run_c_algo,
    run_c_N,
    run_c_n,
    run_c_rts,
    run_c_gamma,
    run_c_umap_n_neighbors,
    run_c_nr_simulations,
    run_d_algo,
    run_d_N,
    run_d_n,
    run_d_rts,
    run_d_gamma,
    run_d_umap_n_neighbors,
    run_d_nr_simulations,
    metric_name="Kendall",
    show_std=True,
):
    """Generate the bar chart for a single metric (no diff panels, no table)."""
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
        # Build base masks
        mask_a_base = _build_base_mask(df, run_a_algo, run_a_N, run_a_n, run_a_rts, run_a_gamma)
        mask_b_base = _build_base_mask(df, run_b_algo, run_b_N, run_b_n, run_b_rts, run_b_gamma)
        mask_c_base = _build_base_mask(df, run_c_algo, run_c_N, run_c_n, run_c_rts, run_c_gamma)
        mask_d_base = _build_base_mask(df, run_d_algo, run_d_N, run_d_n, run_d_rts, run_d_gamma)

        # Show valid sub‑options
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

        # Build full masks
        mask_a = (
            mask_a_base
            & (df["umap_n_neighbors"] == run_a_umap_n_neighbors)
            & (df["nr_simulations"] == run_a_nr_simulations)
        )
        mask_b = (
            mask_b_base
            & (df["umap_n_neighbors"] == run_b_umap_n_neighbors)
            & (df["nr_simulations"] == run_b_nr_simulations)
        )
        mask_c = (
            mask_c_base
            & (df["umap_n_neighbors"] == run_c_umap_n_neighbors)
            & (df["nr_simulations"] == run_c_nr_simulations)
        )
        mask_d = (
            mask_d_base
            & (df["umap_n_neighbors"] == run_d_umap_n_neighbors)
            & (df["nr_simulations"] == run_d_nr_simulations)
        )

        # Validate
        invalid = False
        invalid |= _check_and_report(
            mask_a_base, mask_a, "Run A", run_a_umap_n_neighbors, run_a_nr_simulations, df
        )
        invalid |= _check_and_report(
            mask_b_base, mask_b, "Run B", run_b_umap_n_neighbors, run_b_nr_simulations, df
        )
        invalid |= _check_and_report(
            mask_c_base, mask_c, "Run C", run_c_umap_n_neighbors, run_c_nr_simulations, df
        )
        invalid |= _check_and_report(
            mask_d_base, mask_d, "Run D", run_d_umap_n_neighbors, run_d_nr_simulations, df
        )
        if invalid:
            raise SystemExit("One or more runs have invalid parameter combinations.")

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
        dims_map = df_a.groupby("dim_reduction_level")["dims"].first().to_dict()
        levels_ordered = sorted(levels_present, key=lambda l: dims_map.get(l, 0), reverse=True)
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

        # Aggregate
        agg_a = _aggregate(df_a, levels_ordered, mean_col, std_col)
        agg_b = _aggregate(df_b, levels_ordered, mean_col, std_col)
        agg_c = _aggregate(df_c, levels_ordered, mean_col, std_col)
        agg_d = _aggregate(df_d, levels_ordered, mean_col, std_col)

        # Build run labels (still include the prefix for construction, it will be stripped in the chart)
        run_label_a = f"Run A: {run_a_algo}" + (
            f" (k={run_a_umap_n_neighbors})" if run_a_algo != "PCA-DEA" else ""
        )
        run_label_b = f"Run B: {run_b_algo}" + (
            f" (k={run_b_umap_n_neighbors})" if run_b_algo != "PCA-DEA" else ""
        )
        run_label_c = f"Run C: {run_c_algo}" + (
            f" (k={run_c_umap_n_neighbors})" if run_c_algo != "PCA-DEA" else ""
        )
        run_label_d = f"Run D: {run_d_algo}" + (
            f" (k={run_d_umap_n_neighbors})" if run_d_algo != "PCA-DEA" else ""
        )
        run_labels = [run_label_a, run_label_b, run_label_c, run_label_d]

        # Short labels for filename
        def _short_label(algo, umap_k):
            if algo == "PCA-DEA":
                return "PCA-DEA"
            return f"{algo}_k{umap_k}"

        short_a = _short_label(run_a_algo, run_a_umap_n_neighbors)
        short_b = _short_label(run_b_algo, run_b_umap_n_neighbors)
        short_c = _short_label(run_c_algo, run_c_umap_n_neighbors)
        short_d = _short_label(run_d_algo, run_d_umap_n_neighbors)

        # Filename tag for N/n (only if they differ)
        N_values = {run_a_N, run_b_N, run_c_N, run_d_N}
        n_values = {run_a_n, run_b_n, run_c_n, run_d_n}
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

        # --- Create the bar chart (only output) ---
        x = np.arange(len(levels_ordered))
        width = 0.20
        fig_bar, ax_bar = plt.subplots(figsize=(18, 12))
        fig_bar.suptitle(
            f"N = {run_a_N}, n = {run_a_n}, {metric_name}", fontsize=22, fontweight="bold", y=0.98
        )

        _plot_bar_chart(
            ax_bar,
            x,
            width,
            agg_a,
            agg_b,
            agg_c,
            agg_d,
            show_std,
            run_labels,
            labels_ordered,
            metric_name,
        )

        bar_filename = (
            f"{metric_name}__bar__"
            f"A_{short_a}_B_{short_b}_C_{short_c}_D_{short_d}__"
            f"{dim_tag}.png"
        )
        path = os.path.join(output_dir, bar_filename)
        fig_bar.tight_layout()
        fig_bar.savefig(path, dpi=300, bbox_inches="tight")
        print(f"💾 Saved: {path}")
        plt.show()
        plt.close(fig_bar)

    finally:
        plt.rcParams.update(original_rc)  # restore defaults
