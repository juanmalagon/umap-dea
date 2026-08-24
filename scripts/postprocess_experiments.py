"""Aggregate completed experiment results and generate all comparison plots."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

try:
    from scripts.aggregate_results import aggregate_results
    from scripts.run_all_experiments import EXPERIMENT_CASES, PROJECT_ROOT, k_options_for_n
except ImportError:
    from aggregate_results import aggregate_results
    from run_all_experiments import EXPERIMENT_CASES, PROJECT_ROOT, k_options_for_n


EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from compare_four_runs_data import add_algorithm_column
from compare_four_runs_plot import plot_comparison

DEFAULT_METRICS = ["Kendall", "MAE", "Pearson", "Spearman"]


def comparison_runs_for_case(
    base_config: dict[str, Any], n_inputs: int, n_dmus: int
) -> list[dict[str, Any]]:
    """Build PCA plus unique UMAP-k comparison runs for one (N, n) case."""
    common = {
        "N": n_inputs,
        "n": n_dmus,
        "rts": base_config["rts"],
        "gamma": base_config["gamma"],
        "nr_simulations": base_config["nr_simulations"],
    }
    runs = [
        {
            **common,
            "algo": "PCA-DEA",
            "umap_n_neighbors": 3,
        }
    ]
    for k_value, k_label in k_options_for_n(n_dmus):
        runs.append(
            {
                **common,
                "algo": "UMAP-DEA",
                "umap_n_neighbors": k_value,
                "k_label": k_label,
            }
        )
    return runs


def generate_all_plots(
    df,
    base_config: dict[str, Any],
    plots_root: Path,
    metric_names: list[str],
    show_std: bool,
    allow_missing: bool,
) -> None:
    """Generate comparison plots for every configured (N, n) case."""
    skipped_cases: list[tuple[int, int]] = []
    for n_inputs, n_dmus, _neighborhood_values in EXPERIMENT_CASES:
        case_mask = (df["N"] == n_inputs) & (df["n"] == n_dmus)
        if case_mask.sum() == 0:
            skipped_cases.append((n_inputs, n_dmus))
            logging.warning(
                "Skipping N=%s, n=%s because no matching rows exist in the aggregate data",
                n_inputs,
                n_dmus,
            )
            continue

        runs = comparison_runs_for_case(base_config, n_inputs, n_dmus)
        case_dir = plots_root / f"N_{n_inputs:03d}" / f"n_{n_dmus:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Plotting N=%s, n=%s", n_inputs, n_dmus)
        try:
            plot_comparison(
                df,
                runs,
                metric_names=metric_names,
                show_std=show_std,
                output_dir=case_dir,
                show=False,
            )
        except SystemExit:
            if not allow_missing:
                raise
            logging.warning(
                "Skipping N=%s, n=%s because matching data is missing", n_inputs, n_dmus
            )

    if skipped_cases:
        skipped = ", ".join(f"N={n_inputs}, n={n_dmus}" for n_inputs, n_dmus in skipped_cases)
        logging.warning("Skipped %s completely missing case(s): %s", len(skipped_cases), skipped)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for orientation-specific aggregation and plotting."""
    parser = argparse.ArgumentParser(
        description="Aggregate all results and generate all comparison plots."
    )
    parser.add_argument("--orientation", required=True, choices=("input", "output"))
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.json",
        help="Base config used to fill rts, gamma, and nr_simulations.",
    )
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--plots-root", type=Path, default=None)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--show-std", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    suffix = f"{args.orientation}_oriented_recomputed"
    results_root = args.results_root or PROJECT_ROOT / f"results_{suffix}"
    output_csv = args.output_csv or PROJECT_ROOT / f"all_results_{suffix}.csv"
    plots_root = args.plots_root or EXPERIMENTS_DIR / "plots" / suffix

    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    df = aggregate_results(results_root, output_csv)
    df = add_algorithm_column(df)
    generate_all_plots(
        df=df,
        base_config=base_config,
        plots_root=plots_root,
        metric_names=args.metrics,
        show_std=args.show_std,
        allow_missing=args.allow_missing,
    )
    logging.info("Wrote aggregate CSV to %s", output_csv)
    logging.info("Wrote plots under %s", plots_root)


if __name__ == "__main__":
    main()
