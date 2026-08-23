"""Run the complete UMAP-DEA experiment matrix for one DEA orientation."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_SIM_PATH = Path(__file__).resolve().parent / "run_sim.py"

# Each tuple is (N, n, UMAP neighborhood values). These reproduce the
# study cases used for the existing input- and output-oriented result sets.
EXPERIMENT_CASES = (
    (10, 10, (3, 5)),
    (10, 25, (4, 5, 12)),
    (10, 40, (5, 6, 20)),
    (10, 100, (6, 10, 50)),
    (20, 20, (4, 10)),
    (20, 50, (5, 7, 25)),
    (20, 80, (6, 8, 40)),
    (20, 200, (7, 14, 100)),
    (50, 50, (5, 7, 25)),
    (50, 125, (6, 11, 62)),
    (50, 200, (7, 14, 100)),
    (50, 500, (8, 22, 250)),
    (100, 100, (6, 10, 50)),
    (100, 250, (7, 15, 125)),
    (100, 400, (8, 20, 200)),
    (100, 1000, (9, 31, 500)),
)


def build_experiment_configs(
    base_config: dict[str, Any], orientation: str
) -> list[dict[str, Any]]:
    """Build the PCA and UMAP configurations for the complete study."""
    configs: list[dict[str, Any]] = []
    for n_inputs, n_dmus, neighborhood_values in EXPERIMENT_CASES:
        pca_config = base_config | {
            "N": n_inputs,
            "n": n_dmus,
            "orientation": orientation,
            "pca": True,
            # PCA does not use this parameter; preserve the historical path.
            "umap_n_neighbors": 3,
        }
        configs.append(pca_config)

        for n_neighbors in neighborhood_values:
            configs.append(
                base_config | {
                    "N": n_inputs,
                    "n": n_dmus,
                    "orientation": orientation,
                    "pca": False,
                    "umap_n_neighbors": n_neighbors,
                }
            )
    return configs


def results_dir_for(config: dict[str, Any], results_root: Path) -> Path:
    """Return the established directory path for one experiment result."""
    method = "pca_dea" if config["pca"] else "umap_dea"
    gamma = str(config["gamma"]).replace(".", "p")
    return (
        results_root
        / f"nr_sim_{config['nr_simulations']}"
        / f"gamma_{gamma}"
        / method
        / f"N_{config['N']:03d}"
        / f"n_{config['n']:04d}"
        / f"k_{config['umap_n_neighbors']:03d}"
    )


def run_all_experiments(
    base_config: dict[str, Any], orientation: str, results_root: Path, verbose: bool
) -> None:
    """Run every experiment sequentially, keeping each result set separate."""
    configs = build_experiment_configs(base_config, orientation)
    log_level = "INFO" if verbose else "WARNING"

    for index, config in enumerate(configs, start=1):
        results_dir = results_dir_for(config, results_root)
        results_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            "[%s/%s] N=%s, n=%s, method=%s, k=%s",
            index,
            len(configs),
            config["N"],
            config["n"],
            "PCA" if config["pca"] else "UMAP",
            config["umap_n_neighbors"],
        )
        config_path = results_dir / "run_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(RUN_SIM_PATH),
                    "--config",
                    str(config_path),
                    "--results-dir",
                    str(results_dir),
                    "--log-level",
                    log_level,
                ],
                check=True,
            )
        finally:
            config_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the complete orientation-specific study."""
    parser = argparse.ArgumentParser(
        description="Run all UMAP-DEA experiment cases for one DEA orientation."
    )
    parser.add_argument(
        "--orientation",
        required=True,
        choices=("input", "output"),
        help="DEA orientation for the complete experiment study.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.json",
        help="Base configuration whose non-grid values are retained.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help=(
            "Top-level directory for the result tree. Defaults to a fresh "
            "results_<orientation>_oriented_recomputed directory."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    results_root = args.results_root or (
        PROJECT_ROOT / f"results_{args.orientation}_oriented_recomputed"
    )
    run_all_experiments(base_config, args.orientation, results_root, args.verbose)


if __name__ == "__main__":
    main()