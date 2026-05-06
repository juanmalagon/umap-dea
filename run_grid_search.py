import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
import time


logger = logging.getLogger(__name__)


def create_grid_search_config(base_config_path: str, param_grid: dict):
    """
    Generate all parameter combinations from grid.
    param_grid: dict with parameter names as keys and lists of values
    Example: {"N": [20, 50, 100, 200], "n": [20, 50, 100, 200]}
    """
    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    # Get all combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    combinations = list(itertools.product(*param_values))

    results = []
    for combo in combinations:
        config = base_config.copy()
        for param_name, value in zip(param_names, combo):
            config[param_name] = value
        results.append(config)

    return results


def run_grid_search(
    param_grid: dict,
    base_config: str = "config.json",
    results_base_dir: str = "results_grid_search",
    verbose: bool = False,
):
    """Run simulations for all parameter combinations."""

    os.makedirs(results_base_dir, exist_ok=True)

    # Generate all configurations
    configs = create_grid_search_config(base_config, param_grid)
    total = len(configs)

    logger.info("Running %s parameter combinations...", total)
    log_level = "INFO" if verbose else "WARNING"

    grid_results = []
    start_time = time.monotonic()
    sim_times = []

    for i, config in enumerate(configs):
        sim_start = time.monotonic()

        logger.info(
            "[%s/%s] N=%s, n=%s (elapsed: %s, ETA: %s)",
            i + 1,
            total,
            config['N'],
            config['n'],
            _fmt_duration(time.monotonic() - start_time),
            _eta(time.monotonic() - start_time, i + 1, total),
        )

        # Create temporary config file for this run
        temp_config_path = f"{results_base_dir}/temp_config_{i}.json"
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=2)

        try:
            # Run simulation with this config using the current Python env
            subprocess.run(
                [
                    sys.executable, "run_sim.py",
                    "--config", temp_config_path,
                    "--log-level", log_level,
                ],
                check=True,
            )
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

        elapsed = time.monotonic() - sim_start
        sim_times.append(elapsed)
        logger.info(
            "  -> completed in %s (avg: %s per sim)",
            _fmt_duration(elapsed),
            _fmt_duration(sum(sim_times) / len(sim_times)),
        )

        grid_results.append(config)

    total_elapsed = time.monotonic() - start_time
    logger.info(
        "Completed all %s simulations in %s!",
        total,
        _fmt_duration(total_elapsed),
    )
    return grid_results


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(seconds, 60)
    if mins < 60:
        return f"{int(mins)}m {secs:.0f}s"
    hours, mins = divmod(mins, 60)
    return f"{int(hours)}h {int(mins)}m {secs:.0f}s"


def _eta(elapsed: float, done: int, total: int) -> str:
    """Estimate remaining time."""
    if done == 0:
        return "?"
    per_item = elapsed / done
    remaining = per_item * (total - done)
    return _fmt_duration(remaining)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search over parameter combinations for UMAP-DEA."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-simulation detail (INFO level logs from subprocess)."
    )
    parser.add_argument(
        "--param-grid",
        type=str,
        default=None,
        help=(
            "JSON string of the parameter grid, e.g. "
            '\'{"N": [20, 50], "n": [20, 50]}\'. '
            "Defaults to N=[20,50,100,200], n=[20,50,100,200]."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    if args.param_grid:
        param_grid = json.loads(args.param_grid)
    else:
        param_grid = {"N": [20, 50, 100, 200], "n": [20, 50, 100, 200]}

    run_grid_search(param_grid, verbose=args.verbose)
