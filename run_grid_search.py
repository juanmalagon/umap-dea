import itertools
import json
import logging
import os
import subprocess
import sys


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
):
    """Run simulations for all parameter combinations."""

    os.makedirs(results_base_dir, exist_ok=True)

    # Generate all configurations
    configs = create_grid_search_config(base_config, param_grid)

    logger.info("Running %s parameter combinations...", len(configs))

    grid_results = []
    for i, config in enumerate(configs):
        logger.info(
            "[%s/%s] Running with N=%s, n=%s",
            i + 1,
            len(configs),
            config['N'],
            config['n'],
        )

        # Create temporary config file for this run
        temp_config_path = f"{results_base_dir}/temp_config_{i}.json"
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=2)

        try:
            # Run simulation with this config using the current Python env
            subprocess.run(
                [sys.executable, "run_sim.py", "--config", temp_config_path],
                check=True,
            )
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)

        grid_results.append(config)

    logger.info("Completed all %s simulations!", len(configs))
    return grid_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    # Define your parameter grid
    param_grid = {"N": [20, 50, 100, 200], "n": [20, 50, 100, 200]}

    run_grid_search(param_grid)
