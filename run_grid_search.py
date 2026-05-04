import itertools
import json
import subprocess
import os
from pathlib import Path
import pandas as pd


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

    print(f"Running {len(configs)} parameter combinations...\n")

    grid_results = []
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Running with N={config['N']}, n={config['n']}")

        # Create temporary config file for this run
        temp_config_path = f"{results_base_dir}/temp_config_{i}.json"
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Run simulation with this config
        subprocess.run(
            ["python", "run_sim.py", "--config", temp_config_path], check=True
        )

        grid_results.append(config)

    print(f"\nCompleted all {len(configs)} simulations!")
    return grid_results


if __name__ == "__main__":
    # Define your parameter grid
    param_grid = {"N": [20, 50, 100, 200], "n": [20, 50, 100, 200]}

    run_grid_search(param_grid)
