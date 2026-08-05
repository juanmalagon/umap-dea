"""Module to load configuration for the compare_four_runs notebook."""

import json
import os


def load_config(path=None):
    """Load and resolve the compare_four_runs configuration.

    Parameters
    ----------
    path : str, optional
        Path to the JSON config file. Defaults to
        ``compare_four_runs_config.json`` in the same directory.

    Returns
    -------
    dict
        A dictionary with keys:
        - ``run_a`` .. ``run_d`` : each a dict with keys
          algo, N, n, rts, gamma, umap_n_neighbors, nr_simulations
        - ``metric_names`` : list of str
        - ``show_std`` : bool
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "compare_four_runs_config.json")

    with open(path, "r") as fh:
        raw = json.load(fh)

    global_defaults = raw["global"]

    def _resolve(run_name):
        """Replace every 'global' sentinel with the value from the global block."""
        run = raw[run_name]
        resolved = {}
        for key in global_defaults:
            val = run.get(key, "global")
            resolved[key] = global_defaults[key] if val == "global" else val
        return resolved

    # Support both "metric_names" (new, array) and "metric_name" (legacy, string)
    if "metric_names" in raw:
        metric_names = raw["metric_names"]
    else:
        metric_names = [raw.get("metric_name", "Kendall")]

    config = {
        "run_a": _resolve("run_a"),
        "run_b": _resolve("run_b"),
        "run_c": _resolve("run_c"),
        "run_d": _resolve("run_d"),
        "metric_names": metric_names,
        "show_std": raw.get("show_std", True),
    }
    return config
