"""Shared utilities for LaTeX generation scripts."""

import os
import re


def gamma_to_dirname(gamma) -> str:
    """Format gamma value for directory name, consistent with organize_results.py.

    e.g. '0.5' -> '0p5', '1.0' -> '1', '2.0' -> '2', '1.5' -> '1p5'.
    """
    try:
        f = float(gamma)
    except (ValueError, TypeError):
        return str(gamma)
    if f == int(f):
        return str(int(f))
    # Replace decimal point with 'p' for fractional values
    return str(f).replace(".", "p")


# ---------------------------------------------------------------------------
#  Resolve output subpath from results file path and/or params dict
# ---------------------------------------------------------------------------

# Patterns to extract method + optional k from a results path.
_METHOD_K_RE = re.compile(
    r"/(pca_dea|umap_dea)(?:/k_(\d+))?/"
)


def get_output_subpath(filepath: str, params=None) -> str:
    # params is optional dict or None
    """Return the relative subpath within tex/ for a given results file.

    Inspects the *filepath* (e.g. ``results/gamma_1/umap_dea/k_15/...csv``)
    to extract the method folder and, for UMAP runs, the ``k_XX`` subfolder.
    Falls back to reading ``params["pca"]`` and ``params["umap_n_neighbors"]``
    if the path doesn't contain the expected structure.

    Returns
    -------
    str
        ``"pca_dea"`` or ``"umap_dea/k_15"`` (no trailing slash).
    """
    m = _METHOD_K_RE.search(filepath)
    if m:
        method = m.group(1)   # "pca_dea" or "umap_dea"
        k = m.group(2)        # "05", "15", …  (None for pca_dea)
        if k is not None:
            return f"{method}/k_{int(k)}"
        return method

    # Fallback: use params dict
    if params is not None:
        is_pca = bool(params.get("pca", False))
        if not is_pca:
            nn = int(params.get("umap_n_neighbors", 15))
            return f"umap_dea/k_{nn}"
        return "pca_dea"

    # Ultimate fallback (shouldn't happen)
    return "pca_dea"


def get_output_subpath_from_params(params: dict) -> str:
    """Return the relative subpath within tex/ based solely on params dict.

    Returns
    -------
    str
        ``"pca_dea"`` or ``"umap_dea/k_15"`` (no trailing slash).
    """
    is_pca = bool(params.get("pca", False))
    if is_pca:
        return "pca_dea"
    nn = int(params.get("umap_n_neighbors", 15))
    return f"umap_dea/k_{nn}"