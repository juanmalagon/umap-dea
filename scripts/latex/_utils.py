"""Shared utilities for LaTeX generation scripts."""

import os
import re
import pandas as pd
import numpy as np


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
#  Path extraction helpers
# ---------------------------------------------------------------------------

# Patterns to extract method + optional k from a results path.
_METHOD_K_RE = re.compile(
    r"/(pca_dea|umap_dea)(?:/k_(\d+))?/"
)


def get_output_subpath(filepath: str, params=None) -> str:
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
        k = m.group(2)        # "15", "5", …  (None for pca_dea)
        if k is not None:
            return f"{method}/k_{k}"
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


# ---------------------------------------------------------------------------
#  Shared LaTeX formatting helpers
# ---------------------------------------------------------------------------

def format_hyperparams_suffix(params: dict, include_method_specific: bool = True) -> str:
    r"""Build a consistent hyperparameter suffix string for LaTeX captions.

    Format matches the style used in ``generate_tables.py``, with LaTeX
    inline-math wrapping around each parameter:

      , \\(gamma=1\\), \\(sigma_u=0.1\\), \\(alpha_1=0.25\\), \\(M=1\\)

    with optional UMAP-specific parameters appended.

    Parameters
    ----------
    params:
        Dictionary containing parameter values (typically from params_dict CSV).
    include_method_specific:
        If True, also append UMAP-specific params (k, min_dist, metric) when the
        experiment is not PCA.  Set to False for UMAP-vs-PCA comparisons where
        only shared parameters are relevant.

    Returns
    -------
    str
        A LaTeX-formatted string ready to embed in a caption, e.g.
        ``", \\(\\gamma=1\\), \\(\\sigma_u=0.1\\), \\(\\alpha_1=0.25\\), \\(M=1\\)"``.
        If key parameters are missing, the suffix gracefully omits them.
    """
    parts = []

    # Shared parameters (always included if present)
    gamma = params.get("gamma")
    if gamma is not None:
        try:
            f_gamma = float(gamma)
            if f_gamma == int(f_gamma):
                gamma_str = str(int(f_gamma))
            else:
                gamma_str = str(f_gamma)
        except (ValueError, TypeError):
            gamma_str = str(gamma)
        parts.append(f"\\gamma={gamma_str}")

    sigma_u = params.get("sigma_u")
    if sigma_u is not None and not (isinstance(sigma_u, float) and sigma_u != sigma_u):  # skip NaN
        try:
            parts.append(f"\\sigma_u={float(sigma_u):g}")
        except (ValueError, TypeError):
            pass

    alpha_1 = params.get("alpha_1")
    if alpha_1 is not None and not (isinstance(alpha_1, float) and alpha_1 != alpha_1):
        try:
            parts.append(f"\\alpha_1={float(alpha_1):g}")
        except (ValueError, TypeError):
            pass

    M = params.get("M")
    if M is not None and not (isinstance(M, float) and M != M):
        try:
            M_int = int(float(M))
            parts.append(f"M={M_int}")
        except (ValueError, TypeError):
            pass

    # Method-specific parameters (UMAP only, not PCA)
    is_pca = bool(params.get("pca", False))
    if include_method_specific and not is_pca:
        k = params.get("umap_n_neighbors")
        if k is not None and not (isinstance(k, float) and k != k):
            try:
                parts.append(f"k={int(float(k))}")
            except (ValueError, TypeError):
                pass

        min_dist = params.get("umap_min_dist")
        if min_dist is not None and not (isinstance(min_dist, float) and min_dist != min_dist):
            try:
                parts.append(f"\\text{{min\\_dist}}={float(min_dist):g}")
            except (ValueError, TypeError):
                pass

        metric = params.get("umap_metric")
        if metric is not None and not (isinstance(metric, float) and metric != metric):
            # metric is typically a string like "euclidean"
            parts.append(f"\\text{{metric}}={str(metric)}")

    if not parts:
        return ""

    # Build suffix: ", \(\gamma=1\), \(\sigma_u=0.1\), ...""
    return ", " + ", ".join(f"\\({part}\\)" for part in parts)


def extract_uuid(filename):
    """Extract run UUID from a summary_df or params_dict csv filename."""
    match = re.search(r"(?:summary_df|params_dict)_([0-9a-fA-F-]+)\.csv", filename)
    if match:
        return match.group(1)
    return ""


def fmt_val(val, ndigits=4):
    """Format a single numeric value for LaTeX."""
    if pd.isna(val):
        return "—"
    return f"{val:.{ndigits}f}"


def fmt_pct(val):
    """Format as percentage with one decimal for LaTeX."""
    if pd.isna(val):
        return "—"
    return f"{val * 100:.1f}\\%"


def bold_if_better(val_a, val_b, best_direction, ndigits=4):
    """Return (fmt_a, fmt_b) with the better value bolded.

    best_direction: 'min' means lower is better, 'max' means higher is better.
    On a tie, neither value is bolded.
    """
    if pd.isna(val_a) or pd.isna(val_b):
        return fmt_val(val_a, ndigits), fmt_val(val_b, ndigits)

    str_a = f"{val_a:.{ndigits}f}"
    str_b = f"{val_b:.{ndigits}f}"

    if best_direction == "min":
        better_a = val_a < val_b
    else:
        better_a = val_a > val_b

    if better_a:
        str_a = f"\\textbf{{{str_a}}}"
    else:
        str_b = f"\\textbf{{{str_b}}}"

    return str_a, str_b


def bold_pct_if_better(val_a, val_b, best_direction):
    """Same as bold_if_better for percentage values (lower is better for non-discriminating)."""
    if pd.isna(val_a) or pd.isna(val_b):
        return fmt_pct(val_a), fmt_pct(val_b)

    pct_a = val_a * 100
    pct_b = val_b * 100
    str_a = f"{pct_a:.1f}\\%"
    str_b = f"{pct_b:.1f}\\%"

    if best_direction == "min":
        better_a = pct_a < pct_b
    else:
        better_a = pct_a > pct_b

    if better_a:
        str_a = f"\\textbf{{{str_a}}}"
    else:
        str_b = f"\\textbf{{{str_b}}}"

    return str_a, str_b