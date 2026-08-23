import logging
import warnings

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, ConstantInputWarning

logger = logging.getLogger(__name__)


def nan_mae(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the mean absolute error between two arrays, ignoring NaNs.
    """
    return float(np.nanmean(np.abs(x - y)))


def nan_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient between two arrays, ignoring
    NaNs.
    """
    return float(pd.DataFrame({"x": x, "y": y}).dropna().corr().iloc[0, 1])


def _count_efficient(
    efficiency_scores: np.ndarray,
    tolerance: float = 1e-6,
) -> tuple[int, int, float]:
    """
    Count the number of efficient (frontier) DMUs.

    A DMU is considered efficient if its DEA score is within *tolerance* of
    1.0 (the frontier), after excluding NaN values.

    Parameters
    ----------
    efficiency_scores : np.ndarray
        1-D array of DEA efficiency scores.
    tolerance : float
        Absolute tolerance for the ``== 1.0`` check (default 1e-6).

    Returns
    -------
    nr_efficient : int
        Number of efficient DMUs.
    nr_non_nan : int
        Number of non-NaN DMUs.
    prop_efficient : float
        Proportion of (non-NaN) DMUs on the frontier.  Returns ``np.nan``
        when there are no valid (non-NaN) scores.
    """
    valid_mask = ~np.isnan(efficiency_scores)
    nr_non_nan = int(np.sum(valid_mask))

    if nr_non_nan == 0:
        return 0, 0, float("nan")

    nr_efficient = int(np.sum(np.abs(efficiency_scores[valid_mask] - 1.0) <= tolerance))
    prop_efficient = nr_efficient / nr_non_nan
    return nr_efficient, nr_non_nan, prop_efficient


def _safe_spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """
    Compute Spearman's rank correlation, handling constant-input cases.

    Drops paired NaN entries and catches ``ConstantInputWarning``, returning
    ``(NaN, True)`` when correlation is undefined.
    """
    return _safe_rank_correlation(
        lambda a, b: spearmanr(a, b).statistic,
        x,
        y,
    )


def _safe_kendalltau(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """
    Compute Kendall's tau, handling constant-input cases.

    Drops paired NaN entries and catches ``ConstantInputWarning``, returning
    ``(NaN, True)`` when correlation is undefined.
    """
    return _safe_rank_correlation(
        lambda a, b: kendalltau(a, b).statistic,
        x,
        y,
    )


def _safe_rank_correlation(corr_func, x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """
    Generic wrapper for rank-correlation functions.

    Parameters
    ----------
    corr_func : callable
        A function ``f(x, y) -> float`` that computes a rank correlation.
    x, y : np.ndarray
        Input arrays.

    Returns
    -------
    statistic : float
        Correlation value, or NaN when undefined.
    warning : bool
        True if the correlation is undefined (constant-input) or could not be
        computed.
    """
    # Drop paired NaNs
    valid = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid]
    y_clean = y[valid]

    if len(x_clean) < 3:
        # Need at least 3 points for meaningful rank correlation
        return float("nan"), True

    # Check for sufficient variation in both arrays
    x_unique = len(np.unique(x_clean))
    y_unique = len(np.unique(y_clean))
    if x_unique < 2 or y_unique < 2:
        return float("nan"), True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            stat = float(corr_func(x_clean, y_clean))
        except Exception:
            return float("nan"), True

        # Check if ConstantInputWarning was raised
        has_warning = any(issubclass(w.category, ConstantInputWarning) for w in caught)
        if has_warning:
            return float("nan"), True

    return stat, False


def create_evaluation_df(
    efficiency_scores_dict: dict[str, np.ndarray],
    efficiency_score_by_design: np.ndarray,
    dims_for_embedding_dict: dict[str, int],
    efficiency_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Create an evaluation dataframe for all available embeddings.

    Parameters
    ----------
    efficiency_scores_dict : dict[str, np.ndarray]
        Dictionary mapping embedding names to their DEA efficiency scores.
    efficiency_score_by_design : np.ndarray
        The ground-truth efficiency scores (from the DGP).
    dims_for_embedding_dict : dict[str, int]
        Dictionary mapping embedding names to embedding dimensions.
    efficiency_tolerance : float
        Absolute tolerance for classifying a DMU as efficient
        (score within ``tolerance`` of 1.0).

    Returns
    -------
    pd.DataFrame
        Evaluation dataframe with one row per embedding (including
        ``"original"``), containing columns for MAE, Spearman's r, Pearson's r,
        Kendall's tau, number of non-NaN DMUs, number / proportion of efficient
        DMUs, and correlation warning flags.
    """

    logger.debug("Creating evaluation dataframe...")
    mae_dict: dict[str, float] = {}
    spearmanr_dict: dict[str, float] = {}
    spearmanr_warning_dict: dict[str, bool] = {}
    pearsonr_dict: dict[str, float] = {}
    kendalltau_dict: dict[str, float] = {}
    kendalltau_warning_dict: dict[str, bool] = {}
    nr_non_nan_dict: dict[str, int] = {}
    nr_efficient_dict: dict[str, int] = {}
    prop_efficient_dict: dict[str, float] = {}

    for k, v in efficiency_scores_dict.items():
        mae_dict[k] = nan_mae(efficiency_score_by_design, v)

        spearmanr_val, spearmanr_warn = _safe_spearmanr(efficiency_score_by_design, v)
        spearmanr_dict[k] = spearmanr_val
        spearmanr_warning_dict[k] = spearmanr_warn

        pearsonr_dict[k] = nan_pearsonr(efficiency_score_by_design, v)

        kendalltau_val, kendalltau_warn = _safe_kendalltau(efficiency_score_by_design, v)
        kendalltau_dict[k] = kendalltau_val
        kendalltau_warning_dict[k] = kendalltau_warn

        nr_eff, nr_non_nan, prop_eff = _count_efficient(v, tolerance=efficiency_tolerance)
        nr_non_nan_dict[k] = nr_non_nan
        nr_efficient_dict[k] = nr_eff
        prop_efficient_dict[k] = prop_eff

    mae_df = pd.DataFrame.from_dict(mae_dict, orient="index", columns=["mae"])
    spearmanr_df = pd.DataFrame.from_dict(spearmanr_dict, orient="index", columns=["spearmanr"])
    spearmanr_warning_df = pd.DataFrame.from_dict(
        spearmanr_warning_dict, orient="index", columns=["spearmanr_warning"]
    )
    pearsonr_df = pd.DataFrame.from_dict(pearsonr_dict, orient="index", columns=["pearsonr"])
    kendalltau_df = pd.DataFrame.from_dict(kendalltau_dict, orient="index", columns=["kendalltau"])
    kendalltau_warning_df = pd.DataFrame.from_dict(
        kendalltau_warning_dict, orient="index", columns=["kendalltau_warning"]
    )
    nr_non_nan_df = pd.DataFrame.from_dict(nr_non_nan_dict, orient="index", columns=["nr_non_nan"])
    nr_efficient_df = pd.DataFrame.from_dict(
        nr_efficient_dict, orient="index", columns=["nr_efficient"]
    )
    prop_efficient_df = pd.DataFrame.from_dict(
        prop_efficient_dict, orient="index", columns=["prop_efficient"]
    )

    dims_df = pd.DataFrame.from_dict(dims_for_embedding_dict, orient="index", columns=["dims"])

    evaluation_df = pd.concat(
        [
            dims_df,
            mae_df,
            spearmanr_df,
            spearmanr_warning_df,
            pearsonr_df,
            kendalltau_df,
            kendalltau_warning_df,
            nr_non_nan_df,
            nr_efficient_df,
            prop_efficient_df,
        ],
        axis=1,
    )
    evaluation_df = evaluation_df.reset_index().rename(columns={"index": "dim_reduction_level"})
    logger.info("Evaluation dataframe created.")

    return evaluation_df


def get_efficiency_summary(
    efficiency_scores_dict: dict[str, np.ndarray],
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """
    Create a summary DataFrame of DMU efficiency for each embedding.

    This is a convenience function for obtaining per-embedding counts of
    efficient DMUs and NaN scores, especially useful for the reviewer
    questions about frontier saturation.

    Parameters
    ----------
    efficiency_scores_dict : dict[str, np.ndarray]
        Dictionary mapping embedding names to their DEA efficiency scores.
    tolerance : float
        Absolute tolerance for classifying a DMU as efficient.

    Returns
    -------
    pd.DataFrame
        Dataframe indexed by embedding name with columns:
        ``nr_efficient``, ``nr_non_nan``, ``prop_efficient``.
    """
    rows = []
    for name, scores in efficiency_scores_dict.items():
        nr_eff, nr_non_nan, prop_eff = _count_efficient(scores, tolerance=tolerance)
        rows.append(
            {
                "dim_reduction_level": name,
                "nr_efficient": nr_eff,
                "nr_non_nan": nr_non_nan,
                "prop_efficient": prop_eff,
            }
        )
    return pd.DataFrame(rows).set_index("dim_reduction_level")
