import logging

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau  # , pearsonr


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
    return float(pd.DataFrame({'x': x, 'y': y}).dropna().corr().iloc[0, 1])



def create_evaluation_df(
    efficiency_scores_dict: dict[str, np.ndarray],
    efficiency_score_by_design: np.ndarray,
    dims_for_embedding_dict: dict[str, int],
) -> pd.DataFrame:
    """Create an evaluation dataframe for all available embeddings."""

    logger.debug("Creating evaluation dataframe...")
    mae_dict: dict[str, float] = {}
    spearmanr_dict: dict[str, float] = {}
    pearsonr_dict: dict[str, float] = {}
    kendalltau_dict: dict[str, float] = {}
    nr_non_nan_dict: dict[str, int] = {}

    for k, v in efficiency_scores_dict.items():
        mae_dict[k] = nan_mae(efficiency_score_by_design, v)
        spearmanr_dict[k] = spearmanr(
            a=efficiency_score_by_design, b=v, nan_policy="omit"
        ).statistic  # type: ignore
        pearsonr_dict[k] = nan_pearsonr(efficiency_score_by_design, v)
        kendalltau_dict[k] = kendalltau(
            x=efficiency_score_by_design, y=v, nan_policy="omit"
        ).statistic  # type: ignore
        nr_non_nan_dict[k] = int(np.count_nonzero(~np.isnan(v)))

    mae_df = pd.DataFrame.from_dict(mae_dict, orient="index", columns=["mae"])
    spearmanr_df = pd.DataFrame.from_dict(
        spearmanr_dict, orient="index", columns=["spearmanr"]
    )
    pearsonr_df = pd.DataFrame.from_dict(
        pearsonr_dict, orient="index", columns=["pearsonr"]
    )
    kendalltau_df = pd.DataFrame.from_dict(
        kendalltau_dict, orient="index", columns=["kendalltau"]
    )
    nr_non_nan_df = pd.DataFrame.from_dict(
        nr_non_nan_dict, orient="index", columns=["nr_non_nan"]
    )

    dims_df = pd.DataFrame.from_dict(
        dims_for_embedding_dict, orient="index", columns=["dims"]
    )

    evaluation_df = pd.concat(
        [
            dims_df,
            mae_df,
            spearmanr_df,
            pearsonr_df,
            kendalltau_df,
            nr_non_nan_df,
        ],
        axis=1,
    )
    evaluation_df = evaluation_df.reset_index().rename(
        columns={'index': 'dim_reduction_level'}
    )
    logger.info("Evaluation dataframe created.")

    return evaluation_df
