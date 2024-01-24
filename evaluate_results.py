import pandas as pd
import numpy as np
# from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, kendalltau  # , pearsonr


def nan_mae(x, y):
    return np.nanmean(np.abs(x - y))


def nan_pearsonr(x, y):
    return pd.DataFrame({'x': x, 'y': y}).dropna().corr().iloc[0, 1]


def create_evaluation_df(
    efficiency_scores_dict: dict,
    efficiency_score_by_design: np.ndarray,
    dims_for_embedding_dict: dict,
) -> pd.DataFrame:
    """
    Create a dataframe with the evaluation metrics for each embedding.
    :param efficiency_scores_dict: dictionary with the efficiency scores for
    each embedding
    :param efficiency_score_by_design: theoretical efficiency scores
    :param dims_for_embedding_dict: dictionary with the number of dimensions
    for each embedding
    :return: dataframe with the evaluation metrics for each embedding
    """
    print("Creating evaluation dataframe...")
    mae_dict = {}
    spearmanr_dict = {}
    pearsonr_dict = {}
    kendalltau_dict = {}

    for k, v in efficiency_scores_dict.items():
        mae_dict[k] = nan_mae(efficiency_score_by_design, v)
        spearmanr_dict[k] = spearmanr(
            a=efficiency_score_by_design, b=v, nan_policy="omit"
        ).statistic  # type: ignore
        pearsonr_dict[k] = nan_pearsonr(efficiency_score_by_design, v)
        kendalltau_dict[k] = kendalltau(
            x=efficiency_score_by_design, y=v, nan_policy="omit"
        ).statistic

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

    dims_df = pd.DataFrame.from_dict(
        dims_for_embedding_dict, orient="index", columns=["dims"]
    )

    evaluation_df = pd.concat(
        [dims_df, mae_df, spearmanr_df, pearsonr_df, kendalltau_df], axis=1
    ).sort_values(by="dims", ascending=False)
    print("Evaluation dataframe created.")

    return evaluation_df
