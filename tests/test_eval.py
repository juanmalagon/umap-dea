import pytest
import numpy as np
import pandas as pd
from umap_dea import eval


class TestNanMAE:
    """Test nan_mae function."""

    def test_mae_calculation(self):
        """Test basic MAE calculation."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.1, 2.1, 3.1])

        mae = eval.nan_mae(x, y)

        assert isinstance(mae, (float, np.floating))
        assert mae == pytest.approx(0.1)

    def test_mae_with_nans(self):
        """Test that NaNs are ignored."""
        x = np.array([1.0, 2.0, np.nan, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        mae = eval.nan_mae(x, y)

        assert not np.isnan(mae)
        assert mae == pytest.approx(0.0)

    def test_mae_perfect_match(self):
        """Test MAE when arrays are identical."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        mae = eval.nan_mae(x, y)

        assert mae == pytest.approx(0.0)


class TestNanPearsonr:
    """Test nan_pearsonr function."""

    def test_pearsonr_perfect_correlation(self):
        """Test Pearson correlation with perfect correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        corr = eval.nan_pearsonr(x, y)

        assert corr == pytest.approx(1.0)

    def test_pearsonr_negative_correlation(self):
        """Test Pearson correlation with negative correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([4.0, 3.0, 2.0, 1.0])

        corr = eval.nan_pearsonr(x, y)

        assert corr == pytest.approx(-1.0)

    def test_pearsonr_with_nans(self):
        """Test that NaNs are handled correctly."""
        x = np.array([1.0, 2.0, np.nan, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        corr = eval.nan_pearsonr(x, y)

        assert isinstance(corr, (float, np.floating))
        assert not np.isnan(corr)


class TestCreateEvaluationDF:
    """Test create_evaluation_df function."""

    def test_evaluation_df_structure(self, efficiency_scores):
        """Test that evaluation dataframe has expected structure."""
        efficiency_scores_dict = {
            "embedding1": efficiency_scores,
            "embedding2": efficiency_scores * 0.95,
        }
        efficiency_score_by_design = efficiency_scores
        dims_for_embedding_dict = {"embedding1": 5, "embedding2": 3, "original": 10}

        eval_df = eval.create_evaluation_df(
            efficiency_scores_dict,
            efficiency_score_by_design,
            dims_for_embedding_dict,
        )

        assert isinstance(eval_df, pd.DataFrame)
        assert len(eval_df) == len(dims_for_embedding_dict)
        assert set(eval_df["dim_reduction_level"]) == set(dims_for_embedding_dict)

    def test_evaluation_df_columns(self, efficiency_scores):
        """Test that evaluation dataframe has expected columns."""
        efficiency_scores_dict = {
            "embedding1": efficiency_scores,
            "embedding2": efficiency_scores * 0.95,
        }
        efficiency_score_by_design = efficiency_scores
        dims_for_embedding_dict = {"embedding1": 5, "embedding2": 3, "original": 10}

        eval_df = eval.create_evaluation_df(
            efficiency_scores_dict,
            efficiency_score_by_design,
            dims_for_embedding_dict,
        )

        expected_columns = {
            "dim_reduction_level",
            "dims",
            "mae",
            "spearmanr",
            "pearsonr",
            "kendalltau",
            "nr_non_nan",
        }
        assert set(eval_df.columns) == expected_columns

    def test_evaluation_df_values(self, efficiency_scores):
        """Test that evaluation metrics are reasonable."""
        efficiency_scores_dict = {"embedding1": efficiency_scores}
        efficiency_score_by_design = efficiency_scores
        dims_for_embedding_dict = {"embedding1": 5, "original": 10}

        eval_df = eval.create_evaluation_df(
            efficiency_scores_dict,
            efficiency_score_by_design,
            dims_for_embedding_dict,
        )

        # When scores are identical, MAE should be 0
        assert eval_df.iloc[0]["mae"] == pytest.approx(0.0)
        # Correlation with itself should be 1
        assert eval_df.iloc[0]["spearmanr"] == pytest.approx(1.0)
        assert eval_df.iloc[0]["pearsonr"] == pytest.approx(1.0)
        assert eval_df.iloc[0]["kendalltau"] == pytest.approx(1.0)

    def test_evaluation_df_with_noise(self, efficiency_scores):
        """Test evaluation with noisy efficiency scores."""
        np.random.seed(42)
        efficiency_scores_dict = {
            "embedding1": efficiency_scores + np.random.randn(50) * 0.01
        }
        efficiency_score_by_design = efficiency_scores
        dims_for_embedding_dict = {"embedding1": 5, "original": 10}

        eval_df = eval.create_evaluation_df(
            efficiency_scores_dict,
            efficiency_score_by_design,
            dims_for_embedding_dict,
        )

        # MAE should be small but non-zero
        assert 0 < eval_df.iloc[0]["mae"] < 0.1
        # Correlation should be high but not perfect
        assert 0.9 < eval_df.iloc[0]["spearmanr"] <= 1.0
        assert 0.9 < eval_df.iloc[0]["pearsonr"] <= 1.0

    def test_evaluation_df_with_nans(self):
        """Test evaluation dataframe handles NaNs."""
        n = 50
        efficiency_scores_dict = {
            "embedding1": np.concatenate(
                [np.random.rand(40), np.full(10, np.nan)]
            )
        }
        efficiency_score_by_design = np.random.rand(n)
        dims_for_embedding_dict = {"embedding1": 5, "original": 10}

        eval_df = eval.create_evaluation_df(
            efficiency_scores_dict,
            efficiency_score_by_design,
            dims_for_embedding_dict,
        )

        # Should not raise an error and should record the number of non-NaN values
        assert eval_df.iloc[0]["nr_non_nan"] < n
