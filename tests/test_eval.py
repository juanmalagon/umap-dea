import pytest
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning
from scripts.run_sim import _design_score_for_orientation
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


class TestDesignScoreForOrientation:
    """Ensure DGP references use the same direction as DEA scores."""

    def test_output_reference_is_the_input_efficiency_reciprocal(self):
        y = np.array([[0.5], [0.8], [1.0]])
        y_tilde = np.ones_like(y)

        input_score = _design_score_for_orientation(y, y_tilde, "input")
        output_score = _design_score_for_orientation(y, y_tilde, "output")

        np.testing.assert_allclose(input_score, [0.5, 0.8, 1.0])
        np.testing.assert_allclose(output_score, [2.0, 1.25, 1.0])


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


class TestCountEfficient:
    """Test _count_efficient helper."""

    def test_all_efficient(self):
        """All non-NaN DMUs are exactly 1.0."""
        scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        nr_eff, nr_nn, prop = eval._count_efficient(scores)
        assert nr_eff == 5
        assert nr_nn == 5
        assert prop == pytest.approx(1.0)

    def test_none_efficient(self):
        """No DMU reaches the frontier."""
        scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        nr_eff, nr_nn, prop = eval._count_efficient(scores)
        assert nr_eff == 0
        assert nr_nn == 5
        assert prop == pytest.approx(0.0)

    def test_mixed(self):
        """Three out of five are on the frontier."""
        scores = np.array([1.0, 0.5, 1.0, 0.8, 1.0])
        nr_eff, nr_nn, prop = eval._count_efficient(scores)
        assert nr_eff == 3
        assert nr_nn == 5
        assert prop == pytest.approx(0.6)

    def test_with_nans(self):
        """NaNs are excluded from both counts."""
        scores = np.array([1.0, 1.0, np.nan, 0.5, np.nan, 1.0])
        nr_eff, nr_nn, prop = eval._count_efficient(scores)
        assert nr_eff == 3
        assert nr_nn == 4
        assert prop == pytest.approx(0.75)

    def test_all_nan(self):
        """Entire array is NaN."""
        scores = np.full(10, np.nan)
        nr_eff, nr_nn, prop = eval._count_efficient(scores)
        assert nr_eff == 0
        assert nr_nn == 0
        assert np.isnan(prop)

    def test_tolerance_respected(self):
        """Values clearly inside the tolerance window are flagged efficient."""
        tol = 1e-4
        scores = np.array([1.0 - 0.5 * tol, 1.0, 1.0 + 0.5 * tol])
        nr_eff, nr_nn, _ = eval._count_efficient(scores, tolerance=tol)
        assert nr_eff == 3  # all within 1e-4 of 1.0

    def test_tolerance_boundary(self):
        """Values just inside tolerance are flagged, just outside are not."""
        tol = 1e-4
        scores = np.array([1.0 - 0.99 * tol, 1.0 + 0.99 * tol, 1.0 - 1.01 * tol, 1.0 + 1.01 * tol])
        nr_eff, nr_nn, _ = eval._count_efficient(scores, tolerance=tol)
        assert nr_eff == 2  # first two are inside tolerance

    def test_outside_tolerance(self):
        """Values clearly outside tolerance are not flagged efficient."""
        tol = 1e-4
        scores = np.array([1.0 - 2 * tol, 1.0 + 2 * tol])
        nr_eff, nr_nn, _ = eval._count_efficient(scores, tolerance=tol)
        assert nr_eff == 0


class TestSafeRankCorrelation:
    """Test _safe_spearmanr and _safe_kendalltau wrappers."""

    def test_normal_case(self):
        """Varying arrays produce valid correlation and no warning."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 3.0, 5.0, 7.0, 11.0])
        stat, warn = eval._safe_spearmanr(x, y)
        assert not np.isnan(stat)
        assert not warn

        stat, warn = eval._safe_kendalltau(x, y)
        assert not np.isnan(stat)
        assert not warn

    def test_constant_array(self):
        """Constant array produces NaN and a warning flag."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.ones(5)

        stat, warn = eval._safe_spearmanr(x, y)
        assert np.isnan(stat)
        assert warn

        stat, warn = eval._safe_kendalltau(x, y)
        assert np.isnan(stat)
        assert warn

    def test_too_few_points(self):
        """Fewer than 3 valid points should yield NaN + warning."""
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        stat, warn = eval._safe_spearmanr(x, y)
        assert np.isnan(stat)
        assert warn

    def test_with_nans(self):
        """Paired NaNs are dropped, remaining points are sufficient."""
        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        y = np.array([2.0, 3.0, 5.0, np.nan, 7.0])
        stat, warn = eval._safe_spearmanr(x, y)
        assert not np.isnan(stat)
        assert not warn

    def test_too_many_nans(self):
        """After dropping NaNs, fewer than 3 points remain."""
        x = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        y = np.array([2.0, np.nan, np.nan, 7.0, np.nan])
        stat, warn = eval._safe_spearmanr(x, y)
        assert np.isnan(stat)
        assert warn

    def test_constant_after_nan_removal(self):
        """After removing NaNs, one array is constant."""
        x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
        y = np.array([0.5, 0.5, 0.5, 0.5, np.nan])
        stat, warn = eval._safe_spearmanr(x, y)
        assert np.isnan(stat)
        assert warn


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
        """Test that evaluation dataframe has all expected columns."""
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
            "spearmanr_warning",
            "pearsonr",
            "kendalltau",
            "kendalltau_warning",
            "nr_non_nan",
            "nr_efficient",
            "prop_efficient",
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
        # No warning flags for perfectly varying data
        assert not eval_df.iloc[0]["spearmanr_warning"]
        assert not eval_df.iloc[0]["kendalltau_warning"]

    def test_evaluation_df_with_noise(self, efficiency_scores):
        """Test evaluation with noisy efficiency scores."""
        np.random.seed(42)
        efficiency_scores_dict = {"embedding1": efficiency_scores + np.random.randn(50) * 0.01}
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
            "embedding1": np.concatenate([np.random.rand(40), np.full(10, np.nan)])
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
        # Efficient count should be consistent (only non-NaN scores counted)
        assert eval_df.iloc[0]["nr_efficient"] <= eval_df.iloc[0]["nr_non_nan"]

    def test_evaluation_df_efficient_columns(self):
        """Test that nr_efficient and prop_efficient are populated correctly."""
        n = 50
        # Half efficient, half not
        scores = np.ones(n)
        scores[n // 2 :] = 0.5
        efficiency_scores_dict = {"embedding1": scores}
        efficiency_score_by_design = np.random.rand(n)
        dims_for_embedding_dict = {"embedding1": 5, "original": 10}

        eval_df = eval.create_evaluation_df(
            efficiency_scores_dict,
            efficiency_score_by_design,
            dims_for_embedding_dict,
        )

        row = eval_df[eval_df["dim_reduction_level"] == "embedding1"].iloc[0]
        assert row["nr_efficient"] == 25
        assert row["prop_efficient"] == pytest.approx(0.5)
        assert row["nr_non_nan"] == 50

    def test_evaluation_df_constant_input_warning(self):
        """When DMU scores are all 1.0, correlation should be NaN with warning."""
        n = 50
        # All efficient — constant array
        scores = np.ones(n)
        efficiency_scores_dict = {"embedding1": scores}
        efficiency_score_by_design = np.random.rand(n)  # non-constant reference
        dims_for_embedding_dict = {"embedding1": 5, "original": 10}

        eval_df = eval.create_evaluation_df(
            efficiency_scores_dict,
            efficiency_score_by_design,
            dims_for_embedding_dict,
        )

        row = eval_df[eval_df["dim_reduction_level"] == "embedding1"].iloc[0]
        # Spearman and Kendall should be NaN because scores are constant
        assert np.isnan(row["spearmanr"])
        assert row["spearmanr_warning"]
        assert np.isnan(row["kendalltau"])
        assert row["kendalltau_warning"]
        # MAE is still computable (difference of non-constant reference)
        assert not np.isnan(row["mae"])
        # All DMUs are efficient
        assert row["nr_efficient"] == 50
        assert row["prop_efficient"] == pytest.approx(1.0)


class TestGetEfficiencySummary:
    """Test get_efficiency_summary function."""

    def test_basic_structure(self):
        """Smoke test for structure."""
        scores_dict = {
            "algo_a": np.array([1.0, 0.5, 1.0, 0.8, np.nan]),
            "algo_b": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        }
        df = eval.get_efficiency_summary(scores_dict)
        assert isinstance(df, pd.DataFrame)
        assert set(df.index) == {"algo_a", "algo_b"}
        assert set(df.columns) == {"nr_efficient", "nr_non_nan", "prop_efficient"}
        # algo_b has no efficient DMUs
        assert df.loc["algo_b", "nr_efficient"] == 0
        assert df.loc["algo_b", "prop_efficient"] == pytest.approx(0.0)
        # algo_a has 2 efficient out of 4 non-NaN
        assert df.loc["algo_a", "nr_efficient"] == 2
        assert df.loc["algo_a", "nr_non_nan"] == 4
        assert df.loc["algo_a", "prop_efficient"] == pytest.approx(0.5)
