import pytest
import numpy as np
from src import dgp


class TestGenerateCoefficients:
    """Test generate_coefficients function."""

    def test_coefficients_structure(self):
        """Test that coefficients are generated with correct structure."""
        N, M, alpha_1 = 10, 1, 0.25
        alpha, beta = dgp.generate_coefficients(N, M, alpha_1, verbose=False)

        assert isinstance(alpha, np.ndarray)
        assert isinstance(beta, np.ndarray)
        assert alpha.shape == (N,)
        assert beta.shape == (M,)

    def test_coefficients_sum_to_one(self):
        """Test that coefficients sum to 1."""
        N, M, alpha_1 = 10, 1, 0.25
        alpha, beta = dgp.generate_coefficients(N, M, alpha_1, verbose=False)

        np.testing.assert_almost_equal(np.sum(alpha), 1.0)
        np.testing.assert_almost_equal(np.sum(beta), 1.0)

    def test_alpha_first_element(self):
        """Test that alpha[0] equals alpha_1."""
        N, M, alpha_1 = 10, 1, 0.25
        alpha, beta = dgp.generate_coefficients(N, M, alpha_1, verbose=False)

        np.testing.assert_almost_equal(alpha[0], alpha_1)

    def test_coefficients_positive(self):
        """Test that all coefficients are positive."""
        N, M, alpha_1 = 10, 1, 0.25
        alpha, beta = dgp.generate_coefficients(N, M, alpha_1, verbose=False)

        assert np.all(alpha > 0)
        assert np.all(beta > 0)

    def test_different_N_values(self):
        """Test with different N values."""
        alpha_1 = 0.25
        for N in [5, 10, 20]:
            alpha, _ = dgp.generate_coefficients(N, 1, alpha_1, verbose=False)
            assert alpha.shape == (N,)
            np.testing.assert_almost_equal(np.sum(alpha), 1.0)


class TestGenerateEfficientOutputs:
    """Test generate_efficient_outputs function."""

    def test_efficient_outputs_shape(self):
        """Test that efficient outputs have correct shape."""
        n, M = 50, 1
        y_tilde = dgp.generate_efficient_outputs(n, M, verbose=False)

        assert y_tilde.shape == (n, M)

    def test_efficient_outputs_range(self):
        """Test that efficient outputs are in expected range."""
        n, M = 50, 1
        y_tilde = dgp.generate_efficient_outputs(n, M, verbose=False)

        assert np.all(y_tilde >= 0.1)
        assert np.all(y_tilde <= 1.0)

    def test_multiple_outputs(self):
        """Test generating multiple outputs."""
        n, M = 50, 3
        y_tilde = dgp.generate_efficient_outputs(n, M, verbose=False)

        assert y_tilde.shape == (n, M)


class TestGenerateAllButOneInput:
    """Test generate_all_but_one_input function."""

    def test_input_generation_shape(self):
        """Test that inputs have correct shape."""
        n, N = 50, 10
        x = dgp.generate_all_but_one_input(n, N, verbose=False)

        assert x.shape == (n, N)

    def test_input_generation_range(self):
        """Test that inputs are in expected range."""
        n, N = 50, 10
        x = dgp.generate_all_but_one_input(n, N, verbose=False)

        assert np.all(x >= 0.1)
        assert np.all(x <= 1.0)


class TestGenerateDataDict:
    """Test generate_data_dict function."""

    def test_data_dict_keys(self):
        """Test that generated data has all expected keys."""
        data_dict = dgp.generate_data_dict(
            n=50, N=10, M=1, alpha_1=0.25, gamma=1.0, sigma_u=0.1, verbose=False
        )

        expected_keys = {"x", "y", "y_tilde", "alpha", "beta"}
        assert set(data_dict.keys()) == expected_keys

    def test_data_dict_shapes(self):
        """Test that data has correct shapes."""
        n, N, M = 50, 10, 1
        data_dict = dgp.generate_data_dict(
            n=n, N=N, M=M, alpha_1=0.25, gamma=1.0, sigma_u=0.1, verbose=False
        )

        assert data_dict["x"].shape == (n, N)
        assert data_dict["y"].shape == (n, M)
        assert data_dict["y_tilde"].shape == (n, M)
        assert data_dict["alpha"].shape == (N,)
        assert data_dict["beta"].shape == (M,)

    def test_data_dict_non_negative(self):
        """Test that all data values are non-negative."""
        data_dict = dgp.generate_data_dict(
            n=50, N=10, M=1, alpha_1=0.25, gamma=1.0, sigma_u=0.1, verbose=False
        )

        assert np.all(data_dict["x"] >= 0)
        assert np.all(data_dict["y"] >= 0)
        assert np.all(data_dict["y_tilde"] >= 0)

    def test_y_less_than_y_tilde(self):
        """Test that y <= y_tilde (inefficiency model)."""
        data_dict = dgp.generate_data_dict(
            n=50, N=10, M=1, alpha_1=0.25, gamma=1.0, sigma_u=0.1, verbose=False
        )

        assert np.all(data_dict["y"] <= data_dict["y_tilde"])

    def test_reproducibility_with_seed(self):
        """Test that data generation is reproducible with same seed."""
        np.random.seed(42)
        data1 = dgp.generate_data_dict(
            n=50, N=10, M=1, alpha_1=0.25, gamma=1.0, sigma_u=0.1, verbose=False
        )

        np.random.seed(42)
        data2 = dgp.generate_data_dict(
            n=50, N=10, M=1, alpha_1=0.25, gamma=1.0, sigma_u=0.1, verbose=False
        )

        for key in data1.keys():
            np.testing.assert_array_almost_equal(data1[key], data2[key])
