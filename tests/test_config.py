import pytest
from umap_dea.config import SimulationConfig


class TestSimulationConfigCreation:
    """Test SimulationConfig creation and validation."""

    def test_valid_config_creation(self, valid_config_dict):
        """Test creating a valid configuration."""
        config = SimulationConfig(**valid_config_dict)
        assert config.N == 10
        assert config.M == 1
        assert config.n == 50
        assert config.alpha_1 == 0.25
        assert config.gamma == 1.0
        assert config.sigma_u == 0.1
        assert config.rts == "crs"
        assert config.orientation == "input"
        assert config.nr_simulations == 10
        assert config.seed == 42
        assert config.pca is False

    def test_umap_hyperparameters(self, valid_config):
        """Test that UMAP hyperparameters are properly set."""
        assert valid_config.umap_n_neighbors == 15
        assert valid_config.umap_min_dist == 0.1
        assert valid_config.umap_metric == "euclidean"

    def test_default_values(self):
        """Test that default values are applied correctly."""
        config = SimulationConfig(N=10, M=1, n=50, alpha_1=0.25, gamma=1.0, sigma_u=0.1)
        assert config.rts == "crs"
        assert config.orientation == "input"
        assert config.nr_simulations == 1000
        assert config.seed == 42
        assert config.pca is False
        assert config.umap_n_neighbors == 15
        assert config.umap_min_dist == 0.1
        assert config.umap_metric == "euclidean"

    def test_alpha_1_as_float(self, valid_config):
        """Test alpha_1 as a float value."""
        assert isinstance(valid_config.alpha_1, float)
        assert 0 <= valid_config.alpha_1 <= 1

    def test_alpha_1_as_string(self):
        """Test alpha_1 as string '1/N' resolves to a float."""
        config = SimulationConfig(
            N=10,
            M=1,
            n=50,
            alpha_1="1/N",
            gamma=1.0,
            sigma_u=0.1,
        )
        assert isinstance(config.alpha_1, float)
        assert config.alpha_1 == pytest.approx(0.1)


class TestSimulationConfigValidation:
    """Test SimulationConfig validation."""

    def test_invalid_alpha_1_type(self):
        """Test that invalid alpha_1 type raises TypeError."""
        with pytest.raises(TypeError, match="alpha_1 must be float or string"):
            SimulationConfig(
                N=10,
                M=1,
                n=50,
                alpha_1=[0.25],  # Invalid: list instead of float or string
                gamma=1.0,
                sigma_u=0.1,
            )

    def test_invalid_alpha_1_string(self):
        """Test that invalid alpha_1 string raises ValueError."""
        with pytest.raises(ValueError, match="alpha_1 string value must be '1/N'"):
            SimulationConfig(
                N=10,
                M=1,
                n=50,
                alpha_1="invalid",
                gamma=1.0,
                sigma_u=0.1,
            )

    def test_invalid_rts(self):
        """Test that invalid rts value raises ValueError."""
        with pytest.raises(ValueError, match="rts must be 'crs' or 'vrs'"):
            SimulationConfig(
                N=10,
                M=1,
                n=50,
                alpha_1=0.25,
                gamma=1.0,
                sigma_u=0.1,
                rts="invalid",
            )

    def test_invalid_orientation(self):
        """Test that invalid orientation raises ValueError."""
        with pytest.raises(ValueError, match="orientation must be 'input' or 'output'"):
            SimulationConfig(
                N=10,
                M=1,
                n=50,
                alpha_1=0.25,
                gamma=1.0,
                sigma_u=0.1,
                orientation="invalid",
            )

    def test_valid_vrs(self):
        """Test valid VRS configuration."""
        config = SimulationConfig(
            N=10,
            M=1,
            n=50,
            alpha_1=0.25,
            gamma=1.0,
            sigma_u=0.1,
            rts="vrs",
        )
        config.validate()  # Should not raise

    def test_valid_output_orientation(self):
        """Test valid output orientation."""
        config = SimulationConfig(
            N=10,
            M=1,
            n=50,
            alpha_1=0.25,
            gamma=1.0,
            sigma_u=0.1,
            orientation="output",
        )
        config.validate()  # Should not raise


class TestSimulationConfigConversion:
    """Test converting config to dictionary."""

    def test_config_to_dict(self, valid_config):
        """Test converting config to dictionary."""
        config_dict = valid_config.__dict__
        assert isinstance(config_dict, dict)
        assert "N" in config_dict
        assert "M" in config_dict
        assert "n" in config_dict
        assert "umap_n_neighbors" in config_dict
