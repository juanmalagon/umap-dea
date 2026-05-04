import pytest
import numpy as np
from src.config import SimulationConfig


@pytest.fixture(scope="session")
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture
def valid_config_dict():
    """Provide a valid configuration dictionary."""
    return {
        "N": 10,
        "M": 1,
        "n": 50,
        "alpha_1": 0.25,
        "gamma": 1.0,
        "sigma_u": 0.1,
        "rts": "crs",
        "orientation": "input",
        "nr_simulations": 10,
        "seed": 42,
        "pca": False,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "umap_metric": "euclidean",
    }


@pytest.fixture
def valid_config(valid_config_dict):
    """Provide a valid SimulationConfig instance."""
    return SimulationConfig(**valid_config_dict)


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 10
    x = np.random.rand(n_samples, n_features) * 0.9 + 0.1
    y = np.random.rand(n_samples, 1) * 0.9 + 0.1
    return {"x": x, "y": y}


@pytest.fixture
def efficiency_scores():
    """Provide sample efficiency scores."""
    np.random.seed(42)
    return np.random.rand(50)
