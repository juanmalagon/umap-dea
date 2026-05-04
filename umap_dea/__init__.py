"""Top-level package for UMAP-DEA.

This package provides utilities for:
- synthetic data generation,
- dimensionality reduction with UMAP or PCA,
- DEA efficiency estimation,
- evaluation of estimated efficiencies.
"""

from . import dea, dgp, dim_red, eval
from .config import SimulationConfig

__all__ = [
    "SimulationConfig",
    "dea",
    "dgp",
    "dim_red",
    "eval",
]
