from dataclasses import dataclass
from typing import Union


@dataclass
class SimulationConfig:
    """Configuration for a UMAP-DEA simulation run."""

    N: int
    M: int
    n: int
    alpha_1: Union[float, str]
    gamma: float
    sigma_u: float
    rts: str = 'crs'
    orientation: str = 'input'
    nr_simulations: int = 1000
    seed: int = 42
    pca: bool = False
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = 'euclidean'

    def __post_init__(self) -> None:
        """Validate configuration immediately after initialization."""
        self.validate()
        self.alpha_1 = self._resolve_alpha_1()

    def _resolve_alpha_1(self) -> float:
        """Resolve alpha_1 to a numeric value for downstream computation."""
        if isinstance(self.alpha_1, str):
            normalized = self.alpha_1.replace(' ', '').lower()
            if normalized != '1/n':
                raise ValueError("alpha_1 string value must be '1/N'")
            if self.N <= 0:
                raise ValueError("N must be positive when alpha_1 is '1/N'")
            return 1.0 / self.N

        return float(self.alpha_1)

    def validate(self) -> None:
        """Validate configuration parameter values."""
        if not isinstance(self.alpha_1, (float, str)):
            raise TypeError("alpha_1 must be float or string '1/N'")
        if self.rts not in ('crs', 'vrs'):
            raise ValueError("rts must be 'crs' or 'vrs'")
        if self.orientation not in ('input', 'output'):
            raise ValueError("orientation must be 'input' or 'output'")
