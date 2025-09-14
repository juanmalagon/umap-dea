from dataclasses import dataclass
from typing import Union


@dataclass
class SimulationConfig:
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

    def validate(self):
        """Validate configuration parameters"""
        if not isinstance(self.alpha_1, (float, str)):
            raise TypeError("alpha_1 must be float or string '1/N'")
        if self.rts not in ('crs', 'vrs'):
            raise ValueError("rts must be 'crs' or 'vrs'")
        if self.orientation not in ('input', 'output'):
            raise ValueError("orientation must be 'input' or 'output'")
