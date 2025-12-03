"""
Chem-Gym package scaffold for surrogate-assisted active RL on HEA surfaces.
"""

from .config import EnvConfig, TrainConfig, SurrogateConfig
from .envs.chem_env import ChemGymEnv
from .surrogate.ensemble import SurrogateEnsemble

__all__ = [
    "ChemGymEnv",
    "SurrogateEnsemble",
    "EnvConfig",
    "TrainConfig",
    "SurrogateConfig",
]
