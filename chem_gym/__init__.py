"""
Chem-Gym package scaffold for surrogate-assisted active RL on HEA surfaces.
"""

from .config import EnvConfig, TrainConfig, SurrogateConfig, AdsorptionConfig
from .envs.chem_env import ChemGymEnv
from .envs.adsorption_env import AdsorptionChemGymEnv
from .surrogate.ensemble import SurrogateEnsemble

__all__ = [
    "ChemGymEnv",
    "AdsorptionChemGymEnv",
    "SurrogateEnsemble",
    "EnvConfig",
    "TrainConfig",
    "SurrogateConfig",
    "AdsorptionConfig",
]
