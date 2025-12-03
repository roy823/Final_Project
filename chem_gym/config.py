"""
Centralized configuration dataclasses used across the Chem-Gym scaffold.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EnvConfig:
    mode: str = "image"  # "image" or "graph"
    element_types: List[str] = field(default_factory=lambda: ["Cu", "Ni", "Pt", "Pd"])
    slab_size: Tuple[int, int] = (4, 4)  # x, y
    max_steps: int = 64
    step_penalty: float = 0.01
    init_seed: Optional[int] = None
    render_mode: Optional[str] = None  # "human" or "rgb_array"


@dataclass
class TrainConfig:
    total_timesteps: int = 10_000
    n_envs: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    device: str = "cpu"
    uncertainty_penalty: float = 0.0
    oracle_threshold: Optional[float] = None


@dataclass
class SurrogateConfig:
    n_models: int = 3
    seeds: Optional[List[int]] = None  # None -> range(n_models)
    mean_energy: float = -1.0
    noise_scale: float = 0.1
