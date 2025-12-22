"""
Centralized configuration dataclasses used across the Chem-Gym scaffold.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EnvConfig:
    mode: str = "image"  # "image" or "graph"
    element_types: List[str] = field(default_factory=lambda: [ "Au", "Ag","Cu"])
    slab_size: Tuple[int, int] = (4, 4) 
    n_layers: int = 4                   # Slab 总厚度层数
    n_active_layers: int = 3            # 允许优化的顶部层数 (Active Region)
    graph_cutoff: float = 6.0           # 图的截断半径（Å）
    graph_sigma: float = 2.0            # RBF 权重的尺度（Å）
    max_steps: int = 200                # 适应更大的搜索空间
    step_penalty: float = 0.01
    init_seed: Optional[int] = None
    render_mode: Optional[str] = None  # "human" or "rgb_array"



@dataclass
class TrainConfig:
    total_timesteps: int = 200_000
    n_envs: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    device: str = "cpu"
    uncertainty_penalty: float = 0.0
    oracle_threshold: Optional[float] = None
    oracle_fmax: float = 0.05
    oracle_max_steps: int = 100
    oracle_disable_amp: bool = True


@dataclass
class SurrogateConfig:
    n_models: int = 3
    seeds: Optional[List[int]] = None  # None -> range(n_models)
    mean_energy: float = -1.0
    noise_scale: float = 0.1
