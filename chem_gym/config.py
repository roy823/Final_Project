"""
Centralized configuration dataclasses used across the Chem-Gym scaffold.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EnvConfig:
    mode: str = "image"  # "image" or "graph"
    element_types: List[str] = field(default_factory=lambda: [ "Pt", "Ag"])
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


@dataclass
class AdsorptionConfig:
    """吸附能优化专用配置"""
    # 吸附剂参数
    adsorbate: str = "CO"  # CO, O2, H2, NO
    target_ads_energy: float = -0.5  # 目标吸附能(eV)
    energy_tolerance: float = 0.1    # 允许偏差

    # 物理参数
    gas_reference_energy: float = -1.0  # 气相参考能
    adsorbate_height: float = 2.0       # 初始吸附高度(Å)
    adsorption_site: str = "fcc"        # 吸附位点类型: fcc, hcp, bridge

    # Oracle配置
    use_oracle_directly: bool = True    # 直接使用EquiformerV2
    cache_enabled: bool = True          # 启用吸附能缓存
    enable_relaxation: bool = True      # 启用Oracle自动弛豫

    # 奖励权重
    target_reward_weight: float = 10.0
    proximity_reward_weight: float = 1.0
