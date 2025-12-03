from typing import Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required for ChemGymEnv") from exc

try:
    from ase.build import fcc111
    from ase.visualize.plot import plot_atoms
    from ase import Atoms
except ImportError:  # pragma: no cover
    fcc111 = None
    plot_atoms = None
    Atoms = None

from chem_gym.config import EnvConfig


class ChemGymEnv(gym.Env):
    """
    Gymnasium environment for swapping atoms on an HEA slab.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: EnvConfig, surrogate=None):
        super().__init__()
        self.config = config
        self.surrogate = surrogate
        self.rng = np.random.default_rng(config.init_seed)

        self.element_types = config.element_types
        self.n_elements = len(self.element_types)
        self.n_sites = config.slab_size[0] * config.slab_size[1]
        self.render_mode = config.render_mode

        # 动作空间：任意两点交换
        self.action_space = spaces.Discrete(self.n_sites * (self.n_sites - 1) // 2)
        
        # 观察空间
        if self.config.mode == "image":
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.slab_size[0], config.slab_size[1], self.n_elements),
                dtype=np.float32,
            )
        elif self.config.mode == "graph":
            self.observation_space = spaces.Dict(
                {
                    "node_features": spaces.Box(
                        low=0.0, high=1.0, shape=(self.n_sites, self.n_elements), dtype=np.float32
                    ),
                    "adjacency": spaces.Box(
                        low=0.0, high=1.0, shape=(self.n_sites, self.n_sites), dtype=np.float32
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        self.state = None
        self.atoms = None
        
        # 状态追踪变量
        self.initial_energy = 0.0
        self.prev_energy = 0.0  # 新增：用于计算差分奖励
        self.current_energy = 0.0
        self.current_uncertainty = 0.0
        self.steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # --- FIX: 固定成分，随机洗牌 (Stoichiometry Control) ---
        # 确保每种元素的数量尽可能相等，而不是随机抽取
        n_base = self.n_sites // self.n_elements
        remainder = self.n_sites % self.n_elements
        
        base_state = []
        for i in range(self.n_elements):
            count = n_base + (1 if i < remainder else 0)
            base_state.extend([i] * count)
            
        self.state = np.array(base_state, dtype=np.int32)
        self.rng.shuffle(self.state) # 仅打乱位置
        # ---------------------------------------------------

        self.steps = 0
        self.atoms = self._build_atoms_from_state()

        self.initial_energy, self.current_uncertainty = self._evaluate_energy(self.atoms)
        self.current_energy = self.initial_energy
        self.prev_energy = self.initial_energy # 初始化上一轮能量

        observation = self._state_to_observation()
        info = {
            "energy": self.current_energy, 
            "uncertainty": self.current_uncertainty, 
            "atoms": self.atoms
        }
        return observation, info

    def step(self, action: int):
        i, j = self._action_to_indices(action)
        
        # 执行交换
        self.state[i], self.state[j] = self.state[j], self.state[i]
        self.steps += 1

        # 更新物理状态
        self.atoms = self._build_atoms_from_state()
        self.current_energy, self.current_uncertainty = self._evaluate_energy(self.atoms)

        # --- FIX: 差分奖励 (Differential Reward) ---
        # 奖励 = 能量降低量 = 上一步能量 - 当前能量
        energy_diff = self.prev_energy - self.current_energy
        reward = energy_diff - self.config.step_penalty
        
        # 更新历史
        self.prev_energy = self.current_energy
        # -------------------------------------------

        terminated = False
        truncated = self.steps >= self.config.max_steps

        observation = self._state_to_observation()
        info = {
            "energy": self.current_energy,
            "uncertainty": self.current_uncertainty,
            "swapped_sites": (i, j),
            "energy_improvement": self.initial_energy - self.current_energy,
            "atoms": self.atoms  # <--- [关键修复] 必须传递 atoms 给 OracleWrapper 使用
        }
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        if plot_atoms is None or self.atoms is None:
            return np.reshape(self.state, self.config.slab_size)

        fig = plot_atoms(self.atoms, show_unit_cell=0, rotation='-90x')
        if self.render_mode == "human":
            return fig
        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            return data.reshape((h, w, 3))
        return None

    # --- 内部辅助函数 ---
    def _action_to_indices(self, action: int) -> Tuple[int, int]:
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Action {action} out of bounds")
        remaining = action
        for i in range(self.n_sites - 1):
            span = self.n_sites - i - 1
            if remaining < span:
                return i, i + 1 + remaining
            remaining -= span
        raise ValueError(f"Action {action} could not be decoded")

    def _state_to_observation(self):
        if self.config.mode == "image":
            flat_one_hot = np.eye(self.n_elements, dtype=np.float32)[self.state]
            grid = flat_one_hot.reshape(self.config.slab_size[0], self.config.slab_size[1], self.n_elements)
            return grid

        node_features = np.eye(self.n_elements, dtype=np.float32)[self.state]
        adjacency = np.ones((self.n_sites, self.n_sites), dtype=np.float32) - np.eye(self.n_sites, dtype=np.float32)
        return {"node_features": node_features, "adjacency": adjacency}

    def _build_atoms_from_state(self):
        if fcc111 is None:
            return None
        
        if self.atoms is None:
             self.atoms = fcc111("Cu", size=(self.config.slab_size[0], self.config.slab_size[1], 3), a=3.6, vacuum=7.0)
        
        symbols = self.atoms.get_chemical_symbols()
        # 仅更新最上层表面原子
        start_idx = len(symbols) - self.n_sites
        new_surface_symbols = [self.element_types[idx] for idx in self.state]
        
        for k, sym in enumerate(new_surface_symbols):
            self.atoms[start_idx + k].symbol = sym
        return self.atoms

    def _evaluate_energy(self, atoms: Optional["Atoms"]):
        if self.surrogate is None:
            # 如果没有代理模型，返回一个伪能量（仅用于测试）
            pseudo_energy = float(np.mean(self.state)) 
            return pseudo_energy, 0.0
        return self.surrogate.evaluate(atoms)