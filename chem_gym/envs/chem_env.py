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
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from ase.constraints import FixAtoms
except ImportError:  # pragma: no cover
    fcc111 = None
    plot_atoms = None
    Atoms = None
    EMT = None
    BFGS = None
    FixAtoms = None

from chem_gym.config import EnvConfig


class ChemGymEnv(gym.Env):
    """
    Gymnasium environment for swapping atoms on an HEA slab.
    Integreated with ASE EMT calculator for physical realism.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    # 来源: Materials Project / Kittel / ICSD
    LATTICE_CONSTANTS = {
        "Pt": 3.92,
        "Pd": 3.89,
        "Cu": 3.61,
        "Ni": 3.52,
        "Au": 4.08, # 预留扩展
        "Ag": 4.09  # 预留
    }

    # def __init__(self, config: EnvConfig, surrogate=None):
    #     super().__init__()
    #     self.config = config
    #     self.surrogate = surrogate
    #     self.rng = np.random.default_rng(config.init_seed)

    #     self.element_types = config.element_types
    #     self.n_elements = len(self.element_types)
    #     self.n_sites = config.slab_size[0] * config.slab_size[1]
    #     self.render_mode = config.render_mode

    #     # 动作空间：任意两点交换
    #     self.action_space = spaces.Discrete(self.n_sites * (self.n_sites - 1) // 2)
        
    #     # 观察空间
    #     if self.config.mode == "image":
    #         self.observation_space = spaces.Box(
    #             low=0.0,
    #             high=1.0,
    #             shape=(config.slab_size[0], config.slab_size[1], self.n_elements),
    #             dtype=np.float32,
    #         )
    #     elif self.config.mode == "graph":
    #         self.observation_space = spaces.Dict(
    #             {
    #                 "node_features": spaces.Box(
    #                     low=0.0, high=1.0, shape=(self.n_sites, self.n_elements), dtype=np.float32
    #                 ),
    #                 "adjacency": spaces.Box(
    #                     low=0.0, high=1.0, shape=(self.n_sites, self.n_sites), dtype=np.float32
    #                 ),
    #             }
    #         )
    #     else:
    #         raise ValueError(f"Unsupported mode: {self.config.mode}")
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
        
        # --- [修改开始]：观察空间升级为多层切片 ---
        # 假设我们观察 Slab 的 4 层原子 (与 _build_atoms_from_state 中的 size=(x,y,4) 对应)
        self.n_layers = 4 
        
        if self.config.mode == "image":
            # 形状：(H, W, Channels)
            # Channels = 元素种类 * 层数。例如 5种元素 * 4层 = 20 个通道
            # 这种结构非常适合 CNN 处理，能同时感知空间位置和深度信息
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.slab_size[0], config.slab_size[1], self.n_elements * self.n_layers),
                dtype=np.float32,
            )
        # ----------------------------------------
        elif self.config.mode == "graph":
            # Graph 模式保持不变，由你同学负责
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
        self.prev_energy = 0.0
        self.current_energy = 0.0
        self.current_uncertainty = 0.0
        self.steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # --- Stoichiometry Control ---
        # 确保每种元素的数量尽可能相等
        n_base = self.n_sites // self.n_elements
        remainder = self.n_sites % self.n_elements
        
        base_state = []
        for i in range(self.n_elements):
            count = n_base + (1 if i < remainder else 0)
            base_state.extend([i] * count)
            
        self.state = np.array(base_state, dtype=np.int32)
        self.rng.shuffle(self.state) # 仅打乱位置
        # -----------------------------

        self.steps = 0
        # 重置 atoms 为 None，强制 _build_atoms_from_state 重新计算晶格常数并建模
        self.atoms = None 
        self.atoms = self._build_atoms_from_state()

        self.initial_energy, self.current_uncertainty = self._evaluate_energy(self.atoms)
        self.current_energy = self.initial_energy
        self.prev_energy = self.initial_energy

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

        # --- 奖励函数优化 ---
        # 1. 差分奖励: 能量降低量
        energy_diff = self.prev_energy - self.current_energy
        # 2. 缩放奖励: 放大 10 倍，使梯度更显著
        # 3. 步数惩罚: 鼓励尽快找到最优解
        reward = (energy_diff * 10.0) - self.config.step_penalty
        
        # 更新历史
        self.prev_energy = self.current_energy
        # --------------------

        terminated = False
        truncated = self.steps >= self.config.max_steps
        
        # 可选：安全截断，如果能量异常爆炸（>5.0 eV/atom 是极其不正常的），提前结束
        if self.current_energy > 5.0:
            reward -= 10.0 # 给予重罚
            terminated = True

        observation = self._state_to_observation()
        info = {
            "energy": self.current_energy,
            "uncertainty": self.current_uncertainty,
            "swapped_sites": (i, j),
            "energy_improvement": self.initial_energy - self.current_energy,
            "atoms": self.atoms
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

    # def _state_to_observation(self):
    #     if self.config.mode == "image":
    #         flat_one_hot = np.eye(self.n_elements, dtype=np.float32)[self.state]
    #         grid = flat_one_hot.reshape(self.config.slab_size[0], self.config.slab_size[1], self.n_elements)
    #         return grid

    #     node_features = np.eye(self.n_elements, dtype=np.float32)[self.state]
    #     adjacency = np.ones((self.n_sites, self.n_sites), dtype=np.float32) - np.eye(self.n_sites, dtype=np.float32)
    #     return {"node_features": node_features, "adjacency": adjacency}
    
    def _state_to_observation(self):
        # --- [修改开始]：生成多层切片观测 ---
        if self.config.mode == "image":
            # 我们需要获取整个 Slab (包括表面和内部) 的原子分布
            # 在 ASE 的 fcc111 构建中，原子通常是按层排列的
            # Layer 0 (底部) -> ... -> Layer 3 (表面)
            
            # 1. 获取所有原子的元素索引
            # 如果 atoms 还没构建，就先用 state 构建个临时的
            if self.atoms is None:
                self._build_atoms_from_state()
            
            # 获取所有原子的化学符号
            all_symbols = self.atoms.get_chemical_symbols()
            
            # 将符号转换为索引 (比如 'Cu'->0, 'Pt'->2)
            # 注意：底层原子可能是你在 fcc111 里定义的默认 'Cu'
            type_map = {ele: i for i, ele in enumerate(self.element_types)}
            
            # 2. 构建多层 Grid
            # 形状: (Layers, H, W, Elements)
            layers_grid = np.zeros(
                (self.n_layers, self.config.slab_size[0], self.config.slab_size[1], self.n_elements),
                dtype=np.float32
            )
            
            # 填充数据
            # 假设 atoms 列表顺序是：第0层(0~15), 第1层(16~31), ..., 第3层(表面)
            for l in range(self.n_layers):
                start = l * self.n_sites
                end = start + self.n_sites
                # 获取这一层的原子 (如果 atoms 数量不够，就填 0)
                if start < len(all_symbols):
                    layer_syms = all_symbols[start:end]
                    for idx, sym in enumerate(layer_syms):
                        # 计算在 grid 中的 (x, y) 坐标
                        # ASE fcc111 排列通常是行优先或列优先，这里假设是一一对应
                        row = idx // self.config.slab_size[1]
                        col = idx % self.config.slab_size[1]
                        
                        if sym in type_map:
                            ele_idx = type_map[sym]
                            layers_grid[l, row, col, ele_idx] = 1.0
                        else:
                            # 如果遇到了不在 element_types 里的元素（比如底部的基底元素），可以忽略或归为某类
                            pass

            # 3. 展平层维度到通道维度 (Flatten Layers to Channels)
            # (Layers, H, W, E) -> (H, W, Layers * E)
            # 例如：前5个通道是第0层，接着5个通道是第1层... 最后5个通道是表面层
            obs = layers_grid.transpose(1, 2, 0, 3).reshape(
                self.config.slab_size[0], 
                self.config.slab_size[1], 
                -1 # 自动计算为 n_layers * n_elements
            )
            
            return obs

    def _build_atoms_from_state(self):
        """
        根据当前状态构建 ASE Atoms 对象。
        使用 Vegard 定律动态计算平均晶格常数，避免初始应力过大。
        """
        if fcc111 is None:
            return None
        
        # 1. 如果 atoms 尚未初始化，创建它
        if self.atoms is None:
            # [关键] Vegard's Law: 计算当前成分的加权平均晶格常数
            current_elements = [self.element_types[i] for i in self.state]
            
            # 从 LATTICE_CONSTANTS 查表，默认 3.7
            avg_lattice_constant = float(np.mean([
                self.LATTICE_CONSTANTS.get(el, 3.7) for el in current_elements
            ]))
            
            # 使用计算出的 a 构建底板
            # size=(x, y, 4) 表示 4 层厚度，通常底部 2 层固定用于模拟体相
            self.atoms = fcc111(
                "Cu", # 这里的 'Cu' 只是占位符，后面会替换符号
                size=(self.config.slab_size[0], self.config.slab_size[1], 4), 
                a=avg_lattice_constant, 
                vacuum=10.0
            )

            # [关键] 添加约束：固定底部 2 层原子，防止板子飘动
            if FixAtoms is not None:
                n_total = len(self.atoms)
                n_surface = self.n_sites  # 最表面一层的原子数 (slab_size x * y)
                
                # 固定除了最上层以外的所有原子
                n_fixed = n_total - n_surface
                constraint = FixAtoms(indices=range(n_fixed))
                self.atoms.set_constraint(constraint)
        
        # 2. 更新表面原子的化学符号
        symbols = self.atoms.get_chemical_symbols()
        # ASE 的 fcc111 构建中，表面原子通常在列表末尾
        start_idx = len(symbols) - self.n_sites
        new_surface_symbols = [self.element_types[idx] for idx in self.state]
        
        # 批量更新符号
        for k, sym in enumerate(new_surface_symbols):
            self.atoms[start_idx + k].symbol = sym
            
        return self.atoms

    def _evaluate_energy(self, atoms: Optional["Atoms"]):
        """
        计算当前构型的能量。
        策略：Surrogate (优先) -> EMT + Relaxation (物理基线)。
        """
        if atoms is None:
            return 0.0, 0.0

        # 1. 优先使用代理模型 (AI Surrogate)
        if self.surrogate is not None:
            # 注意：如果 surrogate 返回的是随机数，请确保在训练早期将其关闭
            # 或者在 ensemble.py 中实现逻辑，当模型为空时调用 _evaluate_physics
            return self.surrogate.evaluate(atoms)
        
        # 2. 如果没有代理，使用 ASE 内置 EMT 计算器 (Ground Truth Baseline)
        if EMT is not None:
            try:
                # [关键] 在副本上进行计算和松弛，不破坏环境的主状态 (atoms)的位置
                # 这样可以保持晶格的整齐，只获取能量值
                calc_atoms = atoms.copy()
                calc_atoms.calc = EMT()
                
                # [关键] 局部结构优化 (Pre-relaxation / Soft Minimization)
                # 只跑 10 步，消除原子重叠造成的巨大虚假能量
                if BFGS is not None:
                    # logfile=None 静默模式，不输出冗长的优化日志
                    dyn = BFGS(calc_atoms, logfile=None) 
                    dyn.run(fmax=0.5, steps=10)
                
                potential_energy = calc_atoms.get_potential_energy()
                
                # 归一化能量 (eV/atom) 以利于 RL 训练收敛
                energy_per_atom = potential_energy / len(calc_atoms)
                
                return energy_per_atom, 0.0  # EMT 不确定度为 0
            except Exception as e:
                # 容错处理，防止偶尔的物理计算崩溃中断训练
                print(f"Warning: EMT calculation failed: {e}")
                return 0.0, 0.0

        return 0.0, 0.0