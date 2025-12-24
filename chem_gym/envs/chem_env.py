from typing import Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required for ChemGymEnv") from exc

try:
    from ase.build import fcc111, bulk
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
    Gymnasium environment for optimizing High-Entropy Alloy (HEA) surfaces.
    
    Scientific Improvements:
    - Active Region: Optimizes top N layers (not just surface) to capture Ligand & Strain effects.
    - Layer-aware Graph: Node features include layer depth information.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    # 来源: Materials Project / Kittel / ICSD
    LATTICE_CONSTANTS = {
        "Pt": 3.92,
        "Pd": 3.89,
        "Cu": 3.61,
        "Ni": 3.52,
        "Au": 4.08,
        "Ag": 4.09
    }

    def __init__(self, config: EnvConfig, surrogate=None, oracle=None):
        super().__init__()
        self.config = config
        self.surrogate = surrogate
        self.oracle = oracle

        self.rng = np.random.default_rng(config.init_seed)

        self.element_types = config.element_types
        self.n_elements = len(self.element_types)
        
        # 几何参数
        self.n_sites_per_layer = config.slab_size[0] * config.slab_size[1]
        self.n_active_layers = config.n_active_layers
        
        # [关键] 活性原子总数 = 每层原子数 * 活性层数
        self.n_active_atoms = self.n_sites_per_layer * self.n_active_layers
        
        self.render_mode = config.render_mode

        # 动作空间：Active Region 内任意两点交换
        # 组合数 C(n, 2) = n * (n-1) / 2
        self.action_space = spaces.Discrete(self.n_active_atoms * (self.n_active_atoms - 1) // 2)
        
        # 观察空间
        if self.config.mode == "image":
            # Image mode 仅支持可视化最表层，对于多层优化建议使用 Graph mode
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.slab_size[0], config.slab_size[1], self.n_elements),
                dtype=np.float32,
            )
        elif self.config.mode == "graph":
            # 总原子数 = 每层原子数 * 总层数
            self.n_total_atoms = self.n_sites_per_layer * self.config.n_layers
            
            # 节点特征维度: 
            # 1. One-Hot Element (N_elements)
            # 2. Relative Pos (3)
            # 3. Layer Index (1) -> [新增] 明确告知网络原子的深度
            self.node_feat_dim = self.n_elements + 3 + 1

            self.observation_space = spaces.Dict(
                {
                    "node_features": spaces.Box(
                        low=-100.0, high=100.0,
                        shape=(self.n_total_atoms, self.node_feat_dim), 
                        dtype=np.float32
                    ),
                    "adjacency": spaces.Box(
                        low=0.0, high=1.0,
                        shape=(self.n_total_atoms, self.n_total_atoms), 
                        dtype=np.float32
                    ),
                    "node_mask": spaces.Box(
                        low=0.0, high=1.0,
                        shape=(self.n_total_atoms,),
                        dtype=np.float32
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        self.state = None
        self.atoms = None
        
        self.initial_energy = 0.0
        self.prev_energy = 0.0
        self.current_energy = 0.0
        self.current_uncertainty = 0.0
        self.min_energy_so_far = 0.0 # [新增] 记录本回合最低能量
        self.steps = 0
        # 参考能量 (校准用)
        self.ref_energies = {}
        if self.oracle is not None:
            self._calibrate_references()

    def _calibrate_references(self):
        """
        使用 UMA 时，参考态修正由内部 FormationEnergyCalculator 处理。
        """
        if isinstance(self.oracle, (object,)) and "UMAOracle" in str(type(self.oracle)):
            print("[ChemGymEnv] UMA Oracle detected. Using internal MP2020 corrections.")
            return
        
        # 原有的手动校准逻辑 (仅在非 UMA 模式下运行)
        print("[ChemGymEnv] Calibrating reference energies with Legacy Oracle...")
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # --- Stoichiometry Control (Active Region) ---
        # 确保 Active Region 内每种元素的数量尽可能相等
        n_base = self.n_active_atoms // self.n_elements
        remainder = self.n_active_atoms % self.n_elements
        
        base_state = []
        for i in range(self.n_elements):
            count = n_base + (1 if i < remainder else 0)
            base_state.extend([i] * count)
            
        self.state = np.array(base_state, dtype=np.int32)
        self.rng.shuffle(self.state)
        # -----------------------------

        self.steps = 0
        self.atoms = None 
        self.atoms = self._build_atoms_from_state()

        self.initial_energy, self.current_uncertainty = self._evaluate_energy(self.atoms)
        self.current_energy = self.initial_energy
        self.prev_energy = self.initial_energy
        self.min_energy_so_far = self.initial_energy # [新增] 初始化最低能量

        observation = self._state_to_observation()
        info = {
            "energy": self.current_energy, 
            "uncertainty": self.current_uncertainty, 
            "atoms": self.atoms
        }
        return observation, info

    def action_masks(self) -> np.ndarray:
        """
        计算动作掩码：如果交换的两个原子元素相同，则该动作为非法 (False)。
        返回: 布尔数组，形状为 (n_actions,)
        """
        mask = np.ones(self.action_space.n, dtype=bool)
        for action in range(self.action_space.n):
            i, j = self._action_to_indices(action)
            if self.state[i] == self.state[j]:
                mask[action] = False
        return mask
        
    def step(self, action: int):
        i, j = self._action_to_indices(action)
        
        # [新增] 检查是否交换了相同的元素 (无效动作)
        # 如果交换前后状态不变，直接给予小惩罚并跳过昂贵的能量计算
        if self.state[i] == self.state[j]:
            reward = -0.5 # 给予小惩罚，避免 Agent 偷懒
            self.steps += 1
            truncated = self.steps >= self.config.max_steps
            
            # 状态未变，无需重新计算能量
            info = {
                "energy": self.current_energy,
                "uncertainty": self.current_uncertainty,
                "swapped_sites": (i, j),
                "energy_improvement": 0.0,
                "atoms": self.atoms
            }
            # 观察值也不变，直接返回当前的观察
            return self._state_to_observation(), reward, False, truncated, info

        # 执行交换 (在 Active Region 内)
        self.state[i], self.state[j] = self.state[j], self.state[i]
        self.steps += 1

        # 更新物理状态
        self.atoms = self._build_atoms_from_state()
        self.current_energy, self.current_uncertainty = self._evaluate_energy(self.atoms)

        # --- 奖励函数 ---
        energy_diff = self.prev_energy - self.current_energy
        # 1. 大幅放大能量差信号 (从 200 增加到 1000)
        # 这样 0.001 eV 的改进就能带来 +1.0 的奖励
        reward = energy_diff * 1000.0
        
        # 2. 移除固定的 step_penalty，改为只对“无改进”的步数进行微小惩罚
        if energy_diff <= 0:
            reward -= 0.01  # 极小的惩罚，迫使它寻找更好的构型
            
        # 3. 引入“历史新低”额外奖励 (Record Breaking Bonus)
        if self.current_energy < self.min_energy_so_far:
            # 只要打破了本局的最低能量记录，就额外给奖励
            bonus = (self.min_energy_so_far - self.current_energy) * 2000.0
            reward += bonus
            self.min_energy_so_far = self.current_energy

        self.prev_energy = self.current_energy

        terminated = False
        truncated = self.steps >= self.config.max_steps
        
        if self.current_energy > 5.0: # 物理异常检测
            reward -= 10.0
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
            return None

        # 渲染时旋转视角以便观察侧面（分层结构）
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
        # 解码动作：将单一整数 action 映射为 (i, j) 交换对
        # 适用于任意大小的 n_active_atoms
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Action {action} out of bounds")
        remaining = action
        # 遍历所有可能的第一个原子 i
        for i in range(self.n_active_atoms - 1):
            # i 之后的原子都可以作为 j
            span = self.n_active_atoms - i - 1
            if remaining < span:
                return i, i + 1 + remaining
            remaining -= span
        raise ValueError(f"Action {action} could not be decoded")

    def _state_to_observation(self):
        if self.config.mode == "image":
            # Image mode 只取最表层 (Active Region 的最后一部分)
            # 注意：state 是扁平的 Active Region，最表层在最后
            surface_state = self.state[-self.n_sites_per_layer:]
            flat_one_hot = np.eye(self.n_elements, dtype=np.float32)[surface_state]
            grid = flat_one_hot.reshape(self.config.slab_size[0], self.config.slab_size[1], self.n_elements)
            return grid
            
        if self.atoms is None:
            return {
                "node_features": np.zeros((self.n_total_atoms, self.node_feat_dim), dtype=np.float32),
                "adjacency": np.zeros((self.n_total_atoms, self.n_total_atoms), dtype=np.float32),
                "node_mask": np.zeros((self.n_total_atoms,), dtype=np.float32)
            }
            
        symbols = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions().astype(np.float32)
        n_total = len(symbols)

        # 1. 节点特征构建
        pos_center = positions.mean(axis=0, keepdims=True)
        rel_pos = positions - pos_center

        node_features = np.zeros((n_total, self.node_feat_dim), dtype=np.float32)
        node_mask = np.ones((n_total,), dtype=np.float32)

        for i, sym in enumerate(symbols):
            # One-Hot Encoding
            one_hot = np.zeros((self.n_elements,), dtype=np.float32)
            if sym in self.element_types:
                one_hot[self.element_types.index(sym)] = 1.0
            
            # [科学改进] Layer Index
            # ASE fcc111 从底向上构建，每层 n_sites_per_layer 个原子
            # Layer 0 是最底层，Layer (n_layers-1) 是最表层
            layer_idx = i // self.n_sites_per_layer
            
            # Concat: [OneHot, RelPos, LayerIndex]
            node_features[i] = np.concatenate([one_hot, rel_pos[i], [float(layer_idx)]])

        # 2. 邻接矩阵构建 (RBF)
        dist = self.atoms.get_all_distances(mic=True).astype(np.float32)
        cutoff = float(self.config.graph_cutoff)
        sigma = float(self.config.graph_sigma)
        if sigma <= 0: sigma = 1.0

        adjacency = np.zeros_like(dist, dtype=np.float32)
        mask = (dist > 0) & (dist <= cutoff)
        adjacency[mask] = np.exp(- (dist[mask] / sigma) ** 2)
        np.fill_diagonal(adjacency, 1.0)

        return {
            "node_features": node_features, 
            "adjacency": adjacency, 
            "node_mask": node_mask
        }

    def _build_atoms_from_state(self):
        """
        根据 Active Region 的状态构建 ASE Atoms 对象。
        """
        if fcc111 is None:
            return None
        
        if self.atoms is None:
            # Vegard's Law 计算平均晶格常数
            current_elements = [self.element_types[i] for i in self.state]
            avg_lattice_constant = float(np.mean([
                self.LATTICE_CONSTANTS.get(el, 3.7) for el in current_elements
            ]))
            
            self.atoms = fcc111(
                "Cu", 
                size=(self.config.slab_size[0], self.config.slab_size[1], self.config.n_layers), 
                a=avg_lattice_constant, 
                vacuum=10.0
            )
            self.atoms.set_pbc(True)
            # Tag 0: 固定的体相/底层原子
            # Tag 1: 自由的表面原子 (Active Region)
            # Tag 2: 吸附剂 (本场景无)
            tags = np.zeros(len(self.atoms), dtype=int)
            
            # 标记 Active Region 为 1
            n_total = len(self.atoms)
            start_idx = n_total - self.n_active_atoms
            tags[start_idx:] = 1
            self.atoms.set_tags(tags)

            # [关键] 固定非 Active Region 的原子
            if FixAtoms is not None:
                n_total = len(self.atoms)
                # Active Region 在顶部，所以固定底部 (n_total - n_active_atoms) 个原子
                n_fixed = n_total - self.n_active_atoms
                if n_fixed > 0:
                    constraint = FixAtoms(indices=range(n_fixed))
                    self.atoms.set_constraint(constraint)
        
        # 更新 Active Region 的化学符号
        # Active Region 对应 atoms 列表的最后 n_active_atoms 个元素
        # self.atoms.set_positions(self.ideal_positions)
        n_total = len(self.atoms)
        start_idx = n_total - self.n_active_atoms
        
        new_symbols = [self.element_types[idx] for idx in self.state]
        
        for k, sym in enumerate(new_symbols):
            self.atoms[start_idx + k].symbol = sym
            
        return self.atoms

    def _evaluate_energy(self, atoms: Optional["Atoms"]):
        if atoms is None:
            return 0.0, 0.0

        if self.oracle is not None:
            try:
                # 统一调用 compute_energy 接口
                # 对于 UMAOracle，它返回的是生成能 (eV/atom)；对于 EquiformerV2Oracle，它返回总能 (eV)
                energy = self.oracle.compute_energy(atoms, relax=False)
                
                # 如果是旧模型 (EquiformerV2)，它返回的是总能，需要手动减去参考能并归一化
                if not ("UMAOracle" in str(type(self.oracle))):
                    composition = atoms.get_chemical_symbols()
                    e_ref_total = sum([self.ref_energies.get(sym, 0.0) for sym in composition])
                    # 修正逻辑：(总能 - 参考能) / 原子数
                    energy = (energy - e_ref_total) / len(atoms)
                
                return energy, 0.0
            except Exception as e:
                print(f"Oracle calculation failed: {e}")
                return 5.0, 0.0