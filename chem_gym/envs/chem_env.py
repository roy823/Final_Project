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
    
    Evolution:
    - [v2.0] Action Space: "Element Mutation" + "Global Teleportation".
             This aligns with GNN's node-classification strength and SA's global search capability.
    - Active Region: Optimizes top N layers.
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

        # [重大修改] 动作空间重定义：Element Mutation
        # 旧: 任意两点交换 C(n, 2) -> O(N^2)
        # 新: 指定位置变为指定元素 -> O(N * K)
        # 动作 Action = Site_Index * n_elements + Target_Element_Index
        # 含义: "我希望 Site i 是 Element k"
        self.action_space = spaces.Discrete(self.n_active_atoms * self.n_elements)
        
        # 观察空间 (保持不变)
        if self.config.mode == "image":
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.slab_size[0], config.slab_size[1], self.n_elements),
                dtype=np.float32,
            )
        elif self.config.mode == "graph":
            self.n_total_atoms = self.n_sites_per_layer * self.config.n_layers
            
            # Node Features: [OneHot(N), RelPos(3), LayerIdx(1)]
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
        self.min_energy_so_far = 0.0
        self.steps = 0
        
        # 参考能量 (校准用)
        self.ref_energies = {}
        if self.oracle is not None:
            self._calibrate_references()

    def _calibrate_references(self):
        """
        [保持 UMA 逻辑]
        使用 UMA 时，参考态修正由内部 FormationEnergyCalculator 处理。
        """
        if isinstance(self.oracle, (object,)) and "UMAOracle" in str(type(self.oracle)):
            print("[ChemGymEnv] UMA Oracle detected. Using internal MP2020 corrections.")
            return
        
        # 原有的手动校准逻辑 (仅在非 UMA 模式下运行)
        print("[ChemGymEnv] Calibrating reference energies with Legacy Oracle...")
        for elem in self.element_types:
             # 为了避免未定义的 self.oracle.compute_energy 报错，这里加个简单保护
             if hasattr(self.oracle, "compute_energy"):
                 try:
                    atoms = bulk(elem, 'fcc', a=self.LATTICE_CONSTANTS.get(elem, 3.8))
                    total_energy = self.oracle.compute_energy(atoms, relax=True)
                    self.ref_energies[elem] = total_energy / len(atoms)
                 except: pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # --- Stoichiometry Control (Active Region) ---
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
        self.min_energy_so_far = self.initial_energy

        observation = self._state_to_observation()
        info = {
            "energy": self.current_energy, 
            "uncertainty": self.current_uncertainty, 
            "atoms": self.atoms
        }
        return observation, info

    def action_masks(self) -> np.ndarray:
        """
        [修改] 动作掩码计算
        对于 Mutation 策略，我们需要允许 "No-op" (即确认当前元素正确)，
        因此大部分动作是有效的。
        唯一需要掩码的情况是：当某个元素在 Slab 中彻底耗尽时，我们无法 Mutate *成为* 该元素。
        但由于 reset 保证了化学计量比，且 step 保持守恒，这种情况理论上不会发生。
        """
        mask = np.ones(self.action_space.n, dtype=bool)
        
        # 统计当前各元素数量
        counts = np.bincount(self.state, minlength=self.n_elements)
        
        # 如果某种元素数量为 0，则所有试图变 *成* 该元素的动作都无效
        for el_idx in range(self.n_elements):
            if counts[el_idx] == 0:
                # 动作索引: site * n_elems + el_idx
                # 使用切片将所有目标为 el_idx 的动作设为 False
                mask[el_idx::self.n_elements] = False
                
        return mask
        
    def step(self, action: int):
        # [修改] 全新的 Mutation + Teleportation 逻辑
        
        # 1. 解码动作：(Target Site, Target Element)
        site_idx = action // self.n_elements
        target_elem_idx = action % self.n_elements
        
        current_elem_idx = self.state[site_idx]
        
        # 2. 确认奖励 (Confirmation Bonus / No-op)
        # 如果 Agent 认为 "Site i 应该是 A"，而它本来就是 A
        # 这表明 GNN 对当前结构的理解是正确的，我们给予微小的正向反馈
        if current_elem_idx == target_elem_idx:
            reward = 0.1 # 小奖励，鼓励网络输出稳定的预测
            self.steps += 1
            truncated = self.steps >= self.config.max_steps
            info = {
                "energy": self.current_energy,
                "uncertainty": self.current_uncertainty,
                "action_type": "confirmation",
                "atoms": self.atoms
            }
            # 状态未变，无需重新计算能量
            return self._state_to_observation(), reward, False, truncated, info

        # 3. 寻找补偿原子 (Global Teleportation Candidates)
        # Agent 想把 site_idx 变成 target_elem_idx
        # 我们必须找一个当前是 target_elem_idx 的原子，把它变成 current_elem_idx
        candidate_indices = np.where(self.state == target_elem_idx)[0]
        
        # 理论上 action_masks 已经过滤了这种情况，但为了鲁棒性再检查一次
        if len(candidate_indices) == 0:
            return self._state_to_observation(), -1.0, False, True, {}

        # 4. 随机瞬移 (Random Teleportation)
        # 这一步引入了极其关键的随机性，打破了局部最优的"死循环"
        swap_partner_idx = self.rng.choice(candidate_indices)
        
        # 执行交换
        self.state[site_idx] = target_elem_idx
        self.state[swap_partner_idx] = current_elem_idx
        
        self.steps += 1

        # 更新物理状态
        self.atoms = self._build_atoms_from_state()
        self.current_energy, self.current_uncertainty = self._evaluate_energy(self.atoms)

        # --- 奖励函数 ---
        energy_diff = self.prev_energy - self.current_energy
        reward = energy_diff * 1000.0
        
        # 允许微小的能量上升 (模拟退火思想)
        if energy_diff <= 0:
            reward -= 0.01 
            
        # 历史最优奖励
        if self.current_energy < self.min_energy_so_far:
            bonus = (self.min_energy_so_far - self.current_energy) * 2000.0
            reward += bonus
            self.min_energy_so_far = self.current_energy

        self.prev_energy = self.current_energy

        terminated = False
        truncated = self.steps >= self.config.max_steps
        
        if self.current_energy > 5.0: 
            reward -= 10.0
            terminated = True

        observation = self._state_to_observation()
        info = {
            "energy": self.current_energy,
            "uncertainty": self.current_uncertainty,
            "swapped_sites": (site_idx, swap_partner_idx),
            "action_type": "mutation",
            "energy_improvement": self.initial_energy - self.current_energy,
            "atoms": self.atoms
        }
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if plot_atoms is None or self.atoms is None:
            return None
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
    
    # 注意：旧的 _action_to_indices 已被 step 中的逻辑替代，
    # 但为了兼容性或调试，这里可以保留一个 dummy 版本或者直接删除。
    # 这里我们保留一个简化版以防外部调用。
    def _action_to_indices(self, action: int):
        # 仅用于 debug，不是真实的逻辑
        return action // self.n_elements, action % self.n_elements

    def _state_to_observation(self):
        if self.config.mode == "image":
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

        # 1. 节点特征
        pos_center = positions.mean(axis=0, keepdims=True)
        rel_pos = positions - pos_center

        node_features = np.zeros((n_total, self.node_feat_dim), dtype=np.float32)
        node_mask = np.ones((n_total,), dtype=np.float32)

        for i, sym in enumerate(symbols):
            one_hot = np.zeros((self.n_elements,), dtype=np.float32)
            if sym in self.element_types:
                one_hot[self.element_types.index(sym)] = 1.0
            layer_idx = i // self.n_sites_per_layer
            node_features[i] = np.concatenate([one_hot, rel_pos[i], [float(layer_idx)]])

        # 2. 邻接矩阵 (RBF)
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
        if fcc111 is None: return None
        if self.atoms is None:
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
            tags = np.zeros(len(self.atoms), dtype=int)
            n_total = len(self.atoms)
            start_idx = n_total - self.n_active_atoms
            tags[start_idx:] = 1
            self.atoms.set_tags(tags)
            if FixAtoms is not None:
                n_fixed = n_total - self.n_active_atoms
                if n_fixed > 0:
                    self.atoms.set_constraint(FixAtoms(indices=range(n_fixed)))
        
        n_total = len(self.atoms)
        start_idx = n_total - self.n_active_atoms
        new_symbols = [self.element_types[idx] for idx in self.state]
        for k, sym in enumerate(new_symbols):
            self.atoms[start_idx + k].symbol = sym
        return self.atoms

    def _evaluate_energy(self, atoms: Optional["Atoms"]):
        if atoms is None: return 0.0, 0.0
        
        # [保持 UMA 逻辑]
        if self.oracle is not None:
            try:
                energy = self.oracle.compute_energy(atoms, relax=False)
                # 非 UMA 模型 (如 EquiformerV2) 需要手动扣除参考能
                if not ("UMAOracle" in str(type(self.oracle))):
                    composition = atoms.get_chemical_symbols()
                    e_ref_total = sum([self.ref_energies.get(sym, 0.0) for sym in composition])
                    energy = (energy - e_ref_total) / len(atoms)
                return energy, 0.0
            except Exception as e:
                print(f"Oracle calculation failed: {e}")
                return 5.0, 0.0

        if self.surrogate is not None:
            return self.surrogate.evaluate(atoms)
        
        if EMT is not None:
            try:
                calc_atoms = atoms.copy()
                calc_atoms.calc = EMT()
                if calc_atoms.get_potential_energy() > 100.0: return 10.0, 0.0
                if BFGS is not None:
                    BFGS(calc_atoms, logfile=None).run(fmax=0.5, steps=20)
                energy = calc_atoms.get_potential_energy() / len(calc_atoms)
                if energy > 5.0 or energy < -10.0: return 5.0, 0.0
                return energy, 0.0
            except Exception as e:
                print(f"Warning: EMT calculation failed: {e}")
                return 5.0, 0.0
        return 0.0, 0.0