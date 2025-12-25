from typing import Dict, Optional, Tuple, List
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("gymnasium is required for ChemGymEnv") from exc

try:
    from ase.build import fcc111, bulk
    from ase.visualize.plot import plot_atoms
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from ase.constraints import FixAtoms
except ImportError:
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
    
    Evolution v6.0 (Anti-Reward Hacking):
    - Strategy: Quadratic Debt Penalty (L2 Loss) to prevent mass element deletion.
    - Safety: Credit Limit Masking (Hard stops deviation > 3 atoms).
    - Observation: Global Debt Vector included.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    LATTICE_CONSTANTS = {
        "Pt": 3.92, "Pd": 3.89, "Cu": 3.61, "Ni": 3.52, "Au": 4.08, "Ag": 4.09
    }

    def __init__(self, config: EnvConfig, surrogate=None, oracle=None):
        super().__init__()
        self.config = config
        self.surrogate = surrogate
        self.oracle = oracle

        self.rng = np.random.default_rng(config.init_seed)
        self.element_types = config.element_types
        self.n_elements = len(self.element_types)
        
        self.n_sites_per_layer = config.slab_size[0] * config.slab_size[1]
        self.n_active_layers = config.n_active_layers
        self.n_active_atoms = self.n_sites_per_layer * self.n_active_layers
        
        self.render_mode = config.render_mode

        # [动作空间] 纯突变
        self.action_space = spaces.Discrete(self.n_active_atoms * self.n_elements)
        
        # [观察空间]
        if self.config.mode == "image":
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(config.slab_size[0], config.slab_size[1], self.n_elements),
                dtype=np.float32,
            )
        elif self.config.mode == "graph":
            self.n_total_atoms = self.n_sites_per_layer * self.config.n_layers
            # Feat Dim: Elements + Pos + Layer + CN + Strain + DebtVector
            self.node_feat_dim = self.n_elements + 3 + 1 + 1 + 1 + self.n_elements

            self.observation_space = spaces.Dict({
                "node_features": spaces.Box(low=-100.0, high=100.0, shape=(self.n_total_atoms, self.node_feat_dim), dtype=np.float32),
                "adjacency": spaces.Box(low=0.0, high=1.0, shape=(self.n_total_atoms, self.n_total_atoms), dtype=np.float32),
                "node_mask": spaces.Box(low=0.0, high=1.0, shape=(self.n_total_atoms,), dtype=np.float32),
            })
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        self.state = None
        self.atoms = None
        self.target_counts = None 
        
        self.initial_energy = 0.0
        self.prev_energy = 0.0
        self.current_energy = 0.0
        self.min_energy_so_far = 0.0
        self.prev_debt = 0.0 # [重要] 记录上一步的 L2 债务
        
        self.steps = 0
        self.ref_energies = {}
        
        # [新增] 信用额度 (Credit Limit)
        # 允许最大偏离多少个原子。超过这个值将触发 Hard Masking。
        self.max_deviation = 3 

        if self.oracle is not None:
            self._calibrate_references()

    def _calibrate_references(self):
        if isinstance(self.oracle, (object,)) and "UMAOracle" in str(type(self.oracle)):
            print("[ChemGymEnv] UMA Oracle detected. Using internal MP2020 corrections.")
            return
        print("[ChemGymEnv] Calibrating reference energies...")
        if hasattr(self.oracle, "compute_energy"):
             try:
                for elem in self.element_types:
                    atoms = bulk(elem, 'fcc', a=self.LATTICE_CONSTANTS.get(elem, 3.8))
                    total_energy = self.oracle.compute_energy(atoms, relax=True)
                    self.ref_energies[elem] = total_energy / len(atoms)
             except: pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        n_base = self.n_active_atoms // self.n_elements
        remainder = self.n_active_atoms % self.n_elements
        base_state = []
        for i in range(self.n_elements):
            count = n_base + (1 if i < remainder else 0)
            base_state.extend([i] * count)
        
        self.target_counts = np.bincount(base_state, minlength=self.n_elements)
        self.state = np.array(base_state, dtype=np.int32)
        self.rng.shuffle(self.state)

        self.steps = 0
        self.atoms = None 
        self.atoms = self._build_atoms_from_state()

        self.initial_energy, _ = self._evaluate_energy(self.atoms)
        self.current_energy = self.initial_energy
        self.prev_energy = self.initial_energy
        self.min_energy_so_far = self.initial_energy
        self.prev_debt = 0.0

        return self._state_to_observation(), {
            "energy": self.current_energy, 
            "atoms": self.atoms
        }

    def action_masks(self) -> np.ndarray:
        """
        [关键改进] 信用熔断机制 (Credit Limit Circuit Breaker)
        如果某种元素已经少得离谱(超过 max_deviation)，禁止继续把该元素突变成其他元素。
        如果某种元素已经多得离谱，禁止继续把其他元素突变成该元素。
        """
        mask = np.ones(self.action_space.n, dtype=bool)
        
        current_counts = np.bincount(self.state, minlength=self.n_elements)
        # diff = current - target
        # diff < -3 : 说明该元素太少了，必须买入，不能卖出
        # diff > +3 : 说明该元素太拥挤了，必须卖出，不能买入
        diffs = current_counts - self.target_counts
        
        for el_idx in range(self.n_elements):
            diff = diffs[el_idx]
            
            # 情况 A: 元素太少 (Undersupply > Limit)
            # 禁止所有 "X -> Not_X" 的动作 (即禁止进一步消耗该元素)
            # 动作逻辑: Action = Site * n_elems + Target
            if diff <= -self.max_deviation:
                # 遍历所有位置
                for site_idx in range(self.n_active_atoms):
                    # 如果这个位置当前就是该元素
                    if self.state[site_idx] == el_idx:
                        # 禁止把它变成其他任何元素 (Mask 掉该 Site 的所有动作)
                        start_act = site_idx * self.n_elements
                        end_act = start_act + self.n_elements
                        mask[start_act : end_act] = False
                        
                        # (可选) 但允许 No-op? 不，既然太少了，最好逼它别动或者通过别处变回来
                        # 这里我们只禁止它变走。
            
            # 情况 B: 元素太多 (Oversupply > Limit)
            # 禁止所有 "Not_X -> X" 的动作 (即禁止进一步生成该元素)
            if diff >= self.max_deviation:
                # 动作索引: 任意 Site * n_elems + el_idx
                # 使用切片禁用所有目标为 el_idx 的动作
                mask[el_idx::self.n_elements] = False
                
        return mask

    def step(self, action: int):
        site_idx = action // self.n_elements
        target_elem_idx = action % self.n_elements
        current_elem_idx = self.state[site_idx]
        
        # 1. No-op 检查
        if current_elem_idx == target_elem_idx:
            reward = 0.0 
            self.steps += 1
            truncated = self.steps >= self.config.max_steps
            return self._state_to_observation(), reward, False, truncated, {
                "energy": self.current_energy,
                "action_type": "no_op",
                "atoms": self.atoms
            }

        # 2. 执行突变
        self.state[site_idx] = target_elem_idx
        self.steps += 1

        # 3. 更新物理
        self.atoms = self._build_atoms_from_state()
        self.current_energy, _ = self._evaluate_energy(self.atoms)

        # 4. 计算奖励 (Enhanced Delta Reward)
        
        # (A) 能量奖励
        energy_diff = self.prev_energy - self.current_energy
        reward = energy_diff * 1000.0
        
        # (B) 债务奖励 (Quadratic/L2 Logic)
        current_counts = np.bincount(self.state, minlength=self.n_elements)
        diffs = current_counts - self.target_counts
        
        # [关键改动] 使用平方和 (L2 Norm) 而不是绝对值和 (L1 Norm)
        # 例如: 偏差 1 -> Debt=1; 偏差 3 -> Debt=9
        current_debt_l2 = np.sum(np.square(diffs)) 
        
        # 计算 Delta: 
        # 如果债务增加 (e.g. 1->4, diff=-3)，重罚
        # 如果债务减少 (e.g. 4->1, diff=+3)，重赏
        debt_diff = self.prev_debt - current_debt_l2
        
        # 系数微调: 因为是平方，数值会变大，系数可以稍微小一点
        # 偏差 1->2 (delta -3) * 2.0 = -6.0 分 (足以抵消普通的能量收益)
        # 偏差 2->3 (delta -5) * 2.0 = -10.0 分 (严厉禁止)
        debt_coeff = 2.0 
        reward += debt_coeff * debt_diff
        
        # (C) 滞留惩罚 (基于平方)
        if current_debt_l2 > 0:
            reward -= 0.1 # 稍微增加利息，逼它还钱

        # (D) 历史最佳 (仅在严格守恒时触发)
        # 既然我们用了 L2 惩罚，对于打破记录的要求就要严格一些
        # 只允许 L2 < 2 (即最多偏离 1 个原子) 时算有效记录
        if self.current_energy < self.min_energy_so_far and current_debt_l2 <= 2.0:
            bonus = (self.min_energy_so_far - self.current_energy) * 2000.0
            reward += bonus
            self.min_energy_so_far = self.current_energy

        # 更新历史状态
        self.prev_energy = self.current_energy
        self.prev_debt = current_debt_l2

        terminated = False
        truncated = self.steps >= self.config.max_steps
        
        if self.current_energy > 5.0: 
            reward -= 10.0
            terminated = True

        return self._state_to_observation(), reward, terminated, truncated, {
            "energy": self.current_energy,
            "stoich_debt": current_debt_l2,
            "action_type": "mutation",
            "atoms": self.atoms
        }

    def _state_to_observation(self):
        # Image 模式兼容
        if self.config.mode == "image":
            surface_state = self.state[-self.n_sites_per_layer:]
            flat_one_hot = np.eye(self.n_elements, dtype=np.float32)[surface_state]
            return flat_one_hot.reshape(self.config.slab_size[0], self.config.slab_size[1], self.n_elements)
            
        if self.atoms is None:
            return {
                "node_features": np.zeros((self.n_total_atoms, self.node_feat_dim), dtype=np.float32),
                "adjacency": np.zeros((self.n_total_atoms, self.n_total_atoms), dtype=np.float32),
                "node_mask": np.zeros((self.n_total_atoms,), dtype=np.float32)
            }
            
        symbols = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions().astype(np.float32)
        n_total = len(symbols)

        # 物理几何特征
        dist_matrix = self.atoms.get_all_distances(mic=True).astype(np.float32)
        cutoff = float(self.config.graph_cutoff)
        sigma = float(self.config.graph_sigma)
        if sigma <= 0: sigma = 1.0
        
        adjacency = np.zeros_like(dist_matrix, dtype=np.float32)
        adj_mask = (dist_matrix > 0) & (dist_matrix <= cutoff)
        adjacency[adj_mask] = np.exp(- (dist_matrix[adj_mask] / sigma) ** 2)
        np.fill_diagonal(adjacency, 1.0)

        coord_num = np.sum(adj_mask, axis=1, dtype=np.float32)
        avg_bond_len = np.zeros(n_total, dtype=np.float32)
        for k in range(n_total):
            neighbors = dist_matrix[k][adj_mask[k]]
            avg_bond_len[k] = np.mean(neighbors) if len(neighbors) > 0 else 0.0

        # Raw Debt Vector
        current_counts = np.bincount(self.state, minlength=self.n_elements)
        debt_vector = (self.target_counts - current_counts).astype(np.float32)
        
        pos_center = positions.mean(axis=0, keepdims=True)
        rel_pos = positions - pos_center

        node_features = np.zeros((n_total, self.node_feat_dim), dtype=np.float32)
        node_mask = np.ones((n_total,), dtype=np.float32)

        for i, sym in enumerate(symbols):
            one_hot = np.zeros((self.n_elements,), dtype=np.float32)
            if sym in self.element_types:
                one_hot[self.element_types.index(sym)] = 1.0
            
            layer_idx = i // self.n_sites_per_layer
            
            node_features[i] = np.concatenate([
                one_hot,              
                rel_pos[i],           
                [float(layer_idx)],   
                [coord_num[i] / 12.0],
                [avg_bond_len[i]],    
                debt_vector           
            ])

        return {
            "node_features": node_features, 
            "adjacency": adjacency, 
            "node_mask": node_mask
        }

    # 其他辅助函数保持不变
    def _action_to_indices(self, action: int):
        return action // self.n_elements, action % self.n_elements

    def _build_atoms_from_state(self):
        if fcc111 is None: return None
        if self.atoms is None:
            current_elements = [self.element_types[i] for i in self.state]
            avg_lat = float(np.mean([self.LATTICE_CONSTANTS.get(el, 3.7) for el in current_elements]))
            self.atoms = fcc111(
                "Cu", 
                size=(self.config.slab_size[0], self.config.slab_size[1], self.config.n_layers), 
                a=avg_lat, 
                vacuum=10.0
            )
            self.atoms.set_pbc(True)
            tags = np.zeros(len(self.atoms), dtype=int)
            start_idx = len(self.atoms) - self.n_active_atoms
            tags[start_idx:] = 1
            self.atoms.set_tags(tags)
            if FixAtoms:
                n_fixed = len(self.atoms) - self.n_active_atoms
                if n_fixed > 0: self.atoms.set_constraint(FixAtoms(indices=range(n_fixed)))
        
        n_total = len(self.atoms)
        start_idx = n_total - self.n_active_atoms
        new_symbols = [self.element_types[idx] for idx in self.state]
        for k, sym in enumerate(new_symbols):
            self.atoms[start_idx + k].symbol = sym
        return self.atoms

    def _evaluate_energy(self, atoms):
        if atoms is None: return 0.0, 0.0
        if self.oracle:
            try:
                energy = self.oracle.compute_energy(atoms, relax=False)
                if not ("UMAOracle" in str(type(self.oracle))):
                    composition = atoms.get_chemical_symbols()
                    e_ref_total = sum([self.ref_energies.get(sym, 0.0) for sym in composition])
                    energy = (energy - e_ref_total) / len(atoms)
                return energy, 0.0
            except: return 5.0, 0.0
        if self.surrogate: return self.surrogate.evaluate(atoms)
        if EMT:
            try:
                c = atoms.copy()
                c.calc = EMT()
                if c.get_potential_energy() > 100.0: return 10.0, 0.0
                if BFGS: BFGS(c, logfile=None).run(fmax=0.5, steps=20)
                e = c.get_potential_energy()/len(c)
                if e > 5.0 or e < -10.0: return 5.0, 0.0
                return e, 0.0
            except: return 5.0, 0.0
        return 0.0, 0.0

    def render(self):
        if self.render_mode is None: return None
        if plot_atoms is None or self.atoms is None: return None
        fig = plot_atoms(self.atoms, show_unit_cell=0, rotation='-90x')
        if self.render_mode == "human": return fig
        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            return data.reshape((h, w, 3))
        return None