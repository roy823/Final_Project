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

    def __init__(self, config: EnvConfig, surrogate=None):
        super().__init__()
        self.config = config
        self.surrogate = surrogate
        self.rng = np.random.default_rng(config.init_seed)

        self.element_types = config.element_types
        self.n_elements = len(self.element_types)
        
        # 网格尺寸
        self.h, self.w = config.slab_size
        self.n_sites = self.h * self.w
        
        self.render_mode = config.render_mode

        # === 修改 1: 动作空间 ===
        # 格式: [动作类型, X坐标, Y坐标, 额外参数]
        # 动作类型 (3): 0=Swap, 1=Add, 2=Delete
        # X, Y: 坐标
        # 额外参数: 
        #    - 如果是 Swap: 代表方向 (0:上, 1:下, 2:左, 3:右)
        #    - 如果是 Add: 代表原子类型索引 (0..n_elements-1)
        #    - 如果是 Delete: 忽略
        max_arg = max(4, self.n_elements) # 确保够用
        self.action_space = spaces.MultiDiscrete([3, self.h, self.w, max_arg])
        # ========================
        
        # 观察空间保持不变 (Image 模式下，空位将由全0向量表示)
        if self.config.mode == "image":
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.h, self.w, self.n_elements),
                dtype=np.float32,
            )
        # ... (Graph 模式略，保持原样即可) ...

        # 内部状态：使用 2D 数组更方便，-1 代表空位
        self.state = np.full((self.h, self.w), -1, dtype=np.int32)
        
        # ... 其他初始化保持不变 ...
        self.atoms = None
        self.initial_energy = 0.0
        # ...

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # === 修改 2: 初始化 2D 状态 ===
        # 简单的初始化逻辑：随机填充所有格子
        # 如果你想随机留空，可以把部分位置设为 -1
        flat_state = self.rng.integers(0, self.n_elements, size=self.n_sites)
        self.state = flat_state.reshape(self.h, self.w)
        # ============================

        self.steps = 0
        self.atoms = None
        self.atoms = self._build_atoms_from_state() # 重新构建物理对象

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

    def step(self, action):
        # 1. 解析动作 [Type, X, Y, Arg]
        # act_type: 0=Swap, 1=Add, 2=Delete
        # x, y: 坐标
        # arg: Swap的方向 或 Add的原子类型
        act_type, x, y, arg = action
        
        reward = 0.0
        # 基础惩罚配置
        invalid_penalty = -1.0  # 非法操作（撞墙、删空气）
        cost_of_add = -0.5      # 添加原子的材料费（防止无脑填满）
        cost_of_delete = -0.2   # 删除原子的操作费
        
        # === 动作逻辑分支 ===
        
        # Case 0: Swap (交换)
        if act_type == 0:
            direction = arg % 4 # 0=Up, 1=Down, 2=Left, 3=Right
            tx, ty = x, y
            
            # 计算目标坐标
            if direction == 0: tx -= 1
            elif direction == 1: tx += 1
            elif direction == 2: ty -= 1
            elif direction == 3: ty += 1
            
            # 边界检查
            if 0 <= tx < self.h and 0 <= ty < self.w:
                # 执行交换 (即使一边是空位也可以交换，相当于移动)
                self.state[x, y], self.state[tx, ty] = self.state[tx, ty], self.state[x, y]
            else:
                reward += invalid_penalty # 撞墙了

        # Case 1: Add (增加)
        elif act_type == 1:
            target_element = arg % self.n_elements
            # 只有当前是空位 (-1) 才能添加
            if self.state[x, y] == -1:
                self.state[x, y] = target_element
                reward += cost_of_add # 扣除材料费
                # 可选：给一点点微小的激励抵消一部分成本，如果鼓励构建的话
                reward += 0.1 
            else:
                reward += invalid_penalty # 已经有原子了，不能覆盖

        # Case 2: Delete (删除)
        elif act_type == 2:
            # 只有当前有原子 (!= -1) 才能删除
            if self.state[x, y] != -1:
                self.state[x, y] = -1 # 变为空位
                reward += cost_of_delete # 扣除操作费
            else:
                reward += invalid_penalty # 删除空气

        # =======================

        self.steps += 1

        # === 物理与规则检查 (防作弊) ===
        
        # 1. 更新物理状态
        # _build_atoms_from_state 会根据 state 里的 -1 自动处理空位
        self.atoms = self._build_atoms_from_state()
        self.current_energy, self.current_uncertainty = self._evaluate_energy(self.atoms)

        # 2. 检查原子数量是否过少 (防止“删库跑路”策略)
        # 假设至少保留 25% 的原子
        current_atom_count = np.sum(self.state != -1)
        min_atoms = int(self.n_sites * 0.25) 
        
        terminated = False
        truncated = self.steps >= self.config.max_steps

        if current_atom_count < min_atoms:
            reward -= 10.0 # 巨额惩罚
            terminated = True # 强制失败结束
        
        # 3. 计算能量奖励 (Energy Improvement)
        if not terminated:
            energy_diff = self.prev_energy - self.current_energy
            # 放大奖励信号
            reward += (energy_diff * 10.0) 
            
        # 扣除每步的时间成本
        reward -= self.config.step_penalty
        
        self.prev_energy = self.current_energy

        # 4. 安全截断 (防止能量爆炸)
        if self.current_energy > 10.0: 
            reward -= 5.0
            terminated = True

        observation = self._state_to_observation()
        info = {
            "energy": self.current_energy,
            "uncertainty": self.current_uncertainty,
            "atom_count": current_atom_count,
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
    
    def _state_to_observation(self):
        # 模式检查
        if self.config.mode == "image":
            # 初始化一个多层矩阵
            # 形状: (H, W, Channels)
            # Channels = 元素种类数 (n_elements)
            # 比如: Channel 0 是 Cu 的分布图, Channel 1 是 Pt 的分布图...
            grid = np.zeros((self.h, self.w, self.n_elements), dtype=np.float32)
            
            # 遍历我们的逻辑网格
            for r in range(self.h):
                for c in range(self.w):
                    # 获取当前位置的原子类型索引
                    idx = self.state[r, c]
                    
                    # [关键] 处理空位
                    # 如果 idx 是 -1 (空位)，说明这里没原子，所有 Channel 保持 0 (全黑)
                    # 如果 idx >= 0 (有原子)，就在对应的 Channel 填 1
                    if idx != -1:
                        grid[r, c, idx] = 1.0
            
            return grid

        # ... (Graph 模式部分保持原样) ...

    def _build_atoms_from_state(self):
        if fcc111 is None:
            return None
        
        # 1. 计算平均晶格常数 (只考虑存在的原子)
        valid_indices = self.state[self.state != -1]
        if len(valid_indices) == 0:
            # 如果全空，给一个默认值防止报错
            avg_lattice_constant = 3.7 
        else:
            current_elements = [self.element_types[i] for i in valid_indices]
            avg_lattice_constant = float(np.mean([
                self.LATTICE_CONSTANTS.get(el, 3.7) for el in current_elements
            ]))
        
        # 2. 生成完整的底板 (包含满的表面层)
        atoms = fcc111(
            "Cu", 
            size=(self.h, self.w, 4), 
            a=avg_lattice_constant, 
            vacuum=10.0
        )
        
        # 3. 识别表面原子并更新/标记删除
        # ASE fcc111 生成的原子顺序通常是层优先。
        # 最上面一层 (表面) 是列表的最后 n_sites 个原子。
        surface_start_idx = len(atoms) - self.n_sites
        
        indices_to_delete = []
        
        # 遍历网格 (注意 ASE 的原子排列顺序通常与 reshape 顺序对应，但也可能需要微调)
        # 这里假设 fcc111 生成顺序是 行优先 或 列优先。
        # 通常 ASE fcc111 的表面原子顺序对应于 nested loop: for x in range(h): for y in range(w)
        # 我们可以展平 state 来一一对应
        flat_state = self.state.flatten() 
        
        for k, type_idx in enumerate(flat_state):
            atom_idx = surface_start_idx + k
            
            if type_idx == -1:
                # 这是一个空位，标记删除
                indices_to_delete.append(atom_idx)
            else:
                # 这是一个原子，更新类型
                atoms[atom_idx].symbol = self.element_types[type_idx]
        
        # 4. 执行删除 (必须倒序删除，防止索引偏移)
        # 或者使用 del atoms[list] (ASE 支持列表删除)
        if indices_to_delete:
            del atoms[indices_to_delete]

        # 5. 添加约束 (固定底部原子)
        if FixAtoms is not None:
            # 重新计算高度，固定 Z 坐标较小的原子
            # 简单的做法：固定 Z < 10.0 的原子 (假设表面在上面)
            # 或者固定原来的数量 (稍微不准但可行)
            z_positions = atoms.get_positions()[:, 2]
            # 找到最高 Z
            max_z = np.max(z_positions)
            # 固定距离表面一定距离以下的原子
            fixed_indices = [i for i, z in enumerate(z_positions) if z < max_z - 3.0]
            
            constraint = FixAtoms(indices=fixed_indices)
            atoms.set_constraint(constraint)
            
        return atoms
        
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