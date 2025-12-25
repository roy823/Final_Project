import math
from typing import Dict, List
import numpy as np
from chem_gym.envs.chem_env import ChemGymEnv

def random_search(env: ChemGymEnv, total_steps: int) -> Dict:
    """
    严格守恒的随机搜索：通过随机交换两个不同元素的位置来优化。
    """
    best_energy = math.inf
    history = []
    env.reset()
    
    # 每次交换消耗 2 个 Oracle 调用步数
    n_swaps = total_steps // 2
    
    for _ in range(n_swaps):
        # 1. 随机选两个位置
        idx_i, idx_j = np.random.choice(env.n_active_atoms, 2, replace=False)
        elem_i = env.state[idx_i]
        elem_j = env.state[idx_j]
        
        if elem_i == elem_j:
            # 如果元素相同，交换没意义，跳过但不计步数（或计入 history 保持长度）
            history.extend([best_energy, best_energy])
            continue

        # 2. 执行交换（分两步变异）
        # 第一步：i 变成 j 的元素
        env.step(idx_i * env.n_elements + elem_j)
        # 第二步：j 变成 i 原来的元素 (此时配比恢复)
        _, _, _, _, info = env.step(idx_j * env.n_elements + elem_i)
        
        current_energy = info["energy"]
        if current_energy < best_energy:
            best_energy = current_energy
        
        # 记录两次（因为消耗了 2 步）
        history.extend([best_energy, best_energy])
        
    return {"best_energy": best_energy, "history": history[:total_steps]}

def simulated_annealing(env: ChemGymEnv, total_steps: int, t_start: float = 0.1, t_end: float = 0.001) -> Dict:
    """
    严格守恒的模拟退火：基于交换操作的 Metropolis 采样。
    """
    _, info = env.reset()
    current_energy = info["energy"]
    best_energy = current_energy
    history = []

    n_swaps = total_steps // 2

    for step in range(n_swaps):
        temp = t_start * (t_end / t_start) ** (step / n_swaps)
        
        # 备份当前状态
        old_state = env.state.copy()
        old_energy = current_energy
        
        # 1. 尝试交换
        idx_i, idx_j = np.random.choice(env.n_active_atoms, 2, replace=False)
        elem_i = old_state[idx_i]
        elem_j = old_state[idx_j]
        
        if elem_i == elem_j:
            history.extend([best_energy, best_energy])
            continue

        # 执行交换
        env.step(idx_i * env.n_elements + elem_j)
        _, _, _, _, info = env.step(idx_j * env.n_elements + elem_i)
        new_energy = info["energy"]
        
        delta_e = new_energy - old_energy
        
        # 2. Metropolis 准则
        if delta_e < 0 or math.exp(-delta_e / max(temp, 1e-8)) > np.random.rand():
            # 接受
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
        else:
            # 拒绝：必须撤销交换（再换回来）
            env.step(idx_i * env.n_elements + elem_i)
            env.step(idx_j * env.n_elements + elem_j)
            # 注意：撤销操作不计入对比步数，或者你可以选择计入
            
        history.extend([best_energy, best_energy])
        
    return {"best_energy": best_energy, "history": history[:total_steps]}