import math
from typing import Dict, List
import numpy as np
from chem_gym.envs.chem_env import ChemGymEnv

def random_search(env: ChemGymEnv, total_steps: int) -> Dict:
    """
    随机搜索：完全靠运气，随机交换原子。
    """
    best_energy = math.inf
    history = []
    env.reset()
    
    for _ in range(total_steps):
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        current_energy = info["energy"]
        if current_energy < best_energy:
            best_energy = current_energy
        history.append(best_energy)
        
    return {"best_energy": best_energy, "history": history}

def simulated_annealing(env: ChemGymEnv, total_steps: int, t_start: float = 0.1, t_end: float = 0.001) -> Dict:
    """
    模拟退火：物理界经典算法。
    初期接受高能量状态以跳出局部最优，后期逐渐冷却。
    """
    _, info = env.reset()
    current_energy = info["energy"]
    best_energy = current_energy
    history = []

    for step in range(total_steps):
        # 指数冷却进度表
        temp = t_start * (t_end / t_start) ** (step / total_steps)
        
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        new_energy = info["energy"]
        
        delta_e = new_energy - current_energy
        
        # Metropolis 准则
        if delta_e < 0 or math.exp(-delta_e / max(temp, 1e-8)) > np.random.rand():
            # 接受新状态
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
        else:
            # [关键修复] 如果不接受，必须把原子交换回来！
            # 在我们的环境中，再次执行相同的 action 就会交换回来
            env.step(action) 
            
        history.append(best_energy)
    return {"best_energy": best_energy, "history": history}