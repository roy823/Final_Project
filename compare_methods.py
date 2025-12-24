import matplotlib.pyplot as plt
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.config import EnvConfig
from chem_gym.surrogate.ocp_model import UMAOracle
from chem_gym.baselines import random_search, simulated_annealing
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.maskable.utils import get_action_masks
import torch
from pathlib import Path
import numpy as np

# 1. 初始化环境
print("Initializing Environment and Oracle...")
oracle = UMAOracle("checkpoints/uma-s-1p1.pt")
config = EnvConfig(mode="graph", n_active_layers=3)

def make_env():
    return ChemGymEnv(config, oracle=oracle)

base_venv = DummyVecEnv([make_env])

# [关键] 加载 0.9 分模型对应的归一化参数
stats_path = "checkpoints/vec_normalize_ppo_maskable.pkl"
print(f"✅ Loading CORRECT normalization stats from {stats_path}...")
venv = VecNormalize.load(stats_path, base_venv)
venv.training = False
venv.norm_reward = False

STEPS = 500 

# 2. 运行基准算法
print("\nRunning Random Search...")
res_random = random_search(venv.unwrapped.envs[0], STEPS)

print("Running Simulated Annealing...")
res_sa = simulated_annealing(venv.unwrapped.envs[0], STEPS)

# 3. 运行 0.9 分的 MaskablePPO Agent
print("\nRunning Trained Maskable DRL Agent...")
model_path = "checkpoints/ppo_maskable"
model = MaskablePPO.load(model_path, env=venv)
print("✅ Successfully loaded MaskablePPO model!")

res_drl = {"history": []}
obs = venv.reset()
best_e = float('inf')

for i in range(STEPS):
    masks = get_action_masks(venv)
    # [修改] 将 True 改为 False，增加探索性
    action, _ = model.predict(obs, action_masks=masks, deterministic=False) 
    
    obs, rewards, dones, infos = venv.step(action)
    current_energy = infos[0]['energy']
    
    if current_energy < best_e:
        best_e = current_energy
    res_drl["history"].append(best_e)
    
    if i % 50 == 0:
        print(f"DRL Step {i:03d} | Best Energy: {best_e:.6f} eV/atom")

# 4. 绘图对比
plt.figure(figsize=(10, 6))
plt.plot(res_random["history"], label="Random Search (Luck)", color='gray', linestyle='--')
plt.plot(res_sa["history"], label="Simulated Annealing (Physics)", color='blue')
plt.plot(res_drl["history"], label="Our Maskable DRL Agent (AI)", color='red', linewidth=2)
plt.xlabel("Oracle Calls (Steps)")
plt.ylabel("Best Energy Found (eV/atom)")
plt.title("Optimization Performance: DRL vs Baselines (HEA Surface)")
plt.legend()
plt.grid(True)
plt.savefig("comparison_plot.png")
print("\n✅ Comparison finished! Check [comparison_plot.png] for the final result.")