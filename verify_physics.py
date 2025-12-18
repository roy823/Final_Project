from chem_gym.config import EnvConfig
from chem_gym.envs.chem_env import ChemGymEnv
import numpy as np

def verify():
    print("="*60)
    print("🧪 Chem-Gym 物理环境验证脚本 (EMT + Relaxation)")
    print("="*60)
    
    # 1. 初始化环境，不带 Surrogate (强制使用 EMT)
    print("[*] 初始化环境 (surrogate=None)...")
    config = EnvConfig(mode="image", slab_size=(4, 4))
    try:
        env = ChemGymEnv(config, surrogate=None)
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        return

    # 2. Reset 环境
    print("[*] 正在 Reset 环境 (第一次运行需要编译 EMT 计算器，请稍候)...")
    obs, info = env.reset()
    
    initial_energy = info['energy']
    print(f"✅ Reset 完成。初始能量: {initial_energy:.5f} eV/atom")
    
    # 检查点 1: 能量必须是负数 (结合能)
    if initial_energy >= 0:
        print("❌ 严重错误: 初始能量非负！这意味着原子极其不稳定或计算器未工作。")
        print("   -> 请检查晶格常数 (Lattice Constants) 是否设置正确。")
    else:
        print("✅ 能量检查通过: 能量为负值，符合结合能物理规律。")

    # 3. 检查原子约束
    atoms = info['atoms']
    constraints = atoms.constraints
    if len(constraints) > 0:
        print(f"✅ 约束检查通过: 检测到 {len(constraints)} 个约束对象 (固定底部原子)。")
    else:
        print("❌ 警告: 未检测到原子固定约束！整个 slab 可能会在空间中飘移。")

    # 4. 执行随机动作测试
    print("\n[*] 执行 5 步随机交换测试...")
    
    energies = [initial_energy]
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        curr_energy = info['energy']
        
        diff = curr_energy - energies[-1]
        energies.append(curr_energy)
        
        swapped = info.get('swapped_sites', ('?', '?'))
        print(f"   Step {step+1}: Swap {swapped} | Energy={curr_energy:.5f} eV | Reward={reward:.4f}")

    # 检查点 2: 能量变化
    energy_range = max(energies) - min(energies)
    if energy_range < 1e-6:
        print("\n⚠️ 警告: 5步操作后能量几乎没有变化。")
        print("   可能性 1: 你的交换刚好发生在同种元素之间 (如 Cu 换 Cu)。")
        print("   可能性 2: EMT 计算器没有更新 (请检查 step 函数中是否调用了 _evaluate_energy)。")
    else:
        print(f"\n✅ 动力学检查通过: 能量发生波动 (Range: {energy_range:.5f} eV)，说明物理引擎正在响应动作。")

    print("="*60)
    print("验证完成。如果以上检查均为 ✅，你可以运行 main.py 开始 PPO 训练。")
    print("="*60)

if __name__ == "__main__":
    verify()