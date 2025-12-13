from chem_gym.config import EnvConfig
from chem_gym.envs.chem_env import ChemGymEnv
import numpy as np

def verify():
    print("="*60)
    print("🧪 Chem-Gym 物理环境验证脚本 (Add/Delete/Swap)")
    print("="*60)
    
    # 1. 初始化环境
    print("[*] 初始化环境 (surrogate=None)...")
    config = EnvConfig(mode="image", slab_size=(4, 4))
    try:
        env = ChemGymEnv(config, surrogate=None)
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        return

    # 2. Reset 环境
    print("[*] 正在 Reset 环境...")
    obs, info = env.reset()
    
    initial_energy = info['energy']
    print(f"✅ Reset 完成。初始能量: {initial_energy:.5f} eV/atom")
    print(f"   初始原子分布:\n{env.state}")

    # 3. 执行随机动作测试
    print("\n[*] 执行 10 步随机动作测试...")
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 解析动作以便打印
        act_type, x, y, arg = action
        curr_energy = info['energy']
        atom_count = info.get('atom_count', '?')
        
        # 动作名称映射
        act_name = ["Swap", "Add ", "Del "][act_type]
        
        # 打印详细日志
        print(f"Step {step+1:02d}: {act_name} @ ({x},{y}) | Arg={arg} | Atoms={atom_count} | E={curr_energy:.4f} | R={reward:.3f}")

        if terminated:
            print(f"   ⚠️ 回合提前结束 (Terminated)")
            break

    print("="*60)
    print("验证完成。请检查上面的日志：")
    print("1. 动作类型是否包含 Swap, Add, Del")
    print("2. Atoms 数量是否在变化")
    print("3. Energy 是否随动作波动")
    print("="*60)

if __name__ == "__main__":
    verify()