import torch
from ase.build import fcc111
from chem_gym.surrogate.ocp_model import UMAOracle
import numpy as np

def run_comparison():
    # 1. 初始化 Oracle (确保路径正确)
    checkpoint = "checkpoints/uma-s-1p1.pt"
    oracle = UMAOracle(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")

    # 2. 构建一个 4x4x4 的基础 Cu 晶格
    # 我们将修改最顶层的 16 个原子
    size = (4, 4, 4)
    atoms = fcc111("Cu", size=size, a=3.61, vacuum=10.0)
    atoms.pbc = [True, True, True] # 关键修复：fairchem 要求全维度周期性一致
    n_total = len(atoms)
    surface_indices = list(range(n_total - 16, n_total)) # 最顶层的 16 个原子

    print(f"\n--- 体系信息: {atoms.get_chemical_formula()} ---")
    print(f"总原子数: {n_total}, 表面原子数: 16")

    # ---------------------------------------------------------
    # 构型 A (假设是“好”构型): 将 16 个 Ag 原子放在表面
    # Ag 的表面能通常比 Cu 低，因此 Ag 覆盖表面更稳定
    # ---------------------------------------------------------
    atoms_good = atoms.copy()
    for idx in surface_indices:
        atoms_good[idx].symbol = 'Ag'
    
    energy_good = oracle.compute_energy(atoms_good)
    print(f"\n[构型 A - Ag在表面] 生成能: {energy_good:.6f} eV/atom")

    # ---------------------------------------------------------
    # 构型 B (假设是“坏”构型): 将 16 个 Au 原子放在表面
    # 或者保持 Cu 在表面，将 Ag 埋在下面第二层
    # ---------------------------------------------------------
    atoms_bad = atoms.copy()
    # 我们把 Ag 换到倒数第二层 (indices: n-32 到 n-16)
    subsurface_indices = list(range(n_total - 32, n_total - 16))
    for idx in subsurface_indices:
        atoms_bad[idx].symbol = 'Ag'
    # 表面保持为 Cu (默认)
    
    energy_bad = oracle.compute_energy(atoms_bad)
    print(f"[构型 B - Ag在深层] 生成能: {energy_bad:.6f} eV/atom")

    # 3. 计算能量差
    diff = (energy_bad - energy_good) * n_total
    print(f"\n结论:")
    print(f"能量差 (Total Delta E): {diff:.4f} eV")
    if energy_good < energy_bad:
        print("✅ 验证成功: Ag 在表面确实比埋在深层更稳定 (能量更低)。")
    else:
        print("❓ 结果出乎意料，可能需要进一步弛豫或检查元素属性。")

if __name__ == "__main__":
    run_comparison()