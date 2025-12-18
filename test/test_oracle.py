import torch
import numpy as np
from ase.build import fcc111, add_adsorbate
from ase import Atoms
from ase.visualize import view
import os

from chem_gym.surrogate.ocp_model import EquiformerV2Oracle 

# 请将此路径替换为你实际下载的 EquiformerV2 checkpoint 路径
# 如果只是想测试逻辑而不加载大模型，请将 MOCK_MODE 设为 True
CHECKPOINT_PATH = "checkpoints/eq2_83M_2M.pt" 
MOCK_MODE = False  

class MockCalculator:
    """用于在没有模型权重时测试逻辑流的伪计算器"""
    def __init__(self, checkpoint=None, cpu=True):
        pass
    def get_potential_energy(self, atoms=None):
        # 返回一个伪随机能量
        return -100.0 + np.random.random()
    def get_forces(self, atoms):
        return np.random.random((len(atoms), 3)) * 0.1

if MOCK_MODE:
    print("!!! 运行在 MOCK 模式：仅验证代码逻辑，不加载真实模型 !!!")
    # 猴子补丁：替换 OCPCalculator 以便在没有权重时也能跑通流程
    import sys
    from unittest.mock import MagicMock
    # 模拟 fairchem 导入以防未安装
    sys.modules["fairchem"] = MagicMock()
    sys.modules["fairchem.core"] = MagicMock()
    sys.modules["fairchem.core.common.relaxation.ase_utils"] = MagicMock()
    # 替换 Oracle 内部的 calculator 初始化逻辑
    # 注意：如果你在运行真实测试，请确保 MOCK_MODE = False
    class EquiformerV2OracleMocked(EquiformerV2Oracle):
        def __init__(self, *args, **kwargs):
            self.device = "cpu"
            self.fmax = kwargs.get('fmax', 0.05)
            self.max_steps = kwargs.get('max_steps', 100)
            self.calculator = MockCalculator()
    
    OracleClass = EquiformerV2OracleMocked
else:
    # 正常模式：使用你的类
    OracleClass = EquiformerV2Oracle

# ==========================================
# 3. 构造符合 OC20 标准的数据 (Pt111 + O)
# ==========================================
def create_oc20_style_data():
    """
    创建一个模拟 OC20 格式的 ASE Atoms 对象。
    OC20 Tags 约定:
    0: Subsurface slab (固定)
    1: Surface slab (固定)
    2: Adsorbate (弛豫)
    """
    # 1. 创建 Slab (Pt 111)
    slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
    
    # 手动设置 Tags (模拟 OC20 数据集)
    # 假设底部 2 层是体相(0)，顶部 2 层是表面(1)
    tags = np.zeros(len(slab), dtype=int)
    z_positions = slab.get_positions()[:, 2]
    median_z = np.median(z_positions)
    
    for i, z in enumerate(z_positions):
        if z > median_z:
            tags[i] = 1 # Surface
        else:
            tags[i] = 0 # Subsurface
    slab.set_tags(tags)
    
    # 2. 创建 Adsorbed System (添加氧原子)
    atoms_with_ads = slab.copy()
    add_adsorbate(atoms_with_ads, 'O', height=1.2, position='fcc')
    
    # 设置吸附剂 Tag 为 2
    # add_adsorbate 默认把新原子加在最后
    new_tags = np.append(atoms_with_ads.get_tags()[:-1], 2)
    atoms_with_ads.set_tags(new_tags)
    
    return slab, atoms_with_ads

# ==========================================
# 4. 执行测试流程
# ==========================================
def run_validation_test():
    print("\n" + "="*40)
    print("开始验证 EquiformerV2Oracle 功能")
    print("="*40)

    # --- 步骤 0: 初始化 Oracle ---
    try:
        oracle = OracleClass(
            checkpoint_path=CHECKPOINT_PATH, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            fmax=0.05,
            max_steps=20 # 测试时步数设少一点
        )
        print("[Pass] 模型初始化成功")
    except Exception as e:
        print(f"[Fail] 模型初始化失败: {e}")
        return

    # --- 准备数据 ---
    slab, atoms_ads = create_oc20_style_data()
    print(f"数据准备完毕: Slab原子数={len(slab)}, 吸附体系原子数={len(atoms_ads)}")

    # --- 步骤 1: 验证约束逻辑 (_get_fixed_indices) ---
    print("\n[Test 1] 验证固定原子索引逻辑 (Constraint Logic)...")
    fixed_indices = oracle._get_fixed_indices(atoms_ads)
    
    # 验证：所有 tag != 2 的原子都应该被固定
    tags = atoms_ads.get_tags()
    expected_fixed = [i for i, t in enumerate(tags) if t != 2]
    
    if set(fixed_indices) == set(expected_fixed):
        print(f"[Pass] 固定索引正确。固定了 {len(fixed_indices)} 个原子 (Slab)，放开了 {len(slab) - len(fixed_indices) + 1} 个原子 (吸附剂)。")
    else:
        print(f"[Fail] 索引错误! \n计算值: {fixed_indices}\n期望值: {expected_fixed}")

    # --- 步骤 2: 计算纯 Slab 能量 ---
    print("\n[Test 2] 计算 Slab 能量 (compute_energy)...")
    try:
        e_slab = oracle.compute_energy(slab, relax=False) # 纯表面通常不弛豫或微弛豫
        print(f"[Pass] E_slab = {e_slab:.4f} eV")
    except Exception as e:
        print(f"[Fail] E_slab 计算失败: {e}")
        e_slab = -1000.0 # Dummy value to continue

    # --- 步骤 3: 计算吸附能 (predict_ads_energy) ---
    print("\n[Test 3] 计算吸附能与弛豫 (predict_ads_energy)...")
    gas_ref = -3.0 # 假设的氧气参考能
    
    try:
        # 这里会触发内部的 LBFGS 弛豫
        e_ads = oracle.predict_ads_energy(
            atoms_with_ads=atoms_ads,
            slab_energy=e_slab,
            gas_reference_energy=gas_ref,
            return_force=False
        )
        print(f"[Pass] 弛豫成功。E_ads = {e_ads:.4f} eV")
        print("       (注: E_ads = E_total - E_slab - E_gas)")
    except Exception as e:
        print(f"[Fail] 吸附能计算失败: {e}")

    # --- 步骤 4: 验证受力返回 (Force Return) ---
    print("\n[Test 4] 验证受力返回与离域化检测接口...")
    try:
        e_ads_2, max_f = oracle.predict_ads_energy(
            atoms_with_ads=atoms_ads,
            slab_energy=e_slab,
            gas_reference_energy=gas_ref,
            return_force=True
        )
        print(f"[Pass] 接口调用成功。")
        print(f"       E_ads: {e_ads_2:.4f} eV")
        print(f"       Max Force on Adsorbate: {max_f:.4f} eV/A")
        
        if max_f < 0.05:
            print("       -> 状态: 收敛良好 (Force < fmax)")
        else:
            print("       -> 状态: 仍有残余力 (可能步数不足)")
            
    except Exception as e:
        print(f"[Fail] 受力返回接口失败: {e}")

    print("\n" + "="*40)
    print("测试结束")
    print("="*40)

if __name__ == "__main__":
    # 如果没有定义类，请先定义，然后运行：
    run_validation_test()