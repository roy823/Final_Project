import torch
import numpy as np
from ase.build import fcc111, add_adsorbate
from ase import Atoms
from ase.constraints import FixAtoms

from chem_gym.surrogate.ocp_model import EquiformerV2Oracle 

# =================配置区域=================
CHECKPOINT_PATH = "checkpoints/eq2_83M_2M.pt"  # 请修改为你的实际路径
# =========================================

def build_and_calc_raw_delta(oracle, element_symbol, lattice_const=None):
    """
    构建表面+吸附剂，计算 Raw Delta E (E_total - E_slab)
    不涉及任何参考能，只看模型输出的能量差。
    """
    print(f"   正在计算 {element_symbol}(111) + O ...")
    
    # 1. 构建体系 (ASE 默认晶格常数通常够用，也可以手动指定)
    # size=(3,3,4) 保证覆盖度较低 (1/9 ML)
    slab = fcc111(element_symbol, size=(3, 3, 4), a=lattice_const, vacuum=10.0)
    slab.pbc = [True, True, True]
    
    # 设置 Tags
    tags = np.zeros(len(slab), dtype=int)
    z_pos = slab.get_positions()[:, 2]
    tags[z_pos > np.median(z_pos)] = 1
    slab.set_tags(tags)
    
    # 2. 计算纯表面能量 (Fix Slab)
    e_slab = oracle.compute_energy(slab, relax=False)
    
    # 3. 添加吸附剂 (fcc hollow)
    atoms_ads = slab.copy()
    # 注意：Au 的晶格常数大，Pd 的小，add_adsorbate 会自动处理高度吗？
    # 最好给一个通用的高度 guess，让模型自己弛豫
    add_adsorbate(atoms_ads, 'O', height=1.3, position='fcc')
    atoms_ads.set_tags(np.append(atoms_ads.get_tags()[:-1], 2))
    
    # 4. 弛豫吸附剂并计算总能
    # 使用 relax=True
    e_total = oracle.compute_energy(atoms_ads, relax=True)
    
    # Raw Delta = E_total - E_slab
    # 物理含义：把 O 放上去后，系统能量变了多少 (此时还没减去 O 的来源能量)
    raw_delta = e_total - e_slab
    return raw_delta

def run_transferability_test():
    print("========================================")
    print("多金属体系迁移性验证 (Pt / Pd / Au)")
    print("========================================")
    
    oracle = EquiformerV2Oracle(
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fmax=0.05
    )
    
    # --- 1. 定义测试集 ---
    # 理论值参考: DFT PBE/RPBE 文献
    test_cases = [
        {"element": "Pt", "target": -1.50, "role": "Calibration (校准)"},
        {"element": "Pd", "target": -1.45, "role": "Validation (验证-强)"}, 
        {"element": "Au", "target": +0.20, "role": "Validation (验证-弱)"} 
    ]
    # 注：Au(111)上O吸附是吸热的(不稳定)，或者是极弱的放热(-0.05)，取决于具体泛函。
    # 这里我们设定为 +0.20 作为区分度，只要它是正值或接近0，就说明模型是对的。
    
    results = []
    
    # --- 2. 收集 Raw Data ---
    raw_deltas = {}
    for case in test_cases:
        elem = case["element"]
        delta = build_and_calc_raw_delta(oracle, elem)
        raw_deltas[elem] = delta
        print(f"   -> {elem} Raw Delta E: {delta:.4f} eV")

    # --- 3. 执行校准 (以 Pt 为基准) ---
    print("\n[执行校准]")
    # E_ads = Delta - E_ref  =>  E_ref = Delta - E_ads_target
    pt_delta = raw_deltas["Pt"]
    pt_target = -1.50
    calibrated_ref = pt_delta - pt_target
    
    print(f"基于 Pt(111) 的校准参考能 (E_ref_O): {calibrated_ref:.4f} eV")
    
    # --- 4. 验证其他金属 ---
    print("\n[验证结果]")
    print(f"{'Metal':<6} | {'Role':<15} | {'Raw Delta':<10} | {'Pred E_ads':<10} | {'Lit. Target':<12} | {'Diff':<6}")
    print("-" * 75)
    
    for case in test_cases:
        elem = case["element"]
        raw = raw_deltas[elem]
        # 使用统一的 calibrated_ref 计算预测值
        pred_ads = raw - calibrated_ref
        target = case["target"]
        diff = pred_ads - target
        
        print(f"{elem:<6} | {case['role']:<15} | {raw:<10.4f} | {pred_ads:<10.4f} | {target:<12.2f} | {diff:+.2f}")
        
    print("-" * 75)
    
    # --- 5. 自动判定 ---
    print("\n[结论判定]")
    pred_au = raw_deltas["Au"] - calibrated_ref
    pred_pd = raw_deltas["Pd"] - calibrated_ref
    
    if pred_au > -0.5 and pred_pd < -1.0:
        print("✅ 通过！模型成功区分了惰性金属(Au)和活性金属(Pd)。")
        print(f"   Au 预测值 ({pred_au:.2f} eV) 明显高于 Pd 预测值 ({pred_pd:.2f} eV)。")
        print("   你可以放心地使用这个 calibrated_ref 进行大规模筛选。")
    else:
        print("❌ 失败。模型未能正确区分活性差异。")
        print("   可能原因：模型权重问题 (2M checkpiont 可能在 Au 上表现不佳)。")

if __name__ == "__main__":
    run_transferability_test()