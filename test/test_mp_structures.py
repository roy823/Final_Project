#!/usr/bin/env python3
"""
测试 Materials Project 下载的结构
"""

import sys
import os
from pathlib import Path

# 添加父目录到 Python 路径，以便导入 chem_gym
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
from chem_gym.surrogate.ocp_model import EquiformerV2Oracle
from chem_gym.surrogate.ensemble import SurrogateEnsemble

def download_mp_structure(mat_id, api_key=None):
    """下载 Materials Project 结构"""
    try:
        from pymatgen.ext.matproj import MPRester
    except ImportError:
        print("错误: 需要安装 pymatgen: pip install pymatgen")
        return None

    if not api_key:
        api_key = os.environ.get("MP_API_KEY")
        if not api_key:
            print(f"错误: 未设置 MP_API_KEY 环境变量，无法下载 {mat_id}")
            return None

    save_dir = Path("mp_structures")
    save_dir.mkdir(exist_ok=True)

    try:
        with MPRester(api_key) as mpr:
            print(f"  正在下载 {mat_id}...")
            structure = mpr.get_structure_by_material_id(mat_id)

            # 获取能量信息
            docs = mpr.materials.summary.search(material_ids=[mat_id])
            energy_per_atom = None
            formation_energy_per_atom = None
            spacegroup = None

            if docs:
                doc = docs[0]
                # 处理可能的字典或对象格式
                if isinstance(doc, dict):
                    energy_per_atom = doc.get('energy_per_atom')
                    formation_energy_per_atom = doc.get('formation_energy_per_atom')
                    spacegroup = doc.get('symmetry', {}).get('symbol') if doc.get('symmetry') else None
                else:
                    # 对象格式
                    energy_per_atom = getattr(doc, 'energy_per_atom', None)
                    formation_energy_per_atom = getattr(doc, 'formation_energy_per_atom', None)
                    spacegroup = getattr(doc.symmetry, 'symbol', None) if hasattr(doc, 'symmetry') else None

            data = {
                'material_id': mat_id,
                'structure': structure,
                'formula': structure.formula,
                'spacegroup': spacegroup,
                'energy_per_atom': energy_per_atom,
                'formation_energy_per_atom': formation_energy_per_atom,
            }

            # 保存
            pickle_file = save_dir / f"{mat_id}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(data, f)

            print(f"  ✓ 已保存到 {pickle_file}")
            if energy_per_atom:
                print(f"  ✓ DFT 能量: {energy_per_atom:.4f} eV/atom")
            return data

    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return None

def load_mp_structure(mat_id):
    """加载 Materials Project 结构（如果不存在则下载）"""
    structure_file = Path(f"mp_structures/{mat_id}.pkl")

    # 如果文件不存在，尝试下载
    if not structure_file.exists():
        api_key = os.environ.get("MP_API_KEY")
        if api_key:
            print(f"\n结构文件 {structure_file} 不存在，尝试下载...")
            data = download_mp_structure(mat_id, api_key)
            if data is None:
                raise FileNotFoundError(f"无法下载结构文件 {structure_file}")
            return data['structure']
        else:
            raise FileNotFoundError(f"结构文件 {structure_file} 不存在且未设置 MP_API_KEY")

    with open(structure_file, 'rb') as f:
        data = pickle.load(f)

    return data['structure']

def convert_to_ase(atoms):
    """将 pymatgen Structure 转换为 ASE Atoms"""
    if hasattr(atoms, 'to_ase_atoms'):
        return atoms.to_ase_atoms()
    return atoms

def main():
    print("="*80)
    print("Materials Project 结构能量预测测试 + DFT 验证")
    print("="*80)

    # 检查命令行参数
    use_real_oracle = len(sys.argv) > 1 and sys.argv[1].lower() in ['y', 'yes', 'true', '1']

    try:
        if use_real_oracle:
            oracle = EquiformerV2Oracle("checkpoints/eq2_83M_2M.pt", device="cuda")
            oracle_name = "EquiformerV2"
        else:
            # 使用简单的占位符
            oracle = SurrogateEnsemble()
            oracle_name = "SurrogateEnsemble"
    except Exception as e:
        print(f"无法加载 EquiformerV2: {e}")
        print("使用 SurrogateEnsemble 作为后备")
        oracle = SurrogateEnsemble()
        oracle_name = "SurrogateEnsemble"

    # 定义统一的 predict_energy 方法
    if hasattr(oracle, 'predict_energy'):
        # EquiformerV2Oracle 有 predict_energy 方法
        predict_fn = oracle.predict_energy
    elif hasattr(oracle, 'evaluate'):
        # SurrogateEnsemble 有 evaluate 方法
        def predict_fn(atoms):
            energy, _ = oracle.evaluate(atoms)
            return energy
    else:
        raise AttributeError("Oracle 对象没有 predict_energy 或 evaluate 方法")

    # 要测试的结构
    test_ids = ['mp-30', 'mp-23', 'mp-13', 'mp-134', 'mp-20305', 'mp-101']

    print(f"\n使用模型: {oracle_name}")
    print("="*80)

    results = []
    has_dft_results = False

    for mat_id in test_ids:
        try:
            # 加载结构时顺便获取 DFT 能量（如果可用）
            structure_file = Path(f"mp_structures/{mat_id}.pkl")
            if structure_file.exists():
                with open(structure_file, 'rb') as f:
                    data = pickle.load(f)
                structure = data['structure']
                dft_energy = data.get('energy_per_atom')
                if dft_energy is not None:
                    has_dft_results = True
            else:
                structure = load_mp_structure(mat_id)
                # 重新加载以获取最新数据（包含 DFT 能量）
                with open(structure_file, 'rb') as f:
                    data = pickle.load(f)
                dft_energy = data.get('energy_per_atom')
                if dft_energy is not None:
                    has_dft_results = True

            print(f"\n{mat_id}:")
            print(f"  化学式: {structure.formula}")
            print(f"  原子数: {len(structure)}")

            # 显示 DFT 参考值（如果有）
            if dft_energy is not None:
                print(f"  DFT 参考能量: {dft_energy:.4f} eV/atom")

            # 转换为 ASE Atoms（如果需要）
            ase_structure = convert_to_ase(structure)

            # 预测能量
            energy = predict_fn(ase_structure)
            e_per_atom = energy / len(structure)
            print(f"  {oracle_name} 预测: {e_per_atom:.4f} eV/atom")

            # 计算误差（如果有 DFT 参考）
            error = None
            if dft_energy is not None:
                error = e_per_atom - dft_energy
                print(f"  误差: {error:.4f} eV/atom")

                # 计算相对误差
                if abs(dft_energy) > 1e-6:
                    rel_error = abs(error) / abs(dft_energy) * 100
                    print(f"  相对误差: {rel_error:.2f}%")

            results.append({
                'mat_id': mat_id,
                'formula': structure.formula,
                'n_atoms': len(structure),
                'pred_energy': e_per_atom,
                'dft_energy': dft_energy,
                'error': error
            })

        except Exception as e:
            print(f"\n{mat_id}: ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("总结:")
    print("-" * 80)

    # 按预测能量排序
    results.sort(key=lambda x: x['pred_energy'])

    if has_dft_results:
        # 使用更详细的表格格式（包含 DFT 参考值）
        print(f" {'ID':8s} | {'Formula':10s} | {'N':3s} | {'Pred (eV)':>10s} | {'DFT (eV)':>10s} | {'Error (eV)':>11s} | {'Rel Err':>8s}")
        print("-" * 80)
        for r in results:
            dft_str = f"{r['dft_energy']:.4f}" if r['dft_energy'] is not None else "N/A"
            error_str = f"{r['error']:.4f}" if r['error'] is not None else "N/A"
            rel_error_str = ""
            if r['error'] is not None and r['dft_energy'] and abs(r['dft_energy']) > 1e-6:
                rel_error = abs(r['error']) / abs(r['dft_energy']) * 100
                rel_error_str = f"{rel_error:.2f}%"

            print(f" {r['mat_id']:8s} | {r['formula']:10s} | {r['n_atoms']:3d} | " +
                  f"{r['pred_energy']:10.4f} | {dft_str:>10s} | {error_str:>11s} | {rel_error_str:>8s}")

        # 计算平均误差
        errors = [r['error'] for r in results if r['error'] is not None]
        if errors:
            print("\n" + "-" * 80)
            print(f"平均绝对误差 (MAE): {sum(abs(e) for e in errors) / len(errors):.4f} eV/atom")
            print(f"均方根误差 (RMSE): {(sum(e**2 for e in errors) / len(errors))**0.5:.4f} eV/atom")
    else:
        # 简单排序格式（没有 DFT 参考值）
        print(f" {'ID':8s} | {'Formula':10s} | {'N':3s} | {'Energy (eV)':>12s}")
        print("-" * 80)
        for r in results:
            print(f" {r['mat_id']:8s} | {r['formula']:10s} | {r['n_atoms']:3d} | {r['pred_energy']:12.4f}")

    print("\n" + "="*80)
    print("✅ MP 结构测试完成！")
    print("="*80)

if __name__ == "__main__":
    main()
