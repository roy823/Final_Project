import sys
import typing

# --- 热修复: 为 Python 3.10 注入 NotRequired ---
# 某些库(如 mp-api)在 Python 3.10 下可能会错误地从 typing 导入 NotRequired
if sys.version_info < (3, 11):
    try:
        from typing_extensions import NotRequired
        # 强制将其注入到 typing 模块中，欺骗后续导入的库
        if not hasattr(typing, "NotRequired"):
            typing.NotRequired = NotRequired
    except ImportError:
        pass
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple, Any

# --- Pymatgen 核心导入 ---
try:
    from pymatgen.core import Structure, Lattice, Molecule
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.cif import CifWriter
    from pymatgen.analysis.local_env import CrystalNN
except ImportError:
    raise ImportError("需要安装 pymatgen: pip install pymatgen")

# --- MP API 导入 (用于获取真实数据) ---
try:
    # 新版 MP API
    from mp_api.client import MPRester
except ImportError as e:
    MPRester = None
    # [修改] 打印具体的错误信息，以便调试
    print(f"提示: 无法导入 'mp-api' (错误详情: {e})")
    print("      请尝试运行: /base/mambaforge/bin/python -m pip install mp-api")


class PymatgenInterface:
    """
    Chem-Gym 的 Pymatgen 接口模块。
    功能：
    1. 轨迹记录 (Trajectory Recording) -> 生成 CIF 动画
    2. 切片分析 (Slice Analysis) -> 分析每一层的成分
    3. 数据校准 (Data Calibration) -> 使用 API Key 获取真实晶格常数
    """

    # API KEY
    DEFAULT_API_KEY = "LWy9cEJNTrC8Bk1b5QEfsL6EU9tLnZiw"

    def __init__(self, api_key: str = None):
        self.api_key = api_key if api_key else self.DEFAULT_API_KEY
        self.adaptor = AseAtomsAdaptor()

    def ase_to_structure(self, ase_atoms) -> Structure:
        """将 ASE Atoms 转换为 Pymatgen Structure"""
        return self.adaptor.get_structure(ase_atoms)

    def save_trajectory_cif(self, structures: List[Structure], filename: str = "trajectory.cif"):
        """
        将一系列结构保存为多帧 CIF 文件 (可直接拖入 VESTA 播放动画)
        """
        print(f"[PMG] 正在保存 {len(structures)} 帧轨迹到 {filename} ...")
        with open(filename, "w") as f:
            for idx, s in enumerate(structures):
                # 写入 CIF 数据块头，VESTA 识别为不同的帧
                f.write(f"data_frame_{idx}\n")
                cw = CifWriter(s)
                f.write(cw.__str__())
                f.write("\n")
        print(f"[PMG] ✅ 保存成功: {filename}")

    def analyze_layers(self, structure: Structure, layer_height: float = 2.0) -> Dict[int, Dict[str, float]]:
        """
        [可视化切片核心]
        自动识别 Slab 的层数，并统计每一层的元素分布。
        """
        # 1. 获取所有原子 Z 坐标
        sites = sorted(structure, key=lambda x: x.coords[2])
        z_coords = np.array([s.coords[2] for s in sites])
        
        # 2. 简单的层聚类 (假设层间距 > 1.0 埃)
        layers = []
        if len(z_coords) > 0:
            current_layer = [z_coords[0]]
            for z in z_coords[1:]:
                if z - current_layer[-1] > 1.0: # 判定为新的一层
                    layers.append(np.mean(current_layer))
                    current_layer = [z]
                else:
                    current_layer.append(z)
            layers.append(np.mean(current_layer))

        # 3. 统计每层成分
        layer_stats = {}
        for layer_idx, z_center in enumerate(layers):
            # 找到属于该层的原子 (Z 坐标在中心附近)
            layer_sites = [
                s for s in structure 
                if abs(s.coords[2] - z_center) < 0.8
            ]
            
            composition = {}
            for s in layer_sites:
                el = s.specie.symbol
                composition[el] = composition.get(el, 0) + 1
            
            # 归一化比例
            total = len(layer_sites)
            comp_ratio = {k: round(v/total, 3) for k, v in composition.items()}
            
            layer_stats[layer_idx] = {
                "z_height": round(z_center, 2),
                "composition": comp_ratio,
                "count": composition
            }
            
        return layer_stats

    def fetch_real_lattice_constants(self, elements: List[str]) -> Dict[str, float]:
        """
        使用 Materials Project API 获取最稳定的 FCC 晶格常数。
        这比使用教科书的平均值更准确。
        """
        if MPRester is None:
            print("[PMG] ⚠️ 未安装 mp-api，无法连接 Materials Project。请 pip install mp-api")
            return {}

        print(f"[PMG] 正在通过 API ({self.api_key[:4]}...) 连接 Materials Project 查询晶格参数...")
        results = {}
        
        try:
            with MPRester(self.api_key) as mpr:
                for el in elements:
                    # 查询该元素最稳定的 FCC 结构
                    docs = mpr.materials.summary.search(
                        chemsys=el, 
                        crystal_system="Cubic", 
                        is_stable=True,
                        fields=["structure", "symmetry", "material_id"]
                    )
                    
                    # 筛选 Spacegroup 225 (Fm-3m, 即标准 FCC)
                    fcc_docs = [d for d in docs if d.symmetry.number == 225]
                    
                    if fcc_docs:
                        # 取最稳定的一个
                        best_doc = fcc_docs[0]
                        a = best_doc.structure.lattice.a
                        results[el] = round(a, 4)
                        print(f"   ✅ {el} (FCC): a = {a:.4f} Å (ID: {best_doc.material_id})")
                    else:
                        print(f"   ⚠️ 未找到 {el} 的标准 FCC 结构，跳过。")
                        
        except Exception as e:
            print(f"[PMG] API 查询失败: {e}")
            
        return results

# --- 使用示例 ---
if __name__ == "__main__":
    # 简单的本地测试
    try:
        from ase.build import fcc111
        atoms = fcc111('Pt', size=(4,4,4), vacuum=10.0)
        
        pmg = PymatgenInterface()
        
        # 1. 测试转换
        struct = pmg.ase_to_structure(atoms)
        print("结构转换成功:", struct.formula)
        
        # 2. 测试层分析
        stats = pmg.analyze_layers(struct)
        print("\n--- 切片分析 (Layer Analysis) ---")
        for idx, info in stats.items():
            print(f"Layer {idx} (Z={info['z_height']}Å): {info['composition']}")
            
        # 3. 测试 API (可选)
        # constants = pmg.fetch_real_lattice_constants(["Pt", "Cu"])
        # print("\n真实晶格常数:", constants)
        
    except ImportError:
        pass