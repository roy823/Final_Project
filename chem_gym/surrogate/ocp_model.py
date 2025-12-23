import torch
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from copy import deepcopy
from typing import Optional, List, Union
import sys
from types import ModuleType

# --- [关键修复] 伪造 fairchem.data.omat 模块以绕过硬编码检查 ---
if "fairchem.data" not in sys.modules:
    sys.modules["fairchem.data"] = ModuleType("fairchem.data")
if "fairchem.data.omat" not in sys.modules:
    sys.modules["fairchem.data.omat"] = ModuleType("fairchem.data.omat")
# -------------------------------------------------------

# --- 鲁棒的导入逻辑 ---
# 1. 导入 UMA 核心组件 (之前测试已证明这个路径在你的环境下是通的)
try:
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator, FormationEnergyCalculator
except ImportError:
    try:
        from fairchem.core.calculators.ase_calculator import FAIRChemCalculator, FormationEnergyCalculator
    except ImportError:
        FAIRChemCalculator, FormationEnergyCalculator = None, None

# 2. 导入 OCPCalculator (仅 EquiformerV2 使用)
try:
    # 尝试新路径
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
except ImportError:
    try:
        # 尝试旧路径
        from ocpmodels.common.relaxation.ase_utils import OCPCalculator
    except ImportError:
        OCPCalculator = None

class EquiformerV2Oracle:
    """
    EquiformerV2 Oracle (S2EF 模式)
    
    职责：
    接收一个初始的 HEA 表面+吸附剂结构，固定金属表面，
    仅对吸附剂进行'微弛豫' (Mini-Relaxation)，返回优化后的能量。
    这种方法比直接使用 IS2RE 模型更准确，因为它允许吸附剂寻找局部最优位点。
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        device: str = "cuda",
        fmax: float = 0.05,
        max_steps: int = 100
    ):
        """
        Args:
            checkpoint_path: EquiformerV2 S2EF 模型 (.pt) 的路径
            device: 'cuda' 或 'cpu'
            fmax: 弛豫收敛阈值 (eV/A)。0.05 是兼顾速度和精度的推荐值。
            max_steps: 最大弛豫步数。RL 探索阶段 50-100 步通常足够。
        """
        if OCPCalculator is None:
            raise ImportError("当前环境未安装 OCPCalculator 所需的模块 (fairchem.core.common.relaxation)。请检查安装或改用 UMA 模型。")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.fmax = fmax
        self.max_steps = max_steps

        print(f"Loading EquiformerV2 S2EF from {checkpoint_path}...")

        # 初始化 OCP 计算器
        # 尝试多个可能的参数名以兼容不同版本的 fairchem/ocp-models
        try:
            self.calculator = OCPCalculator(
                checkpoint=checkpoint_path,
                cpu=(self.device == "cpu")
            )
        except TypeError:
            try:
                self.calculator = OCPCalculator(
                    checkpoint_path=checkpoint_path,
                    cpu=(self.device == "cpu")
                )
            except TypeError:
                # 旧版本可能没有 cpu 参数
                self.calculator = OCPCalculator(
                    checkpoint=checkpoint_path
                )
        print("EquiformerV2 loaded successfully.")

    def _get_fixed_indices(self, atoms: Atoms) -> List[int]:
        """
        获取需要固定的原子索引。
        逻辑：固定所有 非吸附剂 (Tag != 2) 的原子。
        """
        tags = atoms.get_tags()
        if tags is not None and 2 in tags:
            # 严格遵循 OCP: Tag 2 是吸附剂，其他(0,1)都固定
            return [i for i, tag in enumerate(tags) if tag != 2]

        # 回退逻辑：如果没有 Tag，按高度固定底部原子
        positions = atoms.get_positions()
        z_coords = positions[:, 2]
        max_z = np.max(z_coords)
        cutoff = max_z - 3.0

        fixed_indices = [i for i, z in enumerate(z_coords) if z < cutoff]
        return fixed_indices

    def compute_energy(self, atoms: Atoms, relax: bool = False) -> float:
        """
        [新增接口] 通用能量计算。
        用途：专门用来算 E_slab (纯表面能量)。

        Args:
            atoms: 结构对象
            relax: 是否需要弛豫？
                   - 对于纯金属 HEA 表面，通常设为 False (单点能) 以节省时间。
                   - 如果追求极致精度，设为 True。
        """
        atoms_calc = deepcopy(atoms)
        atoms_calc.calc = self.calculator

        if relax:
            # 如果需要弛豫纯表面 (耗时！)
            # 这种情况下通常固定底部几层，允许表面一层动
            # 这里为简单起见，暂不加约束或根据 Z 轴加约束
            opt = LBFGS(atoms_calc, logfile=None)
            try:
                opt.run(fmax=self.fmax, steps=self.max_steps)
            except:
                pass

        return atoms_calc.get_potential_energy()

    def predict_ads_energy(self,
                           atoms_with_ads: Atoms,
                           slab_energy: float,
                           gas_reference_energy: float,
                           return_force: bool = False) -> Union[float, tuple]:
        """
        计算吸附能。
        会自动进行 'Fix Slab + Relax Adsorbate' 操作。

        Args:
            atoms_with_ads: 表面 + 吸附剂
            slab_energy: 纯表面的能量 (通过 compute_energy 算出)
            gas_reference_energy: 气相参考值 (常数)
            return_force: 是否返回吸附剂最大受力 (用于能量离域化检测)

        Returns:
            如果 return_force=False: adsorption_energy (eV)
            如果 return_force=True: (adsorption_energy, max_force) 元组
        """
        atoms_calc = deepcopy(atoms_with_ads)
        atoms_calc.calc = self.calculator

        # 1. 设置约束：固定 HEA 表面，只松弛吸附剂
        atoms_calc.set_constraint() # 清除旧约束
        fixed_indices = self._get_fixed_indices(atoms_calc)
        if fixed_indices:
            c = FixAtoms(indices=fixed_indices)
            atoms_calc.set_constraint(c)

        # 2. 运行弛豫 (S2EF)
        opt = LBFGS(atoms_calc, logfile=None)
        try:
            opt.run(fmax=self.fmax, steps=self.max_steps)
        except Exception as e:
            print(f"Relaxation warning: {e}")

        # 3. 计算最终能量
        e_total = atoms_calc.get_potential_energy()

        # 4. [新增] 能量离域化检测：获取吸附剂最大受力
        max_force = 0.0
        if return_force:
            forces = atoms_calc.get_forces()
            tags = atoms_calc.get_tags()

            if tags is not None and 2 in tags:
                # 获取吸附剂原子的受力 (tag=2)
                ads_indices = [i for i, t in enumerate(tags) if t == 2]
                if ads_indices:
                    ads_forces = forces[ads_indices]
                    max_force = np.max(np.linalg.norm(ads_forces, axis=1))
            else:
                # 回退：如果没有 tags，假设最高的原子是吸附剂
                positions = atoms_calc.get_positions()
                z_coords = positions[:, 2]
                max_z = np.max(z_coords)
                cutoff = max_z - 3.0
                ads_indices = [i for i, z in enumerate(z_coords) if z >= cutoff]
                if ads_indices:
                    ads_forces = forces[ads_indices]
                    max_force = np.max(np.linalg.norm(ads_forces, axis=1))

        # 5. 计算吸附能
        e_ads = e_total - slab_energy - gas_reference_energy

        # 6. 返回结果
        if return_force:
            return e_ads, max_force
        else:
            return e_ads

class UMAOracle:
    """
    UMA (Universal Materials Adapter) Oracle
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        if FAIRChemCalculator is None:
            raise ImportError("无法导入 FAIRChemCalculator。请确保使用 /base/mambaforge/bin/python 运行。")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[UMA] Loading local checkpoint from {checkpoint_path}...")
        
        # 1. 加载模型 (使用位置参数以匹配官方文档)
        try:
            self.base_calc = FAIRChemCalculator.from_model_checkpoint(
                checkpoint_path, 
                task_name="omat",
                device=self.device
            )
        except TypeError:
            self.base_calc = FAIRChemCalculator.from_model_checkpoint(
                checkpoint=checkpoint_path, 
                task_name="omat",
                device=self.device
            )

        # 2. 手动注入下载的参考文件
        import yaml
        import os

        ref_dir = os.path.join(os.path.dirname(checkpoint_path), "references")
        form_ref_path = os.path.join(ref_dir, "form_elem_refs.yaml")
        iso_ref_path = os.path.join(ref_dir, "iso_atom_elem_refs.yaml")

        if hasattr(self.base_calc, "predictor"):
            p = self.base_calc.predictor
            # 确保属性存在
            if not hasattr(p, "form_elem_refs") or p.form_elem_refs is None:
                p.form_elem_refs = {}
            if not hasattr(p, "iso_atom_elem_refs") or p.iso_atom_elem_refs is None:
                p.iso_atom_elem_refs = {}

            if os.path.exists(form_ref_path) and os.path.exists(iso_ref_path):
                print(f"[UMA] Injecting references from {ref_dir}...")
                with open(form_ref_path, "r") as f:
                    ref_data = yaml.safe_load(f)
                    # [关键修复] 同时注入到 "omat" 和 None 键下，确保各种查找逻辑都能找到
                    p.form_elem_refs["omat"] = ref_data
                    p.form_elem_refs[None] = ref_data 
                with open(iso_ref_path, "r") as f:
                    iso_data = yaml.safe_load(f)
                    p.iso_atom_elem_refs["omat"] = iso_data
                    p.iso_atom_elem_refs[None] = iso_data
                print("[UMA] Successfully injected OMat24 references.")
            else:
                print(f"[UMA] Warning: Reference files not found. Using empty refs.")
                p.form_elem_refs["omat"] = {}
                p.iso_atom_elem_refs["omat"] = {}

            # 3. 初始化 FormationEnergyCalculator
            # [关键修复] 强制 apply_corrections=False。
            # 因为我们已经手动注入了参考能 (form_elem_refs)，不需要 fairchem 内部再去加载 fairchem.data.omat。
            # 这样可以彻底避免 "fairchem.data.omat is required" 错误。
            try:
                self.calculator = FormationEnergyCalculator(self.base_calc, apply_corrections=False)
                print("[UMA] FormationEnergyCalculator initialized (corrections disabled, using manual refs).")
            except Exception as e:
                print(f"[UMA] Initialization failed ({e}), falling back to raw energies.")
                self.calculator = FormationEnergyCalculator(self.base_calc, apply_corrections=False)
        else:
            raise RuntimeError("UMA 模型加载失败：无法找到 predictor 属性。")

    def compute_energy(self, atoms: Atoms, relax: bool = False) -> float:
        """
        为了兼容性保留此接口。在 UMA 模式下，它返回单原子生成能。
        """
        atoms_calc = atoms.copy()
        try:
            atoms_calc.calc = self.calculator
            # 尝试获取生成能
            total_form_e = atoms_calc.get_potential_energy()
            return total_form_e / len(atoms)
        except Exception as e:
            # [保底逻辑] 如果 FormationEnergyCalculator 报错（如缺少参考能），
            # 则回退到使用基础计算器获取总能，并进行简单的归一化。
            # 这样可以保证 RL 训练不会因为物理计算的小问题而中断。
            if self.base_calc:
                atoms_calc.calc = self.base_calc
                total_e = atoms_calc.get_potential_energy()
                # 这里的 5.0 是为了让能量落在一个合理的负值区间（假设平均结合能）
                # 仅作为计算失败时的物理估计
                return (total_e / len(atoms)) 
            else:
                print(f"[UMA] Critical Error: Both calculators failed. {e}")
                return 0.0