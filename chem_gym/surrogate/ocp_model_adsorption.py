import torch
import numpy as np
from pathlib import Path
from ase import Atoms
from typing import List, Union, Optional, Tuple
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import molecule
import sys
import warnings

# 忽略 torch.load 的 Future warning
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 动态寻找 Fairchem 导入路径 ---
FAIRCHEM_AVAILABLE = False
AtomsToGraphs = None
registry = None
Batch = None

try:
    # 1. 尝试导入基础依赖
    from torch_geometric.data import Batch

    # 2. 尝试导入 Registry
    try:
        from fairchem.core.common.registry import registry
    except ImportError:
        try:
            from fairchem.core.registry import registry
        except ImportError:
            print("Warning: Could not find 'registry' in fairchem.core")

    # 3. 关键修复：显式导入模型以触发注册 (Registry)
    try:
        import fairchem.core.models
        import fairchem.core.models.equiformer_v2
    except Exception as e_models:
        print(f"Warning: 'import fairchem.core.models' failed: {e_models}")

    # 4. 尝试导入 AtomsToGraphs
    try:
        from fairchem.core.preprocessing import AtomsToGraphs
    except ImportError:
        try:
            from fairchem.core.datasets import AtomsToGraphs
        except ImportError:
            try:
                import fairchem.core
                if hasattr(fairchem.core, "AtomsToGraphs"):
                    AtomsToGraphs = fairchem.core.AtomsToGraphs
            except ImportError:
                pass

    if registry is not None and AtomsToGraphs is not None:
        FAIRCHEM_AVAILABLE = True

except ImportError as e:
    print(f"Warning: Failed to import OCP dependencies: {e}")


# 预定义的吸附分子
PREDEFINED_ADSORBATES = {
    "H": (molecule("H"), "hydrogen"),
    "O": (molecule("O"), "oxygen"),
    "OH": (molecule("OH"), "hydroxyl"),
    "CO": (molecule("CO"), "carbon_monoxide"),
    "N": (molecule("N"), "nitrogen"),
    "NH": (molecule("NH"), "imine"),
    "NH2": (molecule("NH2"), "amide"),
    "CH3": (molecule("CH3"), "methyl"),
 }


class EquiformerV2AdsorptionOracle:
    """
    增强版 EquiformerV2 Oracle，支持吸附能预测

    可以预测：
    - 表面能量 (E_surface)
    - 吸附物能量 (E_adsorbate)
    - 吸附后总能量 (E_total)
    - 吸附能 (ΔE_ads = E_total - E_surface - E_adsorbate)
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 adsorbate_prefs: Optional[List[str]] = None):
        if not FAIRCHEM_AVAILABLE:
            raise ImportError(
                "Failed to initialize OCP. Ensure 'fairchem-core' is installed properly."
            )

        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.checkpoint_path = checkpoint_path

        # 设置可用的吸附物类型
        self.available_adsorbates = list(PREDEFINED_ADSORBATES.keys())
        if adsorbate_prefs:
            self.available_adsorbates = [a for a in adsorbate_prefs
                                         if a in PREDEFINED_ADSORBATES]

        print(f"Loading EquiformerV2 from {checkpoint_path} to {self.device}...")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # --- [FIX] 加载 Normalizers (均值和标准差) ---
            self.energy_mean = 0.0
            self.energy_std = 1.0

            if "normalizers" in checkpoint:
                # 新的OCP格式使用 'target' 作为键名
                if "target" in checkpoint["normalizers"]:
                    normalizers = checkpoint["normalizers"]["target"]
                    self.energy_mean = normalizers.get("mean", 0.0)
                    self.energy_std = normalizers.get("std", 1.0)
                # 旧的格式可能使用 'energy'
                elif "energy" in checkpoint["normalizers"]:
                    normalizers = checkpoint["normalizers"]["energy"]
                    self.energy_mean = normalizers.get("mean", 0.0)
                    self.energy_std = normalizers.get("std", 1.0)

                # 确保转为 Tensor 并移到 GPU
                if torch.is_tensor(self.energy_mean):
                    self.energy_mean = self.energy_mean.to(self.device, dtype=torch.float32)
                else:
                    self.energy_mean = torch.tensor(self.energy_mean, dtype=torch.float32, device=self.device)

                if torch.is_tensor(self.energy_std):
                    self.energy_std = self.energy_std.to(self.device, dtype=torch.float32)
                else:
                    self.energy_std = torch.tensor(self.energy_std, dtype=torch.float32, device=self.device)

                print(f"✓ Loaded normalizers: mean={self.energy_mean.item():.4f}, std={self.energy_std.item():.4f}")
            else:
                print("⚠ WARNING: No normalizers found in checkpoint! Predictions will be unscaled (wrong).")
                print("  This usually happens if the checkpoint doesn't contain 'normalizers'.")
                print("  You may need to find mean/std from model documentation.")
            # --- [FIX END] ---

            # 处理 Config
            if "config" in checkpoint:
                config = checkpoint["config"]
            else:
                config = checkpoint.get("args", {})

            # 获取模型名称
            if "model_attributes" in config:
                model_name = config["model_attributes"].get("model", config.get("model"))
                model_args = config["model_attributes"]
            else:
                model_name = config.get("model", "equiformer_v2")
                model_args = config.get("model_args", config)

            # 获取模型类
            model_cls = registry.get_model_class(model_name)
            if model_cls is None:
                raise ValueError(f"Model '{model_name}' not found in registry.")

            self.model = model_cls(**model_args).to(self.device)

            # 加载权重
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()

            # 初始化图转换器
            self.a2g = AtomsToGraphs(
                max_neigh=50,
                radius=6.0,
                r_energy=False,
                r_forces=False,
                r_distances=False,
                r_fixed=True,
            )

            # 初始化 ASE 到 pymatgen 的转换器
            self.adaptor = AseAtomsAdaptor()

            print("EquiformerV2 Adsorption Oracle loaded successfully!")

        except Exception as e:
            print(f"Error loading OCP model: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def predict_energy(self, atoms) -> float:
        """
        计算给定原子结构的总能量（应用反归一化）
        """
        ase_atoms = self._convert_to_ase(atoms)
        self._set_tags(ase_atoms)

        data_list = self.a2g.convert_all([ase_atoms], disable_tqdm=True)
        batch = Batch.from_data_list(data_list).to(self.device)

        with torch.no_grad():
            output = self.model(batch)

        # --- [FIX] 应用反归一化 ---
        # 公式: E_real = E_raw * std + mean
        raw_energy = output["energy"]
        energy = raw_energy * self.energy_std + self.energy_mean
        # --- [FIX END] ---

        return energy.item()

    def predict_surface_energy(self, atoms: Atoms) -> float:
        """
        预测表面能量（移除吸附物后的结构）

        Args:
            atoms: ASE Atoms对象，可能包含吸附物

        Returns:
            表面能量 (eV)
        """
        # 创建只包含表面的副本（移除所有吸附物）
        surface_atoms = self._extract_surface(atoms)
        if len(surface_atoms) == 0:
            # 如果没有识别到表面原子，返回原始能量
            return self.predict_energy(atoms)

        return self.predict_energy(surface_atoms)

    def predict_adsorption_energy(self, atoms: Atoms) -> Tuple[float, dict]:
        """
        预测特定吸附构型的吸附能

        Args:
            atoms: ASE Atoms对象，包含表面和吸附物

        Returns:
            (adsorption_energy, info_dict)
            adsorption_energy: 吸附能 = E_total - E_surface - E_adsorbate
            info_dict: 包含详细能量分解信息
        """
        # 总能量
        e_total = self.predict_energy(atoms)

        # 表面能量（移除吸附物）
        e_surface = self.predict_surface_energy(atoms)

        # 提取吸附物能量（单独计算）
        adsorbate_atoms = self._extract_adsorbate(atoms)
        e_adsorbate = 0.0
        if len(adsorbate_atoms) > 0:
            e_adsorbate = self.predict_energy(adsorbate_atoms)

        # 吸附能
        adsorption_energy = e_total - e_surface - e_adsorbate

        info = {
            "e_total": e_total,
            "e_surface": e_surface,
            "e_adsorbate": e_adsorbate,
            "adsorption_energy": adsorption_energy
        }

        return adsorption_energy, info

    def find_adsorption_sites(self, surface_atoms: Atoms,
                            adsorbate: Optional[str] = None,
                            height: float = 2.0) -> dict:
        """
        找到表面的吸附位点

        Args:
            surface_atoms: 表面结构（ASE Atoms）
            adsorbate: 吸附物类型（从 PREDEFINED_ADSORBATES 中选择）
            height: 吸附物放置的高度

        Returns:
            dict with keys: "ontop", "bridge", "hollow"
            每个包含坐标列表和对应的能量预测
        """
        # 转换为 pymatgen 结构
        structure = self.adaptor.get_structure(surface_atoms)

        # 使用 pymatgen 的 AdsorbateSiteFinder
        asf = AdsorbateSiteFinder(structure)

        # 获取所有类型的吸附位点
        sites = {
            "ontop": asf.find_adsorption_sites(distance=height,
                                              put_inside=False,
                                              symm_reduce=0.01)["ontop"],
            "bridge": asf.find_adsorption_sites(distance=height,
                                               put_inside=False,
                                               symm_reduce=0.01)["bridge"],
            "hollow": asf.find_adsorption_sites(distance=height,
                                               put_inside=False,
                                               symm_reduce=0.01)["hollow"]
        }

        # 如果指定了吸附物类型，预测每个位点的吸附能
        if adsorbate and adsorbate in self.available_adsorbates:
            adsorbate_mol, _ = PREDEFINED_ADSORBATES[adsorbate]

            for site_type, site_coords in sites.items():
                energies = []

                for coord in site_coords:
                    # 创建吸附结构
                    adsorbed_atoms = self._place_adsorbate(surface_atoms.copy(),
                                                         adsorbate_mol,
                                                         coord,
                                                         height)
                    # 预测吸附能
                    e_ads, _ = self.predict_adsorption_energy(adsorbed_atoms)
                    energies.append(float(e_ads))

                # 添加到结果
                sites[site_type] = {
                    "coordinates": site_coords,
                    "adsorption_energies": energies
                }

        return sites

    def get_best_adsorption_site(self, surface_atoms: Atoms,
                                adsorbate: str) -> dict:
        """
        找到特定吸附物在表面的最佳吸附位点

        Args:
            surface_atoms: 表面结构
            adsorbate: 吸附物类型

        Returns:
            dict with keys: "site_type", "coordinates", "adsorption_energy"
        """
        if adsorbate not in self.available_adsorbates:
            raise ValueError(f"Unsupported adsorbate: {adsorbate}. "
                            f"Available: {self.available_adsorbates}")

        # 找到所有位点并预测能量
        sites = self.find_adsorption_sites(surface_atoms, adsorbate)

        best_site = None
        best_energy = float('inf')

        for site_type, site_data in sites.items():
            if isinstance(site_data, dict) and "adsorption_energies" in site_data:
                for i, energy in enumerate(site_data["adsorption_energies"]):
                    if energy < best_energy:  # 更低的能量 = 更稳定的吸附
                        best_energy = energy
                        best_site = {
                            "site_type": site_type,
                            "coordinates": site_data["coordinates"][i],
                            "adsorption_energy": energy
                        }

        return best_site

    def _convert_to_ase(self, atoms) -> Atoms:
        """转换输入到 ASE Atoms"""
        if hasattr(atoms, 'to_ase_atoms'):
            return atoms.to_ase_atoms()
        elif hasattr(atoms, 'get_positions'):
            return atoms
        else:
            raise TypeError(f"Unsupported atom structure type: {type(atoms)}")

    def _set_tags(self, atoms: Atoms):
        """为原子设置标签（用于OCP模型）"""
        tags = np.ones(len(atoms), dtype=np.int64)

        if hasattr(atoms, 'constraints') and atoms.constraints:
            for constraint in atoms.constraints:
                if hasattr(constraint, 'index'):
                    tags[constraint.index] = 0

        atoms.set_tags(tags)

    def _extract_surface(self, atoms: Atoms) -> Atoms:
        """从结构中分离表面原子（移除吸附物）"""
        # 简单启发式：移除z坐标最高的原子（假设是吸附物）
        # 实际应用中可能需要更复杂的逻辑
        positions = atoms.get_positions()
        z_coords = positions[:, 2]

        # 计算z方向的统计信息
        z_mean = np.mean(z_coords)
        z_std = np.std(z_coords)

        # 如果z坐标的标准差很大，说明可能有吸附物
        if z_std > 0.5:  # 阈值
            # 考虑移除高于平均值 + 2*标准差的原子
            surface_mask = z_coords <= (z_mean + 2 * z_std)
            surface_indices = np.where(surface_mask)[0]

            if len(surface_indices) < len(atoms):
                return atoms[surface_indices]

        # 如果没有明显的吸附物，返回原结构
        return atoms

    def _extract_adsorbate(self, atoms: Atoms) -> Atoms:
        """从结构中分离吸附物原子"""
        positions = atoms.get_positions()
        z_coords = positions[:, 2]

        z_mean = np.mean(z_coords)
        z_std = np.std(z_coords)

        # 如果z坐标的标准差很大，分离顶部原子作为吸附物
        if z_std > 0.5:
            adsorbate_mask = z_coords > (z_mean + 1.5 * z_std)
            adsorbate_indices = np.where(adsorbate_mask)[0]

            if len(adsorbate_indices) > 0:
                return atoms[adsorbate_indices]

        # 返回空结构
        return Atoms()

    def _place_adsorbate(self, surface_atoms: Atoms,
                        adsorbate_mol: Atoms,
                        site_coord: np.ndarray,
                        height: float = 2.0) -> Atoms:
        """
        在指定位点上放置吸附物

        Args:
            surface_atoms: 表面结构
            adsorbate_mol: 吸附物分子
            site_coord: 吸附位点的坐标
            height: 吸附物放置的高度
        """
        # 复制吸附物并移动到位点
        adsorbate = adsorbate_mol.copy()

        # 计算吸附物的质心
        mol_centroid = np.mean(adsorbate.get_positions(), axis=0)

        # 移动吸附物使最低原子位于指定高度
        positions = adsorbate.get_positions()
        z_min = np.min(positions[:, 2])

        # 位移
        translation = site_coord - mol_centroid
        translation[2] = height - z_min  # 设置z方向高度

        adsorbate.translate(translation)

        # 合并到表面
        combined = surface_atoms + adsorbate

        return combined


if __name__ == "__main__":
    from ase.build import fcc111

    # 测试代码
    try:
        # 创建一个表面
        surface = fcc111('Cu', size=(4, 4, 3), vacuum=10.0)
        surface.set_pbc([True, True, True])

        ckpt = "checkpoints/eq2_83M_2M.pt"
        if not Path(ckpt).exists():
            ckpt = "../../checkpoints/eq2_83M_2M.pt"

        if Path(ckpt).exists():
            oracle = EquiformerV2AdsorptionOracle(ckpt, device="cuda")

            # 测试1: 预测表面能量
            e_surface = oracle.predict_surface_energy(surface)
            print(f"✓ Surface energy: {e_surface:.4f} eV")

            # 测试2: 找到吸附位点
            sites = oracle.find_adsorption_sites(surface, adsorbate="H")
            print(f"✓ Found adsorption sites: {len(sites['ontop']['coordinates'])} ontop, "
                  f"{len(sites['bridge']['coordinates'])} bridge, "
                  f"{len(sites['hollow']['coordinates'])} hollow")

            # 测试3: 找到最佳吸附位点
            best_site = oracle.get_best_adsorption_site(surface, "OH")
            print(f"✓ Best adsorption site for OH: {best_site['site_type']} "
                  f"at {best_site['coordinates']} with energy {best_site['adsorption_energy']:.4f} eV")

            print("\n✓ EquiformerV2 Adsorption Oracle is working correctly!")
        else:
            print(f"Checkpoint not found at {ckpt}")

    except Exception as e:
        import traceback
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
