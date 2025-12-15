import torch
import numpy as np
from pathlib import Path
from ase import Atoms
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
    # 必须显式导入 fairchem.core.models，否则 registry 会是空的
    try:
        import fairchem.core.models
        # [新增] 显式导入 equiformer_v2 模块以触发注册
        # 某些版本的 fairchem 不会在导入 models 时自动导入所有子模块，导致 registry 为空
        import fairchem.core.models.equiformer_v2
    except Exception as e_models:
        print(f"Warning: 'import fairchem.core.models' failed: {e_models}")
        # 备选：尝试导入具体模型文件
        try:
            from fairchem.core.models import equiformer_v2
        except Exception as e_eq2:
            print(f"Warning: Failed to import equiformer_v2 specifically: {e_eq2}")

    # 4. 尝试导入 AtomsToGraphs
    # 既然之前的诊断脚本没找到，我们这里尽可能尝试所有可能的路径
    try:
        # Path A: 标准路径
        from fairchem.core.preprocessing import AtomsToGraphs
    except ImportError:
        try:
            # Path B: 兼容路径
            from fairchem.core.datasets import AtomsToGraphs
        except ImportError:
            try:
                # Path C: 顶层
                import fairchem.core
                if hasattr(fairchem.core, "AtomsToGraphs"):
                    AtomsToGraphs = fairchem.core.AtomsToGraphs
                # Path D: 可能是 common.utils?
                elif hasattr(fairchem.core.common, "utils") and hasattr(fairchem.core.common.utils, "AtomsToGraphs"):
                    AtomsToGraphs = fairchem.core.common.utils.AtomsToGraphs
            except ImportError:
                pass

    if registry is not None and AtomsToGraphs is not None:
        FAIRCHEM_AVAILABLE = True

except ImportError as e:
    print(f"Warning: Failed to import OCP dependencies: {e}")


class EquiformerV2Oracle:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        if not FAIRCHEM_AVAILABLE:
            raise ImportError(
                "Failed to initialize OCP. Ensure 'fairchem-core' is installed properly."
            )

        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.checkpoint_path = checkpoint_path
        
        print(f"Loading EquiformerV2 from {checkpoint_path} to {self.device}...")
        
        try:
            # 加载 Checkpoint
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
            # 注意：这里删除了之前导致 Crash 的 registry.get_model_names() 调试代码
            try:
                model_cls = registry.get_model_class(model_name)
            except Exception as e:
                # 如果 get_model_class 失败（比如 registry 是空的），我们再抛出错误
                print(f"CRITICAL: Failed to get model class '{model_name}' from registry.")
                print("This usually means 'import fairchem.core.models' failed silently.")
                raise e

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
                r_energy=False,  # Don't read energy from atoms (we're predicting it)
                r_forces=False,  # Don't read forces from atoms
                r_distances=False,
                r_fixed=True,
            )
            print("EquiformerV2 loaded successfully!")
            
        except Exception as e:
            print(f"Error loading OCP model: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def predict_energy(self, atoms) -> float:
        """
        计算给定原子结构的总能量

        支持 ASE Atoms 和 pymatgen Structure 两种对象
        """
        # 如果是 pymatgen Structure，转换为 ASE Atoms
        ase_atoms = atoms

        if hasattr(atoms, 'to_ase_atoms'):
            ase_atoms = atoms.to_ase_atoms()
        elif hasattr(atoms, 'get_positions'):
            # 检查是否是 ASE Atoms
            pass
        else:
            raise TypeError(f"Unsupported atom structure type: {type(atoms)}")

        # 处理原子标签（tags）：约束设置为0，表面设置为1
        tags = np.ones(len(ase_atoms), dtype=np.int64)

        # 检查是否有约束属性
        if hasattr(ase_atoms, 'constraints') and ase_atoms.constraints:
            for constraint in ase_atoms.constraints:
                if hasattr(constraint, 'index'):
                    tags[constraint.index] = 0

        ase_atoms.set_tags(tags)

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

if __name__ == "__main__":
    from ase.build import fcc111
    try:
        atoms = fcc111('Cu', size=(4,4,3), vacuum=10.0)

        # Set periodic boundary conditions
        atoms.set_pbc([True, True, True])

        ckpt = "checkpoints/eq2_83M_2M.pt"
        if not Path(ckpt).exists():
             ckpt = "../../checkpoints/eq2_83M_2M.pt"

        if Path(ckpt).exists():
            oracle = EquiformerV2Oracle(ckpt, device="cuda")
            e = oracle.predict_energy(atoms)
            print(f"✓ Test Prediction: {e:.4f} eV")
            print("✓ EquiformerV2 Oracle is working correctly!")
        else:
            print(f"Checkpoint not found at {ckpt}")
    except Exception as e:
        import traceback
        print(f"✗ Test failed: {e}")
        traceback.print_exc()