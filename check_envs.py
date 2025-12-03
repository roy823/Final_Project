import sys
import subprocess

def get_package_status(pkg_name):
    try:
        # 尝试导入并获取版本
        if pkg_name == "fairchem":
            import fairchem.core
            return f"✅ Installed (Version: {fairchem.core.__version__})"
        elif pkg_name == "torch_geometric":
            import torch_geometric
            return f"✅ Installed (Version: {torch_geometric.__version__})"
        elif pkg_name == "ase":
            import ase
            return f"✅ Installed (Version: {ase.__version__})"
        elif pkg_name == "stable_baselines3":
            import stable_baselines3
            return f"✅ Installed (Version: {stable_baselines3.__version__})"
        else:
            mod = __import__(pkg_name)
            return f"✅ Installed (Version: {getattr(mod, '__version__', 'Unknown')})"
    except ImportError:
        return "❌ Not Found"
    except Exception as e:
        return f"⚠️ Error: {e}"

print(f"==================================================")
print(f"🔍 当前 Python 解释器路径:")
print(f"   {sys.executable}")
print(f"==================================================")

packages = ["torch", "fairchem", "torch_geometric", "gymnasium", "numpy", "ase", "stable_baselines3", "tensorboard"]

for pkg in packages:
    status = get_package_status(pkg)
    print(f"{pkg:<20} : {status}")

if "torch" in sys.modules:
    import torch
    print(f"--------------------------------------------------")
    print(f"CUDA Available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device          : {torch.cuda.get_device_name(0)}")
print(f"==================================================\n")