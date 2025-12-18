import fairchem.core
import pkgutil
import importlib
import sys

print(f"==================================================")
print(f"🔍 Starting Deep Search in fairchem.core (v{fairchem.core.__version__})")
print(f"   Root Path: {fairchem.core.__path__[0]}")
print(f"==================================================\n")

TARGET_CLASS = "AtomsToGraphs"
found_paths = []

def safe_import_and_check(module_name):
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, TARGET_CLASS):
            return True
    except Exception:
        # Ignore import errors from internal dependencies
        return False
    return False

# 1. 扫描一级子模块
print("Scanning top-level modules...")
for importer, modname, ispkg in pkgutil.iter_modules(fairchem.core.__path__):
    full_mod_name = f"fairchem.core.{modname}"
    if safe_import_and_check(full_mod_name):
        print(f"🎉 FOUND in top-level: {full_mod_name}")
        found_paths.append(full_mod_name)

# 2. 如果没找到，尝试递归扫描常见嫌疑人 (datasets, common)
if not found_paths:
    print("\nScanning sub-packages (common, datasets)...")
    sub_packages = ["common", "datasets"]
    
    for sub in sub_packages:
        sub_path = f"fairchem.core.{sub}"
        try:
            sub_module = importlib.import_module(sub_path)
            if hasattr(sub_module, "__path__"):
                for importer, modname, ispkg in pkgutil.iter_modules(sub_module.__path__):
                    full_mod_name = f"{sub_path}.{modname}"
                    if safe_import_and_check(full_mod_name):
                        print(f"🎉 FOUND in sub-package: {full_mod_name}")
                        found_paths.append(full_mod_name)
        except ImportError:
            pass

print(f"\n==================================================")
if found_paths:
    print(f"✅ Conclusion: Please verify imports from:")
    for p in found_paths:
        print(f"   from {p} import {TARGET_CLASS}")
else:
    print(f"❌ Failed: '{TARGET_CLASS}' not found anywhere.")
    print("   Possibility 1: The class has been renamed in v2.12.0.")
    print("   Possibility 2: Installation is corrupted (missing files).")
print(f"==================================================")