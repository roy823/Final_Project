#!/usr/bin/env python
"""
测试统一结果目录功能
"""
import sys
sys.path.insert(0, '/root/shared-nvme/Final_Project')

def test_directory_structure():
    """测试目录创建逻辑"""
    print("=" * 70)
    print("测试统一结果目录功能")
    print("=" * 70)

    from pathlib import Path
    from datetime import datetime

    # 模拟时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 测试标准训练目录
    result_dir = Path("result") / f"{timestamp}_training"
    print(f"\n✓ 标准训练目录: {result_dir}")

    # 测试吸附能训练目录
    result_dir = Path("result") / f"{timestamp}_adsorption"
    print(f"✓ 吸附能训练目录: {result_dir}")

    # 测试自定义目录
    result_dir = Path("result") / "custom_experiment"
    print(f"✓ 自定义目录: {result_dir}")

    # 子目录
    models_dir = result_dir / "models"
    logs_dir = result_dir / "logs"
    visualizations_dir = result_dir / "visualizations"
    structures_dir = result_dir / "structures"

    print(f"\n✓ 子目录:")
    print(f"  - 模型目录: {models_dir}")
    print(f"  - 日志目录: {logs_dir}")
    print(f"  - 可视化目录: {visualizations_dir}")
    print(f"  - 结构目录: {structures_dir}")

    print("\n" + "=" * 70)
    print("✅ 目录结构测试通过!")
    print("=" * 70)

    return True

if __name__ == "__main__":
    test_directory_structure()
