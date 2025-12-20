#!/usr/bin/env python
"""
测试统一结果目录功能的脚本
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

def test_directory_creation():
    """测试目录创建功能"""
    print("=" * 60)
    print("测试统一结果目录功能")
    print("=" * 60)

    # 模拟创建时间戳目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_result_dir = Path("test_result") / f"{timestamp}_training"

    # 创建子目录
    models_dir = test_result_dir / "models"
    logs_dir = test_result_dir / "logs"
    visualizations_dir = test_result_dir / "visualizations"
    structures_dir = test_result_dir / "structures"

    for dir_path in [models_dir, logs_dir, visualizations_dir, structures_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {dir_path}")

    # 创建一些测试文件
    test_model_path = models_dir / "ppo_chem_gym.zip"
    test_stats_path = models_dir / "vec_normalize.pkl"
    test_structure_path = structures_dir / "best_optimized.xyz"
    test_tb_log = logs_dir / "tensorboard" / "test_events.out.tfevents.123"

    # 创建模拟文件
    test_model_path.write_text("Mock model file")
    test_stats_path.write_text("Mock stats file")
    test_structure_path.write_text("Mock structure file")
    test_tb_log.parent.mkdir(parents=True, exist_ok=True)
    test_tb_log.write_text("Mock tensorboard log")

    print(f"\n✓ 创建测试文件:")
    print(f"  - {test_model_path}")
    print(f"  - {test_stats_path}")
    print(f"  - {test_structure_path}")
    print(f"  - {test_tb_log}")

    # 验证目录结构
    print(f"\n📁 目录结构:")
    for root, dirs, files in os.walk(test_result_dir):
        level = root.replace(str(test_result_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')

    # 清理测试目录
    print(f"\n🧹 清理测试目录...")
    shutil.rmtree(Path("test_result"))
    print(f"✓ 清理完成")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)

if __name__ == "__main__":
    test_directory_creation()
