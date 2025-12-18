#!/usr/bin/env python3
"""
演示 Oracle 参数配置的使用方法
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chem_gym.config import TrainConfig, EnvConfig
from chem_gym.surrogate.ensemble import SurrogateEnsemble
from chem_gym.surrogate.ocp_model import EquiformerV2Oracle


def demo_default_config():
    """演示默认配置"""
    print("=" * 70)
    print("演示 1: 使用默认 Oracle 参数")
    print("=" * 70)
    print()
    print("命令行:")
    print("  python main.py --mode train --oracle-ckpt checkpoints/eq2_83M_2M.pt")
    print()
    print("配置效果:")
    print("  - oracle_fmax: 0.05 eV/A (默认收敛阈值)")
    print("  - oracle_max_steps: 100 (默认最大步数)")
    print("  - oracle_disable_amp: True (默认启用稳定性优化)")
    print()


def demo_custom_config():
    """演示自定义配置"""
    print("=" * 70)
    print("演示 2: 使用自定义 Oracle 参数")
    print("=" * 70)
    print()
    print("命令行:")
    print("  python main.py --mode train \\")
    print("    --oracle-ckpt checkpoints/eq2_83M_2M.pt \\")
    print("    --oracle-fmax 0.03 \\")
    print("    --oracle-max-steps 50")
    print()
    print("配置效果:")
    print("  - oracle_fmax: 0.03 eV/A (更严格的收敛阈值，更精确)")
    print("  - oracle_max_steps: 50 (减少弛豫步数，更快)")
    print("  - oracle_disable_amp: True (保持稳定性)")
    print()


def demo_aggressive_config():
    """演示激进配置"""
    print("=" * 70)
    print("演示 3: 激进优化配置 (更快但精度稍低)")
    print("=" * 70)
    print()
    print("命令行:")
    print("  python main.py --mode train \\")
    print("    --oracle-ckpt checkpoints/eq2_83M_2M.pt \\")
    print("    --oracle-fmax 0.10 \\")
    print("    --oracle-max-steps 30")
    print()
    print("配置效果:")
    print("  - oracle_fmax: 0.10 eV/A (较宽松的收敛阈值)")
    print("  - oracle_max_steps: 30 (快速弛豫，适合探索阶段)")
    print("  - 适用于: 初步探索，筛选候选结构")
    print()


def demo_precise_config():
    """演示精确配置"""
    print("=" * 70)
    print("演示 4: 精确优化配置 (更慢但精度更高)")
    print("=" * 70)
    print()
    print("命令行:")
    print("  python main.py --mode train \\")
    print("    --oracle-ckpt checkpoints/eq2_83M_2M.pt \\")
    print("    --oracle-fmax 0.01 \\")
    print("    --oracle-max-steps 200")
    print()
    print("配置效果:")
    print("  - oracle_fmax: 0.01 eV/A (非常严格的收敛阈值)")
    print("  - oracle_max_steps: 200 (充分弛豫)")
    print("  - 适用于: 精细优化，验证最终结果")
    print()


def demo_with_active_learning():
    """演示主动学习场景"""
    print("=" * 70)
    print("演示 5: 主动学习场景 (完整配置)")
    print("=" * 70)
    print()
    print("命令行:")
    print("  python main.py --mode train \\")
    print("    --obs-mode image \\")
    print("    --total-steps 10000 \\")
    print("    --oracle-ckpt checkpoints/eq2_83M_2M.pt \\")
    print("    --oracle-fmax 0.05 \\")
    print("    --oracle-max-steps 100 \\")
    print("    --oracle-threshold 0.3 \\")
    print("    --uncertainty-penalty 0.05")
    print()
    print("配置说明:")
    print("  - 当 surrogate 不确定性 > 0.3 eV 时触发 Oracle")
    print("  - Oracle 使用中等精度配置平衡速度和精度")
    print("  - 奖励函数包含不确定性惩罚项")
    print()


def demo_comparison():
    """演示不同配置的性能对比"""
    print("=" * 70)
    print("演示 6: 性能与精度权衡")
    print("=" * 70)
    print()

    configs = [
        ("快速探索", {"fmax": 0.10, "max_steps": 30}, "快", "低"),
        ("平衡推荐", {"fmax": 0.05, "max_steps": 100}, "中", "中"),
        ("精确验证", {"fmax": 0.01, "max_steps": 200}, "慢", "高"),
    ]

    print(f"{'配置名称':<12} {'fmax':<8} {'max_steps':<10} {'速度':<6} {'精度':<6}")
    print("-" * 70)
    for name, params, speed, acc in configs:
        print(f"{name:<12} {params['fmax']:<8.2f} {params['max_steps']:<10} {speed:<6} {acc:<6}")
    print()


if __name__ == "__main__":
    print("\n" + "🔬" * 35)
    print("Oracle 参数配置演示")
    print("🔬" * 35 + "\n")

    demo_default_config()
    demo_custom_config()
    demo_aggressive_config()
    demo_precise_config()
    demo_with_active_learning()
    demo_comparison()

    print("=" * 70)
    print("💡 总结:")
    print("=" * 70)
    print()
    print("1. 默认配置 (fmax=0.05, max_steps=100) 适合大多数场景")
    print("2. 探索阶段: 使用较宽松参数 (fmax=0.10, max_steps=30)")
    print("3. 验证阶段: 使用严格参数 (fmax=0.01, max_steps=200)")
    print("4. 主动学习: 平衡参数 + 不确定性阈值")
    print("5. 所有参数都可以通过 CLI 灵活配置")
    print()
    print("=" * 70)
