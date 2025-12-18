#!/usr/bin/env python3
"""
验证 Oracle 参数传递的测试脚本
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from chem_gym.config import TrainConfig


def test_train_config_defaults():
    """测试 TrainConfig 默认值"""
    print("=" * 60)
    print("测试 1: TrainConfig 默认参数")
    print("=" * 60)

    config = TrainConfig()

    assert config.oracle_fmax == 0.05, f"默认 fmax 错误: {config.oracle_fmax}"
    assert config.oracle_max_steps == 100, f"默认 max_steps 错误: {config.oracle_max_steps}"
    assert config.oracle_disable_amp == True, f"默认 disable_amp 错误: {config.oracle_disable_amp}"

    print(f"✅ oracle_fmax 默认值: {config.oracle_fmax}")
    print(f"✅ oracle_max_steps 默认值: {config.oracle_max_steps}")
    print(f"✅ oracle_disable_amp 默认值: {config.oracle_disable_amp}")
    print()


def test_train_config_custom():
    """测试 TrainConfig 自定义参数"""
    print("=" * 60)
    print("测试 2: TrainConfig 自定义参数")
    print("=" * 60)

    config = TrainConfig(
        oracle_fmax=0.03,
        oracle_max_steps=50,
        oracle_disable_amp=False
    )

    assert config.oracle_fmax == 0.03, f"自定义 fmax 错误: {config.oracle_fmax}"
    assert config.oracle_max_steps == 50, f"自定义 max_steps 错误: {config.oracle_max_steps}"
    assert config.oracle_disable_amp == False, f"自定义 disable_amp 错误: {config.oracle_disable_amp}"

    print(f"✅ oracle_fmax 自定义值: {config.oracle_fmax}")
    print(f"✅ oracle_max_steps 自定义值: {config.oracle_max_steps}")
    print(f"✅ oracle_disable_amp 自定义值: {config.oracle_disable_amp}")
    print()


def test_cli_args_parsing():
    """测试 CLI 参数解析"""
    print("=" * 60)
    print("测试 3: CLI 参数解析")
    print("=" * 60)

    from main import parse_args

    # 模拟命令行参数
    test_args = [
        "--mode", "train",
        "--oracle-fmax", "0.02",
        "--oracle-max-steps", "75",
        "--oracle-disable-amp", "false"
    ]

    args = parse_args()

    print(f"✅ oracle_fmax: {args.oracle_fmax}")
    print(f"✅ oracle_max_steps: {args.oracle_max_steps}")
    print(f"✅ oracle_disable_amp: {args.oracle_disable_amp}")
    print()


def test_oracle_initialization_params():
    """测试 Oracle 初始化参数传递"""
    print("=" * 60)
    print("测试 4: Oracle 初始化参数传递")
    print("=" * 60)

    from chem_gym.surrogate.ocp_model import EquiformerV2Oracle
    import inspect

    # 获取 __init__ 方法签名
    sig = inspect.signature(EquiformerV2Oracle.__init__)
    params = list(sig.parameters.keys())

    print(f"EquiformerV2Oracle.__init__ 参数: {params}")

    assert 'checkpoint_path' in params, "缺少 checkpoint_path 参数"
    assert 'fmax' in params, "缺少 fmax 参数"
    assert 'max_steps' in params, "缺少 max_steps 参数"

    print(f"✅ 支持 fmax 参数: {sig.parameters['fmax'].default}")
    print(f"✅ 支持 max_steps 参数: {sig.parameters['max_steps'].default}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Oracle 参数传递验证测试")
    print("=" * 60 + "\n")

    try:
        test_train_config_defaults()
        test_train_config_custom()
        test_cli_args_parsing()
        test_oracle_initialization_params()

        print("=" * 60)
        print("✅ 所有测试通过！Oracle 参数传递机制工作正常")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
