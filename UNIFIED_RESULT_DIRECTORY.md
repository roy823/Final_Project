# 统一结果目录功能说明

## 📝 修改概述

本次修改将Chem-Gym框架的训练输出文件统一保存到 `/result/{task_id}/` 目录中，其中 `task_id` 为时间戳（格式：`YYYYMMDD_HHMMSS`）。这样每次训练的结果都会被有序地组织在独立的目录中，便于管理和查找。

## 🗂️ 目录结构

```
result/
├── {timestamp}_{mode}/
│   ├── models/                  # 模型文件
│   │   ├── ppo_chem_gym.zip     # PPO模型权重
│   │   └── vec_normalize.pkl    # 归一化参数
│   ├── logs/                    # 日志文件
│   │   └── tensorboard/         # TensorBoard日志
│   │       └── PPO_X/
│   │           └── events.out.tfevents.*
│   ├── visualizations/          # 可视化文件
│   │   ├── step_XXXXXX.cif      # CIF结构文件
│   │   ├── step_XXXXXX.png      # 静态图像
│   │   └── step_XXXXXX.html     # 交互式3D图像
│   └── structures/              # 结构文件
│       └── best_optimized.xyz   # 最佳结构（实时更新）
```

## 🔧 修改的文件

### 1. `main.py`
- ✅ 添加 `datetime` 导入
- ✅ 添加 `--result-dir` 参数（可选）
- ✅ 在 `launch_train()` 中创建时间戳目录和子目录
- ✅ 将目录路径传递给 `train_agent()`
- ✅ 修改模型和参数保存路径
- ✅ 更新评估模式以支持新的目录结构

### 2. `chem_gym/agent/trainer.py`
- ✅ 修改 `train_agent()` 函数签名，添加路径参数
- ✅ 更新 TensorBoard 日志路径
- ✅ 更新可视化回调保存目录
- ✅ 修改模型和归一化参数保存路径

### 3. `chem_gym/analysis/vis_callback.py`
- ✅ 修改 `VisualizationCallback.__init__()`，添加 `best_structure_path` 参数
- ✅ 添加最佳结构实时跟踪功能
- ✅ 在 `_on_step()` 中实现最佳结构保存逻辑

## 🚀 使用方法

### 标准训练模式

```bash
# 标准表面优化训练（自动创建时间戳目录）
python main.py --mode train --total-steps 5000

# 评估模式（自动创建时间戳目录）
python main.py --mode eval --n-active-layers 3
```

### 吸附能训练模式 ✅ **新增功能**

```bash
# 基础吸附能训练（默认CO，目标-0.5eV）
python main.py --mode adsorption_train --total-steps 5000

# 自定义吸附剂和目标能量
python main.py --mode adsorption_train \
  --adsorbate O2 \
  --target-ads-energy -0.8 \
  --total-steps 10000

# 自定义吸附高度和容差
python main.py --mode adsorption_train \
  --adsorbate CO \
  --target-ads-energy -0.5 \
  --adsorbate-height 2.5 \
  --energy-tolerance 0.15 \
  --total-steps 10000
```

### 自定义目录

```bash
# 标准训练使用自定义目录
python main.py --mode train --result-dir result/my_experiment_1 --total-steps 5000

# 吸附能训练使用自定义目录
python main.py --mode adsorption_train --result-dir result/co_adsorption_exp1 --total-steps 5000

# 评估时使用相同的目录
python main.py --mode eval --result-dir result/my_experiment_1
```

### 吸附能训练示例

```bash
# 训练CO在Pt-Ag表面吸附（目标-0.5eV）
python main.py --mode adsorption_train \
  --adsorbate CO \
  --target-ads-energy -0.5 \
  --energy-tolerance 0.1 \
  --total-steps 10000 \
  --device cuda

# 训练O2在Pt-Ag表面吸附（目标-0.8eV）
python main.py --mode adsorption_train \
  --adsorbate O2 \
  --target-ads-energy -0.8 \
  --energy-tolerance 0.15 \
  --total-steps 15000 \
  --device cuda
```

## 📊 文件类型说明

| 文件类型 | 保存位置 | 大小 | 说明 |
|---------|---------|------|------|
| **模型文件** | `models/ppo_chem_gym.zip` | ~315KB | 训练后的PPO模型权重 |
| **归一化参数** | `models/vec_normalize.pkl` | ~136KB | 环境观察值归一化参数 |
| **TensorBoard日志** | `logs/tensorboard/PPO_X/` | ~几MB | 训练指标日志，用于可视化分析 |
| **CIF结构文件** | `visualizations/step_XXXXXX.cif` | ~6KB | 晶体信息文件，可用于结构分析 |
| **静态图像** | `visualizations/step_XXXXXX.png` | ~175KB | 俯视图和侧视图 |
| **交互式3D图** | `visualizations/step_XXXXXX.html` | ~4.7MB | Plotly生成的3D可视化 |
| **最佳结构** | `structures/best_optimized.xyz` | ~4KB | 训练过程中发现的最佳结构 |

## ✨ 主要优势

1. **组织清晰**: 每次训练的结果都在独立目录中，不会相互覆盖
2. **易于管理**: 可以通过目录名快速识别训练任务和时间
3. **完整记录**: 包含训练过程中的所有输出文件
4. **便于比较**: 不同训练实验的结果可以并排比较
5. **向下兼容**: 原有代码仍然可以正常工作

## 🔍 关键特性

### 时间戳生成
- 格式：`YYYYMMDD_HHMMSS`（例如：`20251220_143025`）
- 确保每次运行的目录名唯一
- 便于排序和查找

### 实时最佳结构跟踪
- 在 `VisualizationCallback` 中实现
- 每次保存可视化文件时，同时更新最佳结构
- 训练过程中可以实时查看当前最佳结果

### 自动目录创建
- 程序自动创建所需的子目录
- 无需手动创建目录结构
- 目录创建失败时会抛出清晰的错误信息

## 📝 注意事项

1. **磁盘空间**: HTML可视化文件较大（每个约4.7MB），大量训练步数会占用较多空间
2. **路径分隔符**: 代码中使用 `/` 作为路径分隔符，在Windows系统上可能需要调整
3. **目录清理**: 建议定期清理旧的训练结果以释放磁盘空间
4. **并发训练**: 如果同时运行多个训练任务，每个任务会创建不同的目录，不会相互干扰

## 🧪 测试

运行测试脚本验证功能：
```bash
python test_result_dir.py
```

测试会验证：
- ✅ 目录创建功能
- ✅ 子目录结构
- ✅ 文件保存功能
- ✅ 目录清理功能

## 📦 相关文件

- `main.py` - 主程序入口
- `chem_gym/agent/trainer.py` - 训练逻辑
- `chem_gym/analysis/vis_callback.py` - 可视化回调
- `test_result_dir.py` - 测试脚本

## 🎯 后续扩展建议

1. **配置文件保存**: 在结果目录中保存训练配置参数
2. **训练曲线**: 自动生成训练曲线图像并保存
3. **结果摘要**: 生成训练结果的Markdown摘要报告
4. **压缩归档**: 训练完成后自动压缩结果目录
5. **远程同步**: 支持将结果同步到远程存储或云端

## 🎯 吸附能训练特性

### 吸附能训练与标准训练的区别

| 特性 | 标准训练 | 吸附能训练 |
|------|---------|----------|
| **目标** | 表面能最小化 | 吸附能达到目标值 |
| **环境** | ChemGymEnv | AdsorptionChemGymEnv |
| **Oracle使用** | 可选（用于主动学习） | 推荐（用于准确吸附能） |
| **模型文件** | `ppo_chem_gym.zip` | `ppo_adsorption_chem_gym.zip` |
| **目录后缀** | `_training` | `_adsorption` |
| **特殊参数** | - | `--adsorbate`, `--target-ads-energy` |

### 支持的吸附剂

目前支持的吸附剂类型：
- **CO** - 一氧化碳（最常用）
- **O2** - 氧气
- **H2** - 氢气
- **NO** - 一氧化氮

### 吸附能训练输出

吸附能训练的结果目录结构与标准训练相同，但会包含：
- 更详细的吸附能相关日志
- 实时显示当前吸附能与目标值的偏差
- 特殊的环境包装器用于监控训练进度

### 吸附能训练示例输出

```
[Main] Adsorption Training Result directory: result/20251220_143025_adsorption
[Main] Timestamp: 20251220_143025_adsorption

[Main] Loading Oracle from checkpoints/eq2_83M_2M.pt for adsorption training...
[Main] EquiformerV2 Oracle loaded successfully.

[Main] Starting adsorption training on cuda...
[Main] Adsorbate: CO
[Main] Target energy: -0.50 eV
[Main] Energy tolerance: ±0.10 eV

[Trainer] Initializing PPO with adsorption environment
[Trainer] Adsorbate: CO, Target: -0.50 eV
[Trainer] Using AdsorptionEnergyLoggerCallback

Step 5 | Adsorption Energy: -0.245 eV | Target: -0.50 eV | Deviation: 0.2550 eV | Within Target: False | Reward: -2.55
Step 10 | Adsorption Energy: -0.512 eV | Target: -0.50 eV | Deviation: 0.0120 eV | Within Target: True | Reward: 9.88
```
