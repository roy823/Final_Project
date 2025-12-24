# Chem-Gym: 基于 EquiformerV2 代理模型的高熵合金 (HEA) 表面优化框架

## 📖 项目背景
高熵合金（High-Entropy Alloys, HEA）因其独特的“鸡尾酒效应”在催化领域展现出巨大潜力。然而，HEA 表面原子的排列组合空间极其庞大（对于一个 32 原子的体系，二元合金的排列数约为 $10^8$，五元合金则超过 $10^{18}$）。

**Chem-Gym** 是一个将深度强化学习（DRL）与最先进的原子尺度模型（EquiformerV2）相结合的框架，旨在通过智能搜索而非穷举法，寻找热力学最稳定的 HEA 表面构型。

---

## 🛠️ 技术架构

项目遵循“环境-代理-预言机”三层架构：

1.  **环境 (chem_gym/envs/chem_env.py)**: 
    - 基于 Gymnasium 接口。
    - 负责维护原子坐标、执行原子交换动作、计算生成能。
    - 支持 `graph`（图神经网络）和 `image`（多通道网格）两种观测模式。
2.  **代理 (chem_gym/agent/trainer.py)**:
    - 使用 Stable Baselines3 的 PPO 算法。
    - 集成了自定义的 `CrystalGraphFeatureExtractor`，能够识别晶体结构的局部配位环境。
3.  **预言机 (chem_gym/surrogate/ocp_model.py)**:
    - 封装了 Open Catalyst Project (OCP) 的 **EquiformerV2** 模型。
    - 提供高精度的能量预测和结构弛豫功能。

---

## 🚀 安装与配置

### 1. 核心依赖
```bash
pip install fairchem-core torch-geometric stable-baselines3 gymnasium ase pymatgen plotly
```

### 2. 下载预训练权重
项目默认使用 EquiformerV2 (83M 参数) 模型：
```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_83M_2M.pt -O checkpoints/eq2_83M_2M.pt
```

---

## 🧪 物理原理详解

### 生成能 (Formation Energy) 计算
为了消除体系大小和元素种类的影响，我们计算每原子的生成能：
$$E_{form} = \frac{E_{slab} - \sum_{i} n_i \mu_i}{N}$$
其中：
- $E_{slab}$ 是由 `EquiformerV2Oracle` 计算的总能。
- $\mu_i$ 是元素 $i$ 在纯金属体相（Bulk）中的化学势。
- 环境在启动时会自动调用 `_calibrate_references` 函数，使用相同的 Oracle 模型计算所有元素的 $\mu_i$，确保能量基准的一致性。

### 奖励函数塑形 (Reward Shaping)
为了引导 Agent 在巨大的搜索空间中快速收敛，我们设计了复合奖励：
1.  **能量差奖励**：$1000 \times (E_{t-1} - E_t)$，即能量每下降 0.001 eV，奖励 +1.0。
2.  **记录突破奖励**：当 Agent 找到本轮训练中的最低能量构型时，给予双倍奖励。
3.  **无效动作惩罚**：交换相同元素或能量上升时给予微小惩罚。

---

## 🤖 强化学习优化策略

### 1. 熵减量调度 (Entropy Decay)
在训练初期，我们设置较高的 `ent_coef`（如 0.01）鼓励 Agent 尝试各种排列。随着训练进行，`EntropyDecayCallback` 会线性降低熵系数，迫使 Agent 在后期锁定已发现的最优物理规律。

### 2. 观测值归一化 (VecNormalize)
由于 GNN 提取的特征值范围波动较大，我们使用了 `VecNormalize` 包装器。它能实时计算观测值的均值和方差并进行归一化，这对于稳定 PPO 的 `explained_variance` 指标至关重要。

---

## 💻 使用指南

### 进入工作区
```bash
cd /root/shared-nvme/ChemGymProject
```

### 训练模型
针对 3 层活性层（48 个原子）的复杂体系：
```bash
/base/mambaforge/bin/python main.py     --mode train     --oracle-ckpt checkpoints/uma-s-1p1.pt     --obs-mode graph     --total-steps 40000     --n-active-layers 3     --learning-rate 1e-4     --device cuda 

/base/mambaforge/bin/python main.py     --mode train     --oracle-ckpt checkpoints/uma-s-1p1.pt     --obs-mode graph     --total-steps 40000     --n-active-layers 3     --learning-rate 1e-4     --device cuda   --use-masking
```
### 查看TensorBoard
```bash
tensorboard --logdir ./chem_gym_tensorboard/
```
### 评估与推理 (Greedy Quench)
加载训练好的模型，进行 200 步的随机采样优化，并实时保存最优构型：
```bash
python main.py --mode eval --n-active-layers 3
```

### 交互式可视化
使用 `advanced_vis.py` 生成可缩放、可旋转的 3D 结构图：
```bash
python3 -c "from ase.io import read; from chem_gym.analysis.advanced_vis import plot_structure_plotly; atoms = read('best_optimized.xyz'); plot_structure_plotly(atoms, 'Optimized HEA Structure', 'best_structure.html')"
```

---

## 📈 监控与分析

建议使用 TensorBoard 监控以下关键指标：
- **`rollout/ep_rew_mean`**: 奖励是否稳步上升。
- **`train/explained_variance`**: 核心指标。若该值 > 0.5，说明 Agent 已经理解了原子排列与能量之间的物理联系。
- **`train/entropy_loss`**: 观察熵是否按照预期下降。

---

## 🗺️ 路线图 (Roadmap)
- [x] 集成 EquiformerV2 作为 Oracle。
- [x] 实现基于生成能的奖励机制。
- [x] 引入熵减量回调函数。
- [ ] **多组分扩展**：支持 5 元及以上的高熵合金体系。
- [ ] **吸附能优化**：引入 CO/H 等吸附质，直接优化催化活性位点。
- [ ] **主动学习**：当模型不确定度高时，自动触发高精度 DFT 计算。

--- 
**日期**: 2025年12月20日