# Chem-Gym：用于高熵合金表面的代理辅助主动强化学习

面向课程大作业的完整框架，包含环境 (A)、代理集成 (B)、策略与主动学习 (C)，并以 EquiformerV2 作为高保真“Oracle”。

## 目录结构与职责
- `chem_gym/__init__.py`：包入口，导出 `ChemGymEnv`、`SurrogateEnsemble`。
- `chem_gym/envs/chem_env.py`（模块 A：环境）：
  - Gymnasium 环境；两种观测：
    - `image`：`(H, W, N_elements)` one-hot 网格，适合 CNN/MLP 策略。
    - `graph`：节点特征 + 邻接矩阵占位（可扩展为边索引/距离，供 GNN 策略）。
  - 动作：离散交换两格（`n_sites*(n_sites-1)/2`），映射在 `_action_to_indices`。
  - 奖励：`-(E_t - E_0) - step_penalty`，步数达 `max_steps` 截断。
  - 状态 → ASE `Atoms`（fcc(111)），`render` 调用 `plot_atoms`。
  - `surrogate.evaluate(atoms)` 输出能量与不确定度。
- `chem_gym/surrogate/ensemble.py`（模块 B：代理集成）：
  - 集成 + 缓存：hash(符号+坐标) 避免重复推理。
  - `evaluate` 返回 (均值能量, 方差)；`update_with_oracle` 写入高保真标签（主动学习）。
  - 待替换为 OCP 预训练（GemNet-OC/PaiNN 快速代理 + EquiformerV2 高保真 Oracle）。
- `chem_gym/agent/trainer.py`（模块 C：策略与主动学习）：
  - `TrainConfig`：PPO 超参 + `uncertainty_penalty`、`oracle_threshold`。
  - 包装器：`UncertaintyPenaltyWrapper`（奖励扣 sigma），`OracleWrapper`（sigma>阈值触发 Oracle，写回缓存）。
  - `make_vec_env` 构建矢量化环境；`train_agent` 运行 PPO（image 用 CNN，graph 可换自定义 GNN）。
- `chem_gym/active_learning.py`：轻量回放、Oracle 触发、缓存更新工具。
- `chem_gym/baselines.py`：随机搜索、模拟退火基线。
- `main.py`：CLI 运行 PPO 或基线，Oracle 钩子占位。
- `requirements.txt`：依赖列表（含 EquiformerV2 所需堆栈）。

## 深度学习设计（思路与目的）
- 目标：在固定 HEA 表面上通过最少代理调用获得最低吸附能；利用不确定度驱动主动学习。
- 代理集成（速度+稳健）：
  - 快速模型：3–5 个 GemNet-OC/PaiNN 检查点组成深度集成，输出均值与 std 作为 (E_hat, σ)。
  - 高保真 Oracle：EquiformerV2 作为慢模型，仅在 sigma 超阈值时调用，写回缓存以降低不确定度。
- 数据集选择与清洗：
  - 来源：OC20/OC22。筛选目标元素（如 Cu/Ni/Pt/Pd/Au）和少量吸附物（*H, *CO）。
  - 规模：几千到一两万条子集，便于小规模微调/线下推理；可保留验证集作为 Oracle 查表。
  - 预处理：去除缺失坐标/异常能量；标准化晶格；记录 adsorbate 标签用于 hash。
- 训练/推理策略：
  - 代理微调：可对最后 MLP 头进行轻量微调以适配子集；不同随机种子/检查点提升不确定度质量。
  - 批量推理：在矢量化环境中合批多个 Atoms 送入模型，降低 kernel 启动开销。
  - 缓存：基于符号+位置(+吸附物) 的稳定 hash，重复状态直接返回 (E, σ=0)。
- 不确定度与主动学习：
  - 奖励塑形：`R = -(E_t - E_0) - λ·σ - step_penalty`，鼓励低能且低不确定度。
  - Oracle 触发：`sigma > tau` 时调用 EquiformerV2/查表，将能量写入缓存（sigma→0），形成闭环。
  - 统计：记录 sigma 分布、Oracle 触发频次、缓存命中率，以证明主动学习有效。
- 策略与基线：
  - 主算法：PPO (on-policy) 先跑通 image 模式；graph 模式可换自定义 GNN policy。
  - 基线：随机搜索、模拟退火，作为对比曲线（能量 vs 代理调用数）。

## 数据与模型流
1) 环境状态 -> ASE `Atoms` slab（可控元素比例、随机置换）。
2) 代理集成 -> (能量均值, sigma)；缓存命中则 O(1) 返回。
3) PPO 策略接收观测（image/graph），奖励包含能量与不确定度项。
4) 主动学习：sigma 超阈值触发 EquiformerV2/查表，缓存更新，sigma 下降。

## 快速开始
1) 安装基础依赖（CPU 可用，推荐 GPU）：
```
pip install -U gymnasium stable-baselines3[extra] ase torch numpy tensorboard rich
```
2) 安装 OCP 堆栈（EquiformerV2 & GemNet/PaiNN，需要匹配 CUDA 的 PyTorch）：
```
pip install ocp-torch           # EquiformerV2
pip install ocp-models          # GemNet-OC / PaiNN
# 下载对应 checkpoint，参见 OCP 官方说明
```
3) 运行 PPO（占位代理）：
```
python main.py --mode train --obs-mode image --total-steps 5000 --uncertainty-penalty 0.05
```
4) 运行基线：
```
python main.py --mode baseline --obs-mode image
```

## 如何接入真实 EquiformerV2 / OCP 代理
- 在 `surrogate/ensemble.py` 中实现：
  - `load_ocp_model(cfg, ckpt, device)`：加载 EquiformerV2/GemNet-OC。
  - `ocp_predict(atoms)`：ASE `Atoms` -> OCP 图（节点/边/距离 + adsorbate 标签），前向得到能量。
  - 用多个检查点构成 `self.models`，取均值/方差；支持批量推理。
- 强化 hash：符号 + 四舍五入坐标 + adsorbate id，确保重复状态命中缓存。
- 将 `oracle_energy_fn` 绑定 EquiformerV2（或 OC20/OC22 查表），`OracleWrapper` 自动写回缓存。

## 环境与策略注意点
- image 模式：默认 CNN policy；最快实现路径。
- graph 模式：扩展邻接为边索引/距离，接入自定义 GNN policy（可继承 SB3 ActorCriticPolicy，或换 d3rlpy/CQL/IQL）。
- 奖励：可调 `step_penalty` 与 `uncertainty_penalty`；截断在 `max_steps`。
- 可视化：`render` 生成 PNG/GIF 用于展示。

## 实施优先级
1) 补充 `requirements.txt`：`ocp-torch`, `ocp-models`, `tensorboard`, `rich`。
2) 在 `ensemble.py` 接 EquiformerV2/GemNet-OC 推理 + 稳定 hash + 批量化。
3) 扩展 graph 观测为边索引/距离，接入 GNN policy。
4) 打开 TensorBoard 日志，记录能量/σ/Oracle 触发/缓存命中。
5) 评测脚本：能量 vs 代理调用、sigma 衰减、随机/SA/PPO 对比；渲染优化后的表面。

## 价值与可行性
- EquiformerV2 提供高保真 Oracle，GemNet-OC/PaiNN 作为快速代理，配合缓存显著减少调用次数。
- 主动学习闭环利用 sigma 触发高精度评估，提升策略可靠性；image 模式快速出结果，graph+GNN 可在最终报告展示更高物理保真度。
