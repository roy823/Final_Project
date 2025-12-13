from typing import Callable, Optional

import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from chem_gym.config import EnvConfig, TrainConfig
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.surrogate.ensemble import SurrogateEnsemble
from chem_gym.analysis.vis_callback import VisualizationCallback


# === [新增] 自定义小网格 CNN ===
class CustomSmallCNN(BaseFeaturesExtractor):
    """
    专门适配 4x4 或 10x10 小网格的 CNN 特征提取器。
    解决 SB3 默认 NatureCNN 在小尺寸输入下报错的问题。
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # SB3 会自动将 channel-last (H,W,C) 转为 channel-first (C,H,W)
        n_input_channels = observation_space.shape[0] 
        
        self.cnn = nn.Sequential(
            # 第一层卷积: 保持空间尺寸 (Padding=1, Kernel=3)
            # 输入: (C, H, W) -> 输出: (32, H, W)
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # 第二层卷积
            # 输入: (32, H, W) -> 输出: (64, H, W)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            # 展平
            nn.Flatten(),
        )

        # 动态计算 Flatten 后的维度以连接全连接层
        with th.no_grad():
            # 创建一个伪造的样本输入 (Batch=1)
            sample = th.as_tensor(observation_space.sample()[None]).float()
            # 模拟 SB3 的预处理 (Permute dimensions)
            if sample.shape[-1] == n_input_channels: # 如果是 (N, H, W, C)
                sample = sample.permute(0, 3, 1, 2)
            
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


# === 原有的 Wrappers 保持不变 ===

class UncertaintyPenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, coefficient: float):
        super().__init__(env)
        self.coefficient = coefficient

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 只有当 info 中确实包含 uncertainty 且值非零时才扣分
        if self.coefficient and "uncertainty" in info:
            reward -= self.coefficient * float(info.get("uncertainty", 0.0))
        return obs, reward, terminated, truncated, info


class OracleWrapper(gym.Wrapper):
    """
    Emulates an oracle call: if uncertainty > threshold, request oracle_energy_fn
    and push it into the surrogate cache so future calls are lower-uncertainty.
    """

    def __init__(self, env: ChemGymEnv, surrogate: SurrogateEnsemble, threshold: float,
                 oracle_energy_fn: Callable[[Optional[object]], float]):
        super().__init__(env)
        self.surrogate = surrogate
        self.threshold = threshold
        self.oracle_energy_fn = oracle_energy_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 只有在 surrogate 存在时才有意义进行主动学习
        current_uncertainty = info.get("uncertainty", 0.0)
        
        if current_uncertainty > self.threshold:
            oracle_energy = self.oracle_energy_fn(info.get("atoms"))
            # 更新 surrogate 缓存
            self.surrogate.update_with_oracle(info.get("atoms"), oracle_energy)
            info["oracle_energy"] = oracle_energy
            
        return obs, reward, terminated, truncated, info


def make_vec_env(env_config: EnvConfig, surrogate: Optional[SurrogateEnsemble], train_config: TrainConfig,
                 oracle_energy_fn: Optional[Callable] = None):
    def _make_single():
        # 这里 surrogate 可能是 None，ChemGymEnv 会自动处理回退到 EMT
        env = ChemGymEnv(env_config, surrogate=surrogate)
        
        # 只有当 surrogate 存在时，才需要考虑不确定性惩罚和 Oracle 调用
        if surrogate is not None:
            if train_config.uncertainty_penalty > 0:
                env = UncertaintyPenaltyWrapper(env, coefficient=train_config.uncertainty_penalty)
            
            if train_config.oracle_threshold is not None and oracle_energy_fn:
                env = OracleWrapper(env, surrogate, train_config.oracle_threshold, oracle_energy_fn)
                
        return env

    return DummyVecEnv([_make_single for _ in range(train_config.n_envs)])


def train_agent(env_config: EnvConfig, surrogate: Optional[SurrogateEnsemble], train_config: TrainConfig,
                oracle_energy_fn: Optional[Callable] = None):
    vec_env = make_vec_env(env_config, surrogate, train_config, oracle_energy_fn)
    
    # === [修改] 策略配置 ===
    policy_kwargs = {}
    
    if env_config.mode == "image":
        policy = "CnnPolicy"
        # 注入我们自定义的 CNN 提取器
        policy_kwargs = {
            "features_extractor_class": CustomSmallCNN,
            "features_extractor_kwargs": {"features_dim": 128},
        }
        print(f"[Trainer] Using CnnPolicy with CustomSmallCNN for 4x4 Grid.")
    else:
        # Graph 模式保持不变
        policy = "MultiInputPolicy" 
        print(f"[Trainer] Using MultiInputPolicy for Graph mode.")
    
    model = PPO(
        policy,
        vec_env,
        policy_kwargs=policy_kwargs, # <--- 关键：传入自定义网络配置
        verbose=1,
        learning_rate=train_config.learning_rate,
        gamma=train_config.gamma,
        gae_lambda=train_config.lam,
        device=train_config.device,
        # 开启 Tensorboard 日志，用于观察物理能量曲线
        n_steps=128,
        batch_size=64,
        tensorboard_log="./chem_gym_tensorboard/"
    )
    
    vis_callback = VisualizationCallback(save_freq=50, save_dir="./vis_results")
    model.learn(total_timesteps=train_config.total_timesteps, progress_bar=True, callback=vis_callback)
    return model