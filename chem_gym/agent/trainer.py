from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize # [新增] 引入 VecNormalize

from chem_gym.config import EnvConfig, TrainConfig
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.surrogate.ensemble import SurrogateEnsemble
from chem_gym.analysis.vis_callback import VisualizationCallback
from chem_gym.agent.graph_feature_extractor import CrystalGraphFeatureExtractor


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

    # 1. 创建基础向量化环境
    env = DummyVecEnv([_make_single for _ in range(train_config.n_envs)])
    
    # 2. [关键修改] 使用 VecNormalize 归一化观测值和奖励
    # norm_obs=True: 归一化观测值 (对 GNN 输入特征很有帮助)
    # norm_reward=True: 归一化奖励 (解决 explained_variance 低的核心)
    # clip_obs=10.0, clip_reward=10.0: 防止极端值破坏训练
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    return env


def train_agent(env_config: EnvConfig, surrogate: Optional[SurrogateEnsemble], train_config: TrainConfig,
                oracle_energy_fn: Optional[Callable] = None):
    vec_env = make_vec_env(env_config, surrogate, train_config, oracle_energy_fn)
    
    # 策略网络选择
    policy_kwargs = {}

    if env_config.mode == "image":
        policy = "MlpPolicy" # 对于 4x4 网格，MLP 足以处理且比 CNN 更快
    elif env_config.mode == "graph":
        policy = "MultiInputPolicy" # 用于 graph 模式
        policy_kwargs = dict(
            features_extractor_class=CrystalGraphFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256, hidden_dim=128, n_layers=3),
        )
    else:
        raise ValueError(f"Unsupported mode: {env_config.mode}")
    
    print(f"[Trainer] Initializing PPO with policy: {policy}")
    
    model = PPO(
        policy,
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=train_config.learning_rate,
        gamma=train_config.gamma,
        gae_lambda=train_config.lam,
        device=train_config.device,
        # 开启 Tensorboard 日志，用于观察物理能量曲线
        n_steps=2048, # [建议] 增加采样步数，让梯度更稳
        batch_size=64,
        clip_range=0.1, # [建议] 限制更新幅度
        ent_coef=0.01, # [建议] 增加探索
        tensorboard_log="./chem_gym_tensorboard/"
    )
    vis_callback = VisualizationCallback(save_freq=50, save_dir="./vis_results")
    model.learn(total_timesteps=train_config.total_timesteps, progress_bar=True, callback=vis_callback)
    
    # [新增] 保存 VecNormalize 的统计数据 (均值和方差)
    # 否则加载模型时无法还原真实的奖励尺度
    model.save("ppo_chem_gym")
    vec_env.save("vec_normalize.pkl")
    
    return model