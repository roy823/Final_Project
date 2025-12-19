from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback  # 新增导入
from stable_baselines3.common.callbacks import CallbackList

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

class EnergyLoggerCallback(BaseCallback):
    """
    自定义回调函数：每隔指定步数打印一次当前能量
    """
    def __init__(self, verbose=0, log_freq=10):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # n_calls 是总步数。我们只在特定频率打印
        if self.n_calls % self.log_freq == 0:
            # 从 VecEnv 的 info 中获取能量
            # 因为是向量化环境，infos 是一个列表
            infos = self.locals.get("infos", [])
            if infos:
                energy = infos[0].get("energy", "N/A")
                reward = self.locals.get("rewards")[0]
                print(f"[Step {self.n_calls}] Energy: {energy:.4f} eV/atom | Reward: {reward:.4f}")
        return True

def make_vec_env(env_config: EnvConfig, surrogate: Optional[SurrogateEnsemble], train_config: TrainConfig,
                 oracle_energy_fn: Optional[Callable] = None, oracle=None):
    def _make_single():
        # 这里 surrogate 可能是 None，ChemGymEnv 会自动处理回退到 EMT
        env = ChemGymEnv(env_config, surrogate=surrogate, oracle=oracle)
        
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
                oracle_energy_fn: Optional[Callable] = None, oracle=None):
    vec_env = make_vec_env(env_config, surrogate, train_config, oracle_energy_fn, oracle=oracle)

    
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
        batch_size=128,
        clip_range=0.2, # [建议] 限制更新幅度
        ent_coef=0.001, # [建议] 增加探索
        tensorboard_log="./chem_gym_tensorboard/"
    )
    vis_callback = VisualizationCallback(save_freq=50, save_dir="./vis_results")
    energy_callback = EnergyLoggerCallback(log_freq=5) # 每 5 步打印一次
    
    # 使用 CallbackList 包装
    callbacks = CallbackList([vis_callback, energy_callback])

    print(f"Starting training for {train_config.total_timesteps} steps...")
    
    # [关键修改：只调用一次 learn]
    model.learn(
        total_timesteps=train_config.total_timesteps, 
        progress_bar=True, 
        callback=callbacks
    )
    # ------------------------------

    # 保存模型和归一化参数
    model.save("ppo_chem_gym")
    vec_env.save("vec_normalize.pkl")
    
    return model
