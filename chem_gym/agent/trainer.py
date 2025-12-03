from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from chem_gym.config import EnvConfig, TrainConfig
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.surrogate.ensemble import SurrogateEnsemble


class UncertaintyPenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, coefficient: float):
        super().__init__(env)
        self.coefficient = coefficient

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.coefficient and "uncertainty" in info:
            reward -= self.coefficient * float(info["uncertainty"])
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
        if info.get("uncertainty", 0.0) > self.threshold:
            oracle_energy = self.oracle_energy_fn(info.get("atoms"))
            self.surrogate.update_with_oracle(info.get("atoms"), oracle_energy)
            info["oracle_energy"] = oracle_energy
        return obs, reward, terminated, truncated, info


def make_vec_env(env_config: EnvConfig, surrogate: SurrogateEnsemble, train_config: TrainConfig,
                 oracle_energy_fn: Optional[Callable] = None):
    def _make_single():
        env = ChemGymEnv(env_config, surrogate=surrogate)
        if train_config.uncertainty_penalty > 0:
            env = UncertaintyPenaltyWrapper(env, coefficient=train_config.uncertainty_penalty)
        if train_config.oracle_threshold is not None and oracle_energy_fn:
            env = OracleWrapper(env, surrogate, train_config.oracle_threshold, oracle_energy_fn)
        return env

    return DummyVecEnv([_make_single for _ in range(train_config.n_envs)])


def train_agent(env_config: EnvConfig, surrogate: SurrogateEnsemble, train_config: TrainConfig,
                oracle_energy_fn: Optional[Callable] = None):
    vec_env = make_vec_env(env_config, surrogate, train_config, oracle_energy_fn)
    
    # --- 核心修复 ---
    # 原始代码使用了 "CnnPolicy"，这对于 4x4 的小网格是不合适的（且会导致 Crash）。
    # 改为 "MlpPolicy"，它会将 4x4xN 的输入展平为向量处理，非常适合这种小规模状态。
    if env_config.mode == "image":
        policy = "MlpPolicy"
    else:
        policy = "MultiInputPolicy" # 用于 graph 模式 (Dict observation)
    # ----------------
    
    model = PPO(
        policy,
        vec_env,
        verbose=1,
        learning_rate=train_config.learning_rate,
        gamma=train_config.gamma,
        gae_lambda=train_config.lam,
        device=train_config.device,
    )
    model.learn(total_timesteps=train_config.total_timesteps)
    return model