import os
from typing import Callable, Optional
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
# [新增] 引入 ActionMasker
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from chem_gym.config import EnvConfig, TrainConfig
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.surrogate.ensemble import SurrogateEnsemble
from chem_gym.analysis.vis_callback import VisualizationCallback
from chem_gym.agent.graph_feature_extractor import CrystalGraphFeatureExtractor
import datetime # [新增] 导入时间模块


class UncertaintyPenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, coefficient: float):
        super().__init__(env)
        self.coefficient = coefficient

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.coefficient and "uncertainty" in info:
            reward -= self.coefficient * float(info.get("uncertainty", 0.0))
        return obs, reward, terminated, truncated, info


class OracleWrapper(gym.Wrapper):
    def __init__(self, env: ChemGymEnv, surrogate: SurrogateEnsemble, threshold: float,
                 oracle_energy_fn: Callable[[Optional[object]], float]):
        super().__init__(env)
        self.surrogate = surrogate
        self.threshold = threshold
        self.oracle_energy_fn = oracle_energy_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_uncertainty = info.get("uncertainty", 0.0)
        
        if current_uncertainty > self.threshold:
            oracle_energy = self.oracle_energy_fn(info.get("atoms"))
            self.surrogate.update_with_oracle(info.get("atoms"), oracle_energy)
            info["oracle_energy"] = oracle_energy
            
        return obs, reward, terminated, truncated, info

class EnergyLoggerCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=10):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get("infos", [])
            if infos:
                energy = infos[0].get("energy", "N/A")
                reward = self.locals.get("rewards")[0]
                print(f"[Step {self.n_calls}] Energy: {energy:.4f} eV/atom | Reward: {reward:.4f}")
        return True

class SelectiveVecNormalize(VecNormalize):
    """自定义归一化器，跳过邻接矩阵和掩码"""
    def _normalize_obs(self, obs, var_type):
        if isinstance(obs, dict):
            for key in obs.keys():
                # 只归一化节点特征
                if key == "node_features":
                    obs[key] = super()._normalize_obs(obs[key], var_type)
            return obs
        return super()._normalize_obs(obs, var_type)

def make_vec_env(env_config: EnvConfig, surrogate: Optional[SurrogateEnsemble], train_config: TrainConfig,
                 oracle_energy_fn: Optional[Callable] = None, oracle=None, 
                 use_masking: bool = False): # [修改] 接收 use_masking 参数
    
    def _make_single():
        env = ChemGymEnv(env_config, surrogate=surrogate, oracle=oracle)
        
        if surrogate is not None:
            if train_config.uncertainty_penalty > 0:
                env = UncertaintyPenaltyWrapper(env, coefficient=train_config.uncertainty_penalty)
            
            if train_config.oracle_threshold is not None and oracle_energy_fn:
                env = OracleWrapper(env, surrogate, train_config.oracle_threshold, oracle_energy_fn)
        
        # [关键修改] 如果开启掩码，必须包裹 ActionMasker
        # 这里的 lambda env: env.action_masks() 是告诉 Wrapper 怎么获取掩码
        if use_masking:
            env = ActionMasker(env, lambda env: env.action_masks())
            
        return env

    env = DummyVecEnv([_make_single for _ in range(train_config.n_envs)])
    
    # VecNormalize 放在最后
    env = SelectiveVecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    return env


def train_agent(env_config: EnvConfig, surrogate: Optional[SurrogateEnsemble], train_config: TrainConfig,
                oracle_energy_fn: Optional[Callable] = None, oracle=None, save_dir: str = ".",
                use_masking: bool = False): 
    
    # [修改] 将 use_masking 传递给 make_vec_env
    vec_env = make_vec_env(env_config, surrogate, train_config, oracle_energy_fn, oracle=oracle, use_masking=use_masking)

    policy_kwargs = {}
    if env_config.mode == "image":
        policy = "MlpPolicy"
    elif env_config.mode == "graph":
        policy = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=CrystalGraphFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256, hidden_dim=128, n_layers=3),
        )
    else:
        raise ValueError(f"Unsupported mode: {env_config.mode}")
    
    if use_masking:
        print(f"[Trainer] Initializing MaskablePPO (With Action Masking)")
        model_class = MaskablePPO
        tb_log_name = "MaskablePPO_Experiment"
    else:
        print(f"[Trainer] Initializing Standard PPO (No Masking)")
        model_class = PPO
        tb_log_name = "StandardPPO_Baseline"
    
    model = model_class(
        policy,
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=train_config.learning_rate,
        gamma=train_config.gamma,
        gae_lambda=train_config.lam,
        device=train_config.device,
        n_steps=2048,
        batch_size=128,
        clip_range=0.2,
        ent_coef=0.001,
        tensorboard_log="./chem_gym_tensorboard/"
    )
    
    # [修改] 生成唯一的时间戳和运行 ID
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_type = "maskable" if use_masking else "standard"
    steps_k = train_config.total_timesteps // 1000
    run_id = f"{run_type}_{steps_k}k_{timestamp}"
    
    # 创建独立的运行文件夹
    run_save_dir = os.path.join(save_dir, run_id)
    os.makedirs(run_save_dir, exist_ok=True)

    # [优化] 可视化结果也存入该运行文件夹下的 vis 子目录
    run_vis_dir = os.path.join(run_save_dir, "vis")
    vis_callback = VisualizationCallback(save_freq=200, save_dir=run_vis_dir) # 频率调快一点方便观察
    energy_callback = EnergyLoggerCallback(log_freq=5)
    callbacks = CallbackList([vis_callback, energy_callback])

    print(f"Starting training: {run_id}")
    
    model.learn(
        total_timesteps=train_config.total_timesteps, 
        callback=callbacks,
        tb_log_name=tb_log_name
    )

    # 保存到独立文件夹
    model_path = os.path.join(run_save_dir, "model")
    stats_path = os.path.join(run_save_dir, "vec_normalize.pkl")
    
    print(f"[Trainer] Saving weights to {run_save_dir}...")
    model.save(model_path)
    vec_env.save(stats_path)
    
    # [可选] 同时在根目录保存一个 "latest" 副本，方便 eval 脚本默认加载
    model.save(os.path.join(save_dir, "latest_model"))
    vec_env.save(os.path.join(save_dir, "latest_vec_normalize.pkl"))
    
    return model