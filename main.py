import argparse
from pathlib import Path
import torch

from chem_gym.agent.trainer import train_agent
from chem_gym.baselines import random_search, simulated_annealing
from chem_gym.config import EnvConfig, SurrogateConfig, TrainConfig
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.surrogate.ensemble import SurrogateEnsemble
from chem_gym.surrogate.ocp_model import EquiformerV2Oracle

def parse_args():
    parser = argparse.ArgumentParser(description="Chem-Gym scaffold launcher.")
    parser.add_argument("--mode", choices=["train", "baseline"], default="train")
    parser.add_argument("--obs-mode", choices=["image", "graph"], default="image")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0)
    parser.add_argument("--oracle-threshold", type=float, default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    
    # [更新] Device 默认为 auto
    parser.add_argument("--device", type=str, default="auto") 
    parser.add_argument("--seed", type=int, default=None)
    
    # Surrogate params
    parser.add_argument("--surrogate-models", type=int, default=3)
    parser.add_argument("--surrogate-mean", type=float, default=-1.0)
    parser.add_argument("--surrogate-noise", type=float, default=0.1)
    
    # [新增] Oracle params
    parser.add_argument("--oracle-ckpt", type=str, default="checkpoints/eq2_83M_2M.pt", help="Path to OCP checkpoint")
    
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def launch_train(args):
    # 1. 初始化 Oracle (Equiformer V2)
    real_oracle = None
    if args.oracle_ckpt and Path(args.oracle_ckpt).exists():
        print(f"[Main] Loading real Oracle from {args.oracle_ckpt}...")
        # 自动检测 GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            eq2_model = EquiformerV2Oracle(args.oracle_ckpt, device=device)
            real_oracle = eq2_model.predict_energy
        except Exception as e:
            print(f"[Main] Failed to load Oracle: {e}. Falling back to dummy.")
            real_oracle = dummy_oracle
    else:
        print(f"[Main] Checkpoint {args.oracle_ckpt} not found. Using Dummy Oracle.")
        real_oracle = dummy_oracle

    env_config = EnvConfig(mode=args.obs_mode, init_seed=args.seed)
    
    # 2. 初始化 Surrogate
    surrogate_cfg = SurrogateConfig(
        n_models=args.surrogate_models,
        mean_energy=args.surrogate_mean,
        noise_scale=args.surrogate_noise,
    )
    surrogate = SurrogateEnsemble(config=surrogate_cfg)
    
    # 3. 训练配置
    train_config = TrainConfig(
        total_timesteps=args.total_steps,
        uncertainty_penalty=args.uncertainty_penalty,
        oracle_threshold=args.oracle_threshold,
        n_envs=args.n_envs,
        device=args.device,
    )
    
    # 4. 开始训练
    print(f"[Main] Starting training on {train_config.device}...")
    model = train_agent(env_config, surrogate, train_config, oracle_energy_fn=real_oracle)
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.save_dir / "ppo_chem_gym.zip")
    print(f"Model saved to {args.save_dir}")


def launch_baselines(args):
    env_config = EnvConfig(mode=args.obs_mode, init_seed=args.seed)
    surrogate_cfg = SurrogateConfig(
        n_models=args.surrogate_models,
        mean_energy=args.surrogate_mean,
        noise_scale=args.surrogate_noise,
    )
    env = ChemGymEnv(env_config, surrogate=SurrogateEnsemble(config=surrogate_cfg))
    
    print("Running random search...")
    rand_result = random_search(env, episodes=10, horizon=50)
    print(f"Random best energy: {rand_result['best_energy']}")

    print("Running simulated annealing...")
    sa_result = simulated_annealing(env, steps=200)
    print(f"SA best energy: {sa_result['best_energy']}")


def dummy_oracle(atoms):
    return -1.5

if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "train":
        launch_train(cli_args)
    else:
        launch_baselines(cli_args)