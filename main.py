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
    
    # Device defaults to auto
    parser.add_argument("--n-active-layers", type=int, default=2, help="Number of top layers to optimize")
    parser.add_argument("--device", type=str, default="auto") 
    parser.add_argument("--seed", type=int, default=None)
    
    # Surrogate params
    # 设置为 0 时，将禁用 Surrogate 并使用环境内置的 EMT 物理计算
    parser.add_argument("--surrogate-models", type=int, default=3)
    parser.add_argument("--surrogate-mean", type=float, default=-1.0)
    parser.add_argument("--surrogate-noise", type=float, default=0.1)
    
    # Oracle params
    parser.add_argument("--oracle-ckpt", type=str, default="checkpoints/eq2_83M_2M.pt", help="Path to OCP checkpoint")
    
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def launch_train(args):
    # 1. 初始化 Oracle (Equiformer V2)
    real_oracle = None
    if args.oracle_ckpt and Path(args.oracle_ckpt).exists():
        print(f"[Main] Loading real Oracle from {args.oracle_ckpt}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            eq2_model = EquiformerV2Oracle(args.oracle_ckpt, device=device)
            real_oracle = eq2_model.predict_energy
        except Exception as e:
            print(f"[Main] Failed to load Oracle: {e}. Running without Oracle.")
            real_oracle = None
    else:
        print(f"[Main] Oracle checkpoint not found at {args.oracle_ckpt}. Running without Oracle.")
        real_oracle = None

    env_config = EnvConfig(
        mode=args.obs_mode, 
        init_seed=args.seed,
        n_active_layers=args.n_active_layers
    )
    # 2. 初始化 Surrogate
    surrogate = None
    if args.surrogate_models > 0:
        print(f"[Main] Initializing Surrogate Ensemble with {args.surrogate_models} models...")
        surrogate_cfg = SurrogateConfig(
            n_models=args.surrogate_models,
            mean_energy=args.surrogate_mean,
            noise_scale=args.surrogate_noise,
        )
        surrogate = SurrogateEnsemble(config=surrogate_cfg)
    else:
        print("[Main] Surrogate models set to 0. Using internal Environment Physics (EMT) instead.")
    
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
    # 对于 Baseline，如果不指定 surrogate-models > 0，也应使用 EMT
    if args.surrogate_models > 0:
        surrogate_cfg = SurrogateConfig(n_models=args.surrogate_models)
        surrogate = SurrogateEnsemble(config=surrogate_cfg)
    else:
        surrogate = None
        
    env_config = EnvConfig(
        mode=args.obs_mode, 
        init_seed=args.seed,
        n_active_layers=args.n_active_layers
    )
    
    print("Running random search...")
    rand_result = random_search(env, episodes=10, horizon=50)
    print(f"Random best energy: {rand_result['best_energy']}")

    print("Running simulated annealing...")
    sa_result = simulated_annealing(env, steps=200)
    print(f"SA best energy: {sa_result['best_energy']}")


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "train":
        launch_train(cli_args)
    else:
        launch_baselines(cli_args)