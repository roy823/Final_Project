import argparse
from pathlib import Path
import torch

from chem_gym.agent.trainer import train_agent
from chem_gym.baselines import random_search, simulated_annealing
from chem_gym.config import EnvConfig, SurrogateConfig, TrainConfig
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.surrogate.ensemble import SurrogateEnsemble
from chem_gym.surrogate.ocp_model import EquiformerV2Oracle
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO

def parse_args():
    parser = argparse.ArgumentParser(description="Chem-Gym scaffold launcher.")
    parser.add_argument("--mode", choices=["train", "baseline", "eval"], default="train") # 增加 eval
    parser.add_argument("--obs-mode", choices=["image", "graph"], default="image")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0)
    parser.add_argument("--oracle-threshold", type=float, default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for PPO")
    
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
    parser.add_argument("--oracle-fmax", type=float, default=0.05, help="Oracle 弛豫收敛阈值 (eV/A)")
    parser.add_argument("--oracle-max-steps", type=int, default=100, help="Oracle 最大弛豫步数")
    parser.add_argument("--oracle-disable-amp", type=bool, default=True, help="禁用 AMP 提升稳定性")
    # [新增] 增加动作掩码开关
    parser.add_argument("--use-masking", action="store_true", help="使用动作掩码 (MaskablePPO)")
    # [新增] 允许指定加载路径
    parser.add_argument("--load-dir", type=str, default=None, help="Path to a specific run directory to load weights from")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()

def launch_train(args):
    # 1. 初始化 Oracle (支持 UMA 或 EquiformerV2)
    eq2_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.oracle_ckpt and ("uma" in args.oracle_ckpt.lower()):
        print(f"[Main] Initializing UMA Oracle from {args.oracle_ckpt}...")
        from chem_gym.surrogate.ocp_model import UMAOracle
        eq2_model = UMAOracle(checkpoint_path=args.oracle_ckpt, device=device)
    elif args.oracle_ckpt and Path(args.oracle_ckpt).exists():
        print(f"[Main] Loading real Oracle from {args.oracle_ckpt}...")
        try:
            eq2_model = EquiformerV2Oracle(
                args.oracle_ckpt,
                device=device,
                fmax=args.oracle_fmax,
                max_steps=args.oracle_max_steps
            )
            print("[Main] EquiformerV2 Oracle loaded successfully.")
        except Exception as e:
            print(f"[Main] Failed to load Oracle: {e}. Running without Oracle.")
            eq2_model = None
    else:
        print(f"[Main] Oracle checkpoint not found at {args.oracle_ckpt}. Running without Oracle.")
        eq2_model = None # 确保这里也是 eq2_model

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
        if eq2_model is not None:
            print("[Main] Surrogate models set to 0. Using EquiformerV2 Oracle for direct physics calculation.")
        else:
            print("[Main] Surrogate models set to 0. Using internal Environment Physics (EMT) instead.")
    # 3. 训练配置
    train_config = TrainConfig(
        total_timesteps=args.total_steps,
        n_envs=args.n_envs,
        device=args.device,
        uncertainty_penalty=args.uncertainty_penalty,
        oracle_threshold=args.oracle_threshold,
        learning_rate=args.learning_rate,
        oracle_fmax=args.oracle_fmax,
        oracle_max_steps=args.oracle_max_steps,
        oracle_disable_amp=args.oracle_disable_amp,
    )
    
    # 4. 启动训练
    train_agent(
        env_config=env_config,
        train_config=train_config,
        surrogate=surrogate,
        oracle_energy_fn=eq2_model,
        oracle=eq2_model,
        save_dir=args.save_dir,
        use_masking=args.use_masking # [新增] 传递开关参数
    )


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


def launch_eval(args):
    # 1. 初始化 Oracle (支持 UMA 或 EquiformerV2)
    eq2_model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.oracle_ckpt and ("uma" in args.oracle_ckpt.lower()):
        print(f"[Eval] Initializing UMA Oracle from {args.oracle_ckpt}...")
        from chem_gym.surrogate.ocp_model import UMAOracle
        eq2_model = UMAOracle(checkpoint_path=args.oracle_ckpt, device=device)
    elif args.oracle_ckpt and Path(args.oracle_ckpt).exists():
        print(f"[Eval] Loading Oracle from {args.oracle_ckpt}...")
        eq2_model = EquiformerV2Oracle(args.oracle_ckpt, device=device)
    
    # 2. 环境配置
    env_config = EnvConfig(mode=args.obs_mode, n_active_layers=args.n_active_layers)
    
    from chem_gym.envs.chem_env import ChemGymEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import MaskablePPO
    
    def _make_single():
        return ChemGymEnv(env_config, oracle=eq2_model)
    
    base_venv = DummyVecEnv([_make_single])
    
    # 3. [优化] 智能路径搜索逻辑
    if args.load_dir:
        # 如果指定了文件夹，直接从那里加载
        load_path = Path(args.load_dir)
        stats_path = load_path / "vec_normalize.pkl"
        model_path = load_path / "model"
    else:
        # 否则，按优先级搜索：latest -> ppo_maskable -> vec_normalize.pkl
        stats_candidates = [
            args.save_dir / "latest_vec_normalize.pkl",
            args.save_dir / "vec_normalize_ppo_maskable.pkl",
            Path("vec_normalize.pkl")
        ]
        stats_path = next((p for p in stats_candidates if p.exists()), None)
        
        model_candidates = [
            args.save_dir / "latest_model",
            args.save_dir / "ppo_maskable",
            Path("ppo_chem_gym")
        ]
        model_path = next((p for p in model_candidates if p.with_suffix(".zip").exists()), None)

    # 4. 加载归一化统计数据
    if stats_path and stats_path.exists():
        print(f"[Eval] Loading normalization stats from {stats_path}...")
        try:
            venv = VecNormalize.load(str(stats_path), base_venv)
            venv.training = False
            venv.norm_reward = False
        except Exception as e:
            print(f"[Error] Failed to load stats: {e}. Dimension mismatch likely.")
            venv = base_venv
    else:
        print(f"[Warning] Normalization stats not found. Using unnormalized environment.")
        venv = base_venv

    # 5. 加载模型
    if not model_path or not model_path.with_suffix(".zip").exists():
        print(f"Error: Model not found! Searched in: {model_path}")
        return
        
    print(f"[Eval] Loading model from {model_path}...")
    try:
        model = MaskablePPO.load(model_path, env=venv)
        is_maskable = True
        print("Successfully loaded MaskablePPO model.")
    except Exception:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=venv)
        is_maskable = False
        print("Successfully loaded standard PPO model.")

    # 6. 开始优化过程
    obs = venv.reset()
    print("\n" + "="*30)
    print("  STARTING STOCHASTIC OPTIMIZATION  ")
    print("="*30)
    
    best_energy = float('inf')
    from sb3_contrib.common.maskable.utils import get_action_masks

    for step in range(1000): # 增加到 1000 步
        if is_maskable:
            masks = get_action_masks(venv)
            action, _ = model.predict(obs, action_masks=masks, deterministic=False)
        else:
            action, _ = model.predict(obs, deterministic=False)
        
        obs, rewards, dones, infos = venv.step(action)
        current_energy = infos[0]['energy']
        
        if step % 50 == 0 or current_energy < best_energy:
            print(f"Step {step+1:03d} | Energy: {current_energy:.6f} eV/atom")
        
        if current_energy < best_energy:
            best_energy = current_energy
            infos[0]['atoms'].write("best_optimized.xyz")

    print("="*30)
    print(f"Final Best Energy: {best_energy:.6f} eV/atom")
    from chem_gym.analysis.advanced_vis import plot_structure_plotly
    plot_structure_plotly(infos[0]['atoms'], f"Eval Best: {best_energy:.4f} eV", "eval")

if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "train":
        launch_train(cli_args)
    elif cli_args.mode == "eval": 
        launch_eval(cli_args)
    else:
        launch_baselines(cli_args)