import argparse
from pathlib import Path
import torch
from datetime import datetime

from chem_gym.agent.trainer import train_agent
from chem_gym.baselines import random_search, simulated_annealing
from chem_gym.config import EnvConfig, SurrogateConfig, TrainConfig, AdsorptionConfig
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.surrogate.ensemble import SurrogateEnsemble
from chem_gym.surrogate.ocp_model import EquiformerV2Oracle
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO

def parse_args():
    parser = argparse.ArgumentParser(description="Chem-Gym scaffold launcher.")
    parser.add_argument("--mode", choices=["train", "baseline", "eval", "adsorption_train"], default="train")
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

    # Adsorption params
    parser.add_argument("--adsorbate", type=str, default="CO", help="Adsorbate type (CO, O2, H2, NO)")
    parser.add_argument("--target-ads-energy", type=float, default=-0.5, help="Target adsorption energy (eV)")
    parser.add_argument("--adsorbate-height", type=float, default=2.0, help="Initial adsorbate height (Å)")
    parser.add_argument("--energy-tolerance", type=float, default=0.1, help="Energy tolerance for target (eV)")

    # Result directory
    parser.add_argument("--result-dir", type=Path, default=None, help="Custom result directory. If not specified, uses timestamp-based directory in ./result/")
    return parser.parse_args()


def launch_train(args):
    # Create result directory with timestamp
    if args.result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path("result") / f"{timestamp}_training"
    else:
        result_dir = args.result_dir
        timestamp = result_dir.name

    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Main] Result directory: {result_dir}")
    print(f"[Main] Timestamp: {timestamp}\n")

    # Create subdirectories
    models_dir = result_dir / "models"
    logs_dir = result_dir / "logs"
    visualizations_dir = result_dir / "visualizations"
    structures_dir = result_dir / "structures"

    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    visualizations_dir.mkdir(exist_ok=True)
    structures_dir.mkdir(exist_ok=True)

    # 1. 初始化 Oracle (Equiformer V2)
    eq2_model = None
    real_oracle = None
    if args.oracle_ckpt and Path(args.oracle_ckpt).exists():
        print(f"[Main] Loading real Oracle from {args.oracle_ckpt}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # 4. 开始训练
    print(f"[Main] Starting training on {train_config.device}...")
    model = train_agent(
        env_config,
        surrogate,
        train_config,
        oracle_energy_fn=real_oracle,
        oracle=eq2_model,
        result_dir=str(result_dir),
        tensorboard_log=str(logs_dir / "tensorboard"),
        vis_dir=str(visualizations_dir),
        best_structure_path=str(structures_dir / "best_optimized.xyz")
    )

    # Save model and normalization stats
    model.save(models_dir / "ppo_chem_gym.zip")
    print(f"[Main] Model saved to {models_dir / 'ppo_chem_gym.zip'}")
    print(f"[Main] All results saved to {result_dir}")


def launch_adsorption_train(args):
    """Launch adsorption energy optimization training."""
    # Create result directory with timestamp
    if args.result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path("result") / f"{timestamp}_adsorption"
    else:
        result_dir = args.result_dir
        timestamp = result_dir.name

    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Main] Adsorption Training Result directory: {result_dir}")
    print(f"[Main] Timestamp: {timestamp}\n")

    # Create subdirectories
    models_dir = result_dir / "models"
    logs_dir = result_dir / "logs"
    visualizations_dir = result_dir / "visualizations"
    structures_dir = result_dir / "structures"

    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    visualizations_dir.mkdir(exist_ok=True)
    structures_dir.mkdir(exist_ok=True)

    # 1. 初始化 Oracle (Equiformer V2)
    eq2_model = None
    if args.oracle_ckpt and Path(args.oracle_ckpt).exists():
        print(f"[Main] Loading Oracle from {args.oracle_ckpt} for adsorption training...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            eq2_model = EquiformerV2Oracle(
                args.oracle_ckpt,
                device=device,
                fmax=args.oracle_fmax,
                max_steps=args.oracle_max_steps
            )
            print("[Main] EquiformerV2 Oracle loaded successfully.")
        except Exception as e:
            print(f"[Main] Failed to load Oracle: {e}")
            print("[Main] Warning: Oracle is recommended for accurate adsorption energies!")
    else:
        print(f"[Main] Oracle checkpoint not found. Using EMT (less accurate for adsorption).")

    # 2. 创建环境配置
    env_config = EnvConfig(
        mode=args.obs_mode,
        init_seed=args.seed,
        n_active_layers=args.n_active_layers
    )

    # 3. 创建吸附能配置
    ads_config = AdsorptionConfig(
        adsorbate=args.adsorbate,
        target_ads_energy=args.target_ads_energy,
        adsorbate_height=args.adsorbate_height,
        energy_tolerance=args.energy_tolerance
    )

    # 4. 训练配置
    train_config = TrainConfig(
        total_timesteps=args.total_steps,
        n_envs=args.n_envs,
        device=args.device,
        learning_rate=args.learning_rate,
        oracle_fmax=args.oracle_fmax,
        oracle_max_steps=args.oracle_max_steps,
        oracle_disable_amp=args.oracle_disable_amp,
    )

    # 5. 开始吸附能训练
    print(f"[Main] Starting adsorption training on {train_config.device}...")
    print(f"[Main] Adsorbate: {ads_config.adsorbate}")
    print(f"[Main] Target energy: {ads_config.target_ads_energy:.2f} eV")
    print(f"[Main] Energy tolerance: ±{ads_config.energy_tolerance:.2f} eV")

    model = train_agent(
        env_config,
        surrogate=None,  # 吸附能训练直接使用Oracle
        train_config=train_config,
        oracle_energy_fn=None,
        oracle=eq2_model,
        ads_config=ads_config,
        use_adsorption_env=True,
        result_dir=str(result_dir),
        tensorboard_log=str(logs_dir / "tensorboard"),
        vis_dir=str(visualizations_dir),
        best_structure_path=str(structures_dir / "best_optimized.xyz")
    )

    # Save model and normalization stats
    model.save(models_dir / "ppo_adsorption_chem_gym.zip")
    print(f"[Main] Adsorption model saved to {models_dir / 'ppo_adsorption_chem_gym.zip'}")
    print(f"[Main] All results saved to {result_dir}")


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
    # Create result directory for evaluation
    if args.result_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path("result") / f"{timestamp}_eval"
    else:
        result_dir = args.result_dir

    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Eval] Result directory: {result_dir}\n")

    structures_dir = result_dir / "structures"
    structures_dir.mkdir(exist_ok=True)

    # 初始化 Oracle (用于获取真实能量)
    eq2_model = None
    if args.oracle_ckpt and Path(args.oracle_ckpt).exists():
        print(f"[Eval] Loading Oracle from {args.oracle_ckpt}...")
        eq2_model = EquiformerV2Oracle(args.oracle_ckpt, device="cuda" if torch.cuda.is_available() else "cpu")

    # 环境配置
    env_config = EnvConfig(mode=args.obs_mode, n_active_layers=args.n_active_layers)

    # 创建基础环境 (不带归一化，因为我们要加载保存的归一化参数)
    from chem_gym.envs.chem_env import ChemGymEnv
    from stable_baselines3.common.vec_env import DummyVecEnv

    def _make_single():
        return ChemGymEnv(env_config, oracle=eq2_model)

    base_venv = DummyVecEnv([_make_single])

    # 加载归一化统计数据 (均值和方差)
    if args.result_dir is not None:
        stats_path = args.result_dir / "models" / "vec_normalize.pkl"
    else:
        stats_path = Path("vec_normalize.pkl")

    if stats_path.exists():
        print(f"[Eval] Loading normalization stats from {stats_path}...")
        venv = VecNormalize.load(stats_path, base_venv)
        # 关键：推理模式下关闭统计更新和奖励归一化
        venv.training = False
        venv.norm_reward = False
    else:
        print(f"[Warning] {stats_path} not found. Using unnormalized environment.")
        venv = base_venv

    # 加载 PPO 模型
    if args.result_dir is not None:
        model_path = args.result_dir / "models" / "ppo_chem_gym.zip"
    else:
        model_path = Path("ppo_chem_gym.zip")

    if not model_path.exists():
        print(f"Error: Model {model_path} not found!")
        return

    print(f"[Eval] Loading model from {model_path}...")
    model = PPO.load(model_path, env=venv)

    # 开始确定性优化过程
    obs = venv.reset()
    print("\n" + "="*30)
    print("  STARTING STOCHASTIC OPTIMIZATION  ")
    print("="*30)

    best_energy = float('inf')
    best_atoms = None

    # 记录已经尝试过的动作，防止死循环
    attempted_actions = set()

    best_structure_path = structures_dir / "best_optimized.xyz"

    for step in range(200): # 运行 200 步
        # --- [关键修改：不要用 deterministic=True] ---
        # 因为模型还没学稳，我们用随机采样，但增加采样次数
        action, _ = model.predict(obs, deterministic=False)

        # 获取当前环境的内部状态（用于检查是否是无效交换）
        # 注意：venv 是 VecNormalize，需要访问原始环境
        raw_env = venv.unwrapped.envs[0]
        i, j = raw_env._action_to_indices(int(action))

        # 如果交换的是相同元素，或者是已经试过没效果的动作，就重新采样
        retry = 0
        while raw_env.state[i] == raw_env.state[j] and retry < 100:
            action, _ = model.predict(obs, deterministic=False)
            i, j = raw_env._action_to_indices(int(action))
            retry += 1

        obs, rewards, dones, infos = venv.step(action)
        current_energy = infos[0]['energy']

        # 打印有意义的进度
        if step % 10 == 0 or current_energy < best_energy:
            print(f"Step {step+1:03d} | Energy: {current_energy:.6f} eV/atom | Action: ({i},{j})")

        if current_energy < best_energy:
            best_energy = current_energy
            best_atoms = infos[0]['atoms'].copy()
            best_atoms.write(str(best_structure_path)) # 实时保存最好的

    print("="*30)
    print(f"Final Best Energy: {best_energy:.6f} eV/atom")
    print(f"Best structure saved to {best_structure_path}")


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "train":
        launch_train(cli_args)
    elif cli_args.mode == "adsorption_train":
        launch_adsorption_train(cli_args)
    elif cli_args.mode == "eval":
        launch_eval(cli_args)
    else:
        launch_baselines(cli_args)