"""
Demo script for AdsorptionChemGymEnv.

This script demonstrates how to use the adsorption energy optimization environment.
"""
import sys
sys.path.insert(0, '/root/shared-nvme/Final_Project')

from chem_gym.config import EnvConfig, AdsorptionConfig
from chem_gym.envs.adsorption_env import AdsorptionChemGymEnv
import numpy as np


def demo_basic_usage():
    """Demonstrate basic usage of the adsorption environment."""
    print("=" * 70)
    print("AdsorptionChemGymEnv - Basic Usage Demo")
    print("=" * 70)

    # 1. Create configurations
    print("\n1. Creating configurations...")
    env_config = EnvConfig(
        mode="image",
        element_types=["Pt", "Ag"],
        slab_size=(4, 4),
        n_layers=4,
        n_active_layers=3,
        max_steps=50
    )

    ads_config = AdsorptionConfig(
        adsorbate="CO",
        target_ads_energy=-0.5,
        adsorbate_height=2.0,
        energy_tolerance=0.1,
        enable_relaxation=True,
        cache_enabled=True
    )

    print(f"   ✓ Environment config: {env_config.slab_size}, {env_config.n_active_layers} active layers")
    print(f"   ✓ Adsorption config: {ads_config.adsorbate}, target: {ads_config.target_ads_energy:.2f} eV")

    # 2. Create environment
    print("\n2. Creating environment...")
    env = AdsorptionChemGymEnv(env_config, ads_config, oracle=None, surrogate=None)
    print(f"   ✓ Environment created")
    print(f"   ✓ Action space size: {env.action_space.n}")
    print(f"   ✓ Observation shape: {env.observation_space.shape}")

    # 3. Reset environment
    print("\n3. Resetting environment...")
    obs, info = env.reset(seed=42)
    print(f"   ✓ Environment reset")
    print(f"   ✓ Initial energy: {info['energy']:.4f} eV")
    print(f"   ✓ Number of atoms: {len(info['atoms'])}")
    print(f"   ✓ Adsorbate: {info['adsorbate']}")

    # 4. Run a few steps
    print("\n4. Running optimization steps...")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"   Step {step + 1:2d}: energy={info['energy']:8.4f} eV, "
              f"reward={reward:8.2f}, action={action:4d}")

        if terminated or truncated:
            print(f"   Episode ended: terminated={terminated}, truncated={truncated}")
            break

    # 5. Test different adsorbates
    print("\n5. Testing different adsorbates...")
    for adsorbate in ["CO", "O2", "H2"]:
        ads_config.adsorbate = adsorbate
        test_env = AdsorptionChemGymEnv(env_config, ads_config, oracle=None, surrogate=None)
        obs, info = test_env.reset(seed=42)
        print(f"   ✓ {adsorbate}: initial energy = {info['energy']:.4f} eV")

    # 6. Test different targets
    print("\n6. Testing different target energies...")
    for target in [-0.3, -0.5, -0.8]:
        ads_config.target_ads_energy = target
        test_env = AdsorptionChemGymEnv(env_config, ads_config, oracle=None, surrogate=None)
        obs, info = test_env.reset(seed=42)
        reward = test_env._calculate_adsorption_reward(info['energy'])
        print(f"   ✓ Target {target:.1f} eV: initial reward = {reward:.2f}")

    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


def demo_with_oracle():
    """
    Demonstrate usage with Oracle (if available).

    This requires a trained EquiformerV2 model.
    """
    print("\n" + "=" * 70)
    print("AdsorptionChemGymEnv - With Oracle Demo")
    print("=" * 70)

    try:
        # This would require loading an actual Oracle model
        # from chem_gym.surrogate.ocp_model import EquiformerV2Oracle

        # oracle = EquiformerV2Oracle(checkpoint_path="checkpoints/eq2_83M_2M.pt")

        print("\nNote: Oracle integration requires:")
        print("  1. Trained EquiformerV2 checkpoint")
        print("  2. fairchem-core installation")
        print("  3. GPU with sufficient memory")

        print("\nExample usage with Oracle:")
        print("""
        from chem_gym.surrogate.ocp_model import EquiformerV2Oracle

        oracle = EquiformerV2Oracle(
            checkpoint_path="checkpoints/eq2_83M_2M.pt",
            fmax=0.05,
            max_steps=100
        )

        env = AdsorptionChemGymEnv(env_config, ads_config, oracle=oracle)
        """)

    except Exception as e:
        print(f"\nOracle not available: {e}")


def demo_training_setup():
    """Demonstrate how to set up for RL training."""
    print("\n" + "=" * 70)
    print("AdsorptionChemGymEnv - Training Setup Demo")
    print("=" * 70)

    print("\nFor RL training, you would typically:")
    print("""
    1. Create multiple environments for vectorized training:
       from stable_baselines3 import PPO
       from stable_baselines3.common.env_util import make_vec_env

       def make_env():
           return AdsorptionChemGymEnv(env_config, ads_config, oracle=oracle)

       envs = make_vec_env(make_env, n_envs=4)

    2. Create and train PPO agent:
       model = PPO("MlpPolicy", envs, verbose=1)
       model.learn(total_timesteps=100000)

    3. Evaluate the trained model:
       obs, _ = envs.reset()
       for _ in range(100):
           action, _ = model.predict(obs)
           obs, reward, terminated, truncated, _ = envs.step(action)
    """)


def main():
    """Run all demos."""
    demo_basic_usage()
    demo_with_oracle()
    demo_training_setup()

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Load EquiformerV2 Oracle for accurate adsorption energies")
    print("2. Train PPO agent on the environment")
    print("3. Monitor training with TensorBoard")
    print("4. Analyze optimized structures")
    print("=" * 70)


if __name__ == "__main__":
    main()
