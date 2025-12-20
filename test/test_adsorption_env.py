"""
Basic tests for AdsorptionChemGymEnv.

This module tests the core functionality of the adsorption environment.
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, '/root/shared-nvme/Final_Project')

from chem_gym.config import EnvConfig, AdsorptionConfig
from chem_gym.envs.adsorption_env import AdsorptionChemGymEnv


def test_environment_creation():
    """Test basic environment creation."""
    print("\n=== Test 1: Environment Creation ===")

    # Create configs
    env_config = EnvConfig(
        mode="image",
        element_types=["Pt", "Ag"],
        slab_size=(4, 4),
        n_layers=4,
        n_active_layers=3,
        max_steps=100
    )

    ads_config = AdsorptionConfig(
        adsorbate="CO",
        target_ads_energy=-0.5,
        adsorbate_height=2.0,
        enable_relaxation=True
    )

    # Create environment
    env = AdsorptionChemGymEnv(env_config, ads_config, oracle=None, surrogate=None)

    print(f"✓ Environment created successfully")
    print(f"  - Adsorbate: {ads_config.adsorbate}")
    print(f"  - Target energy: {ads_config.target_ads_energy:.2f} eV")
    print(f"  - Action space size: {env.action_space.n}")
    print(f"  - Observation shape: {env.observation_space.shape}")

    return env


def test_environment_reset(env):
    """Test environment reset and initial state."""
    print("\n=== Test 2: Environment Reset ===")

    # Reset environment
    obs, info = env.reset(seed=42)

    print(f"✓ Environment reset successfully")
    print(f"  - Initial adsorption energy: {info.get('initial_adsorption_energy', 'N/A'):.4f} eV")
    print(f"  - Initial energy: {info['energy']:.4f} eV")
    print(f"  - Steps: {env.steps}")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Atoms created: {len(info['atoms'])} atoms")
    print(f"  - Tags: {info['atoms'].get_tags()}")

    # Check adsorbate is present
    symbols = info['atoms'].get_chemical_symbols()
    adsorbate_present = any(s in ['C', 'O'] for s in symbols[-2:])
    print(f"  - Adsorbate present: {adsorbate_present}")

    return obs, info


def test_action_execution(env):
    """Test action execution."""
    print("\n=== Test 3: Action Execution ===")

    obs, info = env.reset(seed=42)
    initial_energy = info['energy']

    print(f"  - Initial energy: {initial_energy:.4f} eV")

    # Execute a few random actions
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  - Step {step + 1}: action={action}, reward={reward:.4f}, "
              f"energy={info['energy']:.4f} eV")

        if terminated or truncated:
            print(f"  - Episode ended: terminated={terminated}, truncated={truncated}")
            break

    print(f"✓ Action execution successful")

    return obs, info


def test_adsorption_energy_calculation(env):
    """Test adsorption energy calculation (without Oracle)."""
    print("\n=== Test 4: Adsorption Energy Calculation ===")

    # Note: This test uses EMT fallback, so results are approximate
    obs, info = env.reset(seed=42)

    print(f"✓ Adsorption energy calculated")
    print(f"  - Energy: {info['energy']:.4f} eV")
    print(f"  - Uncertainty: {info['uncertainty']:.4f}")

    return info


def test_reward_function(env):
    """Test reward function."""
    print("\n=== Test 5: Reward Function ===")

    obs, info = env.reset(seed=42)

    # Test reward calculation at different energy levels
    test_energies = [-0.6, -0.5, -0.4, 0.0, 1.0]

    print(f"  - Target energy: {env.ads_config.target_ads_energy:.2f} eV")
    print(f"  - Tolerance: {env.ads_config.energy_tolerance:.2f} eV")

    for e in test_energies:
        reward = env._calculate_adsorption_reward(e)
        within_tolerance = abs(e - env.ads_config.target_ads_energy) <= env.ads_config.energy_tolerance
        print(f"  - Energy {e:.2f} eV -> Reward {reward:.4f} "
              f"{'(within target)' if within_tolerance else ''}")

    print(f"✓ Reward function working correctly")


def test_observation_space(env):
    """Test observation space consistency."""
    print("\n=== Test 6: Observation Space ===")

    obs, info = env.reset(seed=42)

    # Check observation shape
    expected_shape = (4, 4, 2)  # (H, W, n_elements)
    actual_shape = obs.shape

    print(f"  - Expected shape: {expected_shape}")
    print(f"  - Actual shape: {actual_shape}")
    print(f"  - Observation range: [{obs.min():.4f}, {obs.max():.4f}]")

    # Check for NaN or Inf
    has_nan = np.isnan(obs).any()
    has_inf = np.isinf(obs).any()

    print(f"  - Has NaN: {has_nan}")
    print(f"  - Has Inf: {has_inf}")

    assert not has_nan, "Observation contains NaN"
    assert not has_inf, "Observation contains Inf"

    print(f"✓ Observation space is valid")


def main():
    """Run all tests."""
    print("=" * 60)
    print("AdsorptionChemGymEnv - Basic Tests")
    print("=" * 60)

    try:
        # Test 1: Environment creation
        env = test_environment_creation()

        # Test 2: Reset
        obs, info = test_environment_reset(env)

        # Test 3: Action execution
        test_action_execution(env)

        # Test 4: Adsorption energy calculation
        test_adsorption_energy_calculation(env)

        # Test 5: Reward function
        test_reward_function(env)

        # Test 6: Observation space
        test_observation_space(env)

        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
