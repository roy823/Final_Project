import numpy as np
from chem_gym.config import EnvConfig
from chem_gym.envs.chem_env import ChemGymEnv

def test_active_region():
    print("=== Testing Active Region Logic ===")
    
    # 配置: 4层 Slab, 优化顶部 2 层
    config = EnvConfig(
        mode="graph",
        slab_size=(4, 4),
        n_layers=4,
        n_active_layers=2
    )
    
    env = ChemGymEnv(config)
    obs, info = env.reset()
    atoms = info["atoms"]
    
    print(f"Total atoms: {len(atoms)}")
    print(f"Active layers: {config.n_active_layers}")
    print(f"Sites per layer: {env.n_sites_per_layer}")
    print(f"Active atoms count: {env.n_active_atoms}")
    
    # 1. 检查动作空间大小
    expected_actions = env.n_active_atoms * (env.n_active_atoms - 1) // 2
    print(f"Action Space: {env.action_space.n} (Expected: {expected_actions})")
    assert env.action_space.n == expected_actions, "Action space size mismatch!"
    
    # 2. 检查约束 (Constraints)
    # 底部 (Total - Active) 个原子应该被固定
    constraints = atoms.constraints
    if len(constraints) > 0:
        fixed_indices = constraints[0].get_indices()
        n_fixed = len(atoms) - env.n_active_atoms
        print(f"Fixed atoms count: {len(fixed_indices)} (Expected: {n_fixed})")
        
        # 验证被固定的确实是底部原子 (索引 0 到 n_fixed-1)
        expected_fixed_indices = np.arange(n_fixed)
        if np.array_equal(fixed_indices, expected_fixed_indices):
            print("✅ Constraints are correctly applied to bottom layers.")
        else:
            print("❌ Constraints indices mismatch!")
    else:
        print("❌ No constraints found!")

    # 3. 检查观察空间 (Layer Index)
    node_feats = obs["node_features"]
    layer_indices = node_feats[:, -1] # 最后一维是 Layer Index
    print(f"Layer indices in observation: {np.unique(layer_indices)}")
    
    assert len(np.unique(layer_indices)) == config.n_layers, "Layer indices should match n_layers"
    print("✅ Layer indices are correctly encoded.")

    print("\n=== Test Passed ===")

if __name__ == "__main__":
    test_active_region()