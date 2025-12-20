# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Chem-Gym is a surrogate-assisted active reinforcement learning framework for high-entropy alloy (HEA) surface optimization. The system uses RL agents to find low-energy surface configurations, leveraging surrogate models for fast predictions and EquiformerV2 as a high-fidelity Oracle for active learning.

## Development Commands

### Running Training
```bash
# Basic PPO training with image observations (MLP policy)
python main.py --mode train --obs-mode image --total-steps 5000

# Training with uncertainty penalty and oracle threshold
python main.py --mode train --obs-mode image --total-steps 10000 \
  --uncertainty-penalty 0.05 --oracle-threshold 0.3 \
  --oracle-ckpt checkpoints/eq2_83M_2M.pt

# Multi-environment parallel training
python main.py --mode train --obs-mode image --n-envs 4 --device cuda

# Graph-based observations (requires custom GNN policy)
python main.py --mode train --obs-mode graph --total-steps 5000
```

### Running Baselines
```bash
# Run random search and simulated annealing baselines
python main.py --mode baseline --obs-mode image
```

### Testing and Validation
```bash
# Test EquiformerV2 Oracle loading and inference
python chem_gym/surrogate/ocp_model.py

# Diagnose fairchem installation issues
python diagnose_fairchem.py

# Check environment setup
python check_envs.py

# Verify physics engine (EMT) is working correctly
python test/verify_physics.py

# Test Oracle integration with mock mode
python test/test_oracle.py

# Test real physics calculations with Oracle
python test/test_oracle_real_physics.py

# Verify active region configuration
python test/verify_active_region.py

# Test and demo Oracle parameters
python test/test_oracle_params.py
python test/demo_oracle_params.py
```

### Running Evaluation
```bash
# Load trained model and perform greedy quench optimization
python main.py --mode eval --n-active-layers 3

# Evaluation with specific configuration
python main.py --mode eval --obs-mode graph --n-envs 1 --device cuda
```

### Dependencies
Basic installation (CPU-compatible):
```bash
pip install -r requirements.txt
```

For OCP/EquiformerV2 support (requires GPU):
```bash
# Install fairchem packages for OCP models
pip install fairchem-core torch-geometric
# Download checkpoint from OCP repository
```

## Architecture

### Core Module Structure

**Module A: Environment** (`chem_gym/envs/chem_env.py`)
- Gymnasium-based RL environment for atom swapping on fcc(111) HEA surfaces
- Two observation modes:
  - `image`: (H, W, N_elements) one-hot grid for CNN/MLP policies
  - `graph`: node features + adjacency matrix placeholder (extensible to edge indices/distances for GNN)
- Action space: discrete pairwise atom swaps on surface sites (n_sites*(n_sites-1)/2 actions)
- Reward structure: differential energy improvement with step penalty and optional uncertainty penalty
- State tracking: initial_energy, prev_energy, current_energy, current_uncertainty
- Renders ASE Atoms objects for visualization

**Module B: Surrogate Ensemble** (`chem_gym/surrogate/`)
- `ensemble.py`: Lightweight ensemble wrapper with caching
  - Evaluates multiple models and returns (mean_energy, std_uncertainty)
  - Hash-based cache (symbols + positions) prevents duplicate inference
  - `update_with_oracle()`: writes high-fidelity labels for active learning
- `ocp_model.py`: EquiformerV2 Oracle integration
  - Loads OCP checkpoints with fairchem-core
  - Converts ASE Atoms to PyG graphs via AtomsToGraphs
  - Returns high-fidelity energy predictions (slow but accurate)

**Module C: Policy & Active Learning** (`chem_gym/agent/trainer.py`)
- `TrainConfig`: PPO hyperparameters + uncertainty_penalty + oracle_threshold
- Wrappers:
  - `UncertaintyPenaltyWrapper`: penalizes reward by uncertainty (R -= λ·σ)
  - `OracleWrapper`: triggers Oracle when σ > threshold, updates cache
- `make_vec_env()`: constructs vectorized environments with wrappers
- `train_agent()`: runs PPO with MlpPolicy (image mode) or MultiInputPolicy (graph mode)

**Supporting Modules**
- `chem_gym/config.py`: Centralized dataclass configs (EnvConfig, TrainConfig, SurrogateConfig)
- `chem_gym/active_learning.py`: ReplayBuffer and oracle querying utilities
- `chem_gym/baselines.py`: Random search and simulated annealing baselines
- `main.py`: CLI entry point with argument parsing

### Key Design Patterns

**Active Learning Loop**
1. Agent takes action, environment evaluates with surrogate ensemble
2. Surrogate returns (energy_mean, uncertainty_std)
3. If uncertainty > threshold, OracleWrapper triggers EquiformerV2
4. Oracle's high-fidelity energy is cached with σ=0
5. Future queries to same state return cached value instantly

**Caching Strategy**
- Hash function: MD5 of (symbols + positions rounded to 3 decimals)
- Cache hit returns (energy, 0.0) with zero uncertainty
- Dramatically reduces expensive Oracle calls during training

**Reward Engineering**
- Base: differential reward = prev_energy - current_energy
- Step penalty: discourages excessive exploration
- Uncertainty penalty (optional): R -= λ·σ encourages low-uncertainty regions
- Stoichiometry control: fixed element ratios, only positions shuffled

**Policy Selection**
- Image mode (4x4 grid): Uses MlpPolicy (NOT CnnPolicy - too small for CNN)
- Graph mode: Uses MultiInputPolicy for Dict observations
- Custom GNN policies can replace MultiInputPolicy by inheriting SB3's ActorCriticPolicy

### State Management

**Environment State Variables** (chem_env.py:69-74)
- `self.state`: Flat array of element indices (length = n_sites)
- `self.atoms`: ASE Atoms object rebuilt from state
- `self.initial_energy`: Energy at reset
- `self.prev_energy`: Previous step's energy (for differential rewards)
- `self.current_energy`: Current energy after action
- `self.current_uncertainty`: Current surrogate uncertainty (σ)

**Surrogate Cache** (ensemble.py:43)
- Dictionary mapping hash strings to (energy, uncertainty) tuples
- Populated during inference and by Oracle updates
- Persists across episodes within same training run

## Important Implementation Details

### Hash Function Precision
The default hash in `ensemble.py:17-26` uses `np.array2string(positions, precision=3)` which rounds coordinates to 3 decimals. This is critical for cache hits - atoms must have identical hashed positions to match.

### Action Mapping
Action integers are decoded to (i, j) swap indices via `_action_to_indices` (chem_env.py:161-170). The mapping iterates through all pairs in lexicographic order.

### Policy Architecture Fix
Original code used CnnPolicy which crashed on 4x4 grids. Changed to MlpPolicy in trainer.py:67 which flattens the input. For larger grids (>8x8), CnnPolicy may be appropriate.

### Oracle Integration Points
When implementing OracleWrapper or custom oracle functions:
- Oracle receives `info["atoms"]` from environment step (chem_env.py:139)
- Must return float energy value
- Called only when `info["uncertainty"] > threshold`
- Result written to cache via `surrogate.update_with_oracle(atoms, energy)`

### Stoichiometry Control
Environment reset ensures fixed elemental composition (chem_env.py:81-93) by:
1. Calculating base count per element (n_sites // n_elements)
2. Distributing remainder to first elements
3. Shuffling only positions, not composition

This prevents composition drift and ensures comparable energy evaluations.

## File Organization

```
chem_gym/
├── __init__.py              # Package exports
├── config.py                # Centralized configuration dataclasses
├── envs/
│   ├── __init__.py
│   └── chem_env.py          # ChemGymEnv Gymnasium environment
├── surrogate/
│   ├── ensemble.py          # SurrogateEnsemble with caching
│   └── ocp_model.py         # EquiformerV2Oracle OCP integration
├── agent/
│   ├── trainer.py           # PPO training, wrappers, vec_env
│   └── graph_feature_extractor.py  # CrystalGraphFeatureExtractor for GNN
├── analysis/
│   ├── advanced_vis.py      # Interactive 3D visualization with Plotly
│   ├── pmg_utils.py         # Pymatgen interface for data calibration
│   └── vis_callback.py      # Visualization callback for training
├── active_learning.py       # ReplayBuffer and oracle utilities
└── baselines.py             # Random search, simulated annealing

main.py                      # CLI entry point
check_envs.py                # Environment validation script
diagnose_fairchem.py         # OCP installation diagnostic

test/
├── check_envs.py            # Check environment dependencies
├── diagnose_fairchem.py     # Diagnose OCP installation issues
├── verify_physics.py        # Verify EMT physics engine
├── verify_active_region.py  # Verify active region configuration
├── test_oracle.py           # Test Oracle integration (mock mode)
├── test_oracle_real_physics.py  # Test real physics with Oracle
├── test_oracle_params.py    # Test Oracle parameters
└── demo_oracle_params.py    # Demo Oracle parameter usage
```

## Common Development Patterns

### Adding New Surrogate Models
Extend `SurrogateEnsemble.models` list in ensemble.py:
1. Implement model as callable `(Atoms) -> float`
2. Add to `self.models` during `__post_init__`
3. Ensemble automatically computes mean/std across all models

### Extending Graph Observations
Current graph mode (chem_env.py:53-62) uses full adjacency matrix placeholder:
1. Replace adjacency with edge_index (COO format) for GNN compatibility
2. Add edge attributes (distances, bond types)
3. Implement custom GNN policy by subclassing SB3's ActorCriticPolicy

### Custom Reward Shaping
Modify reward calculation in chem_env.py:121-127:
- Current: differential + step penalty
- Extensions: add distance to target composition, surface roughness penalty, etc.
- Uncertainty penalty handled by UncertaintyPenaltyWrapper separately

### Batched Surrogate Inference
For vectorized environments (n_envs > 1):
1. Collect multiple Atoms objects from parallel environments
2. Batch convert via AtomsToGraphs in ocp_model.py
3. Single forward pass through model for all graphs
4. Reduces GPU kernel launch overhead

## Analysis and Visualization

### Interactive 3D Visualization (`chem_gym/analysis/advanced_vis.py`)
- `plot_structure_plotly()`: Generate interactive 3D plots using Plotly
- Supports element-specific colors and radii
- Renders crystal lattice framework
- Output: HTML files for web-based viewing
- Requires: pymatgen, plotly

```python
from ase.io import read
from chem_gym.analysis.advanced_vis import plot_structure_plotly

atoms = read('best_optimized.xyz')
plot_structure_plotly(atoms, 'Optimized HEA Structure', 'output.html')
```

### Pymatgen Interface (`chem_gym/analysis/pmg_utils.py`)
- `PymatgenInterface`: Core utilities for structure analysis
- Features:
  - Trajectory recording (CIF animations)
  - Layer composition analysis
  - Data calibration via Materials Project API
- Uses Materials Project API for real crystal data
- Includes API key: `"LWy9cEJNTrC8Bk1b5QEfsL6EU9tLnZiw"`

### Visualization Callback (`chem_gym/analysis/vis_callback.py`)
- `VisualizationCallback`: Automatically saves structures during training
- Generates:
  - Static PNG plots (top/side views)
  - CIF files for each checkpoint
  - Interactive HTML (if Plotly available)
- Integrates with SB3 training loop
- Configurable save frequency

## Test Suite

### Environment and Physics Tests
- **`test/verify_physics.py`**: Validates EMT physics engine
  - Checks energy calculations are negative (proper binding)
  - Verifies atom constraints are applied
  - Tests energy changes during atom swaps
  - Run before first training to ensure setup is correct

- **`test/verify_active_region.py`**: Confirms active layer configuration
  - Validates surface atom selection
  - Checks constraint boundaries

### Oracle Integration Tests
- **`test/test_oracle.py`**: Tests Oracle integration with mock mode
  - Safe to run without large model weights
  - Validates code logic flow
  - MOCK_MODE flag for testing without GPU

- **`test/test_oracle_real_physics.py`**: Tests with real Oracle calculations
  - Requires EquiformerV2 checkpoint
  - Validates actual energy predictions
  - Tests structure relaxation

- **`test/test_oracle_params.py` & `demo_oracle_params.py`**:
  - Tests Oracle parameter configuration
  - Demonstrates fmax, max_steps, disable_amp usage
  - Validates parameter effects on relaxation

### Setup and Diagnostics
- **`test/check_envs.py`**: Comprehensive environment check
  - Lists all package versions
  - Checks CUDA availability
  - Validates GPU device info

- **`test/diagnose_fairchem.py`**: Diagnoses OCP installation issues
  - Tests fairchem imports
  - Validates checkpoint loading
  - Reports configuration problems

## Graph Feature Extractor

### CrystalGraphFeatureExtractor (`chem_gym/agent/graph_feature_extractor.py`)
- Custom neural network for processing graph observations
- Inherits from SB3's `BaseFeaturesExtractor`
- Architecture:
  - Node embedding layer (Linear → GELU → Linear → LayerNorm)
  - GraphResidualBlocks (message passing with residual connections)
  - Multi-layer GCN with layer normalization
- Input: Graph observations with node features and adjacency matrix
- Output: 256-dimensional feature vector for policy network
- Compatible with `MultiInputPolicy` for graph mode

### GraphResidualBlock
- Message passing layer with residual connections
- Performs: A_norm * (W * H) for message passing
- Includes layer normalization for training stability
- GELU activation functions

## Evaluation Mode

### Running Trained Models
The evaluation mode (`--mode eval`) loads a trained PPO model and performs greedy optimization:

```bash
# Basic evaluation
python main.py --mode eval --n-active-layers 3

# With specific configuration
python main.py --mode eval --obs-mode graph --device cuda
```

### Evaluation Process
1. Loads checkpoint from `checkpoints/` directory
2. Runs greedy policy (no exploration)
3. Performs atom swaps to minimize energy
4. Saves best structures during optimization
5. Outputs final optimized configuration

## Oracle Configuration Parameters

### Key Oracle Parameters
- **`--oracle-ckpt`**: Path to EquiformerV2 checkpoint (default: `checkpoints/eq2_83M_2M.pt`)
- **`--oracle-fmax`**: Relaxation convergence threshold in eV/A (default: 0.05)
  - Lower values = more precise relaxation, more computation
  - Higher values = faster convergence, less precise
- **`--oracle-max-steps`**: Maximum relaxation steps (default: 100)
- **`--oracle-disable-amp`**: Disable Automatic Mixed Precision (default: True)
  - AMP improves speed but may cause instability
  - Set to False for faster inference if stable

### Oracle Usage in Training
```bash
# High-precision relaxation (slower but more accurate)
python main.py --mode train --oracle-fmax 0.01 --oracle-max-steps 200

# Fast relaxation (faster training, less accurate)
python main.py --mode train --oracle-fmax 0.1 --oracle-max-steps 50
```

## Testing and Validation

**Basic Environment Test**
```python
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.config import EnvConfig

config = EnvConfig(mode="image", max_steps=10)
env = ChemGymEnv(config)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

**Surrogate Cache Test**
```python
from chem_gym.surrogate.ensemble import SurrogateEnsemble
from ase.build import fcc111

surrogate = SurrogateEnsemble()
atoms = fcc111("Cu", size=(4, 4, 3))
e1, u1 = surrogate.evaluate(atoms)  # Computes
e2, u2 = surrogate.evaluate(atoms)  # Cache hit
assert e1 == e2 and u2 == 0.0  # Cache returns zero uncertainty
```

## Known Issues and Gotchas

**CnnPolicy Crashes**: Small grids (4x4) cause dimension errors with CnnPolicy. Always use MlpPolicy for image mode with grids <8x8.

**Fairchem Import Paths**: fairchem-core has varying import paths across versions. ocp_model.py includes fallback logic for registry, AtomsToGraphs locations.

**Cache Key Collisions**: Hash precision of 3 decimals means atoms within 0.001 Angstrom are treated as identical. Increase precision if needed for finer distinctions.

**ASE Tags Requirement**: Oracle prediction requires proper tags on atoms (ocp_model.py:148-152). Surface atoms get tag=1, constrained atoms get tag=0.

**Graph Mode Policy**: Default MultiInputPolicy may not support graph structure properly. Implement custom GNN policy or use external RL libraries (d3rlpy, RLlib) for graph-based agents.
