"""
Adsorption energy optimization environment based on ChemGymEnv.

This module extends the base ChemGymEnv to support adsorption energy optimization
by fixing the adsorbate position and using Oracle's automatic relaxation.
"""
from typing import Dict, Optional, Tuple
import numpy as np

try:
    from ase.build import fcc111, bulk, molecule
    from ase import Atoms
    from ase.calculators.emt import EMT
except ImportError:
    fcc111 = None
    bulk = None
    molecule = None
    Atoms = None
    EMT = None

from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.config import EnvConfig, AdsorptionConfig
from chem_gym.utils import AdsorptionCacheManager


class AdsorptionChemGymEnv(ChemGymEnv):
    """
    Adsorption energy optimization environment.

    Key features:
    - Fixed adsorbate position (CO, O2, H2, etc.)
    - Oracle automatic relaxation for optimal adsorption geometry
    - Simplified action space (only surface atom swaps)
    - Target-oriented reward function based on adsorption energy
    """

    def __init__(self, config: EnvConfig, ads_config: AdsorptionConfig,
                 surrogate=None, oracle=None):
        """
        Initialize adsorption environment.

        Args:
            config: Environment configuration
            ads_config: Adsorption-specific configuration
            surrogate: Surrogate model (not used - direct Oracle)
            oracle: EquiformerV2 Oracle for high-fidelity calculations
        """
        # Initialize base environment
        super().__init__(config, surrogate=None, oracle=oracle)

        self.ads_config = ads_config
        self.adsorbate_atoms = None

        # Initialize intelligent cache manager
        if self.ads_config.cache_enabled:
            max_cache_size = 2000  # Increased cache size for better performance
            self.cache_manager = AdsorptionCacheManager(max_size=max_cache_size)
            print(f"[AdsorptionChemGymEnv] Cache manager initialized (max_size={max_cache_size})")
        else:
            self.cache_manager = None
            print(f"[AdsorptionChemGymEnv] Cache manager disabled")

        # Pre-compute and cache gas reference energy
        if self.cache_manager is not None:
            self._precompute_reference_energy()

        # Setup adsorbate
        self._setup_adsorbate()

    def _precompute_reference_energy(self):
        """
        Pre-compute and cache gas reference energy for the adsorbate.

        This is a simplified approach - in practice, reference energies
        should come from quantum chemistry calculations or databases.
        """
        adsorbate = self.ads_config.adsorbate

        # Known gas reference energies (approximate values in eV)
        # These should be replaced with accurate values from quantum chemistry
        reference_energies = {
            "CO": -1.0,   # Carbon monoxide
            "O2": 0.0,    # Oxygen (reference state)
            "H2": 0.0,    # Hydrogen (reference state)
            "NO": -0.5,   # Nitric oxide
            "N2": 0.0,    # Nitrogen (reference state)
            "CH4": -2.0,  # Methane
        }

        # Get reference energy or use default
        ref_energy = reference_energies.get(adsorbate, self.ads_config.gas_reference_energy)

        # Cache the reference energy
        self.cache_manager.put_gas_reference_energy(adsorbate, ref_energy)
        print(f"[AdsorptionChemGymEnv] Cached reference energy for {adsorbate}: {ref_energy:.2f} eV")

    def _setup_adsorbate(self):
        """
        Initialize adsorbate structure (CO molecule by default).
        Place it at the specified height above the fcc site.
        """
        if molecule is None or Atoms is None:
            raise ImportError("ASE is required for adsorption environment")

        # Build adsorbate molecule
        if self.ads_config.adsorbate == "CO":
            # CO molecule: C-O bond length ~1.13 Å
            self.adsorbate_atoms = molecule("CO")
            # Position CO with C atom pointing down (towards surface)
            # C is closer to surface, O is farther
            self.adsorbate_atoms.set_positions([
                [0.0, 0.0, -0.565],  # C atom (closer to surface)
                [0.0, 0.0, 0.565],   # O atom (farther from surface)
            ])
        elif self.ads_config.adsorbate == "O2":
            # O2 molecule: O-O bond length ~1.21 Å
            self.adsorbate_atoms = molecule("O2")
            self.adsorbate_atoms.set_positions([
                [0.0, 0.0, -0.605],  # First O
                [0.0, 0.0, 0.605],   # Second O
            ])
        elif self.ads_config.adsorbate == "H2":
            # H2 molecule: H-H bond length ~0.74 Å
            self.adsorbate_atoms = molecule("H2")
            self.adsorbate_atoms.set_positions([
                [0.0, 0.0, -0.37],   # First H
                [0.0, 0.0, 0.37],    # Second H
            ])
        else:
            # Generic diatomic molecule
            self.adsorbate_atoms = molecule(self.ads_config.adsorbate)

        # Get the center position of adsorbate
        center = self.adsorbate_atoms.get_center_of_mass()

        # Position adsorbate above the surface center at specified height
        # For fcc(111) surface, center is at (0, 0, 0) in relative coordinates
        # Place adsorbate at height = adsorbate_height above the surface
        height_offset = np.array([0.0, 0.0, self.ads_config.adsorbate_height])

        # Center the adsorbate and move it to the correct height
        current_positions = self.adsorbate_atoms.get_positions()
        new_positions = current_positions - center + height_offset
        self.adsorbate_atoms.set_positions(new_positions)

        print(f"[AdsorptionChemGymEnv] Initialized {self.ads_config.adsorbate} adsorbate "
              f"at height {self.ads_config.adsorbate_height:.2f} Å")

    def _build_atoms_from_state(self) -> "Atoms":
        """
        Build Atoms object from state, including adsorbate.

        Returns:
            Atoms object with surface atoms + adsorbate
        """
        # Build surface atoms from state (inherited from ChemGymEnv)
        surface_atoms = super()._build_atoms_from_state()

        # Create combined structure: surface + adsorbate
        # Make a copy to avoid modifying the original
        combined_atoms = surface_atoms.copy()

        # Add adsorbate atoms to the structure
        for atom in self.adsorbate_atoms:
            combined_atoms.append(atom)

        # Set atomic tags:
        # 0 = bottom layers (fixed)
        # 1 = active surface layers (can be swapped)
        # 2 = adsorbate atoms (fixed during swaps, but relaxed by Oracle)
        n_surface_atoms = len(surface_atoms)
        tags = np.zeros(n_surface_atoms, dtype=int)

        # Tag active surface atoms (last n_active_atoms)
        tags[-self.n_active_atoms:] = 1

        # Tag adsorbate atoms
        ads_tags = np.full(len(self.adsorbate_atoms), 2, dtype=int)

        # Combine tags
        all_tags = np.concatenate([tags, ads_tags])
        combined_atoms.set_tags(all_tags)

        return combined_atoms

    def _evaluate_energy(self, atoms: Optional["Atoms"]) -> Tuple[float, float]:
        """
        Evaluate adsorption energy using Oracle with intelligent caching.

        Overrides the parent method to compute adsorption energy instead of
        formation energy. Uses caching to avoid redundant calculations.

        Args:
            atoms: Atoms object with surface + adsorbate

        Returns:
            Tuple of (adsorption_energy, uncertainty)
        """
        if atoms is None:
            return 0.0, 0.0

        # Check cache first (if enabled)
        if self.cache_manager is not None:
            cached_result = self.cache_manager.get_adsorption_energy(atoms)
            if cached_result is not None:
                return cached_result

        # Use Oracle for direct adsorption energy calculation
        if self.oracle is not None:
            try:
                # Calculate slab energy (surface without adsorbate)
                slab_atoms = atoms.copy()
                # Remove adsorbate atoms (last n_ads atoms)
                n_ads = len(self.adsorbate_atoms)
                slab_atoms = slab_atoms[:-n_ads].copy()

                # Check slab cache
                e_slab = None
                if self.cache_manager is not None:
                    e_slab = self.cache_manager.get_slab_energy(slab_atoms)

                if e_slab is None:
                    # Compute slab energy
                    e_slab = self.oracle.compute_energy(slab_atoms, relax=False)
                    # Cache slab energy
                    if self.cache_manager is not None:
                        self.cache_manager.put_slab_energy(slab_atoms, e_slab)

                # Get gas reference energy from cache
                gas_ref_energy = self.ads_config.gas_reference_energy
                if self.cache_manager is not None:
                    cached_ref = self.cache_manager.get_gas_reference_energy(self.ads_config.adsorbate)
                    if cached_ref is not None:
                        gas_ref_energy = cached_ref

                # Compute adsorption energy with automatic relaxation
                if self.ads_config.enable_relaxation:
                    # Oracle will relax the adsorbate position automatically
                    e_ads = self.oracle.predict_ads_energy(
                        atoms_with_ads=atoms,
                        slab_energy=e_slab,
                        gas_reference_energy=gas_ref_energy,
                        return_force=False
                    )
                else:
                    # No relaxation, just compute binding energy
                    e_total = self.oracle.compute_energy(atoms, relax=False)
                    e_ads = e_total - e_slab - gas_ref_energy

                # Cache the result
                if self.cache_manager is not None:
                    self.cache_manager.put_adsorption_energy(atoms, e_ads, 0.0)

                return e_ads, 0.0

            except Exception as e:
                print(f"[AdsorptionChemGymEnv] Oracle calculation failed: {e}")
                return 5.0, 0.0  # Return penalty value on failure

        # Fallback to EMT if Oracle not available
        if EMT is not None:
            try:
                calc_atoms = atoms.copy()
                calc_atoms.calc = EMT()

                # Calculate adsorption energy with EMT
                # This is a rough approximation
                e_total = calc_atoms.get_potential_energy()

                # Get slab energy (without adsorbate)
                slab_atoms = atoms.copy()
                n_ads = len(self.adsorbate_atoms)
                slab_atoms = slab_atoms[:-n_ads].copy()
                slab_atoms.calc = EMT()
                e_slab = slab_atoms.get_potential_energy()

                # Simple adsorption energy
                e_ads = e_total - e_slab - self.ads_config.gas_reference_energy

                return e_ads, 0.0

            except Exception as e:
                print(f"[AdsorptionChemGymEnv] EMT calculation failed: {e}")
                return 5.0, 0.0

        return 0.0, 0.0

    def get_cache_stats(self) -> Optional[Dict]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics or None if cache disabled
        """
        if self.cache_manager is not None:
            return self.cache_manager.get_cache_stats()
        return None

    def print_cache_stats(self):
        """Print cache statistics to console."""
        if self.cache_manager is not None:
            self.cache_manager.print_stats()
        else:
            print("Cache manager is disabled.")

    def clear_cache(self):
        """Clear all caches."""
        if self.cache_manager is not None:
            self.cache_manager.clear_all()
            print("[AdsorptionChemGymEnv] All caches cleared")

    def _calculate_adsorption_reward(self, current_energy: float) -> float:
        """
        Calculate reward based on target adsorption energy.

        Args:
            current_energy: Current adsorption energy (eV)

        Returns:
            Reward value
        """
        target = self.ads_config.target_ads_energy
        tolerance = self.ads_config.energy_tolerance

        # 1. Target achievement reward (Gaussian)
        distance = abs(current_energy - target)
        if distance <= tolerance:
            # Exact hit: Gaussian reward (max 100)
            target_reward = self.ads_config.target_reward_weight * \
                          np.exp(- (distance ** 2) / (2 * tolerance ** 2))
        else:
            # Close to target: linear reward
            target_reward = -distance * self.ads_config.proximity_reward_weight

        # 2. Trend reward (encourage continuous improvement)
        if current_energy < self.prev_energy:
            trend_reward = (self.prev_energy - current_energy) * 100
        else:
            trend_reward = -0.01  # Small penalty for energy increase

        # 3. Boundary check (prevent physically impossible values)
        if current_energy > 0:  # Physically impossible positive adsorption energy
            boundary_penalty = -50
        elif current_energy < -2.0:  # Too stable adsorption
            boundary_penalty = -10
        else:
            boundary_penalty = 0

        total_reward = target_reward + trend_reward + boundary_penalty

        return total_reward

    def step(self, action: int):
        """
        Execute action and compute adsorption energy reward.

        Overrides parent step method to use adsorption reward function.
        """
        # Use parent's step method for action execution
        i, j = self._action_to_indices(action)

        # Check if swapping identical elements (ineffective action)
        if self.state[i] == self.state[j]:
            reward = -0.5
            self.steps += 1
            truncated = self.steps >= self.config.max_steps

            info = {
                "energy": self.current_energy,
                "uncertainty": self.current_uncertainty,
                "swapped_sites": (i, j),
                "energy_improvement": 0.0,
                "adsorption_energy": self.current_energy,
                "atoms": self.atoms
            }
            return self._state_to_observation(), reward, False, truncated, info

        # Execute swap (in active region)
        self.state[i], self.state[j] = self.state[j], self.state[i]
        self.steps += 1

        # Update physical state
        self.atoms = self._build_atoms_from_state()
        self.current_energy, self.current_uncertainty = self._evaluate_energy(self.atoms)

        # Calculate reward using adsorption-specific function
        reward = self._calculate_adsorption_reward(self.current_energy)

        self.prev_energy = self.current_energy

        terminated = False
        truncated = self.steps >= self.config.max_steps

        # Termination condition: achieve target adsorption energy
        if abs(self.current_energy - self.ads_config.target_ads_energy) <= \
           self.ads_config.energy_tolerance:
            terminated = True

        # Physical anomaly detection
        if self.current_energy > 5.0:
            reward -= 10.0
            terminated = True

        observation = self._state_to_observation()
        info = {
            "energy": self.current_energy,
            "uncertainty": self.current_uncertainty,
            "swapped_sites": (i, j),
            "energy_improvement": self.initial_energy - self.current_energy,
            "adsorption_energy": self.current_energy,
            "atoms": self.atoms
        }
        return observation, reward, terminated, truncated, info

    def _state_to_observation(self):
        """
        Convert state to observation.

        For now, use parent's observation (surface only).
        Adsorbate is fixed, so no need to include in observation.
        """
        # Use parent's method - adsorbate is fixed
        return super()._state_to_observation()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset environment with adsorbate.

        Overrides parent reset to include adsorbate in initial state.
        """
        # Use parent's reset for surface initialization
        observation, info = super().reset(seed=seed, options=options)

        # The parent's reset already builds atoms with adsorbate via _build_atoms_from_state
        # Update info with adsorbate-related data
        info["adsorbate"] = self.ads_config.adsorbate
        info["target_adsorption_energy"] = self.ads_config.target_ads_energy
        info["initial_adsorption_energy"] = self.current_energy

        return observation, info
