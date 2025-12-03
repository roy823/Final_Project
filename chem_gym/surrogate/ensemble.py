import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from ase import Atoms
except ImportError:  # pragma: no cover - allows scaffold without ASE
    Atoms = None

from chem_gym.config import SurrogateConfig

Prediction = Tuple[float, float]  # mean energy, std uncertainty


def default_hash_fn(atoms: Optional["Atoms"]) -> Optional[str]:
    """
    Hash an ASE Atoms object by chemical symbols and positions.
    """
    if atoms is None or Atoms is None:
        return None
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    payload = ",".join(symbols) + "|" + np.array2string(positions, precision=3)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


@dataclass
class SurrogateEnsemble:
    """
    Lightweight ensemble wrapper to mimic OCP-style inference + uncertainty.
    """

    config: SurrogateConfig = field(default_factory=SurrogateConfig)
    models: List[Callable[[Optional["Atoms"]], float]] = field(default_factory=list)
    hash_fn: Callable[[Optional["Atoms"]], Optional[str]] = default_hash_fn

    def __post_init__(self):
        if not self.models:
            seeds = self.config.seeds or list(range(self.config.n_models))
            self.models = [self._random_model(seed) for seed in seeds[: self.config.n_models]]
        self.cache: Dict[str, Prediction] = {}

    def evaluate(self, atoms: Optional["Atoms"]) -> Prediction:
        key = self.hash_fn(atoms)
        if key and key in self.cache:
            return self.cache[key]

        preds = np.array([model(atoms) for model in self.models], dtype=np.float32)
        mean_energy = float(preds.mean())
        std_energy = float(preds.std())

        if key:
            self.cache[key] = (mean_energy, std_energy)
        return mean_energy, std_energy

    def update_with_oracle(self, atoms: Optional["Atoms"], oracle_energy: float):
        """
        Insert higher-fidelity label into the cache to emulate active learning feedback.
        """
        key = self.hash_fn(atoms)
        if key:
            self.cache[key] = (oracle_energy, 0.0)

    def _random_model(self, seed: int) -> Callable[[Optional["Atoms"]], float]:
        rng = np.random.default_rng(seed)

        def _fn(_: Optional["Atoms"]) -> float:
            return float(rng.normal(loc=self.config.mean_energy, scale=self.config.noise_scale))

        return _fn
