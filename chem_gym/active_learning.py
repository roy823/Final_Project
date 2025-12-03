"""
Minimal active learning utilities to plug into the surrogate ensemble.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    from ase import Atoms
except ImportError:  # pragma: no cover
    Atoms = None

from chem_gym.surrogate.ensemble import SurrogateEnsemble


@dataclass
class ReplayBuffer:
    entries: List[Tuple[Optional["Atoms"], float]] = field(default_factory=list)

    def push(self, atoms: Optional["Atoms"], energy: float):
        self.entries.append((atoms, energy))

    def sample(self, k: int):
        k = min(k, len(self.entries))
        idx = np.random.choice(len(self.entries), size=k, replace=False)
        return [self.entries[i] for i in idx]


def should_query_oracle(uncertainty: float, threshold: float) -> bool:
    return uncertainty > threshold


def update_surrogate_from_buffer(surrogate: SurrogateEnsemble, buffer: ReplayBuffer):
    for atoms, energy in buffer.entries:
        surrogate.update_with_oracle(atoms, energy)
