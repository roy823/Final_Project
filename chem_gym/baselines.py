import math
from typing import Dict, Tuple

import numpy as np

from chem_gym.envs.chem_env import ChemGymEnv


def random_search(env: ChemGymEnv, episodes: int, horizon: int) -> Dict:
    best_energy = math.inf
    best_state = None
    for _ in range(episodes):
        _, info = env.reset()
        best_energy = min(best_energy, info["energy"])
        for _ in range(horizon):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if info["energy"] < best_energy:
                best_energy = info["energy"]
                best_state = np.array(env.state)
            if terminated or truncated:
                break
    return {"best_energy": best_energy, "best_state": best_state}


def simulated_annealing(env: ChemGymEnv, steps: int, t_start: float = 1.0, t_end: float = 0.01) -> Dict:
    _, info = env.reset()
    current_energy = info["energy"]
    best_energy = current_energy
    best_state = np.array(env.state)

    for step in range(steps):
        temp = max(t_end, t_start * ((t_end / t_start) ** (step / max(steps - 1, 1))))
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        delta_e = info["energy"] - current_energy
        accept = delta_e < 0 or math.exp(-delta_e / max(temp, 1e-6)) > np.random.rand()
        if accept:
            current_energy = info["energy"]
            if current_energy < best_energy:
                best_energy = current_energy
                best_state = np.array(env.state)
        if terminated or truncated:
            break
    return {"best_energy": best_energy, "best_state": best_state}
