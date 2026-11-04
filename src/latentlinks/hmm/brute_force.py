"""Exhaustive HMM inference: slow, obvious, and invaluable as an oracle."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor

from .model import CategoricalHMM


@dataclass(frozen=True)
class BruteForceResult:
    likelihood: Tensor
    gamma: Tensor
    xi: Tensor


def path_joint_probability(model: CategoricalHMM, observations: Tensor, path: Tensor) -> Tensor:
    """Compute ``P(x[0:T], z[0:T])`` one factor at a time."""
    probability = model.initial[path[0]] * model.emission[path[0], observations[0]]
    for time in range(1, observations.numel()):
        probability = probability * model.transition[path[time - 1], path[time]]
        probability = probability * model.emission[path[time], observations[time]]
    return probability


def brute_force_inference(model: CategoricalHMM, observations: Tensor) -> BruteForceResult:
    """Enumerate all ``K**T`` state paths and form exact posterior counts."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    if observations.ndim != 1 or observations.numel() == 0:
        raise ValueError("observations must be a non-empty vector")
    length = observations.numel()
    gamma_mass = torch.zeros(length, model.n_states, dtype=model.initial.dtype)
    xi_mass = torch.zeros(
        max(length - 1, 0), model.n_states, model.n_states, dtype=model.initial.dtype
    )
    likelihood = torch.zeros((), dtype=model.initial.dtype)

    for state_tuple in product(range(model.n_states), repeat=length):
        path = torch.tensor(state_tuple, dtype=torch.long)
        path_mass = path_joint_probability(model, observations, path)
        likelihood = likelihood + path_mass
        for time, state in enumerate(state_tuple):
            gamma_mass[time, state] += path_mass
        for time in range(length - 1):
            xi_mass[time, state_tuple[time], state_tuple[time + 1]] += path_mass

    return BruteForceResult(likelihood, gamma_mass / likelihood, xi_mass / likelihood)
