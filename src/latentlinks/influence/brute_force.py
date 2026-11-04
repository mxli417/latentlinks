"""Exhaustive trajectory enumeration for tiny Influence Models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor

from .model import InfluenceModel


@dataclass(frozen=True)
class InfluenceBruteForceResult:
    likelihood: Tensor
    chain_marginals: Tensor


def trajectory_joint_probability(
    model: InfluenceModel, observations: Tensor, states: Tensor
) -> Tensor:
    """Compute the joint mass of one ``(time, chains)`` state trajectory."""
    probability = torch.ones((), dtype=model.initial.dtype)
    for chain in range(model.n_chains):
        probability *= model.initial[chain, states[0, chain]]
        probability *= model.emission[chain, states[0, chain], observations[0, chain]]
    for time in range(1, observations.shape[0]):
        for target in range(model.n_chains):
            transition_probability = torch.zeros((), dtype=model.initial.dtype)
            for source in range(model.n_chains):
                transition_probability += (
                    model.influence[target, source]
                    * model.transition[
                        target,
                        source,
                        states[time - 1, source],
                        states[time, target],
                    ]
                )
            probability *= transition_probability
            probability *= model.emission[target, states[time, target], observations[time, target]]
    return probability


def brute_force_inference(model: InfluenceModel, observations: Tensor) -> InfluenceBruteForceResult:
    """Enumerate ``K**(C*T)`` trajectories; use only as a tiny oracle."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    if observations.ndim != 2 or observations.shape[1] != model.n_chains:
        raise ValueError("observations must have shape (time, chains)")
    length = observations.shape[0]
    marginal_mass = torch.zeros(model.n_chains, length, model.n_states, dtype=model.initial.dtype)
    likelihood = torch.zeros((), dtype=model.initial.dtype)
    for flat_states in product(range(model.n_states), repeat=length * model.n_chains):
        states = torch.tensor(flat_states, dtype=torch.long).reshape(length, model.n_chains)
        mass = trajectory_joint_probability(model, observations, states)
        likelihood += mass
        for time in range(length):
            for chain in range(model.n_chains):
                marginal_mass[chain, time, states[time, chain]] += mass
    return InfluenceBruteForceResult(likelihood, marginal_mass / likelihood)
