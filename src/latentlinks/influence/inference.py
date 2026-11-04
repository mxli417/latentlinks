"""Exact product-state inference for small Influence Models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor

from .model import InfluenceModel


@dataclass(frozen=True)
class ExactInfluenceResult:
    log_likelihood: Tensor
    joint_states: Tensor
    joint_posterior: Tensor
    chain_marginals: Tensor


def enumerate_joint_states(model: InfluenceModel) -> Tensor:
    return torch.tensor(
        list(product(range(model.n_states), repeat=model.n_chains)), dtype=torch.long
    )


def compile_joint_initial(model: InfluenceModel, joint_states: Tensor) -> Tensor:
    probabilities = []
    for state_vector in joint_states:
        probability = torch.ones((), dtype=model.initial.dtype)
        for chain in range(model.n_chains):
            probability *= model.initial[chain, state_vector[chain]]
        probabilities.append(probability)
    return torch.stack(probabilities)


def compile_joint_transition(model: InfluenceModel, joint_states: Tensor) -> Tensor:
    """Compile the structured transition into a ``K**C`` square matrix."""
    rows = []
    for previous in joint_states:
        row = []
        for current in joint_states:
            probability = torch.ones((), dtype=model.initial.dtype)
            for target in range(model.n_chains):
                target_probability = torch.zeros((), dtype=model.initial.dtype)
                for source in range(model.n_chains):
                    target_probability += (
                        model.influence[target, source]
                        * model.transition[target, source, previous[source], current[target]]
                    )
                probability *= target_probability
            row.append(probability)
        rows.append(torch.stack(row))
    return torch.stack(rows)


def compile_joint_emission(
    model: InfluenceModel, observations: Tensor, joint_states: Tensor
) -> Tensor:
    rows = []
    for time in range(observations.shape[0]):
        state_likelihoods = []
        for state_vector in joint_states:
            likelihood = torch.ones((), dtype=model.initial.dtype)
            for chain in range(model.n_chains):
                likelihood *= model.emission[chain, state_vector[chain], observations[time, chain]]
            state_likelihoods.append(likelihood)
        rows.append(torch.stack(state_likelihoods))
    return torch.stack(rows)


def exact_inference(model: InfluenceModel, observations: Tensor) -> ExactInfluenceResult:
    """Run exact log-space forward-backward on the joint state space."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    joint_states = enumerate_joint_states(model)
    initial = compile_joint_initial(model, joint_states)
    transition = compile_joint_transition(model, joint_states)
    emission = compile_joint_emission(model, observations, joint_states)
    log_transition = torch.log(transition)
    log_emission = torch.log(emission)

    alpha_rows = [torch.log(initial) + log_emission[0]]
    for time in range(1, observations.shape[0]):
        next_row = []
        for current in range(joint_states.shape[0]):
            incoming = alpha_rows[-1] + log_transition[:, current]
            next_row.append(torch.logsumexp(incoming, dim=0) + log_emission[time, current])
        alpha_rows.append(torch.stack(next_row))
    log_alpha = torch.stack(alpha_rows)
    log_likelihood = torch.logsumexp(log_alpha[-1], dim=0)

    reverse_beta = [torch.zeros(joint_states.shape[0], dtype=model.initial.dtype)]
    for time in range(observations.shape[0] - 2, -1, -1):
        next_beta = reverse_beta[-1]
        row = []
        for previous in range(joint_states.shape[0]):
            terms = log_transition[previous] + log_emission[time + 1] + next_beta
            row.append(torch.logsumexp(terms, dim=0))
        reverse_beta.append(torch.stack(row))
    log_beta = torch.stack(list(reversed(reverse_beta)))
    joint_posterior = torch.exp(log_alpha + log_beta - log_likelihood)

    marginals = torch.zeros(
        model.n_chains,
        observations.shape[0],
        model.n_states,
        dtype=model.initial.dtype,
    )
    for chain in range(model.n_chains):
        for state in range(model.n_states):
            matching = joint_states[:, chain] == state
            marginals[chain, :, state] = joint_posterior[:, matching].sum(dim=1)
    return ExactInfluenceResult(log_likelihood, joint_states, joint_posterior, marginals)
