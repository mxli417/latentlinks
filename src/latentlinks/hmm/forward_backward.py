"""Reference forward-backward algorithms with inspectable intermediate tensors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import CategoricalHMM


@dataclass(frozen=True)
class ForwardBackwardResult:
    log_likelihood: Tensor
    log_alpha: Tensor
    log_beta: Tensor
    gamma: Tensor
    xi: Tensor


def forward_probabilities(model: CategoricalHMM, observations: Tensor) -> Tensor:
    """Compute ``alpha[t,j] = P(x[0:t], z[t]=j)`` with explicit loops."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    emission = model.emission_distribution.probabilities_for(observations)
    rows = [model.initial * emission[0]]
    for time in range(1, observations.numel()):
        next_row = []
        for next_state in range(model.n_states):
            incoming_mass = torch.sum(rows[-1] * model.transition[:, next_state])
            next_row.append(incoming_mass * emission[time, next_state])
        rows.append(torch.stack(next_row))
    return torch.stack(rows)


def backward_probabilities(model: CategoricalHMM, observations: Tensor) -> Tensor:
    """Compute ``beta[t,i] = P(x[t+1:T] | z[t]=i)`` with explicit loops."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    emission = model.emission_distribution.probabilities_for(observations)
    reverse_rows = [torch.ones(model.n_states, dtype=model.initial.dtype)]
    for time in range(observations.numel() - 2, -1, -1):
        next_beta = reverse_rows[-1]
        row = []
        for state in range(model.n_states):
            terms = model.transition[state] * emission[time + 1] * next_beta
            row.append(torch.sum(terms))
        reverse_rows.append(torch.stack(row))
    return torch.stack(list(reversed(reverse_rows)))


def forward_log_probabilities(model: CategoricalHMM, observations: Tensor) -> Tensor:
    """Stable differentiable forward recursion using ``torch.logsumexp``."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    log_initial = torch.log(model.initial)
    log_transition = torch.log(model.transition)
    log_emission = model.emission_distribution.log_prob(observations)
    rows = [log_initial + log_emission[0]]
    for time in range(1, observations.numel()):
        next_row = []
        for next_state in range(model.n_states):
            incoming = rows[-1] + log_transition[:, next_state]
            next_row.append(torch.logsumexp(incoming, dim=0) + log_emission[time, next_state])
        rows.append(torch.stack(next_row))
    return torch.stack(rows)


def forward_log_likelihood(model: CategoricalHMM, observations: Tensor) -> Tensor:
    return torch.logsumexp(forward_log_probabilities(model, observations)[-1], dim=0)


def forward_log_probabilities_vectorized(model: CategoricalHMM, observations: Tensor) -> Tensor:
    """Tensorized-state counterpart to the loop-based teaching recursion."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    log_transition = torch.log(model.transition)
    log_emission = model.emission_distribution.log_prob(observations)
    rows = [torch.log(model.initial) + log_emission[0]]
    for time in range(1, observations.numel()):
        incoming = rows[-1][:, None] + log_transition
        rows.append(torch.logsumexp(incoming, dim=0) + log_emission[time])
    return torch.stack(rows)


def forward_backward_vectorized(
    model: CategoricalHMM, observations: Tensor
) -> ForwardBackwardResult:
    """Vectorize state and state-pair axes while retaining the time recursion."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    log_alpha = forward_log_probabilities_vectorized(model, observations)
    log_transition = torch.log(model.transition)
    log_emission = model.emission_distribution.log_prob(observations)
    log_likelihood = torch.logsumexp(log_alpha[-1], dim=0)

    reverse_rows = [torch.zeros(model.n_states, dtype=model.initial.dtype)]
    for time in range(observations.numel() - 2, -1, -1):
        terms = log_transition + log_emission[time + 1, None, :] + reverse_rows[-1][None, :]
        reverse_rows.append(torch.logsumexp(terms, dim=1))
    log_beta = torch.stack(list(reversed(reverse_rows)))
    gamma = torch.exp(log_alpha + log_beta - log_likelihood)

    if observations.numel() > 1:
        log_xi = (
            log_alpha[:-1, :, None]
            + log_transition[None, :, :]
            + log_emission[1:, None, :]
            + log_beta[1:, None, :]
            - log_likelihood
        )
        xi = torch.exp(log_xi)
    else:
        xi = torch.empty(0, model.n_states, model.n_states, dtype=model.initial.dtype)
    return ForwardBackwardResult(log_likelihood, log_alpha, log_beta, gamma, xi)


def forward_backward(model: CategoricalHMM, observations: Tensor) -> ForwardBackwardResult:
    """Compute stable ``alpha``, ``beta``, state posterior ``gamma``, and ``xi``."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    log_alpha = forward_log_probabilities(model, observations)
    log_transition = torch.log(model.transition)
    log_emission = model.emission_distribution.log_prob(observations)
    log_likelihood = torch.logsumexp(log_alpha[-1], dim=0)

    reverse_rows = [torch.zeros(model.n_states, dtype=model.initial.dtype)]
    for time in range(observations.numel() - 2, -1, -1):
        next_beta = reverse_rows[-1]
        row = []
        for state in range(model.n_states):
            terms = log_transition[state] + log_emission[time + 1] + next_beta
            row.append(torch.logsumexp(terms, dim=0))
        reverse_rows.append(torch.stack(row))
    log_beta = torch.stack(list(reversed(reverse_rows)))

    gamma = torch.exp(log_alpha + log_beta - log_likelihood)
    xi_rows = []
    for time in range(observations.numel() - 1):
        log_xi = (
            log_alpha[time, :, None]
            + log_transition
            + log_emission[time + 1, None, :]
            + log_beta[time + 1, None, :]
            - log_likelihood
        )
        xi_rows.append(torch.exp(log_xi))
    xi = torch.stack(xi_rows) if xi_rows else torch.empty(0, model.n_states, model.n_states)
    return ForwardBackwardResult(log_likelihood, log_alpha, log_beta, gamma, xi)
