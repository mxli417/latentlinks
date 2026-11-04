"""A transparent factorized marginal approximation for larger systems.

This is an independently derived teaching approximation. It is not labeled as
the nonlinear inference algorithm from Dong's thesis; see ``docs/literature.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..utils.normalization import normalize
from .model import InfluenceModel


@dataclass(frozen=True)
class FactorizedInferenceResult:
    marginals: Tensor
    max_changes: tuple[float, ...]
    converged: bool


def effective_transitions(model: InfluenceModel, marginals: Tensor, target: int) -> Tensor:
    """Construct ``(time-1, own_previous_state, own_next_state)`` dynamics."""
    length = marginals.shape[1]
    transitions = []
    for time in range(length - 1):
        matrix = model.influence[target, target] * model.transition[target, target].clone()
        for source in range(model.n_chains):
            if source == target:
                continue
            expected_next = torch.zeros(model.n_states, dtype=model.initial.dtype)
            for source_state in range(model.n_states):
                expected_next += (
                    marginals[source, time, source_state]
                    * model.transition[target, source, source_state]
                )
            matrix += model.influence[target, source] * expected_next[None, :]
        transitions.append(matrix)
    return torch.stack(transitions)


def _inhomogeneous_marginals(
    initial: Tensor, transitions: Tensor, emission_likelihood: Tensor
) -> Tensor:
    log_alpha_rows = [torch.log(initial) + torch.log(emission_likelihood[0])]
    for time in range(1, emission_likelihood.shape[0]):
        row = []
        for next_state in range(initial.numel()):
            incoming = log_alpha_rows[-1] + torch.log(transitions[time - 1, :, next_state])
            row.append(
                torch.logsumexp(incoming, dim=0) + torch.log(emission_likelihood[time, next_state])
            )
        log_alpha_rows.append(torch.stack(row))
    log_alpha = torch.stack(log_alpha_rows)
    reverse_beta = [torch.zeros_like(initial)]
    for time in range(emission_likelihood.shape[0] - 2, -1, -1):
        row = []
        for state in range(initial.numel()):
            terms = (
                torch.log(transitions[time, state])
                + torch.log(emission_likelihood[time + 1])
                + reverse_beta[-1]
            )
            row.append(torch.logsumexp(terms, dim=0))
        reverse_beta.append(torch.stack(row))
    log_beta = torch.stack(list(reversed(reverse_beta)))
    return torch.softmax(log_alpha + log_beta, dim=1)


def factorized_inference(
    model: InfluenceModel,
    observations: Tensor,
    max_iterations: int = 100,
    tolerance: float = 1e-7,
) -> FactorizedInferenceResult:
    """Coordinate-update chain marginals while holding other chains fixed."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    length = observations.shape[0]
    marginals = torch.empty(model.n_chains, length, model.n_states, dtype=model.initial.dtype)
    for chain in range(model.n_chains):
        local_evidence = model.emission[chain, :, observations[:, chain]].transpose(0, 1)
        marginals[chain] = normalize(local_evidence, dim=1)
        marginals[chain, 0] = normalize(marginals[chain, 0] * model.initial[chain])

    max_changes = []
    converged = False
    for _ in range(max_iterations):
        previous = marginals.clone()
        for target in range(model.n_chains):
            dynamics = effective_transitions(model, marginals, target)
            evidence = model.emission[target, :, observations[:, target]].transpose(0, 1)
            marginals[target] = _inhomogeneous_marginals(model.initial[target], dynamics, evidence)
        change = torch.max(torch.abs(marginals - previous)).item()
        max_changes.append(change)
        if change <= tolerance:
            converged = True
            break
    return FactorizedInferenceResult(marginals, tuple(max_changes), converged)
