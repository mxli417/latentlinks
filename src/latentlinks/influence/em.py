"""Expected-count learning paired with Dong's linear inference algorithm."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from ..utils.normalization import normalize
from .dong import dong_linear_inference
from .model import InfluenceModel


@dataclass(frozen=True)
class InfluenceEMHistory:
    log_likelihoods: tuple[float, ...]
    converged: bool


def dong_linear_em_step(
    model: InfluenceModel,
    sequences: Iterable[Tensor],
    pseudocount: float = 1e-6,
) -> tuple[InfluenceModel, Tensor]:
    """Perform the M-step associated with Dong's linear sufficient statistics."""
    data = [torch.as_tensor(sequence, dtype=torch.long) for sequence in sequences]
    if not data:
        raise ValueError("at least one sequence is required")
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")

    initial_counts = torch.full_like(model.initial, pseudocount)
    transition_counts = torch.full_like(model.transition, pseudocount)
    emission_counts = torch.full_like(model.emission, pseudocount)
    total_log_likelihood = torch.zeros((), dtype=model.initial.dtype)

    for observations in data:
        posterior = dong_linear_inference(model, observations)
        total_log_likelihood += posterior.log_likelihood
        initial_counts += posterior.marginals[:, 0]
        transition_counts += posterior.expected_pair_transitions
        for chain in range(model.n_chains):
            for symbol in range(model.n_symbols):
                selected = observations[:, chain] == symbol
                emission_counts[chain, :, symbol] += posterior.marginals[chain, selected].sum(dim=0)

    # Every source-target block is a conditional transition table.
    transition = normalize(transition_counts, dim=3)
    # Summing a block assigns responsibility to that source; normalize over
    # sources separately for every target.
    influence_counts = transition_counts.sum(dim=(2, 3))
    influence = normalize(influence_counts, dim=1)
    updated = InfluenceModel(
        normalize(initial_counts, dim=1),
        transition,
        normalize(emission_counts, dim=2),
        influence,
    )
    return updated, total_log_likelihood


def fit_dong_linear_em(
    model: InfluenceModel,
    sequences: Iterable[Tensor],
    max_iterations: int = 100,
    tolerance: float = 1e-7,
    pseudocount: float = 1e-6,
) -> tuple[InfluenceModel, InfluenceEMHistory]:
    """Fit an Influence Model with the linear E-step and its count M-step."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    data = list(sequences)
    history: list[float] = []
    converged = False
    current = model
    for _ in range(max_iterations):
        current, objective = dong_linear_em_step(current, data, pseudocount)
        history.append(float(objective))
        if len(history) > 1 and abs(history[-1] - history[-2]) <= tolerance * (
            1 + abs(history[-2])
        ):
            converged = True
            break
    return current, InfluenceEMHistory(tuple(history), converged)
