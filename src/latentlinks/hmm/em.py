"""Baum-Welch written as literal expected-count updates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

from ..utils.normalization import normalize
from .forward_backward import forward_backward
from .model import CategoricalHMM


@dataclass(frozen=True)
class EMHistory:
    log_likelihoods: tuple[float, ...]
    converged: bool


def baum_welch_step(
    model: CategoricalHMM, sequences: Iterable[Tensor], pseudocount: float = 0.0
) -> tuple[CategoricalHMM, Tensor]:
    """Perform one exact EM update using posterior expected counts."""
    data = [torch.as_tensor(sequence, dtype=torch.long) for sequence in sequences]
    if not data:
        raise ValueError("at least one sequence is required")
    initial_counts = torch.full_like(model.initial, pseudocount)
    expected_transitions = torch.full_like(model.transition, pseudocount)
    expected_emissions = torch.full_like(model.emission, pseudocount)
    total_log_likelihood = torch.zeros((), dtype=model.initial.dtype)

    for observations in data:
        posterior = forward_backward(model, observations)
        total_log_likelihood += posterior.log_likelihood
        initial_counts += posterior.gamma[0]

        # N_ij = E[#(i -> j) | X, theta] = sum_t xi[t, i, j].
        expected_transitions += posterior.xi.sum(dim=0)

        # Expected emissions count how much posterior state mass generated symbol m.
        for symbol in range(model.n_symbols):
            times_with_symbol = observations == symbol
            expected_emissions[:, symbol] += posterior.gamma[times_with_symbol].sum(dim=0)

    updated = CategoricalHMM(
        normalize(initial_counts),
        normalize(expected_transitions, dim=1),
        normalize(expected_emissions, dim=1),
    )
    return updated, total_log_likelihood


def fit_baum_welch(
    model: CategoricalHMM,
    sequences: Iterable[Tensor],
    max_iterations: int = 100,
    tolerance: float = 1e-7,
    pseudocount: float = 1e-6,
) -> tuple[CategoricalHMM, EMHistory]:
    data = list(sequences)
    history: list[float] = []
    converged = False
    current = model
    for _ in range(max_iterations):
        current, objective = baum_welch_step(current, data, pseudocount)
        history.append(float(objective))
        if len(history) > 1 and abs(history[-1] - history[-2]) <= tolerance * (
            1 + abs(history[-2])
        ):
            converged = True
            break
    return current, EMHistory(tuple(history), converged)
