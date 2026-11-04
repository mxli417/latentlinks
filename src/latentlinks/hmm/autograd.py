"""Gradient estimation after the classical inference path is understood."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .forward_backward import forward_log_likelihood
from .model import CategoricalHMM


class LogitHMM(nn.Module):
    """An unconstrained parameterization mapped to probabilities by softmax."""

    def __init__(self, model: CategoricalHMM) -> None:
        super().__init__()
        self.initial_logits = nn.Parameter(torch.log(model.initial))
        self.transition_logits = nn.Parameter(torch.log(model.transition))
        self.emission_logits = nn.Parameter(torch.log(model.emission))

    def probability_model(self) -> CategoricalHMM:
        return CategoricalHMM(
            torch.softmax(self.initial_logits, dim=-1),
            torch.softmax(self.transition_logits, dim=-1),
            torch.softmax(self.emission_logits, dim=-1),
        )

    def log_likelihood(self, observations: Tensor) -> Tensor:
        return forward_log_likelihood(self.probability_model(), observations)


@dataclass(frozen=True)
class GradientHistory:
    log_likelihoods: tuple[float, ...]


def fit_gradient(
    model: CategoricalHMM,
    sequences: Iterable[Tensor],
    iterations: int = 200,
    learning_rate: float = 0.03,
) -> tuple[CategoricalHMM, GradientHistory]:
    """Optimize observed-data likelihood; autograd does not replace inference."""
    data = [torch.as_tensor(sequence, dtype=torch.long) for sequence in sequences]
    parameterization = LogitHMM(model)
    optimizer = torch.optim.Adam(parameterization.parameters(), lr=learning_rate)
    history = []
    for _ in range(iterations):
        optimizer.zero_grad()
        objective = torch.zeros((), dtype=model.initial.dtype)
        for sequence in data:
            objective = objective + parameterization.log_likelihood(sequence)
        (-objective).backward()
        optimizer.step()
        history.append(float(objective.detach()))
    return parameterization.probability_model(), GradientHistory(tuple(history))
