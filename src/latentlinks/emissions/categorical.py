"""Categorical emission probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..utils.normalization import normalize


@dataclass(frozen=True)
class CategoricalEmission:
    """State-conditional categorical probabilities with shape ``(K, M)``."""

    probabilities: Tensor

    def __post_init__(self) -> None:
        probabilities = torch.as_tensor(self.probabilities)
        if probabilities.ndim != 2 or min(probabilities.shape) < 1:
            raise ValueError("categorical probabilities must have shape (states, symbols)")
        object.__setattr__(self, "probabilities", normalize(probabilities, dim=1))

    @property
    def n_states(self) -> int:
        return self.probabilities.shape[0]

    @property
    def n_symbols(self) -> int:
        return self.probabilities.shape[1]

    def probabilities_for(self, observations: Tensor) -> Tensor:
        """Return ``P(x[t] | z[t]=k)`` with shape ``(time, states)``."""
        observations = torch.as_tensor(observations, dtype=torch.long)
        if observations.ndim != 1 or observations.numel() == 0:
            raise ValueError("observations must be a non-empty vector")
        if (observations < 0).any() or (observations >= self.n_symbols).any():
            raise ValueError("observation index is outside the emission alphabet")
        return self.probabilities[:, observations].transpose(0, 1)

    def log_prob(self, observations: Tensor) -> Tensor:
        return torch.log(self.probabilities_for(observations))

    def sample(self, state: int, generator: torch.Generator | None = None) -> Tensor:
        return torch.multinomial(self.probabilities[state], 1, generator=generator).squeeze(0)
