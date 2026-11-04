"""Diagonal Gaussian emissions, prepared for continuous-model tutorials."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import torch
from torch import Tensor


@dataclass(frozen=True)
class DiagonalGaussianEmission:
    """State-conditional diagonal Gaussians with parameters ``(K, features)``."""

    mean: Tensor
    variance: Tensor

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean)
        variance = torch.as_tensor(self.variance, dtype=mean.dtype, device=mean.device)
        if mean.ndim != 2 or variance.shape != mean.shape:
            raise ValueError("mean and variance must share shape (states, features)")
        if (variance <= 0).any():
            raise ValueError("Gaussian variances must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)

    def log_prob(self, observations: Tensor) -> Tensor:
        """Return state log densities with shape ``(time, states)``."""
        observations = torch.as_tensor(observations, dtype=self.mean.dtype, device=self.mean.device)
        if observations.ndim != 2 or observations.shape[1] != self.mean.shape[1]:
            raise ValueError("observations must have shape (time, features)")
        difference = observations[:, None, :] - self.mean[None, :, :]
        terms = difference.square() / self.variance[None, :, :]
        normalizer = torch.log(self.variance) + log(2 * pi)
        return -0.5 * (terms + normalizer[None, :, :]).sum(dim=-1)
