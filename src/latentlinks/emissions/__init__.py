"""Observation models used by latent-state processes."""

from .categorical import CategoricalEmission
from .gaussian import DiagonalGaussianEmission

__all__ = ["CategoricalEmission", "DiagonalGaussianEmission"]
