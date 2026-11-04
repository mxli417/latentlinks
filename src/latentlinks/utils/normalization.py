"""Readable probability normalization helpers for PyTorch tensors."""

from __future__ import annotations

import torch
from torch import Tensor


def normalize(values: Tensor, dim: int = -1) -> Tensor:
    """Normalize non-negative values along ``dim``.

    Keeping this operation named makes the probability-simplex constraint
    visible at call sites instead of hiding it inside clever tensor expressions.
    """
    values = torch.as_tensor(values)
    if not values.is_floating_point():
        values = values.to(torch.get_default_dtype())
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    total = values.sum(dim=dim, keepdim=True)
    if (total <= 0).any():
        raise ValueError("each probability vector must have positive mass")
    return values / total


def assert_stochastic(values: Tensor, dim: int = -1, atol: float = 1e-7) -> None:
    """Raise ``ValueError`` unless ``values`` is stochastic along ``dim``."""
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    expected = torch.ones_like(values.sum(dim=dim))
    if not torch.allclose(values.sum(dim=dim), expected, atol=atol, rtol=0):
        raise ValueError("probability vectors must sum to one")
