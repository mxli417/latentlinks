"""Translate influence coefficients into conventional graph orientation."""

from __future__ import annotations

import torch
from torch import Tensor


def influence_to_adjacency(influence: Tensor, include_self: bool = False) -> Tensor:
    """Return ``adjacency[source,target] = influence[target,source]``.

    Values express predictive dynamical coupling, not automatically causality.
    """
    influence = torch.as_tensor(influence)
    if influence.ndim != 2 or influence.shape[0] != influence.shape[1]:
        raise ValueError("influence must be a square (target, source) matrix")
    adjacency = influence.transpose(0, 1).clone()
    if not include_self:
        adjacency.fill_diagonal_(0)
    return adjacency
