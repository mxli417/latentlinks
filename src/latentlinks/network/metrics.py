"""Simple network-recovery metrics for synthetic experiments."""

from __future__ import annotations

import torch
from torch import Tensor


def edge_mean_squared_error(reference: Tensor, estimate: Tensor) -> Tensor:
    reference = torch.as_tensor(reference)
    estimate = torch.as_tensor(estimate, dtype=reference.dtype)
    if reference.shape != estimate.shape:
        raise ValueError("network matrices must have the same shape")
    return torch.mean((reference - estimate) ** 2)
