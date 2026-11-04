"""State-label alignment utilities for synthetic recovery experiments."""

from __future__ import annotations

from itertools import permutations

import torch
from torch import Tensor


def best_state_permutation(reference: Tensor, estimate: Tensor) -> tuple[int, ...]:
    """Find the small-state permutation minimizing squared parameter error.

    HMM state labels are not identifiable. This exhaustive helper is intentionally
    limited to tiny pedagogical examples, where its behavior is obvious.
    """
    if reference.shape != estimate.shape or reference.ndim < 1:
        raise ValueError("reference and estimate must have the same non-empty shape")
    n_states = reference.shape[0]
    best: tuple[int, ...] | None = None
    best_error = float("inf")
    for permutation in permutations(range(n_states)):
        candidate = estimate[list(permutation)]
        error = torch.sum((reference - candidate) ** 2).item()
        if error < best_error:
            best = permutation
            best_error = error
    assert best is not None
    return best
