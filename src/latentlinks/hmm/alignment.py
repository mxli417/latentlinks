"""State-label alignment for HMM recovery experiments."""

from __future__ import annotations

from itertools import permutations

import torch

from .model import CategoricalHMM


def best_hmm_state_permutation(
    reference: CategoricalHMM, estimate: CategoricalHMM
) -> tuple[int, ...]:
    """Minimize total parameter error over tiny-model label permutations."""
    if (reference.n_states, reference.n_symbols) != (
        estimate.n_states,
        estimate.n_symbols,
    ):
        raise ValueError("models must have matching state and symbol counts")
    best: tuple[int, ...] | None = None
    best_error = float("inf")
    for order in permutations(range(reference.n_states)):
        indices = list(order)
        error = (
            torch.sum((reference.initial - estimate.initial[indices]) ** 2)
            + torch.sum((reference.transition - estimate.transition[indices][:, indices]) ** 2)
            + torch.sum((reference.emission - estimate.emission[indices]) ** 2)
        ).item()
        if error < best_error:
            best, best_error = order, error
    assert best is not None
    return best


def align_hmm_states(reference: CategoricalHMM, estimate: CategoricalHMM) -> CategoricalHMM:
    """Return ``estimate`` with its latent labels aligned to ``reference``."""
    order = list(best_hmm_state_permutation(reference, estimate))
    return CategoricalHMM(
        estimate.initial[order],
        estimate.transition[order][:, order],
        estimate.emission[order],
    )
