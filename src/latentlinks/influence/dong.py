"""Dong's linear marginal-state inference for the Influence Model.

This module implements the linear recursion described in Wen Dong's thesis and
exercised by the public MIT Influence Model toolbox.  It propagates ``C * K``
marginal state probabilities rather than a ``K**C`` Cartesian-product state.
Consequently it is a structured approximation to the full joint model, not the
exact product-state algorithm in :mod:`latentlinks.influence.inference`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import InfluenceModel


@dataclass(frozen=True)
class DongLinearResult:
    """Inspectable messages and sufficient statistics from linear inference."""

    log_likelihood: Tensor
    alpha: Tensor
    beta: Tensor
    marginals: Tensor
    expected_pair_transitions: Tensor
    influence_matrix: Tensor
    scales: Tensor


def compile_influence_matrix(model: InfluenceModel) -> Tensor:
    """Return the block matrix mapping source marginals to target marginals.

    Rows index ``(source, source_state)`` and columns index
    ``(target, target_state)``.  This is the transpose convention required by
    row-vector forward messages; the public model API remains
    ``influence[target, source]``.
    """
    chains, states = model.n_chains, model.n_states
    matrix = torch.empty(chains * states, chains * states, dtype=model.initial.dtype)
    for source in range(chains):
        source_slice = slice(source * states, (source + 1) * states)
        for target in range(chains):
            target_slice = slice(target * states, (target + 1) * states)
            matrix[source_slice, target_slice] = (
                model.influence[target, source] * model.transition[target, source]
            )
    return matrix


def _emission_likelihoods(model: InfluenceModel, observations: Tensor) -> Tensor:
    rows = []
    for time in range(observations.shape[0]):
        rows.append(
            torch.cat(
                [
                    model.emission[chain, :, observations[time, chain]]
                    for chain in range(model.n_chains)
                ]
            )
        )
    return torch.stack(rows)


def dong_linear_inference(
    model: InfluenceModel, observations: Tensor, *, epsilon: float = 1e-30
) -> DongLinearResult:
    """Run Dong's linear forward-backward approximation.

    The implementation follows the equations literally.  Each chain block is
    normalized separately; ``scales[t, c]`` stores that block's normalizer.
    ``expected_pair_transitions[target, source, r, s]`` is the expected count
    assigned to source state ``r`` influencing target state ``s``.
    """
    observations = torch.as_tensor(observations, dtype=torch.long)
    if observations.ndim != 2 or observations.shape[1] != model.n_chains:
        raise ValueError("observations must have shape (time, chains)")
    if observations.shape[0] < 1:
        raise ValueError("at least one observation is required")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    chains, states = model.n_chains, model.n_states
    influence_matrix = compile_influence_matrix(model)
    emission = _emission_likelihoods(model, observations).clamp_min(epsilon)
    prior = model.initial.reshape(-1)

    alpha_rows: list[Tensor] = []
    scale_rows: list[Tensor] = []
    unnormalized = prior * emission[0]
    blocks = unnormalized.reshape(chains, states)
    chain_scales = blocks.sum(dim=1)
    alpha_rows.append((blocks / chain_scales[:, None]).reshape(-1))
    scale_rows.append(chain_scales)

    for time in range(1, observations.shape[0]):
        unnormalized = (alpha_rows[-1] @ influence_matrix) * emission[time]
        blocks = unnormalized.reshape(chains, states)
        chain_scales = blocks.sum(dim=1)
        alpha_rows.append((blocks / chain_scales[:, None]).reshape(-1))
        scale_rows.append(chain_scales)

    alpha_flat = torch.stack(alpha_rows)
    scales = torch.stack(scale_rows)
    repeated_scales = scales.repeat_interleave(states, dim=1)
    log_likelihood = torch.log(scales).sum()

    beta_reverse = [torch.ones(chains * states, dtype=model.initial.dtype)]
    for time in range(observations.shape[0] - 2, -1, -1):
        coefficient = emission[time + 1] * beta_reverse[-1] / repeated_scales[time + 1]
        beta_reverse.append(influence_matrix @ coefficient)
    beta_flat = torch.stack(list(reversed(beta_reverse)))
    gamma_flat = alpha_flat * beta_flat

    pair_counts_flat = torch.zeros_like(influence_matrix)
    for time in range(observations.shape[0] - 1):
        coefficient = emission[time + 1] * beta_flat[time + 1] / repeated_scales[time + 1]
        pair_counts_flat += alpha_flat[time, :, None] * influence_matrix * coefficient[None, :]

    pair_counts = torch.empty(chains, chains, states, states, dtype=model.initial.dtype)
    for target in range(chains):
        target_slice = slice(target * states, (target + 1) * states)
        for source in range(chains):
            source_slice = slice(source * states, (source + 1) * states)
            pair_counts[target, source] = pair_counts_flat[source_slice, target_slice]

    alpha = alpha_flat.reshape(observations.shape[0], chains, states).permute(1, 0, 2)
    beta = beta_flat.reshape(observations.shape[0], chains, states).permute(1, 0, 2)
    marginals = gamma_flat.reshape(observations.shape[0], chains, states).permute(1, 0, 2)
    return DongLinearResult(
        log_likelihood,
        alpha,
        beta,
        marginals,
        pair_counts,
        influence_matrix,
        scales,
    )
