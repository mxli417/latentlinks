"""Tensorized counterparts to the readable Dong linear implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from .dong import DongLinearResult
from .model import InfluenceModel


def compile_influence_matrix_vectorized(model: InfluenceModel) -> Tensor:
    """Compile source-state rows and target-state columns without Python blocks."""
    weighted = model.transition * model.influence[:, :, None, None]
    return weighted.permute(1, 2, 0, 3).reshape(
        model.n_chains * model.n_states,
        model.n_chains * model.n_states,
    )


def dong_linear_inference_vectorized(
    model: InfluenceModel, observations: Tensor, *, epsilon: float = 1e-30
) -> DongLinearResult:
    """Vectorize chain/state assembly and all two-slice expected counts."""
    observations = torch.as_tensor(observations, dtype=torch.long)
    if observations.ndim != 2 or observations.shape[1] != model.n_chains:
        raise ValueError("observations must have shape (time, chains)")
    if observations.shape[0] < 1:
        raise ValueError("at least one observation is required")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    chains, states = model.n_chains, model.n_states
    flat_states = chains * states
    influence_matrix = compile_influence_matrix_vectorized(model)
    symbol_indices = observations.T[:, None, :].expand(chains, states, -1)
    emission = (
        model.emission.gather(2, symbol_indices)
        .permute(2, 0, 1)
        .reshape(observations.shape[0], flat_states)
        .clamp_min(epsilon)
    )

    first = model.initial.reshape(-1) * emission[0]
    first_blocks = first.reshape(chains, states)
    scale_rows = [first_blocks.sum(dim=1)]
    alpha_rows = [(first_blocks / scale_rows[0][:, None]).reshape(-1)]
    for time in range(1, observations.shape[0]):
        unnormalized = (alpha_rows[-1] @ influence_matrix) * emission[time]
        blocks = unnormalized.reshape(chains, states)
        scale_rows.append(blocks.sum(dim=1))
        alpha_rows.append((blocks / scale_rows[-1][:, None]).reshape(-1))

    alpha_flat = torch.stack(alpha_rows)
    scales = torch.stack(scale_rows)
    repeated_scales = scales.repeat_interleave(states, dim=1)
    beta_reverse = [torch.ones(flat_states, dtype=model.initial.dtype)]
    for time in range(observations.shape[0] - 2, -1, -1):
        coefficient = emission[time + 1] * beta_reverse[-1] / repeated_scales[time + 1]
        beta_reverse.append(influence_matrix @ coefficient)
    beta_flat = torch.stack(list(reversed(beta_reverse)))
    marginals_flat = alpha_flat * beta_flat

    coefficient = emission[1:] * beta_flat[1:] / repeated_scales[1:]
    pair_counts_flat = torch.einsum("ti,ij,tj->ij", alpha_flat[:-1], influence_matrix, coefficient)
    pair_counts = pair_counts_flat.reshape(chains, states, chains, states).permute(2, 0, 1, 3)

    alpha = alpha_flat.reshape(-1, chains, states).permute(1, 0, 2)
    beta = beta_flat.reshape(-1, chains, states).permute(1, 0, 2)
    marginals = marginals_flat.reshape(-1, chains, states).permute(1, 0, 2)
    return DongLinearResult(
        torch.log(scales).sum(),
        alpha,
        beta,
        marginals,
        pair_counts,
        influence_matrix,
        scales,
    )
