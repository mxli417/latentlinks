"""Publicly documented experiment parameters, never historical source code."""

from __future__ import annotations

import torch

from .model import InfluenceModel


def mit_six_chain_model() -> InfluenceModel:
    """Construct the six-chain synthetic example from the MIT usage page."""
    chains = 6
    pairwise = torch.tensor([[0.99, 0.01], [0.08, 0.92]], dtype=torch.float64)
    emission_one = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)
    transition = pairwise.repeat(chains, chains, 1, 1)
    emission = emission_one.repeat(chains, 1, 1)
    influence = torch.zeros(chains, chains, dtype=torch.float64)
    influence[:3, :3] = 1 / 3
    influence[3:, 3:] = 1 / 3
    initial = torch.full((chains, 2), 0.5, dtype=torch.float64)
    return InfluenceModel(initial, transition, emission, influence)
