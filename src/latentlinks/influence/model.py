"""The latent-state Influence Model as a distinct structured model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..utils.normalization import normalize


@dataclass(frozen=True)
class InfluenceModel:
    """Homogeneous-chain categorical Influence Model.

    Shapes are intentionally uniform for the first pedagogical implementation:

    - ``initial``: ``(chains, states)``
    - ``transition``: ``(target, source, source_state, target_state)``
    - ``emission``: ``(chains, states, symbols)``
    - ``influence``: ``(target, source)``
    """

    initial: Tensor
    transition: Tensor
    emission: Tensor
    influence: Tensor

    def __post_init__(self) -> None:
        dtype = torch.float64
        initial = torch.as_tensor(self.initial, dtype=dtype)
        transition = torch.as_tensor(self.transition, dtype=dtype)
        emission = torch.as_tensor(self.emission, dtype=dtype)
        influence = torch.as_tensor(self.influence, dtype=dtype)
        if initial.ndim != 2 or min(initial.shape) < 1:
            raise ValueError("initial must have shape (chains, states)")
        chains, states = initial.shape
        if transition.shape != (chains, chains, states, states):
            raise ValueError("transition must have shape (target, source, state, state)")
        if emission.ndim != 3 or emission.shape[:2] != (chains, states):
            raise ValueError("emission must have shape (chains, states, symbols)")
        if influence.shape != (chains, chains):
            raise ValueError("influence must have shape (target, source)")
        object.__setattr__(self, "initial", normalize(initial, dim=1))
        object.__setattr__(self, "transition", normalize(transition, dim=3))
        object.__setattr__(self, "emission", normalize(emission, dim=2))
        object.__setattr__(self, "influence", normalize(influence, dim=1))

    @property
    def n_chains(self) -> int:
        return self.initial.shape[0]

    @property
    def n_states(self) -> int:
        return self.initial.shape[1]

    @property
    def n_symbols(self) -> int:
        return self.emission.shape[2]

    def target_transition(self, previous_states: Tensor, target: int) -> Tensor:
        """Compute the mixture distribution for one target chain explicitly."""
        distribution = torch.zeros(self.n_states, dtype=self.initial.dtype)
        for source in range(self.n_chains):
            source_state = previous_states[source]
            contribution = self.transition[target, source, source_state]
            distribution += self.influence[target, source] * contribution
        return distribution

    def sample(self, length: int, seed: int | None = None) -> tuple[Tensor, Tensor]:
        if length < 1:
            raise ValueError("length must be positive")
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        states = torch.empty(length, self.n_chains, dtype=torch.long)
        observations = torch.empty(length, self.n_chains, dtype=torch.long)
        for chain in range(self.n_chains):
            states[0, chain] = torch.multinomial(self.initial[chain], 1, generator=generator)
            observations[0, chain] = torch.multinomial(
                self.emission[chain, states[0, chain]], 1, generator=generator
            )
        for time in range(1, length):
            for target in range(self.n_chains):
                next_distribution = self.target_transition(states[time - 1], target)
                states[time, target] = torch.multinomial(next_distribution, 1, generator=generator)
                observations[time, target] = torch.multinomial(
                    self.emission[target, states[time, target]], 1, generator=generator
                )
        return observations, states
