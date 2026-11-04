"""The ordinary categorical hidden Markov model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..emissions import CategoricalEmission
from ..utils.normalization import normalize


@dataclass(frozen=True)
class CategoricalHMM:
    """A finite-state HMM represented by three explicit probability tables.

    ``initial[k]`` is ``P(z[0]=k)``; ``transition[i,j]`` is
    ``P(z[t]=j | z[t-1]=i)``; and ``emission[k,m]`` is
    ``P(x[t]=m | z[t]=k)``.
    """

    initial: Tensor
    transition: Tensor
    emission: Tensor

    def __post_init__(self) -> None:
        dtype = torch.float64
        initial = torch.as_tensor(self.initial, dtype=dtype)
        transition = torch.as_tensor(self.transition, dtype=dtype)
        emission = torch.as_tensor(self.emission, dtype=dtype)
        if initial.ndim != 1 or initial.numel() == 0:
            raise ValueError("initial must have shape (states,)")
        if transition.shape != (initial.numel(), initial.numel()):
            raise ValueError("transition must have shape (states, states)")
        if emission.ndim != 2 or emission.shape[0] != initial.numel():
            raise ValueError("emission must have shape (states, symbols)")
        object.__setattr__(self, "initial", normalize(initial))
        object.__setattr__(self, "transition", normalize(transition, dim=1))
        object.__setattr__(self, "emission", normalize(emission, dim=1))

    @property
    def n_states(self) -> int:
        return self.initial.numel()

    @property
    def n_symbols(self) -> int:
        return self.emission.shape[1]

    @property
    def emission_distribution(self) -> CategoricalEmission:
        return CategoricalEmission(self.emission)

    @classmethod
    def random(cls, n_states: int, n_symbols: int, seed: int | None = None) -> CategoricalHMM:
        if n_states < 1 or n_symbols < 1:
            raise ValueError("state and symbol counts must be positive")
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        return cls(
            torch.rand(n_states, generator=generator, dtype=torch.float64),
            torch.rand(n_states, n_states, generator=generator, dtype=torch.float64),
            torch.rand(n_states, n_symbols, generator=generator, dtype=torch.float64),
        )

    def sample(self, length: int, seed: int | None = None) -> tuple[Tensor, Tensor]:
        """Draw observations and latent states using deliberately explicit steps."""
        if length < 1:
            raise ValueError("length must be positive")
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        states = torch.empty(length, dtype=torch.long)
        observations = torch.empty(length, dtype=torch.long)
        states[0] = torch.multinomial(self.initial, 1, generator=generator)
        observations[0] = torch.multinomial(self.emission[states[0]], 1, generator=generator)
        for time in range(1, length):
            states[time] = torch.multinomial(
                self.transition[states[time - 1]], 1, generator=generator
            )
            observations[time] = torch.multinomial(
                self.emission[states[time]], 1, generator=generator
            )
        return observations, states
