"""Small, inspectable utilities shared by the teaching implementations."""

from .alignment import best_state_permutation
from .normalization import assert_stochastic, normalize

__all__ = ["assert_stochastic", "best_state_permutation", "normalize"]
