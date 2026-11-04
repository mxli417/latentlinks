"""Pedagogical PyTorch inference for interacting latent-state systems."""

from .hmm import (
    CategoricalHMM,
    baum_welch_step,
    brute_force_inference,
    fit_baum_welch,
    fit_gradient,
    forward_backward,
    forward_log_likelihood,
)
from .influence import (
    InfluenceModel,
    dong_linear_inference,
    exact_inference,
    factorized_inference,
    fit_dong_linear_em,
    mit_six_chain_model,
)
from .network import influence_to_adjacency

__all__ = [
    "CategoricalHMM",
    "InfluenceModel",
    "baum_welch_step",
    "brute_force_inference",
    "dong_linear_inference",
    "exact_inference",
    "factorized_inference",
    "fit_baum_welch",
    "fit_dong_linear_em",
    "fit_gradient",
    "forward_backward",
    "forward_log_likelihood",
    "influence_to_adjacency",
    "mit_six_chain_model",
]

__version__ = "0.1.0"
