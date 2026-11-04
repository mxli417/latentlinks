"""Influence Models, kept distinct from ordinary and general coupled HMMs."""

from .brute_force import (
    InfluenceBruteForceResult,
    brute_force_inference,
    trajectory_joint_probability,
)
from .dong import DongLinearResult, compile_influence_matrix, dong_linear_inference
from .em import InfluenceEMHistory, dong_linear_em_step, fit_dong_linear_em
from .inference import ExactInfluenceResult, exact_inference
from .model import InfluenceModel
from .optimized import (
    compile_influence_matrix_vectorized,
    dong_linear_inference_vectorized,
)
from .reference import mit_six_chain_model
from .variational import (
    FactorizedInferenceResult,
    effective_transitions,
    factorized_inference,
)

__all__ = [
    "ExactInfluenceResult",
    "DongLinearResult",
    "FactorizedInferenceResult",
    "InfluenceBruteForceResult",
    "InfluenceModel",
    "InfluenceEMHistory",
    "brute_force_inference",
    "compile_influence_matrix",
    "compile_influence_matrix_vectorized",
    "dong_linear_em_step",
    "dong_linear_inference",
    "dong_linear_inference_vectorized",
    "effective_transitions",
    "exact_inference",
    "factorized_inference",
    "fit_dong_linear_em",
    "mit_six_chain_model",
    "trajectory_joint_probability",
]
