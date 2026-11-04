import torch

from latentlinks.hmm import (
    CategoricalHMM,
    forward_backward,
    forward_backward_vectorized,
)
from latentlinks.influence import (
    InfluenceModel,
    dong_linear_inference,
    dong_linear_inference_vectorized,
)


def test_vectorized_hmm_matches_reference_messages():
    model = CategoricalHMM.random(4, 3, seed=5)
    observations, _ = model.sample(25, seed=6)
    reference = forward_backward(model, observations)
    optimized = forward_backward_vectorized(model, observations)
    torch.testing.assert_close(optimized.log_likelihood, reference.log_likelihood)
    torch.testing.assert_close(optimized.log_alpha, reference.log_alpha)
    torch.testing.assert_close(optimized.log_beta, reference.log_beta)
    torch.testing.assert_close(optimized.gamma, reference.gamma)
    torch.testing.assert_close(optimized.xi, reference.xi)


def test_vectorized_dong_linear_matches_reference_messages_and_counts():
    generator = torch.Generator().manual_seed(7)
    model = InfluenceModel(
        torch.rand(3, 2, generator=generator),
        torch.rand(3, 3, 2, 2, generator=generator),
        torch.rand(3, 2, 3, generator=generator),
        torch.rand(3, 3, generator=generator),
    )
    observations, _ = model.sample(30, seed=8)
    reference = dong_linear_inference(model, observations)
    optimized = dong_linear_inference_vectorized(model, observations)
    torch.testing.assert_close(optimized.log_likelihood, reference.log_likelihood)
    torch.testing.assert_close(optimized.alpha, reference.alpha)
    torch.testing.assert_close(optimized.beta, reference.beta)
    torch.testing.assert_close(optimized.marginals, reference.marginals)
    torch.testing.assert_close(
        optimized.expected_pair_transitions,
        reference.expected_pair_transitions,
    )
    torch.testing.assert_close(optimized.influence_matrix, reference.influence_matrix)
