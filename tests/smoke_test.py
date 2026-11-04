"""Minimal installed-distribution test used by the release workflow."""

import torch

import latentlinks

model = latentlinks.CategoricalHMM(
    initial=torch.tensor([0.6, 0.4]),
    transition=torch.tensor([[0.8, 0.2], [0.3, 0.7]]),
    emission=torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
)
score = latentlinks.forward_log_likelihood(model, torch.tensor([0, 0, 1, 1]))
assert torch.isfinite(score)

influence_model = latentlinks.mit_six_chain_model()
observations, _ = influence_model.sample(3, seed=1)
linear = latentlinks.dong_linear_inference(influence_model, observations)
assert linear.marginals.shape == (6, 3, 2)
assert latentlinks.__version__ == "0.1.0"
