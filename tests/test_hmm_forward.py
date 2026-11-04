import torch

from latentlinks.hmm import (
    CategoricalHMM,
    backward_probabilities,
    forward_log_likelihood,
    forward_probabilities,
)


def test_probability_and_log_space_forward_agree():
    model = CategoricalHMM(
        [0.55, 0.45],
        [[0.8, 0.2], [0.3, 0.7]],
        [[0.85, 0.15], [0.1, 0.9]],
    )
    observations = torch.tensor([0, 0, 1, 1, 0])
    alpha = forward_probabilities(model, observations)
    log_likelihood = forward_log_likelihood(model, observations)
    torch.testing.assert_close(alpha[-1].sum(), torch.exp(log_likelihood))


def test_backward_terminal_condition_is_one():
    model = CategoricalHMM.random(3, 2, seed=4)
    beta = backward_probabilities(model, torch.tensor([0, 1, 0]))
    torch.testing.assert_close(beta[-1], torch.ones(3, dtype=torch.float64))
