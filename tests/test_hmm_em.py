import torch

from latentlinks.hmm import (
    CategoricalHMM,
    baum_welch_step,
    fit_gradient,
    forward_log_likelihood,
)


def total_likelihood(model: CategoricalHMM, sequences: list[torch.Tensor]) -> torch.Tensor:
    return sum(forward_log_likelihood(model, sequence) for sequence in sequences)


def test_exact_em_does_not_decrease_observed_likelihood():
    source = CategoricalHMM(
        [0.8, 0.2],
        [[0.9, 0.1], [0.15, 0.85]],
        [[0.9, 0.1], [0.1, 0.9]],
    )
    sequences = [source.sample(40, seed)[0] for seed in range(5)]
    initial = CategoricalHMM.random(2, 2, seed=22)
    before = total_likelihood(initial, sequences)
    updated, _ = baum_welch_step(initial, sequences)
    after = total_likelihood(updated, sequences)
    assert after >= before - 1e-8
    torch.testing.assert_close(updated.transition.sum(dim=1), torch.ones(2, dtype=torch.float64))


def test_autograd_likelihood_is_differentiable_and_improves():
    source = CategoricalHMM([0.7, 0.3], [[0.9, 0.1], [0.2, 0.8]], [[0.8, 0.2], [0.1, 0.9]])
    sequences = [source.sample(25, seed)[0] for seed in range(3)]
    initial = CategoricalHMM.random(2, 2, seed=9)
    before = total_likelihood(initial, sequences)
    fitted, history = fit_gradient(initial, sequences, iterations=25, learning_rate=0.03)
    assert total_likelihood(fitted, sequences) > before
    assert len(history.log_likelihoods) == 25
