import torch

from latentlinks.hmm import CategoricalHMM, align_hmm_states, fit_baum_welch


def test_multi_iteration_baum_welch_recovers_parameters_up_to_labels():
    truth = CategoricalHMM(
        torch.tensor([0.65, 0.35]),
        torch.tensor([[0.92, 0.08], [0.16, 0.84]]),
        torch.tensor([[0.94, 0.06], [0.08, 0.92]]),
    )
    sequences = [truth.sample(220, seed=seed)[0] for seed in range(18)]
    start = CategoricalHMM(
        torch.tensor([0.5, 0.5]),
        torch.tensor([[0.65, 0.35], [0.35, 0.65]]),
        torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
    )
    fitted, history = fit_baum_welch(start, sequences, max_iterations=60)
    aligned = align_hmm_states(truth, fitted)
    assert len(history.log_likelihoods) > 2
    assert history.log_likelihoods[-1] > history.log_likelihoods[0]
    torch.testing.assert_close(aligned.transition, truth.transition, atol=0.08, rtol=0)
    torch.testing.assert_close(aligned.emission, truth.emission, atol=0.06, rtol=0)
