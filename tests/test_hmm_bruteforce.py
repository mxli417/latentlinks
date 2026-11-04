import torch

from latentlinks.hmm import CategoricalHMM, brute_force_inference, forward_backward


def tiny_hmm() -> CategoricalHMM:
    return CategoricalHMM(
        initial=torch.tensor([0.6, 0.4]),
        transition=torch.tensor([[0.7, 0.3], [0.2, 0.8]]),
        emission=torch.tensor([[0.9, 0.1], [0.25, 0.75]]),
    )


def test_forward_backward_matches_all_sixteen_paths():
    observations = torch.tensor([0, 1, 1, 0])
    brute = brute_force_inference(tiny_hmm(), observations)
    dynamic = forward_backward(tiny_hmm(), observations)

    torch.testing.assert_close(torch.exp(dynamic.log_likelihood), brute.likelihood)
    torch.testing.assert_close(dynamic.gamma, brute.gamma)
    torch.testing.assert_close(dynamic.xi, brute.xi)


def test_oracle_posteriors_are_normalized():
    result = brute_force_inference(tiny_hmm(), torch.tensor([0, 1, 0, 1]))
    torch.testing.assert_close(result.gamma.sum(dim=1), torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close(result.xi.sum(dim=(1, 2)), torch.ones(3, dtype=torch.float64))
