import torch

from latentlinks.influence import InfluenceModel, exact_inference, factorized_inference


def independent_model() -> InfluenceModel:
    initial = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
    own = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    transition = own.repeat(2, 2, 1, 1)
    emission = torch.tensor([[[0.85, 0.15], [0.1, 0.9]], [[0.8, 0.2], [0.2, 0.8]]])
    influence = torch.eye(2)
    return InfluenceModel(initial, transition, emission, influence)


def test_factorized_method_is_exact_for_independent_chains():
    observations = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]])
    model = independent_model()
    approximate = factorized_inference(model, observations)
    exact = exact_inference(model, observations)
    torch.testing.assert_close(approximate.marginals, exact.chain_marginals)


def test_factorized_marginals_and_effective_dynamics_normalize():
    observations, _ = independent_model().sample(8, seed=2)
    result = factorized_inference(independent_model(), observations)
    torch.testing.assert_close(result.marginals.sum(dim=2), torch.ones(2, 8, dtype=torch.float64))
