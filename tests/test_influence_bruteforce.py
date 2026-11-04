import torch

from latentlinks.influence import InfluenceModel
from latentlinks.influence.brute_force import brute_force_inference
from latentlinks.influence.inference import exact_inference


def tiny_influence_model() -> InfluenceModel:
    initial = torch.tensor([[0.6, 0.4], [0.5, 0.5]])
    transition = torch.tensor(
        [
            [
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.8, 0.2], [0.3, 0.7]],
            ],
            [
                [[0.7, 0.3], [0.1, 0.9]],
                [[0.85, 0.15], [0.25, 0.75]],
            ],
        ]
    )
    emission = torch.tensor([[[0.9, 0.1], [0.2, 0.8]], [[0.8, 0.2], [0.15, 0.85]]])
    influence = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    return InfluenceModel(initial, transition, emission, influence)


def test_joint_state_inference_matches_all_tiny_trajectories():
    observations = torch.tensor([[0, 1], [1, 1], [0, 0]])
    brute = brute_force_inference(tiny_influence_model(), observations)
    exact = exact_inference(tiny_influence_model(), observations)
    torch.testing.assert_close(torch.exp(exact.log_likelihood), brute.likelihood)
    torch.testing.assert_close(exact.chain_marginals, brute.chain_marginals)


def test_joint_transition_is_stochastic():
    model = tiny_influence_model()
    from latentlinks.influence.inference import compile_joint_transition, enumerate_joint_states

    transition = compile_joint_transition(model, enumerate_joint_states(model))
    torch.testing.assert_close(
        transition.sum(dim=1), torch.ones(transition.shape[0], dtype=torch.float64)
    )
