import json
from pathlib import Path

import torch

from latentlinks.influence import InfluenceModel, dong_linear_inference, mit_six_chain_model
from latentlinks.network import influence_to_adjacency

REFERENCE_DIR = Path(__file__).parent / "reference"


def test_documented_mit_six_chain_parameters():
    model = mit_six_chain_model()
    expected_transition = torch.tensor([[0.99, 0.01], [0.08, 0.92]], dtype=torch.float64)
    expected_emission = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float64)
    torch.testing.assert_close(model.transition[0, 0], expected_transition)
    torch.testing.assert_close(model.emission[0], expected_emission)
    torch.testing.assert_close(
        model.influence[:3, :3], torch.full((3, 3), 1 / 3, dtype=torch.float64)
    )
    assert torch.count_nonzero(model.influence[:3, 3:]) == 0
    assert torch.count_nonzero(model.influence[3:, :3]) == 0


def test_network_orientation_is_source_by_target():
    influence = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    adjacency = influence_to_adjacency(influence, include_self=True)
    torch.testing.assert_close(adjacency, influence.T)


def test_dong_linear_inference_matches_matlab_toolbox_fixture():
    fixture = json.loads((REFERENCE_DIR / "mit_toolbox_linear_tiny.json").read_text())
    observations = torch.tensor(fixture["observations_one_based"], dtype=torch.long).T - 1
    initial = torch.tensor(fixture["initial"], dtype=torch.float64)

    # MATLAB cat(3, B{1}, B{2}) is (state, symbol, chain).
    emission_matlab = torch.tensor(fixture["emission"], dtype=torch.float64)
    emission = emission_matlab.permute(2, 0, 1)

    # The fixture concatenates A{source,target} in the order 11, 12, 21, 22.
    blocks = torch.tensor(fixture["transition_source_target"], dtype=torch.float64)
    transition = torch.empty(2, 2, 2, 2, dtype=torch.float64)
    transition[0, 0] = blocks[:, :, 0]
    transition[1, 0] = blocks[:, :, 1]
    transition[0, 1] = blocks[:, :, 2]
    transition[1, 1] = blocks[:, :, 3]

    influence_source_target = torch.tensor(fixture["influence_source_target"], dtype=torch.float64)
    model = InfluenceModel(initial, transition, emission, influence_source_target.T)
    actual = dong_linear_inference(model, observations)

    expected_gamma = torch.tensor(fixture["gamma_time_by_flat_state"], dtype=torch.float64)
    actual_gamma = actual.marginals.permute(1, 0, 2).reshape(observations.shape[0], -1)

    expected_xi = torch.tensor(fixture["xi_flat_source_by_target"], dtype=torch.float64)
    actual_xi = torch.empty_like(expected_xi)
    for source in range(2):
        source_slice = slice(2 * source, 2 * source + 2)
        for target in range(2):
            target_slice = slice(2 * target, 2 * target + 2)
            actual_xi[source_slice, target_slice] = actual.expected_pair_transitions[target, source]

    torch.testing.assert_close(
        actual.log_likelihood,
        torch.tensor(fixture["log_likelihood"], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(actual_gamma, expected_gamma, atol=1e-12, rtol=0)
    torch.testing.assert_close(actual_xi, expected_xi, atol=1e-12, rtol=0)
