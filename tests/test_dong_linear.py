import torch

from latentlinks.influence import (
    InfluenceModel,
    compile_influence_matrix,
    dong_linear_em_step,
    dong_linear_inference,
    exact_inference,
    fit_dong_linear_em,
)


def independent_model() -> InfluenceModel:
    initial = torch.tensor([[0.55, 0.45], [0.4, 0.6]])
    transition = torch.empty(2, 2, 2, 2)
    transition[0, 0] = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    transition[0, 1] = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    transition[1, 0] = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    transition[1, 1] = torch.tensor([[0.8, 0.2], [0.15, 0.85]])
    emission = torch.tensor([[[0.9, 0.1], [0.1, 0.9]], [[0.85, 0.15], [0.2, 0.8]]])
    return InfluenceModel(initial, transition, emission, torch.eye(2))


def block_model() -> InfluenceModel:
    chains = 4
    initial = torch.full((chains, 2), 0.5)
    transition = torch.empty(chains, chains, 2, 2)
    persistent = torch.tensor([[0.94, 0.06], [0.08, 0.92]])
    copying = torch.tensor([[0.88, 0.12], [0.12, 0.88]])
    uninformative = torch.full((2, 2), 0.5)
    for target in range(chains):
        for source in range(chains):
            same_block = target // 2 == source // 2
            transition[target, source] = (
                persistent if target == source else copying if same_block else uninformative
            )
    emission = torch.tensor([[[0.96, 0.04], [0.04, 0.96]]] * chains)
    influence = torch.tensor(
        [
            [0.75, 0.25, 0.0, 0.0],
            [0.30, 0.70, 0.0, 0.0],
            [0.0, 0.0, 0.80, 0.20],
            [0.0, 0.0, 0.25, 0.75],
        ]
    )
    return InfluenceModel(initial, transition, emission, influence)


def test_linear_inference_is_exact_for_independent_chains():
    model = independent_model()
    observations, _ = model.sample(7, seed=4)
    linear = dong_linear_inference(model, observations)
    exact = exact_inference(model, observations)
    torch.testing.assert_close(linear.log_likelihood, exact.log_likelihood)
    torch.testing.assert_close(linear.marginals, exact.chain_marginals)
    torch.testing.assert_close(linear.marginals.sum(dim=2), torch.ones(2, 7, dtype=torch.float64))


def test_compiled_matrix_has_source_rows_and_target_columns():
    model = independent_model()
    matrix = compile_influence_matrix(model)
    torch.testing.assert_close(matrix[:2, :2], model.transition[0, 0])
    assert torch.count_nonzero(matrix[:2, 2:]) == 0


def test_one_em_step_does_not_decrease_linear_objective():
    truth = block_model()
    sequences = [truth.sample(80, seed=seed)[0] for seed in range(4)]
    start = InfluenceModel(
        truth.initial,
        truth.transition + 0.08,
        truth.emission + 0.05,
        truth.influence + 0.10,
    )
    updated, before = dong_linear_em_step(start, sequences)
    after = sum(dong_linear_inference(updated, sequence).log_likelihood for sequence in sequences)
    assert after >= before - 1e-8


def test_recovers_synthetic_block_network():
    truth = block_model()
    sequences = [truth.sample(180, seed=seed)[0] for seed in range(12)]
    generator = torch.Generator().manual_seed(12)
    start = InfluenceModel(
        truth.initial + 0.05 * torch.rand(truth.initial.shape, generator=generator),
        truth.transition + 0.08 * torch.rand(truth.transition.shape, generator=generator),
        truth.emission + 0.05 * torch.rand(truth.emission.shape, generator=generator),
        torch.full_like(truth.influence, 0.25),
    )
    fitted, history = fit_dong_linear_em(start, sequences, max_iterations=35)
    # Directly compare total mass assigned inside versus outside each known block.
    within_mass = torch.stack(
        [
            fitted.influence[target, (target // 2) * 2 : (target // 2) * 2 + 2].sum()
            for target in range(4)
        ]
    )
    assert torch.all(within_mass > 0.80)
    assert len(history.log_likelihoods) > 1
    assert history.log_likelihoods[-1] > history.log_likelihoods[0]
