# Latent Links

**Can several observed time series reveal both their hidden behavioral states and
the interaction network connecting those states?**

Latent Links is an independent, pedagogical PyTorch treatment of that question.
It begins with an ordinary hidden Markov model, proves inference against every
latent path in tiny examples, then moves carefully toward interacting HMMs, the
Dong–Pentland Influence Model, and directed network extraction.

```text
ordinary HMM
    → interacting latent chains
    → Influence Model
    → influence matrix
    → directed predictive-interaction graph
```

The historical MIT MATLAB toolbox is external evidence, not the source of this
implementation. No MATLAB source is included or translated.

> **Status:** the ordinary categorical HMM, exact tiny Influence Model, Dong's
> linear marginal-state inference/EM formulation, and a separately labeled
> project-derived factorized approximation are implemented and tested. Synthetic
> HMM parameter recovery and Influence Model block-network recovery are included.

## The method is the project

Every important algorithm follows the same progression:

```text
derive → implement simply → prove on tiny examples → compare → optimize
```

That means Latent Links keeps readable reference implementations even when a
faster tensorized implementation is added later. You should be able to stop at a
breakpoint, inspect `alpha`, `beta`, `gamma`, `xi`, or an effective transition
matrix, and connect it directly to an equation.

The evidence hierarchy is similarly explicit:

```text
brute-force mathematics
        > synthetic recovery
        > historical reference fixtures
        > mock-based software tests
```

## What is implemented

| Layer | Reference implementation | Evidence |
| --- | --- | --- |
| HMM generation | explicit PyTorch sampling | probability normalization tests |
| HMM likelihood | all `K**T` paths, then forward recursion | exact equality for `K=2, T=4` |
| HMM posteriors | forward–backward `gamma` and `xi` | exhaustive posterior equality |
| HMM learning | literal expected-count Baum–Welch | non-decreasing observed likelihood |
| Gradient HMM | softmax logits and Adam | differentiable likelihood and improvement test |
| Influence Model | `D[target, source]` pairwise transition mixture | stochasticity tests |
| Tiny coupled inference | all `K**(C*T)` trajectories, then joint-state DP | exact equality for `C=2, K=2, T=3` |
| Factorized inference | coordinate updates with effective transitions | exact when chains are independent |
| Dong linear inference | scaled `C*K` marginal recursion and expected counts | exact for independent chains; MATLAB driver prepared |
| Influence learning | Dong linear EM updates for `Pi`, `A`, `B`, and `D` | objective improvement and block-network recovery |
| MIT experiment | independently constructed public parameters | documented provenance test |
| Network extraction | `adjacency = influence.T` | orientation test |

## Install

```console
pip install latentlinks
```

For a development checkout:

```console
git clone https://github.com/mxli417/latentlinks.git
cd latentlinks
uv sync
```

Latent Links supports Python 3.10+ and uses PyTorch for all model computation.

## Start with a two-state HMM

```python
import torch

from latentlinks.hmm import (
    CategoricalHMM,
    brute_force_inference,
    forward_backward,
)

model = CategoricalHMM(
    initial=torch.tensor([0.6, 0.4]),
    transition=torch.tensor([[0.7, 0.3], [0.2, 0.8]]),
    emission=torch.tensor([[0.9, 0.1], [0.25, 0.75]]),
)
observations = torch.tensor([0, 1, 1, 0])

oracle = brute_force_inference(model, observations)  # sums all 2**4 paths
posterior = forward_backward(model, observations)

torch.testing.assert_close(
    oracle.likelihood,
    torch.exp(posterior.log_likelihood),
)
torch.testing.assert_close(oracle.gamma, posterior.gamma)
torch.testing.assert_close(oracle.xi, posterior.xi)
```

The forward–backward result exposes every teaching quantity:

- `log_alpha[t, k]`: evidence accumulated up to time `t` ending in state `k`;
- `log_beta[t, k]`: future evidence after time `t` given state `k`;
- `gamma[t, k]`: posterior probability that state `k` generated observation `t`;
- `xi[t, i, j]`: posterior probability of transition `i → j` at time `t`.

## EM means expected counting

If states were visible, learning a transition matrix would mean counting
transitions. With hidden states, Baum–Welch replaces the missing counts by their
posterior expectations:

```text
N_ij = E[#(i → j) | X, θ] = sum_t xi[t, i, j]
```

The corresponding implementation is intentionally literal:

```python
expected_transitions = posterior.xi.sum(dim=0)
transition_new = expected_transitions / expected_transitions.sum(
    dim=1,
    keepdim=True,
)
```

See [`hmm/em.py`](src/latentlinks/hmm/em.py) before reading the autograd version.
Autograd replaces hand-derived parameter gradients; it does not replace
latent-variable inference.

## The Influence Model

For target chain `c` and source chain `j`, define a pairwise transition table

```text
A[target=c, source=j, source_state=r, target_state=s].
```

The next-state distribution for the target is a convex mixture:

```text
P(z_c[t+1]=s | z_1:C[t])
  = sum_j D[c,j] A[c,j,z_j[t],s].
```

`D[c,j]` is the contribution of source `j` to target `c`. Rows of `D` sum to one.
For conventional graph orientation:

```python
adjacency[source, target] = influence[target, source]
adjacency = influence.T
```

These weights describe **predictive coupling** or **directed dynamical
dependency**. They are not automatically causal effects.

## Exactness, approximation, and honesty

With `C` chains and `K` states, the joint latent state has `K**C` possibilities.
Latent Links therefore provides two deliberately separate paths:

1. `exact_inference` constructs the Cartesian-product state model. It is slow but
   trusted for tiny systems.
2. `factorized_inference` updates one chain posterior while holding the current
   marginals of other chains fixed. It discards posterior dependence between
   chains and is labeled as an independently derived approximation.

Dong's thesis develops linear and nonlinear Influence Model inference procedures.
`dong_linear_inference` implements the selected linear marginal-state recursion;
the coordinate method in `variational.py` is **not** presented as Dong's. See
[`docs/literature.md`](docs/literature.md) for the precise lineage and
[`docs/architecture.md`](docs/architecture.md) for tensor conventions.

## Historical MIT experiment

```python
from latentlinks.influence import mit_six_chain_model

model = mit_six_chain_model()
observations, hidden_states = model.sample(length=1000, seed=7)
```

This reconstructs only parameters documented on the public MIT usage page: six
chains, two states, two symbols, noisy emissions, and two three-chain influence
blocks. Reference provenance and the rules for future external fixtures live in
[`tests/reference/README.md`](tests/reference/README.md).

## Repository map

```text
src/latentlinks/
├── hmm/          # brute force → forward/backward → EM → autograd
├── influence/    # exact, Dong linear, and project-derived inference paths
├── emissions/    # categorical and diagonal-Gaussian observation models
├── network/      # influence-to-adjacency conversion and recovery metrics
└── utils/        # normalization and state-label alignment

tutorials/        # equation-led executable learning path
tests/            # mathematical oracles, synthetic checks, reference provenance
docs/             # architecture, literature, and publishing
```

## Development

```console
uv sync
uv run pytest --cov
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build --no-sources
```

To install notebook tooling as well:

```console
uv sync --group tutorials
```

## Scientific lineage

- Brand, Oliver & Pentland, [“Coupled Hidden Markov Models for Complex Action
  Recognition”](https://doi.org/10.1109/CVPR.1997.609450), CVPR 1997.
- Wen Dong, [*Influence Modeling of Complex Stochastic
  Processes*](https://vismod.media.mit.edu/vismod/demos/influence-model/wdong06thesis.pdf),
  MIT, 2006.
- [MIT Influence Model toolbox usage
  page](https://vismod.media.mit.edu/vismod/demos/influence-model/software-usage.htm),
  used only as an external behavioral specification.

This repository contains independently written code based on mathematical
descriptions. It does not include or translate the historical MATLAB source.

## License

Latent Links is released under the [MIT License](LICENSE).
