# Architecture: equations before acceleration

Latent Links follows one dependency and one evidence hierarchy:

```text
derive → implement simply → prove on tiny examples → compare → optimize

brute-force mathematics > synthetic recovery > historical fixtures > mocks
```

## Package map

```text
src/latentlinks/
├── hmm/
│   ├── model.py              categorical HMM and explicit sampling
│   ├── brute_force.py        K**T path-enumeration oracle
│   ├── forward_backward.py   loop-based probability and log-space inference
│   ├── em.py                 literal expected-count Baum–Welch
│   └── autograd.py           logits and Adam, introduced last
├── influence/
│   ├── model.py              D[target, source] Influence Model
│   ├── brute_force.py        K**(C*T) trajectory oracle
│   ├── inference.py          exact Cartesian-product inference
│   ├── dong.py               Dong linear marginal-state forward/backward
│   ├── variational.py        clearly labeled project-derived approximation
│   ├── em.py                 Dong linear expected-count learning
│   └── reference.py          public MIT example parameters only
├── emissions/                categorical and diagonal-Gaussian observations
├── network/                  influence-to-graph conversion and metrics
└── utils/                    normalization and label alignment
```

The reference implementations favor named intermediates and Python loops.
Optimized implementations should eventually live beside them with names such as
`forward_log_probabilities_vectorized`, never replace them.

## Model boundaries

An **ordinary HMM** has one latent state per time step.

A **general coupled HMM** has several latent states and permits a target chain's
next state to depend on the complete previous joint state. Its conditional table
can therefore become very large.

The **Influence Model** is a structured member of that broader family. It forms a
target transition by mixing pairwise source-to-target tables:

```text
P(z_target[t+1] | z_1:C[t])
  = sum_source D[target, source]
      A[target, source, z_source[t], z_target[t+1]].
```

The three terms are related, not interchangeable.

## Tensor conventions

- HMM observations: `(time,)`
- Influence observations and states: `(time, chains)`
- Chain marginals: `(chains, time, states)`
- Pairwise transitions: `(target, source, source_state, target_state)`
- Influence: `(target, source)`
- Graph adjacency: `(source, target) = influence.T`

All public probability tensors are row-stochastic and use zero-based indices.
Reference computations default to `torch.float64` so tiny equality tests diagnose
algorithms rather than single-precision rounding.

## Approximation boundary

`influence.inference.exact_inference` enumerates `K**C` joint states and is the
trusted dynamic-programming oracle for small systems.

`influence.dong.dong_linear_inference` builds a `C*K` block matrix with source
state rows and target state columns. It performs the separately normalized
forward/backward recursions of Dong's linear formulation and returns the pairwise
sufficient statistics consumed by `fit_dong_linear_em`. It is exact for
independent chains but remains an approximation to the general joint posterior.

`influence.variational.factorized_inference` assumes chain-trajectory marginals
can be updated separately while other chains are temporarily fixed. This drops
posterior dependence between chains. It is an independently derived teaching
approximation and is **not** claimed to reproduce either nonlinear algorithm in
Dong's thesis. It is tested against exact inference in the independent-chain case.

The two approximations remain separate APIs. Only the Dong linear sufficient
statistics feed `influence.em`; the generic coordinate approximation has no
implied closed-form M-step.
