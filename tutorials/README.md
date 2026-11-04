# Learning path

The notebooks are meant to be read in order. Each chapter starts from a question,
states its assumptions, connects an equation to named tensors, and ends with a
check or an explicit open boundary.

1. HMM generative model
2. Exhaustive likelihood and the forward recursion
3. Backward messages, `gamma`, and `xi`
4. Baum–Welch as expected counting
5. Autograd after inference
6. Ordinary HMMs, coupled HMMs, and state explosion
7. The Influence Model
8. A factorized marginal approximation
9. Dong's linear marginal-state inference and EM
10. Public MIT six-chain experiment
11. Synthetic block-network recovery
12. Rebuilding every HMM update from equations and a calculator
13. HMMs and Influence Models as generative models

Run them with `uv sync --group tutorials` followed by `uv run jupyter lab`.
