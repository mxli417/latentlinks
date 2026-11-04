"""Small dependency-free timing comparison for reference and tensorized paths."""

from __future__ import annotations

from collections.abc import Callable
from statistics import median
from time import perf_counter

import torch

from latentlinks.hmm import (
    CategoricalHMM,
    forward_backward,
    forward_backward_vectorized,
)
from latentlinks.influence import (
    InfluenceModel,
    dong_linear_inference,
    dong_linear_inference_vectorized,
)


def benchmark(function: Callable[[], object], repeats: int = 5) -> float:
    function()  # warm-up
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return median(samples)


def report(name: str, reference: Callable[[], object], optimized: Callable[[], object]) -> None:
    reference_seconds = benchmark(reference)
    optimized_seconds = benchmark(optimized)
    speedup = reference_seconds / optimized_seconds
    print(
        f"{name:20s} reference={reference_seconds:.6f}s "
        f"vectorized={optimized_seconds:.6f}s speedup={speedup:.2f}x"
    )


def main() -> None:
    hmm = CategoricalHMM.random(12, 5, seed=20)
    hmm_observations, _ = hmm.sample(400, seed=21)

    generator = torch.Generator().manual_seed(22)
    influence = InfluenceModel(
        torch.rand(8, 3, generator=generator),
        torch.rand(8, 8, 3, 3, generator=generator),
        torch.rand(8, 3, 4, generator=generator),
        torch.rand(8, 8, generator=generator),
    )
    influence_observations, _ = influence.sample(300, seed=23)

    report(
        "HMM forward-backward",
        lambda: forward_backward(hmm, hmm_observations),
        lambda: forward_backward_vectorized(hmm, hmm_observations),
    )
    report(
        "Dong linear",
        lambda: dong_linear_inference(influence, influence_observations),
        lambda: dong_linear_inference_vectorized(influence, influence_observations),
    )


if __name__ == "__main__":
    main()
