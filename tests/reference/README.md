# Historical reference fixtures

This directory is the provenance boundary between the independent Python code
and historical implementations.

No MATLAB source code is included in Latent Links. The package is derived from
published mathematical descriptions. The MIT Influence Model toolbox may be run
separately as an external behavioral oracle, and only minimal numerical outputs
may be recorded here.

## Currently recorded evidence

The directory now contains one MATLAB-generated numerical fixture in addition
to parameters printed on the public MIT usage page:

- source: <https://vismod.media.mit.edu/vismod/demos/influence-model/software-usage.htm>
- retrieved: 2026-10-03
- toolbox version: not stated on the usage page
- experiment: six chains, two states, two observations, length 1000
- pairwise transition: `[[0.99, 0.01], [0.08, 0.92]]`
- categorical emission: `[[0.9, 0.1], [0.1, 0.9]]`
- influence: equal weights within blocks 1–3 and 4–6; zero across blocks

`tests/test_reference_fixtures.py` checks that our independently constructed
experiment uses exactly those documented parameters. It does not claim matching
learned outputs or random samples.

## Prepared external-oracle run

`generate_mit_fixture.m` is independently authored glue code: it defines a fixed
two-chain input, invokes the public toolbox API, and writes only `gamma`, `xi`,
and the linear log likelihood. It contains no toolbox implementation.

- archive: <https://vismod.media.mit.edu/vismod/demos/influence-model/influence.zip>
- retrieved: 2026-10-06
- SHA-256: `6B1F2C1613FAAEF8F326A533DA326D4F3F8EA0BF41DB77DB88796AE020B2C3F0`
- MATLAB attempted: R2016b on Windows
- toolbox version: unstated by the publisher
- deterministic input: fully recorded in `generate_mit_fixture.m`; no RNG output is used
- intended Python comparator: `dong_linear_inference`, tolerance `1e-12`

The fixture `mit_toolbox_linear_tiny.json` was generated on 2026-10-07 with
MATLAB R2016b using an activated student license. It records the complete fixed
input plus the toolbox's linear log likelihood, one-slice `gamma`, and aggregated
two-slice `xi`. `tests/test_reference_fixtures.py` independently reconstructs
the Python model, converts the MATLAB source/target orientations explicitly, and
requires agreement to absolute tolerance `1e-12`.

To reproduce it, set `LATENTLINKS_MIT_TOOLBOX` to an external extraction and
`LATENTLINKS_FIXTURE_OUT` to the JSON destination, then execute
`generate_mit_fixture.m`. The external archive and all historical source remain
outside this repository.

## Adding a fixture

Every future fixture must record:

1. toolbox archive URL and version or retrieval date;
2. MATLAB and toolbox versions used for execution;
3. complete input parameters and random seed, if reproducible;
4. the command used to produce the quantity;
5. the smallest useful numerical output;
6. tolerance and the Python algorithm expected to match it;
7. confirmation that no historical implementation is included.
