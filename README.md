# WarpLab

**Differentiable sequence alignment you can inspect, perturb, and independently verify.**

WarpLab implements established soft-DTW mathematics in dependency-free scalar Python and browser JavaScript. Stable forward/backward messages expose every cell occupancy and edge flow. Directional expectation messages give cost Hessian-vector products without an autodiff library. A separate exhaustive path oracle checks the complete report.

This is a bounded mathematical instrument, not a trained model, performance benchmark, new alignment algorithm, or production-scale library.

## What to inspect

- Raw soft-DTW, hard minimum, exact hard-tie count and a canonical minimum path.
- Gibbs probabilities for every monotone path, with a complete-path slider.
- Cell occupancies, joint edge flows, expected cost and expected visited-cell count.
- Sequence-coordinate gradients and the temperature derivative, equal to negative path entropy.
- Signed cost Hessian-vector products, checked against path-incidence covariance and finite differences.
- Strict versioned JSON certificates, atomic CLI output and an independent verifier.
- Responsive controls, accessible row/column selectors, layer switching and certificate export.

Run `python -m http.server 8000 --directory docs` and open `http://localhost:8000` for the inspector. All computation stays in the browser. There are no external model calls, keys, downloads or paid services.

## Reproduce

```bash
python -m pip install .
python -m unittest discover -s tests -v
npm test  # Node 22+ and Chrome, with CHROME_BIN if needed
bash tools/check_wheel.sh
```

The Python implementation has no runtime dependencies. Node uses built-in test, HTTP, child-process and WebSocket APIs. Browser tests need an installed Chrome; they fail rather than silently skip if it is missing.

```bash
printf '%s\n' '{"x":[0,1,0],"y":[0,0,1,0],"gamma":0.6}' > request.json
warplab analyze request.json -o certificate.json
warplab verify certificate.json
python -m warplab verify certificate.json
```

`analyze` and `verify` accept `-` for stdin. Inputs are capped at 4 MiB; duplicate keys, nonfinite values, booleans as numbers, unknown fields and unsupported shapes are rejected. Existing output survives failed requests. The wheel check installs in a fresh environment and runs module and console commands outside the checkout for all eight controls.

## Controls that expose the model

| Experiment | What it checks |
| --- | --- |
| Stretched pulse | Soft occupancy spreads beyond one hard minimum |
| Negative self-score | Three zero samples against themselves have 13 equal-cost paths, score `-log(13)` at temperature one, not zero |
| Sharp / diffuse alignment | The same two sequences at temperatures 0.05 and 8; descriptive, not an accuracy comparison |
| One legal path | Zero entropy, zero temperature derivative and zero cost Hessian |
| All 8,989 paths tie | The full 7 by 7 zero-cost path space; exact Delannoy count |
| Underflow stress | Large local costs at low temperature; centered/log-space calculations retain a finite report |
| Signed perturbation | Cost-direction curvature and signed occupancy response |

## Mathematical boundaries

Paths start at `(0, 0)`, end at `(n-1, m-1)`, include both endpoints and use right, down or diagonal moves. Local cost is `(x[i]-y[j])**2`, with no square root applied to the final score. Both sequence lengths are 1..7; sample values are -8..8; temperature is 0.05..8; cost-direction entries are -4..4. Larger inputs are rejected, not clipped.

Raw soft-DTW can be negative and is not a metric. Occupancy sums to expected path length, not one; it is not a normalized transport coupling. Edge values are joint traversal probabilities, not conditional transition probabilities. The Hessian product is in independent cost coordinates, not the sequence-coordinate Hessian. Floating-point underflow can round tiny probabilities to zero.

The hard control compares **exact rational sums of binary64 local cost entries**, then rounds the score for output. This avoids tolerance-defined ties and addition-order-dependent tie counts. It does not recover exact real-number input arithmetic.

The self-corrected divergence is deliberately absent. The 2021 paper's nonnegativity theorem depends on the ground cost; it does not prove the plain squared-Euclidean case used here. Broad statements in library/teaching docs do not replace that theorem's assumptions. See [CONTRACT.md](CONTRACT.md).

## Verification, not screenshots alone

Python tests compare dynamic messages to exhaustive paths, check conservation, soft-min bounds, transpose/scaling identities, coordinate and temperature finite differences, Hessian-vector finite differences and Hessian symmetry, singleton and maximum path spaces, forged certificates, input rejection and CLI atomicity. The verifier still works with generator functions disabled.

JavaScript tests compare 168 complete Python/JavaScript reports and pass both versions through the independent Python oracle. A second direct path check recovers all occupancies and edges from path probabilities. Chrome tests use the real page at 320, 390 and 1280 pixels, inspect cells, switch layers and complete paths, exercise row/column selectors, verify hard-path selection, export and stale/error clearing. CI runs Python 3.10, 3.11, 3.12 and 3.13 plus Node 22/Chrome.

## Exact path-length atlas

The second inspector, `docs/lengths.html`, separates the Gibbs law into every legal visited-cell count. Inspect conditional occupancies/edge flows, exact per-length path counts and hard minima, conditional sequence gradients and the entropy chain rule. An explicit per-cell penalty changes only the length mixture, not the conditional laws. Its score gradient is expected length and its curvature is negative length variance divided by temperature. This changes the objective, not algorithm quality.

`warplab lengths request.json -o atlas.json` generates the separate `warplab-lengths-v1` certificate; `warplab verify-lengths atlas.json` checks it by independently grouping complete paths and directly evaluating the penalized path law. The request wraps the base input as `alignment` and adds optional `penalty` in [-4,4]. Base certificate and bounds remain unchanged. See [LENGTHS.md](LENGTHS.md) for exact conventions and derivations.

## Sources and credit

- [Cuturi & Blondel, ICML 2017](https://proceedings.mlr.press/v70/cuturi17a.html), [paper](https://proceedings.mlr.press/v70/cuturi17a/cuturi17a.pdf): soft-DTW and differentiable expected alignments.
- [Blondel, Mensch & Vert, AISTATS 2021](https://proceedings.mlr.press/v130/blondel21a.html), [paper](https://proceedings.mlr.press/v130/blondel21a/blondel21a.pdf): entropy decomposition, Hessian products and divergence assumptions.
- [2017 author code](https://github.com/mblondel/soft-dtw/) and [2021 author code](https://github.com/google-research/soft-dtw-divergences): reference context. No code is copied from either implementation.
- [Romain Tavenard's notes](https://rtavenar.github.io/ml4ts_ensai/contents/align/softdtw.html) and [tslearn documentation](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.SoftDTWLossPyTorch.html): teaching/library context, with the primary-paper caveat above.

Original implementation and experiment interface by Midya Rahmani. MIT license. Synthetic examples only.
