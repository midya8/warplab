# Invert the expected-length response

WarpLab's `warplab-inverse-v1` report asks a bounded inverse question: which per-visited-cell penalty in [-4,4] gives a requested expected alignment length? It uses the established Gibbs length law in [LENGTHS.md](LENGTHS.md), not a learned calibration model or a new algorithm. Every visited cell, including both endpoints, pays the penalty.

## Input and fixed numerical policy

```json
{"alignment":{"x":[0,0,0],"y":[0,0,0],"gamma":1},"target":4}
```

Exactly `alignment` and `target` are required. The alignment has the unchanged base bounds (scalar sequences of length 1..7, values -8..8, temperature .05..8, optional cost direction -4..4). Target is a finite number in [1,13]. Booleans, unknown fields and nonfinite values fail. A target need not be physically or finitely reachable: that is an explicit result, not an input error. Cost direction is preserved in the normalized request but does not affect length probabilities.

The fixed penalty endpoints are -4 and +4. Absolute mean tolerance is 1e-9. The pre-update penalty-width tolerance is 1e-9, with at most 40 midpoint evaluations. These are numerical tolerances, not rigorous interval bounds.

## Forward response and monotonicity

Compute the length partitions once. At penalty lambda, tilt each length's unpenalized log partition by `-lambda*length/gamma`, normalize with a max-shift, and compute mean and centered variance.

```
P_lambda(length=k) proportional to Z_k exp(-lambda*k/gamma)
d E_lambda[length] / d lambda = -Var_lambda(length) / gamma
```

This derivative follows by differentiating the finite Gibbs sum. In exact arithmetic with positive temperature, every legal path has positive weight. More than one legal length gives positive variance and a strictly decreasing response. One legal length gives constant response and no identifiable penalty. Underflow and rounding can make a multi-length floating-point response appear constant; the implementation does not promote that to mathematical unidentifiability.

A finite penalty need not reach the shortest or longest legal length exactly. The finite endpoint response range, not just the integer support range, determines bounded reachability. Raising the penalty changes the objective; it is not a measured alignment-quality improvement.

## Status and witness

- `unidentifiable`: only one legal length and target matches within mean tolerance. `selected` and `residual` are null because there is no identified penalty.
- `target_above_range` / `target_below_range`: outside the finite endpoint mean range, allowing the stated mean tolerance. The selected point is the nearest evaluated endpoint, not a solution.
- `numerically_flat`: more than one legal length, target is within the numerical endpoint range, but endpoint means differ by at most mean tolerance. The selected point is a diagnostic sample, not a unique estimate.
- `lower_endpoint` / `upper_endpoint`: the corresponding finite endpoint matches within mean tolerance. No exact or unique-parameter proof is implied.
- `residual_tolerance`: a midpoint's absolute mean residual is at most 1e-9.
- `bracket_width`: the pre-update bracket is at most 1e-9 wide. This does not promise a 1e-9 mean residual; inspect the actual residual.

Endpoint range checks precede flat/endpoint matches. The single-legal-length test precedes all multi-length classifications. If both endpoint residuals are within tolerance but the range is not numerically flat, the lower endpoint wins. Bisection records the full pre-update bracket, midpoint penalty, mean, centered variance, derivative and every length probability. Mean above target raises the lower penalty; mean below target lowers the upper penalty. For generated nonempty traces the fixed width policy forces termination within the 40-step cap.

Among endpoints and midpoints, select the smallest absolute residual, breaking exact numerical ties by smaller penalty. The selected point can differ from the last midpoint. Report the actual residual, never silently equate a width stop with a residual stop. A floating-point report is not an interval-arithmetic certificate.

## Independent verifier

`verify-inverse` does not call the inverse generator, length dynamic program or its tilt response routine. It enumerates all complete monotone paths, sums exact rational binary64 local costs before conversion, and directly weights full path energies at every witness penalty. It reconstructs expected lengths, variances, length probabilities, endpoint status, bracket decisions, stopping rules, selected response and residual. Structural fields must agree and numbers use the existing exhaustive verifier tolerance. Forged statuses, truncated/extended traces, altered brackets, derivatives and extra fields fail.

Numerical replay can disagree for adversarial targets within floating-point rounding of a branch/stop boundary. Such a report fails verification rather than being silently accepted. The verifier is independent numerical evidence, not a formal real-arithmetic proof. Complete-path enumeration is exponential and bounded to at most 8,989 paths.

The forward partitions and conditional laws use the existing length implementation. Once obtained, each response evaluation is O(K) arithmetic for K legal lengths; at most 42 response points are evaluated. The current solver calls the full length atlas, so total preprocessing includes its O(nmM²) conditional-law work for M=n+m-1; it does not claim an optimized partition-only solver. Trace storage is O(40K). Independent verification is proportional to full path count times the number of evaluated points, plus path-energy construction.

## Reproduce

```sh
python -m unittest discover -s tests -v
npm test
bash tools/check_wheel.sh
python -m warplab inverse request.json -o witness.json
python -m warplab verify-inverse witness.json
```

The CLI retains the 4 MiB input cap, duplicate-key checks, nonfinite rejection and atomic output. Controls include reachable interior targets, both directions of finite-range failure, single-length unidentifiability, a single-length impossible target, the 7 by 7 zero-cost grid, and a low-temperature numerically flat response. Python tests include endpoint matches, random interior targets, finite-difference derivatives, monotonicity, generator-disabled verification and forgeries. Browser tests inspect every evaluated witness at 320/390/1280, distributions, trace selection, export, stale/error clearing and responsive layout.

Underlying soft-DTW mathematics is credited to [Cuturi & Blondel (2017)](https://proceedings.mlr.press/v70/cuturi17a.html) and [Blondel, Mensch & Vert (2021)](https://proceedings.mlr.press/v130/blondel21a.html). Bisection and finite exponential-family differentiation are standard. No learned model, dataset/speed/accuracy benchmark, new algorithm, self-corrected divergence or ground-cost-independent theorem is claimed. Raw/penalized soft scores are not metrics.
