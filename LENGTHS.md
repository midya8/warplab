# Exact path-length atlas

The base `warplab-v1` certificate remains unchanged. This extension uses `warplab-lengths-v1`, embedding the full base certificate and adding every legal visited-cell-count stratum plus an explicitly penalized mixture. Underlying soft-DTW mathematics is credited in [CONTRACT.md](CONTRACT.md); this is an inspectable dynamic program and bounded verification experiment, not a new algorithm.

## Input and objective

```
{"alignment":{"x":[0,1,0],"y":[0,0,1,0],"gamma":0.6},"penalty":0.5}
```

`alignment` is an ordinary base request, including optional cost direction. Its existing bounds are unchanged. The optional `penalty` defaults to zero and must be a finite number in [-4,4]. Unknown keys, booleans as numbers, unsupported shapes and nonfinite values fail. The base cost direction is preserved in its embedded certificate but does not affect the length distribution.

A path visits both endpoints, and its length is the number of visited cells, not its edge count. Each visited cell pays its squared scalar cost plus penalty λ. Thus the penalized path cost is `base_cost + λ * length`. Negative λ rewards visited cells; positive λ charges them. This is a different objective, not an improvement to the alignment algorithm. No path is removed.

Legal lengths are exactly `max(n,m)..n+m-1`. For length ell, the numbers of down, right and diagonal steps are `ell-m`, `ell-n` and `n+m-1-ell`. The number of paths is therefore `(ell-1)! / ((ell-m)! (ell-n)! (n+m-1-ell)!)`. Tests check this formula against the length-resolved recurrence on all 49 admitted grid shapes.

## Length-resolved dynamic program

A prefix message F[i,j,k] is the log weight of paths ending at cell (i,j) after k visited cells, including the current cell's cost. A suffix B[i,j,k] uses k visited cells including the current cell. Missing states have log weight minus infinity; valid start/end single-cell states have log weight `-C[i,j]/g`.

```
F[i,j,k] = -C[i,j]/g + logsumexp(F[predecessors,k-1])
log Z_ell = F[n-1,m-1,ell]
S_ell = -g log Z_ell
```

For fixed ell, conditional occupancy convolves prefix length a and suffix length b with `a+b-1=ell`; the current-cell cost is restored once. Conditional edge flow convolves disjoint prefix/suffix cell sets with `a+b=ell`, without a current-cell correction. The local Gibbs law at fixed length does not depend on λ because every path in the stratum gets the same additive `λ*ell`.

Each stratum reports length, exact path count, log partition, unpenalized soft score, expected base cost, path entropy, temperature gradient, all conditional occupancies and joint edge flows, conditional sequence gradients, and exact-rational hard minimum/tie count/canonical path. Hard tie conventions match the base contract. Occupancy sums to ell; total edge flow is ell-1. The fixed-length soft score is not divided by the path count.

Length messages take O(n*m*K) time and space, with K=n+m-1. Reporting conditional cell/edge laws for every length requires length convolutions, O(n*m*K^2) work in this implementation. Rational/BigInt hard comparisons have operand-dependent arithmetic cost. Exhaustive verification remains exponential and capped by the base 8,989-path maximum.

## Penalized mixture and derivatives

```
log Z_lambda = logsumexp_ell(log Z_ell - lambda*ell/g)
P_lambda(length=ell) = exp(log Z_ell - lambda*ell/g - log Z_lambda)
S_lambda = -g log Z_lambda
```

The mixture reports this length law, expected base/penalized cost, expected length, length variance, total path entropy, length entropy, average conditional entropy, full mixture occupancies/edge flows and sequence gradients.

```
H(path) = H(length) + E_length[H(path | length)]
S_lambda = E[base cost] + lambda*E[length] - g*H(path)
dS_lambda/dlambda = E[length]
d2S_lambda/dlambda2 = -Var(length)/g <= 0
dS_lambda/dg = -H(path)
```

Thus expected length cannot increase when the penalty increases. Conditional sequence gradients are unchanged by λ; mixture sequence gradients are length-weighted conditional gradients. Zero penalty recovers the embedded base law. Neither the raw nor penalized score is a metric; both can be negative. Length-conditional occupancy is not a probability distribution over cells, and edge flow is joint probability rather than a conditional transition probability.

## Independent verification and reproduction

`length_oracle.py` does not use the length-resolved messages or the generator's mixture function. It enumerates complete paths, groups them by visited-cell count, centers each group's Gibbs weights, and directly sums cell/edge indicators. Hard minima/ties use exact rational sums of binary64 local costs. The penalized mixture is independently summed over complete penalized paths, not reconstructed through the generator's mixture code. The embedded base certificate is also independently verified. Generator-disabled tests preserve verification.

The complete report is compared with the same explicit numerical tolerance and integer strictness as the base contract. Binary64 rounding, concentrated-law entropy cancellation and probability underflow remain limitations, not silently omitted paths. Counts and legal strata remain present even when their Gibbs probability underflows to zero.

```
warplab lengths request.json -o atlas.json
warplab verify-lengths atlas.json
python -m warplab verify-lengths atlas.json
```

The 4 MiB transport cap, duplicate-key/nonfinite rejection and atomic output behavior apply unchanged. The installed wheel check runs both module and console commands outside the checkout for all eight base controls and all eight new length controls.

New tests compare complete length atlases to exhaustive paths, check all-grid closed-form counts, zero-penalty recovery, conditional conservation, entropy decomposition, penalty finite differences, coordinate/temperature derivatives, agreement with a separately shifted-cost kernel, penalty monotonicity, strict forgeries and generator independence. Cross-language tests compare 98 full Python/JavaScript atlases and independently verify both. Chrome exercises every control at 320/390/1280 pixels, every legal length card, row/column inspection, penalty changes, export and stale/error states.

## Attribution and limits

The monotone path model, Gibbs soft minimum and differentiable alignment come from [Cuturi & Blondel 2017](https://proceedings.mlr.press/v70/cuturi17a.html). Entropy and expected-alignment derivative context comes from [Blondel, Mensch & Vert 2021](https://proceedings.mlr.press/v130/blondel21a.html). The length grouping and penalty identities are explicitly derived above and checked independently, not claimed as novel. The base primary-paper caveat about ground-cost-specific self-corrected-divergence theorems remains unchanged. No correction, training, dataset benchmark, accuracy or speed claim is introduced.
