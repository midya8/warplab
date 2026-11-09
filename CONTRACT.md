# The WarpLab mathematical contract

## Path space and score

A legal path is a list of grid cells from `(0,0)` to `(n-1,m-1)`. Each step is `(0,1)`, `(1,0)` or `(1,1)`. Both endpoints count toward cost. Every cell can occur at most once. No band, skip, blank symbol, sample normalization, transport marginal constraint or path-length normalization is used.

For local cost `C[i,j] = (x[i]-y[j])^2` and path incidence matrix `A`, write `L(A) = <A,C>` and temperature `g > 0`:

```
Z = sum_A exp(-L(A)/g)
S(C,g) = -g log Z
p(A) = exp(-L(A)/g) / Z
```

There is no square root. The path count is `Delannoy(n-1,m-1)`. The largest admitted 7 by 7 grid has 8,989 paths. Both sequences have 1..7 samples, each in [-8,8]. Temperature is [.05,8]. Optional independent cost direction V has shape n by m and entries in [-4,4]; default V[i,j] = (-1)^(i+j). Nonfinite numbers, booleans, unknown keys and unsupported requests fail.

## Stable messages

Prefix F includes the current cell's cost. Suffix B also includes the current cost. Their log-weights obey:

```
F[i,j] = -C[i,j]/g + logsumexp(F[predecessors])
B[i,j] = -C[i,j]/g + logsumexp(B[successors])
```

The start-prefix and terminal-suffix have a virtual predecessor/successor of log-weight zero. Missing neighbors contribute nothing. The terminal prefix is log Z. Max subtraction stabilizes every logsumexp.

```
occupancy[i,j] = exp(F[i,j] + B[i,j] + C[i,j]/g - log Z)
edge_joint[(i,j),(a,b)] = exp(F[i,j] + B[a,b] - log Z)
```

The current-cell term is restored once in occupancy because both messages included it. An edge connects disjoint prefix/suffix cell sets and needs no correction. Exponents are capped above at zero to remove tiny positive rounding excursions, not to truncate the path space.

Expected cost is `<occupancy,C>`; expected path length is the occupancy sum. Incoming and outgoing edge sums equal occupancy except at the start/terminal. Total edge flow is expected length minus one. Individual values are probabilities; the full occupancy matrix is not a probability distribution over cells and does not satisfy transport marginal constraints.

## First and second derivatives

```
dS/dC[i,j] = P(A visits (i,j))
dS/dx[i] = 2 sum_j occupancy[i,j] (x[i]-y[j])
dS/dy[j] = 2 sum_i occupancy[i,j] (y[j]-x[i])
H(path) = -sum_A p(A) log p(A)
S = E[L] - g H(path)
dS/dg = -H(path)
```

The cost Hessian is negative semidefinite:

```
H_C = -Cov(vec(A)) / g
(H_C V)[i,j] = -Cov(A[i,j], <A,V>) / g
<V,H_C V> = -Var(<A,V>) / g <= 0
```

The engine propagates prefix and suffix expectations of `<A,V>` using the same local Gibbs mixture weights. Conditional on visiting a cell, their sum minus V at that cell is the full-path directional expectation. Subtract the unconditional directional expectation and multiply by `-occupancy/g` to get the covariance product. No full Hessian is materialized. This is the **cost** Hessian, not the sequence-coordinate Hessian. The covariance identity and differentiable-DP method are established mathematics, not an algorithmic novelty claim.

The main messages, occupancies, edges and directional products are O(nm) time and space. Hard controls also use grid dynamic programming, but exact rational/BigInt comparisons have operand-dependent arithmetic cost. Browser complete-path enumeration and the exhaustive verifier are exponential and are intentionally bounded by 8,989 paths.

## Hard control and ties

The hard control minimizes the exact rational sum of the already-computed binary64 local costs. Python uses Fraction; JavaScript converts the nonnegative binary64 values into integers at a common binary exponent. All exact minimum paths are counted. The canonical path is lexicographically first by successive `(row,column)` pairs. The emitted hard score is rounded to binary64. The hard calculation is not differentiated. It is not a fuzzy-tolerance tie count and not exact arithmetic on the ideal real-valued input sequences.

## Independent verification

`warplab.oracle` shares request validation but not the dynamic program, forward/backward messages, tie recurrence or covariance implementation. It lists every complete path, computes exact rational costs for hard ties, centers Gibbs exponentials at the minimum cost, sums incidence and edge indicators directly, computes entropy from `-p log p`, and computes the Hessian product from centered path-direction values. It reconstructs the complete report and compares every key and value. Numeric comparisons allow relative/absolute scale tolerance `2e-7 * max(1,|expected|,|actual|)`; integer counts and cell indices remain strict integers. Version and key sets are exact. This is numerical verification, not a formal proof or adversarial cryptographic certificate.

Tests separately perturb sequence coordinates, temperature and independent cost entries; check transpose, scale, soft-min bounds, flow conservation and Hessian symmetry; and disable generator functions while verifying. All eight controls are reproduced through an installed wheel outside the source tree. Browser JSON exports contain the same report shape.

## Numerical limits

Binary64 is used, so extremely small probabilities can underflow to zero. The entropy identity can suffer cancellation for concentrated distributions; a tiny negative computed entropy is clamped to zero. Curvature may have tiny roundoff residuals. Bounds and conservation are numerical, not bit-exact identities. Underflow stress controls keep all reported fields finite. Input values are not clipped or silently shortened.

## Source conflict: do not borrow a theorem without its cost

Raw soft-DTW may be negative and is not a metric. Constant zero sequences have all paths at zero cost, hence S = -g log(path count). In particular, three zeros against themselves have 13 paths and score -log(13) at g=1. Occupancy's sum changes with the path-length distribution.

A common correction subtracts one half of each sequence's self-score. This project does not implement it. In [Blondel, Mensch & Vert 2021, Proposition 3](https://proceedings.mlr.press/v130/blondel21a/blondel21a.pdf), the valid-divergence theorem covers their special cost (11) or one-dimensional absolute cost (12). For ordinary squared Euclidean cost (1), Proposition 4 proves stationarity at equality; the paper describes nonnegativity as a conjecture. The paper's cost (1) includes a factor one half, whereas WarpLab uses the full square. Multiplying the square changes scaling, not the missing theorem assumption. Current teaching/tslearn pages make broader claims; WarpLab preserves the primary paper's narrower statement and makes no claim that the conjecture is now proved or disproved.

## Scope and attribution

This is a short scalar-sequence educational instrument. It has no training data, external dataset, learned predictor, classification metric, accuracy benchmark, causal ablation, speed claim or production deployment claim. Sharp/diffuse controls only illustrate two temperatures on identical supplied sequences. No private data or paid API is involved. The implementation is original; the algorithms and identities are credited to [Cuturi & Blondel 2017](https://proceedings.mlr.press/v70/cuturi17a.html) and [Blondel, Mensch & Vert 2021](https://proceedings.mlr.press/v130/blondel21a.html). The inspector is different from CTC alignment (no label/blank emission model) and optimal transport (no coupling marginal constraints).
