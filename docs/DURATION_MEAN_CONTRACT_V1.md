# Duration forcing point-mean computation contract v1

Frozen before implementation on 23 October 2026. Parent is
`5d44fc6d76ef3517dbe5e354f9943d76511a048e`. The original exhaustive
`affine_rollout` and `evaluate_rollout` remain unchanged independent references.
This is a separate computation API for the same expected affine point targets;
it changes no forcing law, A operator, training protocol, actual fit or result.

`mean_affine_rollout` accepts the exhaustive API's arguments. Check the immutable
model, loaded dependencies, exact current identity, issue/model/checkpoint
boundaries, known age, two-mode checkpoint shape and A finite/PSD covariance
semantics, two-by-two historical basis, horizon 1:6, nonnegative lambda/sigma,
fixed finite pressure and explicit nonnegative target observation variance.
No future data are read. Unsupported support on any positive mass returns the
same unavailable reason; no probability mass is dropped or renormalized.

The exact distribution uses keys `(state, capped_age, mark_id)`. North has id0;
new South marks have id1:4, kept separate even if their centroids coincide; the
observed initial South q has id5 even if it equals a centroid. South stays keep
their mark identity; flips reset it. Merge only identical keys. Exact
`Rational{BigInt}` masses and exact rational conversions of supplied Float64 q
give `qmean[h] = sum(mass * q)`. Every depth has exact mass one.

For the already thresholded q and fixed A coefficients, linearity gives

`E[B_end[j,k]] = phi[j] E[B_prev[j,k]] - (j==k) u[j] E[q[h]]`,

`E[C[h,k]] = sum(c[j] E[B_prev[j,k]]) - v[k] E[q[h]]`.

Therefore marginal expected q at each hour suffices for point means, despite
temporal dependence of the forcing law. Averaging raw wind before the nonlinear
injection threshold does not obey this identity. Mixture variance cannot be
recovered from these marginal means and is never returned or inferred.

Return immutable exact state distributions/q expectations, zero-forcing interval
offsets, expected gain coefficients and expected endpoint gain bases. Use A's
unrounded precise coefficients and per-hour Float64 rounding for zero-forcing
base means. Use the exhaustive B public one-hour Float64 coefficients for gain
bases, with wide intermediate expectation arithmetic. Existing tolerances remain
`atol=5e-11, rtol=5e-14` on the declared synthetic domain of the B contract.
Nonfinite means/bases reject rather than clip. `evaluate_mean_rollout` accepts
two finite nonnegative gains and returns point means and endpoint point means.

Every available result names `computation_scope=:point_mean_only`,
`measured_receipts_available=false`, and `claim_ready=false`. It has no variance,
interval, covariance or mixture-component fields. Sigma/observation variance
are validated for interface consistency; no future covariance is propagated.
An arithmetic overflow specific to future covariance therefore does not block
an otherwise finite point mean. This is an explicitly narrower arithmetic
domain than a full probabilistic rollout, not a probabilistic qualification.

Pre-code independent Fraction full-tree oracles freeze nonconstant hazards,
beta5/20, N/S starts, ages2/48/49, h1:6, duplicate centroid IDs with unequal
masses, exact q expectations/state distributions and Brownian affine targets.
Focused tests compare unchanged exhaustive B for all9 tau/sigma corners,
Brownian limits, historical basis/pressure/checkpoint extremes, support failures,
identity/boundary errors, save/reload/as-of causality and synthetic NNLS gain fit.
Require mutation sensitivity, Julia1.12.6/1.13.1, strict docs, and a bounded warmed
BenchmarkTools CPU/allocation profile including at least1000 timing samples for
sub-millisecond claims. Full package/experiment/harness gates belong to root.

No actual data, actual gain fit, new outcomes, network/serving changes, paper
claims or expected forecast-skill advantage are authorized by this component.
