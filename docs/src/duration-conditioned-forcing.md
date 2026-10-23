# Duration-conditioned forcing research

`SolarSINDy.DurationConditionedForcing` implements a separately named research
component for a marked South/North forcing continuation. It does not change the
original interval-mean A experiment, deploy a forecaster or establish skill or
novelty. The fixed empirical edge and mark conventions are recorded in
`docs/DURATION_FORCING_CONTRACT.md`; the literal/rational expectations were frozen
before implementation in `test/duration_forcing/precode_oracles_v1.json`.

Fit input consists of tagged hourly Bz and nonnegative thresholded injection q,
with explicit completion and assumed receipt clocks. Only known next-hour
transitions mature before the fitting boundary minus168 hours contribute risk.
Gaps, missing hours, initially unknown ages and right censoring remain explicit.
Empty cells shrink to the training-only state flip rate. Missing global or
required first-S mark support produces unavailable forecasts.

The forecast keeps every positive-mass path and duplicate mark identity through
at most six future hours, with exact rational probability mass. It shares A's
zero-forcing state/covariance calculation by depth and carries each branch's
two-gain affine basis. The returned mixture separates conditional and
between-forcing variance. Target observation variance is a required argument;
pressure and fitted parameters remain fixed, and calibration is unqualified.

Models use immutable counts and marks, a training-prefix digest and implementation
identity. Publication writes bounded data-only JSON to a new directory with a
manifest and completion marker; loading requires the caller's expected manifest
digest. These controls detect changed bytes and incompatible semantics. They do
not constitute measured historical receipt evidence or an external timestamp.
The combined known-next risk count is bounded by499998 for the500000-hour input
limit: the first unknown-age row and final right-censored row supply no risk.
This necessary aggregate bound does not prove chronology or source provenance.

For repeated point-model design evaluation, `mean_affine_rollout` propagates
exact rational masses over distinct state/age/mark identities and computes the
expected already thresholded injection at each future hour. A's linear affine
response then gives the same expected hourly point targets without retaining
every chronological path. Duplicate centroid identities and the observed initial
South injection identity remain separate. Any unsupported positive-probability
state or mark makes the result unavailable; no mass is removed or renormalized.
`evaluate_mean_rollout` evaluates these expected designs at fixed gains.

The mean-only result has `computation_scope=:point_mean_only` and returns no
variance or probabilistic interval. It validates the supplied checkpoint with A's
finite/positive-semidefinite covariance semantics and validates sigma and target
observation variance, but does not propagate future covariance. Its finite-mean
arithmetic domain can therefore extend beyond a full mixture's covariance range.
Use the exhaustive API when conditional or forcing uncertainty is required.
The pre-code derivation is in `docs/DURATION_MEAN_CONTRACT_V1.md`.

The exhaustive evaluator preserves the original tuple paths/components, field
keys/order, values, caller metadata and repeated-evaluation behavior. Its returned
NamedTuple deliberately widens its field types to `Any`, avoiding compilation
growth from embedding the number of branches in every caller's inferred result
type. The affine rollout's return and numerical recurrences remain unchanged.
The resource contract and frozen numerical reference are described in
`docs/DURATION_EVALUATOR_RESOURCE_CONTRACT.md`. These computation checks establish
no forecast-skill or novelty result.

The synthetic example is `examples/duration_forcing_synthetic.jl`. It learns a
small duration model, saves and reloads it, and checks a hand-calculated
two-hour Gaussian mixture through the A interval kernel. Actual-data execution,
matched causal baselines, full software qualification, resource measurements
and independent forecasting evidence remain separate gates.

```@docs
SolarSINDy.DurationConditionedForcing
SolarSINDy.DurationConditionedForcing.WindHour
SolarSINDy.DurationConditionedForcing.DurationModel
SolarSINDy.DurationConditionedForcing.fit_duration
SolarSINDy.DurationConditionedForcing.current_state
SolarSINDy.DurationConditionedForcing.hazard
SolarSINDy.DurationConditionedForcing.affine_rollout
SolarSINDy.DurationConditionedForcing.evaluate_rollout
SolarSINDy.DurationConditionedForcing.mean_affine_rollout
SolarSINDy.DurationConditionedForcing.evaluate_mean_rollout
SolarSINDy.DurationConditionedForcing.save_model
SolarSINDy.DurationConditionedForcing.load_model
SolarSINDy.DurationConditionedForcing.source_identity
```
