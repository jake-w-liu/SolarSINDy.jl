# Completed-source dB/dt research replay

`SolarSINDy.CompletedSourceDBDT` prepares issue features before admitting future
labels and exposes synthetic static, phase, FIFO and maximum-window replay paths.
It uses the sibling `CompletedMinuteReplay` clock convention: a default whole
minute issue uses the preceding completed ground center; the next30 rate centers
form the target. The final center is ready45 seconds later, giving a default
sufficient whole-minute phase period of31 minutes.

These are retrospective assumed-availability research tools. Their records state
`availability_kind=:assumed_completed_source_intervals` and
`measured_receipts_available=false`. They do not establish historical receipt
vintage, observational forecast skill or a live serving policy.

```@example completed_source_dbdt
using Dates, SolarSINDy
using SolarSINDy.CompletedSourceDBDT
issue = DateTime(2024, 1, 1)
ground = Dict(issue + Minute(k) => (Float64(k*k), 0.0) for k in -40:31)
rows = prepare_schedule(ground, [issue])
@assert only(rows).source_row.dbdt_now == 3.0
@assert only(rows).source_row.dbdt_mean30 == 32.0
@assert only(rows).source_row.dbdt_max30 == 61.0
prediction = bind_predictions(rows, X -> zeros(size(X, 1)))
@assert only(case_audit(rows, ground; asof=issue+Minute(31))).target == 59.0
calibration = CompletedCalibration(zeros(9), fill(issue-Millisecond(1), 9);
    model_schema="dbdt_completed_source_point_v1",
    point_model_sha256=repeat("1", 64), purpose=:synthetic)
run = replay_phase(rows, prediction.centers, prediction.scales,
                   Dict(issue=>59.0), calibration; gamma=0.1, asof=issue+Minute(31))
@assert run.phase_period_minutes == 31
@assert run.certificate.issued == run.certificate.resolved == run.certificate.misses == 1
@assert !run.measured_receipts_available
nothing # hide
```

Ground dictionaries contain `(X,Y)` vectors in nT at centered minute tags. The
adapter derives rates in nT/min from adjacent vectors. `mode=:ground11` uses the
four completed ground-rate summaries, their transforms and the actual issue
calendar; it never consults wind. `mode=:mixed8` additionally requires the admitted
wind dictionary with `(V,Bz,By,n)` rows in km/s, nT, nT and cm^-3. A missing target
or wind value cannot silently filter the ground-only issue schedule.

`prepare_schedule` retains feature rejection reasons. `bind_predictions` calls a
provided predictor only on accepted features and fixes the positive scale
`1+log1p(max30)`. `label_audit` and `case_audit` take an explicit `asof` and check
readiness before source access. `split_eligible` uses a strict label-readiness
boundary. Numeric inputs must remain finite in Float64; nonzero-to-zero
underflow is rejected while signed zeros and representable subnormals remain
supported.

`CompletedCalibration` retains chronological and sorted signed scores, a point
model SHA256, policy and layout. It requires the completed-source schemas and
strict readiness before stream start. `purpose=:refitted` is a declaration; the
constructor does not establish source-bound fit/export provenance. The internal
feature/model reference is a local raw-byte hash-pinned copy. Its legacy raw-tag
case builder is excluded from this namespace's public bindings and exports.

`static_bounds`, `replay_phase` and `replay_fifo` use an assumed complete-label
dictionary. Missing ready labels are censored and remain unknown in the gap
certificate. They do not implement late receipt handling. `replay_receipts` and
`CompletedMaximumController` instead consume explicit assumed receipt batches,
validate a complete same-time batch before mutation, and process receipts before
issues at the same time. Receipt grace belongs to that separate path. Only early
mode can apply a proven miss before the full maximum is known; a completed score
always requires all30 valid rates and label readiness. Expiry is strictly after
the scheduled feedback clock plus grace. Diagnostic flushes preserve issued
bounds. The child `CompletedLabelEvents` module provides a common synthetic
completion-event path for fixed consumers. These original static-map routines
retain their complete-label assumptions; matched actual-data comparisons still
require source-bound refits and the remaining qualification gates.

The standalone synthetic example is `examples/completed_source_dbdt_synthetic.jl`;
it is also registered in the package experiment entry. The polynomial example
and focused tests validate software contracts, not an actual refit or scientific
result. Full package, experiment, harness and native cross-platform qualification
remain separate required gates.

```@docs
SolarSINDy.CompletedSourceDBDT
SolarSINDy.CompletedSourceDBDT.PreparedIssue
SolarSINDy.CompletedSourceDBDT.prepare_issue
SolarSINDy.CompletedSourceDBDT.prepare_schedule
SolarSINDy.CompletedSourceDBDT.label_audit
SolarSINDy.CompletedSourceDBDT.case_audit
SolarSINDy.CompletedSourceDBDT.split_eligible
SolarSINDy.CompletedSourceDBDT.bind_predictions
SolarSINDy.CompletedSourceDBDT.CompletedCalibration
SolarSINDy.CompletedSourceDBDT.static_bounds
SolarSINDy.CompletedSourceDBDT.replay_phase
SolarSINDy.CompletedSourceDBDT.replay_fifo
SolarSINDy.CompletedSourceDBDT.CompletedMaximumController
SolarSINDy.CompletedSourceDBDT.issue!
SolarSINDy.CompletedSourceDBDT.receive!
SolarSINDy.CompletedSourceDBDT.advance!
SolarSINDy.CompletedSourceDBDT.certificate
SolarSINDy.CompletedSourceDBDT.materialize_receipts
SolarSINDy.CompletedSourceDBDT.replay_receipts
```

## Immutable legacy reference

The following helper belongs to the internal byte-pinned reference library. It
uses archive-tag indices for histories and filters rows by complete future
targets. It is not the completed-source cohort builder and is not re-exported
from `CompletedSourceDBDT`. Use
[`prepare_schedule`](@ref SolarSINDy.CompletedSourceDBDT.prepare_schedule) to
retain feature-only issue eligibility, then
[`case_audit`](@ref SolarSINDy.CompletedSourceDBDT.case_audit) for explicit
readiness-gated labels.

```@docs
SolarSINDy.CompletedSourceDBDT.DBDTCausalForecaster.ground_cases
```
