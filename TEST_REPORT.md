# V2.2 Research Test Report

## Coverage

The focused suite contains 1,753 assertions across the V2.2 research surfaces:

| Surface | Assertions |
|---|---:|
| Constrained SINDy-dominant stack | 51 |
| Sparse residual | 52 |
| Portable boosted residual | 76 |
| Causal served replay | 100 |
| Leakage-safe primary cross-fit | 63 |
| Causal sparse-history kernel | 100 |
| Purged M1 cross-fit helpers | 12 |
| Prospective L1 receipt collector | 322 |
| Explicit-only V2.2 collector launchd integration | 67 |
| Prospective half-hour issue and Dst receipt capture | 227 |
| Offline L1 issue pairing | 112 |
| Receipt-causal M2 arrival queue and ungated sparse candidate | 151 |
| Low-level M2 matrix-to-core wrapper | 26 |
| Combined-mechanism recoverability helpers | 17 |
| Stable group-sparse M2 driver kernel | 52 |
| Causal AR-only M3 error-state control | 70 |
| Full exogenous M3 error-state candidate | 106 |
| Checksum-bound V2.2 shadow chain | 149 |

The tests cover constraint projection, pooled fallback, synthetic recovery,
causal feature construction, split embargoes, target maturity, lag and horizon
semantics, exact live-kernel identity, deterministic fitting, portable tree
inference, checksummed artifact round trips, and corruption rejection.

## Independent expectations

- Mixture forecasts are checked against hand-computed convex combinations and
  an exact SINDy-family weight floor.
- Synthetic tables recover known weights and known sparse supports.
- Extracted flat-tree inference is checked against EvoTrees CPU predictions,
  including values exactly on split thresholds.
- Replay predictions are checked against the live V2.1 kernel at all supported
  model steps.
- Cross-fit rows are checked for exact key coverage, whole anchors, and a 168 h
  target-to-evaluation-block gap.
- Sparse-history one-step and multi-step trajectories are checked against hand
  calculations, exact zero-augmentation continuity, all-support synthetic
  recovery, stability bounds, and artifact mutation.
- Receipt capture is checked against deterministic clocks and responses,
  content hashes, complete chain traversal, source/URL identity, chronology,
  rollback, orphaned and missing records, intermediate symbolic links,
  transport failures, strict JSON parsing, exact NOAA `source`/Boolean `active`
  extraction, issue-causal ephemeris receipt order, exact and linearly
  interpolated GSE positions, gap/extrapolation fail-closure, ephemeris
  transport failures, DSCOVR normal/suspect/error quality semantics, required
  mag/wind field gates, ACE and malformed-quality fail-closure, and
  metadata-forgery rejection after independent receipt rehashing.
- Offline pairing is checked against hand-written same-minute mag, wind, and
  orbit rows. Its oracles cover immutable checksummed issue cutoffs, exact-prefix
  replay, invariance to corruption of later records, raw bodies, and latest
  pointers, latest exact common timestamps, conflicting latest and historical
  revisions, missing GSE Vx, position disagreement, raw corruption, all v2
  product/unit/frame/quality fields, returned hashes, collector-lock exclusion,
  receipt-tree identity, and lock cleanup.
- Prospective issue capture is checked with synthetic clocks and HTTP responses.
  The oracles cover exact half-hour scheduling, raw Dst response and header
  identity, first receipt and revision lineage, causal Dst-anchor selection,
  immutable L1 and Dst cutoffs, exact L1-pair binding or explicit
  unavailability, post-issue-object invariance, five-minute
  commitment failure, crash recovery, pending/completion guards, chain heads,
  and the absence of every numeric forecast field while the fitted gate is
  unavailable.
- The receipt-causal M2 queue is checked against hand-derived GSE ballistic
  delays, the exact trailing-window endpoints, half-open UTC bins, physical
  medians, the 20/120 min delay boundaries, the exact 90 min freshness
  boundary, one isolated causal fill, the strict greater-than-one-bin
  overtaking rule, exact issue grid, and post-issue mutation invariance after
  reading only the future candidate's issue timestamp. Queue, pair-provenance,
  and path checksum mutations fail closed. Invalid plasma, Vx, delay, and
  overtaking retain their safe transported prefix and produce fourteen-step
  persistence when a causal seed exists. The Stage-A/B handoff copies known
  future bins exactly and emits a 14-by-5 `(Bx,By,Bz,logV,logn)` research
  candidate. Direct path-to-core use is rejected; the bound overload reverifies
  queue, driver artifact, and pinned frozen-core identity, then fails closed
  because no frozen support/gate artifact exists.
- The recoverability probe checks exact matured innovation lags, invariance to
  a post-issue target mutation, duplicate and invalid-lag rejection, pre-2023
  enforcement, identical issue support across leads, deterministic 168 h block
  resampling, and a closed-form constant-gain result.
- The M2 driver kernel is checked against a hand-computed fourteen-step
  rollout, synthetic joint-support recovery, deterministic refitting, exact
  companion indexing, selected-group threshold integrity, capped-iteration
  fail-closure, spectral-radius rejection without rescaling, finite recursion,
  and checksummed artifact corruption.
- The M3 error-state control is checked against hand-computed matured
  innovations and recursive corrections, post-issue mutation invariance,
  missing-history fallback, deterministic sparse fitting, stability rejection,
  correction caps, record and current-call base-center binding, artifact
  corruption, and non-regular output-target preservation.
- The shadow-chain contract is checked against hand-computed AR and full-M3
  corrections, exact wrapper-result handoff, post-issue invariance, all six
  lead-specific exogenous artifact identities, semantic core and conformal
  hashes, explicit receipt/transport/anchor-pressure/calibration provenance,
  feature and product schemas, issue-relative horizons, same-hour anchoring,
  exact-center fallback, checksummed round trips, and corrupt or non-regular
  artifact targets. Synthetic numeric and low-level path centers require
  explicit research scopes, while every ungated operational prediction call
  fails before forecast arithmetic.

## Tolerances

Exact structural, checksum, key, split, and identity claims use equality.
Well-conditioned floating-point hand calculations use tolerances at or below
`1e-12`. Synthetic constrained-weight recovery uses `2e-8`, commensurate with
the projected-gradient stopping tolerance and tight enough to catch a changed
weight, projection, or component order. The portable tree comparison preserves
EvoTrees' Float32 accumulation and uses exact or near-machine-precision
expectations at threshold edges. Scientific promotion margins are contract
values, not adjustable test tolerances.

## Anti-false-test checks

Mutation tests alter targets, post-issue drivers, split boundaries, artifact
payloads, feature order, lag metadata, source hashes, Boolean metadata types,
ephemeris source bytes, source availability, coordinate units, interpolation
rules, quality authority, value, row identity, required-field status,
decisions, receipt admissibility fields, paired receipt times, same-time row
revisions, nonlatest historical-row revisions, cross-feed orbit records,
post-issue pairs, pre-issue raw hashes, Dst response bodies and revisions,
cutoff heads, scheduler guards, completion records, deadline markers, queued
future bins, sparse-tail artifacts, queue/path checksums, delay limits,
freshness, and overtaking.
Each mutation must
either leave a causal prediction unchanged or trigger the specified fail-closed
error. This makes the tests sensitive to sign, indexing, leakage, schema,
serialization, and boundary errors rather than only checking successful
execution.

## Full verification

- All 1,753 focused V2.2 assertions pass on the final source tree under bounds
  checking with deprecations treated as errors. The bounded invocations read no
  post-2022 challenge values and performed no network or service action.
- Current metadata, ephemeris, quality, and source-URL identity extension:
  focused collector suite 322/322.
- Current explicit-only launchd integration: 67/67 under bounds checking with
  deprecations treated as errors, plus clean Bash syntax and plist validation.
  Collector-containing requests render without loading unless explicitly
  enabled; no real service action occurred.
- Current prospective half-hour issue and Dst receipt capture: 227/227 under
  bounds checking with deprecations treated as errors. It is off by default,
  emits no numeric forecast, and has not contacted a live endpoint or started a
  service.
- Current offline issue pairing: 112/112 under bounds checking with deprecations
  treated as errors. It performs no capture, network request, or serving action.
  The live overload writes the immutable cutoff; saved-cutoff replay is read-only.
- Current receipt-causal M2 queue and ungated sparse candidate: 151/151 under
  bounds checking with deprecations treated as errors. The separate low-level
  matrix-to-core suite passes 26/26. Both are pure, start no service, and have no
  observational accuracy or promotion result.
- Current recoverability helpers: 17/17 under bounds checking. The complete
  pre-2023 diagnostic passed the 0.25 nT point margin at all six leads; its
  noncausal driver input makes it a mechanism bound, not a promotion result.
- Current pure M2 group-sparse driver kernel: 52/52 under bounds checking. It
  has not been selected or fitted on a receipt-causal observational cohort.
- Current causal AR-only M3 error-state control: 70/70 under bounds checking.
  It excludes the exogenous M2 and issue-time features required by the full M3
  candidate and therefore does not establish end-to-end skill.
- Current full exogenous M3 error-state candidate: 106/106 under bounds checking
  with deprecations treated as errors. These are synthetic contract tests and
  do not establish observational forecast skill.
- Current checksum-bound V2.2 shadow chain: 149/149 under bounds checking with
  deprecations treated as errors. The safe focused suites spanning M2, the core
  path, AR M3, full exogenous M3, and the composite passed 403/403 assertions
  in aggregate. These are
  synthetic contract tests, not an observational fit or accuracy result.
- The full package suite was not rerun because this bounded task prohibited
  access to the local challenge-data values exercised by broader tests.
- Prior frozen V2.2 state before the current bounded V2.2 extensions: clean full
  package suite with 5,098/5,098 assertions passed.
- Current deterministic package experiment: `SolarSINDy experiments: V2.1
  deterministic smoke PASS` under bounds checking with deprecations treated as
  errors.
- Prior development harness: 155 passes, 3 warnings, and 0 failures. One warning is
  the conservative loose-tolerance scan documented above; the other two report
  that this package-only worktree does not contain the workspace manuscript
  data and figure directories.

## Debug and reverify record

The first full-suite run reached 4,891 passing assertions and one setup error:
the isolated worktree did not contain the ignored OMNI source cache required by
an existing V2.1 replay oracle. Pointing the test at a differently named source
path was correctly rejected by the existing provenance guard. The canonical
workspace source artifact was then mounted at the worktree's expected ignored
cache path, preserving both file content and path identity. No source behavior,
test expectation, or tolerance was weakened. A clean default-path rerun is the
decisive result.

The first collector verification exposed a Julia ownership detail:
`String(::Vector{UInt8})` consumed the response buffer before its size was
recorded. Copying the buffer for diagnostics fixed the source. An independent
adversarial audit then reproduced intermediate-symlink escape, head rollback,
orphan acceptance, missing-predecessor append, chronology regression, source-URL
reuse, empty verification, transport-exception loss, and permissive nonstandard
JSON. The archive implementation and mutation tests were strengthened at the
source; the final focused collector suite is 90/90 and the subsequent full
package suite is 5,098/5,098. The metadata-contract extension first raised the
focused collector suite to 139/139; issue-causal ephemeris capture and its
mutation oracles raised it to 196/196. The verified DSCOVR row-quality gate,
required-field checks, and quality metadata mutations raised it to 319/319. The
normalized duplicate-source-URL capture and verification gates raised the
current focused suite to 322/322. The
offline exact-time pairing layer now has 112/112 independent and mutation-sensitive
assertions. A follow-up audit found that an earlier implementation observed
only each response's latest row, which could miss a revision hidden below a
newer row. The implementation now compares every occurrence of the selected
timestamp across all pre-issue HTTP-200 receipts, and the focused suite includes
that regression. Its
broader-suite limitation is recorded above rather than treating the earlier
full run as post-change evidence.

The first prospective issue-capture audit found that a post-issue Dst revision
could hide an earlier causal anchor, a locally rehashed cutoff could truncate
its causal prefix, and a post-deadline record lacked a durable invalid-cohort
state. The causal-anchor fallback now searches the exact eligible prefix, the
cutoff records its first excluded receipt when present, and late completion
writes a fail-closed cohort marker. A second audit found missing pending-guard,
orphan-head, and monotonic-clock checks. Scheduled records now require archived
pending and completion evidence, full verification rejects orphan issue
records, and all scheduler clocks are nonnegative and nondecreasing. Exact
issue-time pairing is now materialized and rederived during verification; an
unavailable pair is recorded explicitly. The final focused suite is 227/227.
Coordinated rewriting of every local hash-bound object
cannot be detected without the external witness required by the blind protocol;
the current records therefore retain an explicit unavailable-witness status.

The first aggregate standalone invocation exposed that the M2 driver test file
relied on imports supplied by the full test runner. Explicit `CSV`,
`DataFrames`, `LinearAlgebra`, `Random`, `Statistics`, `Test`, and `SolarSINDy`
imports were added without changing an expectation. Its standalone
bounds-checked suite then passed 52/52, and the remaining focused chain passed
under the same strict settings.

## V2.3 integration: served static stack and V2.3 shadow

### Coverage

`test/test_operational_v22_serving.jl` (93 assertions) covers the served
static-stack contract: the pinned identity strings, the coupling gate at both
edges of its two conditions, the depth-safe alerting center, digest and label
verification of the shipped weights including a tampered copy and a relabelled
copy, the per-step and per-regime weighted sum against a hand-rolled expectation,
the SINDy-dominance and convexity properties of every cell the product can select,
the fail-closed paths (unsupported step, non-finite component), the neutral
treatment of a non-finite one-hour rate, and the physical projection above the
+50 nT ceiling. A streaming prefix of the archived base table supplies a real-data
oracle inside the unit suite; the full-scale version is the identity script.

`test/test_operational_v23_serving.jl` (169 assertions) covers the shadow
deployment. A synthetic 480-hour deployment fixture — frame, origins,
standardisation, a real 26-feature calibration, blend weights, a fitted boosted
error layer and a digest manifest — exercises the load path without the
87,000-origin archive. Checks: identity strings and per-step caps; the ballistic
transit helper against its closed form; the all-or-nothing innovation block; the
rebuilt archive and its standardisation recomputed independently of the shipped
table; five distinct load failures (tampered artifact, missing digest row, wrong
origin count, drifted standardisation, wrong correction cap); the analog key's own
incompleteness reporting, including the distinction between a hole inside the
mandatory seven-hour window and one in the run-length tail; the ensemble raw center
against a hand-rolled member rollout and the retrieval order against an
independently computed weighted distance; the correction, blend and error layer
against their definitions, including which five features move with the raw core;
and the shipped deployment's layer composition.

`test/test_live_forecast_verify.jl` adds an issued-row testset (inside the 623
assertions of that suite): the served identity and driver assumption, the served
center recomputed from the six logged components, the logged regime and coupling
gate, the shifted served band, both disclosed stack fallbacks (absent weights and
tampered weights, each falling back to the V2.1 center and the V2.1 label), and the
shadow row (status, ensemble size, pre-layer and post-layer centers, the identity
layer on a fresh log, and the shadow center never reaching the served columns).

### Independent expectations

| Claim | Independent expectation |
|---|---|
| Served center is the fitted stack cell | Hand-rolled `sum(cell.weights .* components)` from the parsed stack rows, and a full reproduction of the archived `static_v2_2_dst_nt` column |
| Coupling gate matches the archived definition | Restated from the archived rule and compared with the archived `coupling_active_mvm` column on every scorable row |
| Shadow center is the scored center | The scored `V2_3_final` column, with the pre-layer center checked against `V2_3_LAT` and the blend partner against `frozen_v2_1_dst_nt` |
| Ensemble raw center | A hand-rolled per-member frozen-core rollout through `init_assimilation` / `assimilation_predict!` |
| Retrieval order | An independently computed weighted squared distance over the standardised archive |
| Frozen-tail blend partner | `operational_core_forecast` under the held issue driver plus the deployed ridge correction |
| Analog feature values | Closed forms for `bz0`, `v0`, `dv6`, `bz_mean6`, `dst0`, `ddst1` and `vbs0` from the supplied driver history |
| Ballistic transit | `1.5e6 / V / 3600`, checked at 500 km/s against `5/6` h |

### Tolerances

Identity claims use 1e-9 nT, the tolerance the base table and the confirmatory
runner already use for reconstruction oracles; both oracles report exactly 0 nT,
so the tolerance is not load-bearing. The rebuilt analog standardisation uses
1e-9 in feature units. Hand-rolled arithmetic comparisons use `atol=1e-12` where a
different summation order is possible and exact equality where the same operation
is repeated.

### Anti-false-test checks

Each new test would fail under at least one plausible bug: a permuted component
order (the hand-rolled stack sum), a coupling gate that never disengages (the
archived-column comparison and the zero-rate boundary cases), a permuted analog
feature order (the independently recomputed standardised archive and the closed-form
feature checks), an unpinned stack (the tampered and relabelled copies), a shadow
deployment that is not the scored one (the origin-count, bounds and standardisation
checks), an innovation history taken against the post-layer center (the pre-layer
identity on a fresh log), an off-by-one in the lead-aware weight lookup (the
per-step blend recomputation), and a cap widened beyond the published value (the
`correction_cap_nt` load check).

### Full verification

Measured at the V2.3 integration (2026-08); superseded by the 2026-08-19 table at the end of this
report. The per-file counts below are the counts of that tree, not of the current one.

| Check | Result |
|---|---|
| `Pkg.test()` | 20313 / 20313 pass, 0 failures, 11m52s |
| `dev-harness-audit.sh` | PASS 314, WARN 3, FAIL 0 (`Pkg.test()` and `examples/experiments.jl` both PASS) |
| `test/test_operational_v22_serving.jl` | 93 / 93 pass |
| `test/test_operational_v23_serving.jl` | 169 / 169 pass |
| `test/test_serving_identity_oracles.jl` | 98 / 98 pass |
| `test/test_live_forecast_verify.jl` | 630 / 630 pass |
| `test/test_v2_2_served_replay.jl` | 106 / 106 pass |
| `test/test_realtime_monitor.jl` | 176 / 176 pass |
| `app/test/runtests.jl` | 769 / 769 pass |
| `examples/experiments.jl` | PASS |
| `validation/operational/v2_2_served_identity.jl` | PASS — 832,368 archived DEV/TEST rows, max abs delta 0 nT for the stack sum and 0 mV/m for the coupling gate |
| `validation/operational/v2_3_serving_identity.jl` | PASS — 701 anchors / 4,206 rows, max abs delta 0 nT at every model step for the final center, the pre-layer center and the frozen blend partner |
| `validation/operational/v2_3_build_deploy.jl --from-test` | PASS — 86,968 origins, matching the confirmatory archive record |

### Post-integration audit: serving-path corrections

#### Coverage added

`app/test/runtests.jl` gains five testsets covering the alerting and disclosure
gaps. The watch testset reproduces the escalation the stack stage could otherwise
drop: the previous product's -95 nT center with a [-105, -85] nT band raises a
watch into the intense tier and an alert level of 3, and the stacked product's
-88 nT center with a [-98, -78] nT band must now produce the same tier, the same
alert level, and an alert message quoting the same -105 nT edge. The same testset
checks that a deeper stacked center's band is *not* pulled down, that a pre-stack
row without the continuity column keeps its band unchanged, and that the app file
no longer contains its own copy of the depth-safe comparison. The remaining
testsets cover the per-horizon severity fields, the product name and driver
assumption taken from the served row (stacked, fallback and unrecorded), a
mixed-label cycle staying available under its weakest label while an unknown label
still fails closed, verified rows counted per served label, and the health
endpoint's served identity and trailing fallback rate.

`test/test_live_forecast_verify.jl` gains three testsets. The error-layer chain
testset issues eight consecutive one-hour-lagged anchors so the sixth innovation
lag of the last anchor is complete, then requires the two steps that carry a
fitted layer (2 h and 7 h) to have applied it and the two identity steps (3 h and
4 h) not to, with one lag recomputed from its two logged ingredients and the
earliest anchor still pending. The unpinned-stack testset checks that an empty
digest override is refused by default and that an accepted staged load carries a
separate identity. The short-feed testset checks both sides of the analog key's
depth boundary: an eight-hour feed fails closed with `missing_driver_lagN` and a
ten-hour feed is admissible with a truncated run-length window.

`test/test_operational_v23_serving.jl` gains the innovation-rule testset (the
`Dst(anchor + 1 h)` pairing, unmatured and non-finite drops, and a one-hour-shifted
observation series that must not silently pair) and an E-layer manifest testset
that deletes the boosted model's digest row and requires a load error, then shows
the same directory still loads with hash verification disabled so the refusal is
attributable to that check.

`test/test_v2_readiness_selftest.jl` runs `v2_readiness_audit.jl --self-test`
inside the package suite. The audit's own self-test gained nine cases: the served
driver-assumption sentence, a disclosed fallback payload that must warn rather than
fail, a served label disagreeing with the newest logged cycle, an unpinned label
that must not be accepted, a fully stacked 24-cycle window that passes, a
newest-cycle fallback with a loadable artifact that must fail, the same window with
an unusable artifact, a shadow window whose error layer never engaged, and a
tampered and an absent stack artifact.

#### Independent expectations added

| Claim | Independent expectation |
|---|---|
| The watch tier cannot fall below the previous product | The prior product's own payload, built as a separate fixture and evaluated through the same `build_status` / `compute_alert_state` path |
| The depth-safe edge is the point shift applied to the edge | Hand values: `-98 + min(0, -95 - (-88)) = -105` nT, with the deeper-center case asserted unchanged |
| The live innovation rule is the scored rule | The offline oracle builds the history from the scored table's step-1 `V2_3_LAT` and observations, then requires the shared live rule to reproduce it (51,754 anchors, max abs delta 0 nT) |
| One live innovation lag | `Dst(anchor + 1 h)` from the synthetic Dst function minus the logged `v23_step1_center_dst_nt` of that anchor, compared at `atol=0` |
| The E-layer digest check is what refuses the trimmed manifest | The same directory loads under `verify_hashes = false` |
| The analog key's depth boundary | An eight-hour feed fails, a ten-hour feed passes, with `v23_history_hours` bracketed against `V23_HISTORY_LAGS_H` and `V23_SOUTH_RUN_CAP_H` |
| Served fallback rate | Fixture 24-cycle windows built from explicit label sequences, with the pass/fail verdict asserted per window |

#### Anti-false-test checks added

Each added test fails under at least one plausible bug: reverting the watch edge to
the served band (the stacked watch tier drops to 2 and the alert level to 2);
reverting it to the served band shifted by the change in the centre, which is what
the V2.4 integration first published (the narrow-conformal-band case drops the watch
tier from 3 to 2 on the same physics the V2.1 operator warned on);
making the edge shift two-sided (the deeper-center band assertion); restoring a
second copy of the depth-safe comparison in the app (the source scan); keying the
innovation history on one-hour *rows* again (the eight-cycle chain finds no history
and both fitted steps stay pending); feeding the row's target-step baseline panel
into the one-hour center (the recomputed lag no longer matches at `atol=0`, because
the logged center would not be the one-hour center); accepting an empty digest pin
(the refused-fallback assertions); publishing the unpinned load under the pinned
identity (the label inequality, and the audit's unpinned-label case); hardcoding
the product name or driver assumption again (the fallback payload and the app
source scans); rejecting a mixed-label cycle (the availability assertion);
downgrading the newest-cycle fallback to a warning (the self-test's FAIL
assertion); and counting `ok:e_layer_pending` as unavailable (the disclosure
testset's availability pass).

#### Full verification after the corrections

| Check | Result |
|---|---|
| `Pkg.test()` | 20475 / 20475 pass, 0 failures, 13m17.5s |
| `dev-harness-audit.sh` | PASS 319, WARN 3, FAIL 0 |
| `test/test_operational_v22_serving.jl` | 93 / 93 pass |
| `test/test_operational_v23_serving.jl` | 183 / 183 pass |
| `test/test_serving_identity_oracles.jl` | 98 / 98 pass |
| `test/test_v2_readiness_selftest.jl` | 3 / 3 pass |
| `test/test_live_forecast_verify.jl` | 684 / 684 pass |
| `test/test_v2_2_served_replay.jl` | 106 / 106 pass |
| `test/test_realtime_monitor.jl` | 176 / 176 pass |
| `app/test/runtests.jl` | 859 / 859 pass |
| `validation/operational/v2_readiness_audit.jl --self-test` | PASS — 24 independent checks, exit 0 |
| `validation/operational/v2_2_served_identity.jl` | PASS — 832,368 rows, max abs delta 0 nT (stack sum) and 0 mV/m (coupling gate), unchanged |
| `validation/operational/v2_3_serving_identity.jl` | PASS — 701 anchors / 4,206 rows, max abs delta 0 nT at every step, unchanged; the live error-layer chain reproduces 51,754 scored innovations with max abs delta 0 nT |

### Debug and reverify record — served-column consumers

Promoting the served center changed the meaning of `served_pred_dst_nt`, and one
downstream oracle depended on the old meaning. The first full-suite run after the
change returned 20203 / 20209 with exactly six failures, all in
`test/test_v2_2_served_replay.jl` at the assertion
`replay.served_v2_1_dst_nt == live.served_pred_dst_nt`: the replay helper's V2.1
center is now the `v2_1_served_pred_dst_nt` continuity column, because the served
column carries the stack stage on top of it.

The fix retargets that assertion to the continuity column and additionally pins the
difference between the two served columns to the disclosed stack cell, recomputed
from the six logged components. The file then passes 106 / 106 and the full suite
passes 20313 / 20313 with zero failures. The regression is recorded here rather
than silently corrected because it is the exact class of failure the change was
expected to produce, and the new assertion is what would catch a future divergence
between the two served columns.

### Deployment-boundary audit: schema-change and cycle-keying corrections

#### Coverage added

`app/test/runtests.jl` gains two testsets. The first builds a trailing window that
straddles the shadow-schema change: twenty cycles carrying the previous served label
and no shadow columns, then four cycles carrying the stacked label and the shadow
columns, joined so the earlier rows hold `missing` in every shadow field. It requires
the health summary to report twenty-four cycles, the stacked identity and product of
the newest cycle, twenty fallback cycles, a newest cycle that is not a fallback, and
four available shadow cycles with an engaged error layer; it then drops the newest
row's served label entirely and requires twenty-one fallback cycles with no reportable
identity rather than a lost summary. The second testset builds a cycle whose horizons
carry different accepted served labels and different driver-assumption tokens, and
requires the published assumption to be the one belonging to the cycle's weakest label,
in both mixing directions, with a uniform stacked cycle still reporting the stacked
sentence.

The audit self-test gains seven cases: a pre-stage window with one stacked cycle on top
(fallback rate passes, twenty-three excluded cycles disclosed, window of one), the same
window through the shadow check, one isolated older fallback in a four-day window that
must pass, two fallback cycles in that window that must fail, the issue-hour newest
cycle under a stalled L1 vintage together with the weakest-label and unaccepted-label
readings, the dashboard comparison snapshot re-reading the newest cycle's served label
(and clearing it when that label is not accepted), and an identity artifact whose shadow
manifest digest is absent.

`test/test_live_forecast_verify.jl` gains cache-key assertions inside the shadow testset:
the one-hour center's key is read out of the live cache after a real issuance and must
carry eleven entries ending in the two content hashes, and the hash helper must separate
different values and the same values under different field names while treating integer
and float spellings of the same number as equal.

#### Independent expectations added

| Claim | Independent expectation |
|---|---|
| The health summary survives the schema change | A window built from two fixture generations, asserted to contain `missing` shadow fields before the summary is taken, with fallback and shadow counts stated per generation |
| A mixed-label cycle names the stage it is served under | The assumption of the weakest label's own rows, asserted equal across both mixing directions and distinguished from the stacked sentence |
| Pre-stage cycles leave the fallback window | The staged window size and the excluded count, asserted separately from the verdict, against a fixture whose generations are known by construction |
| The four-day window and the two-cycle failure rule | Fixtures with one and with two fallback cycles at known positions in a 96-cycle window |
| The newest cycle is one issue hour | A stalled fixture in which four rows share one solar-wind vintage across two issue hours: the issue-hour reading returns two rows, the vintage reading returns four |
| The comparison label is re-read | A pre-seeded stale label that must be replaced by the fixture log's newest-cycle label, and cleared when that label is unaccepted |
| The absent shadow manifest digest is a reported state | The deployed identity artifact with its digest field emptied, requiring a reported check rather than a raise |

#### Anti-false-test checks added

Each added test fails under at least one plausible bug, verified by mutation:
restoring `==` against the shadow flag raises `TypeError: non-boolean (Missing) used
in boolean context` inside the health summary and errors the schema-change testset;
restoring the common-field reading of the driver assumption returns `"unrecorded"`
and fails three assertions of the mixed-cycle testset; counting pre-stage cycles in
the fallback window fails the pre-stage pass assertion; counting them in the shadow
window fails the shadow availability assertion; lowering the failure threshold to one
cycle fails the isolated-fallback pass; shortening the window to twenty-four cycles
fails the window-size assertion; keying the newest cycle on the solar-wind vintage
fails the stalled-cycle row count; reading the shadow manifest digest as a string
raises `MethodError: no method matching String(::Missing)`; and dropping the label
recomputation from the comparison snapshot fails the refreshed-label assertion.

The shadow cache-key change is latent hardening and is covered structurally, by the
key's arity and the hash helper's sensitivity, rather than behaviourally: no natural
two-issuance fixture separates a stale drivers/memory key from the analog-feature hash
that already keys the same cache, because both change together whenever the L1 stream
advances.

#### Full verification after the corrections

| Check | Result |
|---|---|
| `test/runtests.jl` | 20503 / 20503 pass, 0 failures, 10m22.7s, exit 0 |
| `dev-harness-audit.sh` | PASS 323, WARN 3, FAIL 0, exit 0 (the three warnings are the pre-existing loose-tolerance scan and the two absent paper directories) |
| `app/test/runtests.jl` | 880 / 880 pass, exit 0 |
| `test/test_v2_readiness_selftest.jl` | 3 / 3 pass, exit 0 |
| `validation/operational/v2_readiness_audit.jl --self-test` | PASS — 31 independent checks, exit 0 |
| `test/test_live_forecast_verify.jl` | 691 / 691 pass, exit 0 |
| `test/test_serving_identity_oracles.jl` | 98 / 98 pass, exit 0 |
| `test/test_operational_v23_serving.jl` | 183 / 183 pass, exit 0 |

The served center, the shadow center and both identity oracles are unchanged by this
round: `test/test_serving_identity_oracles.jl` and the V2.3 shadow serving suite pass
at the same counts as before, which is what pins the shadow cache-key change to the
cache rather than to the center it returns.

#### Real-artifact exercise of the changed audit path

Two mixed-generation live-log fixtures preserved from the audit were run through
`audit_live_log!` end to end. On the mixed-label log the served window admits one
staged cycle and discloses thirty-seven excluded pre-stage cycles, the shadow window
admits one and discloses twenty-three, and the newest staged cycle's fallback is a
FAIL — the intended verdict for that fixture, reached without an exception anywhere in
the path. On the stalled-L1 log the two newest-cycle definitions are shown to disagree
on real data: the vintage-keyed reading pools eight rows spanning two issue cycles and
two different served labels, while the issue-hour reading returns the four rows of the
cycle the API published and its stacked label. That disagreement is the spurious
served-label FAIL the single definition removes.

## V2.4e integration: the served super-learner

### Coverage

`test/test_operational_v24_serving.jl` (425 assertions) covers the served center. A
synthetic bundle built by `test/v2_4_serving_fixture.jl` — a 720-hour frame with its analog
archive and standardisation, a real 26-feature correction, six tiny boosted models on the real
29-column design schema, a floor-satisfying ten-expert stack with resolved, regime-pooled and
fully pooled cells, a complete conformal grid, the guard and selection records and a digest
manifest — exercises the load path without the 138,715-origin archive. Checks: the published
contract (identity, served variant and stack label, expert order, the four-member SINDy family
and the static expert's slot in it, the floor, the deepening thresholds, the depth edges, and
that the identity claims no guard stage); the depth bins at both edges including the non-finite
case; the cell chain and the grid it must terminate in; the deepening cell at both edges of both
of its conditions; the guard arithmetic in and out of a deepening cell and its one-sidedness;
the depth-safe severity over one, two and three partners and its agreement with the deployed
two-stage rule; cell resolution walking the chain from a resolved cell through a regime-pooled
cell to the fully pooled one; the served center as a hand-rolled weighted sum over all ten
experts, including a perturbation of the static expert alone that must move the center by
exactly its own weight; the deployed bundle's guard switch read from `guard.json` and reported
as inactive, with a deepening row served as the stack center itself; a second fixture bundle
whose record enables the guard, served with it, so the retained code path is exercised rather
than assumed dead; the four fail-closed input refusals (unsupported step, non-finite anchor Dst,
non-finite static expert, non-finite or short expert panel); the climatology relaxation against
its closed form and its monotonicity in lead; the direct design against `v23_direct_features` on
the whole frame, against the analog key column by column, and its refusal when the two disagree;
the increment inversion against a hand-rolled `predict + dst0`; every gap in the Dst ladder
reported by the lag it is missing; the interval per stratum with the pooled fallback; and
manifest verification as a digest check.

Four additions closed test debt in this suite. The deepening cell's coupling branch is a strict
`> 0`, and is now pinned against a `!= 0` reading with a negative and a `-1e-9` coupling on a
deep, slowly recovering ring current, where the coupling term is the only thing deciding the
cell. The pooled conformal stratum's fixture half-width was equal to the shallow bin's, so a row
that resolved to the pooled stratum and a row that resolved to the shallow one returned the same
number and the interval fallback was untestable; the pooled width is now distinct at every step
and the fallback is asserted against both the pooled width and the shallow width it must not
return, and against the resolved stratum's own depth label. The physical `clamp(., -2000, +50)` nT
projection has its own test on a synthetic panel at the ceiling and at the floor, with the
interval formed on the projected center, and with strictly interior panels asserted unflagged.
The direct-GBM reader's manifest-digest gate is reached directly, with a hashed-name set that
omits one model file: through a full load that defect is caught earlier by the manifest's
required-artifact rule, so the gate would otherwise be a branch no test enters.

Twenty-eight bundle defects are injected one at a time — sub-floor SINDy mass, non-unit mass, a
negative weight, a missing pooled cell, a missing conformal bin, a zero half-width, a
relabelled identity, a claimed residual layer, a drifted guard rate, a drifted depth edge, a
permuted expert order, a permuted boosted design, a tampered file, an unlisted model, a
drifted standardisation, a wrong origin count, an out-of-fold pool year at the fold year, a
non-physical timescale, a served cell whose recorded expert set is the nine-expert one, a
served cell whose recorded expert count is nine, a floor group that drops the static stack, a
renamed served variant, a renamed stack variant, a guard enabled without its reference, a guard
disabled while still naming one, a guard record with no switch at all, a whole stack table
relabelled as the nine-expert fit, and conformal rows keyed on another variant's name — and each
must fail to load. The nine Amendment A3 defects are additionally asserted to be present in the
mutation list, and the list's length and uniqueness are asserted, so a silently dropped case is a
failure rather than a smaller loop.

Each defect is asserted against the message of the check it exists to exercise, through
`V24_FIXTURE_EXPECTED_ERRORS`, and the suite asserts that every mutation has such an entry. The
previous `@test_throws Exception` form passes whenever anything throws, and two defects were in
fact being caught by a different check than the fixture documented, which left the documented
check untested while the suite looked green: an unlisted direct-GBM model is refused by the
manifest's required-artifact rule rather than by the reader's digest gate (now reached directly,
above), and a renamed served variant is refused by the selection record rather than by the
conformal keying the fixture comment described. `:conformal_variant_mismatch` — the conformal rows
alone carrying another variant's name — was added so that keying is exercised on its own.

`test/test_live_forecast_verify.jl` (852 assertions in that suite) adds the
served-V2.4e testsets: the served identity, driver assumption and interval source; every new log
column; the served center recomputed from the logged state through the serving function,
including the frozen expert recomputed as the held-driver rollout, the assertion that
substituting the logged core center moves the stack center, and the assertion that perturbing
the static expert alone moves it too; the logged guard flag false and the published center equal
to the logged stack center; the absent-bundle and tampered-bundle fallbacks to the static stack;
the static-expert fallback to the V2.1 operator when the stack stage cannot act; a short L1 feed
failing the analog key closed; a short Dst ladder failing the direct expert closed; the bounded
retry cool-down healing a staged bundle; and the deployed `deploy/v2_4/` bundle serving a finite
center with all three severity partners present. The V2.2 and V2.3 testsets point
`SOLARSINDY_V2_4_DEPLOY_DIR` at a nonexistent path so each still isolates the stage it names.

Three additions. The predecessor band edges are asserted per row: both are finite on a served row,
both carry the same half-width (the band the pre-V2.4 machinery would have served either center
under), that half-width differs from the served conformal half-width — which is the condition
under which shifting the served edge under-warns — the stack column is `missing` on a row whose
stack stage could not act, and on a row the stack or the V2.1 operator actually served the
partner edge equals the published edge exactly, so the alerting minimum is idempotent by
construction. The `v24_pred_dst_nt == v24_l1_center_dst_nt` invariant is asserted under
`!v24_projection_applied`, and the flag is compared with the serving function's own
`projection_applied` rather than assumed false.

The third is a served-stage status matrix: every documented `v24_status` value with a reachable
code path is produced by a code path in one testset. The early refusals — unsupported model step,
absent calibration, unavailable and unpinned static expert, absent and invalid deployment, absent
anchor Dst, absent previous Dst, and any exception inside the stage — are driven directly against
the state a served row logged, because they are decided before any expert is formed; the two
short-feed cases run end to end; and a single-step v1 issuance produces the row-level default.
Three statuses are defence-in-depth branches that cannot fire under the deployed loader and are
reported as such rather than faked: `fallback:non_finite_center` (a non-finite expert or
combination is refused earlier and a conformal half-width must load positive and finite),
`fallback:incomplete_analog_key`, and `fallback:incomplete_direct_design`. For the second, the
suite enumerates every single-input defect of the analog key — absent anchor Dst, absent previous
Dst, an absent driver record, each of the five driver channels non-finite, and a non-positive
density, at three of the seven mandatory lags — and shows each landing on a *named* reason, which
is what makes that branch unreachable rather than untested; a new rejection condition in the
feature block would surface there.

`app/test/runtests.jl` (1,035 assertions) covers the three-label chain and the three-partner
severity: every accepted label publishes, the chain's order decides a mixed cycle's label at
two and at three stages, the depth-safe center is held to the stack partner alone and to the
V2.1 partner alone, the payload discloses
`v24_status`/`v24_pred_dst_nt`/`v24_guard_applied`/`v24_projection_applied`/`v24_regime_cell`
per horizon and reports `nothing` on a fallback row, the health window is keyed on `v24_status`
and counts which stage it landed on, and the dashboard names the new pipeline stages.

The watch edge has its own testset, on the case the shift rule gets wrong: a served center of
-88 nT with a +-4 nT conformal band against a V2.1 operator that warned at -95 nT with a +-10 nT
band. The shift rule publishes -99 nT and a watch tier of 2; the minimum over the logged
predecessor edges publishes -105 nT and a tier of 3, which is the tier the operator raised on the
same physics. The testset also pins the stack partner deciding on its own, a deeper served edge
being published unchanged (so the rule is idempotent and never widens a band the product did not
issue), a non-finite partner edge being dropped rather than propagated, the per-horizon
disclosure of the published edge with its source and both partner edges, and a row written before
those columns existed keeping the earlier shift rule and disclosing that it did.

Two dashboard behaviours are executed rather than read. Both blocks of `app/public/app.js` are
extracted verbatim between their own sentinels and run under `node`; the tests skip when no JS
runtime is present. The capability block pins that an unrecognised stage token falls back to the raw
label for the whole pipeline instead of being dropped from the list, and that a token resolving
through `Object.prototype` (`toString`, `constructor`) is not presented as a capability. The
severity-line block is run against a synthetic payload with a DOM stub and its rendered text
asserted: the centre and edge appear as numbers (`severity centre -95 nT`, `watch edge -105 nT`),
the stage that set the edge is named in reader-facing words, and a payload with no alerting values
leaves the line empty and hidden rather than showing an em dash where a warning number belongs. That
the element exists in the page and that the renderer is reached on both the populated and the empty
path are asserted at the source level.

`test/test_serving_identity_oracles.jl` (199 assertions) adds the V2.4 oracle's
contracts: its declared base-table columns, its per-column report covering every expert — the
static stack among them — and every stage, the absence of any residue of the earlier variant's
column names, the raw pass-through semantics of its driver history (a record with non-positive
density is passed through, not filtered, because the study's run-length and coupling-lag features
read such a record's other fields), the depth of its Dst ladder, the determinism and storm
coverage of its anchor sample, and the published artifact: every one of the fifteen reported
columns within 1e-9 nT, all six model steps present, the deepening state and the deep cells
exercised, all three regimes present, no row guarded, every published center equal to its stack
center, and every half-width positive.

`test/test_v2_readiness_selftest.jl` and `v2_readiness_audit.jl --self-test` (38 independent
checks) cover the audit's own fixtures under the three-label chain: a fully served window
passes, a newest-cycle fallback fails whether or not the artifacts load here, both stages
disclose separately, pre-stage cycles are excluded and disclosed, the four-day window's
two-cycle failure rule holds, and the weakest-label reading resolves two- and three-stage
cycles.

Post-deployment fix (2026-08-18, first live readiness run on the merged main): the identity
contract coerced the record's `served_bundle_training_max_target_utc` with `String(...)`, but
CSV.read parses that ISO-8601 field into a `DateTime`, so the audit aborted with a
`MethodError` before the verdict line whenever the served bundle had been loaded (the self-test
never populated the bundle metrics, so its identity fixture short-circuited before the
coercion). The comparison now goes through `_identity_datetime_agrees` (instants when both
sides parse, text otherwise, never coerced), and a new self-test case loads the deployed bundle,
evaluates the contract with the metrics populated (pass on the deployed record, fail — not raise
— on a record carrying another refit's training bound) and unit-checks the helper; the old code
fails that case with the original `MethodError` (mutation check run).

Two added fixtures. A window whose newest cycle predates the served stage while its newest
*staged* cycle fell back: the two readings give opposite verdicts there — deciding the rule on the
newest staged cycle fails the window, deciding it on the newest cycle withholds the verdict and
the single isolated older fallback then passes on the window rules — so the fixture discriminates
rather than decorating. Restoring the previous `last(fallback_flags)` reading fails it. And the
bundle-identity tie: the deployed bundle's selection record and manifest build row both carry the
published identity, a manifest with no build row reads as no identity, served rows publishing that
identity pass, a served label that is not it fails, and a newest cycle with no served row or a log
predating the status column warns rather than failing.

### Independent expectations

| Claim | Independent expectation |
|---|---|
| Served center is the fitted cell | Hand-rolled `sum(cell.weights .* experts)` over all ten experts from the parsed weight rows, and the full reproduction of the study's `v2_4e` column |
| The static stack is an expert, not a discarded reference | Perturbing `static_v2_2` alone moves the center by exactly `w_static_v2_2` times the perturbation, in the unit suite and in the live suite |
| The guard is off but not gone | The oracle asserts `raw_center == l1_center` on every row of a bundle recording no guard; a second fixture bundle that enables the guard reproduces `min(center, static)` in a deepening cell and the unguarded center outside one |
| Interval endpoints | The study's `v2_4e_lo_nt` / `v2_4e_hi_nt` columns, and `center ± half_width` from the parsed stratum |
| Direct-GBM design | `v23_direct_features` over the whole hourly frame, and the analog key column by column |
| Direct-GBM center | `v23_predict(model, design) + design[dst0]`, and the study's `direct_gbm` column |
| Climatology center | `Dst(t)·exp(−h/τ)` in closed form, and the study's `climatology` column |
| T1r analog center | The study's `t1r_analog` column, with the ensemble raw core checked against `t1_analog_raw` |
| Frozen V2.1 expert | `operational_core_forecast` under the held issue driver plus the deployed ridge correction, and the study's `frozen_v2_1` column |
| Regime, depth bin, cell and deepening flag | The study's `regime`, `depth_bin`, `l1_cell_regime`, `l1_cell_depth` and `deepening_cell` columns, per row |
| Gated coupling proxy | Recomputed from the row's `VBsouth_mvm` and one-hour rate, compared with the archived `coupling_active_mvm` |
| Bundle fits are the study's fits | The study's persisted `v2_4_l1_weights.csv`, `v2_4_conformal.csv` and fold manifest, compared inside the builder before publication |
| Depth-safe severity | The shared dependency-free definition, checked to reduce to the deployed two-stage rule |
| Depth-safe watch edge | The predecessor's own edge, computed independently in the fixture as its center minus its own half-width, against which the shift rule is shown to under-warn by one storm tier |
| The physical projection is a stage | A synthetic panel above the `+50` nT ceiling and below the `-2000` nT floor, with the unprojected combination computed by hand |
| The bundle identity is the artifact's | The `selected.json` record and the manifest's `build/identity` row, and the served label of the newest cycle's rows |

### Tolerances

The identity claims use 1e-9 nT, the tolerance the base table and both earlier oracles use;
the published run reports exactly 0.0 nT on all fifteen columns, so the tolerance is not
load-bearing. The builder's agreement with the study uses 1e-12 and also reports exactly 0.0.
Hand-rolled arithmetic in the unit suite uses `atol=1e-12`; the stack's unit-mass and
SINDy-floor checks use 1e-9 in weight units, loose enough for the sum of ten doubles and tight
enough to catch an edited weight; the rebuilt analog standardisation uses 1e-9 in feature
units, which the injected 1e-3 drift exceeds by six orders of magnitude.

### Anti-false-test checks

The fixture bundle's weights sit exactly on the 0.60 floor, so a floor check written with the
wrong inequality fails. The resolved `(active_deepening, deep)` cell carries different weights
from the pooled cell, so a chain that silently took the pooled cell would change the center.
The guard cases pair a deeper and a shallower reference in the same deepening cell on a bundle
that enables the guard, so a guard written as `max` or applied unconditionally fails; the same
inputs on the deployed bundle's switch must leave the center untouched, so a guard applied
regardless of the bundle record fails too. The static expert's weight is non-zero in the fixture
cells, so a loader that read the served weights from the nine-expert rows would change the
center. The interval cases give each depth bin a
different width, and the pooled stratum a fourth width distinct from all three, so a row served
the pooled width fails and a pooled fallback that silently returned the shallow width fails too.
The projection cases use strictly interior panels for the inert direction, because an
exactly-boundary panel lands a hair outside the range under a unit mass held only to 1e-9 and the
projection then legitimately acts — an "inert at the boundary" assertion would be a false test.
The deepening cases pin the coupling comparison with a negative and a `-1e-9` coupling, so a
`!= 0` reading fails. Every bundle defect is asserted against the message of its own check, so a
defect caught by an unrelated check fails instead of passing as coverage. The direct-design test compares
against the package function on the whole frame, so a mini frame that got a lag wrong fails
rather than agreeing with itself. Each bundle mutation is injected alone and asserted to fail
to load, and the mutation list's length, uniqueness and membership are asserted, so a case that
stopped being generated is a failure rather than a shorter loop.

### Full verification

Measured at the V2.4e integration (2026-08-18); superseded by the 2026-08-19 table at the end of this
report. `test/test_serving_identity_oracles.jl` at 199, `test/test_operational_v22_serving.jl` at 93
and `v2_readiness_audit.jl --self-test` at 38 checks are counts that require untracked locally
generated artifacts; a clean checkout reaches fewer, and the difference is now recorded rather than
implied.

| Check | Result |
|---|---|
| `test/test_operational_v24_serving.jl` | 425 / 425 pass |
| `test/test_live_forecast_verify.jl` | 852 / 852 pass |
| `app/test/runtests.jl` | 1,035 / 1,035 pass standalone, 1,036 / 1,036 inside the package suite |
| `test/test_serving_identity_oracles.jl` | 199 / 199 pass |
| `v2_readiness_audit.jl --self-test` | PASS, 38 independent checks |
| `validation/operational/v2_4_serving_identity.jl` | PASS, max abs Δ = 0.0 nT on all fifteen columns over 753 anchors / 4,518 rows, rerun after the projection flag was added to the serving return |
| `examples/experiments.jl` | PASS |
| `Pkg.test()` | 282,144 / 282,144 pass, 0 failures, 14m22.7s |
| `dev-harness-audit.sh` | PASS 338, WARN 3, FAIL 0; the warnings are the pre-existing loose V2.2 tolerances and two absent paper directories |
| Scratch live cycle (`live_monitor.jl --once`) | 4 rows, all `v24_status=ok`, `v24_guard_applied=false`, `v24_projection_applied=false`, `v24_pred_dst_nt = v24_l1_center_dst_nt`, `interval_source=v24_conformal_depth`, cell `active_deepening/shallow`, `v24_history_hours=12`, both predecessor edge columns finite and each equal to its stage's center less the lead's pre-V2.4 half-width |

## Study-side embargo of the L2 inner split

### Coverage

`test/test_v2_4_learn.jl` gains one testset, "the inner split embargoes its training block by
168 h, per row's own step" (93 assertions), and three assertions inside the existing residual
acceptance testset. Two hourly pools carrying all six model steps at every issue hour exercise
both branches of `v24_inner_split`: 20,000 hours, long enough for the plan's last-24-months
rule, and 9,001 hours, whose 9,000 h span makes the chronological two-thirds boundary land on
an exact hour so single rows can be named by their target. Each is checked for the same six
properties — the halves are disjoint, the cutoff is exactly 168 h before the validation start,
every training row's `issue + step` clears the cutoff, every validation row is issued at or
after the boundary, the rows in neither half are exactly those whose issue precedes the window
while their target matures inside it, and their number equals the closed form
`sum over steps of (167 + h)` = 1,025. Two named-row blocks follow: at each step the row whose
target lands exactly on the cutoff is in the training half and the next issue hour at the same
step is in neither half, and at one issue hour three hours before the cutoff the six steps
split 3/3 between training and the gap. The acceptance testset additionally asserts that
`v24_fit_l2` forwards the cutoff and the dropped count and that the three row counts partition
the pool, and the whole-stage testset asserts that `v2_4_l2_selection.csv` persists
`inner_target_cutoff_utc` 168 h before `inner_boundary` and that the persisted counts partition
the pool.

### Independent expectations

The dropped-row count is a closed form derived from the geometry rather than from the
implementation: at step `h` the gap holds the issues strictly between `cutoff - h` and the
boundary, which is `V24_EMBARGO_HOURS - 1 + h` rows, so the pool-wide count is 1,025 for the
six-step grid and is independent of pool length. The kept/dropped pair at each step is named by
its target, not by its index, so it survives any change to the fixture's length or ordering.

### Anti-false-test checks

The testset was run against the previous contiguous split, restored for the purpose: it
reports 22 failures, and the direct oracle on the acceptance testset's own 4,000-hour pool
returns `n_embargoed = 0` against the expected 1,025, so the new assertions fail for the reason
they were written for rather than passing on both implementations. The per-step block is what
separates a per-row rule from a single-nominal-step rule: a split that embargoed every row by
the same step would put all six steps of the probe hour on one side and fail. The whole-stage
assertion is deliberately `== 0` rather than `> 0`, because the stage fixture's pool years are
30-day blocks eleven months apart and the two-thirds boundary falls in the empty gap between
them; asserting a positive count there would be a false test, so the gap arithmetic is pinned
on the hourly pools and only the partition identity and the persisted cutoff are pinned on the
stage.

### Full verification

| Check | Result |
|---|---|
| `test/test_v2_4_learn.jl` before the change | 259,863 / 259,863 pass over 35 testsets |
| `test/test_v2_4_learn.jl` after the change | 259,963 / 259,963 pass over 36 testsets, 0 failures |
| Mutation: contiguous inner split restored | 22 failures in the new testset; `n_embargoed` 0 against the expected 1,025 |
| `validation/operational/v2_4_learn.jl` (run 6, 8 threads) | complete in 1,477.9 s wall / 1,465.5 s recorded, selection `v2_4e`, decision `SHADOW`, serve rule `SERVE_ELIGIBLE_PENDING_G4` on ALL and E2 — all identical to run 5 |
| `validation/operational/v2_4_serving_identity.jl` (against the regenerated `learn_year_2025.csv`) | PASS, max abs Δ = 0.0 nT on all fifteen columns over 753 anchors / 4,518 rows |
| `Pkg.test()` after the change | 282,260 / 282,260 pass, 0 failures, 33m24.1s; the whole-stage V2.4 testset contributes 137,183. The total is not comparable with the 282,144 recorded earlier in this file, which predates unrelated test additions elsewhere in the tree, and the wall time reflects a concurrent study re-run on the same machine |

## Test and documentation pass: the clean-checkout suite (2026-08-19)

### What was not observable before

Three of the findings in this pass are about the evidence rather than the product.

**A fixture that hid a swapped expert.** `v24_fixture_weights` gave each of the six non-SINDy-family
expert slots `rest / length(others)`, so any permutation among `persistence, burton, burton_full,
obrien, direct_gbm, climatology` was a mathematical no-op and the whole live-verification suite
passed with the engine's `burton` and `burton_full` slots exchanged. The deployed bundle is not
uniform: 27 of the 60 cells in `deploy/v2_4/stack_weights.csv` carry `w_burton != w_burton_full`, so
the same exchange in production would have moved the published center. All ten fixture weights are
now distinct and strictly positive, each group summing exactly to its mass, and
`test/test_operational_v24_serving.jl` asserts for each of the 45 expert pairs that exchanging two
experts moves the center by exactly `(w_a - w_b)(x_b - x_a)`.

**Two gates that were never recomputed.** The V2.1 split audit's validation→holdout forecast-origin
comparison and the V2.1 served-holdout promotion gate were both asserted only through frozen CSV
columns. `test/test_v2_1_calibration.jl` now adds a holdout that leaks past the last validation
target while staying clear of the fit targets — the only shape that can distinguish the second
comparison from the first — plus the strict-equality boundary, and checks the refusal message names
the boundary crossed. `test/test_v2_1_served_holdout.jl` recomputes `pooled_gate_pass` on cohorts at
84, 85, 86 and 8,657/10,000 hits and on a cohort shaped exactly like the published one
(117,575 / 135,817 = 0.86569), asserts the published gate equals the comparison of its own published
coverage against its own published floor, and asserts that only the pooled cohort carries the gate.
The published coverage and the 0.85 floor are unchanged.

**Oracles that disappeared on a clean checkout.** `Pkg.test()` on a fresh clone of `80ad1e1` did not
pass: `test/test_v2_broad_replay.jl` errored on the untracked OMNI archive and
`test/test_v2_3_runners.jl` failed `@test source.real` after `oracle_frame()` silently fell back to a
synthetic frame, while four further real-data oracles degraded to bare `@test_skip`. All six now
register in `Main.LocalArtifactGate` before deciding whether to run; `test/runtests.jl` prints the
ledger and pins its membership, and `SOLARSINDY_REQUIRE_LOCAL_ARTIFACTS=1` makes any absence a
failure.

### Mutations re-applied and killed

| Mutation | Location | Suite | Result |
|---|---|---|---|
| `burton` ↔ `burton_full` slot exchange in the served expert panel | `examples/live_forecast_verify.jl:2445` | `test/test_live_forecast_verify.jl` | 894 pass / **4 fail** (was 852/852 pass at HEAD): the recomputed `l1_center` -65.9252 nT against the logged -65.8605 nT, and the two band edges with it |
| `rows[2]` → `rows[1]` in the validation→holdout origin check | `validation/operational/v2_1_calibration.jl:213` | `test/test_v2_1_calibration.jl` | 20 pass / **3 fail**: the leaking holdout is no longer rejected, its refusal message is empty, and the strict boundary is accepted |
| promotion floor 0.85 → 0.90 in the pooled gate | `validation/operational/v2_1_served_holdout.jl:197` | `test/test_v2_1_served_holdout.jl` | 100 pass / **5 fail**: the 86 %, 85 % and 8,657/10,000 cohorts and the published-shape cohort all lose the gate |

Each mutation was applied in place, run, then reverted from a byte copy and confirmed identical by
SHA-256 (`examples/live_forecast_verify.jl`
`fc8f5f83957dfa0e44f6b86ef0c53343d709d0997c3f1b9e81fbffda6685ddef` before and after).

### Full verification (2026-08-19)

Clean checkout means a `git archive` of the tree with this pass applied and **no**
`validation/output/`, which is what a fresh clone gets.

| Check | Result |
|---|---|
| `Pkg.test()` on a clean checkout | **282,900 pass, 6 broken, 282,906 total, exit 0, 22m51.8s.** The 6 broken are the six ledger-recorded skips; at `80ad1e1` the same environment gave 1 error, 1 failure and 4 silent skips |
| Local-artifact oracle ledger (clean checkout) | 6 registered, 0 exercised, 6 skipped, each named with its missing artifact; the ledger testset itself contributes 16 assertions |
| Repository-internal method-overwrite warnings during `Pkg.test()` | **0** (3,380 before, of which 57 were the colliding `main` entry points). The shared replay and study scripts are now included under a guard, and the five colliding `main()` definitions were renamed to script-specific entry points; the 34 warnings that remain are PlotlySupply overriding PlotlyBase, outside this repository |
| `.agents/scripts/dev-harness-audit.sh` | PASS 340, WARN 3, FAIL 0. The warnings are the pre-existing loose V2.2/app tolerances and two absent paper directories; the scan now covers `app/test` as well as `test/` |
| `test/test_operational_v24_serving.jl` | 783 / 783 pass (425 / 425 before this pass and the loader-hardening pass) |
| `test/test_live_forecast_verify.jl` | 898 / 898 pass |
| `test/test_v2_1_calibration.jl` | 23 / 23 pass |
| `test/test_v2_1_served_holdout.jl` | 105 / 105 pass |
| `test/test_conformal.jl` | 167 / 167 pass, standalone as well as through `runtests.jl` |
| `test/test_compat.jl` (new) | 56 / 56 pass |
| `validation/operational/v2_readiness_audit.jl --self-test` | PASS, 44 independent checks with the local artifacts present; PASS, 42 with `SOLARSINDY_OPERATIONAL_OUTPUT_DIR` pointed at an empty directory. The count is artifact-dependent, which the earlier "38 checks" claim did not say |
| `test/test_serving_identity_oracles.jl` | 199 / 199 with the identity tables present; 163 pass + 3 recorded skips without them |
| `test/test_operational_v22_serving.jl` | 93 / 93 with the base table present; 89 pass + 1 recorded skip without it |
| `test/test_v2_broad_replay.jl` | 30 / 30 and 33 / 33 with the OMNI archive present; 29 pass + 1 recorded skip without it |
| `test/test_v2_3_runners.jl`, self-origin oracle | 77 / 77 on the archived hourly frame; 76 pass + 1 recorded skip without it |
| `app/test/runtests.jl` inside the package suite | 1,343 / 1,343 pass |
| `examples/experiments.jl` | PASS, and now loads `deploy/v2_4` and serves one row through `v24_serving_center` — cell `active_deepening/deep`, center -126.734 nT, band ±31.857 nT, recomputed from the resolved cell's own weights and the bundle's own conformal stratum |
| `julia --project=docs docs/make.jl` | exit 0, zero errors, zero warnings (258 docstrings were missing from the manual before, and the documented build previously failed at instantiation) |
| `Pkg.resolve()` on the declared minimum Julia 1.10.11 | RESOLVED, then `Pkg.instantiate()` and `using SolarSINDy` load 427 exported bindings. Before dropping the `Logging` compat bound: `Unsatisfiable requirements detected for package Logging` |

Counts that require untracked locally generated artifacts are marked above. Every other count is
what a clean clone reaches.
