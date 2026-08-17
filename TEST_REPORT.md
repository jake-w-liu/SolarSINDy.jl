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
