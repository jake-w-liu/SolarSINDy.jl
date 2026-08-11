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
