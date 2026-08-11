# V2.2 Research Test Report

## Coverage

The focused suite contains 977 assertions across the new V2.2 surfaces:

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
| Offline L1 issue pairing | 66 |
| Combined-mechanism recoverability helpers | 17 |
| Stable group-sparse M2 driver kernel | 52 |
| Causal AR-only M3 error-state control | 66 |

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
  orbit rows. Its oracles cover issue-time receipt cutoffs, latest exact common
  timestamps, unchanged duplicate rows, conflicting latest and historical-row
  revisions, missing GSE Vx, position disagreement, raw corruption, rehashed
  metadata changes, returned record/response hashes, and read-only filesystem
  identity.
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
revisions, nonlatest historical-row revisions, and cross-feed orbit records.
Each mutation must
either leave a causal prediction unchanged or trigger the specified fail-closed
error. This makes the tests sensitive to sign, indexing, leakage, schema,
serialization, and boundary errors rather than only checking successful
execution.

## Full verification

- Current metadata, ephemeris, quality, and source-URL identity extension:
  focused collector suite 322/322.
- Current offline issue pairing: 66/66 under bounds checking. It performs no
  capture, network request, write, or serving action.
- Current recoverability helpers: 17/17 under bounds checking. The complete
  pre-2023 diagnostic passed the 0.25 nT point margin at all six leads; its
  noncausal driver input makes it a mechanism bound, not a promotion result.
- Current pure M2 group-sparse driver kernel: 52/52 under bounds checking. It
  has not been selected or fitted on a receipt-causal observational cohort.
- Current causal AR-only M3 error-state control: 66/66 under bounds checking.
  It excludes the exogenous M2 and issue-time features required by the full M3
  candidate and therefore does not establish end-to-end skill.
- The full package suite was not rerun because this bounded task prohibited
  access to the local challenge-data values exercised by broader tests.
- Prior frozen V2.2 state before the current bounded V2.2 extensions: clean full
  package suite with 5,098/5,098 assertions passed.
- Prior deterministic package experiment: `SolarSINDy experiments: V2.1 deterministic
  smoke PASS`.
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
offline exact-time pairing layer added 66/66 independent and mutation-sensitive
assertions. A follow-up audit found that an earlier implementation observed
only each response's latest row, which could miss a revision hidden below a
newer row. The implementation now compares every occurrence of the selected
timestamp across all pre-issue HTTP-200 receipts, and the focused suite includes
that regression. Its
broader-suite limitation is recorded above rather than treating the earlier
full run as post-change evidence.
