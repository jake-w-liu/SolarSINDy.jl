# V2.2 Research Development Ledger

## Development Ledger

| Item | Record |
|---|---|
| Objective | Test whether a product layer anchored to the frozen V2.1 SINDy-20/11 model can beat every same-row local comparator at 1, 2, 3, 4, 6, and 7 h without weakening storm or recovery safety. |
| Contract | Preserve the V2.1 coefficients; use issue-time information only; retain at least 60% combined served/frozen SINDy weight; distinguish anchor-relative model step from issue-relative product horizon; fail closed on missing rows, unsupported leads, lag mismatch, or corrupt artifacts. |
| Evidence | The source replay, primary stack, residual table, cross-fit table, feature schema, and learner version are hash-pinned. The development partitions end before 2023. |
| Independent oracles | Hand-computed constrained blends, synthetic known-weight recovery, exact live-kernel/replay identity, post-issue mutation invariance, whole-anchor embargoes, exact portable-EvoTrees inference, and artifact corruption tests. |
| Test plan | Focused tests cover the primary stack, sparse residual, portable boosted residual, served replay, expanding-window cross-fit, causal sparse-history kernel, M1 cross-fit helpers, and prospective receipt collector. The full package suite and deterministic package experiment are required before handoff. |
| Baseline verification | Every development row carries served and frozen V2.1, raw SINDy, persistence, Burton, Burton full, and O'Brien--McPherron predictions under a common finite-row mask. |
| Data regeneration trigger | Changes to split, feature, primary-stack, or learner code require rebuilding the affected replay and audit tables and rechecking their pinned hashes before interpreting scores. |
| Harness | `julia --project=. -e 'using Pkg; Pkg.test()'`; `julia --project=. examples/experiments.jl`; the repository development-harness audit against this worktree. |
| Risk | The main risks are post-issue leakage, repeated selection on one validation period, lag-zero/lag-one horizon confusion, hidden SINDy removal, artifact drift, and optimistic promotion from a small effect. |

## Implementation correctness gate

The implementation is isolated on `research/v2.2-accuracy`. The production
V2.1 checkout is unchanged. Its monitor, dashboard, and watchdog jobs are
stopped; no research module is wired into serving.

The primary constrained stack, sparse residual, portable boosted residual,
causal served replay, and expanding-window cross-fit have independent focused
tests. The nonlinear development runner reads only the pinned pre-2023 replay,
uses fixed 168 h issue clusters, and records zero exposed-benchmark rows read.

## Scientific outcome

The primary stack improved on the best same-row local comparator by only
0.0216--0.0604 nT on the four primary leads. The sparse residual passed its
regime guards only at 1 h. The frozen nonlinear family had no safe cross-fitted
setting at 2, 3, or 4 h; its final diagnostic gains remained below 0.25 nT at
every supported lead and the 6 and 7 h settings worsened safety cells.

The result is `NO_GO`. These modules are retained as reproducible research
infrastructure, not as a deployable V2.2 artifact.

## M1 mechanism test

The next bounded study inserted one causal exponential coupling-memory state
inside the frozen V2.1 trajectory. The core supports all eight subsets of
`{m, E-m, Dst* m}`, 2/6/12 h memory constants, lag-zero and lag-one anchors,
sign and nonoscillatory-stability constraints, varying-driver rollout, and
checksummed artifact I/O.

The expanding 2013--2017 cross-fit selected a 2 h `{Dst* m}` candidate only as
the least-bad screen. It lost to the strongest same-row comparator by
0.0610--0.6858 nT across the six leads, reached 97.41% availability, and failed
the active/recovery guards at 6 and 7 h. Its 10,000-replicate simultaneous
one-sided lower confidence bound was -0.7720 nT. M1 is therefore research-only
and is not a product upgrade.

## Input-trajectory pivot

The realized-driver oracle retains substantial 2--7 h headroom, whereas M1
adds only 0.0141--0.1181 nT over the conservative core. The next candidate is
therefore receipt-causal L1 advection followed by a stable group-sparse delay
continuation of the solar-wind trajectory, with the frozen SINDy core downstream.

No documented public pre-2023 DSCOVR or ACE holding currently reconstructs the
per-record receipt state required for an exact historical replay. A prospective
raw-response collector is implemented but not started. It writes
content-addressed raw bodies and a durable per-source sequence/hash chain,
records transport failures, rejects path and chronology corruption, and leaves
forecast serving untouched. Issue-causal GSE ephemeris provenance is now
implemented together with a DSCOVR-specific row-quality gate. A sufficiently
large prospective storm-and-recovery sample remains a prerequisite before any
M2 fit.

## Prospective L1 metadata contract

The receipt schema binds each HTTP response to metadata derived only from its
immutable raw bytes and hash-bound source URL. Only the canonical NOAA RTSW
endpoints receive authoritative status. NOAA Service Change Notice 26-21
defines `source` as the originating satellite and `active` as SWPC's Boolean
forecaster designation at that time. The collector therefore preserves exact,
sorted source tokens and accepts only literal Boolean active values; it does not
normalize spacecraft names or reinterpret `active` as measurement quality. The
notice is available at
`https://www.weather.gov/media/notification/pdf_2026/scn26-21_Data_Format_Changes_Impacting_SWPC_Products.pdf`.

The two official one-minute JSON feeds expose `overall_quality`. NCEI's
DSCOVR `m1m` and `f1m` metadata define that exact field as 0 for normal, 1 for
suspect, and 2 for error. The collector transfers those semantics only when the
latest unique row comes from the canonical NOAA endpoint, names `DSCOVR`
exactly, and has literal `active=true`. A normal magnetometer row must also have
finite `bx_gsm`, `by_gsm`, and `bz_gsm`; a normal wind row must have finite
`proton_speed`, `proton_density`, and `proton_vx_gse`. The last field supplies
the GSE radial component required by the transport calculation. The numerical
bounds match the live product's physical range checks. Missing,
nonintegral, suspect, or error quality codes, missing forecast fields, and all
non-DSCOVR sources remain inadmissible. The authoritative product metadata are
`https://data.noaa.gov/waf/NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/dscovr_m1m.xml`
and
`https://data.noaa.gov/waf/NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/dscovr_f1m.xml`.
NASA CDAWeb flags remain product-specific and are not transferred to the live
JSON rows.

NOAA Service Change Notice 26-21 identifies
`rtsw_ephemerides_1h.json` as the replacement operational ephemeris. The NCEI
`pop_dscovr` metadata defines DSCOVR GSE position in kilometres, but NCEI's
archive contract schedules the day-file after 00 UTC. An archived
`pop_dscovr` day-file therefore cannot prove that its positions were available
at an earlier same-day forecast issue. The collector instead retrieves and
timestamps the live SWPC ephemeris before requesting the measurement feeds. It
archives the exact source object, HTTP metadata, UTC and monotonic receipt
clocks, and a content checksum. Receipt completion, rather than an HTTP date or
last-modified header, is the conservative availability time. The position
oracle requires the same source, literal `active=true`, finite non-fill GSE
coordinates, and either an exact timestamp or a bracketing pair no more than
one hour apart. It uses linear interpolation and never extrapolates. The schema
and archive references are
`https://data.noaa.gov/waf/NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/dscovr_pop.xml`
and
`https://www.ncei.noaa.gov/archive/atrac/export/2015-06-15T18-40-09.pdf?id=24749`.

The `active` field selects the spacecraft used operationally; it is not a
measurement-quality flag. `rows_admissible` refers only to the response's latest
unique active measurement row. It becomes true only when DSCOVR identity,
normal row quality, required numerical fields, and issue-causal GSE position are
all bound. It does not declare every historical row in the response admissible.

Independent oracles recompute the complete metadata object from archived raw
bytes, replay the source-matched GSE interpolation from the archived ephemeris
object, require receipt completion before measurement request start, reject
non-Boolean `active` encodings, and reject independently rehashed changes to
units, interpolation, availability, quality authority, quality value, target
row identity, quality decisions, or admissibility.
Transport failures carry an explicit no-response metadata state. This extension
changes receipt provenance only; the collector and all forecast services remain
off.

## Offline L1 issue pairing contract

Exact timestamp pairing is defensible as a conservative same-reported-time
rule. Service Change Notice 26-21 retains `time_tag` in both replacement
products, while NCEI identifies `m1m` and `f1m` as one-minute averages. Equal
timestamps therefore identify the same nominal UTC minute. This does not prove
that the instruments used identical lower-level acquisition windows, so the
pairing layer makes no such claim and never interpolates measurement values.

`select_v2_2_l1_issue_pair` first verifies the complete v4 archive. It considers
only canonical magnetometer and wind records received no later than the issue
time, then selects the latest exact timestamp shared by individually admitted
DSCOVR rows. It returns finite GSM Bx, By, and Bz; proton speed, density, and GSE
Vx; the common bound GSE position; and the selected record, response, and
ephemeris hashes. Repeated identical rows are allowed and resolve to the latest
receipt. Every occurrence of the selected timestamp is checked across all
pre-issue HTTP-200 responses, including historical rows below a newer latest
row. Conflicting revisions, different bound positions, altered metadata,
or corrupt raw content fail closed. Records received after the issue are
excluded rather than treated as archive errors; selection fails if no exact
admitted pair remains. Verification is repeated after selection, and the
selected objects are rehashed before return.

The pairing layer performs no network request, write, capture, or forecast. If
the newest feed rows have different timestamps, it may return an older exact
common minute; downstream transport must enforce its own freshness limit using
the returned measurement and receipt times. No collector or serving process is
started by importing or calling it.

## Combined-mechanism recoverability bound

A development-only probe joins two hash-pinned pre-2023 out-of-fold tables and
combines the noncausal realized-driver oracle with matured one-hour forecast
innovations. It writes no artifact and rejects post-2022 targets, incomplete
six-lead anchors, duplicate keys, fold disagreement, and nonfinite features.

On 30,408 common rows per lead, the combined upper bound improved over the
strongest same-row comparator by 0.3336, 0.8838, 1.3889, 1.8260, 2.6879, and
3.1163 nT at 1, 2, 3, 4, 6, and 7 h. A 10,000-draw paired bootstrap over 191
nonoverlapping 168 h issue blocks gave one-sided per-lead lower bounds of
0.2771, 0.7770, 1.2646, 1.6603, 2.4359, and 2.8178 nT; the simultaneous
minimum-lead lower bound was 0.2771 nT. Because the study uses future realized
drivers and development-fold model selection, it demonstrates joint
recoverability only. It does not authorize a V2.2 artifact, serving change, or
paper claim.

## Pure M2 continuation kernel

`OperationalV22DriverArtifact` implements the frozen five-state order
`(Bx, By, Bz, logV, logn)` and 30-minute lags `(0, 1, 2, 6, 12, 24)`. Its
fitter applies joint predictor/lag group thresholding across all five output
equations, leaves the intercept unpenalized, and rejects any companion map with
spectral radius above `1 + 1e-8`. Rollout is fixed at fourteen 30-minute steps;
unstable maps are never rescaled. Normalization, selected groups, coefficients,
hyperparameters, fit-row count, and spectral radius are bound into the portable
artifact checksum.

This is a pure numerical kernel. Hyperparameter selection, causal transport,
training-fold preprocessing, support envelopes, correction bounds, activation,
and driver-level evaluation remain outside the artifact and must follow the
M2 nested protocol. No observational fit or score has been produced.

## Causal error-state control

The first M3 control constructs one-hour forecast innovations only after their
observations are available. It requires a contiguous 24 h buffer and uses the
frozen lags `(1, 2, 3, 4, 6, 9, 12, 18, 24)` h. The artifact is bound to the
exact M2-plus-core center hash, accepts at most three autoregressive terms under
the current sparse BIC screen, rejects spectral radius above `0.98`, recursively
predicts the error state, and caps the final correction at `5 + 5h` nT.

This AR-only component is a mechanism control, not the full M3 candidate. It
does not yet include the predeclared exogenous M2 trajectory, Dst, driver, and
issue-time features. It has only synthetic verification and cannot support an
accuracy or deployment claim.
