# V2.2 Research Development Ledger

## Development Ledger

| Item | Record |
|---|---|
| Objective | Test whether a product layer anchored to the frozen V2.1 SINDy-20/11 model can beat every same-row local comparator at 1, 2, 3, 4, 6, and 7 h without weakening storm or recovery safety. |
| Contract | Preserve the V2.1 coefficients; use issue-time information only; retain at least 60% combined served/frozen SINDy weight; distinguish anchor-relative model step from issue-relative product horizon; fail closed on missing rows, unsupported leads, lag mismatch, or corrupt artifacts. |
| Evidence | The source replay, primary stack, residual table, cross-fit table, feature schema, and learner version are hash-pinned. The development partitions end before 2023. |
| Independent oracles | Hand-computed constrained blends, synthetic known-weight recovery, exact live-kernel/replay identity, post-issue mutation invariance, whole-anchor embargoes, exact portable-EvoTrees inference, and artifact corruption tests. |
| Test plan | Focused tests cover the primary stack, residual candidates, served replay, chronological cross-fit, sparse history and driver kernels, prospective L1/Dst receipt capture, half-hour issue commitments, causal pairing and arrival transport, the core path, both M3 candidates, and the composite boundary. The full package suite and deterministic package experiment are required before a scientific handoff. |
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

The live `select_v2_2_l1_issue_pair` overload verifies the complete v4 archive
under the collector lock, requires every current source head to have been
received by the issue, and atomically records a checksummed issue-cutoff file.
The replay overload requires that saved cutoff path. It verifies and reads only
the two exact hash-chain prefixes named by the cutoff; it does not inspect a
later latest pointer, record, response body, or ephemeris object. Later replay
therefore cannot infer a historical boundary from the first future record, and
corruption strictly after a saved head cannot change the bound result.

Within the bound prefixes, selection considers canonical magnetometer and wind
records received no later than the issue and chooses the latest exact timestamp
shared by individually admitted DSCOVR rows. The v2 pair contract retains the
two product identifiers, units and frames, original quality values and binding
decisions, first eligible half-hour issue, cutoff identity, GSE position, and
the selected record, response, and ephemeris hashes. Every field is covered by
the pair checksum. Repeated byte-identical rows are allowed; receipt, revision,
quality, position, or metadata disagreement fails closed.

`select_v2_2_l1_issue_pairs` applies the same revision and provenance checks to
every exact common timestamp in an inclusive measurement-time window. It
acquires the collector lock once, constructs the two candidate sets once per
call, returns rows in chronological order, and rehashes every selected record,
raw response, and ephemeris object before releasing the lock. This is the causal
input vector for the arrival queue; history is not reconstructed by repeatedly
requesting one latest row.

The cutoff-bound pairing layer performs no network request, receipt-data write,
capture, or forecast. The live overload writes only its immutable issue-cutoff
record. Both overloads acquire and remove the collector lock so the selected
prefix cannot advance during selection. If
the newest feed rows have different timestamps, it may return an older exact
common minute; downstream transport must enforce its own freshness limit using
the returned measurement and receipt times. No collector or serving process is
started by importing or calling it.

## Prospective issue and Dst capture

`examples/v2_2_prospective_issue_capture.jl` is an explicit, off-by-default
research scheduler. It archives the raw SWPC Kyoto-Dst response, HTTP headers,
UTC and monotonic receipt clocks, parser outcome, first-body receipt, and
same-observation revision lineage in a content-addressed hash chain. Every
scheduled issue is an exact UTC half-hour, binds immutable L1 and Dst cutoffs,
records the causal Dst anchor and fixed 1/2/3/4/6/7 h targets, and links to the
preceding issue. A non-grid time is rejected rather than shifted or backfilled.

The scheduler creates a pending guard before an issue becomes durable and
requires matching archived guard and completion records during replay. A
record completed after `H+5 min` writes a durable invalid-cohort marker; later
capture and verification then refuse the root. Full-chain checks reject orphan
records and clock regression. Saved cutoffs isolate historical verification
from later receipts and raw objects, while any bound-object mutation fails.

No fitted or gated V2.2 model exists. Issue records therefore state
`research_capture_only_unavailable` and contain no numeric forecast. The exact
L1 pair is selected from the saved cutoff and hash-bound when available; lack of
a common admitted minute is recorded explicitly and rederived during replay.
The scheduler is not registered as a service and has never contacted a live
endpoint. Local hash chains cannot detect a coordinated rewrite of every bound
object; the blind protocol consequently requires an independent pre-target
commitment witness before any cohort can support promotion.

## Receipt-causal M2 arrival queue

`build_operational_v22_arrival_queue` consumes only checksum-valid selector-v2
pair outputs from one issue-cutoff snapshot whose magnetometer and wind
receipts are no later than the forecast issue. It requires the issue and every
admitted pair issue to lie exactly on the 30 min UTC grid, and checks product
identity, retained quality decisions, DSCOVR, GSM magnetic components, GSE Vx
and position, kilometres, positive speed and density, and `Vx < 0`. The Earth-UT driver
boundary is fixed at `x_ref_gse_km = 0`; V2.1's `1.5e6 km` scalar L1 distance is
retained only as a compatibility diagnostic. The causal Vx median uses the
measurement-time interval `(s - 15 min, s]`, and accepted delays are 20--120
min. UTC arrival bins are half-open and use componentwise physical medians.

The sparse seed contains 25 complete half-hour bins in chronological order and
has state order `(Bx, By, Bz, logV, logn)`. Exactly one isolated missing bin may
be copied from its immediate predecessor. A complete observed seed remains
fresh at exactly 90 min and falls back one millisecond later. A later packet may
arrive exactly 30 min before the preceding arrival maximum; a larger reversal
falls back. A partially elapsed issue bin is excluded from both history and the
future queue. A post-issue candidate is skipped after parsing only its issue
timestamp; none of its remaining fields are read. On invalid plasma, Vx, delay,
or overtaking, every safe transported prefix and bin is retained so any causal
seed can produce a fourteen-step persistence fallback with the original reason.

`build_operational_v22_arrival_path` copies a contiguous prefix of known future
arrival bins without modification, recursively applies the stable sparse
driver artifact only to the remaining tail, and returns exactly fourteen rows.
`operational_v22_arrival_path_matrix` exposes those rows as the low-level
research matrix `(Bx, By, Bz, logV, logn)`. Queue, selector record/raw hashes,
driver-artifact identity, origins, and
physical states are bound into composite checksums. Expected transport,
history, queue-prefix, or numerical-domain failures return an explicit
transported-persistence path and reason; corrupt structural provenance or a
checksum mismatch raises an error.

This layer is pure and performs no archive write, network request, capture, or
serving action. Its path schema is explicitly `ungated_candidate`. The direct
path-to-core overload is disabled; the bound overload reverifies the queue and
driver artifact plus a caller-pinned semantic frozen-core SHA, then still fails
closed. Training-fold support envelopes, activation, asymmetric clipping, Bz
sign protection, and an unobserved-shock gate have not been fitted or frozen,
and no thresholds are invented in their place.
No observational M2 fit, prospective storm score, promotion decision, or
service change follows from the synthetic verification.

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

## Checksum-bound shadow chain

`OperationalV22ShadowChainArtifact` is the offline identity boundary for the
complete V2.2 candidate. Its center hash binds the receipt-pair and
transport/support contracts, the anchor-pressure contract, the M2 artifact,
the fixed two-substep hourly aggregation and pressure-inversion policy, the
frozen V2.1 point core, the issue-relative horizon schema, and the same-hour
anchor rule. The manifest separately binds the product version, exact feature
order, conformal object's semantic hash, conformal sidecar, paired point
calibration, and either the AR control or all six lead-specific full-M3
artifacts. The frozen-core identity includes the canonical nonzero executable
term-code tuple, and conformal strata must retain the exact finite-sample
coverage floor implied by their sample counts and nominal coverage.

The research-only arithmetic accepts an explicit base-center record tied to its
issue, anchor, target, horizon, and center identity. Numeric centers require
`execution_scope=:synthetic_research_only`; the low-level M2-to-core result
requires and carries `execution_scope=:low_level_research_only`. Neither is
issued-path provenance. `operational_v22_shadow_predict` therefore fails closed
until a frozen issued-path gate artifact and proof exist; only the explicitly
named `operational_v22_shadow_research_predict` performs the existing offline
arithmetic. Component, feature, horizon, anchor, calibration, and provenance
mismatches throw before a research forecast is returned. Missing causal M3
history can return only the supplied exact base center; history associated with
another center is rejected. Checksummed
one-row artifact I/O rejects malformed, corrupt, symbolic-link, directory, and
other non-regular targets. Writes use the shared two-check atomic replacement
path so a target changed to a non-regular object after staging is preserved and
the installation fails closed.

Receipt-pair, transport/support, anchor-pressure, conformal-sidecar, and point
calibration checksums are explicit evidence tokens. This layer verifies their
presence and exact equality but cannot infer or independently prove the
upstream provenance semantics represented by those tokens.

This layer performs no observational fit, skill scoring, live fetch, or
serving action. Its synthetic verification establishes identity and causality
mechanics only; it does not establish forecast accuracy or authorize V2.2
promotion.

## V2.3 integration and the served static-stack promotion

### Development Ledger — served center and shadow center

| Item | Record |
|---|---|
| Objective | Promote a served point center that is a fitted, hash-pinned regime stack over the six existing point components, and integrate the V2.3 analog driver continuation as a shadow forecast whose live center is provably the center that was scored. |
| Contract | The served center is `operational_v22_predict` applied to the six live components, with the regime derived from issue-time state alone and the coupling input the gated proxy; the stack weights must match a pinned digest and fit label or the engine serves the V2.1 center and labels the row with the V2.1 identity. The shadow center is the V2.3 analog candidate: an 18-feature issue-time key, a K = 25 magnetic-weighted retrieval from the 86,968-origin 2010–2019 archive, one frozen-core rollout per member, the analog-core refit of the V2.1 ridge layer with no V2.1 safeguard, a lead-aware blend against a recomputed frozen-tail center, and a per-step error layer capped at 5 + 5h nT. The shadow center never reaches the served columns, the threat level, or an alert. |
| Evidence | Stack weights `operational_v2_2_primary_sindy60_fit407598`, SHA-256 `66e7347f71f5cdf407e85d4612702bb19c82dcbcd74d8c79526173f839472d7d`. V2.3 artifacts from `validation/output/operational/v2_3_test/artifacts/` (confirmatory decision `NO_GO`, failing gates A1 and A2), the all-DEV T1r calibration, the lead-aware weights, and the causal hourly frame; every shipped file carries its digest in `deploy/v2_3_shadow/manifest.csv`. |
| Independent oracles | (1) The archived `static_v2_2_dst_nt` column reproduced through the serving function on every scorable DEV/TEST row, with the coupling gate recomputed rather than read back. (2) The scored `V2_3_final` centers reproduced through the serving functions at every model step, with the frozen-tail blend partner recomputed and checked against the archived `frozen_v2_1_dst_nt` column. (3) Hand-rolled member rollouts, hand-computed stack cells, hand-computed lead-aware blends, and an independent restatement of the retrieval ordering in the unit tests. (4) Artifact corruption, missing-digest, wrong-origin-count, drifted-standardisation, and wrong-cap load failures. |
| Test plan | `test/test_operational_v22_serving.jl` and `test/test_operational_v23_serving.jl` cover the serving contracts against independent expectations; `test/test_live_forecast_verify.jl` covers the issued row end to end, including both disclosed fallbacks; the two identity scripts cover the full-scale reproduction. |
| Baseline verification | The served row keeps `v2_1_served_pred_dst_nt` (the V2.1 operator's own center) and the frozen-tail `improved_*` columns, so every served center can be audited against the operator it replaced. |
| Data regeneration trigger | A change to the analog features, retrieval, kernel, correction, blend, or error layer invalidates `deploy/v2_3_shadow/`; rebuild it with `validation/operational/v2_3_build_deploy.jl --from-test` and rerun both identity scripts. |
| Harness | `julia --project=. -e 'using Pkg; Pkg.test()'`; `julia --project=. examples/experiments.jl`; both identity scripts; the repository development-harness audit. |
| Risk | The named risks are a component-definition mismatch between the live and archived component panels, an unpinned stack silently serving different weights, a shadow center drifting from the scored artifact, and the error layer's innovation history being taken against the wrong center. |

### Archive membership is shipped, not derived

The analog archive is the set of DEV base-table anchors whose issue-time features
are complete and whose seven continuation records exist. That set is not a
function of the shipped hourly frame: an origin is an archive member only if it
was also a V2.1 calibration anchor, which additionally requires a
quality-flagged, non-gap-filled L1 driver record at `t-1`. The causal hourly
frame carries forward-filled drivers and no quality flag, so a frame-only rule
admits 87,466 origins where the scoring run used 86,968.

The deployment therefore ships the origin identities in `analog_origins.csv` and
the loader re-derives everything else: it recomputes each origin's features from
the shipped frame, re-checks the completeness and continuability rules, compares
the origin count and bounds with the values the scoring run recorded, and
recomputes the feature standardisation and compares it with the shipped table to
1e-9. A deployment whose archive is not the scored archive fails at load rather
than at the first served forecast.

### The lead-aware blend partner is recomputed

The scored candidate blends against a frozen-tail center: the deployed core
rolled with the issue driver held for every step, corrected by the deployed V2.1
ridge layer, with no safeguard. The live engine's own `v2_pred_dst_nt` is a
different quantity — its core rollout admits L1-measured hours and freezes the
trailing wind hour rather than the issue record beyond them. `v23_serving_frozen_center`
recomputes the scored definition, and the identity oracle checks that
recomputation against the archived `frozen_v2_1_dst_nt` column before the blend is
trusted; the reproduction is exact (max |Δ| = 0 nT over 4,206 scored rows).

### The error layer's innovation history is the pre-layer center

The error layer regresses the observed-minus-forecast residual of the one-step
center *before* the layer acts. The served log therefore records
`v23_center_dst_nt`, the center after the correction and the blend, separately
from `v23_shadow_pred_dst_nt`, the center after the layer. Taking the history
against the post-layer value would feed the layer its own output and would not
reproduce the scored centers. The layer is the identity whenever fewer than six
matured innovations exist, which is the state of a fresh log.

### The live error layer needs a one-hour center that is never an issued horizon

The first live implementation built the history by filtering the log for rows at
`model_step_hours == 1`. No such row exists in production: the requested wall
horizons are 1/2/3/6 h and the Kyoto anchor lags the issue hour by one hour, so
the issued model steps are 2/3/4/7 h. A census of the production hot log found
148 rows at steps {2, 3, 4, 7} and none at step 1, so `v23_e_layer_applied` was
false on every row while the shadow identity string still advertised the error
layer. The logged shadow center was the lead-aware blend, which is a different
model from the scored candidate.

The chain is now built from a quantity that does not depend on which horizons are
issued. Every cycle computes and logs `v23_step1_center_dst_nt`, the one-hour
pre-layer center of its anchor, and the innovation of anchor `a` is
`Dst(a + 1 h) - v23_step1_center(a)` taken from the observed Kyoto series the
issuance already holds. Maturity is therefore a property of the observation
series rather than of a verification pass, and no separate refresh step is needed
before the layer can engage.

Two details are load-bearing. First, the one-hour center must see the *one-hour*
baseline panel: its correction features read the baseline panel, and the scored
table's one-hour row carries one-hour baselines, so the engine captures the panel
on the first rollout step instead of reusing the row's target-step panel. Second,
the rule that turns logged centers into innovations lives in
`v23_serving_innovations_from_step1_centers` and is called by both the engine and
the offline identity oracle; the oracle asserts that the rule reproduces the
history it builds from the scored table, which it does on 51,754 anchors with
max |Δ| = 0 nT.

A step whose selected layer is a fitted model but whose history is incomplete
records `v23_status = "ok:e_layer_pending"`. The `ok` prefix is what availability
is counted on, so the row is available while the disclosure stays explicit; a
step whose selected layer is the identity by construction records plain `ok`.
Readiness reports the fraction of trailing cycles that applied a layer and warns
when it is still zero after eight cycles.

### The depth-safe center governs the watch edge, not only the point level

The published threat level is taken on `min(served, v2_1_served)`, so a stack
that blends toward persistence cannot lower a warning. The watch flag, however,
was assessed on the served band's lower edge, and the band is shifted onto the
served center: a shallower stacked center moves the whole band up, so the same
physics could produce a lower outbound alert level than the previous product did.
The reproduction was a V2.1 center of -95 nT with a [-105, -85] nT band, which
raises a watch into the intense tier and an alert level of 3, against a stacked
center of -88 nT with a [-98, -78] nT band, which raised no watch and an alert
level of 2.

The edge is now taken on the depth-safe center as well:
`lb_safe = served_ci05 + min(0, v2_1_served - served)`, which lowers the edge by
exactly the amount the point was lowered and leaves a deeper stacked center's
band untouched. The one-sided form matters: a symmetric shift would pull the band
down for a deeper stacked center and manufacture watches.

The comparison itself has a single definition. `v22_serving_depth_safe_center`
now lives in `src/serving_depth_safe.jl`, a dependency-free file the package
includes and the dashboard application includes as well: the application runs in
its own environment and cannot load the package, and a second copy of the rule in
the application is exactly how the published severity would drift from the served
contract. The container image copies that file beside the application sources.

### An unpinned served stack is refused rather than served under the pinned identity

`SOLARSINDY_V2_2_STACK_SHA256` set to an empty string disabled the digest check
while the label check remained, and the label check passes for any file that
copies the published label — so an edited weights file was served under the
pinned identity. The engine now refuses an empty digest override outright: the
row is served by the V2.1 operator and records
`v2_2_status = "fallback_v2_1:stack_unpinned"`. A staged run can still exercise
the path with `SOLARSINDY_ALLOW_UNPINNED_STACK=1`, in which case the stack center
is served under `V2_2_UNPINNED_SERVED_TAIL_VERSION`, a label that is deliberately
outside the accepted set of both the dashboard and the readiness audit, so an
unpinned production configuration fails closed instead of passing as the product.

### Per-row served-stage status and shadow provenance

The shadow stage had a per-row status while the served stage had none, so a
fallback's reason lived only in the daemon's console log and readiness attributed
every fallback to unavailable weights even though `stack_error`,
`non_finite_center` and `unsupported_model_step` produce it too. The row now
carries `v2_2_status`, and readiness counts it. Three further columns close
provenance and depth gaps: `v23_manifest_sha256` records the digest of the shadow
deployment's own manifest, because the artifact cache is keyed on the directory
and the shadow identity is fixed at build time, so a redeployment into the same
directory was previously invisible; `v23_history_hours` records how many hourly
L1 driver means the analog key could draw on, which is the distance to the
fail-closed `missing_driver_lagN` boundary; and `v23_step1_center_dst_nt` carries
the one-hour center described above.

The analog key's mandatory depth is seven lags, not twelve — the further five
lags feed only the consecutive-southward run length and truncate exactly as they
do at the start of the archive. At a one-hour anchor lag, seven lags reach back to
the issue hour minus seven, plus the ballistic transit and the hourly averaging
window: roughly nine and a half hours of upstream minute data. An eight-hour feed
therefore fails closed with `missing_driver_lagN`, while a ten-hour feed supplies
the mandatory lags and records a truncated run-length window. This is a
correction to the earlier estimate that a ten-hour feed would fail the key.

### Deliberate non-changes

Two audit suggestions were not adopted, and the reasons are recorded so they are
decisions rather than omissions.

`V23_SERVING_REQUIRED_FILES` still lists only the seven files every deployment
carries. Its documented contract is to be independent of which error layers were
selected, and the artifact names are configuration-dependent (`e2_step1.bson` and
`e1_step7.csv` exist only for this candidate), so adding them would make the list
wrong for any other selection. The loader-side check is strictly stronger: it
requires a verified digest row for every artifact *any* configuration names.

The served adaptive-conformal residual stream is still keyed on
`_aci_required_model_version`, which reads the row's `model_version` (`v2.1` for
every served row) rather than the served label. Re-keying it would change the
served band numerics of the transition period, and the direction of the current
mixing is conservative: the pooled band over-covers by at most 1.3 pp. The mixed
pool is disclosed as a known boundary instead.

### The E-layer artifacts must be in the digest-verified set

`V23_SERVING_REQUIRED_FILES` lists the seven files every deployment carries, but
the E-layer models are named per step by `e_layers.json` rather than by that
list, and the loader only checked that the named file existed. A manifest with
the E-layer digest rows deleted therefore loaded and served those models
unverified. The loader now requires every artifact named by the configuration to
appear in the hashed set `v23_serving_verify_manifest` returned, so a trimmed
manifest is a load error.

### Known fidelity boundary of the served component panel

The served center is the stack applied to the live engine's own six component
values. Three of those components are constructed differently from the archived
columns the stack was fitted on: the live core and baseline rollouts admit
L1-measured hours, and the live frozen tail holds the trailing wind hour rather
than the issue record. The static-stack identity oracle therefore establishes
that the stack is *applied* correctly, not that the live component panel equals
the archived one. This boundary is a property of serving a minute-cadence feed
with weights fitted on an hourly archive; it is recorded here rather than
asserted away, and closing it would require either fitting the stack on
live-definition components or serving archive-definition components.

## V2.3 comparator correction

### What was wrong

The direct-GBM comparator regressed the level `Dst(t+h)`. With the preregistered
`nbins = 64` the boosted learner bins each feature at its training quantiles, and
the `dst0` quantile ladder is set by the quiet hours that dominate the archive: its
lowest boundary sits at −59.0 nT, so one bin holds every issue below that, from
moderate storms to the deepest anchor in the record. A level fit therefore cannot
separate a −230 nT issue from a −60 nT issue on the feature that matters most,
and its deepest attainable prediction is close to the mean of that bin. On the
2015 development block the archived comparator never predicted below −112.75 nT
against a deepest observation of −234.0 nT, and it lost to persistence at one
hour (out-of-fold RMSE 6.895 nT against 4.758 nT).

### The fix

The comparator now regresses the increment `Dst(t+h) - Dst(t)` and the reported
center is the fitted increment plus the issue-time Dst. The binning limitation is
unchanged, but it no longer binds the reachable level: the increment is the small,
well-mixed quantity across the whole archive, and the level enters through the
anchor that is added back. The preregistered grid is untouched — depth {4, 6},
rounds {200, 400}, `eta = 0.05`, `min_weight = 64`, `nbins = 64`, fixed seed.
`v23_direct_target` builds the target, `v23_direct_center` inverts it, and
`v23_direct_check_anchor` fails closed unless the anchor is exactly the `dst0`
column of the design matrix, because a persisted model can only be inverted with
a quantity a loader reads from its own feature vector. The persisted artifact
contract is unchanged in shape (`direct_gbm_step<h>.bson`) and now records what
must be added back: `target = increment` and `target_anchor = latest_dst_nt`, in
the configuration parameters, in the run manifests, and in `e_layers.json`.

On the 2015 block, at identical rows (n = 8,739, no fallback):

| step | comparator (increment) | comparator (archived level) | persistence | served V2.1 |
|---|---|---|---|---|
| 1 h | 4.328 | 6.895 | 4.758 | 4.525 |
| 6 h | 12.024 | 12.519 | 13.552 | 13.330 |

On the disturbed subset of that block (latest Dst ≤ −50 nT, n = 710) the
increment comparator scores 7.258 nT at one hour against 7.986 nT for
persistence, and 22.529 nT at six hours against 24.379 nT; its deepest one-hour
prediction is −236.7 nT against the deepest observation of −234.0 nT. Editing the
shared runner source changes `v23_code_signature`, so every persisted V2.3
configuration is already invalidated for resume and no development artifact can
be mixed across the two formulations.

### The 2026-08-17 confirmatory artifacts predate this correction

The confirmatory run archived on 2026-08-17, its `direct_gbm_step<h>.bson`
models, its `e_layers.json`, and every TEST comparator number derived from them
carry the level-target comparator. They are not re-scored by this change and must
not be read as the corrected comparator; the B2 comparator rows of that run
understate the comparator on disturbed anchors for the reason above. A corrected
TEST number requires a fresh confirmatory run under the preregistered
single-shot rule, and the code-drift guard on the development contract will
demand a written reason before that run starts.

## Deployment-boundary correctness: a log that spans its own schema change

### Development Ledger — first-day-of-deployment behaviour

| Item | Record |
|---|---|
| Objective | The served health endpoint, the readiness fallback and shadow windows, and the newest-cycle label comparison must behave correctly during the window in which the hot log contains both pre-stage and post-stage cycles |
| Contract | A trailing window may contain cycles issued by an earlier build; those cycles carry `missing` in the columns that build did not write, and they carry the served label that build published |
| Evidence | The hot log's own schema: the shadow and served-stage columns were appended to a file already being written, so the earlier rows read back as `missing`; the identity writer records an empty shadow digest when no shadow deployment is present |
| Independent oracles | Fixture logs whose generations are known by construction, and mutation of each fix back to its previous form |
| Test plan | Two application testsets on two-generation windows; seven audit self-test cases on staged/pre-stage windows, window policy, issue-hour cycle keying, snapshot refresh and the empty identity digest |
| Data regeneration trigger | None: no served number, artifact or figure changes; the served center, the shadow center and both identity oracles are untouched |
| Harness | `app/test/runtests.jl`, `test/test_v2_readiness_selftest.jl`, `v2_readiness_audit.jl --self-test`, `test/test_live_forecast_verify.jl`, `test/test_serving_identity_oracles.jl`, `test/test_operational_v23_serving.jl` |
| Risk | The window policy is a readiness-verdict change and is documented as such; everything else restores intended behaviour |

### Three-valued logic is the failure mode a mixed-generation log creates

`missing == 1` is `missing`, not `false`, and `any` propagates it: a predicate
written with `==` over a column that a previous build did not write makes `any`
return `missing`, and the next `&&` raises. In the health summary that raise was
caught by the endpoint and served as no served block at all, so the deployment's
first day reported no served identity, no fallback rate and no shadow state —
exactly the day those fields exist to cover. Every predicate over a log column is
now written with `isequal` or `isa`, which are two-valued by construction. The
same rule applies to the served label itself: a row with no label is a fallback
cycle with no reportable identity, not an exception.

### A cycle is published under one label, so it must be described by that label

The stack stage is loaded per issuance and can heal or fail between the horizons
of one cycle, so a cycle can legitimately carry more than one accepted label and
more than one driver-assumption token. Reading the assumption as a common field of
the cycle then finds no single value and reports it as never recorded, which
describes a logging failure rather than the disclosed per-row degradation the log
recorded — and readiness failed the payload for it. The assumption is now read
from the rows carrying the cycle's weakest served label, which is the label the
cycle is published under, so the sentence names the stage the reported product was
actually served by.

### A fallback rate needs a window that can express its target

Two independent problems met in the same check. Cycles issued before the served
stage existed carry the previous label and no served-stage status; counting them as
fallbacks made the rate report the age of the log rather than the health of the
deployment, and for a full window after a deployment onto an existing hot log that
read as a near-total served-stage failure. And a one-day window cannot resolve a
one-percent target at all: one fallback out of twenty-four cycles is 4.2 percent,
so the target could only ever be met by a window containing no fallback, which
makes any single redeploy a FAIL.

The window now admits only cycles that carry a served-stage status, discloses how
many older cycles it excluded, and spans four days. The verdict is stated on cycles
rather than on the rate alone: a fallback on the newest staged cycle fails, because
that is the cycle being served now, and an over-target rate fails once two or more
cycles in the window fell back. One isolated older fallback is reported and passes.
The shadow window follows the same staged-cycle rule, where a staged cycle is one
that records a shadow status — an unavailable shadow path still counts against
availability, while a shadow path that did not exist yet does not.

### One definition of the newest cycle

The audit had two. Stage health and the dashboard API key a cycle on its issue
hour; the newest-cycle check keyed on the newest solar-wind vintage. Under a
stalled L1 feed those disagree: several hourly issues share one vintage, so the
vintage-keyed "newest cycle" pools rows from cycles the API never published, and
the label compared against the payload can belong to a different cycle than the
one served. The newest cycle is now the last issue-hour group, with the
vintage-keyed reading kept only as the fallback for a log with no parseable issue
time. The weakest-label rule lives in one function that both the log check and the
post-request snapshot call.

### The comparison snapshot is a snapshot

The dashboard comparison re-reads the live log after the API request, because a
cycle boundary can fall between the two. The newest cycle's served label is part
of that comparison and is now re-read with the rest of it; a label left over from
the pre-request read would report a mislabelled product on every boundary the audit
happened to straddle. When the newest cycle's label is not one this build accepts,
the stale label is cleared rather than kept, so the payload check warns that no
comparable cycle was available instead of comparing against a cycle it never served.

### Latent hardening: the shadow one-hour cache key

The one-hour pre-layer center is cached per anchor and reused by every horizon of
the cycle. Its key now also carries content hashes of the issue-anchor drivers and
the memory features, both of which are recomputed from the L1 stream at each
issuance and both of which enter the blend. This is hardening, not an observed
defect: the analog-feature hash already in the key changes whenever the L1 stream
advances, so no natural fixture separates the two, and the change is covered
structurally by the key's arity and the hash helper's sensitivity. The served
center and both identity oracles are unaffected — the cache belongs to the shadow
path only.

## V2.4e integration: the served super-learner and its fallback chain

### Development Ledger — served V2.4e center

| Item | Record |
|---|---|
| Objective | Serve the V2.4e center — an NNLS super-learner over ten causal experts, per (model step, regime, depth bin), 0.60 mass floor on a SINDy family that counts the static V2.2 stack — from a verified bundle, with the same number the rolling-origin study scored |
| Contract | One implementation of the center, driven by the live engine and by the offline oracle; every expert either logged or recomputed from logged state; fail closed on an unverified bundle, an incomplete issue-time key, an absent static-stack expert or a non-finite center; the served band is the interval the study calibrated on this center; every stage that can move the reported center is logged, including the physical projection; the published severity **and the published watch edge** are never shallower than what either stage V2.4e replaces would have published |
| Evidence | The rolling study's 12 folds and 623,184 scored rows; its persisted `v2_4_l1_weights.csv`, `v2_4_conformal.csv`, `v2_4_decision.csv`, `v2_4_serve_rule.csv`, fold manifests and `learn_year_*.csv`; the base table and hourly frame by SHA-256 |
| Independent oracles | The builder's fold-2025 replica against the study's own fitted artifacts (weights, half-widths, timescale, boosted configurations, correction label); the offline identity oracle against every scored column at 1e-9 nT; hand-rolled weighted sums, guard arithmetic and interval endpoints in the unit suite; the package's own `v23_direct_features` as the reference construction of the live design block |
| Test plan | `test/test_operational_v24_serving.jl` (arithmetic, cell chain, the configurable guard, the physical projection, interval and its pooled fallback, expert helpers, and one injected bundle defect at a time, each asserted against the message of the check it exercises); served-V2.4e testsets in `test/test_live_forecast_verify.jl`, including a matrix that produces every reachable `v24_status`; app payload/severity/watch-edge/health tests plus an executed extract of the dashboard's capability block; readiness self-test fixtures for the three-label chain, the unstaged newest cycle and the bundle-identity tie |
| Baseline verification | The static-stack center and the V2.1 operator center stay logged per row; the static stack is both expert ten and a depth-safe severity partner, and the study's comparator set is the baseline the decision record scores |
| Data regeneration trigger | A change to `src/operational_v24_serving.jl`, to the bundle builder, or to any expert path requires rebuilding `deploy/v2_4/` and rerunning the identity oracle before the served center may be trusted |
| Harness | `julia --project=. -e 'using Pkg; Pkg.test()'`; `examples/experiments.jl`; `v2_readiness_audit.jl --self-test`; `validation/operational/v2_4_serving_identity.jl`; one scratch-directory live cycle |
| Risk | Environment split between the hourly archive and the L1 feed — the live anchor is Kyoto provisional Dst and the study's is final OMNI Dst, which no offline check can bound; thread-count sensitivity of the boosted fits; a shallower candidate or a narrower band lowering a published warning; an unverified bundle serving silently; the analog key's run-length feature truncating silently under an L1 gap at lags 8–12 (visible as `v24_history_hours < 12`); the study's `SHADOW` verdict on the early era |

### The two experts the live engine has to build, and why they are built that way

Eight of the ten experts are quantities the engine already computes or the bundle already
carries. Two are not, and both have a trap.

The **frozen V2.1 expert** is a rollout that holds the issue driver for every step. The
engine's own `v2_pred_dst_nt` is not that quantity: it admits L1-measured hours into the core
rollout, which is the whole point of the served operator. Handing the stack the logged column
would give it an expert whose weights were fitted on a different number. It is therefore
recomputed through `v23_serving_frozen_center`, exactly as the V2.3 shadow path recomputes
it, and the live test asserts the recomputed value differs from the logged column and that
substituting the logged column moves the stack center.

The **direct increment-GBM expert** needs the 29-column design the study fitted on: the 18
analog features plus eight Dst lags (1–6, 12, 24 h) and three coupling lags. Restating those
definitions in the engine is how the live design would drift from the fitted one, so the
engine builds a 25-hour mini frame — Earth-arrival hourly means where L1 coverage exists,
Kyoto Dst where it is observed — and calls the package's own `v23_direct_features` on it. The
first 18 columns of the result must equal the analog key exactly; a disagreement means the
two blocks saw different hourly means, and the expert is refused rather than served from an
inconsistent state. The increment is inverted with the design's own `dst0`, not with the
row's `latest_dst_nt`, because the model regresses `Dst(t+h) − Dst(t)` and only the design
can say which `Dst(t)` it was fitted against.

### Why the guard became an expert, and what took over depth safety

The earlier candidate took the served center as `min(stack center, static V2.2 stack)` in a
deepening cell. Independent verification of that run showed why that was the wrong shape: the
weights are dominated by moderate rows and the deep cells are data-poor in the early folds, so
the physics composition the static product already encodes was never recovered by the
combination and had to be imposed afterwards — and imposing it still left bootstrap-supported
losses in the deep cells. Amendment A3 puts the static stack into the combination as expert
ten and counts it inside the mass floor, because that product is itself a composition of the
deployed SINDy operators. The optimiser then uses it where it is best and adapts elsewhere,
and the selected variant needs no guard on the point forecast.

The static-stack center remains an input to the served center rather than a continuity
column: a cycle whose stack stage cannot act cannot form the ten-expert combination at all.
The stage fails closed to the V2.1 operator, `v24_status` says
`fallback:static_expert_unavailable`, and the row carries the V2.1 label. This is why the
chain is three stages deep rather than two.

Depth safety did not disappear with the guard; it moved to the layer that publishes warnings.
The severity is taken through `v24_serving_depth_safe_center` as the deepest of the V2.4e
center, the static-stack center and the V2.1 center, so a combination that blends toward a
shallower state cannot lower a warning either predecessor would have raised.

The watch edge needs the same rule applied to the *edges*, not to the center. The first
integration shifted the served edge down by the amount the point was lowered, which reads like
the same statement and is not: the edge a predecessor would have published is its own center
minus its own half-width, and the V2.4e half-width is not that half-width. So
`served_ci05 + min(0, depth_safe − served)` lands at `deepest_partner_center − hw_V2.4`, not at
`deepest_partner_center − hw_partner`. Whenever `hw_V2.4` is the smaller of the two the published
edge is shallower than the predecessor's own — a whole storm tier, in the case the app suite now
pins. The scratch live cycle shows the two widths differing at every issued lead (conformal
shallow 4.86 / 8.01 / 10.02 / 13.09 nT at steps 1/2/3/6 against 4.78 / 7.11 / 8.69 / 10.89 nT from
the shifted frozen tail), which is the point: a rule stated on the centers cannot reproduce an edge
formed from a different width, in either direction.

The engine therefore logs the predecessor edges. The band the pre-V2.4 machinery produces —
the frozen tail shifted onto the served center, or the adaptive band centered on it — is in both
cases a pure translation of a width that does not depend on the center (`_shift_interval_to_center`
adds `lo − rc` to the new center; `_adaptive_conformal_step!` returns `point ± hw` with `hw` read
from the residual history). The predecessor edge is that same width applied to that stage's own
center, so it is derived arithmetically from the band already computed rather than by re-running
the interval machinery. Re-running it would take the forecast-log lock, rebuild or reload the
persisted live state and step the ACI stream once more per partner per row; that step happens to be
non-mutating here only because the interval is requested with a non-finite observation, which is a
property of the call site rather than of the function, and three calls per row would make the served
band depend on it. The published edge is then `min(served_ci05, v2_2_stack_ci05, v2_1_served_ci05)`,
which is idempotent, leaves a deeper served edge untouched, and drops a non-finite partner rather
than propagating it — the same handling the center min uses, because it is the same function.
A row written before those columns existed has no deeper edge on record, so it keeps the shift
rule and the payload discloses `severity_ci05_source = "legacy_center_shift"`.

The guard arithmetic itself is retained and is driven by the bundle's own
`guard.json`: the loader reads `guard_applied` and `guard_reference`, refuses a bundle whose
two fields disagree, and applies the guard only when the deployed artifact asks for it. That
keeps a later guarded variant servable without a source change, and keeps the arithmetic under
test either way.

### Two stacks share one weight schema, so the served cells have to name their own

The study fits and ships three stacks per fold — the unconstrained nine-expert `L1`, the
floor-constrained nine-expert `L1a` and the floor-constrained ten-expert `L1e` — and writes
them into one table whose weight columns cover the widest set. A nine-expert row therefore
carries a hard zero in `w_static_v2_2` and is otherwise indistinguishable from a ten-expert row
whose optimiser happened to put no mass there. Serving such a row would silently drop expert
ten while every other check still passed: the weights are non-negative, they sum to one, and
the three-member floor is satisfied.

The bundle's rows therefore carry `expert_set` and `n_experts`, the fit's own record of what it
was fitted on, and the loader requires the served variant's cells to name exactly the ten served
experts in the served order. The bundle also records the floor group in `guard.json`, which the
loader checks against `V24_SERVING_SINDY_FAMILY`, so a floor that no longer counts the static
stack is refused rather than reinterpreted. The fixture injects both defects, plus a whole table
relabelled as the nine-expert fit, and asserts each one fails to load.

### A fold refit is the only honest way to build a deployable bundle

No rolling fold's training window reaches the end of the data, so no fold's fits are the
model a live engine should load. The builder therefore performs a "fold-2026" refit: the same
analog archive rule, the same refit ridge correction, the same inner-validation choice and
per-step boosted fit, the same climatology timescale, the same NNLS super-learner and the
same conformal calibration, with the training window extended to the base table's own
end-of-data embargo and the out-of-fold pool extended to every persisted fold. Reusing the
code paths is not a convenience: a rebuilt fit produced by different code would not be the
model the study scored, and there would be nothing to compare it against.

Which is exactly what makes the fold-2025 replica valuable. Building the same bundle at the
rolling fold-2025 window puts the builder against a fold the study did score, so its weights,
half-widths, timescale, boosted configurations and correction label can be compared with the
study's persisted artifacts. They agree to 0.0. That comparison is what licenses the claim
that the fold-2026 bundle is the same model at a later window, and it is also what proves the
`v24_direct_stage!` split (into selection, fit and prediction, so the builder can fit without
a scored year) left the arithmetic alone.

### Splitting a study file has a signature consequence

`v24_fold_signature` hashes `v2_4_rolling.jl`, so refactoring that file makes every persisted
fold read as stale even when the arithmetic is provably unchanged. That is the intended
behaviour of a signature, and the escape hatch already exists: a re-run that means to reuse
the folds passes `--reuse-stale-folds="<reason>"`, and the reason here is the fold-2025
replica agreement recorded above. Without the flag the engine recomputes, which is the safe
default and costs hours.

### Boosted fits are only reproducible at a fixed thread count

EvoTrees accumulates histogram sums in parallel, so a fit is bit-identical only at the thread
count it was produced under. The study ran with 8 threads; the builder refuses any other
count unless the operator passes `--allow-thread-drift`, and then records the deviation in
both the manifest and `selected.json`. The identity oracle would catch a drifted fit as a
non-zero deviation on `direct_gbm`, which is the check that makes the refusal verifiable
rather than a claim.

### The served band changed source, and that is a disclosure

A V2.4e row carries the study's split-conformal half-widths for the V2.4e center, stratified
by model step and depth bin. The shifted frozen-tail band and the adaptive band are
calibrated on residuals of the centers they were pooled from; transplanting either onto the
super-learner center would publish an interval nothing has verified. The study's G3 gate
scored the conformal band on this center (pooled coverage 0.890, storm rows 0.823, ≤ −100 nT
rows 0.76), so it is the band with evidence. The switch travels with the row in
`interval_source`, and a fallback row keeps the previous machinery.

That the band changed source is also why the watch edge cannot be the served edge shifted: the
two bands are not the same width, so a rule stated on the center does not carry to the edge. The
predecessor edges above exist because of this.

### The physical projection is a stage, so it is logged

`v24_serving_center` ends in `clamp(·, −2000, +50)` nT and returns `projection_applied` alongside
`guard_applied`. Nothing about the arithmetic changed; what changed is that the reported center no
longer disagrees silently with the combination it came from. The identity oracle asserts the
projection inert on the compared anchors, which is a statement about those anchors and not about
the projection: the study's window never approaches `+50` nT, but a real anchor Dst above roughly
`+52` nT with a panel to match does, and the log had no column that would show it. The invariant
`v24_pred_dst_nt == v24_l1_center_dst_nt` for an unguarded bundle is consequently asserted under
`!v24_projection_applied` rather than unconditionally, and the unit suite exercises the clamp at
both edges on a synthetic panel. The exactly-boundary case is deliberately not used as an
"inert" case: a convex combination whose weights sum to unity only within the shipped tolerance
lands a hair outside a boundary panel, and the projection then legitimately acts.

### Identity checks have to read the artifact, not the constant

`V24ServingArtifacts` is constructed with `identity = V24_SERVED_IDENTITY`, so
`artifacts.identity == V24_SERVED_IDENTITY` is an identity of the source constant with itself. It
appeared in both the engine's `_v2_4_artifacts()` and the readiness audit's `audit_v2_4_bundle!`,
reading like a check and enforcing nothing. Both now read what the bundle records on disk — the
`selected.json` identity, and in the audit also the manifest's `build/identity` row. The loader's
own refusal (`load_v24_serving_artifacts` compares the selection record with the constant before
anything else is parsed) is unchanged and remains the real gate; these are the restatements that
survive a loader change.

The remaining independent statement about which product is being served is the label the served
rows carry, so `audit_served_bundle_identity!` compares the bundle's recorded identity with the
`sub_hourly_model_version` of the newest cycle's rows whose `v24_status` is `ok`. Rows served by an
earlier stage legitimately carry that stage's label and are excluded.
