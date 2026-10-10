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
and no thresholds are invented in their place. The later OMNI diagnostic is
not an admissible receipt-causal fit, prospective storm score, or promotion
decision, and it does not authorize a service change.

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

## Pre-2023 near-Earth predictability diagnostic

The next executable research gate is a deliberately non-promotable driver-level
screen using NASA OMNI HRO2 1-minute data ending before the exposed 2023--2025
archive. OMNI is retrospectively shifted to the bow-shock nose, so this screen
cannot establish historical issue-time receipt causality or authorize a V2.2
artifact. Its narrower purpose is to falsify M2 early if the frozen sparse delay
map cannot improve even an idealized near-Earth continuation problem.

The diagnostic aggregates only jointly finite `BX_GSE`, `BY_GSM`, `BZ_GSM`,
speed, `Vx`, and proton-density minutes into half-open 30-minute bins. A bin is
usable only with at least 15 joint minutes. Missing bins split the time series;
fitting and scoring never build a lag or target across a gap. Training-fold
normalization and all coefficients are fit from earlier contiguous segments
only. Expanding calendar blocks use a 168-hour target embargo. Grid selection
uses origins six hours apart to limit serial duplication, whereas the final
score uses every eligible hourly origin in the already exposed 2017--2020
development interval.

The frozen hyperparameter grid and fourteen-step standardized-state loss are
the M2 plan's exact grid and horizon. The one-standard-error rule prefers fewer
active groups, then larger ridge and threshold values. Sparse M2 is compared on
identical complete histories and targets against transported persistence and a
dense ridge VAR with the same state and lag set. Primary driver checks are
`Bz`, southward `Bz`, and `Ey` RMSE/MAE at 1, 2, 3, 4, 6, and 7 hours; all five
state errors and availability are also recorded. A fixed known-prefix
sensitivity may quantify the value of an already observed upstream segment,
but it is labelled as an oracle and cannot count as a causal forecast.

Independent oracles are: segmented fitting equals ordinary fitting for one
segment; separating two segments removes a deliberately adversarial boundary
transition; persistence repeats the exact last known state; a hand-generated
stable delay map is recovered on synthetic contiguous segments; post-cutoff
state mutation cannot change a fitted artifact; and every scoring row binds a
complete 25-row history plus 14-row target on one 30-minute grid. Any sparse
loss to dense ridge or persistence at `Bz`/`Ey`, any gap crossing, or any
post-cutoff dependence is a diagnostic failure, not a reason to weaken the
contract.

The frozen run used 125 content-addressed HAPI months from January 2010 through
May 2020. It retained 152,154 half-hour bins and 3,011 of 22,677 scheduled
hourly validation origins with complete 12 h histories and 7 h targets. The
one-standard-error rule selected ridge `100`, threshold `0.3`, and five active
groups; the dense comparator used ridge `100` and all 30 groups. Sparse M2 beat
transported persistence in `Bz` and `Ey` at every lead, but it did not beat the
dense ridge VAR. The dense-minus-sparse `Bz` RMSE difference ranged from
`-0.0162` to `-0.0015` nT without an oracle prefix, and the corresponding `Ey`
difference ranged from `-0.0420` to `-0.0018` mV/m. All dense comparisons had
nonpositive one-sided 95% lower bounds. The frozen outcome is therefore
`PREDICTABILITY_NOT_SUPPORTED`; neither the selected sparse driver nor its
artifact is promotable.

As a downstream recoverability check only, the unchanged V2.1 20/11 core was
then driven by the selected sparse, dense, and persistence trajectories on the
same 3,011 origins. Hourly Dst came from the official OMNI2 HAPI product, whose
half-hour midpoint tags were mapped to the corresponding hourly index. The
sparse-driver core ranked first in overall RMSE at every lead (`2.942`, `4.792`,
`5.943`, `6.661`, `7.513`, and `7.857` nT), but its smallest advantage over a
same-driver physics comparator was only `0.012`--`0.151` nT, below the frozen
`0.25` nT margin; several safety strata also favored another comparator.
Sparse and dense end-to-end gates both failed. This result supports testing the
already preregistered M3 error state, but it cannot rescue the M2 driver gate or
authorize service integration.

## Pre-2023 M3 recoverability diagnostic

The next diagnostic keeps the exact 73-feature M3 schema and never treats OMNI
as receipt-causal. M2 trajectories are generated out of fold for the 2013--2017
calendar blocks from driver models fitted only before each block. The current
22-feature block reuses the audited physical and frozen-V2.1 expert values, but
all fields named `primary_minus_*` are recomputed from the matching out-of-fold
M2-driven SINDy center; no fitted static-V2.2 prediction is reused as the M3
primary. Nine H1 innovations are admitted only after their hourly Dst targets
have matured. The five endpoint states and the final center come from the same
M2 path and issue.

Per-lead M3 ridge and group threshold use the frozen `9 × 7` grid. Selection is
nested over the 2014--2017 OOF blocks with a 168 h target embargo, uses bounded
`5 + 5h` corrections, and applies the one-standard-error sparsity rule. The
2017 selection block stops early enough that its latest target is 168 h before
the final diagnostic origin at 2017-10-20 23:00 UTC. Final artifacts fit only
OOF rows whose targets mature by 2017-10-13 23:00 UTC. The already exposed
2017--2020 window is then evaluated once, on complete common anchors, against
the uncorrected M2 center, served and frozen V2.1, raw SINDy, persistence,
Burton variants, O'Brien--McPherron, dense-driver controls, and the rejected
static primary where available.

The available-comparator diagnostic requires at least `0.25` nT RMSE
improvement at all six leads, a simultaneous one-sided 95% calendar-month
block-bootstrap lower bound above zero, no MAE increase, and no RMSE increase
in target-storm, active/deepening, or recovery rows with at least 20 examples.
The source has no frozen event identifiers, so the inherited event/month
inference and full comparator gate remain unavailable. Promotion is false
regardless because the upstream sparse driver gate failed, exact historical
receipt causality is unavailable, the mandatory comparator and ablation set is
incomplete, and no prospective storm-and-recovery cohort exists.

The lag-feature tree comparison reuses the already frozen
recoverability family: depths `(3, 4)`, rounds `(64, 128, 256)`, correction
shrinkages `(0.25, 0.50, 0.75, 1.00)`, learning rate `0.03`, `L2=2`,
`lambda=10`, minimum leaf weight `96`, 64 bins, and seed `22,022,026`. It uses
the identical 73 features, expanding 2014--2017 M2-plus-core folds, contiguous
24 h readiness mask, cap, and unchanged-center fallback. The exposed
2017--2020 point-validation window is diagnostic only.

The completed diagnostic had full 24 h history for `939/3,011` M2-eligible
origins (`31.19%`). Because M2 retained only `3,011/22,677` scheduled issues,
end-to-end readiness was `4.14%`. Fold predictions retain their actual
M2-plus-core identities; the final artifacts bind the final center and the full
cross-fit-manifest checksum. After the disturbed-state safety screen, the
sparse correction remained only at 2 h and reduced RMSE by `0.061` nT relative
to the uncorrected center, while recovery RMSE worsened by `0.126` nT. The other
five leads used exact unchanged-center fallbacks. The tree model kept only its
depth-3, 256-round, unit-shrinkage 1 h model, whose gain was `0.044` nT; it also
fell back at the other five leads. The sparse and tree candidates lost to the
strongest available comparator by `0.137`--`0.350` and `0.093`--`0.350` nT,
respectively. Both simultaneous one-sided lower bounds were `-0.497` nT.

Only one storm row was correction-ready at 1 h and none at 2--7 h; there were
no Dst <= -100 nT rows or storm-event labels. The complete comparator and
SINDy-removal set was unavailable. The formal result is `NO_GO`, not `PIVOT`.
The M2/M3 family is closed to further selection on the exposed 2017--2020 rows.

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
