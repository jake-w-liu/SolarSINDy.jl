# V2.1 migration debug log

## Scope

This log records defects found while migrating the operational package from the
historical V2.0 21-candidate/10-active-term core to the revised V2.1
20-candidate/11-active-term core. V2.0 remains available only through its
explicit historical artifact boundary.

## Confirmed defects and fixes

1. **Bare V2 paths loaded the historical discovery core.** The monitor,
   dashboard, replay code, and examples could resolve unqualified V2 artifacts
   to the 21-term file. A versioned artifact resolver now maps `v2` to V2.1 and
   requires an explicit `v2.0` request for the archived core. It validates term
   order, candidate count, active support, pressure terms, joint-draw shape, and
   removal or presence of `n*V^2`, as appropriate.
2. **Current and historical calibration could be mixed.** Calibration paths are
   now resolved through the same version boundary as the core. V2.0 point and
   conformal files are retained under `deploy/historical/v2_0/`; unqualified
   deployment files are V2.1.
3. **Replay row slicing used a Dst mask on a shorter driver table.** Storm
   windows with unequal driver and Dst support could throw a bounds error or
   select the wrong rows. `_slice_replay_window` now applies independent masks
   and all affected replay paths use it. A regression fixture deliberately uses
   unequal support.
4. **EKF replays omitted a current calibration feature.** The revised
   26-feature schema requires `baseline_spread_nt`; the EKF development paths
   used an older feature tuple. `_v2_calibration_features` now constructs the
   shared feature record, and the EKF scripts use the current core and schema.
5. **Two operational diagnostics called a stale `_run_v2` signature.** The
   research scorecard and sustained-southward-Bz stress replay now pass the
   current core and calibration explicitly. Both scripts execute successfully.
6. **Standalone live verification defaulted to V1.** The CLI default now issues
   V2.1, matching the monitor, API, and dashboard. Explicit `--model=v1` remains
   supported. A regression assertion pins the default issue model to `:v2`.
7. **Experimental tail reports used an ambiguous comparator label.** Look-ahead,
   envelope, fixed-composition, sub-hourly, ballistic, and EKF reports now state
   that they share the revised 20/11 core but replace or omit served-tail
   components. Their comparator is the V2.1 frozen-tail ablation, not the full
   served product, and they are labeled development lineage rather than
   promotion evidence.
8. **The one-minute OMNI HRO acquisition path was not reproducible.** A
   deterministic NASA CDAWeb monthly fetcher now validates month identity and
   file structure and writes SHA-256 provenance. Its initial temporary-file call
   used an incompatible `mktemp` form under Julia 1.12.6; the call was corrected
   and all eight required months were fetched and verified.
9. **Unqualified coefficient and research-result snapshots remained on the
   retired library.** `real_sindy_coefficients.csv` and the operational-paper
   mirrors still contained 21 rows, while phase, coupled, and legacy synthetic
   snapshots still contained `n*V^2`. The real-data snapshots were synchronized
   byte-for-byte from the verified final canonical revision run; synthetic
   snapshots and PlotlySupply figures were regenerated with the current
   identifiable library. A package test now rejects `n*V^2` in every
   unqualified CSV below `data/`.
10. **Paper staging did not validate or mirror the complete core artifact set.**
    The V2.1 staging program now checks the 20/11 point fit, stability table,
    inclusion summary, and 500 joint draws; it checks active pressure terms and
    cross-file equality before atomically staging both source records and
    canonical paper mirrors.
11. **The broad operational replay consumed the superseded storm-catalog
    schema.** Refreshing the package catalog from the revised canonical run
    changed `min_dst`/`min_dst_time` to the scientifically explicit
    `min_dst_star`/`min_dst_star_time`, reduced duplicate/invalid event entries,
    and renumbered event identifiers. The replay now requires the revised
    714-event schema and independently pins its 193/30/8 events below
    -100/-200/-300 nT. A regression fixture verifies that a legacy-schema file
    fails closed instead of silently changing the cohort.
12. **The state-inertia constant still reflected the superseded broad cohort.**
    Re-running the fail-closed selector on the corrected 193-storm archive chose
    a two-hour near-quiet model weight of 0.625 rather than the deployed 0.75.
    The live constant, identity audit, replay self-test, and independent operator
    regression now pin 0.625. All affected operational replays and paper
    products are regenerated from that selected center.
13. **Two final checks encoded environment-dependent or ambiguous V2 labels.**
    The operational-evidence path test inferred precedence from directory
    existence even when the generated directory contained the complete replay
    evidence, and the live comparison report labeled historical rows as V2.1
    whenever the newer served columns were absent. The path regression now
    applies the production artifact-completeness rule. The report boundary now
    labels legacy-only rows as Historical V2.0 and uses V2.1 only when served
    V2.1 columns are present; focused regressions pass for both schemas.
14. **The forecast-log retention fixture still emitted the historical served
    label.** The production validator correctly rejected those rows after the
    V2.1 boundary was tightened, but three retention assertions still built
    their synthetic row with `model_version="v2"` and the older served-model
    label. The fixture now uses the exported current-version constants. Its
    focused 300 assertions and the complete package suite pass without
    weakening the validator. Generic script entry points were also given
    script-specific names so package-owned includes no longer overwrite a
    shared `main()` method.

## Verification record

- Operational artifact-boundary regression: 68/68 assertions passed after the
  snapshot cleanup.
- Development-lineage replays for hourly look-ahead, fixed envelope, and fixed
  A+B composition completed successfully with zero current/frozen fairness gap.
- Measured-timeshift and ballistic sub-hourly component replays completed from
  SHA-verified NASA HRO files and improved over the V2.1 frozen-tail ablation at
  every tested lead; these remain component results, not served-product claims.
- The served V2.1 severe, broad, and exact Kp/G3+ replays use matched rows and the
  explicit historical V2.0 comparator. The complete package suite passes
  4,368/4,368 assertions, the deterministic experiment entry point passes, the
  strict documentation build completes without warnings or errors, the
  development harness records 136 PASS, one justified tolerance advisory, and
  zero failures, and the readiness audit records 132 PASS, four disclosed
  evidence-boundary warnings, and zero failures.
- The operational path regression passed 28/28 assertions, and the complete
  live verification workflow passed 550/550 assertions after the final
  V2.0/V2.1 report-label correction.

## V2.4e A3 causal-history freshness (2026-08-30)

- **Symptom:** Prospective A3 coverage fell to 312/422 (0.7393), below the
  frozen 0.88--0.92 pooled gate. A causal replay using the current Kyoto Dst
  snapshot did not always reproduce the location shift recorded at issuance.
- **Root cause:** The monitor issued forecasts before refreshing observations.
  Although `issue_forecast` already held the current causal Dst snapshot, the
  A3 history helper read outcomes only from the pre-refresh CSV log. Newly
  matured outcomes and revisions therefore reached the shadow history one
  cycle late.
- **Fix:** `_v2_4_calibration_shadow_from_log` now accepts the issuance Dst
  snapshot and overlays matching historical targets before computing the
  frozen trailing-24 median. Missing snapshot targets still use the logged
  observation. Served V2.4e fields and all frozen A3 parameters are unchanged.
- **Files modified:** `examples/live_forecast_verify.jl` and
  `test/test_live_forecast_verify.jl`.
- **Regression evidence:** The new fixture supplies correct current outcomes
  beside stale and missing log values. It errors against the old helper because
  the causal snapshot is unsupported, and passes against the repaired helper
  with exact equality to the clean-log interval. The focused workflow passes
  980/980 assertions; the calibration suite passes 47/47 and the claim-audit
  suite passes 51/51.
- **Scientific result:** Independent arithmetic confirms the recorded A3
  endpoint formula exactly. Replaying current outcomes changes only eight hit
  states, so this defect is too small to account for the observed
  undercoverage. The shadow remains non-serving and fails its prospective
  promotion gates; no parameter was retuned.
- **Lesson:** When issuance has already fetched a causal observation snapshot,
  every issuance-time adaptive statistic must consume that snapshot directly
  rather than depend on a verification write scheduled later in the cycle.

## Live validation and recovery checks (2026-09-07)

### Prospective cohort validation

The original claim auditor accepted inconsistent model steps and Dst-anchor
geometry; its positive test fixture even preceded A3 deployment. It also
aborted on identical pending duplicates because ordinary equality propagated
`missing`, silently treated non-finite observations as pending, and threw on
oversized numeric fields, malformed warm-up counts, and conflicting observations
for the same target.

The repaired auditor checks deployment chronology and exact step geometry,
uses missing-safe duplicate equality, and records malformed or contradictory
evidence as integrity failures. Conflicting observed targets are excluded before
aggregation. Its positive fixture now has actual future targets after deployment.
The final focused claim suite passes 166 assertions; the separate calibration
suite passes 49. No acceptance threshold or tolerance was relaxed.

The operational history reader also accepted fractional steps by rounding them.
It now requires exact step equality and valid issue/anchor/target chronology.
The live-workflow suite passes 990 assertions, including malformed-history and
unused quoted-column cases. Reading only the nine consumed CSV columns cuts the
final paired fixed-snapshot median time from 260.95 to 118.51 ms and allocations
by 86.46% (Julia 1.12.6, two threads, eleven alternating pairs).

An independent raw-row calculation passes 13,484 assertions on the 2,104-row
snapshot with SHA-256
`7251ac06a49ce16b6f9a7ffd15fdae66af22a602186d19302c176610beff5e30`.
The complete claim payload, except its generation timestamp, is identical before
and after the fixes. Its 1,178 matured A3 rows still cover 78.438% of outcomes.
All stored A3 endpoints, hit flags, newest causal histories, and pinned-weight
served centers reproduce. These defects therefore do not account for the
undercoverage in this snapshot; this does not establish absence of every possible
implementation defect.

The 24 recorded integrity violations are six genuine V2.2 fallback cycles during
2026-09-03T23:55 through 2026-09-04T04:05, not an observed bundle replacement.
Their missing-driver-lag statuses and absent A3 endpoints remain in the record.

### Cumulative evidence and missing identities

A synthetic 3,200-row claim-ready cohort plus one earlier bad-manifest row
changes from failed to passing if retention moves that bad row into the archive
and the auditor reads only the hot log. Archived storm events also disappear
from the counts. The claim reader now takes the shared cross-process log lock,
reads all numeric archive segments and the hot log with schema union, and
validates segment order and receipt row/byte accounting. Missing or inconsistent
sources and pending transactions persist a failed claim gate. Old receipts hash
only the last append, so this is not whole-history content authentication.

An empty or missing post-freeze shadow identity similarly hid a row from the
integrity audit; removing the identity column also reported integrity as true.
Empty identities are now skipped only when a valid issue timestamp establishes
that the row predates deployment. The actual snapshot has 860 pre-freeze blank
identities and zero post-freeze blanks; this repair does not change its result.

### Retention, experiment, and dashboard recovery

The deterministic serving experiment compared its deepening label with a helper
call that swapped observed Dst and hourly Dst rate. Its chosen deep, rapidly
falling fixture masked the error because both calls returned true. A flat,
uncoupled state at -120 nT distinguishes them: the correct label is false, while
the swapped call returns true. The experiment now checks independent true/false
expectations for both served states. The deployed label function is unchanged.

A failed hot-log replacement after a successful archive commit duplicates rows
when retention retries: six input occurrences become eight across archive and
hot log. A second reproduction includes intervening real verification and append
operations and duplicates the same two occurrences. Retention now durably
journals the append, pre/post hot-log receipts, and sidecar rebuild. Recovery
finishes before another writer modifies the log. The 323 recovery assertions
cover fifteen commit boundaries, partial appends, process death, contradictory
receipts, and bounded EOF handling. A separate 129-column stress test preserves
100,004 input occurrences and independent full-file hashes in 407 assertions.
With one Julia thread, three warmups and seven working-host samples, normal
retention takes 8.438 s median and archive-committed recovery 3.152 s. Allocated
bytes are 1,907,270,608 and 384,292,992 respectively, not peak RSS. Larger-archive
scaling remains unmeasured.

The app's server flush loop and configured notification loop outlive a closed
HTTP server. Closeable timers and shutdown cleanup have focused regressions.
The desktop startup probe also lacked an HTTP deadline despite advertising a
60-second limit; bounded probes now have a stalled-server regression. One
canonical run reached its compilation cap, and a partitioned diagnostic passed
1,616 assertions but crashed on native process exit. Neither counts as a pass.
The app test wrapper now compiles the same 82 bodies separately, and its
formatting fixture starts no real upstream workers. Two final canonical runs
exit successfully with 1,618 assertions each. Precise native-crash causation is
not established.

The dashboard's historical SWPC cache had been frozen since August 26. Graceful
TERM did not stop that process; dashboard-only launchd recovery installed PID
80690, and fresh SWPC updates resumed at 16:48:09, 16:49:07, and 16:50:10 UTC.
Health remained complete/ok/V2.4e, while monitor PID 8325 and its phase cadence
were unchanged. The historical frozen-worker cause remains unproved.

The first full package run exposed a missing Sockets stdlib test dependency at
the app import boundary. Project.toml now declares it in extras and the test
target; the focused environment suite passes 58 assertions. No test, tolerance,
oracle requirement, or scientific gate was weakened.

The final development harness passes 226 checks with one reviewed tolerance
warning and no failures. Pkg.test passes 284,024 assertions with two Julia threads,
all six registered local-artifact oracles exercised, and zero skips; the
deterministic experiment also passes. The 184-file source/configuration receipt
still matches after execution and deployment. The unique post-restart cycle at
18:05:00.005 UTC preserves all 117 immutable prior columns in 130 assertions;
13,556 independent arithmetic assertions reproduce its claim payload, endpoints,
causal histories, and served centers. The NOAA responses captured before and
after issuance are byte-identical. Strict readiness after the cycle correctly
reports one failure: six historical fallbacks in its trailing
96-cycle window. A3's cumulative integrity failures do not age out with it.

### Unresolved dashboard task-switch failure

Final deployment checks detected an unplanned dashboard exit at
2026-09-07T18:07:03.892Z. Launchd's service-inactive event and the process start
time identify the transition from PID 80690, runs 2, to PID 2118, runs 3, with
last exit code 1. Stderr includes `attempt to switch to exited task` in
`task_done_hook`/`wait`, followed by an atexit error while closing the profile
listener. No new macOS crash report was present. The exact initiating task and
cause have not been established; preceding USGS timeouts are context, not proof
of causation. Production already used normal compilation, not `--compile=min`.

The monitor remained PID 1024, runs 6, and its new-cycle forecast preservation
and arithmetic checks pass. Dashboard API responses recovered and exposed fresh
SWPC values. Recovery does not establish sustained stability.

A bounded isolated probe exercises 8,000 immediate timeout-helper calls, 8,000
deliberately blocked helper timeouts, 800 successful local HTTP requests, and
64 local HTTP read timeouts using HTTP 1.11.0 and ConcurrentUtilities 2.5.1.
Both installed Julia 1.12.6 and 1.12.7 processes exit 0. Worker assertions are
not aggregated into the parent test-summary count; the retained execution logs
and call counts must not be described as a larger package-test total. This
short probe did not reproduce the production failure or verify a repair.

The [Julia patch comparison](https://github.com/JuliaLang/julia/compare/v1.12.6...v1.12.7)
includes a [scheduler queue-growth repair](https://github.com/JuliaLang/julia/pull/62372)
for lost tasks during concurrent pool initialization. That source describes a
hang, not this observed fatal switch to an exited task. Hypothesis: the runtime
version contributes to the dashboard failure. Verification requires controlled
reproduction and a longer real-workload comparison. No production runtime, dependency version,
thread setting, or launch configuration was changed. Full correction remains
incomplete while this operational failure is unresolved.

Raw receipts, replay scripts, benchmarks, and failure evidence are retained in
validation/output/operational/deep_debug_20260907. The parent upgrade proposal
does not authorize a new study, identity, cohort, or served policy.

### Outbound TLS finalizer defect and request-path repair

Subsequent isolated tests established a narrower dependency defect. Installed
OpenSSL 1.6.1 registers `close` as an SSLStream finalizer. Its TLS shutdown calls
the BIO write callback, which attempts a Julia socket write and suppresses any
resulting exception. Instrumenting that callback without writing a log from
inside the finalizer records 100 forbidden task-switch errors in 100 GC
cleanups on both Julia 1.12.6 and 1.12.7. Explicitly closing the same number of
streams records no finalizer writes or errors.

The real HTTP 1.11.0 connection path also reaches this defect: a local server
finishes its TLS handshake after the caller's one-second connection deadline.
Twenty trials check both the client timeout and the eventual server handshake;
discarded OpenSSL streams produce 20 finalizer writes and 19 suppressed
task-switch errors. Repeating the same trials with MbedTLS produces no OpenSSL
finalizer writes/errors, and the cleanup probe finds no remaining TCP file
descriptors. The reproducer deliberately disables certificate verification for
its local self-signed fixture only. It does not alter production trust settings.

These observations agree with the finalizer-I/O discussion in
[Julia issue 59716](https://github.com/JuliaLang/julia/issues/59716) and
[OpenSSL issue 44](https://github.com/JuliaWeb/OpenSSL.jl/issues/44).
They establish the dependency defect and its reachable timeout sequence, not
the precise initiating sequence of the dashboard's historical native exit.
The Julia patch-version comparison does not repair this reproduced defect.

All ten project HTTP.jl request sites now explicitly use
`socket_type_tls=MbedTLS.SSLContext`. The backend was already installed as
HTTP's dependency; root and app projects now declare it directly, with no
package-version upgrade. The global HTTP default is untouched. Existing
timeouts, retries, parsing, status handling, and injection interfaces are
retained; USGS and webhook wrappers expose their actual request boundary to
tests. Scientific arithmetic and frozen identities are unchanged.

The new transport regression exercises trusted localhost HTTPS through the
actual package ingestion path, rejects an untrusted certificate and a trusted
certificate at the wrong hostname, and confirms the global default is unchanged.
It passes ten assertions. Removing only the package backend keyword makes two
of those assertions fail. A SHA-256-signed public localhost certificate uses
HTTP's public test fixture key; the older SHA-1 fixture is rejected by MbedTLS's
normal certificate profile. That setup failure and a rejected `forcenew` pool
option are retained separately, not counted as production defects or passes.
The real NOAA probe passes 15 assertions, and the isolated repaired forecast API
response is byte-identical to the production response on the same live log.

Two old-backend dashboard replicas each had one health-response deadline
failure: one of 480 requests and one of 864. For the second, the handler started
at 19:08:47.233 UTC and returned a 200 response in 8.217 ms; the Node client
received 200 headers at 19:08:47.243 but did not finish reading the body before
19:09:02.225. The other 47 concurrent requests completed successfully. This
locates the observed delay after handler calculation, but does not distinguish
server delivery from client consumption. The diagnostic connection timer had
also stopped logging at 18:52:09; a later native sample does not identify that
waiting Julia task. Instrumentation effects and exact causation remain
unresolved. Both diagnostic servers were intentionally TERM-stopped, exit 143.

At 19:33 UTC, the repaired, unmodified app was undergoing a 90-round real-upstream test on a
separate loopback port, without a supervisor restart. Each round has eight
clients and 48 concurrent complete-JSON requests with 15-second deadlines;
rounds are one minute apart. At 19:33 UTC, 28 rounds / 1,344 requests have no
failure. The completed workload and deployment are recorded below.

The first post-TLS full harness found one independent test defect: a launchd
assertion counted a service-name substring over whole logged commands, so
`dashboard` in `TMPDIR` caused six matches instead of four. The same test fails
under that path and passes under `/tmp`. The replacement checks exact command
targets under a fixture path containing all three service names, preserving
the twelve-command requirement. All 193 assertions pass; restoring the old
substring count now fails three assertions even under `/tmp`. The first full
harness (284,138 passes / one failure; experiments passed) remains recorded as
failed. The full rerun against the new 189-file source receipt subsequently
passes 284,139 package assertions in 17m38.4s, with all six local-data oracles
exercised and no skips. Experiments pass; the harness exits 0 with 244 PASS,
one reviewed tolerance warning, and zero failures. Every source/configuration
hash still matches after that run. Those results precede the deployment below.

A further local scheduler control completes sixty late handshakes per backend
(240 assertions each) and exits 0. OpenSSL records sixty GC writes / 59
suppressed errors; MbedTLS records none. Independent Node HTTP clients overlap
the last approximately 25 seconds of handshake activity and then continue:
each finishes 1,440 exact-body requests over three minutes without failure.
Both one-second timers keep firing. An initial diagnostic-client assumption
that every response had Content-Length was corrected after direct curl
inspection showed valid chunked framing; the original eight mismatches per
backend are retained as test-setup failures. This narrower control does not
reproduce the historical crash or body-consumption timeout.

### Completed workload, deployment, and external timing limitation

The unmodified repaired server completed all 90 rounds at 20:35:33 UTC:
4,320 requests, zero transport failures, one unchanged unsupervised PID.
All health/status/forecast responses remained available. The dB/dt endpoint
reported unavailable in eight rounds (64 responses), and network in round 1
(eight responses); these responses still completed within the deadline. The
server was deliberately TERM-stopped afterward, exit 143. The raw transcripts
and independently checked summary retain these availability gaps.

The dashboard and monitor were deliberately reloaded at 20:40:03 and 20:41:43
UTC. Launchd now runs PIDs 89275 and 89363 with unchanged plist hashes. Startup
skipped all four pending forecast duplicates and finished at 20:43:09. All
2,120 forecasts and 117 immutable columns were preserved (120 assertions).
Four outcomes matured and four were revised. External collection retained
all 49,667 prior rows and 15 immutable fields (17 assertions), added 38 rows,
and updated 64 prior score records. The deployed dashboard completed another
240 requests without transport failures or unavailable data. Source checksums
still match the full-harness tree.

The new independent arithmetic replay passes 13,648 assertions; A3 covers
941/1,198 outcomes and retains all 24 historical integrity violations. The
strict readiness audit at the documented status endpoint reports 162/9/1,
with historical fallback availability still failing. Wrong-endpoint diagnostic
invocations are retained separately, not reported as code failures or passes.

A separate external-baseline issue is not covered by those passing code tests.
The 20:12:57 receipt contains 4,040 of 49,667 targets fetched at or after their
target times; 4,025 were already scored. The current summary is source-issue-
relative, not receipt-causal. A two-row fixture fails three independent
receipt-causal expectations. The default fetched timestamp precedes HTTP I/O,
so historical completion times cannot be inferred from it. Five raw-response
hashes match. The post-reload archive has 49,705 rows, including 4,046 known
late rows and 4,036 scored late rows; the source records were preserved.

This external source does not feed A3's arithmetic. The user was asked whether
to correct prospective scoring and future completion-time provenance while
preserving all raw history. No answer has arrived, and no scoring/cohort change
was made. Full correction remains incomplete pending that decision. The
interval-first model-upgrade proposal is also unapproved and unimplemented.

## Approved external receipt correction (2026-09-08)

The user approved exclusion of late-collected targets from prospective external
scores and recording completion timestamps for new captures, while preserving
raw history. This supersedes the external approval blocker above; it does not
authorize the separate model-development study or any A3 cohort change.

The original two-row failure was reproduced again before editing: three failed
assertions and RMSE 70.7424908 nT instead of the receipt-future row's 3 nT.
That old reproducer remains unchanged as dated failure evidence. The approved
contract preserves both source-issue-future raw rows, so its new regression
requires two archived records but only one prospective score, exactly 3 nT.

New captures record receipt_completed_utc after the forecast body and any
source-run metadata request finish. Eligibility requires issue and fetch-start
at/before completion, with completion strictly before target. Milliseconds are
retained. Legacy logs gain a nullable column; original fields and first-receipt
identity remain unchanged. Missing completion is not reconstructed. Historical
error fields remain archived but do not enter the new prospective metrics.
Collector and readiness use the same timing and summary functions; independent
tests use hand-set errors and boundary expectations.

The first renewed collector test exposed another concrete defect: entirely
unscored CSV columns reload as MissingVector and reject their first numeric
assignment. The scorer now widens only those all-missing score columns to the
existing nullable numeric/string types. The failed 271-pass/one-error transcript
is retained; the exact self-test and explicit round-trip regression now pass.

Focused verification passes 75 receipt assertions, 272 collector assertions,
and 27 readiness assertions, together with the monitor's retention suites.
Seven independent mutation runs fail as required: removing scoring or summary
eligibility, accepting target equality, substituting fetch-start, sampling the
clock before source metadata, inventing a legacy receipt, or removing column
widening. A fresh truth-table/repeatability pass checks 750 boundary combinations
and 15 real-archive assertions without finding another defect.

The isolated migration passes 3,821 assertions on all 49,941 preserved rows:
every original column and all 1,878 raw hashes remain intact. Of these rows,
4,081 are known late and 45,860 lack completion evidence; neither group has
prospective scores. Three new controlled records produce one eligible score
of 3 nT after maturation. A subsequent pre-deployment copy preserves the
04:05 live archive's 49,968 rows and 1,879 raw responses.

Seven repeated summaries of the 49,941-row archive have identical outputs and
leave input values unchanged. Median elapsed time is 0.1463 seconds; steady
allocations are 75,318,752 bytes. This is a bounded resource check during
concurrent system work, not a comparative speed or leak-freedom claim.
The separate convention scan's 18 warnings are false-positive imaginary-unit
matches: five real integer-index expressions, eleven join calls, one JavaScript
string variable, and one deliberately malformed timestamp string. The Python
figure guard passes; no figure or manuscript was changed.

The independent 04:05 live replay passes 13,916 assertions, reproducing all
current A3 statistics, 1,226 interval checks, newest residual-history medians,
and 1,940 exact-served center/interval calculations. The maximum high-precision
stack difference is 7.11e-15 nT. The independently fetched Kyoto body is a later
cross-check, not a contemporaneous issuance receipt. A3 coverage is still
962/1,226 = 0.7846655791, with all 24 historical integrity violations retained.
Strict readiness reports 163 PASS / 12 WARN / 0 FAIL: the trailing fallback
window is now clear, while cumulative calibration/storm claims remain false.
The dashboard completes another 240 requests with no failure or unavailable
response and retains PID 89275.

Evidence is under
validation/output/operational/deep_debug_20260908_external_timing/.
The completed full package suite passes 284,230 assertions in 25m27.6s, with
all six local-data oracles exercised and no skips. Experiments pass; the harness
exits 0 with 255 PASS / 1 reviewed tolerance WARN / 0 FAIL. All 191 source hashes
match. Monitor PID 69932 / runs 8 finishes the repaired collector at 04:47:29
and its startup cycle at 04:47:32. Forecast preservation passes 120 assertions;
external preservation passes 3,937. All 49,968 old external rows are unchanged,
29 are added, and none are deleted. The 21 new receipt-future rows have no
recorded prospective score at capture. Full before/after raw copies are retained.
Post-reload arithmetic passes 13,936 assertions and reproduces A3 coverage
964/1,230, with the same cumulative integrity failures. The next scheduled
cycle is 05:55 UTC. Detailed receipts and verification limits are recorded in
TEST_REPORT.md.
