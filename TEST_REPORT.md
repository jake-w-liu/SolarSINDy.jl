# SolarSINDy Research Test Report

## Julia 1.12 release verification — 2026-09-19 UTC

Package, dashboard, documentation, launchers, images, and CI now support only
the Julia 1.12 series, with the minimum patch specified in Project.toml.
The focused environment/deployment checks pass 92 assertions; CLI checks pass
54, and the added pinned-release failure/success fixture passes nine. The
dashboard suite and strict documentation build also pass on Julia 1.12.7.

The receipt-time candidate and daily-report results are recorded below. The
container build contract includes source-before-precompile, runtime data and
audit-script availability, a depot writable by the non-root user, and a
reachable Compose dashboard address. Actual image build and startup results
come from the separate GitHub deployment-image job, not from static assertions.

The final full-package, experiment, harness, live-reload and pushed-commit CI
receipts are collected in
`validation/output/operational/live_upgrade_20260919/RELEASE_VERIFICATION.md`.
An earlier local full run rejected a changed Git baseline after an in-run
commit; it is not passing evidence. Its provenance checks remain unchanged.
The subsequent full run must keep the checkout revision fixed throughout.

The next GitHub run exposed GNU-stat output contamination and three ARM-only
golden literals. Their reproduction and repair are recorded under "Linux
portability checks" in CODE_NOTES.md. The watchdog and CLI regressions cover
both native stat and GNU stat. Exact forecast continuity is checked against a
hash-pinned pre-refactor function on ARM and x86; the original ARM numeric pins
remain. The dated release receipt records the subsequent full local and GitHub
results without rewriting the earlier failed runs as passes.

The mobile inspection additionally reproduced overlapping date labels and
overflowing long model text. Each new frontend assertion failed before its
repair; all nine frontend tests pass afterward. The dated release receipt
also records actual label geometry and screenshots across four viewport widths.
The in-progress full run was stopped before these edits, not counted as a pass.

Earlier checkpoints below retain their original dates and outcomes.

## Recovery verification (2026-09-19)

The [remediation report](validation/output/operational/deep_debug_20260919/remediation/REPORT.md)
contains independent USGS finite differences, source-selection boundary tests,
ground-only model and causal expert-weight oracles, three storm-window checks,
calendar-day uncertainty, browser checks, and the complete development harness.
No historical forecast or acceptance tolerance is changed. Numerical model
failures remain separate from successful software tests.

## Bounded upgrade studies completed — 2026-09-08 UTC

The point-error and preceding interval-only studies are complete. Neither
selects a candidate, and no new shadow or serving change is authorized.
The [point-study result](validation/output/operational/v2_4_point_upgrade_20260908/RESULTS.md)
contains all six candidates, annual folds, feedback-delay comparisons, support,
storm/state results, tails, uncertainty, and measured execution costs.

| Current verification | Result |
|---|---|
| Complete input/baseline audit | 3,350,062 assertions |
| Focused point/comparison tests | 633 assertions |
| Mutation sensitivity | All 16 kernel/metric and ten comparison/selection mutations detected |
| Full package regression after scientific-source repair | 285,056 assertions; all six required local-data oracles, zero skipped |
| Development harness / deterministic experiments | 295 PASS / 1 existing tolerance WARN / 0 FAIL; experiments pass |
| Development prediction / summary reconstruction | 60,785,135 / 3,875,252 assertions |
| Validation prediction / summary reconstruction | 60,301,687 / 3,866,424 assertions |
| Independent decision | 4,917 assertions; 1,632 criteria; no candidate selected |
| Post-checker-repair full package regression | 285,056 assertions in 21m41.5s; experiments pass again |
| Date-label regression / round-trip fixtures | 78 / 322,784 assertions |
| Exact diagnostic projections / displayed report tables | 495 per partition / 65 assertions |

The two-expression hashing repair preserves exact outputs; the scientific
source manifest still matches all 198 entries. All nine CSVs from the
interrupted development attempt are byte-identical to the completed run.
The independent summary check initially stopped because CSV inferred Date
labels. The normalization repair passed the old/new regression, full fixtures,
both complete study partitions, and another package regression. No numerical
formula, input, candidate, tolerance, or scientific criterion changed for it.
See the [hashing record](validation/output/operational/v2_4_point_upgrade_20260908/OPTIMIZATION_REPORT.md)
and [checker repair record](validation/output/operational/v2_4_point_upgrade_20260908/DEBUG_LOG.md).

At the 12:06:13 UTC operational capture, both service PIDs and plists are
unchanged. Forty-eight dashboard requests have zero errors or unavailable
responses; four served static assets match disk. Readiness reports 163 PASS /
11 WARN / 0 FAIL. Independent A3 arithmetic passes 6,296 assertions and
reproduces 985/1,258 covered outcomes. Both claim flags remain false, and all
24 historical integrity violations remain. Preservation checks pass 312
assertions over the original 117 immutable fields and seven normal added cycles.
Submitted-tree and upstream-main checks remain unchanged. No restart, extra
issuance, pull, push, or dependency update occurred. The earlier 11:58 workload
had eight unavailable dB/dt responses despite healthy core endpoints; USGS
timeouts were logged. The panel recovered without intervention. That record
is preserved separately from the latest all-available workload.

The convention scan's 19 imaginary-unit flags are reviewed false positives:
five bound loop indices and fourteen string/identifier matches. The Python
figure guard passes. No manuscript, figure, or publication data was changed.
Three optional generated historical comparisons lack their separate local
inputs; their paths are retained in the transcript. This does not describe
the complete archive audit or the six required local-data oracles as skipped.
The existing tolerance warning remains disclosed, not suppressed.

## Point-study implementation history

The dated records below preserve the earlier verification sequence and counts.
Their descriptions of unfinished work refer to those earlier checkpoints;
the completed outcome and current checks are above.

The [point-study protocol](validation/output/operational/v2_4_point_upgrade_20260908/PROTOCOL.md)
is frozen. Input and baseline verification passes 3,350,062 assertions over
623,184 historical forecasts, 103,864 anchors, 216 published summary rows, and
the 1,944-row live panel. The live frozen V2.1 expert is independently rebuilt
with 256-bit core/ridge arithmetic and checked against the ten-expert sum. It
differs from the logged L1-admitting ablation on all 1,944 rows; both quantities
remain separately identified. The live panel has zero storm target outcomes,
although eight issue-anchor rows are at or below -50 nT.

The point replay, metric arithmetic, and event grouping have 561 focused passing
assertions. Means, medians, caps, projected centers, asymmetric bands, warmup,
future/revised/delayed witnesses, model-epoch and step isolation, input guards,
historical column mapping, and unequal bootstrap-block sizes have independent
expectations. Point-error bias is observation minus prediction. Tolerances are
1e-12 nT for elementary independent summations and 1e-9 nT for the established
serving-identity comparison against separately ordered high-precision arithmetic.
Integer, timestamp, count, sign, rank, and deliberately representable worked
examples use exact equality.

The first input-audit run exposed a reversed bias convention in the new audit
code; no source data or model was changed. A focused test fixture initially
converted a Boolean to Float64 before the input guard saw it; the corrected
fixture preserves its type. The model-epoch expected count was rederived as four
targets, including the current anchor. A surviving storm-gap mutation exposed
a weak test; five longer gapped sequences now distinguish 72 known quiet hours
from 72 consecutive quiet hours. Earlier transcripts remain preserved.

The separate comparison/selection layer adds 68 passing assertions. A missing
comparator-row test exposed an incomplete decision-input check; selection now
requires every comparator and candidate, unique entries, equal scored counts,
equal row-key digests, and valid corrected counts. The initial thirteen
replay/metric mutations and ten comparison/selection mutations cause assertion failures.
The independent prediction/statistics checker passes 322,780 fixture assertions;
the independent decision checker passes 9,838 fixture assertions. No production
calculation functions are called by these checkers when verifying study outputs.

The first full package run passes 284,934/284,934 assertions in 22m43.9s, with
experiments and a 276 PASS / 1 existing tolerance WARN / 0 FAIL harness. Its 195
source entries still matched afterward. The comparison/selection suite was
then registered in Pkg.test and its no-winner smoke added to experiments.
The integrated run on the earlier 198-entry source tree was deliberately stopped
after a further input check exposed an erroneous baseline-range restriction.
Three archived 2024 V2.4e values exceed 50 nT, while the frozen protocol restricts
corrected outputs and explicitly retains sparse-history baselines. The input
restriction and two mistaken test expectations were replaced by exact
fallback/projection checks for every candidate and both signs. Finite-width
guards were also applied before fallback. The original source and interrupted
transcripts remain preserved. No source data, residual cap, corrected-output
projection, numerical tolerance, candidate, or advancement threshold changed.

All 12 source folds now pass the panel guards and 36 extra interval/comparator
assertions. Focused tests pass 629 assertions. All 16 kernel/metric mutations
and all ten comparison/selection mutations are caught. Independent prediction
and statistics fixtures again pass 322,780 assertions with out-of-range inputs
in both sparse and corrected states. The integrated run passes 285,052/285,052
assertions in 23m17.6s, all six required local-data oracles (zero skipped), and
deterministic experiments. Its harness reports 287 PASS / 1 existing tolerance
WARN / 0 FAIL. The three optional generated historical-artifact comparisons
still lack their separate local inputs; this does not describe the complete
archived-input audit above as skipped or substitute for those missing checks.

The first development command stopped before creating its output directory.
The launcher's cutoff variable shadowed Base.split; the new direct preflight
regression also exposed a parent-directory error caused by a trailing slash.
Both launcher-only defects are corrected without changing any of the other
197 frozen files. Ten boundary/source checks pass. The required full rerun
passes 285,051 assertions in 22m12.4s, experiments, and a 293/1/0 harness.
All six local-data oracles ran, with zero skipped. The one-count difference
from the preceding run is its optional empty-skips ledger assertion: the
latest invocation omitted SOLARSINDY_REQUIRE_LOCAL_ARTIFACTS. The full printed
ledger verifies the same six-oracle coverage; no test was removed. Previous
passing evidence and the failed launch remain preserved with explicit suffixes.
Strict syntax passes 279 assertions over 278 files; sealed-source preflight
passes ten. Its first boundary test expected ENOTDIR, but Julia's recursive
mkpath correctly raises EEXIST on the regular-file parent, independently
reproduced against the standard library. Development evaluation started at
10:15 UTC, then stopped for the hashing repair recorded above; validation and
whole-output reconstruction were not yet complete at that checkpoint.
No point candidate has been selected or promoted. Results, input receipts, and test logs
belong under validation/output/operational/v2_4_point_upgrade_20260908; the
historical completed interval-stage receipt below remains a separate result.

At the 2026-09-08T09:05 UTC capture the dashboard API reports status ok, a complete cycle,
no outage, and no cached-log fallback. Monitor PID 69932 and dashboard PID 89275
are unchanged. The scheduled 08:05 cycle completed at 08:05:26 with 2,168 log
rows and 16 pending outcomes. Neither service was restarted in this study.
Readiness passes 163/11/0; those warnings remain distinct from scientific
claims. The 48-request dashboard check has zero errors or unavailable responses.
Independent A3 arithmetic passes 6,216 assertions and reproduces coverage
969/1,242, both claim flags false, and all 24 historical violations. The original
117 immutable forecast fields survive all three additional normal cycles
(204 passing preservation assertions). Submitted-tree and launchd-plist hashes
remain unchanged. Remote-main checks at 09:16 show no missing upstream commits;
the tested dependencies and existing local changes were preserved.

## Completed interval-development stage — 2026-09-08 UTC

The bounded six-candidate study is now authorized. Its protocol and input are
frozen in validation/output/operational/v2_4_interval_upgrade_20260908.
No serving or A3 change is authorized by a retrospective result.

The new module has 193 passing focused assertions; the existing calibration
module retains all 49 passing assertions. Independent hand calculations cover
translated asymmetric bands, finite-sample ranks, all six trailing windows,
issued-location standardization, and paired interval scores. Five-seed scaling
checks, chronology/revision mutations, per-step isolation, pending observations,
duplicate records, invalid inputs, uneven day-block weights, fixed selection,
and immutable persisted outputs cover the remaining public behavior. Roundoff
tolerances are limited to 1e-12 absolute / 1e-14 relative for independently
ordered floating-point arithmetic. Exact integer/rank and timestamp checks do
not use tolerances. A 32-ulp allowance at the 1.25 width boundary accommodates
mean-of-width rounding and does not change the statistical gate.

The one-time validation and independent oracle select no candidate. No change
was made after opening validation. The coverage/width failures and missing
step-7 witness support are documented in
[the study result](validation/output/operational/v2_4_interval_upgrade_20260908/RESULTS.md).
The interval stage is complete. The separate point-error study subsequently
completed with no qualifying candidate, as recorded above.

| Verification | Result |
|---|---|
| Independent development / validation replay | 242,418 / 332,969 assertions pass |
| Independent development / validation witness and control diagnostics | 18,667 / 30,885 assertions pass |
| Pre-miss location/scale diagnostic | 11,484 assertions pass |
| Mutation sensitivity | Nine deliberate faults cause actual assertion failures |
| Repeated development replay | Five runs; all 15 output-byte checks pass |
| Full Pkg.test | 284,423 / 284,423 in 23m20.4s |
| Required local-data oracle ledger | Six exercised, zero skipped |
| Deterministic experiments | Serving/predecessor and new interval arithmetic checks pass |
| Development harness | 265 PASS / 1 reviewed tolerance WARN / 0 FAIL; exit 0 |
| Source/configuration receipt | All 193 entries match after the harness |
| Latest normal-cycle preservation | 130 assertions; four new forecasts, 117 prior immutable columns unchanged |
| Current A3 claim and issued-band arithmetic | 6,176 assertions pass |

Two diagnostic-runner failures were corrected: include_string did not resolve
a relative helper include from the simulated source path, and a source-manifest
reader retained leading whitespace in paths. Corrected runs pass; failed
transcripts remain. The valid manifest is verified_source_2.sha256. The initial
empty verified_source.sha256 is not verification evidence. The source capture
was corrected while the unchanged package suite was already running.

The one harness warning is the existing, reviewed tolerance scan. Three
optional historical generated-artifact comparisons were unavailable, distinct
from the six required local-data oracles that all executed. The convention
scan's 18 matches are reviewed false positives. No tolerance was loosened,
and no manuscript, bibliography, or figure appearance was revalidated.

The external archive cross-check passes 51,976 assertions over 50,000 rows and
1,883 raw-response hashes. Its 21 eligible SWPC scores share one target outcome
and one receipt, not 21 independent forecasts. Strict readiness at the start
passes 163 checks with 12 existing warnings and zero failures; the end check
passes 163/11/0. Start and end dashboard checks each have 48 successful
responses with no reported unavailability. Monitor PID 69932 / runs 8 and
dashboard PID 89275 / runs 4 were not restarted, and their plists are unchanged.
The end external archive again passes 51,976 independent checks. A3's current
966/1,234 coverage and 24 cumulative violations do not change either false
claim flag. Complete start/end captures and detailed limitations are retained
with the study result. Neither repository is missing upstream main commits;
the package remains one local commit ahead, with existing dirty work preserved.

Development harness complete for the interval implementation: 2 diagnostic
issues detected → 2 confirmed → 2 fixed, 0 require user repair action.
No candidate qualifies; a new prospective shadow or serving promotion remains
unauthorized. The upgrade goal continues with the separate point-error study.

## Historical external timing repair — 2026-09-08 04:47 UTC

The user approved the external scoring/provenance correction. Two confirmed
defects are fixed: prospective summaries admitted late-retrieved targets, and
entirely unscored CSV columns could not accept their first numeric score after
reload. The earlier approval blocker is resolved. The model-development study
was still unapproved at that repair's completion; the later study authorization
is recorded above. Served V2.4e, A3, its cohort, and all gates are unchanged.

New external rows carry a millisecond-preserving receipt_completed_utc sampled
after the forecast body and any source-run metadata request. Prospective scores
require issue/fetch-start at or before completion, strictly before target.
Historical timestamps are not reconstructed. All original records and existing
score values remain preserved; legacy rows lacking completion evidence and
known-late rows are excluded from prospective metrics. Repeated retrieval keeps
the first record's timestamps. Collector and readiness share this rule and its
reporting; the separate test oracles use hand-derived expectations.

| Verification | Result |
|---|---|
| Original defect reproduction before editing | Three failed assertions; two-row RMSE 70.7424908 nT instead of the receipt-future row's 3 nT |
| Receipt, collector, readiness regressions | 75 / 272 / 27 assertions pass |
| Mutation sensitivity | All seven deliberately removed safeguards cause regression-test failures |
| Isolated full-archive migration | 3,821 assertions; all 49,941 old rows, all 20 columns, and 1,878 raw hashes preserved |
| Fresh boundary/resource pass | 765 assertions; 750 clock-boundary cases and seven identical, input-preserving real-archive summaries |
| Real-source preflight | 93 assertions; SWPC 24 eligible / 2 late, Temerin–Li 0 eligible / 3 late |
| Full Pkg.test | 284,230 / 284,230 in 25m27.6s; six local-data oracles exercised, zero skipped |
| Deterministic experiments | PASS on the frozen V2.4e operator |
| Development harness | 255 PASS / 1 reviewed tolerance WARN / 0 FAIL, exit 0 |
| Source/configuration receipt | All 191 hashes match after the harness and before deployment |
| Main forecast preservation after reload | 120 assertions; all 2,152 rows and 117 immutable columns preserved |
| External deployment preservation | 3,937 assertions; all 49,968 prior rows and all 20 original columns preserved, 29 new rows, no deletion |
| Post-reload independent live arithmetic | 13,936 assertions; claim payload, A3 endpoints and histories, and 1,940 exact-served rows agree |
| Strict post-reload readiness | 163 PASS / 12 WARN / 0 FAIL at the documented status endpoint |
| Deployed dashboard workload | 240 requests before the monitor reload and 48 afterward; zero failures or unavailable responses; same dashboard PID |

The first renewed collector run passed 271 assertions but errored on the
all-missing CSV column case. Its transcript is retained; the corrected self-test
and explicit round-trip regression pass. The original pre-fix reproducer is
also retained unchanged. Its row-deletion expectation is not the approved
repair contract: the new tests preserve both raw records and exclude only the
unsupported prospective score.

The harness's tolerance warning is the existing scan requiring review of
documented statistical, optimization, quadrature, and roundoff tolerances; no
tolerance was loosened. The convention scan's 18 imaginary-unit warnings are
false positives: integer indices, join calls, a JavaScript string variable,
and malformed timestamp test text. The Python figure guard passes. No figures
or manuscript text were changed or visually revalidated.

The monitor was deliberately restarted through launchd at 04:45:45 UTC and
completed its startup cycle at 04:47:32. It runs as PID 69932 / runs 8 under the
unchanged plist, Julia 1.12.6, and two-thread setting. All four pending issue-hour
duplicates were skipped. Four outcomes matured and four prior observations were
revised; no forecast changed. Dashboard PID 89275 / runs 4 was not restarted.
The next normal cycle at that checkpoint was 05:55 UTC; no extra issuance was required.

External records now total 49,997: 21 receipt-future, 4,092 known-late, and
45,884 unknown-completion legacy rows, with zero invalid chronology. The 29
new records include 21 eligible and eight late rows. No new prospective score
had yet been recorded; old numerical errors remain archived, not prospective
evidence. The complete before/after copies preserve 1,879 and 1,881 raw
responses. Existing rolling retention limits are unchanged.

A3 covers 964 of 1,230 outcomes (78.37398%), with day-block interval
[0.697019773, 0.857816236], width ratio 1.219118670, and mean paired
interval-score difference -12.147034645 nT. There are 297 complete cycles,
14 consecutive days, zero qualifying storms, and the same 24 cumulative
integrity violations. Both claim flags remain false. The maximum independent
256-bit stack difference is 7.11e-15 nT. The newest-history cross-check uses a
separately captured later Kyoto body that reproduces the stored medians; it is
not a contemporaneous 04:05 issuance receipt. Historical revision vintages and
the exact cause of earlier native process failures remain unproved.

The six historical fallback cycles have left readiness's trailing 96-cycle
window. They have not disappeared from the cumulative A3 record. Passing
operational readiness does not establish calibration or storm skill.

The live CSV SHA-256 is
74fb77c68fd6e78ad850807da1e6d5f7cc7cd53744e102642720f7f9817958f0.
The external CSV SHA-256 is
7261de8fbf6155202b12d3d518635f03fd2aa8934f3f37749247d88220c6831d.
The submitted-tree aggregate remains
7dfd4556fe8b6f1ae133be80a9a878d255df4ea1b715ef273d257520e728ca74.
All evidence, including failed runs, is retained in
[the dated repair directory](validation/output/operational/deep_debug_20260908_external_timing/).

Development harness complete for this repair: 2 detected → 2 confirmed →
2 fixed, 0 repair decisions require user action. The bounded fresh pass found
no further confirmed defect. This is not proof of indefinite uptime, every
historical crash cause, or universal code optimality.

## Live-validation verification — 2026-09-07 UTC

### TLS repair verification — 19:50 UTC

The current source selects MbedTLS for ten outbound HTTP request boundaries.
Focused verification passes: trusted/untrusted/hostname HTTPS cases (10),
standalone dashboard (1,635), realtime ingestion (182), live workflow (996),
external snapshot collector (272), L1 receipts (326), and prospective issue
capture (299). The real NOAA transport probe passes 15 assertions. Removing
the backend selection makes two of the ten transport regressions fail.

The first post-change full package run reports 284,138 passes and one failure;
the harness exits 1 with 236 PASS / 1 WARN / 1 FAIL. The deterministic experiment
passes. The failure is a path-dependent launchd test assertion, independently
reproduced under a `TMPDIR` containing `dashboard` and absent under `/tmp`.
Its replacement checks exact command targets under a deliberately confounding
fixture path and passes all 193 launchd assertions. The complete new harness
passes: 284,139 / 284,139 package assertions in 17m38.4s, all six local-data
oracles exercised with no skips, deterministic experiments successful, and
244 PASS / 1 reviewed tolerance WARN / 0 FAIL. The harness exits 0 and all
189 source/configuration hashes still match afterward. This is the
current-source pass; the earlier failures remain disclosed separately.

The isolated workload and subsequent deployment results are recorded below.
No runtime upgrade or scientific-policy change was made. Precise historical
crash causation and indefinite production stability are not established.

### Scheduled cycles and additional evidence finding — 20:18 UTC

The normal 19:55 and 20:05 cycles each preserve all 117 immutable prior columns
in 130 assertions. Their independent raw-row checks pass 13,612 and 13,628
assertions; the NOAA bodies bracketing each issuance match. The 20:05 receipt
contains 2,120 raw / 1,908 exact-V2.4e rows and 1,194 scored A3 outcomes, with
936 hits (78.39196%). The frozen identities, phase minutes, and 24 historical
integrity violations are unchanged. The repaired dashboard's ongoing workload
had passed 72 rounds / 3,456 requests at that checkpoint.

At that checkpoint an additional external-baseline finding was unresolved and was not covered by
the passing package assertions. The lock-protected 20:12:57 receipt has 4,040
of 49,667 rows fetched at or after the target; 4,025 have scores. The summary
includes these source-issue-relative forecasts in its prospective report.
A two-row fixture fails three receipt-causal assertions and shows the score
contribution directly. The existing readiness script reports 162/9/1 after
20:05, but does not check this receipt-time boundary. Its one reported failure
must not be interpreted as the absence of other evidence problems.

The default fetch timestamp precedes network I/O; historical response-completion
times are not known. A scoring/provenance correction then awaited the user's decision;
no external row was removed or reclassified. This separate data source does
not feed A3's calculation or change its independently reproduced undercoverage.
See [the receipt-time evidence](validation/output/operational/deep_debug_20260907/external_receipt_timing/).
The correction request was incomplete at that checkpoint; its later repair is
recorded in the September 8 section above.

### Completed TLS workload and deployment — 20:48 UTC

The same unsupervised repaired dashboard PID completed 90 rounds / 4,320
complete-JSON responses from 19:06:14 to 20:35:33 UTC, with zero request
failures and a maximum latency of 8.005 seconds against the 15-second deadline.
Health, status, and forecasts remained available. There were 64 unavailable
dB/dt responses across eight rounds and eight unavailable network responses
in the first round; all completed with HTTP 200. These data-availability gaps
are not counted as timeouts or concealed by the successful transport result.
The server was deliberately stopped afterward, exit 143. Source, process,
and statistics were not reset during the workload.

The tested source is deployed under the unchanged launchd configuration:
dashboard PID 89275 / runs 4 and monitor PID 89363 / runs 7. The monitor startup
cycle completed at 20:43:09, skipping all four pending issue-hour duplicates.
The 120-assertion preservation check confirms all 2,120 forecast rows and all
117 immutable columns are unchanged. Four observations matured and four were
revised. A separate 17-assertion check preserves every prior external receipt's
15 immutable columns: 49,667 prior rows retained, 38 new rows captured, and
64 prior observation/score records updated. Scoring and timestamp semantics
were not changed. Both launchd plist hashes and all 189 source hashes match.

The post-reload independent replay passes 13,648 assertions, reproducing the
persisted claim payload and the four latest causal histories using their
preserved issuance-time NOAA vintage. A3 covers 941/1,198 outcomes (78.5476%),
with 289 complete cycles, 14 days, zero qualifying storms, and the same
24 integrity violations. The nominal 90% coverage requirements remain unmet.
The forecast-log SHA-256 is
`40587084d6e9727a807e16f2c65c9ffb9e881392e3ffe8686c4429ff03681754`.

The deployed dashboard then completed five rounds / 240 requests without
timeouts, unavailable data, or another restart. The strict readiness run against
its documented status endpoint reports 162 PASS / 9 WARN / 1 FAIL, exit 1:
the six historical fallback cycles remain the reported failure. Earlier
wrong-endpoint diagnostic invocations are retained separately and are not
product failures or successful audits. The receipt-time omission is still
outside the readiness script's checks. The protected submission digest matches.

See [deployment receipts](validation/output/operational/deep_debug_20260907/post_tls_reload/)
and [the completed TLS workload](validation/output/operational/deep_debug_20260907/tls_runtime/).
Monitoring continued on the frozen cadence; the external timing correction and
the separate model-upgrade study were still unapproved at that checkpoint.
Their later authorization and results are recorded above. The finite workload does
not prove the initiating cause of historical native failures.

### Earlier same-day verification, before the TLS change

Final-source `Pkg.test()` passes **284,024 / 284,024 assertions**, the deterministic
experiment passes, and the workspace development harness passes **226 checks /
1 reviewed tolerance warning / 0 failures**. The run completed at approximately
17:36 UTC with Julia 1.12.6, two threads, and
`SOLARSINDY_REQUIRE_LOCAL_ARTIFACTS=1`; all six registered oracles ran, with zero
skips. Package-test elapsed time was 18m22.3s. Source receipts remain unchanged.
Both the initial missing-Sockets test failure and the earlier native app
teardown failure are preserved; neither is counted as a pass.

**Operational correction remains incomplete.** After the verified monitor's
18:05 cycle, the dashboard exited unexpectedly at 18:07:03.892 UTC with a task-
switch error in stderr. Launchd recovered it as PID 2118, runs 3. The monitor
remained PID 1024. The initiating cause is unproved; passing tests and the short
isolated timeout exercises do not establish sustained production stability.

| Completed check | Result |
|---|---:|
| Final focused A3 claim suite | 166 assertions, PASS |
| A3 calibration suite | 49 assertions, PASS |
| Live forecast workflow | 990 assertions, PASS |
| Retention recovery cases | 323 assertions, PASS |
| 129-column / 100,004-occurrence conservation stress test | 407 assertions, PASS |
| Standalone canonical dashboard | 1,618 assertions, PASS on two clean process exits |
| Independent raw live-row oracle | 13,484 assertions, PASS |
| Post-restart raw live-row oracle | 13,556 assertions, PASS |
| Post-restart forecast preservation | 130 assertions, PASS |
| Final paired history benchmark equivalence | 11 assertions, PASS |
| Sockets declaration and compatibility checks | 58 assertions, PASS |
| Fresh V2.2 serving identity | 832,368 rows, PASS |
| Fresh V2.3 identity / innovation chain | 4,206 / 51,754 rows, exact agreement |
| Fresh V2.4e identity | 4,518 rows / 15 fields, exact agreement |

The focused retention/collector integration and final package suite also test
that archival does not erase a claim failure. Source loading joins numeric cold
segments with the hot log under the shared lock and rejects incomplete receipts,
pending transactions, malformed metadata, and post-freeze missing identities.
Existing receipts establish row/byte completeness, not authentication of all
historical CSV contents.

The restored base and hourly tables match the prior worktree's manifest digests
`9dcbe8f2be5e1dcb1ca314628d9d2f900ae7ec73a129dffc10b3fcfe0d3b700b`
and `1bff81e2da08134fd06f71d0a2e5bcc830f3cd890cc0cebe7823e7b0c7bc9ac3`.
The final package run exercised all six registered local-data oracles instead
of retaining the five historical skips. Its bundled dashboard contributes 1,619
assertions, including the package-level fixture-presence assertion.

On the fixed 2,104-row live snapshot, the independent calculation reproduces
the complete claim payload, stored interval formulas and hit flags, newest
causal histories, and pinned-weight centers. A3 coverage is still 0.7843803 on
1,178 matured rows, with no qualifying storm. The reproduced defects do not
explain this snapshot's undercoverage; the tests do not prove every possible
implementation path defect-free.

The history reader's final alternating-pair benchmark (three warmups, eleven
pairs, four queries on 2,100 rows/129 columns, two threads) reduces median time
from 260.95 to 118.51 ms and allocations from 187,966,064 to 25,456,496 bytes.
Outputs are identical. The larger retention test uses one thread and measures
8.438 s normal-retention and 3.152 s crash-recovery medians, with approximately
1.91 GB and 0.384 GB allocated, respectively. These working-host measurements
are not peak-memory or arbitrary-archive scaling guarantees.

Strict live readiness after the 18:05 cycle reports **162 PASS / 9 WARN /
1 FAIL**: six historical fallback cycles remain in its trailing 96-cycle
availability window. The A3 audit additionally retains 24 cumulative integrity
violations and failed coverage/sample gates. No threshold, tolerance, identity,
cohort, or submitted artifact was changed. The reloaded monitor issued exactly
four new rows at 18:05:00.005 and completed at 18:05:13. All 117 immutable prior
columns are preserved. The new independent replay agrees with the deployed
audit and the four causal histories; NOAA responses bracketing issuance are
byte-identical. The 2,112-row log has 1,186 matured A3 rows, 931 hits (78.499%),
and zero qualifying storms. Source/configuration checks pass for all 184 files,
and the submitted-tree digest matches the initial receipt.

[Evidence and reproducible checks](validation/output/operational/deep_debug_20260907/)
and [the debug record](DEBUG_LOG.md) retain the exact limits and failure history.

## Historical verification

> **Current verification (2026-08-26).** The package now serves Operational
> V2.4e and passes **283,460 assertions with 5 registered local-artifact skips
> (283,465 total)**.
> This includes **1,582/1,582** bundled-dashboard assertions, **979/979**
> live-forecast assertions, **47/47** frozen live-calibration assertions, and
> **51/51** prospective-claim assertions. The deterministic V2.4e experiment
> passes, and the workspace development harness reports **225 PASS / 1 WARN /
> 0 FAIL**. The sections below preserve the earlier V2.2 test history; they are
> not the current served-product summary. The prospective suites verify
> evidence integrity and conservative gates, not live calibration or storm
> skill.

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

## Live-feed scalar and UTC parsing (2026-08-24)

### Reproduced defects

Before the change, the dashboard and realtime parsers accepted timestamps such as
`2026-08-24T01:02:03garbage` and `2026-08-24T01:02:03+08:00` by reading only their first 19
characters. Fractional seconds were discarded. The SWPC and USGS numeric helpers also converted JSON
booleans to `1.0` or `0.0`; a boolean could therefore select a solar-wind or Kp row, or contribute a
finite dB/dt value. A structured timestamp could raise `MethodError` instead of being skipped.

### Independent expectations

The regression cases name exact accepted values and exact rejected shapes. They check one- and
six-digit fractions against hand-written `DateTime` values, reject suffixes and offsets, reject
objects and booleans, and exercise the behavior through `fetch_swpc_dst`, RTSW/Kp selection, USGS
dB/dt calculation, monitor freshness and subhourly trajectory serialization. The external-Dst
collector tests cover both supported date layouts and the same complete-input rule.

### Results

| Check | Result |
|---|---|
| `app/test/runtests.jl` | 1,527 / 1,527 pass |
| Realtime parser and monitor testsets | 182 / 182 pass |
| External-Dst collector and live-monitor testsets | 434 / 434 pass |
| `Pkg.test()` | 283,249 pass, 5 recorded local-artifact skips, 283,254 total, 0 failures, 17m22.6s |
| `examples/experiments.jl` | V2.4e serving and predecessor smoke completed; center -126.734 nT and band +/-31.857 nT for the bundled probe row |
| `.agents/scripts/dev-harness-audit.sh .` | 221 checks completed, 1 existing tolerance advisory, 0 failures |
| `julia --project=docs docs/make.jl` | exit 0; doctests, cross-references and HTML rendering completed |

## Dashboard runtime crash isolation and V2.4e live check (2026-08-26)

### Reproduction and control

The dashboard's previous minimal-compilation launch produced five unplanned macOS crash reports.
A sixth report was captured when that old process exited during its planned replacement; its
process had started at 15:55 local time, before the installed launchd property list was updated at
16:31:57. The five earlier reports ended in either `SIGSEGV` or `SIGBUS`, and all six entered Julia's
interpreted execution path. The forecast monitor was a separate process and did not restart during
this work.

An isolated server using the deployed application, live log, two Julia threads and
`--compile=min` exited 139 during a bounded 1,200-request mixed-endpoint run. Its terminal output
included `attempt to switch to exited task`. The same server configuration, with only the
minimal-compilation argument removed, completed 6,000 requests and remained healthy. After the
normal-compilation launchd property list was installed, the production dashboard also completed
6,000 mixed requests at concurrency 24 with one unchanged process identifier. No crash report has a
process launch time at or after the replacement dashboard's 16:32:15 start.

The isolated normal-compilation process used approximately 1.1 GiB RSS after the load; the
minimal-compilation process used approximately 0.5 GiB shortly before it crashed. This is an
explicit memory-for-stability tradeoff. The production service remained within the host's 24 GiB
physical-memory capacity during the load and subsequent checks.

### Verification results

| Check | Result |
|---|---|
| CLI syntax and launch assertions | `bash -n` passed; 54 / 54 CLI smoke assertions passed; both CLI and launchd template reject every `--compile=` argument |
| Launchd property lists | Repository template and installed dashboard property list both passed `plutil -lint`; installed arguments use `--startup-file=no` and the dashboard project without a compilation override |
| Standalone dashboard suite | 1,581 / 1,581 pass |
| Full package suite | 283,460 pass, 5 registered local-artifact skips, 283,465 total, 0 failures, 22m12.1s; bundled dashboard suite 1,582 and live-forecast workflow 979 |
| Deterministic package experiment | PASS; V2.4e bundle probe center -126.734 nT and interval half-width 31.857 nT; predecessor smoke PASS |
| Full serving-identity replay | Not rerun: the local `validation/output/operational/v2_4_rolling/learn_year_2025.csv` study table is absent and is one of the registered local-artifact skips. The previously recorded 4,518-row zero-difference result remains in this report; the current bundle probe and live digest checks passed |
| Development harness | PASS 225, WARN 1, FAIL 0; the warning is the existing loose-tolerance advisory |
| Strict live readiness audit | PASS 166, WARN 6, FAIL 0 with required, fresh dashboard API and strict regime-persistence mode after the 10:05 UTC live cycle |
| Documentation build | Doctests, cross-references and HTML rendering completed with exit 0 |
| Production HTTP load | 6,000 mixed requests at concurrency 24; the same dashboard process remained active, and three post-load health probes plus every primary endpoint passed |

The monitor was restarted once at the scheduled 09:55 UTC slot to load the step-resolved reporter.
Its interrupted predecessor was still selecting the interval policy and had not issued a horizon;
the replacement process completed one valid four-horizon cycle, refreshed observations, passed the
claim audit, and wrote the new report before sleeping for the next phase-balanced slot. It then woke
at 10:05 UTC and completed the next four-horizon cycle on schedule with no monitor error.

The strict readiness audit verified the exact served identity
`v2.4+sindy20x11+superlearner10floor+conformal` and bundle-manifest digest
`057aec0df488314cd682e212e9ba64233e2674a7c641d68b72aa729982093ede` in the newest cycle, with no
fallback among the trailing 96 staged cycles. At the recorded live cutoff, the exact V2.4e cohort
contained 784 matured rows, RMSE 10.92 nT and empirical 90%-target interval coverage 0.694. The
separately identified A3 calibration shadow contained 86 matured rows across 11 complete cycles and
two consecutive days, with provisional coverage 0.907 and width ratio 1.225; its day-block coverage
interval was 0.692--1.000, its paired interval-score interval crossed zero, and no row reached Dst
at or below -50 nT. Its integrity gate passed, but both the marginal-calibration and storm-skill
claim flags remained false.

The step-resolved same-row report used all 784 matured exact-V2.4e rows. V2.4e retained lower RMSE
than static V2.2 at steps 1, 2, 3, 4, and 6, but not at step 7 (18.15 versus 17.92 nT; 33 rows).
BurtonFull and O'Brien remained lower than V2.4e in the pooled comparison. These cohorts are short,
quiet, and highly overlapping, so they are diagnostic evidence rather than a superiority result.

The dashboard and live forecast endpoints named V2.4e as the served product. The `model_version=v2.1`
field remains base-operator metadata; it is neither the served label nor the plotted forecast. The
served V2.4e center and interval fields remained unchanged, the calibration shadow remained
non-serving, and the submitted tree retained aggregate SHA-256
`7dfd4556fe8b6f1ae133be80a9a878d255df4ea1b715ef273d257520e728ca74`.

## V2.4e A3 issuance-snapshot deep-debug and live deployment (2026-08-30)

### Reproduction and root cause

The prospective A3 shadow reached 422 matured rows with 0.7393 pooled coverage,
outside its frozen 0.88--0.92 gate, and every supported model step was below the
0.85 floor. Direct arithmetic reproduced every stored endpoint exactly, but a
cycle-order audit found one independent implementation defect. `cycle!` issued
forecasts before `refresh_observations!`; although `issue_forecast` had already
fetched the current causal Kyoto Dst snapshot, the A3 history helper read only
the older persisted `observation_dst_nt` values. Newly matured or revised
outcomes could therefore enter issuance history one cycle late.

The repair passes the already-fetched `dst_times` and `dst_vals` to the helper,
uses a finite current-snapshot value for a matching historical target, and falls
back to the persisted value only when the snapshot lacks that target. Paired
inputs, equal lengths, and finite values are required. The served center,
served interval, product identity, alert path, A3 identity, widths, location
rule, warm-up, and gate thresholds are unchanged.

### Independent expectations and anti-false-test controls

The regression fixture supplies correct current outcomes alongside stale and
missing logged outcomes. The old implementation produced one error because it
did not accept the causal-snapshot arguments; after the repair, the result is
exactly equal to a clean-log oracle. Separate cases preserve strict prior-issue,
target-maturity, served-identity, manifest, warm-up, and score behavior. Direct
malformed-input checks show that an unpaired snapshot, mismatched vector lengths,
and a non-finite snapshot value each fail closed as `unavailable:history_error`.

An independent arithmetic oracle over the original 422-row prospective cohort
gave zero maximum endpoint-formula error, 312 hits, 45 lower misses, and 65 upper
misses. A current-snapshot retrospective replay changed only eight historical
hit states. Later Kyoto revisions make that replay an intentionally conservative
diagnostic rather than an exact reconstruction of every old issuance snapshot,
but it is sufficient to show that the confirmed freshness defect does not
explain the main undercoverage.

### Verification results

| Check | Result |
|---|---|
| Pre-repair focused reproduction | 973 assertions passed and the new regression errored on the unsupported snapshot arguments; exit 1 |
| Repaired live-workflow suite | 980 / 980 pass |
| Dedicated live-calibration suite | 47 / 47 pass |
| Dedicated live-claim-audit suite | 51 / 51 pass |
| Malformed snapshot boundary | unpaired, unequal-length, and non-finite cases all fail closed; exit 0 |
| Source and test parsing | pass |
| Full `Pkg.test()` | 283,461 pass, 5 registered local-artifact skips, 283,466 total, 0 failures; 30m01.7s |
| `examples/experiments.jl` | PASS; V2.4e bundle probe center -126.734 nT and interval half-width 31.857 nT; predecessor smoke PASS |
| `.agents/scripts/dev-harness-audit.sh .` | PASS 225, WARN 1, FAIL 0; the warning is the existing tolerance advisory |
| Strict fresh/API readiness audit | PASS 166, WARN 6, FAIL 0; warnings are existing evidence boundaries |
| Diff and debug-artifact checks | `git diff --check` clean; no temporary reproduction code or diagnostic print remained in modified paths |

Five registered suite skips remain because their separately generated local
identity/study artifacts are absent. The harness reports them explicitly; no
test, tolerance, experiment, or audit was weakened to obtain a pass.

### Live deployment and fresh-cycle oracle

The prior daemon completed cycles 87 and 88 normally. At the planned 01:55 UTC
replacement boundary it began cycle 89 and wrote four rows immediately before
the restart command. The new process recognized that issue hour and skipped all
four duplicates, so this overlap was not treated as deployment proof. The same
replacement process then completed its unique cycle 2 at
`2026-08-30T02:05:00.007Z`, issued all four horizons under exact V2.4e with no
fallback, refreshed observations, passed retention and claim audits, and
regenerated the comparison report.

An independent replay of those four fresh rows, without calling the claim-audit
endpoint helper, matched the recorded A3 history counts, location shifts, and
both endpoints exactly: maximum location error 0, maximum endpoint error 0, and
maximum history-count error 0. The current API reports health `ok`, a complete
cycle, no outage, and zero fallbacks in the latest 24 exact-V2.4e cycles.

At that cutoff, A3 has 480 canonical deployment rows, 438 matured rows, 99
complete issue cycles, and five consecutive days. Its pooled coverage is
0.7397, every supported step remains below 0.85, and its marginal and storm
claims remain false. The current observed minimum is -32 nT, so there are still
zero qualifying storm rows or events. The repaired implementation therefore
remains shadow-only and was neither retuned nor promoted.

### Measured resource boundary

On the 1,332-row production log, seven warm helper calls took a median 39.37 ms,
with a 223.19 ms maximum and 27.96 MiB median allocation. The repair adds one
bounded in-memory lookup over an existing snapshot, no network call and no
persistent resource. The old long-lived daemon returned from approximately
1.18 GiB immediately after cycle work to approximately 177 MiB after six idle
minutes with 206 open descriptors at both observations. This bounded observation
rejects simple monotonic growth over that interval; it is not a general proof
that no resource leak can exist. The replacement daemon held 1,274,752 KiB RSS
and 203 open descriptors at both approximately four and ten idle minutes after
its first unique cycle. That short flat sample is an operational observation,
not a general leak or long-duration stability proof.

The protected `submitted/` aggregate digest remained
`7dfd4556fe8b6f1ae133be80a9a878d255df4ea1b715ef273d257520e728ca74`.
# CI clean-depot investigation — 2026-09-19

GitHub run `35450234656` failed all three jobs during dependency resolution,
before either package tests or documentation checks began. Julia 1.12 reported
no installed registries; Julia 1.10 reported the first unregistered dependency.
The original command reproduced the missing-registry failure in a fresh local
depot and clean archive. Adding General before resolution then installed and
resolved that same package successfully on Julia 1.12.6. The focused environment
contract suite passes 67 assertions, including a check for registry bootstrap
in both workflow environments. Logs are in
`validation/output/operational/live_upgrade_20260919/ci_empty_depot_{before,after}.log`.
Full matrix and documentation results must be checked on the subsequent pushed
commit; successful local bootstrap alone is not a completed CI verdict.

After registry initialization, the Julia 1.10 reproduction exposed a second
startup failure: the Julia 1.12 deployment manifest pins
`JuliaSyntaxHighlighting` to the standard-library version 1.12.0, which cannot
resolve as a package on Julia 1.10. The minimum-version CI job now preserves that
lockfile in runner scratch and resolves the unchanged Project.toml natively.
The same clean archive then resolves and instantiates successfully on Julia
1.10.11. The Julia 1.12 job retains the deployment lockfile.

## Julia 1.12-only support — 2026-09-19

The support policy now excludes other Julia minor releases. Package, dashboard,
and documentation environments share `~1.12.6`; launchers enforce both bounds.
CI uses one Julia 1.12 test job and one strict documentation job. The temporary
minimum-version lockfile workaround is no longer needed. The registry bootstrap
repair remains in both jobs.

Julia 1.12.7 resolves and instantiates all three environments. Focused tests pass
80 environment assertions and 54 CLI assertions; the full dashboard suite and
strict Documenter build exit successfully. The documentation build has no
Documenter warnings. Dependency precompilation emits existing GeometryBasics
unused-type-variable warnings, distinct from documentation diagnostics.
The full package, harness, and final pushed-commit CI checks remain to be run.

### Receipt-time and daily-review verification

The CMO receipt-time study and independent reconstruction complete all twelve
predeclared period/delay checks. The final reconstruction passes 455,898
assertions, including training-data hashes and high-precision climatologies.
The May 2024 ten-minute-delay coverage is 0.8795950985615344, below the fixed
0.88 floor; all twelve point-accuracy checks pass in both native and log units.
No cutoff, coefficient, history length, or advancement threshold was retuned.
The candidate therefore remains disabled; these exposed historical periods
are development evidence, not prospective confirmation.

A new timestamp-jitter test reproduced a window check that accepted an
irregular interior interval when total duration was unchanged. The corrected
check validates every interval; all 341 focused ground assertions pass. Its
rerun produces the same historical metrics. The separate metric checker does
not call the new evaluator's feature, target, quantile or metric functions.
Numerical tolerances allow only floating-point summation/transformation error;
timestamp, count, rank, hash and qualification comparisons are exact.

The daily-review checks exercise once-per-day publication, concurrency,
restarts, stale/future input, corruption and failed HTTP responses. Additional
tests reproduced discarded malformed payloads; their bodies and hashes are
now preserved while their availability remains false. A real dashboard receipt
reports CMO adjusted and FRD variation measurements separately, retains all
48 A3 findings, and reports both A3 claims false. No forecast log or bundle was
rewritten by the review. Logs are beneath
`validation/output/operational/live_upgrade_20260919`.

GitHub run `35452903794` passed documentation but stopped at a missing test-only
`Pkg` declaration introduced by the new version-policy test. That declaration
is corrected in the test target. Two in-progress local full runs were
deliberately interrupted after the final reporting fix; they are not passing
evidence. A fresh full harness and final-commit CI are required for handoff.

## Freshness and calibration checks — 2026-09-20

The focused checks pass 36 cache/freshness assertions, 40 diagnostic assertions,
1,251 delayed adaptive-calibration assertions and 65 daily-review assertions.
The nested-age reproduction failed nine assertions before the correction;
the already-stale parent reproduction failed three. Cache tests exercise
coalesced workers, completed unavailability and negative-cache throttling.
Daily-review tests require an explicit pending state without rewriting prior
dated reports.

The independent historical checker passes 456,273 adaptive assertions and
10,095 A3 assertions. It does not call the new evaluator's bounds, chronology,
cohort, metric or decision functions. Input columns and hashes, receipt counts,
miss counts, alpha values, bounds, coverage and decisions compare exactly.
Aggregated metrics use BigFloat arithmetic with relative tolerance 2e-13 and
absolute tolerance 2e-12, allowing only floating-point accumulation differences.
The A3 check preserves all 48 recorded fallback findings and verifies the
unchanged interval rule against each retained raw row. Linear receipt counting
produces byte-identical diagnostic CSVs to the initial implementation.

Tests include a future-outcome mutation that flips a covered outcome to a miss;
earlier bounds remain unchanged and later controller feedback changes only after
strict receipt. The initial test increased an outcome already counted as a miss,
which correctly left binary feedback unchanged. That incorrect test expectation
was corrected before the historical evaluation, without altering the candidate
or any acceptance criterion. Invalid windows, duplicate panels, changed input
hashes and output overwrite attempts are rejected.

The frozen experiment and diagnosis outputs are under
`validation/output/operational/qualification_20260920`. Full-package, harness,
documentation, live-reload and pushed-revision CI results belong in that
directory's release receipt; focused checks alone do not complete verification.
