# V2.4e Remaining-Risk Closure

Two items listed under "Remaining risks" in `V2_4_INTEGRATION_REPORT.md` are quantified here against
data: the provisional-versus-final Dst anchor (Task A) and the truncated southward-run feature of the
analog key under an unmeasured L1 hour (Task B). Both are properties of the live information set
rather than of the served arithmetic, so the served identity oracle is silent on them by construction;
this report replaces "not currently quantified" with measured bounds and states what remains
unmeasurable.

Bundle under test: the deployed fold-2026 bundle `deploy/v2_4/` (identity
`v2.4+sindy20x11+superlearner10floor+conformal`, 138,715 analog origins, 60 stack cells,
`guard_applied = false`). Anchor sample: 2,411 anchors of 2025 carrying all six model steps
(14,466 `(anchor, step)` rows), a uniform stride over the 8,728 usable anchors unioned with the 300
deepest, so the storm branch of the stack is exercised rather than sampled away. Sampled anchor Dst
runs from -217 to +54 nT with a median of -12 nT; 230 anchors sit at or below -70 nT. By served cell
the rows split 10,746 shallow / 2,340 moderate / 1,380 deep and 10,014 quiet / 2,568
active-deepening / 1,884 recovery, so every depth bin and every regime of the stack is exercised.

## Method

Every number below comes from one of three harnesses, all under
`<scratch>/risk_closure/` and reproduced in the "Reproduction" section:

- `a1_dst_revision.jl` — read-only extraction of the provisional Kyoto/SWPC Dst record from the
  production monitor logs, and its comparison with final OMNI Dst.
- `b1_history_frequency.jl` — read-only extraction of the per-cycle analog-key history depth and of
  the hourly L1 coverage the live path saw.
- `risk_harness.jl` — offline driver of the *served* code path
  (`v24_serving_analog_features`, `v24_serving_analog_members`, `v24_serving_t1r_center`,
  `v24_serving_direct_features`, `v24_serving_direct_center`, `v24_serving_climatology_center`,
  `v24_serving_center`, `v24_serving_depth_safe_center`) over the fixed anchor sample, once per input
  variant, with `analyze_harness.jl` reducing the per-variant rows to the tables below.

For a Dst variant the harness rebuilds every stage that depends on the Dst ladder rather than reusing
it: the V2.1 core forecast and its 26 calibration features (through
`V23CalibrationSource.build_v2_1_calibration_table` and `score_operational_v2`, the same chain
`build_v2_3_base_table` uses), persistence, Burton, Burton-full, O'Brien–McPherron, the served and
frozen V2.1 centers (`_v2_forecast`), the static V2.2 stack center (`v2_3_static_v22_center`), the
analog key, the direct-GBM design and its Dst lag ladder, the climatology expert, the regime, the
depth bin, the stack cell and the conformal stratum. L1 driver inputs are held fixed throughout, and
the bundle's analog archive stays on final Dst, which is the live situation: a provisional query
against a final archive.

**Harness correctness gate.** Before any variant is compared, the harness's own reference panel is
checked against the study's base table (`v2_3_base_table.csv`) on all 14,466 sampled rows. The
largest absolute deviation is 0.000e+00 nT for every one of `served_v2_1`, `frozen_v2_1`,
`persistence`, `burton`, `burton_full`, `obrien` and `static_v2_2`. A variant difference is therefore
a response of the served path, not a harness artifact.

## Task A — provisional versus final Dst

### A.1 What the archives can and cannot pair

The live path anchors on the Kyoto quicklook Dst served by SWPC
(`https://services.swpc.noaa.gov/products/kyoto-dst.json`, `fetch_swpc_dst` in `src/realtime.jl`);
the study fitted and scored on final OMNI Dst.

**Direct pairing is impossible with the present archives: n = 0.** Final OMNI Dst runs to
2026-03-04T19:00 (553,772 valid hours, last valid record year 2026 DOY 63 hour 19). The live
forecast record starts at 2026-06-06T01:00. The two windows do not overlap by a single hour, so no
provisional anchor the served path has ever used has a final counterpart. This is a structural gap,
not a sampling shortfall, and it will close only when OMNI's final series advances past the start of
the live record.

What the archives *can* measure is the revision of the same provisional product: the same UTC hour
read by the monitor at two different wall-clock times. Every read is timestamped or strictly ordered,
and all reads are of the same feed (the external-forecast scorer's observation URL is the same Kyoto
product the forecaster anchors on). Three reconstructions were made, over 911 distinct UTC hours
between 2026-06-06T01:00 and 2026-08-18T07:00 and 29,427 individual reads.

| # | Comparison | n | exact | mean | median | RMS | p95 abs | max abs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | first read vs last read, both timestamped | 659 | 0.982 | +0.040 | 0.000 | 0.496 | 0.000 | 7.0 |
| 3 | anchor read vs verification read (settled) | 647 | 0.076 | +3.039 | +3.000 | 4.378 | 8.0 | 17.0 |
| 4 | anchor read vs the same hour re-read 1 h later | 640 | 0.081 | +3.127 | +3.000 | 4.487 | 8.0 | 22.0 |
| 4 | anchor read vs the same hour re-read 3 h later | 619 | 0.082 | +3.045 | +3.000 | 4.442 | 8.0 | 22.0 |

Units are nT; the sign convention is `later − anchor`, so a positive mean means the first published
value is the more negative one. Comparison 2 is near-degenerate and is reported as a consistency
check rather than as a revision measurement: its median settle lag is 0.00 h (p90 0.02 h, max 5.00 h)
because the external scorer scores a target hour once, usually in the same cycle that first published
it. Comparisons 3 and 4 carry the time depth. They are independent reconstructions — 3 pairs the
anchor value against the verification value the monitor later refreshed into the row, 4 recovers the
value of hours `t-1` and `t-3` as the issue at `t` saw them from that row's own memory features
(`latest_dst_nt - dst_delta_1h_nt`, `latest_dst_nt - dst_delta_3h_nt`, with neutral-memory rows
excluded) — and they agree to within 0.1 nT on every statistic. The revision is essentially complete
one hour after the anchor read: the 1 h and 3 h figures are the same.

Split by state, on comparison 3:

| Subset | n | exact | mean | median | RMS | p95 abs | max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| quiet (settled Dst > −30 nT) | 637 | 0.077 | +2.967 | +3.000 | 4.275 | 8.0 | 17.0 |
| disturbed (settled Dst ≤ −30 nT) | 10 | 0.000 | +7.600 | +6.000 | 8.729 | 13.6 | 14.0 |

The disturbed subset is 10 hours and carries no weight as an estimate; it is reported because it is
the only disturbed evidence that exists, and it points the same way as the pooled figure, only
larger.

Distribution by absolute difference (comparison 3):

| \|Δ\| bin (nT) | n | fraction |
|---|---:|---:|
| [0, 1] | 144 | 0.223 |
| (1, 2] | 99 | 0.153 |
| (2, 3] | 108 | 0.167 |
| (3, 5] | 162 | 0.250 |
| (5, 10] | 126 | 0.195 |
| (10, ∞) | 8 | 0.012 |

Spot-checked against the raw log: the hour 2026-06-30T18:00 was anchored at +13.0 nT by the cycle
issued 18:59:10 and settled at −4.0 nT, a 17 nT revision; the neighbouring hours 17:00 and 19:00
moved by −11 nT each.

**Honest scope of this measurement.** It bounds the revision *inside the real-time product*, which is
the observable part of the provisional error. Kyoto's provisional and final series apply further
baseline corrections that no local archive can expose, so the figures above are a lower bound on the
provisional-to-final difference, not an estimate of it. Task A.2 therefore also runs the perturbation
at two and three times amplitude, and the comparative conclusion — how V2.4e responds relative to the
pipelines it replaced — is amplitude-independent.

### A.2 Response of the served center to a provisional-quality anchor

Five Dst variants were run against the reference ladder, all with L1 inputs held fixed:

| Variant | Construction | Realised shift on the 17,544 hours |
|---|---|---|
| `prov_iid` | one observed difference drawn independently per hour | mean −2.99 nT, RMS 4.37 nT |
| `prov_block` | contiguous 24 h runs of the observed difference series, preserving its hour-to-hour correlation | mean −3.02 nT, RMS 4.35 nT |
| `prov_iid_x2` | the independent draw at twice amplitude | mean −5.99 nT, RMS 8.75 nT |
| `prov_iid_x3` | the independent draw at three times amplitude | mean −8.99 nT, RMS 13.13 nT |
| `prov_bias` | the observed median difference applied to every hour, no scatter | −3.00 nT exactly |

Center response, pooled over the six model steps, in nT (n = 14,466 rows per cell):

| Variant | Pipeline | mean | median \|Δ\| | p95 \|Δ\| | max \|Δ\| |
|---|---|---:|---:|---:|---:|
| `prov_iid` | **V2.4e** | −2.644 | 2.691 | 7.540 | 24.19 |
| `prov_iid` | static V2.2 stack | −2.511 | 2.694 | 7.514 | 25.22 |
| `prov_iid` | V2.1 served | −2.759 | 2.775 | 7.914 | 63.78 |
| `prov_iid` | depth-safe alerting center | −2.816 | 2.716 | 7.596 | 59.36 |
| `prov_block` | **V2.4e** | −2.455 | 2.519 | 6.586 | 18.01 |
| `prov_block` | static V2.2 stack | −2.396 | 2.483 | 6.466 | 15.90 |
| `prov_block` | V2.1 served | −2.467 | 2.530 | 6.725 | 55.52 |
| `prov_block` | depth-safe alerting center | −2.490 | 2.516 | 6.602 | 55.52 |
| `prov_iid_x2` | **V2.4e** | −5.681 | 5.466 | 16.388 | 40.30 |
| `prov_iid_x2` | static V2.2 stack | −5.259 | 5.395 | 16.351 | 52.01 |
| `prov_iid_x2` | V2.1 served | −6.370 | 5.559 | 22.040 | 86.01 |
| `prov_iid_x2` | depth-safe alerting center | −6.639 | 5.536 | 20.609 | 78.49 |
| `prov_iid_x3` | **V2.4e** | −8.856 | 8.315 | 25.752 | 54.84 |
| `prov_iid_x3` | static V2.2 stack | −8.194 | 8.171 | 25.843 | 78.82 |
| `prov_iid_x3` | V2.1 served | −10.623 | 8.387 | 52.912 | 149.43 |
| `prov_iid_x3` | depth-safe alerting center | −11.077 | 8.410 | 49.066 | 149.43 |
| `prov_bias` | **V2.4e** | −2.356 | 2.426 | 2.890 | 8.75 |
| `prov_bias` | static V2.2 stack | −2.343 | 2.351 | 2.832 | 5.42 |
| `prov_bias` | V2.1 served | −2.360 | 2.399 | 2.806 | 8.34 |
| `prov_bias` | depth-safe alerting center | −2.352 | 2.398 | 2.813 | 8.34 |

The depth-safe alerting center is the minimum of the other three, so its response tracks whichever
pipeline moved furthest down: its mean shift is the most negative of the four on every variant that
carries scatter, while its tail stays bounded by the tails it combines on all five. It is not an
exposure additional to the three.

The V2.4e center per model step:

| Variant | step 1 | step 2 | step 3 | step 4 | step 6 | step 7 |
|---|---:|---:|---:|---:|---:|---:|
| `prov_iid` median \|Δ\| | 3.2813 | 3.0185 | 2.8573 | 2.6627 | 2.3786 | 2.2744 |
| `prov_iid` p95 \|Δ\| | 8.4702 | 8.0591 | 7.6076 | 7.0012 | 6.2648 | 6.0574 |
| `prov_block` median \|Δ\| | 2.9455 | 2.7857 | 2.6306 | 2.4828 | 2.2090 | 2.1132 |
| `prov_block` p95 \|Δ\| | 7.6378 | 7.1665 | 6.7428 | 6.2861 | 5.8013 | 5.4237 |
| `prov_bias` median \|Δ\| | 2.8181 | 2.6401 | 2.4898 | 2.3355 | 2.0998 | 2.0062 |

The gain is close to unity and decays with lead: a −3 nT anchor bias moves the one-hour center by
−2.80 nT and the seven-hour center by −1.93 nT. That is expected rather than anomalous — persistence,
climatology-relaxed persistence, the direct increment-GBM and every SINDy-family expert are anchored
on `Dst(t)` — and it is the same for all three pipelines. The anchor error is therefore essentially
additive on the published center at roughly unit gain, and the exposure is a property of anchoring on
Dst at all, not of the super-learner.

Cell and tier switching, as a fraction of the 14,466 `(anchor, step)` rows:

| Variant | regime | depth bin | stack cell | deepening flag | conformal half-width | severity tier | tier **lowered** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prov_iid` | 0.1273 | 0.0514 | 0.1439 | 0.0324 | 0.0514 | 0.0536 | 0.0058 |
| `prov_block` | 0.0875 | 0.0506 | 0.1070 | 0.0187 | 0.0506 | 0.0451 | 0.0037 |
| `prov_iid_x2` | 0.2190 | 0.1078 | 0.2439 | 0.0850 | 0.1078 | 0.1341 | 0.0088 |
| `prov_iid_x3` | 0.2762 | 0.1651 | 0.3094 | 0.1464 | 0.1651 | 0.2238 | 0.0102 |
| `prov_bias` | 0.0257 | 0.0406 | 0.0406 | 0.0046 | 0.0406 | 0.0339 | 0.0000 |

The regime column is `operational_v22_regime`, which the static V2.2 stack selects its own weights
with as well, so that rate applies to both stages; the depth bin, the stack cell and the conformal
stratum are V2.4e's. The severity tier is `dst_threat_level` (bands −30/−50/−100/−200 nT) evaluated
on `v24_serving_depth_safe_center(V2.4e, static V2.2, V2.1 served)`, which is what the dashboard
publishes. Under the calibrated variants 4.5–5.4 % of rows change tier, and only 0.37–0.58 % change
it *downward*; under the pure systematic shift no row is lowered at all. The direction is the safe
one because the observed revision is signed: the first published quicklook value is the more negative
one, so a provisional anchor deepens rather than shallows the published state.

The analog retrieval is far more sensitive than the center. `dst0` and `ddst1` are two of the
eighteen key features, so 99.4 % of anchors retrieve a different set of 25 archive origins under
`prov_iid` (98.4 % under `prov_block`), yet the analog expert's own center moves by a median of
2.55 nT — less than the stack center — because the ensemble mean over 25 members absorbs a changed
membership. Retrieval instability is not, by itself, center instability.

### A.3 Decision

**Closed as quantified; the V2.4e response is comparable to the pipelines it replaced.** On the four
variants that carry scatter, the V2.4e median response is within 0.15 nT of the static V2.2 stack's
(largest separation 8.32 versus 8.17 nT at three times amplitude) and below the V2.1 served center's
at every amplitude; its 95th percentile is within 1.9 % of the stack's and 2.1 % to 51 % below
V2.1's. Its worst single row is the smallest of the three under the calibrated independent draw
(24.2 against 25.2 and 63.8 nT) and under both stress amplitudes (40.3 against 52.0 and 86.0 nT;
54.8 against 78.8 and 149.4 nT); under the correlated draw it sits between the two, at 18.0 nT
against 15.9 nT for the stack and 55.5 nT for V2.1.

The one variant on which V2.4e is marginally the most responsive of the three is the pure systematic
shift, where a uniform −3 nT anchor offset moves it by a median of 2.426 nT against 2.351 nT for the
stack and 2.399 nT for V2.1, a 95th percentile of 2.890 against 2.832 and 2.806 nT, and a worst row
of 8.75 against 5.42 and 8.34 nT. The distributional separations there are 0.03–0.08 nT, under 3 % of
the response and well under the 4.4 nT scale of the perturbation itself; the tail is a single row
near a cell boundary.

No hardening is applied. The decision rule's condition for proposing one — a response *larger* than
the predecessors' — is not met on any variant by a margin that would matter, and any conformal margin
added for this exposure would have to be added to the V2.1 and V2.2 stages too, because they carry it
in the same amount. What remains unmeasurable is unchanged and is stated as such: the
provisional-to-final step itself cannot be paired until final OMNI Dst advances past 2026-06-06, and
prospective accrual against the feed the daemon reads stays the arbiter of served skill.



## Task B — truncated southward-run feature

### B.1 How often the twelve-hour window is incomplete

The analog key's mandatory driver history is seven hours (`V23_HISTORY_LAGS_H = 7`), but the
consecutive-southward run-length feature reads up to twelve (`V23_SOUTH_RUN_CAP_H = 12`) and stops at
the first hour with no usable record (`operational_v23_features.jl`, the `south_run` loop). The live
history builder `_v2_3_driver_history` returns `nothing` for an hour with no measured L1 coverage, so
a gap at lags 8–12 truncates the run instead of refusing the key. The direct-GBM design does not
catch it either: its extra features read driver records only back to `t-4`
(`V23_DIRECT_VBS_LAG_STEPS = (1, 2, 3)` at `issue - Hour(1 + lag)`), so `design.ok` stays true.

This is a live-only failure mode. The study's hourly frame is causally forward-filled and complete:
over its 140,256 records from 2010-01-01T00:00 to 2025-12-31T23:00 there are zero non-finite `V`,
`Bz`, `By`, `n`, `Pdyn` or `Dst` values and zero missing hourly steps, and the bundle's shipped
analog frame is the same 140,256 records with zero non-finite `Bz`. No scored anchor in the study
ever had a truncated run length, so nothing in the fold tables measures this.

Direct evidence from production, per issued cycle:

| Column | cycles | depth = 12 | depth < 12 | rate |
|---|---:|---:|---:|---:|
| `v24_history_hours` | 10 | 10 | 0 | 0.000 |
| `v23_history_hours` | 20 | 20 | 0 | 0.000 |

The column exists only for cycles issued after it was added, so this is 10 and 20 cycles, not a rate
estimate. For the longer record, the hourly L1 coverage the live path saw was reconstructed from the
finite-sample counts every cycle logs for its trailing solar-wind hour
(`n_speed_finite_trailing_hour`, `n_bz_finite_trailing_hour`, and where present the density and By
counts), applying the same admission threshold the per-lag test applies
(`MIN_HOURLY_DRIVER_SAMPLES = 10`); consecutive cycles cover consecutive hours, so the cycle series is
an hourly coverage series.

| Quantity | Value |
|---|---:|
| reconstructed hours (2026-06-13T13:00 .. 2026-08-18T06:00) | 644 |
| hours with an under-covered L1 window | 3 (0.0047) |
| anchors carrying a complete lag 1..12 coverage record | 409 |
| anchors with at least one uncovered hour at lags 8..12 | 1 (0.0024) |

The three under-covered hours are on record with their counts: 9 finite speed and Bz samples at the
window ending 2026-06-14T07:19, 2 and 2 at 2026-06-14T10:20, and 0 speed with 59 Bz at
2026-06-16T16:31 (the row that set `driver_data_gap`). Staleness cannot produce this mode on its own:
issuance is refused outright when the solar-wind feed is more than
`LIVE_MAX_SOLAR_WIND_AGE_HOURS = 6` h old, well before a 12 h window could be reached, so only an
interior gap in the L1 minute record can truncate the run.

The reconstruction is an estimate, and it is biased optimistic in one direction: an hour in which no
cycle ran at all contributes no record, and it under-counts if such outages correlate with L1 gaps.
It is also blind to the exact per-lag source window, which is offset by the ballistic transit time.

For context on how much of the archive would be exposed if a gap did occur, the 2025 archive
run-length distribution is: 11.43 % of the 8,760 anchors carry `south_run ≥ 8`, 9.68 % `≥ 9`, 8.28 %
`≥ 10`, 7.02 % `≥ 11` and 5.92 % at the 12 h cap. Among deep anchors (Dst ≤ −70 nT) the exposure is
38.26 %.

### B.2 What a truncated run-length does to the served center

Four variants were run over the same 2,411 anchors, each marking the L1 hour at one lag unmeasured
(`driver_history[lag] = nothing`, which is exactly what `_v2_3_driver_history` produces for an hour
with no measured coverage) while leaving the Dst ladder and every other input at the reference value.
In the sample, 343 of 2,411 anchors (14.23 %) carry `south_run ≥ 8` and are therefore exposed to a
lag-8 gap; the sample is enriched with the 300 deepest anchors, so this sits above the 11.43 %
whole-year archive rate.

V2.4e center response over the whole sample (14,466 rows), in nT:

| Gap at lag | mean | median \|Δ\| | p95 \|Δ\| | max \|Δ\| | worst per-step p95 | worst per-step median |
|---|---:|---:|---:|---:|---:|---:|
| 8 | +0.0021 | 0.0000 | 0.167 | 2.017 | 0.213 (step 6) | 0.000 |
| 9 | +0.0011 | 0.0000 | 0.132 | 1.922 | 0.176 (step 7) | 0.000 |
| 10 | +0.0004 | 0.0000 | 0.094 | 1.336 | 0.135 (step 7) | 0.000 |
| 11 | +0.0007 | 0.0000 | 0.049 | 1.232 | 0.071 (step 4) | 0.000 |

Analog expert (`t1r_analog`) response over the same 14,466 rows, with the retrieval and run-length
columns counted per anchor because the retrieval does not depend on the model step:

| Gap at lag | median \|Δ\| | p95 \|Δ\| | max \|Δ\| | anchors whose member set changes | mean Δ`south_run` | worst Δ`south_run` |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 0.000 | 0.610 | 8.015 | 0.1423 | −0.535 h | −5 h |
| 9 | 0.000 | 0.555 | 4.217 | 0.1186 | −0.392 h | −4 h |
| 10 | 0.000 | 0.367 | 5.004 | 0.1037 | −0.274 h | −3 h |
| 11 | 0.000 | 0.243 | 3.491 | 0.0908 | −0.170 h | −2 h |

Conditioned on the anchors that are actually exposed — the 343 anchors whose run reaches lag 8, that
is 2,058 rows — the lag-8 figures are: V2.4e center median 0.089 nT, p95 0.788 nT, max 2.017 nT, with
10.5 % of rows moving by more than 0.5 nT; the analog expert itself moves by median 0.382 nT, p95
2.092 nT, max 8.015 nT. The stack damps the analog expert by roughly a factor of three, which is what
a mass-floor-constrained convex combination over ten experts, only one of which reads the run length,
should do. The single worst row in the whole study is 2025-05-17T02:00 at step 6, anchor Dst −60 nT,
where the run truncates from 10 h to 7 h and the center moves from −55.89 to −57.91 nT — deeper, not
shallower.

Nothing downstream of the center switches. Across all four gap variants no row changes regime, depth
bin, stack cell, deepening flag or conformal half-width — the gap touches the analog key only, and
none of those labels reads it. The published severity tier changes on 9 of 57,864 rows in total, and
those nine come from only four distinct `(anchor, step)` pairs whose reference depth-safe center
already sat within 0.26 nT of a band edge: two pairs crossing −30 nT and two crossing −50 nT. Every
one of the nine is an escalation of a single band; none is a reduction. Every observed center
movement is far inside the served conformal half-width, which is 4.86–17.49 nT at one hour and
13.75–45.25 nT at seven hours depending on depth bin.

### B.3 Decision

**Current behaviour kept; the bound is recorded.** The preregistered rule was to harden fail-closed
if the 95th percentile of \|Δ V2.4e center\| under a lag-8 gap exceeded 0.5 nT at any step, or if the
median exceeded 0.1 nT. The worst per-step 95th percentile is 0.213 nT and the worst per-step median
is 0.000 nT, so neither condition is met and no code change is made.

The quantified bound now on record is: a single unmeasured L1 hour anywhere in the run-length window
moves the served V2.4e center by at most 2.02 nT over 14,466 sampled rows spanning quiet, active
deepening and recovery states and anchor Dst down to −217 nT; it never moves the regime, depth bin,
stack cell or conformal width; and it changed the published severity tier on 9 of 57,864 rows across
the four gap positions, every one an escalation of a single band. Conditioned on the run actually
truncating, the 95th percentile is 0.788 nT.

Two things keep this decision reversible on evidence rather than on assumption. The mode stays
observable per row — `v24_status = ok` with `v24_history_hours < 12` marks a row whose run-length
feature may be truncated — and the fail-closed variant remains cheap if the accrued record ever
contradicts the bound: extending the existing per-lag driver check in `_v2_4_served_center` from the
mandatory seven lags to all twelve would report through the existing
`fallback:missing_driver_lag<N>` status and needs no new status. At the B.1 rate it would cost of
order 0.2 % of cycles, inside the 1 % served-fallback specification target, though its interaction
with the two-cycle failure rule of `v2_readiness_audit.jl` over a 96-cycle window would have to be
checked before adopting it. It is not applied now because the measured response does not justify
trading a served center for a fallback.



## Claim ledger

| Claim | Evidence | Status |
|---|---|---|
| No provisional anchor the served path has used has a final-OMNI counterpart | final OMNI ends 2026-03-04T19:00; live record starts 2026-06-06T01:00; pairing returns n = 0 | VERIFIED |
| The provisional feed revises by mean +3.0 nT, RMS 4.4 nT and a maximum of 17–22 nT within one hour of the anchor read | three independent reconstructions from the monitor logs, n = 647/640/619, agreeing to 0.1 nT on mean, median and RMS | VERIFIED |
| That figure is a lower bound on the provisional-to-final difference | Kyoto's provisional and final baseline corrections are not exposed by any local archive | VERIFIED (as a scope statement) |
| The harness reproduces the study's own base table exactly on the sampled rows | max abs deviation 0.000e+00 nT on 14,466 rows for all seven base-table expert columns | VERIFIED |
| V2.4e's response to a provisional-quality anchor is not larger than the static V2.2 stack's or the V2.1 served center's | five perturbation variants, 14,466 rows each; on the four scatter variants medians within 0.15 nT of the stack, p95 within 1.9 % of it and below V2.1's; on the pure systematic shift V2.4e is the largest of the three by 0.03–0.08 nT | VERIFIED |
| A provisional anchor lowers the published severity tier on at most 0.58 % of rows at the calibrated amplitude | depth-safe tier switching table, all five variants | VERIFIED |
| A truncated southward run is a live-only mode | the study's 140,256-record frame has zero non-finite driver values and no missing hours | VERIFIED |
| An unmeasured L1 hour at lags 8–11 moves the served center by at most 2.02 nT, never changes the cell or the interval, and changed the published tier on 9 of 57,864 rows, every one an escalation | four gap variants, 57,864 rows | VERIFIED |
| The truncation occurs on the order of 0.2 % of anchors | 0 of 10 logged V2.4 cycles and 0 of 20 logged V2.3 cycles; 1 of 409 anchors in the reconstructed coverage record | PARTIAL — a reconstruction over 644 hours, blind to hours in which no cycle ran |

## Reproduction

All scripts are in the session scratch directory
`/private/tmp/claude-501/-Users-jake-EMPIRE-projects-ongoing-2026-045/d34ebeb6-611c-46d6-8ebc-1d0042edd49c/scratchpad/risk_closure/`
and were run from the package root of this worktree.

```bash
# Task A.1 and Task B.1 (read-only against the production monitor tree)
JULIA_NUM_THREADS=1 julia --startup-file=no --project=. <scratch>/a1_dst_revision.jl
JULIA_NUM_THREADS=1 julia --startup-file=no --project=. <scratch>/b1_history_frequency.jl

# Task A.2 and Task B.2 (offline serving harness over the deployed bundle)
JULIA_NUM_THREADS=8 julia --startup-file=no --project=. <scratch>/risk_harness.jl \
    --anchors=2100 --out=<scratch>
JULIA_NUM_THREADS=1 julia --startup-file=no --project=. <scratch>/analyze_harness.jl

# Re-derivation of every figure quoted above, and of the two tier-changing rows
JULIA_NUM_THREADS=1 julia --startup-file=no --project=. <scratch>/verify_report_numbers.jl
JULIA_NUM_THREADS=1 julia --startup-file=no --project=. <scratch>/verify_tier.jl
```

Outputs: `a1_dst_revision_pairs.csv`, `a1_anchor_vs_verification.csv`, `a1_lagged_revision.csv`,
`a1_delta_pool.csv`, `a1_dst_revision_summary.csv`, `a1_dst_revision_bins.csv`,
`b1_direct_history_depth.csv`,
`b1_hourly_coverage.csv`, `b1_reconstructed_truncation.csv`, `risk_harness_rows.csv`,
`sensitivity_by_variant_step.csv`, `switching_by_variant.csv`, `membership_by_variant.csv`.
The harness run log is `harness.log` and the reduced tables are echoed in `analysis.txt`.
`a1_diag_ages.jl` is a diagnostic that established why the timestamped-revision comparison has no
time depth; it is kept for the record.

The production live log is appended by the running daemon, so the extracted revision pool grows as
the record accrues. The harness drew from the pool as of its run: n = 647 hours, median +3.00 nT,
RMS 4.38 nT. A first pass at 1,986 anchors, kept under `run1/`, gives the same decisions and the
same figures to within the sampling difference.
