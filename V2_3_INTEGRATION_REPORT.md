# V2.3 Integration Report

Scope: integrate the Operational V2.3 analog driver continuation into the live
engine, and promote the served point center to the fitted static V2.2 regime
stack. Governing decision: the single-shot confirmatory scoring of the
preregistered V2.3 candidate returned `NO_GO`, so V2.3 is integrated as a shadow
forecast and the served switch goes to the static stack instead.

## Confirmatory decision that set the scope

`validation/output/operational/v2_3_test/decision.csv`

| Field | Value |
|---|---|
| Decision state | `NO_GO` |
| Failing gates | A1, A2 |
| Selected configuration | `T1r_T1_magnetic_K25_Soff` |
| Scored anchors / rows | 51,754 / 310,524 |
| Base-table SHA-256 | `9dcbe8f2be5e1dcb1ca314628d9d2f900ae7ec73a129dffc10b3fcfe0d3b700b` |
| Code SHA-256 | `eb138facfb80a2a7cae443bab4bee55adc3715adeb6f262352ae78c0c709bf36` |

The run's own exposure disclosure records that TEST 2020–2022 was the V2.1
conformal holdout, TEST 2023–2025 was used in V2.1 development and safeguard
selection, and the static V2.2 stack was fitted on 2010–2017: every comparator is
exposure-favoured on TEST and the V2.3 candidate is not.

## Served product

| Item | Value |
|---|---|
| Served identity | `v2.2+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia+staticstack(sindy60_fit407598)` |
| Disclosed fallback identity | `v2.1+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia` |
| Stack weights | `deploy/operational_v2_2_stack.csv`, label `operational_v2_2_primary_sindy60_fit407598`, SHA-256 `66e7347f71f5cdf407e85d4612702bb19c82dcbcd74d8c79526173f839472d7d` |
| Provenance manifest | `deploy/operational_v2_2_stack_manifest.csv` |
| Components | served V2.1, frozen V2.1, persistence, Burton, Burton-full, O'Brien–McPherron |
| Regime inputs | observed Dst, the live memory one-hour Dst rate (neutral on an interior Kyoto gap), the gated coupling proxy |
| Continuity column | `v2_1_served_pred_dst_nt` |
| Per-row stage status | `v2_2_status` (`ok`, `ok_unpinned`, or the exact fallback reason) |
| Alerting center | `min(served, v2_1_served)` per row, then the existing minimum over the cycle |
| Alerting interval edge | `served_ci05 + min(0, v2_1_served - served)` per row, then the minimum over the cycle; the watch tier is taken on this edge |
| Unpinned-stack identity | `...+staticstack(sindy60_fit407598)+unpinned`, served only under `SOLARSINDY_ALLOW_UNPINNED_STACK=1` and accepted by neither the dashboard nor the readiness audit |

## Shadow product

| Item | Value |
|---|---|
| Shadow identity | `v2.3-shadow+sindy20x11+L1A+ADC(magnetic,K25)+T1rcal+LAT+E` |
| Deployment | `deploy/v2_3_shadow/` (12 artifacts plus `manifest.csv`) |
| Analog archive | 86,968 origins, 2010-01-01T07 → 2019-12-24T16 |
| Analog frame | 87,648 hourly rows, 2010-01-01T00 → 2019-12-31T23 |
| Error layers | E2 (depth 3, 128 rounds) at 1 and 2 h, identity at 3 and 4 h, E1 (λ = 1000) at 6 and 7 h |
| Lead-aware weights | 1, 1, 1, 1, 0.75, 0.75 for steps 1, 2, 3, 4, 6, 7 |
| Log columns | `v23_shadow_model_version`, `v23_manifest_sha256`, `v23_status`, `v23_analog_k`, `v23_history_hours`, `v23_raw_dst_nt`, `v23_center_dst_nt`, `v23_step1_center_dst_nt`, `v23_shadow_pred_dst_nt`, `v23_e_layer_applied` |
| Error-layer chain | `Dst(anchor + 1 h)` minus the logged one-hour pre-layer center of that anchor, over the six preceding anchors; `ok:e_layer_pending` until six mature |
| Served? | never; the shadow center is excluded from the served columns, the threat level and every alert |

## Identity oracles

### Served static stack

`validation/operational/v2_2_served_identity.jl` → `validation/output/operational/v2_2_served_identity.csv`

| Scope | n | max abs delta |
|---|---|---|
| all scorable DEV/TEST rows | 832,368 | 0 nT |
| step 1 / 2 / 3 / 4 / 6 / 7 h | 138,728 each | 0 nT each |
| regime quiet / active_deepening / recovery | 704,040 / 80,328 / 48,000 | 0 nT each |
| coupling gate | 832,368 | 0 mV/m |
| pooled-fallback cells | 0 | — |

The archived column carries the unprojected stack sum. The served center
additionally takes the physical projection every served center takes; it moves 10
of 832,368 rows (max 8.685 nT), all of them rows whose observed Dst already sits
above the +50 nT ceiling.

### V2.3 shadow center

`validation/operational/v2_3_serving_identity.jl` → `validation/output/operational/v2_3_serving_identity.csv`

Anchors reproduced: 701 (4,206 scored rows), served-fallback anchors 0, skipped 0.

| Model step | n | max abs delta, final center | pre-layer center | frozen blend partner | error-layer rows |
|---|---|---|---|---|---|
| 1 h | 701 | 0 nT | 0 nT | 0 nT | 694 |
| 2 h | 701 | 0 nT | 0 nT | 0 nT | 694 |
| 3 h | 701 | 0 nT | 0 nT | 0 nT | 0 |
| 4 h | 701 | 0 nT | 0 nT | 0 nT | 0 |
| 6 h | 701 | 0 nT | 0 nT | 0 nT | 694 |
| 7 h | 701 | 0 nT | 0 nT | 0 nT | 694 |

Target was 1e-9 nT over at least 500 anchors; both are met with zero deviation.

The oracle additionally proves that the live error-layer chain is the scored chain.
The live engine cannot filter the log for one-hour rows — the requested wall
horizons are 1/2/3/6 h at a one-hour anchor lag, so the issued model steps are
2/3/4/7 h — so it records the one-hour pre-layer center of every anchor and forms
the innovation as `Dst(anchor + 1 h)` minus that center. The shared rule
`v23_serving_innovations_from_step1_centers` is applied by both the engine and the
oracle; against the history the oracle builds from the scored table's step-1
`V2_3_LAT` column it reproduces 51,754 anchors with max abs delta 0 nT, against a
requirement of 300.

## Verification summary

| Check | Result |
|---|---|
| `Pkg.test()` | 20475 / 20475 pass, 0 failures, 13m17.5s |
| `dev-harness-audit.sh` | PASS 319, WARN 3, FAIL 0 (`Pkg.test()` and `examples/experiments.jl` both PASS) |
| `test/test_operational_v22_serving.jl` | 93 / 93 pass |
| `test/test_operational_v23_serving.jl` | 183 / 183 pass |
| `test/test_serving_identity_oracles.jl` | 98 / 98 pass |
| `test/test_v2_readiness_selftest.jl` | 3 / 3 pass |
| `test/test_live_forecast_verify.jl` | 684 / 684 pass |
| `test/test_v2_2_served_replay.jl` | 106 / 106 pass |
| `test/test_realtime_monitor.jl` | 176 / 176 pass |
| `app/test/runtests.jl` | 859 / 859 pass |
| `validation/operational/v2_readiness_audit.jl --self-test` | PASS, 24 independent checks, exit 0 |
| `examples/experiments.jl` | PASS |
| `validation/operational/v2_3_build_deploy.jl --from-smoke` | PASS (81,129 origins, matching the smoke run's record) |
| `validation/operational/v2_3_build_deploy.jl --from-test` | PASS (86,968 origins, matching the confirmatory record) |

## Known boundaries

1. **Component definitions.** The served center is the stack applied to the live
   engine's own six components. The live core and baseline rollouts admit
   L1-measured hours and the live frozen tail freezes the trailing wind hour, so
   the live component panel is not byte-identical to the archived panel the stack
   was fitted on. The static-stack oracle establishes that the stack is applied
   correctly, not that the panels coincide. The same boundary applies to the shadow
   center: its analog key is built from partial-hour L1 driver means and its member
   rollouts admit L1-measured hours, neither of which the hourly archive carries.
   The shadow center's frozen V2.1 blend partner is also not the served row's
   `v2_pred_dst_nt`: the scored candidate blends against a rollout that holds the
   issue driver for every step, while the logged column admits L1-measured hours, so
   the two frozen quantities differ by construction and the shadow path recomputes
   its own.
2. **Served interval pool.** The served adaptive-conformal band pools
   `served_pred_dst_nt` residuals. During the transition that pool mixes V2.1-served
   and stack-served residuals. The 500-row window is per model step and per activity
   regime, and four horizons mature per hour, so a step's pool turns over after
   roughly 21 days of quiet hourly issuance and considerably longer for the
   disturbed pool. The mixed-pool transition is therefore a multi-week condition,
   not a multi-day one. Readiness reports the served sample and its RMSE per served
   label and warns until the current label has 48 verified rows of its own, so the
   pooled live figures are not read as the stack's own record.
3. **Archive membership.** The analog origin identities are shipped rather than
   derived, because membership additionally requires a quality-flagged,
   non-gap-filled L1 driver record at `t-1`, which the causal hourly frame does
   not carry. Everything else about the archive is rebuilt and checked on load.

## Post-integration audit corrections

A deep-debug audit of the integrated serving path found no wrong served center —
both identity oracles still reproduce their archived answers with zero deviation —
but nineteen operational-safety, tooling and disclosure findings. All of them are
now fixed, each with a regression test.

| Finding | Correction |
|---|---|
| The watch flag and tier were assessed on the served band as issued, so a shallower stacked center could lower the outbound alert level below the previous product on identical physics (reproduced: alert level 3 → 2) | The interval lower edge is taken on the depth-safe center, `served_ci05 + min(0, v2_1_served - served)`, one-sided so a deeper stacked center's band is untouched |
| The live error layer could never engage: its history was keyed on one-hour log rows, and the issued model steps are 2/3/4/7 h | Every cycle records `v23_step1_center_dst_nt` and the innovation is `Dst(anchor + 1 h)` minus that center; the shared rule is proved against the scored history on 51,754 anchors |
| Readiness did not fail closed on served-stack or shadow health | The audit loads the pinned stack and verifies the shadow manifest, measures the served fallback rate over the trailing 24 cycles (FAIL above 1 %, or on a newest fallback cycle while the artifact loads), and reports shadow availability and error-layer engagement |
| `v2_readiness_audit.jl --self-test` exited 1 on a stale fixture sentence | Fixed, extended by nine cases, and added to the package test suite |
| The issue-identity artifact recorded the V2.1 tail label and no stack provenance | It records the served identity, the stack label and digest, the shadow identity and the shadow manifest digest; the audit requires the exact served identity and compares the API's served label with the newest logged cycle |
| Dashboard and API named the product V2.1 and pooled verified rows across served labels | The product name comes from the served label, the 15-minute line is captioned as the V2.1 core trajectory shown for display, and verified rows are reported per label with the current label's own count |
| An empty `SOLARSINDY_V2_2_STACK_SHA256` silently disabled the digest pin | Refused unless `SOLARSINDY_ALLOW_UNPINNED_STACK=1`, and then served under a separate `...+unpinned` identity that fails closed downstream |
| `lead_time.driver_assumption` was hardcoded to the stack sentence even on fallback cycles | Derived from the served row |
| The served stage had no per-row status | `v2_2_status` is logged and counted in readiness |
| E-layer artifacts were outside the digest-verified manifest set | The loader requires every artifact named by `e_layers.json` to carry a verified digest row |
| A shadow redeployment into the same directory was invisible | `v23_manifest_sha256` is logged per row |
| The depth-safe severity was invisible to operators | `/api/forecast` exposes `severity_dst_nt`, `severity_ci05_dst_nt` and `v2_1_served_pred_dst_nt` per horizon |
| `v22_serving_depth_safe_center` was unused while the app kept its own copy | One definition in `src/serving_depth_safe.jl`, included by the package and by the application |
| A cycle whose rows carried different accepted labels blanked the dashboard | Accepted, and reported under the weakest label |
| Served metrics pooled V2.1-served and stack-served rows | Readiness stratifies by `sub_hourly_model_version` and warns until the current label has 48 verified rows |
| Report and documentation inaccuracies | The ACI pool turnover (≈21 days per step, not a few days), the shadow known-boundary and frozen-partner notes, and the product naming in `docs/src/live-verification.md` are corrected |
| The analog key's L1 depth was unobservable | `v23_history_hours` is logged; the fail-closed boundary is ≈9.5 h of upstream minute data, not 10 h as first estimated, because only seven of the twelve lags are mandatory |

## Not performed

No launchd service was installed, reinstalled or restarted; no commit, merge or
push was made; the production checkout at `2026_045/SolarSINDy.jl` was not
touched. A live dry run against the real feeds and the deployment steps of the
original specification remain outstanding.

The readiness checks that read a live hot log — the served fallback rate, the
served-stage status census, shadow availability, error-layer engagement, the
per-served-label live sample, and the comparison of the API's served label with
the newest logged cycle — were exercised on fixture logs and against the real
deployed artifacts, not against a production hot log: this worktree carries no
live log. Whether the error layer actually engages in production, and what the
served fallback rate is there, are therefore observations that the first cycles
after deployment will produce. The full audit's verdict in this worktree is FAIL
for reasons that predate this work: the generated replay evidence and the live log
are not present here.
