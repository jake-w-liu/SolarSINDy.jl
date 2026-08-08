# Live Forecast Verification

Live forecast verification must lock the prediction before the target
observation is available. A forecast is not verified merely because the code
produces a number.

## Evidence Standard

The accepted sequence is:

```text
issue forecast at t0
log target time, prediction, interval, drivers, and model state anchor
wait until observed Dst exists for the target time
compare the locked prediction to that observation
record residual and interval coverage
```

The logged row is the source of truth. Do not recompute or edit the prediction
after the target observation arrives.

## Command

From the repository root:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --wait
```

The script:

- fetches live NOAA SWPC plasma and magnetic-field data,
- fetches the latest SWPC Kyoto Dst observation,
- anchors the rolling Dst* state to the latest observed Dst,
- forecasts a strictly future hourly Dst target,
- appends the locked prediction to `SolarSINDy.jl/var/monitor/live_forecast_log.csv`,
- logs persistence, Burton, BurtonFull, O'Brien--McPherron, and SINDy
  predictions for the same target,
- polls the Dst feed until the target observation appears,
- updates the same log row with the observed Dst, residual, and 90% interval
  coverage flag.

Available modes:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --issue
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --verify-pending
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --refresh-observations
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --backfill-baselines
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --replay-recent
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --replay-omni
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --fit-v2-calibration
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --campaign
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --summary
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --comparison-report
```

Use `--poll-seconds=N`, `--timeout-hours=N`, `--horizon-hours=N`, and
`--log=PATH` to adjust the run. Use `--replay-horizons=1,2,3,6` to emit multiple
lead times per anchor when building a replay/calibration table, and
`--v2-coverage-floor=N` to set the deployment coverage floor for v2.

Use `--campaign --campaign-horizons=1,2,3,6 --model=v2` to issue multiple
future operational-v2 forecasts, poll the observation feed, verify the locked
rows, and regenerate the comparison report from one command. `--campaign`
defaults to `--model=v2` if no model is specified.

Use `--backfill-baselines` after schema changes to fill baseline forecasts and
residuals for older rows that already contain locked SINDy predictions and live
driver values.

Use `--refresh-observations` before final reporting to reconcile logged
observations with the current Dst feed. The live Dst product can revise recent
hours after first publication. Refreshing updates only observation, residual,
and interval-hit fields; locked predictions, drivers, target times, and model
metadata are not recomputed.

Use `--replay-recent --replay-hours=N` to build a longer predicted-versus-
observed comparison table from the recent live feeds. The replay is causal: each
row anchors on the observed Dst at the issue hour, persists the previous
complete solar-wind hour into the one-hour-ahead target, and compares the
prediction with the already-published target Dst. This provides more rows for
diagnostics, but it is not a substitute for locked forecasts that were issued
before observations arrived.

Use `--replay-omni` for a longer research replay from a local extracted OMNI
CSV without adding that large data file to git:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl \
  --replay-omni \
  --omni=paper_v2_monitor/data/omni_extracted.csv \
  --omni-year-start=2024 \
  --omni-year-end=2025 \
  --replay-hours=4000 \
  --table=/private/tmp/solar_v2_omni_replay_2024_2025.csv \
  --table-limit=0
```

The OMNI replay uses the same one-hour-ahead predicted-versus-observed table as
`--replay-recent`, but it can cover thousands of chronological rows. It remains
retrospective evidence; locked live forecasts are still required for operational
claims.

Use `--comparison-report` after `--verify-pending` to create the standard
locked-live report:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl \
  --comparison-report \
  --log=SolarSINDy.jl/var/monitor/live_forecast_log.csv \
  --report=SolarSINDy.jl/var/monitor/live_comparison_report.md
```

The headline comparison uses the same target rows for served V2.1, the V2.1
frozen-tail ablation, SINDy v1, persistence, Burton, BurtonFull, and
O'Brien--McPherron. The served V2.1 column is the product forecast. The
frozen-tail and selector fields are audit evidence, not additional products.

## V2 Calibration

The default live model is V2.1; `--model=v2` is its accepted short alias.
Use `--model=v1` explicitly to reproduce the uncalibrated discovery-core path.
V2.1 applies a causal ridge residual correction to the current revised 20/11
SINDy forecast and supplies a calibrated prediction interval. The correction
uses only issue-time fields: latest Dst, solar-wind
speed, IMF components, density, dynamic pressure, and derived causal coupling
features such as southward IMF, `V Bs`, transverse IMF magnitude, IMF clock-angle
coupling, and square-root dynamic pressure.

Current V2.1 calibration uses a leakage-free chronological split. The fit command
sorts replay/live-log rows by issue time into train, validation, and holdout
sets. The ridge residual correction and the component selector are fit on the
train rows only; the feature set, ridge penalty, and component are chosen on the
validation rows, which fit neither. The holdout is scored exactly once and is
never used for selection or tuning. This calibration-stage score applies to the
V2.1 frozen-tail center. It must not be reported as performance of the served
stack after its center-changing tail and safeguards.

Deployment is decided by an acceptance gate evaluated on the validation rows: a
v2 candidate is deployed only if it beats persistence and O'Brien--McPherron on
both RMSE and MAE and its 90% interval coverage is at least the coverage floor
(`--v2-coverage-floor`, default 0.85). If no candidate passes, a v1-equivalent
(zero-correction) fallback is deployed instead, so a candidate that cannot be
shown to beat the strong baselines is never shipped. Candidate feature sets
include instantaneous coupling terms, causal Dst and solar-wind memory terms,
and expert-disagreement terms when baseline predictions are available. On small
fit sets the selector is held to the corrected SINDy center to avoid choosing a
baseline component on noise. The issued V2 output is one upgraded
forecast; the internal `v2_selected_component` field is audit metadata, not a
separate headline model.

The logged V2.1 frozen-tail center remains available for ablation scores. The
served fields add the validation-selected operational tail: forecast steps with
sufficient upstream coverage use ballistically propagated L1 forcing; later
hours use regime-aware Bz/By relaxation; causal rapid-deepening projection,
one-hour inertia, state-conditioned inertia, and an extreme-Dst guard constrain
known failure regimes. The dashboard displays this served V2.1 result as the
single product forecast.

The frozen package also includes a separate complete-hour served-stack holdout
under `data/operational_validation/v2_1_served_holdout_*`. It applies the
complete-hour implementation of the served tail and safeguards to the locked
2020--2022 holdout, shifts the pre-holdout static conformal offsets to the
served center, and performs zero residual updates from holdout targets. It does
not reconstruct the fractional subhourly upstream windows available during
live issuance. Regenerate it with
`validation/operational/v2_1_served_holdout.jl`. The pooled coverage floor is a
promotion criterion; lead-specific and activity-specific results are reported
as diagnostic boundaries rather than silently pooled away.

Fit the calibration from a prior replay or locked live log:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl \
  --fit-v2-calibration \
  --table=SolarSINDy.jl/validation/output/operational/live_replay_144h.csv \
  --v2-calibration=SolarSINDy.jl/validation/output/operational/operational_v2_calibration.csv \
  --v2-ridge-grid=0,1,10,100,1000 \
  --v2-validation-fraction=0.15 \
  --v2-selector-margin=0.5
```

The monitor's bundled point calibration and conformal sidecar live together in
`SolarSINDy.jl/deploy/`. Replace them as a pair only after evaluating a new fit.

For a longer research calibration, pair `--replay-omni` with the same fit
command:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl \
  --fit-v2-calibration \
  --table=/private/tmp/solar_v2_omni_replay_2024_2025.csv \
  --v2-calibration=/private/tmp/solar_v2_omni_calibration_2024_2025.csv \
  --v2-ridge-grid=0,0.1,1,10,100,1000 \
  --v2-validation-fraction=0.15 \
  --v2-selector-margin=0.5
```

This also writes `operational_v2_calibration_selection.csv` (each tested
candidate with its validation metrics, whether it passed the acceptance gate,
which candidate was deployed, and the once-scored holdout RMSE/MAE/coverage),
`operational_v2_calibration_scored.csv` (the fit/validation/holdout rows scored
by the deployed calibration), and `operational_v2_calibration_conformal.csv`
(the conformal interval calibration; see below).

Then run a calibrated replay or live issue:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl \
  --replay-recent \
  --model=v2 \
  --v2-calibration=SolarSINDy.jl/validation/output/operational/operational_v2_calibration.csv \
  --replay-hours=144 \
  --table=SolarSINDy.jl/validation/output/operational/live_replay_v2_144h.csv
```

The V2.1 calibration is not evidence of readiness by itself. It must be
scored chronologically against held-out rows and then accumulated through locked
live forecasts exactly like v1.

## Conformal Predictive Intervals

`--fit-v2-calibration` also fits a split-conformal interval calibration from the
fixed-driver frozen-center validation residuals (out-of-sample for the ridge
fit) and writes it to a `*_conformal.csv` sidecar next to the V2 calibration.
The conformal half-width is stratified by forecast horizon and by activity
regime (quiet versus disturbed, set from the issue-time Dst), with the standard
finite-sample rank correction and a pooled fallback when a stratum is sparse.
Its marginal finite-sample guarantee requires exchangeability within the
calibration stratum. The fit reports empirical holdout coverage under both the
conformal interval and the legacy interval-scale band.

When the sidecar is present, `--model=v2` issuance sources the logged 90%
interval from the conformal half-width for the row's horizon and activity
regime, instead of the v1-ensemble-spread interval scale. Each row records an
`interval_source` column (`conformal` or `interval_scale`) for audit. The point
forecast is unchanged; only the uncertainty band changes. The static half-width
is shifted from the frozen center to the served V2.1 center. This preserves the
width but does not automatically transfer the frozen-center guarantee; the
complete-hour served-stack holdout supplies the corresponding empirical
coverage evidence.

To populate the horizon strata, build the conformal calibration table with
multiple lead times via `--replay-horizons`:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl \
  --replay-omni \
  --omni=paper_v2_monitor/data/omni_extracted.csv \
  --omni-year-start=2024 --omni-year-end=2025 \
  --replay-hours=4000 --replay-horizons=1,2,3,6 \
  --table=/private/tmp/solar_v2_omni_replay_multi.csv --table-limit=0
```

Then fit the calibration on that multi-horizon table (`--fit-v2-calibration
--table=...`), and validate empirical coverage by horizon and regime from the
`_scored.csv` / comparison report. This is the run that establishes whether the
operational 90% interval actually covers at the nominal rate; it requires live
or archived OMNI data and is the data-dependent step, not a code step.

## Online Assimilation (Research)

The package also provides an Extended Kalman Filter (`init_assimilation`,
`assimilation_predict!`, `assimilation_update!`, `run_assimilation`) that adapts
a small physically-motivated subset of the discovered ODE coefficients (for
example the injection scale and decay rate) online from each observed Dst while
keeping the sparse equation structure fixed. It is a verified library primitive
used for research comparison against fixed-coefficient paths. Causal seven-storm
replays found no promotion-grade gain from either decay-only or
injection-adaptive overlays, so assimilation is intentionally absent from the
live issuance path. See [EKF extension decision](ekf-v3-decision.md).

## Operational Deployment

The long-running monitor, the dashboard/alerting server, and an out-of-process
liveness watchdog are deployed as supervised services. On macOS they are launchd
agents rendered from the tracked templates in `SolarSINDy.jl/deploy/`; on a
Linux host they run from the systemd unit templates in the same directory, and on
a container host as a supervised pair from
`SolarSINDy.jl/deploy/docker-compose.yml`.

### macOS launchd services

Render and install all three services from the templates:

```bash
SolarSINDy.jl/deploy/install_launchd.sh SolarSINDy.jl monitor dashboard watchdog
```

The script resolves the stable juliaup shim (`~/.juliaup/bin/julia`), never a
version-pinned juliaup directory that `juliaup gc`/`update` could delete, creates
the state and log directories, substitutes the template placeholders, lints the
result, and bootstraps each agent into the GUI domain. Because the installed
plists are generated from the templates, the deployed artifacts are reproducible
rather than hand-edited. The monitor and dashboard use `KeepAlive` so a crash is
restarted; the watchdog runs on a fixed `StartInterval`.

### Diagnostics and log rotation

launchd does not rotate console files, so the monitor owns bounded diagnostics.
Every status line is mirrored into a size-capped rotating ring
(`var/monitor/logs/monitor.log` plus numbered archives), and the launchd
`StandardOutPath`/`StandardErrorPath` capture files are rotated once per restart,
so an incident leaves a retrievable record instead of being discarded. Total
diagnostic disk use is bounded.

### Outage surfacing

Issuance loss is surfaced in-band. The monitor writes an outage sentinel beside
the forecast log when its issuance dead-man trips, and the watchdog writes the
same sentinel when it observes a dead or unloaded process. The dashboard reads
the sentinel and the forecast-log staleness: `/api/health` reports an `outage`
object and an `outage` status, and `/api/alerts` raises the stale/unknown flag
with an explicit reason. A stalled or unloaded monitor is therefore visible on
the dashboard even when the monitor process is not running — the case that
outage detection exists to cover.

### External liveness watchdog

`deploy/watchdog.sh` is a dependency-light periodic probe, independent of both
supervised processes. It checks forecast-log freshness (daemon liveness) and the
dashboard health endpoint (server liveness); on a state change it writes the
sentinel and posts the configured webhook. Because it does not run inside either
process, push alerting no longer depends on an interactive session or on the
dashboard being alive.

### Forecast-log durability

The locked-live log is the scientific record, so bounded retention never discards
rows outright. Before the hot log is trimmed to its row cap, the rows about to be
dropped are appended to an append-only cold archive under `var/monitor/archive/`,
guarded by a sidecar manifest that tracks the cumulative row count, the archive
size, and a per-segment hash. A short or corrupted append aborts the trim, so the
rows stay in the hot log rather than being lost.

### Container deployment

For a non-macOS node, `deploy/docker-compose.yml` runs the forecast daemon and
the read-only dashboard as a supervised pair over a shared volume, each with a
restart policy and a `HEALTHCHECK` (log freshness for the daemon, the health
endpoint for the dashboard), both as a non-root user, with Plotly vendored at
build time so the image never depends on an external CDN at runtime.

### Pinned-release deployment

`deploy/deploy_release.sh` snapshots the package at a committed revision into a
versioned run directory (a clean `git archive` export, never the working tree)
and can repoint the services at it, so a restart cannot silently deploy
uncommitted state. State is kept outside the release so the locked-live log
persists across deploys.

## Interpretation

One verified live row proves that the live ingestion, forecast, and observation
comparison loop worked for that case. General reliability requires many locked
forecast rows scored the same way, without discarding misses or changing the
model after seeing the observation.
