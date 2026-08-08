# SolarSINDy.jl

`SolarSINDy.jl` is a Julia package for **data-driven geomagnetic storm (Dst) forecasting**.
It combines sparse equation discovery with a calibrated operational forecasting layer, and
runs end to end from synthetic validation through real NASA OMNI2 discovery to a live
NOAA SWPC–driven monitor.

The package is organized as two layers:

1. **Discovery.** Sparse identification of nonlinear dynamics (SINDy) recovers a
   closed-form solar wind–magnetosphere coupling equation for Dst from storm time
   series, alongside classical physical baselines. The current discovery artifact has
   20 candidate terms and 11 active terms.
2. **Operational forecasting (V2.1).** A causal post-processing layer calibrates the
   revised 20/11 point forecast and attaches **split-conformal predictive intervals
   with finite-sample marginal coverage under exchangeability**. It then applies
   ballistically propagated L1 forcing, regime-aware
   Bz/By relaxation, a causal rapid-deepening projection, a validation-selected
   one-hour inertia blend, and an extreme-Dst inertia guard. The archived 21/10 V2.0
   core is available only through an explicit historical request.

## Capabilities

**Discovery and modeling (v1)**

- synthetic storm generation for controlled validation
- sparse equation discovery (STLSQ, ensemble SINDy, λ sweeps) from storm time series
- a candidate coupling-function library (rectified `VBs`, clock angle, dynamic pressure, …)
- classical baselines: Burton, Burton-full, and O'Brien–McPherron
- NASA OMNI2 ingestion, cleaning, and storm-catalog extraction

**Operational forecasting (V2.1)**

- a causal correction layer, `Dst_v2 = Dst_v1 + β₀ + Σ βⱼ zⱼ`, fit only from prior
  (replay/live) rows so it never looks ahead of the issue time
- stratified split-conformal predictive intervals with finite-sample marginal coverage
  under exchangeability, stratified by lead time × geomagnetic activity regime
- adaptive (online) conformal updating under distribution shift
- V2.1 tail: ballistically propagated L1 forcing when the corresponding upstream window
  has sufficient coverage, then regime-aware Bz/By relaxation, a capped causal
  rapid-deepening projection, one-hour persistence shrinkage selected on development
  partitions, and a separate extreme-core persistence guard
- guarded component selection over corrected SINDy, uncorrected SINDy v1, persistence,
  Burton, Burton-full, and O'Brien–McPherron, deployed only after chronological validation
- online assimilation utilities for reproducibility and shadow experiments; EKF-on-SINDy
  failed promotion gates and is not part of the V2 forecast
- forecast skill metrics (RMSE, correlation, skill score, prediction efficiency, Wilcoxon)

**Real-time**

- NOAA SWPC plasma / magnetic-field / Dst fetchers
- a rolling monitor loop with calibrated storm-severity alarms

## Operational forecast status

The dashboard and live monitor serve a single V2.1 forecast whose SINDy core is the
same 20-candidate/11-active-term artifact used by the revised discovery paper. The
former 21/10 product is labeled V2.0 and retained under `data/historical/v2_0/` and
`deploy/historical/v2_0/` only for reproducible matched comparisons. The `v2` API
alias resolves to V2.1; loading V2.0 requires an explicit version.

EKF-on-SINDy is retained as research infrastructure only. Both tested deployment paths
(decay-only and injection-adaptive EKF) failed promotion gates, so they are not exposed
through the dashboard, daemon, alerting path, or V2 forecast columns. Recent
lower-RMSE relaxed-tail variants also remain diagnostic until they avoid severe-storm
under-warning in sustained southward-Bz stress rows.

The files named `v2_lookahead`, `v2_envelope`, `v2_improved`, `v2_subhourly`, and
the EKF replays are development-lineage experiments. They now load the same revised
20/11 core and V2.1 calibration features as the package, but deliberately substitute
or omit parts of the served tail to isolate one idea. Their frozen-tail comparisons
must not be read as evaluations of the complete V2.1 product. Historical 21/10
execution is confined to the explicit V2.0 artifact resolver and matched provenance
comparisons.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/jake-w-liu/SolarSINDy.jl")
```

Or from a local checkout:

```julia
Pkg.develop(path="SolarSINDy.jl")
```

Then:

```julia
using SolarSINDy
```

Requires Julia 1.10+ for the package API; the operational monitor/dashboard stack targets
Julia 1.12.6+ (enforced by its launchers). `Manifest.toml` is committed for reproducible
research runs.

## Quick start — run the live forecast system

A fresh clone runs the full operational system — the hourly forecast daemon plus the web
dashboard — with one command each:

```bash
git clone https://github.com/jake-w-liu/SolarSINDy.jl.git
cd SolarSINDy.jl

bin/solarsindy setup      # one-time: checks Julia, instantiates + precompiles both environments
bin/solarsindy start      # forecast daemon + dashboard; auto-opens http://127.0.0.1:8723
bin/solarsindy status     # process / health / issuance-freshness summary
bin/solarsindy open       # reopen the dashboard tab any time
bin/solarsindy stop       # stops both daemons
```

`start` opens the dashboard in your browser only from an interactive terminal (never when
scripted, piped, or headless; `SOLARSINDY_NO_OPEN=1` turns it off) — the server keeps
running either way, and a closed tab is just `bin/solarsindy open` away.

`bin/solarsindy help` lists everything else: `start`/`stop`/`restart` take `monitor`,
`dashboard`, or `all`; `once` runs a single forecast cycle in the foreground; `logs
[monitor|dashboard] [-f]` tails the daemon logs; `install-service` /
`uninstall-service` switch to and from the supervised production mode (macOS launchd
with auto-restart and an out-of-process watchdog; systemd templates for Linux live in
[`deploy/`](deploy/)). Configuration is by environment variable — `SWM_PORT`,
`SOLARSINDY_MONITOR_DIR`, `SWM_WEBHOOK_URL` for push alerts — or a gitignored
`solarsindy.env` file in the clone root (template:
[`deploy/solarsindy.env.example`](deploy/solarsindy.env.example)). The CLI keeps its
pidfiles inside the instance state directory, serializes concurrent starts with a lock,
refuses to start a second copy while the installed service mode is active, and never
signals a process that is not running this clone's daemon entry point (so a recycled PID
can never be hit).

## Architecture: discovery → operational calibration

The two layers are deliberately decoupled. The v1 SINDy model is a fixed, interpretable
dynamical core. The v2 calibration is a thin causal layer fit on top of replayed or live
forecasts; it **does not** modify the v1 coefficients or the forecast state. This keeps the
discovered physics auditable while letting the operational forecast adapt:

```
issue-time drivers ──▶ v1 SINDy point forecast ──▶ v2 correction (causal, β·z)
                                                 └─▶ conformal interval (stratified)
                                                 └─▶ V2 tail (L1 look-ahead + regime-aware relaxation)
                                                 └─▶ guarded component selection
                                                          │
                                                          ▼
                                          calibrated operational Dst + 90% band
```

Because the v2 correction and the conformal quantiles are fit only from rows strictly
earlier than the issue time, the operational forecast carries no look-ahead leakage.

## Quick start — discovery (v1)

Generate a synthetic storm, prepare the discovery inputs, discover a sparse equation, and
simulate it forward:

```julia
using SolarSINDy

swd, _ = generate_synthetic_storm(seed=42)
data, dDst = prepare_sindy_data(swd, 1.0; smooth_window=5)
lib = build_minimal_library()

ξ, active_terms, _ = sindy_discover(data, lib, dDst; λ=0.01)
Dst_pred = simulate_sindy(ξ, lib, swd, 1.0)

println(active_terms)
println("Prediction efficiency = ", prediction_efficiency(Dst_pred, swd.Dst_star))
```

## Quick start — calibrated forecast (v2)

Fit the V2 calibration from a replay table of prior forecasts (issue-time
features + realized observations), then issue a corrected point forecast with a
finite-sample conformal interval:

```julia
using SolarSINDy

# 1. Stratified split-conformal calibration from prior residuals (distribution-free band).
#    points / observations / horizons / latest_dsts are equal-length vectors of past forecasts.
cal = fit_conformal(points, observations, horizons, latest_dsts; coverage=0.90)
lo, hi = conformal_interval(cal, point_forecast, horizon, latest_dst)   # scalar query
println("90% band: [", lo, ", ", hi, "]  (target coverage 0.90)")
println("empirical coverage = ", conformal_coverage(cal, points, observations, horizons, latest_dsts))

# 2. V2 causal correction on top of the v1 point forecast.
#    replay_df is a DataFrame of prior issued forecasts + realized observations.
v2cal = fit_operational_v2_calibration(replay_df)            # β·z fit from prior rows only
out = operational_v2_predict(v2cal, v1_point, v1_ci05, v1_ci95, features)
println("v2 Dst = ", out.pred_dst, "  90% band = [", out.ci05_dst, ", ", out.ci95_dst, "]")
```

The conformal layer is distribution-free: it assumes only that calibration and test
residuals are exchangeable, and gives the marginal guarantee
`P(|Y − Ŷ| ≤ Q̂) ≥ ⌈(n+1)(1−α)⌉ / (n+1)`. Quantiles are stratified by forecast horizon
and by activity regime, because storm-time residuals are heavier-tailed than quiet-time
residuals and longer leads carry larger error; a single pooled band would over-cover
quiet/short cases and under-cover storm/long-lead cases.

## Real-time monitoring

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/storm_monitor.jl
```

The monitor:

- fetches near-real-time solar wind from NOAA SWPC (`fetch_realtime_solar_wind`)
- forward-integrates the current eleven-active-term SINDy equation at hourly cadence,
  loading the canonical V2.1 artifacts from `data/`
- propagates the 500-member coefficient ensemble for prediction intervals
- emits configurable storm-severity alarms (`QUIET` / `MODERATE` / `INTENSE` / `SUPERINTENSE`)

Set `STORM_MONITOR_MAX_CYCLES=<n>` to exit after `n` poll cycles (verification / CI).

A locked-live verification harness — which issues a forecast, locks it, and scores it only
after the target hour is observed — is in
[`examples/live_forecast_verify.jl`](examples/live_forecast_verify.jl). The realtime data
path depends on external NOAA SWPC availability.

## Running the live monitor

[`examples/live_monitor.jl`](examples/live_monitor.jl) is the long-running accrual daemon. Each
cycle it issues immutable V2 forecasts at 1/2/3/6 h leads, refreshes observations from the live
Dst feed, scores any pending rows whose target hour has arrived, captures a prospective external
Dst snapshot, and rewrites the comparison report. The managed path is
`bin/solarsindy start monitor` (pidfile, readiness check, then `stop` / `status` /
`logs monitor -f`); the daemon can also be run directly from a fresh clone:

```bash
git clone https://github.com/jake-w-liu/SolarSINDy.jl.git
julia --project=SolarSINDy.jl -e 'using Pkg; Pkg.instantiate()'

# One cycle against the live feeds, then exit:
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_monitor.jl --once

# Continuous daemon (default 3600 s cadence):
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_monitor.jl
```

Configuration is by environment variable:

- `SOLARSINDY_MONITOR_DIR` — output/state directory for the log, comparison report, outage
  sentinel, and external-Dst snapshots (default `<clone>/var/monitor`).
- `SOLARSINDY_V2_CALIBRATION` — V2 calibration CSV (default `<dir>/operational_v2_calibration.csv`).
  When the directory has no calibration, the monitor falls back to the bundled locked calibration
  in [`deploy/`](deploy/) and emits a warning. The conformal interval sidecar is looked up next to
  the calibration (`*_conformal.csv`), also bundled in `deploy/`.
- `SOLARSINDY_MONITOR_ONCE=1` (or `--once`) — run exactly one cycle, then exit (CI/verification).
- `LIVE_MONITOR_INTERVAL_SEC`, `LIVE_MONITOR_MAX_CYCLES`, and
  `LIVE_MONITOR_DEADMAN_CYCLES` — cadence, cycle cap, and issuance dead-man threshold. The
  product horizons are fixed at 1, 2, 3, and 6 h to match the dashboard/API cycle contract. The
  cadence is fixed-rate: cycle runtime is subtracted from the next wait and fully elapsed slots are
  skipped instead of accumulating drift or producing catch-up bursts. Interval selection is also
  atomic across the four-row product: ACI is used only when its point and served residual streams
  are ready at every horizon; otherwise every row uses the static conformal fallback.
- `LIVE_MONITOR_MAX_LOG_ROWS` — maximum retained hot-log rows (default 50,000). Values below
  four are rejected so retention cannot delete part of the latest product cycle.

### macOS launchd service (production)

`bin/solarsindy install-service` (equivalently
[`deploy/install_launchd.sh`](deploy/install_launchd.sh)) renders the tracked launchd
templates and bootstraps three supervised services — the live monitor, the
dashboard/alerting server, and the out-of-process watchdog — with auto-restart
(`KeepAlive`) and the stable juliaup shim, so a `juliaup update` cannot delete the
interpreter out from under a running service. Supervisor console streams go to bounded,
self-rotating files under `var/monitor/logs/` (`launchd.out`/`launchd.err` rotate once per
(re)start, and the daemon mirrors its diagnostic history into the size-capped
`monitor.log` ring), so crash output is retained instead of discarded. Bounded forecast,
state, report, and outage artifacts remain in `var/monitor/`. Remove the services with
`bin/solarsindy uninstall-service`.

### Artifact regeneration

- **Current discovery artifacts** — V2.1 loads
  `data/real_sindy_discovery_coefficients.csv`,
  `data/real_ensemble_inclusion.csv`, and
  `data/real_sindy_ensemble_draws.csv`. These tracked files encode the revised
  20-candidate/11-active-term equation and 500 joint coefficient draws. Regenerate
  them with [`validation/real_data_discovery.jl`](validation/real_data_discovery.jl)
  followed by [`validation/generate_ensemble_draws.jl`](validation/generate_ensemble_draws.jl),
  then rerun the artifact-identity and replay tests before promotion.
- **Historical V2.0 artifacts** — the former 21-candidate/10-active-term files are
  preserved under `data/historical/v2_0/`; their calibration is under
  `deploy/historical/v2_0/`. They are provenance inputs, never current defaults.
- **Conformal interval sidecar** — the primary live interval is the adaptive-conformal band
  derived from the verified log; the stratified sidecar is a cold-start/fallback interval.
  Regenerate it from a verified log with
  [`validation/make_operational_conformal_sidecar.jl`](validation/make_operational_conformal_sidecar.jl)
  (`SOLARSINDY_LIVE_LOG` / `SOLARSINDY_V2_CONFORMAL_SIDECAR` override the paths).
- **One-minute OMNI HRO cache** — the sub-hourly component replays use NASA CDAWeb
  monthly files kept under the ignored validation cache. Fetch and verify the seven-storm
  set with
  `julia --project=. validation/operational/fetch_omni_hro.jl`; pass `YYYYMM`
  arguments to fetch selected months. The fetcher validates the file structure and writes
  a local SHA-256 manifest before the replay consumes the data.

## Web dashboard

A self-contained operational dashboard ships in [`app/`](app/): a minimal-dependency
`HTTP.jl` backend serving a Plotly UI over the locked-live forecast log — current storm level,
the Dst forecast with its calibrated 90% band, a **rolling forecast-vs-observed track** (every
locked forecast plotted against the observation that later arrived), the verified track record,
calibration/skill, a live ground-d*B*/d*t* nowcast, and the Sun → grid
warning chain.

The ground-d*B*/d*t* panel uses the provisional USGS adjusted near-real-time product and is a
GIC-hazard indicator; archival quality control can revise the live magnetic vectors. The bundled
retrospective forecaster uses quasi-definitive ground data and bow-shock-shifted OMNI drivers, so
it is not served from the newest unshifted L1 observation. Its
18/42/66/90 nT/min lines are the unit-converted threshold magnitudes used by
[Pulkkinen et al. (2013)](https://doi.org/10.1002/swe.20056), not a reproduction of that
study's nonoverlapping 20-minute validation protocol or universal grid-risk categories. The optional electric-field value is a
1-D reference-ground estimate; a GIC or grid-impact calculation additionally requires site
conductivity and network topology.

```bash
bin/solarsindy start dashboard    # managed background start (or: cd app && ./run.sh for foreground)
```

The server compiles and caches its log-backed endpoint paths *before* opening the
listener, so the first dashboard hit after a (re)start responds in milliseconds instead
of paying a one-time parse/compile stall.

It reads `<clone>/var/monitor/live_forecast_log.csv` by default (or set
`SOLARSINDY_LOG=/path/to/live_forecast_log.csv`),
falls back to the Plotly CDN when no vendored copy is present, and POSTs webhook alerts on
threat-level changes when `SWM_WEBHOOK_URL` is set. The dashboard has its own lightweight
environment (`app/Project.toml`) and test suite (`app/test/runtests.jl`, also run by the package
`Pkg.test()`); see [`app/README.md`](app/README.md) for endpoints and configuration.

## Data

Pre-computed SINDy coefficients and validation datasets ship in `data/`, available
through both `Pkg.add` and cloned repos. Access them programmatically:

```julia
data_dir = get_data_dir()
coef_csv = joinpath(data_dir, "real_sindy_discovery_coefficients.csv")
```

The current joint SINDy ensemble draws (`data/real_sindy_ensemble_draws.csv`) are versioned
with the 20/11 point artifact. The V2.0 joint draws are retained beside the historical 21/10
core. A fresh clone therefore preserves the cross-term covariance used by each matched
forecast. Regenerate a draw artifact with `validation/generate_ensemble_draws.jl`, then rerun
the artifact-identity and replay gates before replacing a tracked file.

### Fetching the OMNI2 dataset

The large NASA OMNI2 hourly archive used for real-data discovery and backtests is **not**
committed (hundreds of MB, gitignored). Fetch and prepare it from the public NASA SPDF
source with one call:

```julia
df = prepare_omni_data()                                  # download → extract → parse → clean
df = prepare_omni_data(year_start=2010, year_end=2019)    # restrict the year range
```

Or step by step:

```julia
raw       = download_omni2(joinpath(get_data_dir(), "omni_hourly_raw.dat"))
extracted = extract_omni2_columns(raw, joinpath(get_data_dir(), "omni_extracted.csv"))
df        = parse_omni2(extracted; year_start=2010, year_end=2019)
clean_omni_data!(df)
```

The downloaded `omni_hourly_raw.dat` / `omni_extracted.csv` stay under `data/` and remain
gitignored; rerun `prepare_omni_data()` anytime to regenerate them. Every script under
`validation/` reads and writes within the package, so a fresh clone reproduces all results
end to end with no external paths.

## Core API

**Utilities** — `numerical_derivative`, `smooth_moving_average`, `pressure_correct_dst`,
`halfwave_rectify`, `imf_clock_angle`, `get_data_dir`

**Library** — `CandidateLibrary`, `build_solar_wind_library`, `build_minimal_library`,
`evaluate_library`, `get_term_names`

**SINDy** — `stlsq`, `sindy_discover`, `ensemble_sindy`, `sindy_predict`, `simulate_sindy`,
`sweep_lambda`

**Baselines** — `burton_model`, `burton_model_full`, `newell_coupling`,
`obrien_mcpherron_model`, `simulate_burton`, `simulate_burton_full`, `simulate_obrien`

**Synthetic data** — `SolarWindData`, `StormEvent`, `generate_synthetic_storm`,
`generate_multistorm_dataset`, `identify_storm_phases`, `prepare_sindy_data`

**Real-data pipeline** — `download_omni2`, `prepare_omni_data`, `extract_omni2_columns`,
`parse_omni2`, `load_omni2_csv`, `clean_omni_data!`, `StormCatalogEntry`,
`build_storm_catalog`, `extract_storm_data`, `extract_all_storms`, `save_storm_catalog`,
`load_storm_catalog`

**Metrics** — `rmse`, `correlation`, `skill_score`, `prediction_efficiency`,
`metrics_summary`, `wilcoxon_signed_rank_p`

**Forecast (v1 + V2.1 correction and frozen-tail ablation)** — `ForecastState`, `ForecastResult`, `init_forecast`,
`step_forecast!`, `forecast_ahead`, `OperationalV2Calibration`,
`default_operational_v2_calibration`, `operational_v2_feature_tuple`,
`fit_operational_v2_calibration`, `operational_v2_predict`, `score_operational_v2`,
`write_operational_v2_calibration`, `read_operational_v2_calibration`

**Versioned operational core** — `OPERATIONAL_V2_1_MODEL_VERSION`,
`OPERATIONAL_V2_0_MODEL_VERSION`, `OperationalCoreArtifacts`,
`OperationalCalibrationArtifacts`, `OperationalCore`,
`canonical_operational_version`, `operational_core_artifacts`,
`operational_calibration_artifacts`, `validate_operational_core_artifacts`,
`load_operational_core`, `init_operational_forecast`, `operational_core_forecast`

**Conformal UQ** — `ConformalCalibration`, `ConformalStratum`, `fit_conformal`,
`conformal_stratum`, `conformal_halfwidth`, `conformal_interval`, `conformal_coverage`,
`write_conformal_calibration`, `read_conformal_calibration`, `AdaptiveConformal`,
`init_adaptive_conformal`, `adaptive_conformal_step!`, `run_adaptive_conformal`

**Online assimilation (research/shadow only)** — `AssimilationFilter`, `init_assimilation`, `assimilation_predict!`,
`assimilation_update!`, `run_assimilation`, `current_dst`, `current_coeffs`, `dst_variance`

**Alarms** — `StormSeverity` (`QUIET`, `MODERATE`, `INTENSE`, `SUPERINTENSE`), `Alarm`,
`AlarmConfig`, `default_alarm_config`, `check_alarm`, `classify_severity`, `alarm_print`,
`alarm_log`

**Real-time** — `fetch_swpc_plasma`, `fetch_swpc_mag`, `fetch_swpc_dst`,
`fetch_realtime_solar_wind`, `run_monitor`

## Reproducing research results

Research/paper workflows live under `validation/` (not `examples/`):

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/validation/real_data_discovery.jl
```

Useful scripts: `download_omni.jl`, `real_data_discovery.jl`, `phase_dependent_discovery.jl`,
`coupled_discovery.jl`, `significance_tests.jl`, `generate_real_figures.jl`,
`run_validation.jl`.

## Tests

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/test/runtests.jl
```

The suite uses independent expectations—analytical checks, conservation and limiting cases,
and regression baselines—rather than tautologies. Coverage includes:

- analytical checks for the classical baselines
- SINDy synthetic recovery
- forecast-state, operational-v2 correction, and alarm logic
- stratified and adaptive conformal coverage (finite-sample guarantee, exchangeability)
- OMNI parsing, fill-value replacement, cleaning, and storm-catalog extraction
- realtime hourly aggregation and forecast initialization
- live-log duplicate suppression, filesystem locking, and stale-lock recovery

See the package test suites (`test/runtests.jl` and `app/test/runtests.jl`) for coverage,
tolerances, and anti-false-test checks.

## Docs

```bash
julia --project=SolarSINDy.jl/docs -e 'include("SolarSINDy.jl/docs/make.jl")'
```

Doc sources live in `docs/src/` (`index.md`, `api.md`, `examples.md`, `live-verification.md`).

## Notes

- `Manifest.toml` is committed for reproducible research runs.
- The realtime data path depends on external NOAA SWPC availability.
- The monitor writes a local log file and is intended as an operational-prototype workflow.
- Installation is path/URL-based rather than registry-based.

## Release notes

Release notes live in [CHANGELOG.md](CHANGELOG.md).

## Citation

If you use this code in academic work, cite the associated paper/project materials from this
repository; citation metadata is in [CITATION.cff](CITATION.cff). A final archival software
citation can be tightened once the long-term repository URL and paper DOI are fixed.
