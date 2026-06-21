# SolarSINDy.jl

`SolarSINDy.jl` is a Julia package for **data-driven geomagnetic storm (Dst) forecasting**.
It combines sparse equation discovery with a calibrated operational forecasting layer, and
runs end to end from synthetic validation through real NASA OMNI2 discovery to a live
NOAA SWPC–driven monitor.

The package is organized as two layers:

1. **Discovery (v1).** Sparse identification of nonlinear dynamics (SINDy) recovers a
   closed-form solar wind–magnetosphere coupling equation for the Dst index from storm
   time series, alongside classical physical baselines.
2. **Operational forecasting (v2).** A causal post-processing layer corrects the v1 point
   forecast, attaches **distribution-free conformal predictive intervals** with
   finite-sample coverage, and selects a guarded fallback component — producing a
   calibrated operational Dst forecast suitable for live use.

## Capabilities

**Discovery and modeling (v1)**

- synthetic storm generation for controlled validation
- sparse equation discovery (STLSQ, ensemble SINDy, λ sweeps) from storm time series
- a candidate coupling-function library (rectified `VBs`, clock angle, dynamic pressure, …)
- classical baselines: Burton, Burton-full, and O'Brien–McPherron
- NASA OMNI2 ingestion, cleaning, and storm-catalog extraction

**Operational forecasting (v2)**

- a causal correction layer, `Dst_v2 = Dst_v1 + β₀ + Σ βⱼ zⱼ`, fit only from prior
  (replay/live) rows so it never looks ahead of the issue time
- stratified split-conformal predictive intervals with finite-sample coverage, stratified
  by lead time × geomagnetic activity regime
- adaptive (online) conformal updating under distribution shift
- guarded component selection over corrected SINDy, uncorrected SINDy v1, persistence,
  Burton, Burton-full, and O'Brien–McPherron, deployed only after chronological validation
- online assimilation of the ring-current state
- forecast skill metrics (RMSE, correlation, skill score, prediction efficiency, Wilcoxon)

**Real-time**

- NOAA SWPC plasma / magnetic-field / Dst fetchers
- a rolling monitor loop with calibrated storm-severity alarms

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

Requires Julia 1.10+. `Manifest.toml` is committed for reproducible research runs.

## Architecture: discovery → operational calibration

The two layers are deliberately decoupled. The v1 SINDy model is a fixed, interpretable
dynamical core. The v2 calibration is a thin causal layer fit on top of replayed or live
forecasts; it **does not** modify the v1 coefficients or the forecast state. This keeps the
discovered physics auditable while letting the operational forecast adapt:

```
issue-time drivers ──▶ v1 SINDy point forecast ──▶ v2 correction (causal, β·z)
                                                 └─▶ conformal interval (stratified)
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

Fit the operational v2 calibration from a replay table of prior forecasts (issue-time
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

# 2. Operational v2 causal correction on top of the v1 point forecast.
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
- loads the discovered SINDy coefficients from `data/` (via `get_data_dir()`)
- runs rolling forecasts with ensemble + conformal uncertainty bands
- emits configurable storm-severity alarms (`QUIET` / `MODERATE` / `INTENSE` / `SUPERINTENSE`)

A locked-live verification harness — which issues a forecast, locks it, and scores it only
after the target hour is observed — is in
[`examples/live_forecast_verify.jl`](examples/live_forecast_verify.jl). The realtime data
path depends on external NOAA SWPC availability.

## Data

Pre-computed SINDy coefficients and validation datasets ship in `data/` (~830 KB), available
through both `Pkg.add` and cloned repos. Access them programmatically:

```julia
data_dir = get_data_dir()
coef_csv = joinpath(data_dir, "real_sindy_discovery_coefficients.csv")
```

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

**Forecast (v1 + operational v2)** — `ForecastState`, `ForecastResult`, `init_forecast`,
`step_forecast!`, `forecast_ahead`, `OperationalV2Calibration`,
`default_operational_v2_calibration`, `operational_v2_feature_tuple`,
`fit_operational_v2_calibration`, `operational_v2_predict`, `score_operational_v2`,
`write_operational_v2_calibration`, `read_operational_v2_calibration`

**Conformal UQ** — `ConformalCalibration`, `ConformalStratum`, `fit_conformal`,
`conformal_stratum`, `conformal_halfwidth`, `conformal_interval`, `conformal_coverage`,
`write_conformal_calibration`, `read_conformal_calibration`, `AdaptiveConformal`,
`init_adaptive_conformal`, `adaptive_conformal_step!`, `run_adaptive_conformal`

**Online assimilation** — `AssimilationFilter`, `init_assimilation`, `assimilation_predict!`,
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

Current result: **686/686 passing** (deterministic, ~1m40s). The suite uses independent
expectations — analytical checks, conservation/limiting cases, and regression baselines —
rather than tautologies. Coverage includes:

- analytical checks for the classical baselines
- SINDy synthetic recovery
- forecast-state, operational-v2 correction, and alarm logic
- stratified and adaptive conformal coverage (finite-sample guarantee, exchangeability)
- OMNI parsing, fill-value replacement, cleaning, and storm-catalog extraction
- realtime hourly aggregation and forecast initialization

See [TEST_REPORT.md](../TEST_REPORT.md) for coverage, tolerances, and anti-false-test notes.

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
