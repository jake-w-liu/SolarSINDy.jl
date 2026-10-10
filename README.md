# SolarSINDy.jl

A Julia package for data-driven geomagnetic storm (Dst) forecasting, with a live
operational monitor and web dashboard. It has two layers:

1. **Discovery** — sparse identification of nonlinear dynamics (SINDy) recovers a
   compact solar-wind–magnetosphere coupling equation for Dst (20 candidate terms,
   11 active), alongside classical baselines (Burton, Burton-full,
   O'Brien–McPherron).
2. **Operational forecasting (V2.4e)** — a nonnegative super learner combines ten
   causal experts with weights fitted per model step, issue-time regime, and
   ring-current depth, plus depth-stratified split-conformal intervals. Served
   identity:

   ```text
   v2.4+sindy20x11+superlearner10floor+conformal
   ```

## Requirements

- Julia **≥ 1.12.6** (`[compat].julia` in [Project.toml](Project.toml))
- Network access to NOAA SWPC for the live path only

## Quick start — the live system

```bash
git clone https://github.com/jake-w-liu/SolarSINDy.jl.git
cd SolarSINDy.jl

bin/solarsindy setup    # one-time: check Julia, instantiate both environments
bin/solarsindy start    # forecast daemon + dashboard; opens http://127.0.0.1:8723
bin/solarsindy status   # process / health / issuance-freshness summary
bin/solarsindy stop     # stop both services
```

Other useful commands: `bin/solarsindy once` (single forecast cycle in the
foreground, the right smoke test), `bin/solarsindy open` (reopen the dashboard),
`bin/solarsindy logs monitor -f`, `bin/solarsindy install-service` /
`uninstall-service` (supervised mode: macOS launchd with auto-restart and
watchdog; systemd templates in [`deploy/`](deploy/)).

## The served product

**Experts.** The ten inputs are: the V2.1 operator, the frozen-tail V2.1 form,
an analog driver-continuation expert with refit ridge correction, persistence,
Burton, Burton-full, O'Brien–McPherron, a direct increment-GBM, climatology, and
the static V2.2 regime stack. A 0.60 floor on the four SINDy-lineage experts
applies per cell; because the static stack is itself a six-expert mixture, that
constraint is an expert taxonomy, not a lower bound on sparse-equation
contribution.

**Fallback chain (fail-closed):** V2.4e → static V2.2 stack → V2.1 operator.
Every issued row records which stage served it (`model_version`), the chosen
cell, the bundle digest, and the partner centers used for alerting.

**Alerting is separate from the point forecast.** The published severity is the
*deepest* of the served center, the static-stack center, and the V2.1 center, so
a blended forecast can never downgrade a warning a predecessor would have
raised.

**Bundle.** Served artifacts live in `deploy/v2_4/` (stack weights, boosted
models, analog archive, refit correction, climatology timescale, conformal
half-widths, guard record, SHA-256 manifest) and are rebuilt by
`validation/operational/v2_4_build_deploy.jl`. Every file is digest-verified on
load; a bundle that fails any check does not serve.

**Shadow product.** The V2.3 analog-continuation model is recorded for
monitoring only — its confirmatory scoring returned `NO_GO`, so it never reaches
the served center or alerts. It ships in `deploy/v2_3_shadow/`.

**Identity oracles** reproduce the served, stack, and shadow centers offline:

```bash
julia --project=. validation/operational/v2_4_serving_identity.jl
julia --project=. validation/operational/v2_2_served_identity.jl
julia --project=. validation/operational/v2_3_serving_identity.jl
```

**Readiness audit** (while services run):

```bash
julia --project=. validation/operational/v2_readiness_audit.jl \
  --write-report --require-api --require-fresh
```

## Live monitor

The daemon issues immutable forecasts at 1/2/3/6 h (or the shifted step set when
the Dst anchor lags by one hour), scores them after the observation matures,
captures an external SWPC Dst snapshot as an independent comparator, and rewrites
the comparison report. State lives under `var/monitor/` by default.

Key environment variables (full list: `bin/solarsindy help`, template
[`deploy/solarsindy.env.example`](deploy/solarsindy.env.example)):

| Variable | Purpose |
|---|---|
| `SWM_HOST`, `SWM_PORT` | dashboard bind host/port |
| `SOLARSINDY_MONITOR_DIR` | monitor state directory |
| `SWM_WEBHOOK_URL` | POST alerts on threat-level changes |
| `SOLARSINDY_LOG` | dashboard log path override |
| `LIVE_MONITOR_INTERVAL_SEC`, `LIVE_MONITOR_MAX_CYCLES`, `LIVE_MONITOR_DEADMAN_CYCLES` | cadence, cycle cap, dead-man threshold |
| `LIVE_MONITOR_PHASE_SAMPLING=1` | alternate issue phase each hour (prospective calibration evidence) |
| `LIVE_MONITOR_MAX_LOG_ROWS` | hot-log cap (default 50,000; older rows cold-archive) |
| `SOLARSINDY_NO_OPEN=1` | never auto-open the browser |

The hot log is capped and rolls mature rows into an append-only cold archive;
daemon logs are ring-rotated. `var/monitor/source_cache/` (raw feed receipts)
grows unboundedly — prune it periodically.

## Dashboard

`app/` contains a self-contained HTTP.jl + Plotly UI over the locked-live log:
served forecast with its empirical 90% band, forecast-vs-observed track, verified
track record, calibration/skill, a ground-d*B*/d*t* nowcast, and the warning
chain. Coverage is labeled empirical; best-method highlighting waits for enough
matured rows. See [`app/README.md`](app/README.md) for endpoints.

## Using the library

```julia
using Pkg
Pkg.add(url="https://github.com/jake-w-liu/SolarSINDy.jl")
```

Discovery quick start:

```julia
using SolarSINDy

swd, _ = generate_synthetic_storm(seed=42)
data, dDst = prepare_sindy_data(swd, 1.0; smooth_window=5)
lib = build_minimal_library()

ξ, active_terms, _ = sindy_discover(data, lib, dDst; λ=0.01)
Dst_pred = simulate_sindy(ξ, lib, swd, 1.0)
```

Calibration example:

```julia
cal = fit_conformal(points, observations, horizons, latest_dsts; coverage=0.90)
lo, hi = conformal_interval(cal, point_forecast, horizon, latest_dst)

v2cal = fit_operational_v2_calibration(replay_df)          # β·z fit, causal
out = operational_v2_predict(v2cal, v1_point, v1_ci05, v1_ci95, features)
```

## Data

- Pre-computed SINDy coefficients and validation data ship in `data/` (tracked).
- The large NASA OMNI2 hourly archive is **not** committed; fetch and prepare it
  with `prepare_omni_data()` — downloads, parses, and cleans in one call.
- The historical 21/10 V2.0 core is kept under `data/historical/v2_0/` and
  `deploy/historical/v2_0/` for provenance comparisons only; the `v2` API alias
  resolves to V2.1, so loading V2.0 requires an explicit version.

## Development

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/test/runtests.jl      # test suite
julia --project=SolarSINDy.jl/docs -e 'include("SolarSINDy.jl/docs/make.jl")'  # docs
```

- Tests use independent oracles (analytical, conservation, regression). A few
  oracles need generated artifacts under `validation/output/`; absent ones are
  registered as skipped, or as failures with
  `SOLARSINDY_REQUIRE_LOCAL_ARTIFACTS=1`.
- Research scripts live under `validation/` (`real_data_discovery.jl`,
  `generate_ensemble_draws.jl`, `run_validation.jl`, …). The package app has its
  own environment (`app/Project.toml`) and tests (`app/test/runtests.jl`).
- `Manifest.toml` is committed for reproducible runs.

## Repository map

```text
src/            package source (discovery, calibration, serving, monitor glue)
app/            dashboard + alerting server
bin/solarsindy  service CLI (setup/start/stop/status/logs/install-service)
deploy/         served bundles, launchd + systemd templates, env template
validation/     research and operational-validation scripts
data/           tracked coefficients, catalogs, ensemble draws
examples/       runnable examples (live monitor, verification, review)
docs/           Documenter sources; DOCS_REPORT.md records coverage
var/monitor/    live state (gitignored): logs, forecast log, cold archive
```

## Notes

- The realtime path depends on NOAA SWPC availability; missing feeds fail closed.
- EKF-on-SINDy remains research infrastructure — it failed promotion gates and
  is not part of serving.
- Release notes: [CHANGELOG.md](CHANGELOG.md). Citation: [CITATION.cff](CITATION.cff).
