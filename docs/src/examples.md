# Examples

## Realtime Forecasting Example

The dedicated forecasting example is:

- `SolarSINDy.jl/examples/storm_monitor.jl`

Run it from the repository root:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/storm_monitor.jl
```

This example:

- pulls near-real-time solar wind data from NOAA SWPC
- loads the versioned inertial-SINDy discovery artifacts from the package `data/` directory
- advances rolling forecasts with ensemble intervals
- prints storm alarms based on configurable thresholds

This is the package's primary discovery-core forecasting example. The managed
live product uses `examples/live_monitor.jl`, which issues Operational super-learner at
1, 2, 3, and 6 h. The super-learner combines ten experts, including two forms of the inertial-SINDy
predecessor, and records any fallback stage explicitly.

## Live Verification Example

To issue a forecast now and verify it when the future Kyoto Dst observation
arrives:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --wait
```

The script writes locked prediction rows to
`SolarSINDy.jl/var/monitor/live_forecast_log.csv` before the target observation exists,
then updates the same row with the observed Dst and residual after the target
hour is published. It also logs persistence, Burton, BurtonFull, and
O'Brien--McPherron baselines for the same target.

The default `--model=inertial-sindy` option (alias `operational`) selects the base operator used to construct
the served super-learner cycle. Pass `--model=sindy-core` only for an explicit uncalibrated-core
comparison. The served-model field, rather than the base-model option, identifies
which operational stage produced each row.

Other useful modes:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --issue
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --verify-pending
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --backfill-baselines
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/live_forecast_verify.jl --summary
```

## Synthetic Discovery Example

```julia
using SolarSINDy

swd, _ = generate_synthetic_storm(seed=42)
data, dDst = prepare_sindy_data(swd, 1.0; smooth_window=5)
lib = build_minimal_library()

ξ, active_terms, _ = sindy_discover(data, lib, dDst; λ=0.01)
Dst_pred = simulate_sindy(ξ, lib, swd, 1.0)

println(active_terms)
println("PE = ", prediction_efficiency(Dst_pred, swd.Dst_star))
```

## Research Reproduction Scripts

`examples/duration_mean_synthetic.jl` fits a synthetic duration law, publishes and
reloads its immutable model, checks as-of causality, and recovers two nonnegative
gains from independently specified Brownian interval targets using the computed
mean-only affine design. It consumes no actual archive or forecast outcomes.

The paper-style workflows live in `validation/`, not `examples/`.

Common scripts:

- `SolarSINDy.jl/validation/download_omni.jl`
- `SolarSINDy.jl/validation/real_data_discovery.jl`
- `SolarSINDy.jl/validation/phase_dependent_discovery.jl`
- `SolarSINDy.jl/validation/coupled_discovery.jl`
- `SolarSINDy.jl/validation/generate_real_figures.jl`

Example command:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/validation/real_data_discovery.jl
```
