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
- loads the discovered SINDy coefficients from `paper/data/`
- advances rolling forecasts with ensemble intervals
- prints storm alarms based on configurable thresholds

This is the package's primary user-facing forecasting example.

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
