# SolarSINDy.jl

`SolarSINDy.jl` is a Julia package for sparse equation discovery and forecast-style evaluation of solar wind-magnetosphere coupling models.

It is designed for research workflows around:

- synthetic storm generation
- sparse identification of nonlinear dynamics (SINDy)
- comparison against classical empirical Dst baselines
- OMNI2 ingestion and storm extraction
- rolling forecast utilities and storm alarms

## Package Scope

This package is aimed at reproducible research code rather than a large production framework.

Core capabilities:

- discover sparse governing equations from storm data
- simulate discovered equations forward in time
- compare against `Burton`, `BurtonFull`, and `OBrienMcP`
- prepare cleaned storm windows from OMNI2 data
- run rolling Dst forecast utilities from saved coefficients

## Installation

From a local checkout:

```julia
using Pkg
Pkg.develop(path="SolarSINDy.jl")
```

## Quick Start

```julia
using SolarSINDy

swd, _ = generate_synthetic_storm(seed=42)
data, dDst = prepare_sindy_data(swd, 1.0; smooth_window=5)
lib = build_minimal_library()

ξ, active_terms, _ = sindy_discover(data, lib, dDst; λ=0.01)
Dst_pred = simulate_sindy(ξ, lib, swd, 1.0)

println(active_terms)
println(prediction_efficiency(Dst_pred, swd.Dst_star))
```

## Forecasting

The realtime forecasting example is not hidden in the validation pipeline. It is provided directly in:

- `SolarSINDy.jl/examples/storm_monitor.jl`

That example:

- fetches live solar wind data from NOAA SWPC
- loads saved discovered coefficients
- advances rolling forecasts with uncertainty bands
- emits configurable storm alarms

See the [Examples](examples.md) page for the exact command.

## Validation Status

The package currently has deterministic automated tests for:

- SINDy discovery and forward simulation
- classical baselines
- forecast state evolution and alarms
- OMNI parsing, fill-value replacement, cleaning, and storm catalog extraction
- realtime hourly aggregation and forecast initialization

The current local suite result is `157/157` passing tests.
