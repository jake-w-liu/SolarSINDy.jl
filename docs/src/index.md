# SolarSINDy.jl

`SolarSINDy.jl` is a Julia package for sparse equation discovery and operational
Dst forecasting from solar wind–magnetosphere coupling data.

It is designed for research workflows around:

- synthetic storm generation
- sparse identification of nonlinear dynamics (SINDy)
- comparison against classical empirical Dst baselines
- OMNI2 ingestion and storm extraction
- rolling forecast utilities and storm alarms
- an Operational super-learner live product that retains the revised 20-candidate,
  11-active-term SINDy equation within a ten-expert super learner

## Package Scope

The package keeps the discovery equation auditable while providing a guarded,
supervised operational path for locked forecasts.

Core capabilities:

- discover sparse governing equations from storm data
- simulate discovered equations forward in time
- compare against `Burton`, `BurtonFull`, and `OBrienMcP`
- prepare cleaned storm windows from OMNI2 data
- run rolling Dst forecast utilities from saved coefficients
- serve super-learner forecasts with causal ten-expert weighting, depth-stratified
  conformal intervals, ballistically propagated L1 forcing, and explicit
  predecessor fallbacks

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
- `SolarSINDy.jl/examples/live_forecast_verify.jl`

The forecasting paths:

- fetch live solar wind data from NOAA SWPC
- load the versioned inertial-SINDy 20/11 discovery artifacts used by super-learner's sparse experts
- advance rolling forecasts with uncertainty bands
- emit configurable storm alarms

See the [Examples](examples.md) page for the monitor command and the
[Live Verification](live-verification.md) page for the prediction-to-observation
verification workflow.

## Validation Status

The package currently has deterministic automated tests for:

- SINDy discovery and forward simulation
- classical baselines
- forecast state evolution and alarms
- OMNI parsing, fill-value replacement, cleaning, and storm catalog extraction
- realtime hourly aggregation and forecast initialization

The `operational` alias selects the inertial-SINDy base operator from which the current
serving path constructs Operational super-learner. A complete cycle is labelled
`dst-superlearner+sindyfloor+conformal`; static stack and inertial-SINDy are explicit
fallback stages, not the normal served product. Historical SINDy core artifacts are
available only through an explicit version request. Run the complete package
suite and readiness audit before relying on regenerated results or deploying the
monitor.
