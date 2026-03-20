# SolarSINDy.jl

`SolarSINDy.jl` is a Julia package for discovering and evaluating closed-form solar wind-magnetosphere coupling models using sparse identification of nonlinear dynamics (SINDy).

Version: `0.1.0`

The package supports:

- synthetic storm generation for controlled validation
- sparse equation discovery from storm time series
- classical baseline models (`Burton`, `BurtonFull`, `OBrienMcP`)
- real-data ingestion and storm catalog extraction from OMNI2
- rolling forecast utilities and storm-severity alarms


## Installation

From a local checkout:

```julia
using Pkg
Pkg.develop(path="SolarSINDy.jl")
```

Then load the package with:

```julia
using SolarSINDy
```

## Data

The package includes pre-computed SINDy coefficients and validation datasets in `data/`. These files are included with the source repository.

- Access data programmatically with `get_data_dir()`:
  ```julia
  data_dir = get_data_dir()  # Returns path to data/ directory
  coef_csv = joinpath(data_dir, "real_sindy_discovery_coefficients.csv")
  ```

- **From source repo**: Run `julia --project=SolarSINDy.jl` from the cloned directory.
- **Via Pkg.add()**: (Future support) Data will be downloaded as an artifact on first use and cached locally.

## Quick Start

Generate a synthetic storm, prepare the discovery inputs, discover a sparse equation, and simulate it forward:

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

## Core API

Main exported entry points:

- data utilities: `generate_synthetic_storm`, `prepare_sindy_data`, `identify_storm_phases`
- discovery: `build_solar_wind_library`, `build_minimal_library`, `sindy_discover`, `ensemble_sindy`, `simulate_sindy`
- baselines: `simulate_burton`, `simulate_burton_full`, `simulate_obrien`
- real-data pipeline: `download_omni2`, `extract_omni2_columns`, `parse_omni2`, `clean_omni_data!`, `build_storm_catalog`
- forecast utilities: `init_forecast`, `step_forecast!`, `forecast_ahead`
- alarms: `default_alarm_config`, `check_alarm`, `classify_severity`

## Forecasting Example

The forecasting example is in:

- [examples/storm_monitor.jl](/Users/jake/EMPIRE/projects/ongoing/2026_045/SolarSINDy.jl/examples/storm_monitor.jl)

Run it with:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/examples/storm_monitor.jl
```

That example:

- fetches near-real-time solar wind data from NOAA SWPC
- loads the discovered SINDy coefficients from `data/` (via `get_data_dir()`)
- runs rolling forecasts with ensemble uncertainty bands
- emits configurable storm alarms

The example is designed to work from a cloned repository. For package installations (future), data will be available via Julia Artifacts.

## Reproducing Research Results

The paper/research workflows live under `validation/`, not `examples/`.

Useful scripts include:

- `SolarSINDy.jl/validation/download_omni.jl`
- `SolarSINDy.jl/validation/real_data_discovery.jl`
- `SolarSINDy.jl/validation/phase_dependent_discovery.jl`
- `SolarSINDy.jl/validation/coupled_discovery.jl`
- `SolarSINDy.jl/validation/generate_real_figures.jl`

Run them from the repository root, for example:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/validation/real_data_discovery.jl
```

## Tests

Run the package tests with:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/test/runtests.jl
```

Current local result:

- `157/157` tests passing

Coverage now includes:

- analytical checks for the classical baselines
- SINDy synthetic recovery tests
- forecast-state and alarm logic
- OMNI parsing, fill-value replacement, cleaning, storm catalog extraction
- realtime hourly aggregation and forecast initialization

See:

- [TEST_REPORT.md](/Users/jake/EMPIRE/projects/ongoing/2026_045/TEST_REPORT.md)

## Docs

Build the package docs with:

```bash
julia --project=SolarSINDy.jl/docs -e 'include("SolarSINDy.jl/docs/make.jl")'
```

The docs sources live in `SolarSINDy.jl/docs/src/`.

## Release Notes

The current package release notes live in `SolarSINDy.jl/CHANGELOG.md`.

## Notes

- `Manifest.toml` is included here for reproducible research runs, not because this is already a polished registry release.
- The realtime data path depends on external NOAA SWPC availability.
- The monitor writes a local log file and is intended as an example/prototype workflow.

## Citation

If you use this code in academic work, cite the associated paper/project materials from this repository. Citation metadata is provided in `SolarSINDy.jl/CITATION.cff`. A final archival software citation can be tightened further once the long-term repository URL and paper DOI are fixed.
