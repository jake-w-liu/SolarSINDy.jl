# Contributing

`SolarSINDy.jl` is currently maintained as a research package inside the broader project repository.

## Local Development

From the repository root:

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/test/runtests.jl
julia --project=SolarSINDy.jl/docs -e 'include("SolarSINDy.jl/docs/make.jl")'
```

Please run both commands after code changes that affect package behavior, public examples, or docs.

## Scope

Contributions are especially helpful for:

- documentation clarity
- test coverage for deterministic logic
- packaging polish for a future registry-style release
- reproducibility improvements in the realtime forecasting workflow

## Release Notes

The forecasting example lives in `SolarSINDy.jl/examples/storm_monitor.jl`.

The package is intended to remain honest about its scope:

- discovery, simulation, and validation code are actively supported
- the realtime monitor should be treated as a user-facing example/prototype rather than a hardened operational service
