# Changelog

All notable changes to `SolarSINDy.jl` will be documented in this file.

## [0.1.0] - 2026-03-20

Initial public package-polish release for the research codebase.

Highlights:

- package-specific `README.md` with corrected quickstart and forecasting entry point
- MIT `LICENSE`
- `CITATION.cff` metadata
- buildable `Documenter.jl` docs under `docs/`
- expanded deterministic test coverage for baselines, data cleaning, storm catalog logic, realtime aggregation, and forecast initialization
- release housekeeping files including `.gitignore`

Notes:

- installation remains path-based rather than registry-based
- the realtime monitor is included as an example/prototype workflow
