# Changelog

All notable changes to `SolarSINDy.jl` will be documented in this file.

## [Unreleased] - 2026-07-13

Correctness, robustness, and completeness audit campaign.

Causality and leakage:

- causal (forward-fill) OMNI cleaning for replay and serving inputs, so no gap hour is filled with a value measured after it
- purged/embargoed train/validation/test splits so a boundary anchor's multi-hour target is not used both to fit and to score the residual correction
- replay separates the driver-completeness mask (anchors) from the observed-Dst mask (scored targets), removing a bias toward easier hours
- timestamp-based memory features (Dst/Bz/VBsouth deltas and rolling means) so tied multi-horizon issue times and gap-skipped anchors resolve to true hourly differences
- anchor selection by newest timestamp rather than feed position

Physics and statistics:

- canonical single-source Dst*/dynamic-pressure helpers (`dynamic_pressure`, `dst_to_dst_star`, `dst_star_to_dst`, `resolve_pdyn`) shared by training, replay, and serving; carried-forward/quiet-time pressure fallback replaces the `Pdyn = 0` (`Dst + 11`) approximation
- corrected Burton (1975) injection to the threshold-continuous form (`α = 5.4e-3`, offset at the 0.5 mV/m threshold); pressure-correction constants re-attributed to O'Brien & McPherron (2000)
- STLSQ final-threshold fixed point so every returned support satisfies the sparsity contract; `collinearity_diagnostics` and an optional true-bootstrap resampling mode for ensemble SINDy
- interval honesty: monotone-safe stratified-conformal fallback, ACI residual pools keyed on the served model, Eq. (13) projection on served/display centers, and documented bounded-band scope for the adaptive-conformal recursion

Industrial robustness:

- RTSW endpoint migration from the retired array-of-arrays `*-1-day`/`*-7-day.json` products to the named-key `rtsw_{wind,mag}_1m.json` feeds, with active-source selection and physical-range validation
- forecast-log read-modify-write under a shared lock with identity-based row relocation; scored-row dedup by (anchor, target, model)
- forecast cycle keyed on issue epoch (not solar-wind vintage), with staleness/expiry flags, health "stale" status, and a bounded-retry Kyoto Dst fetch
- monitor advances one model hour per new hourly bin (watchdog against free-run), with bounded history, log rotation, and per-target horizon-alarm dedup

Dashboard and API:

- log-independent endpoints stay up when the forecast log is absent; internal errors no longer echo the log path
- input-staleness demotion and served-pipeline capability labels surfaced to the front end

Provenance and tests:

- discovery provenance sidecar, persisted served point fit, and a joint posterior-draws ensemble artifact; fill-fabricated storm rows excluded from discovery
- the joint-draws artifact (`data/real_sindy_ensemble_draws.csv`) is local-only and not committed; regenerate it with `validation/generate_ensemble_draws.jl` (`init_forecast` falls back to marginal per-term sampling when it is absent)
- new deterministic test oracles covering the fixes above

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
