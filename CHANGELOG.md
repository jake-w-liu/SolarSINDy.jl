# Changelog

All notable changes to `SolarSINDy.jl` will be documented in this file.

## [Unreleased] - 2026-07-30

Correctness, robustness, and operational-readiness improvements.

User operation and startup:

- `bin/solarsindy` — one-command control of the live forecast system from a fresh clone:
  `setup` / `start` / `stop` / `restart` (targets `monitor|dashboard|all`), `status`, `once`,
  `logs [-f]`, `open`, `install-service` / `uninstall-service`. Pidfiles live inside the
  instance state dir with a PID-reuse guard (a pidfile PID whose command line is not this
  clone's daemon entry point is stale by definition and never signaled), concurrent starts
  are serialized by a stale-safe lock so a double `start` cannot strand orphan daemons,
  readiness waits are bounded, the installed launchd/systemd service mode is detected and
  never duplicated, and a busy dashboard port is refused with a clear message. Optional
  gitignored `solarsindy.env` supplies defaults (template in
  `deploy/solarsindy.env.example`); explicitly set environment variables win, malformed
  lines are skipped with a warning instead of aborting (recovery commands keep working
  with a broken config), and CRLF files are tolerated. An interactive `start` auto-opens
  the dashboard in the browser (TTY-gated so scripted/headless/service invocations never
  do; `SOLARSINDY_NO_OPEN=1` disables; `open` reopens a closed tab).
- dashboard endpoint warm-up: the server compiles and caches the log-backed endpoint paths
  (log parse + forecast/status/history builders + JSON serialization) before opening the
  listener, converting a measured >20 s first-request JIT stall — which held the log-cache
  lock and blocked `/api/health` behind it — into ordinary startup time (first hit after
  warm-up: milliseconds).
- dashboard startup banner and warm-up lines are explicitly flushed, so nohup/launchd log
  files capture them immediately instead of losing them to block buffering (crash forensics
  and readiness checks read these lines).

Causality and leakage:

- causal (forward-fill) OMNI cleaning for replay and serving inputs, so no gap hour is filled with a value measured after it
- purged/embargoed train/validation/test splits so a boundary anchor's multi-hour target is not used both to fit and to score the residual correction
- replay separates the driver-completeness mask (anchors) from the observed-Dst mask (scored targets), removing a bias toward easier hours
- timestamp-based memory features (Dst/Bz/VBsouth deltas and rolling means) so tied multi-horizon issue times and gap-skipped anchors resolve to true hourly differences
- anchor selection by newest timestamp rather than feed position
- strict causal issuance cutoffs: source-clock tolerance still diagnoses small timestamp skew, but no post-issue plasma, magnetic-field, or Dst sample can enter a forecast
- six-hour hard freshness ceilings for both the solar-wind driver and the observed Dst state

Physics and statistics:

- canonical single-source Dst*/dynamic-pressure helpers (`dynamic_pressure`, `dst_to_dst_star`, `dst_star_to_dst`, `resolve_pdyn`) shared by training, replay, and serving; carried-forward/quiet-time pressure fallback replaces the `Pdyn = 0` (`Dst + 11`) approximation
- corrected Burton (1975) injection to the threshold-continuous form (`α = 5.4e-3`, offset at the 0.5 mV/m threshold); pressure-correction constants re-attributed to O'Brien & McPherron (2000)
- STLSQ final-threshold fixed point so every returned support satisfies the sparsity contract; `collinearity_diagnostics` and an optional true-bootstrap resampling mode for ensemble SINDy
- interval honesty: monotone-safe stratified-conformal fallback, ACI residual pools keyed on the served model, Eq. (13) projection on served/display centers, and documented bounded-band scope for the adaptive-conformal recursion
- exact target-step driver provenance for the served sub-hourly relaxed tail, rather than the pre-relaxation hourly driver

Industrial robustness:

- RTSW endpoint migration from the retired array-of-arrays `*-1-day`/`*-7-day.json` products to the named-key `rtsw_{wind,mag}_1m.json` feeds, with active-source selection and physical-range validation
- forecast-log read-modify-write under a shared lock with identity-based row relocation; scored-row dedup by (issue hour, target, model)
- forecast cycles keyed on the UTC issue hour (not solar-wind vintage or Dst anchor), so a later cycle may legitimately reissue an overlapping target while same-cycle retries remain idempotent
- staleness/expiry flags, health "stale" status, and a bounded-retry Kyoto Dst fetch
- monitor advances one model hour per new hourly bin (watchdog against free-run), with bounded history, log rotation, and per-target horizon-alarm dedup
- fixed-rate live-monitor scheduling subtracts cycle runtime and skips missed slots, preventing cumulative hourly issuance drift and catch-up bursts
- fixed 1, 2, 3, and 6 h product horizons, with health determined from the exact API cycle contract; partial or internally inconsistent rows remain failed even when all four calls returned
- hot-log retention rejects limits below four rows, so it cannot truncate a complete product cycle immediately after validation
- cycle-level interval selection uses ACI only when both point and served residual streams are ready at the model-step lead corresponding to every 1, 2, 3, and 6 h target; otherwise the entire issuance uses the shared static conformal fallback
- per-key single-flight refreshes, bounded waits, and failure cooldowns for USGS dB/dt and station-network requests, preventing duplicate work or repeated requests to a failed service
- one shared NOAA/USGS execution slot for potentially blocking DNS, TLS, and HTTP work, leaving the other server threads available while cached or unavailable responses return immediately
- observed dB/dt nowcast fallback from FRD to CMO for the default dashboard path; explicit station requests remain exact and never fall back silently
- dB/dt artifact schema 3 records quasi-definitive ground provenance, bow-shock-shifted OMNI training drivers, the unshifted L1 live source, and a mandatory disabled-serving flag; offline values are empirical exceedance scores rather than calibrated probabilities
- the live dB/dt route fails closed at the unvalidated L1-to-bow-shock feature transfer: it serves the observed ground nowcast but withholds the retrospective 30-minute forecast instead of evaluating it on a different driver time reference

Dashboard and API:

- log-independent endpoints stay up when the forecast log is absent; internal errors no longer echo the log path
- input-staleness demotion and served-pipeline capability labels surfaced to the front end
- stale or unavailable status responses clear previously displayed metrics and WATCH state instead of leaving an obsolete forecast on screen
- responsive WATCH placement in normal document flow so it cannot cover metric labels
- a bounded four-thread default across shell and Docker launch paths, leaving request capacity during the serialized upstream refresh while preserving the one-request public-data gate
- WCAG-AA contrast for muted labels on every dashboard background surface
- conformal WATCH values are identified as the lower edge of a displayed calibrated 90% interval, without treating the symmetric interval as a one-sided confidence bound or storm probability; value-equivalent legacy API keys remain for already-loaded clients
- ground dB/dt is presented as a GIC-hazard indicator, with the four values identified as unit-converted Pulkkinen et al. threshold magnitudes rather than a reproduction of their nonoverlapping-window protocol or named grid-risk tiers; unsupported generic geoelectric/GIC risk categories were removed
- browser notifications use the stronger of the point-forecast and interval-edge WATCH ranges; unsupported explicit dB/dt stations and malformed query encodings return HTTP 400 instead of aliasing to FRD or surfacing as server errors
- deployment parity across the committed Julia 1.12.6 app environment, Docker image, bundled models, and operational-evidence path; launchers now fail closed when Julia or dependency setup is unsuitable

Provenance and tests:

- discovery provenance sidecar, persisted served point fit, and a joint posterior-draws ensemble artifact; fill-fabricated storm rows excluded from discovery
- the joint-draws artifact (`data/real_sindy_ensemble_draws.csv`) is local-only and not committed; regenerate it with `validation/generate_ensemble_draws.jl` (`init_forecast` falls back to marginal per-term sampling when it is absent)
- canonical and legacy PlotlySupply figure generation preserves the submitted full-width and vertical-panel geometry without loading the desktop synchronization layer
- equal-length normalization of Chromium PDF title and timestamp metadata makes repeated PlotlySupply exports byte-identical without changing PDF object offsets or plotted content
- canonical source identity excludes generated validation output and records portable forward-slash paths
- external-Dst snapshot collection stages network responses before installation and enforces bidirectional raw/log ownership markers under a canonical lock order, preventing shared-path races and unsafe cleanup
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
