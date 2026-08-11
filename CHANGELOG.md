# Changelog

All notable changes to `SolarSINDy.jl` will be documented in this file.

## [0.2.1] - 2026-08-08

Correctness, robustness, and operational-readiness improvements.

Operational V2.1 alignment:

- the default operational artifact boundary now serves the revised
  20-candidate/11-active-term discovery equation; the former 21/10 core and its
  calibration are available only through an explicit V2.0 historical request
- the `v2` alias, live verifier, monitor, API, dashboard, examples, and replay
  scripts resolve to V2.1; explicit `--model=v1` remains available for the
  uncalibrated discovery-core forecast
- current and historical 500-member joint coefficient draws are tracked with
  their matching artifacts, so a fresh clone preserves cross-term covariance
- development-lineage tail and EKF experiments load the revised core and current
  calibration feature schema while remaining clearly separated from the served
  V2.1 composition
- a verified NASA CDAWeb one-minute OMNI HRO fetcher supports the sub-hourly
  component replays and writes a local SHA-256 source manifest
- complete-hour causal replay of the served V2.1 stack is frozen and audited on
  the locked chronological holdout with static conformal offsets and zero
  holdout updates; it does not reconstruct fractional subhourly live windows,
  pooled, lead-specific, and storm-regime coverage remain separately visible

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
- watchdog functional probe: in addition to `/api/health`, the watchdog probes a data
  route (`/api/forecast`, generous timeout, own `dash_wedged` dedup kind) — an observed
  wedge state answered health quickly while data routes hung, which a health-only probe
  cannot see; detection, sentinel, webhook dedup, and recovery verified against a stub
  server reproducing exactly that state.
- start-lock identity semantics: a lock is honored only when its recorded owner PID is a
  live solarsindy CLI process (same `ps` command-line discipline as the pidfile guard), so
  a `SIGKILL`-ed start whose lock PID is later recycled by an unrelated process can no
  longer wedge future starts; a genuine concurrent start still refuses.
- live-monitor per-step cycle timings: every guarded cycle step logs its wall-clock
  duration (`step issue h=1 ok in … s`), so a slow or stalled cycle names the step that
  consumed the time instead of leaving an unattributable gap; all cycle network fetches
  were verified bounded (`connect_timeout=15`, `readtimeout=30` at all three call sites).
- CLI polish: multi-line pidfiles are stale by definition (never "repaired" into their
  first line); with `SWM_HOST=0.0.0.0` the probe/browse URL uses loopback instead of the
  unusable `http://0.0.0.0:…`; config values with unquoted whitespace and parser-reserved
  variable names each get an explicit warning.
- dashboard endpoint warm-up: the server compiles and caches the log-backed endpoint paths
  (log parse + forecast/status/history builders + JSON serialization) before opening the
  listener, converting a measured >20 s first-request JIT stall — which held the log-cache
  lock and blocked `/api/health` behind it — into ordinary startup time (first hit after
  warm-up: milliseconds).
- dashboard startup banner and warm-up lines are explicitly flushed, so nohup/launchd log
  files capture them immediately instead of losing them to block buffering (crash forensics
  and readiness checks read these lines).
- CLI dashboard launch parity: `bin/solarsindy start dashboard` runs the server with
  `--compile=min` (matching the supervised launchd deployment), so a first cold upstream
  refresh cannot monopolize Julia code generation and stall every HTTP route after start;
  the numerical-kernel monitor daemon is intentionally left at full compilation.

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
- tolerant RTSW JSON decoding at both monitor and dashboard boundaries for NOAA's occasional bare `NaN` missing-value tokens; required non-finite or out-of-range physical fields remain rejected before forecasting or display
- forecast-log read-modify-write under a shared lock with identity-based row relocation; scored-row dedup by (issue hour, target, model)
- forecast cycles keyed on the UTC issue hour (not solar-wind vintage or Dst anchor), so a later cycle may legitimately reissue an overlapping target while same-cycle retries remain idempotent
- staleness/expiry flags, health "stale" status, and a bounded-retry Kyoto Dst fetch
- monitor advances one model hour per new hourly bin (watchdog against free-run), with bounded history, log rotation, and per-target horizon-alarm dedup
- fixed-rate live-monitor scheduling subtracts cycle runtime and skips missed slots, preventing cumulative hourly issuance drift and catch-up bursts
- fixed 1, 2, 3, and 6 h product horizons, with health determined from the exact API cycle contract; partial or internally inconsistent rows remain failed even when all four calls returned
- hot-log retention rejects limits below four rows, so it cannot truncate a complete product cycle immediately after validation
- FIFO cold-archive durability and schema guard: rows dropped by hot-log retention are appended, under the same lock, to a cold archive resolved from the log's own directory (a non-default log never writes the production archive), with a sidecar manifest tracking cumulative rows, byte size, and per-segment SHA-256; the append aborts and leaves the rows in the hot log if the archive size drifted since the last manifest or if the pruned columns do not match the existing archive header, so a hot-log schema change can no longer append positionally misaligned rows
- cycle-level interval selection uses ACI only when both point and served residual streams are ready at the model-step lead corresponding to every 1, 2, 3, and 6 h target; otherwise the entire issuance uses the shared static conformal fallback
- per-key single-flight refreshes, bounded waits, and failure cooldowns for USGS dB/dt and station-network requests, preventing duplicate work or repeated requests to a failed service
- one shared NOAA/USGS execution slot for potentially blocking DNS, TLS, and HTTP work, leaving the other server threads available while cached or unavailable responses return immediately
- observed dB/dt nowcast fallback from FRD to CMO for the default dashboard path; explicit station requests remain exact and never fall back silently
- dB/dt artifact schema 3 records quasi-definitive ground provenance, bow-shock-shifted OMNI training drivers, the unshifted L1 live source, and a mandatory disabled-serving flag; offline values are empirical exceedance scores rather than calibrated probabilities
- the live dB/dt route fails closed at the unvalidated L1-to-bow-shock feature transfer: it serves the observed ground nowcast but withholds the retrospective 30-minute forecast instead of evaluating it on a different driver time reference

Dashboard and API:

- documented `--api-url=...` readiness-audit invocations normalize command-line slices to `String`, so strict live-API certification reaches the audit instead of failing dispatch before any checks run
- log-independent endpoints stay up when the forecast log is absent; internal errors no longer echo the log path
- input-staleness demotion and served-pipeline capability labels surfaced to the front end
- stale or unavailable status responses clear previously displayed metrics and WATCH state instead of leaving an obsolete forecast on screen
- the sub-hour display trajectory spans the full anchor-to-target lead, so under a lagging Kyoto Dst anchor (which places the h=6 target one model step beyond a fixed six-hour window) the served display line ends exactly at the furthest issued horizon instead of one hour short; hourly issuance, scoring, and API values are unchanged
- responsive WATCH placement in normal document flow so it cannot cover metric labels
- a bounded four-thread default across shell and Docker launch paths, leaving request capacity during the serialized upstream refresh while preserving the one-request public-data gate
- WCAG-AA contrast for muted labels on every dashboard background surface
- WATCH values are identified as the lower edge of a displayed 90% target interval, without treating the symmetric interval as a one-sided confidence bound, a guaranteed-coverage set, or a storm probability; value-equivalent legacy API keys remain for already-loaded clients
- ground dB/dt is presented as a GIC-hazard indicator, with the four values identified as unit-converted Pulkkinen et al. threshold magnitudes rather than a reproduction of their nonoverlapping-window protocol or named grid-risk tiers; unsupported generic geoelectric/GIC risk categories were removed
- browser notifications use the stronger of the point-forecast and interval-edge WATCH ranges; unsupported explicit dB/dt stations and malformed query encodings return HTTP 400 instead of aliasing to FRD or surfacing as server errors
- deployment parity across the committed Julia 1.12.6 app environment, Docker image, bundled models, and operational-evidence path; launchers now fail closed when Julia or dependency setup is unsuitable

Provenance and tests:

- discovery provenance sidecar, persisted served point fit, and a joint posterior-draws ensemble artifact; fill-fabricated storm rows excluded from discovery
- current and historical joint-draw artifacts are versioned with their matching
  SINDy cores; regeneration remains available through
  `validation/generate_ensemble_draws.jl`
- tracked real-data phase, coupled, cross-cycle, and reconstruction snapshots
  are synchronized to the verified revised-paper run; legacy synthetic
  snapshots and figures are regenerated with the identifiable 20-term library
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
