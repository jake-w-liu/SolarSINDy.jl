# Changelog

All notable changes to `SolarSINDy.jl` will be documented in this file.

## [Unreleased] - 2026-08-17 (served-stack switch; version bump deferred)

Served point center moves from the V2.1 operator to the fitted static V2.2 regime
stack; the V2.3 analog driver continuation is integrated as a shadow forecast after
its confirmatory scoring returned `NO_GO`.

Served product:

- the served point center is now the complete V2.1 operator followed by the fitted
  static V2.2 regime stack over the six point components (served V2.1, frozen V2.1,
  persistence, Burton, Burton-full, O'Brien-McPherron), selected per model step and
  per causal issue-time regime; the served identity is
  `v2.2+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia+staticstack(sindy60_fit407598)`
- the stack weights ship as `deploy/operational_v2_2_stack.csv` with their SHA-256 and
  fit label in `deploy/operational_v2_2_stack_manifest.csv`; the engine refuses weights
  whose digest or label does not match the published fit, serves the V2.1 center instead
  and labels the row with the V2.1 identity, so a degradation is disclosed rather than
  silent
- the regime is derived from issue-time state only, and the coupling input is the gated
  proxy (`v22_serving_coupling_active`): rectified southward coupling counts only while
  the wind drives and the ring current deepens, matching the archived definition the
  stack was fitted under
- every issued row keeps `v2_1_served_pred_dst_nt`, the center the V2.1 operator
  produced, alongside the frozen-tail `improved_*` columns
- the published threat level is taken against the deeper of the stacked and V2.1 centers,
  so a stack that blends toward persistence cannot lower a warning while a deeper stacked
  center still escalates one
- `validation/operational/v2_2_served_identity.jl` reproduces the archived
  `static_v2_2_dst_nt` column through the serving function on every scorable DEV/TEST row

V2.3 analog driver continuation (shadow only):

- the confirmatory single-shot scoring of the preregistered candidate
  (`T1r_T1_magnetic_K25_Soff`) returned `NO_GO` on gates A1 and A2, so the candidate is
  not served; it is computed on the live information set and logged as a shadow forecast
  under `v2.3-shadow+sindy20x11+L1A+ADC(magnetic,K25)+T1rcal+LAT+E`
- new log columns `v23_shadow_model_version`, `v23_status`, `v23_analog_k`,
  `v23_raw_dst_nt`, `v23_center_dst_nt`, `v23_shadow_pred_dst_nt` and
  `v23_e_layer_applied` record the center and why it was or was not available
- `src/operational_v23_serving.jl` is the single implementation of the analog center:
  the 18-feature issue-time key, the K-nearest archive retrieval, the per-member frozen-core
  rollout, the analog-core refit of the V2.1 ridge correction, the lead-aware blend against
  a recomputed frozen-tail center, and the capped per-step error layer
- `deploy/v2_3_shadow/` is built by `validation/operational/v2_3_build_deploy.jl --from-test`
  and verified on load: every file against its SHA-256, and the 86,968-origin analog archive
  rebuilt from the shipped hourly frame against the origin count and feature standardisation
  the scoring run recorded
- `validation/operational/v2_3_serving_identity.jl` reproduces the scored `V2_3_final`
  centers through the serving functions at every model step

Compatibility:

- the dashboard API, the live-cycle validity check and the readiness audit accept both the
  stacked served identity and the V2.1 identity a disclosed fallback row carries
- log-schema readers are unaffected: the new columns are appended and every existing column
  keeps its meaning

Serving-path corrections found by a post-integration audit of the same release:

- the watch flag and its tier are now taken on the depth-safe center rather than on the
  served band as issued. The band is shifted onto the served center, so a stack that
  reported a shallower storm than the V2.1 operator also moved the band up and could lower
  the outbound alert level on identical physics; the interval lower edge is now lowered by
  exactly the amount the point was lowered
- `v22_serving_depth_safe_center` has one definition, in the dependency-free
  `src/serving_depth_safe.jl`, which the dashboard application includes rather than
  restating; the container image ships that file beside the application sources
- the V2.3 error layer can now engage live. The layer's innovation history is defined at a
  one-hour model step, and production issues wall horizons 1/2/3/6 h at a one-hour anchor
  lag, so no logged row ever carried a one-hour step and the layer was permanently the
  identity. Every cycle now records `v23_step1_center_dst_nt`, the one-hour pre-layer
  center of its anchor, and the innovation is `Dst(anchor + 1 h)` minus that center, taken
  from the observed Kyoto series; the shared rule
  `v23_serving_innovations_from_step1_centers` is used by the engine and by the identity
  oracle, which checks the live rule against the scored history
- a shadow row whose fitted layer could not act because the history is incomplete records
  `v23_status = "ok:e_layer_pending"`; the `ok` prefix keeps it available while the
  disclosure stays explicit
- the readiness audit now fails closed on served- and shadow-stage health. It loads
  `deploy/operational_v2_2_stack.csv` under its pinned digest and label, verifies
  `deploy/v2_3_shadow/manifest.csv`, and measures the served fallback rate over a trailing
  window of issue cycles. Shadow availability and the fraction of cycles that applied an
  error layer are reported, and the served identity, shadow identity and fallback rate are
  exposed by `/api/health`
- `v2_readiness_audit.jl --self-test` exited 1 because its fixture payload kept an older
  driver-assumption sentence; it is fixed and now runs inside the package test suite
- `v2_1_issue_identity.jl` records the served identity, the stack label and digest, the
  shadow identity and the shadow manifest digest; the audit requires the exact served
  identity there and compares the API's served label with the newest logged cycle
- an empty `SOLARSINDY_V2_2_STACK_SHA256` is refused instead of silently disabling the
  digest pin. A staged run can accept it with `SOLARSINDY_ALLOW_UNPINNED_STACK=1`, in which
  case the row carries a separate `...+unpinned` identity that neither the dashboard nor the
  readiness audit accepts as the published product
- the dashboard and API derive the product name from the served label instead of naming
  V2.1, caption the 15-minute line as the V2.1 core trajectory shown for display, and
  report verified rows per served label (`by_served_model`,
  `n_verified_current_served_model`) so a record earned by the previous pipeline is not
  presented as the current product's
- `/api/forecast` exposes `severity_dst_nt`, `severity_ci05_dst_nt` and
  `v2_1_served_pred_dst_nt` per horizon; `lead_time.driver_assumption` comes from the served
  row, so a fallback cycle no longer describes a stage that did not run
- a cycle whose rows carry different accepted served labels is accepted and reported under
  the weakest label, instead of blanking the dashboard and suppressing its alerts
- new log columns `v2_2_status` (per-row served-stage status), `v23_manifest_sha256`
  (shadow deployment manifest digest) and `v23_history_hours` (hourly L1 depth the analog
  key drew on)
- error-layer artifacts named by `e_layers.json` must appear in the manifest's
  digest-verified set, so a manifest with their digest rows removed is a load error rather
  than an unverified load

Deployment-boundary corrections, from an audit of the first day of the same release:

- `/api/health` no longer drops its whole served block while the trailing window straddles
  the shadow-schema change. Cycles issued before the shadow columns existed carry `missing`
  in them, `missing == 1` is three-valued, and the health summary therefore raised and was
  reported as no served identity at all during the first day of the deployment it exists to
  report on
- a cycle whose stack stage healed or failed between its horizons now publishes the driver
  assumption of the stage it is reported under, taken from the rows carrying its weakest
  served label. It previously reported the assumption as never recorded, which describes a
  logging failure rather than the disclosed per-row degradation the log recorded, and the
  readiness audit failed the payload for it
- the served-stage fallback window counts only cycles issued by a build that carries the
  stack stage, and discloses how many older cycles it excluded. Cycles that predate the
  stage are not fallbacks of a stage that did not exist, and counting them reported a
  deployment onto an existing log as a near-total served-stage failure. The window spans
  four days, because a one-day window cannot resolve the one-percent target at all: one
  fallback out of twenty-four cycles is already 4.2 percent. A fallback on the newest cycle
  fails, and an over-target rate fails once two or more cycles in the window fell back; one
  isolated older fallback is reported and passes. The shadow window follows the same
  staged-cycle rule
- the readiness audit has one definition of the newest cycle, keyed on the issue hour as the
  dashboard API and the stage windows are. The vintage-keyed reading merged every issue that
  shared a stalled L1 vintage into one cycle, so the served label compared against the
  published payload could belong to a cycle the API never served
- the newest cycle's served label is re-read with the rest of the dashboard comparison
  snapshot after the API request, so a cycle boundary falling between the two is not
  reported as a mislabelled product
- the identity audit accepts the documented empty shadow manifest digest, which an absent
  shadow deployment records and a CSV field reads back as missing
- the shadow one-hour center's per-cycle cache key carries the issue-anchor drivers and the
  memory features, both of which are recomputed from the L1 stream at every issuance

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
