# SolarSINDy.jl

A Julia package for data-driven geomagnetic storm (Dst) forecasting, with a live
operational monitor and web dashboard. It has two layers:

1. **Discovery** — sparse identification of nonlinear dynamics (SINDy) recovers a
   compact solar-wind–magnetosphere coupling equation for Dst (20 candidate terms,
   11 active), alongside classical baselines (Burton, Burton-full,
   O'Brien–McPherron).
2. **Operational forecasting (super-learner)** — a nonnegative super learner combines ten
   causal experts with weights fitted per model step, issue-time regime, and
   ring-current depth, plus depth-stratified split-conformal intervals. Served
   identity:

   ```text
   dst-superlearner+sindyfloor+conformal
   ```

## Requirements

- Julia **≥ 1.12.6** (`[compat].julia` in [Project.toml](Project.toml))
- Network access to NOAA SWPC for the live path only
- GNU Bash and POSIX utilities for shell launchers and shell/data validation
  paths, including `awk` for OMNI extraction. On Windows, install Git for Windows
  and run the shell commands below from Git Bash; WSL is not required. Use native
  Julia for the Julia API. Test helpers verify actual GNU Bash execution;
  `SOLARSINDY_TEST_BASH` can select an explicit Bash executable.

Git attributes preserve the committed source and fixture bytes independently
of `core.autocrlf`. Keep POSIX launchers in LF form; the explicitly declared
CRLF test entry point retains its existing representation. Checksum-pinned
artifacts must retain their exact original bytes.

## Quick start — the live system

```bash
git clone https://github.com/jake-w-liu/SolarSINDy.jl.git
cd SolarSINDy.jl

bin/solarsindy setup    # one-time: check Julia, instantiate both environments
bin/solarsindy start    # forecast daemon + dashboard; opens http://127.0.0.1:8723
bin/solarsindy status   # process / health / issuance-freshness summary
bin/solarsindy stop     # stop both background processes
```

Other useful commands: `bin/solarsindy once` (single forecast cycle in the
foreground, the right smoke test), `bin/solarsindy open` (reopen the dashboard),
`bin/solarsindy logs monitor -f`, `bin/solarsindy install-service` /
`uninstall-service` (supervised mode: macOS launchd with auto-restart and
watchdog; systemd templates in [`deploy/`](deploy/)).

On Windows, CLI startup records the native PID, creation time, executable, physical
clone and actual Julia entrypoint in an exclusive nonce-bound receipt before the
daemon entrypoint runs. This is CLI startup attestation under user-controlled
instance state, rather than independently recovered operating-system arguments.
`status` returns **3** when identity is unknown; `start` and `stop` preserve the
state and refuse to spawn or signal. Older numeric pidfiles have an ambiguous PID
domain: reconcile them only after independently verifying the original process
and instance ownership; do not simply delete them to make a start succeed.
Windows `stop` validates the same native process handle used for forced termination
and waits for exit; it does not deliver POSIX SIGTERM. Supervised installation uses
macOS launchd or Linux systemd; Windows uses the ad-hoc CLI.

Both CLI starts fail before spawning if state-directory preparation or log rotation
fails. On Unix, a positively live start-lock owner is preserved even when its argv
uses a clone alias or relative path. This also preserves an unrelated/recycled live
PID in a start lock: reconcile ownership before changing that state, or wait for
confirmed exit. Daemon pidfiles retain their separate clone-entrypoint checks.
Unix lock acquisition also uses an atomic admission directory during stale-owner
recovery. Existing admission state is unknown and preserved; a crash there requires
verified reconciliation. This directory is not automatically reclaimed, and older
CLI versions that ignore it must not share the same instance state.

Unix starts reserve a regular PID file and verify its exact write before allowing
the intended main to execute. A bounded startup gate prevents a failed PID
publication from leaving an unrecorded main running. Symlink/nonregular PID state
is unknown and preserved. If activation or cleanup becomes uncertain after a
valid PID is committed, start fails and retains the record for verified inspection;
it does not report readiness. Identity uses native retained-descriptor `fstat`
metadata on Darwin and Bash `/dev/fd` comparison on Linux.

Unix numeric PID records accept only a positive decimal PID with an optional
single trailing newline, within 128 bytes. Malformed regular records retain the
legacy stale cleanup policy; read errors, symlink/nonregular paths and known
unsearchable directory ancestors preserve state as unknown. Missing directories
remain absent. These checks do not claim exhaustive filesystem-error diagnosis.

## The served product

**Experts.** The ten inputs are: the inertial-SINDy operator, the frozen-tail inertial-SINDy form,
an analog driver-continuation expert with refit ridge correction, persistence,
Burton, Burton-full, O'Brien–McPherron, a direct increment-GBM, climatology, and
the static stack regime stack. A 0.60 floor on the four SINDy-lineage experts
applies per cell; because the static stack is itself a six-expert mixture, that
constraint is an expert taxonomy, not a lower bound on sparse-equation
contribution.

**Fallback chain (fail-closed):** super-learner → static stack → inertial-SINDy operator.
Every issued row records which stage served it (`model_version`), the chosen
cell, the bundle digest, and the partner centers used for alerting.

**Alerting is separate from the point forecast.** The published severity is the
*deepest* of the served center, the static-stack center, and the inertial-SINDy center, so
a blended forecast can never downgrade a warning a predecessor would have
raised.

**Bundle.** Served artifacts live in `deploy/dst_superlearner/` (stack weights, boosted
models, analog archive, refit correction, climatology timescale, conformal
half-widths, guard record, SHA-256 manifest) and are rebuilt by
`validation/operational/superlearner_build_deploy.jl`. Every file is digest-verified on
load; a bundle that fails any check does not serve.

**Analog machinery.** The analog-continuation ensemble is not a served product of its
own — its confirmatory scoring returned `NO_GO` as a standalone forecast — but the
same kNN driver-continuation machinery is one of the ten experts the super-learner
combines, so its archive, feature and correction code ship inside the
`deploy/dst_superlearner/` bundle.

**Identity oracles** reproduce the served and static-stack centers offline:

```bash
julia --project=. validation/operational/superlearner_serving_identity.jl
julia --project=. validation/operational/static_stack_served_identity.jl
```

**Readiness audit** (while services run):

```bash
julia --project=. validation/operational/operational_readiness_audit.jl \
  --write-report --require-api --require-fresh
```

## Live monitor

The daemon issues immutable forecasts at 1/2/3/6 h (or the shifted step set when
the Dst anchor lags by one hour), scores them after the observation matures,
captures an external SWPC Dst snapshot as an independent comparator, and rewrites
the comparison report. State lives under `var/monitor/` by default.

Observation evidence retains the actual receipt time and raw response. If an
unchanged quicklook value was first received during its hour, the first
post-hour receipt adds mature evidence without rewriting the scored CSV row.
Later identical mature receipts do not add duplicate evidence.

Key environment variables (full list: `bin/solarsindy help`, template
[`deploy/solarsindy.env.example`](deploy/solarsindy.env.example)):

| Variable | Purpose |
|---|---|
| `SWM_HOST`, `SWM_PORT` | dashboard bind host/port |
| `SOLARSINDY_MONITOR_DIR` | monitor state directory |
| `SWM_WEBHOOK_URL` | POST alerts on threat-level changes |
| `SOLARSINDY_LOG` | dashboard log path override |
| `LIVE_MONITOR_INTERVAL_SEC`, `LIVE_MONITOR_MAX_CYCLES`, `LIVE_MONITOR_DEADMAN_CYCLES` | cadence, cycle cap, dead-man threshold |
| `LIVE_MONITOR_PHASE_SAMPLING=1` | alternate issue phase each hour (prospective calibration evidence) |
| `LIVE_MONITOR_MAX_LOG_ROWS` | hot-log cap (default 50,000; older rows cold-archive) |
| `SOLARSINDY_NO_OPEN=1` | never auto-open the browser |

The hot log is capped and rolls mature rows into an append-only cold archive;
daemon logs are ring-rotated. `var/monitor/source_cache/` (raw feed receipts)
grows unboundedly — prune it periodically.

## Dashboard

`app/` contains a self-contained HTTP.jl + Plotly UI over the locked-live log:
served forecast with its empirical 90% band, forecast-vs-observed track, verified
track record, calibration/skill, a ground-d*B*/d*t* nowcast, and the warning
chain. Coverage is labeled empirical; best-method highlighting waits for enough
matured rows. See [`app/README.md`](app/README.md) for endpoints.

## Using the library

```julia
using Pkg
Pkg.add(url="https://github.com/jake-w-liu/SolarSINDy.jl")
```

Discovery quick start:

```julia
using SolarSINDy

swd, _ = generate_synthetic_storm(seed=42)
data, dDst = prepare_sindy_data(swd, 1.0; smooth_window=5)
lib = build_minimal_library()

ξ, active_terms, _ = sindy_discover(data, lib, dDst; λ=0.01)
Dst_pred = simulate_sindy(ξ, lib, swd, 1.0)
```

Calibration example:

```julia
cal = fit_conformal(points, observations, horizons, latest_dsts; coverage=0.90)
lo, hi = conformal_interval(cal, point_forecast, horizon, latest_dst)

calop = fit_inertial_sindy_calibration(replay_df)          # β·z fit, causal
out = operational_inertial_sindy_predict(calop, core_point, core_ci05, core_ci95, features)
```

## Data

- Pre-computed SINDy coefficients and validation data ship in `data/` (tracked).
- The large NASA OMNI2 hourly archive is **not** committed; fetch and prepare it
  with `prepare_omni_data()` — downloads, parses, and cleans in one call.
- The historical 21/10 SINDy core is kept under `data/historical/sindy_core/` and
  `deploy/historical/sindy_core/` for provenance comparisons only; the `operational` API alias
  resolves to inertial-SINDy, so loading SINDy core requires an explicit version.

## Development

```bash
julia --project=SolarSINDy.jl SolarSINDy.jl/test/runtests.jl      # test suite
julia --project=SolarSINDy.jl/docs -e 'include("SolarSINDy.jl/docs/make.jl")'  # docs
```

- Tests use independent oracles (analytical, conservation, regression). A few
  oracles need generated artifacts under `validation/output/`; absent ones are
  registered as skipped, or as failures with
  `SOLARSINDY_REQUIRE_LOCAL_ARTIFACTS=1`.
- Research scripts live under `validation/` (`real_data_discovery.jl`,
  `generate_ensemble_draws.jl`, `run_validation.jl`, …). The package app has its
  own environment (`app/Project.toml`) and tests (`app/test/runtests.jl`).
- `Manifest.toml` is committed for reproducible runs.

## Repository map

```text
src/            package source (discovery, calibration, serving, monitor glue)
app/            dashboard + alerting server
bin/solarsindy  service CLI (setup/start/stop/status/logs/install-service)
deploy/         served bundles, launchd + systemd templates, env template
validation/     research and operational-validation scripts
data/           tracked coefficients, catalogs, ensemble draws
examples/       runnable examples (live monitor, verification, review)
docs/           Documenter sources (built by CI with warnonly=false)
var/monitor/    live state (gitignored): logs, forecast log, cold archive
```

## Notes

- The realtime path depends on NOAA SWPC availability; missing feeds fail closed.
- EKF-on-SINDy remains research infrastructure — it failed promotion gates and
  is not part of serving.
- Citation: [CITATION.cff](CITATION.cff).
# Initial-reference selection adapter

The optional [source-bound selection importer](tools/initial_reference_import/README.md)
consumes separately pinned independent-audit and export records. It preserves
the distinct four-hour and six-hour selection cohorts. This software interface
does not establish actual-data experiment qualification or predictive gains.



The maintained [prospective CME collector](tools/cme_prospective/README.md)
provides source-bound acquisition, fixed-cadence recovery controls and operational
status for a separately reviewed frozen producer and store.
