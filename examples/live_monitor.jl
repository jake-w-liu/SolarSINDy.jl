#!/usr/bin/env julia
# Package-native long-running locked-live Dst forecast monitor.
#
# Each cycle, against the canonical locked-live log:
#   1. issue immutable V2 forecasts at horizons [1,2,3,6] h
#      (locked before their target observations exist; duplicate pending targets
#      are reused by the live verification layer),
#   2. refresh observations from the shared Dst snapshot (this also verifies
#      pending targets; the narrower verifier is retained as an error fallback),
#   3. capture and score a prospective external Dst snapshot,
#   4. rewrite the locked-live comparison report.
# Then sleep and repeat. As the long accrual daemon it never exits on pending==0,
# so locked rows keep accumulating over restarts. Per-step try/catch keeps a
# transient feed/network failure from killing the daemon.
#
# Follows the locked-live verification workflow (docs/src/live-verification.md).
#
# The output/state directory (log, report, calibration, snapshots) is
# parameterized so a fresh clone can run against a scratch directory while the
# package-native service uses `var/monitor` by default.
#
# Env:
#   SOLARSINDY_MONITOR_DIR     output/state directory (default <package>/var/monitor)
#   SOLARSINDY_V2_CALIBRATION  V2 calibration CSV (default <dir>/operational_v2_calibration.csv;
#                              falls back to the package-bundled calibration when absent)
#   SOLARSINDY_MONITOR_ONCE=1  run exactly one cycle, then exit (also --once)
#   LIVE_MONITOR_INTERVAL_SEC  seconds between cycles (default 3600)
#   LIVE_MONITOR_MAX_CYCLES    stop after N cycles (default 0 = run forever; testing)
#   LIVE_MONITOR_DEADMAN_CYCLES consecutive incomplete cycles before the issuance dead-man trips
#   LIVE_MONITOR_MAX_LOG_ROWS   maximum hot-log rows; must hold a full cycle (default 50000)

include(joinpath(@__DIR__, "live_forecast_verify.jl"))
include(joinpath(@__DIR__, "..", "app", "src", "forecast_api.jl"))
include(joinpath(@__DIR__, "external_dst_snapshot_collector.jl"))

using CSV
using DataFrames
using Dates

const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const MONITOR_DIR = get(ENV, "SOLARSINDY_MONITOR_DIR",
                        joinpath(PACKAGE_ROOT, "var", "monitor"))
const LOG = joinpath(MONITOR_DIR, "live_forecast_log.csv")
const REPORT = joinpath(MONITOR_DIR, "live_comparison_report.md")
const OUTAGE_SENTINEL = joinpath(MONITOR_DIR, "OUTAGE.md")   # persistent alert artifact the dashboard can serve

# Package-bundled locked calibration + conformal sidecar (small model metadata), used as the
# graceful fallback when the output directory has no operational calibration (fresh clone).
const BUNDLED_V2_CALIB = normpath(joinpath(@__DIR__, "..", "deploy", "operational_v2_calibration.csv"))

# Resolve the V2 calibration. Prefer the directory-local locked calibration (the deployed live
# location), then the package-bundled copy. The conformal interval sidecar is derived from this
# path by the engine (_conformal_path), so both live and bundled cases stay consistent.
function _resolve_v2_calibration()
    explicit = get(ENV, "SOLARSINDY_V2_CALIBRATION", joinpath(MONITOR_DIR, "operational_v2_calibration.csv"))
    isfile(explicit) && return explicit
    if isfile(BUNDLED_V2_CALIB)
        @warn "V2 calibration not found in monitor directory; using bundled package calibration" requested=explicit bundled=BUNDLED_V2_CALIB
        return BUNDLED_V2_CALIB
    end
    return explicit   # let the engine raise its own clear "run --fit-v2-calibration first" error
end
const V2_CALIB = _resolve_v2_calibration()

const INTERVAL = parse(Int, get(ENV, "LIVE_MONITOR_INTERVAL_SEC", "3600"))
const RUN_ONCE = get(ENV, "SOLARSINDY_MONITOR_ONCE", "0") == "1" || ("--once" in ARGS)
const MAX_CYCLES = RUN_ONCE ? 1 : parse(Int, get(ENV, "LIVE_MONITOR_MAX_CYCLES", "0"))
const HORIZONS = LIVE_CYCLE_HORIZONS  # one shared monitor/API product contract
# Consecutive incomplete horizon cycles before the issuance dead-man trips. Uses the
# package-level feed_deadman_tripped predicate (realtime.jl) so the escalation threshold
# is shared and unit-tested.
const ISSUE_DEADMAN_THRESHOLD = parse(Int, get(ENV, "LIVE_MONITOR_DEADMAN_CYCLES", string(DEFAULT_FEED_DEADMAN_THRESHOLD)))
const MAX_LOG_ROWS = parse(Int, get(ENV, "LIVE_MONITOR_MAX_LOG_ROWS", "50000"))

INTERVAL >= 1 || error("LIVE_MONITOR_INTERVAL_SEC must be at least 1")
MAX_CYCLES >= 0 || error("LIVE_MONITOR_MAX_CYCLES must be nonnegative")
ISSUE_DEADMAN_THRESHOLD >= 1 || error("LIVE_MONITOR_DEADMAN_CYCLES must be at least 1")
MAX_LOG_ROWS >= length(HORIZONS) || error(
    "LIVE_MONITOR_MAX_LOG_ROWS must be at least $(length(HORIZONS)) " *
    "to retain one complete $(join(HORIZONS, '/')) h product cycle",
)

# External Dst snapshot collector config pinned to the monitor directory. repo_root keeps the
# stored raw_path column relative to the directory's parent, matching the deployed layout.
const EXTERNAL_DST_CFG = ExternalDstCollectorConfig(;
    log_path = joinpath(MONITOR_DIR, "external_dst_forecast_log.csv"),
    report_path = joinpath(MONITOR_DIR, "external_dst_forecast_report.md"),
    raw_dir = joinpath(MONITOR_DIR, "source_cache", "external_dst_snapshots"),
    repo_root = normpath(joinpath(MONITOR_DIR, "..")),
)

stamp() = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
logln(args...) = (println("MONITOR ", stamp(), "  ", args...); flush(stdout))

# Run one body step, reporting but never propagating failures. The issuance path counts these
# call results for diagnostics, then validates the completed log cycle independently.
function guarded(label, f)
    try
        f()
        return true
    catch e
        e isa InterruptException && rethrow()
        logln("WARN ", label, " failed: ", sprint(showerror, e))
        return false
    end
end

# Age [h] of the newest issued forecast row vs wall clock, or nothing when the log is
# absent/empty/unparseable. A self-check the cycle prints every pass, so a frozen issuance
# path (e.g. a retired upstream feed) can no longer look healthy in the logs.
function newest_issuance_age_hours()
    isfile(LOG) || return nothing
    rows = try
        CSV.Rows(LOG; select=[:issue_time_utc], reusebuffer=true)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    latest = nothing
    for row in rows
        s = row.issue_time_utc
        ismissing(s) && continue
        str = String(string(s))
        t = tryparse(DateTime, str)
        t === nothing && (t = tryparse(DateTime, split(str, '.')[1]))
        t === nothing && continue
        (latest === nothing || t > latest) && (latest = t)
    end
    latest === nothing && return nothing
    return (now(UTC) - latest) / Millisecond(3_600_000)
end

function write_outage_sentinel(first_fail::AbstractString, consecutive::Int;
                               path::AbstractString=OUTAGE_SENTINEL)
    age = newest_issuance_age_hours()
    age_txt = age === nothing ? "unknown" : string(round(age; digits=1), " h")
    body = string(
        "# LIVE FORECAST ISSUANCE OUTAGE\n\n",
        "Detected UTC: ", stamp(), "\n",
        "First failed cycle UTC: ", first_fail, "\n",
        "Consecutive failed cycles: ", consecutive, "\n",
        "Newest issued forecast age: ", age_txt, "\n\n",
        "The live monitor did not complete every required forecast horizon for ", consecutive,
        " consecutive cycle(s). Cause: upstream feed fetch or issuance error ",
        "(check /api/health and /api/swpc, then confirm that the SWPC and Kyoto Dst ",
        "feeds are current). This file persists ",
        "until issuance recovers, at which point the monitor removes it.\n",
    )
    try
        mkpath(dirname(path))
        open(path, "w") do io; write(io, body); end
    catch e
        e isa InterruptException && rethrow()
        logln("WARN could not write outage sentinel: ", sprint(showerror, e))
    end
    return nothing
end

function _complete_issuance_cycle(log_path::AbstractString, issue_time::DateTime,
                                  interval_policy::Symbol=:auto)
    interval_policy = _checked_interval_policy(interval_policy)
    isfile(log_path) || return false
    df = _load_log(log_path)
    cycle = _latest_cycle_uncached(df)
    _valid_live_cycle(cycle) || return false
    issues = collect(skipmissing(cycle.issue_time_utc_dt))
    isempty(issues) && return false
    floor(maximum(issues), Hour) == floor(issue_time, Hour) || return false
    source = _common_cycle_field(cycle, :interval_source)
    interval_policy == :aci && return source == "aci"
    interval_policy == :static && return source != "aci"
    return true
end

clear_outage_sentinel() = (isfile(OUTAGE_SENTINEL) && rm(OUTAGE_SENTINEL; force=true); nothing)

# Advance a fixed-rate cycle deadline without catch-up bursts. Sleeping for a full interval after
# each completed cycle accumulates fetch/runtime latency and eventually skips an hourly product.
# This schedule stays anchored to the original cadence. If work spans several slots, fully elapsed
# slots are skipped and at most the latest deadline is served immediately, so the daemon never
# emits a multi-cycle catch-up burst.
function _advance_cycle_deadline(previous_deadline::Real, now_seconds::Real,
                                 interval_seconds::Real)
    all(isfinite, (previous_deadline, now_seconds, interval_seconds)) ||
        throw(ArgumentError("cycle scheduling inputs must be finite"))
    interval_seconds > 0 || throw(ArgumentError("cycle interval must be positive"))
    deadline = Float64(previous_deadline) + Float64(interval_seconds)
    skipped = 0
    if now_seconds > deadline
        skipped = floor(Int, (Float64(now_seconds) - deadline) / Float64(interval_seconds))
        deadline += skipped * Float64(interval_seconds)
    end
    return (deadline=deadline, skipped=skipped)
end

_cycle_clock_seconds() = time_ns() / 1.0e9

_monitor_aci_ready(log_path::AbstractString, model_steps::Integer,
                   latest_dst::Real, pred_col::Symbol) =
    _aci_interval_from_log(
        log_path, 0.0, model_steps; latest_dst=latest_dst, pred_col=pred_col,
    ) !== nothing

# Choose one interval policy before any horizon is written. Both the baseline-center and served-
# center residual streams must be mature for every required model-step lead before the batch may use
# ACI; otherwise every horizon uses its shared static/conformal fallback.
function _monitor_interval_policy(inputs;
                                  log_path::AbstractString=LOG,
                                  horizons=HORIZONS,
                                  readiness_fn::Function=_monitor_aci_ready)
    issue_time = inputs.issue_time
    dst_times, dst_vals = inputs.dst
    dst_idx = _latest_causal_index(dst_times, issue_time, "Kyoto Dst")
    latest_dst_time = dst_times[dst_idx]
    latest_dst = Float64(dst_vals[dst_idx])
    for horizon in horizons
        target = _next_hourly_target(issue_time, horizon, latest_dst_time)
        model_steps = Int((target - latest_dst_time) / Hour(1))
        for pred_col in (:v2_pred_dst_nt, :served_pred_dst_nt)
            readiness_fn(log_path, model_steps, latest_dst, pred_col) || return :static
        end
    end
    return :aci
end

# Bound the operational hot log under the same cross-process lock used by
# issuance and verification. Retention is FIFO by append order; rebuilding the
# sidecar clears order-dependent ACI checkpoints so the next query replays only
# the retained authoritative rows.
function _retain_live_forecast_log!(log_path::AbstractString, max_rows::Int)
    max_rows >= length(HORIZONS) || throw(ArgumentError(
        "max_rows must be at least $(length(HORIZONS)) to retain one complete " *
        "$(join(HORIZONS, '/')) h product cycle",
    ))
    isfile(log_path) || return 0
    path = String(log_path)
    return _with_forecast_log_lock(path) do
        _recover_append_transaction!(path)
        _live_require_regular_target(path)
        state = _valid_live_state(path)
        state !== nothing && Int(state["row_count"]) <= max_rows && return 0
        df = CSV.read(path, DataFrame)
        n = nrow(df)
        n <= max_rows && return 0
        previous_state = _valid_live_state(path)
        retained = df[(n - max_rows + 1):n, :]
        _atomic_csv(path, retained)
        _persist_live_state_after_table_write!(
            path, previous_state, retained, Int[]; revised=true,
        )
        return n - max_rows
    end
end

function _issue_horizon_cycle!(inputs;
                               issue_fn::Function=issue_forecast,
                               log_path::AbstractString=LOG,
                               calibration_path::AbstractString=V2_CALIB,
                               complete_fn::Function=_complete_issuance_cycle,
                               interval_policy::Symbol)
    interval_policy = _checked_interval_policy(interval_policy)
    interval_policy == :auto && throw(ArgumentError(
        "live monitor horizon batches require an explicit coherent interval policy",
    ))
    issued_ok = 0
    trajectory_horizon = maximum(HORIZONS)
    for h in HORIZONS
        issued_ok += guarded("issue h=$h", () -> begin
            issue_fn(LiveVerifyConfig(; mode=:issue, model=:v2, horizon_hours=h,
                                      log_path=String(log_path),
                                      v2_calibration_path=String(calibration_path));
                     inputs=inputs, write_trajectory=h == trajectory_horizon,
                     verbose=false, interval_policy=interval_policy)
            nothing
        end)
    end
    complete = guarded("validate issued cycle", () -> begin
        complete_fn(log_path, inputs.issue_time, interval_policy) || error(
            "latest log rows do not form one API-valid $(join(HORIZONS, '/')) h cycle",
        )
    end)
    return (succeeded=issued_ok, complete=complete)
end

# Run one cycle; returns both successful calls and whether the log contains one API-valid cycle.
function cycle!()
    base_cfg = LiveVerifyConfig(; mode=:issue, model=:v2, horizon_hours=first(HORIZONS),
                                log_path=LOG, v2_calibration_path=V2_CALIB)
    inputs = try
        prepare_issue_inputs(base_cfg)
    catch e
        e isa InterruptException && rethrow()
        logln("WARN prepare issuance inputs failed: ", sprint(showerror, e))
        return (succeeded=0, complete=false)
    end
    interval_policy = try
        _monitor_interval_policy(inputs)
    catch e
        e isa InterruptException && rethrow()
        logln("WARN select coherent interval policy failed: ", sprint(showerror, e))
        return (succeeded=0, complete=false)
    end
    logln("forecast interval policy: ", interval_policy)
    issuance = _issue_horizon_cycle!(inputs; interval_policy=interval_policy)
    cfg = LiveVerifyConfig(; log_path=LOG, report_path=REPORT)
    dst_times, dst_vals = inputs.dst
    refreshed = guarded(
        "refresh_observations",
        () -> refresh_observations!(cfg; dst_times=dst_times, dst_vals=dst_vals),
    )
    refreshed || guarded(
        "verify_pending_fallback",
        () -> verify_pending!(cfg; dst_times=dst_times, dst_vals=dst_vals),
    )
    guarded("forecast_log_retention", () -> _retain_live_forecast_log!(LOG, MAX_LOG_ROWS))
    observations = DataFrame(
        observed_time_utc=DateTime.(dst_times),
        observed_dst_nt=Float64.(dst_vals),
    )
    guarded(
        "external_dst_snapshot",
        () -> capture_and_score_external_dst_snapshot!(
            EXTERNAL_DST_CFG; observations=observations,
        ),
    )
    guarded("comparison_report_and_summary", () -> begin
        df = CSV.read(LOG, DataFrame)
        write_live_comparison_report(cfg.log_path, cfg.report_path; df=df)
        pend = count(ismissing, df.observation_dst_nt)
        logln("log rows=", nrow(df), " pending=", pend)
    end)
    return issuance
end

function main()
    logln("start: dir=", MONITOR_DIR, " calibration=", V2_CALIB,
          " interval=", INTERVAL, "s horizons=", HORIZONS,
          " max_cycles=", MAX_CYCLES, " deadman_cycles=", ISSUE_DEADMAN_THRESHOLD,
          " max_log_rows=", MAX_LOG_ROWS)
    cycles = 0
    consecutive_failures = 0
    first_failure = ""
    cycle_deadline = _cycle_clock_seconds()
    while true
        cycles += 1
        logln("cycle ", cycles, " begin")
        issuance = cycle!()

        # Log-freshness self-check every cycle: the report can no longer read healthy during an
        # issuance gap because the age of the newest issued row is surfaced here and in the report.
        let age = newest_issuance_age_hours()
            age === nothing ? logln("newest issuance age: unknown (no log rows)") :
                logln("newest issuance age: ", round(age; digits=2), " h")
        end

        if !issuance.complete
            consecutive_failures += 1
            isempty(first_failure) && (first_failure = stamp())
            logln("WARN incomplete forecast cycle: ", issuance.succeeded, "/", length(HORIZONS),
                  " horizon calls succeeded but the latest rows were not one API-valid cycle ",
                  "(consecutive failed cycles=",
                  consecutive_failures, "/", ISSUE_DEADMAN_THRESHOLD, ")")
            if feed_deadman_tripped(consecutive_failures; threshold=ISSUE_DEADMAN_THRESHOLD)
                write_outage_sentinel(first_failure, consecutive_failures)
                logln("CRITICAL issuance dead-man tripped after ", consecutive_failures,
                      " consecutive failed cycle(s); wrote ", OUTAGE_SENTINEL,
                      " and exiting non-zero so the supervisor flags the outage")
                exit(1)
            end
        else
            if consecutive_failures > 0
                logln("issuance recovered after ", consecutive_failures, " failed cycle(s)")
            end
            # Recovery may occur after the supervisor restarted this process, in
            # which case the in-memory failure counter is zero but a sentinel from
            # the previous process still exists. Every successful issuance clears
            # persistent outage state.
            clear_outage_sentinel()
            consecutive_failures = 0
            first_failure = ""
        end

        logln("cycle ", cycles, " done")
        (0 < MAX_CYCLES <= cycles) && break
        schedule = _advance_cycle_deadline(
            cycle_deadline, _cycle_clock_seconds(), INTERVAL,
        )
        cycle_deadline = schedule.deadline
        schedule.skipped > 0 && logln(
            "WARN cycle runtime passed ", schedule.skipped,
            " fully elapsed scheduled slot(s); cadence remains fixed-rate",
        )
        remaining = cycle_deadline - _cycle_clock_seconds()
        remaining > 0 && sleep(remaining)
    end
    logln("stop after ", cycles, " cycle(s)")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
