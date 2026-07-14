#!/usr/bin/env julia
# Package-native long-running locked-live Dst forecast monitor.
#
# Each cycle, against the canonical locked-live log:
#   1. issue immutable V2 forecasts at horizons [1,2,3,6] h
#      (locked before their target observations exist; duplicate pending targets
#      are reused by the live verification layer),
#   2. refresh observations from the live Dst feed (reconciles revisions),
#   3. verify any pending rows whose target observation has now arrived,
#   4. capture and score a prospective external Dst snapshot,
#   5. rewrite the locked-live comparison report.
# Then sleep and repeat. As the long accrual daemon it never exits on pending==0,
# so locked rows keep accumulating over restarts. Per-step try/catch keeps a
# transient feed/network failure from killing the daemon.
#
# Follows the locked-live verification workflow (docs/src/live-verification.md).
#
# Output/state directory (log, report, calibration, snapshots) is parameterized so
# a fresh clone can run against a scratch directory while the deployed shim keeps
# writing the existing live locations unchanged.
#
# Env:
#   SOLARSINDY_MONITOR_DIR     output/state directory (default "live_forecasts",
#                              resolved against the working directory)
#   SOLARSINDY_V2_CALIBRATION  V2 calibration CSV (default <dir>/operational_v2_calibration.csv;
#                              falls back to the package-bundled calibration when absent)
#   SOLARSINDY_MONITOR_ONCE=1  run exactly one cycle, then exit (also --once)
#   LIVE_MONITOR_INTERVAL_SEC  seconds between cycles (default 3600)
#   LIVE_MONITOR_MAX_CYCLES    stop after N cycles (default 0 = run forever; testing)
#   LIVE_MONITOR_HORIZONS      comma list of horizons (default "1,2,3,6")
#   LIVE_MONITOR_DEADMAN_CYCLES consecutive all-failed cycles before the issuance dead-man trips
#   LIVE_MONITOR_MAX_LOG_ROWS   maximum rows retained in the hot forecast log (default 50000)

include(joinpath(@__DIR__, "live_forecast_verify.jl"))
include(joinpath(@__DIR__, "external_dst_snapshot_collector.jl"))

using CSV
using DataFrames
using Dates

const MONITOR_DIR = get(ENV, "SOLARSINDY_MONITOR_DIR", "live_forecasts")
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
const HORIZONS = unique(parse.(Int, split(get(ENV, "LIVE_MONITOR_HORIZONS", "1,2,3,6"), ",")))
# Consecutive all-horizon-failed cycles before the issuance dead-man trips. Uses the
# package-level feed_deadman_tripped predicate (realtime.jl) so the escalation threshold
# is shared and unit-tested.
const ISSUE_DEADMAN_THRESHOLD = parse(Int, get(ENV, "LIVE_MONITOR_DEADMAN_CYCLES", string(DEFAULT_FEED_DEADMAN_THRESHOLD)))
const MAX_LOG_ROWS = parse(Int, get(ENV, "LIVE_MONITOR_MAX_LOG_ROWS", "50000"))

INTERVAL >= 1 || error("LIVE_MONITOR_INTERVAL_SEC must be at least 1")
MAX_CYCLES >= 0 || error("LIVE_MONITOR_MAX_CYCLES must be nonnegative")
!isempty(HORIZONS) && all(>(0), HORIZONS) ||
    error("LIVE_MONITOR_HORIZONS must contain positive integers")
ISSUE_DEADMAN_THRESHOLD >= 1 || error("LIVE_MONITOR_DEADMAN_CYCLES must be at least 1")
MAX_LOG_ROWS >= 1 || error("LIVE_MONITOR_MAX_LOG_ROWS must be at least 1")

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

# Run one body step, reporting but never propagating failures. Returns true on success so the
# cycle can distinguish a fully-failed issuance (dead-man input) from a healthy one.
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
    df = try
        CSV.read(LOG, DataFrame)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    ("issue_time_utc" in names(df)) || return nothing
    latest = nothing
    for s in df.issue_time_utc
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

function write_outage_sentinel(first_fail::AbstractString, consecutive::Int)
    age = newest_issuance_age_hours()
    age_txt = age === nothing ? "unknown" : string(round(age; digits=1), " h")
    body = string(
        "# LIVE FORECAST ISSUANCE OUTAGE\n\n",
        "Detected UTC: ", stamp(), "\n",
        "First failed cycle UTC: ", first_fail, "\n",
        "Consecutive failed cycles: ", consecutive, "\n",
        "Newest issued forecast age: ", age_txt, "\n\n",
        "The live monitor issued no forecast for ", consecutive,
        " consecutive cycle(s). Cause: upstream feed fetch or issuance error ",
        "(see the WARN issue lines in the monitor stdout log). This file persists ",
        "until issuance recovers, at which point the monitor removes it.\n",
    )
    try
        mkpath(dirname(OUTAGE_SENTINEL))
        open(OUTAGE_SENTINEL, "w") do io; write(io, body); end
    catch e
        e isa InterruptException && rethrow()
        logln("WARN could not write outage sentinel: ", sprint(showerror, e))
    end
    return nothing
end

clear_outage_sentinel() = (isfile(OUTAGE_SENTINEL) && rm(OUTAGE_SENTINEL; force=true); nothing)

# Bound the operational hot log under the same cross-process lock used by
# issuance and verification. Retention is FIFO by append order; rebuilding the
# sidecar clears order-dependent ACI checkpoints so the next query replays only
# the retained authoritative rows.
function _retain_live_forecast_log!(log_path::AbstractString, max_rows::Int)
    max_rows >= 1 || throw(ArgumentError("max_rows must be at least 1"))
    isfile(log_path) || return 0
    path = String(log_path)
    return _with_forecast_log_lock(path) do
        _recover_append_transaction!(path)
        _live_require_regular_target(path)
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

# Run one cycle; returns the number of horizons that issued successfully.
function cycle!()
    issued_ok = 0
    for h in HORIZONS
        guarded("issue h=$h", () ->
            issue_forecast(LiveVerifyConfig(; mode=:issue, model=:v2, horizon_hours=h,
                                            log_path=LOG, v2_calibration_path=V2_CALIB))) &&
            (issued_ok += 1)
    end
    cfg = LiveVerifyConfig(; log_path=LOG, report_path=REPORT)
    guarded("refresh_observations", () -> refresh_observations!(cfg))
    guarded("verify_pending", () -> verify_pending!(cfg))
    guarded("forecast_log_retention", () -> _retain_live_forecast_log!(LOG, MAX_LOG_ROWS))
    guarded("external_dst_snapshot", () -> capture_and_score_external_dst_snapshot!(EXTERNAL_DST_CFG))
    guarded("comparison_report", () -> write_live_comparison_report(cfg.log_path, cfg.report_path))
    guarded("summary", () -> begin
        df = CSV.read(LOG, DataFrame)
        pend = count(ismissing, df.observation_dst_nt)
        logln("log rows=", nrow(df), " pending=", pend)
    end)
    return issued_ok
end

function main()
    logln("start: dir=", MONITOR_DIR, " calibration=", V2_CALIB,
          " interval=", INTERVAL, "s horizons=", HORIZONS,
          " max_cycles=", MAX_CYCLES, " deadman_cycles=", ISSUE_DEADMAN_THRESHOLD,
          " max_log_rows=", MAX_LOG_ROWS)
    cycles = 0
    consecutive_failures = 0
    first_failure = ""
    while true
        cycles += 1
        logln("cycle ", cycles, " begin")
        issued_ok = cycle!()

        # Log-freshness self-check every cycle: the report can no longer read healthy during an
        # issuance gap because the age of the newest issued row is surfaced here and in the report.
        let age = newest_issuance_age_hours()
            age === nothing ? logln("newest issuance age: unknown (no log rows)") :
                logln("newest issuance age: ", round(age; digits=2), " h")
        end

        if issued_ok == 0
            consecutive_failures += 1
            isempty(first_failure) && (first_failure = stamp())
            logln("WARN no forecast issued this cycle (consecutive failed cycles=",
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
        sleep(INTERVAL)
    end
    logln("stop after ", cycles, " cycle(s)")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
