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
# Persistent outage artifact. Written by the daemon when its issuance dead-man trips AND by the
# out-of-process watchdog when the log goes stale (process death or unload), and served in-band by
# the dashboard API (/api/health, /api/alerts) so an outage is visible without the daemon alive.
const OUTAGE_SENTINEL = joinpath(MONITOR_DIR, "OUTAGE.md")

# ---- bounded, self-rotating operational diagnostics --------------------------------------
# launchd does not rotate StandardOutPath/StandardErrorPath, so the daemon owns its own bounded
# diagnostic record. Every logln line is mirrored into monitor.log, which rotates by size into a
# fixed ring (monitor.log, monitor.log.1, …), capping total diagnostic disk use. The launchd
# console streams point at launchd.out/launchd.err (see the plist) and are rotated once per
# (re)start by _rotate_launchd_stream! so out-of-band crash output (precompile errors, backtraces)
# is retained but bounded.
const LOG_DIR = joinpath(MONITOR_DIR, "logs")
const DIAG_LOG = joinpath(LOG_DIR, "monitor.log")
const DIAG_LOG_MAX_BYTES = parse(Int, get(ENV, "LIVE_MONITOR_LOG_MAX_BYTES", string(5 * 1024 * 1024)))
const DIAG_LOG_MAX_FILES = parse(Int, get(ENV, "LIVE_MONITOR_LOG_MAX_FILES", "5"))
const LAUNCHD_OUT = joinpath(LOG_DIR, "launchd.out")
const LAUNCHD_ERR = joinpath(LOG_DIR, "launchd.err")

DIAG_LOG_MAX_BYTES >= 4096 || error("LIVE_MONITOR_LOG_MAX_BYTES must be at least 4096")
DIAG_LOG_MAX_FILES >= 1 || error("LIVE_MONITOR_LOG_MAX_FILES must be at least 1")

# Size-bounded rotating file. Keeps `path` plus up to (max_files-1) archives `path.1 … path.N-1`;
# the oldest is discarded, so total disk use is bounded by ~max_bytes * max_files. Called before an
# append that could cross the cap.
function _rotate_ring!(path::AbstractString, max_bytes::Integer, max_files::Integer)
    (isfile(path) && filesize(path) >= max_bytes) || return nothing
    max_files <= 1 && (rm(path; force=true); return nothing)
    for i in (max_files - 1):-1:2
        src = string(path, '.', i - 1)
        isfile(src) && mv(src, string(path, '.', i); force=true)
    end
    mv(path, string(path, ".1"); force=true)
    return nothing
end

# Append one already-formatted line to the bounded diagnostic ring. Never throws: a diagnostics
# failure must not take down the daemon (the console stream still carries the same line).
function _diag_append(line::AbstractString)
    try
        isdir(LOG_DIR) || mkpath(LOG_DIR)
        _rotate_ring!(DIAG_LOG, DIAG_LOG_MAX_BYTES, DIAG_LOG_MAX_FILES)
        open(DIAG_LOG, "a") do io
            println(io, line)
        end
    catch e
        e isa InterruptException && rethrow()
    end
    return nothing
end

# Rotate a launchd stdout/stderr capture file once at startup. launchd opens these with O_APPEND,
# so truncating the inode in place is safe (the next append lands at offset 0, never a sparse hole).
# The just-ended generation's output is preserved in a single `.1` copy before truncation, so both
# files stay bounded across the KeepAlive restart cycle.
function _rotate_launchd_stream!(path::AbstractString)
    (isfile(path) && filesize(path) > 0) || return nothing
    try
        cp(path, string(path, ".1"); force=true)
        open(io -> truncate(io, 0), path, "r+")
    catch e
        e isa InterruptException && rethrow()
    end
    return nothing
end

# Cold archive of locked-live rows that FIFO retention is about to discard. Append-only CSV with a
# sidecar manifest tracking cumulative archived rows, archive byte size, and per-segment sha256, so
# the scientific record survives the hot-log row cap and integrity is checkable in O(segment).
const ARCHIVE_DIR = joinpath(MONITOR_DIR, "archive")
const FORECAST_ARCHIVE = joinpath(ARCHIVE_DIR, "live_forecast_log_archive.csv")
const FORECAST_ARCHIVE_MANIFEST = string(FORECAST_ARCHIVE, ".manifest.json")

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
function logln(args...)
    line = string("MONITOR ", stamp(), "  ", args...)
    println(stdout, line)
    flush(stdout)
    _diag_append(line)
    return nothing
end

# Append rows about to be dropped by FIFO retention to the cold archive BEFORE they leave the hot
# log, so the locked-live record is never destroyed by the row cap. Runs inside the same forecast-log
# lock as retention, so the archive and the hot-log truncation commit together. Returns the count of
# rows archived; throws on an integrity mismatch so the caller aborts the truncation and the rows
# stay safely in the hot log for the next attempt.
function _archive_pruned_rows!(pruned::DataFrame;
                               archive_path::AbstractString=FORECAST_ARCHIVE,
                               manifest_path::AbstractString=FORECAST_ARCHIVE_MANIFEST)
    nrow(pruned) == 0 && return 0
    # Normalize to absolute paths at entry so a bare relative log path (dirname("live.csv")=="")
    # cannot silently land the archive/manifest under a cwd-relative "archive/" directory.
    archive_path = abspath(archive_path)
    manifest_path = abspath(manifest_path)
    mkpath(dirname(archive_path))
    existed = isfile(archive_path)

    prev_rows = 0
    prev_bytes = 0
    if isfile(manifest_path)
        m = try
            JSON3.read(read(manifest_path, String))
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        if m !== nothing
            prev_rows = Int(get(m, :archived_rows, 0))
            prev_bytes = Int(get(m, :archive_bytes, 0))
        end
    end
    # Detect external truncation/corruption since the last append before we extend the archive.
    existed && prev_bytes != 0 && filesize(archive_path) != prev_bytes && error(
        "cold archive size changed outside retention: expected $prev_bytes bytes, " *
        "found $(filesize(archive_path)) at $archive_path")

    # Manifest-completeness guard: a non-empty archive with no readable manifest (missing, or corrupt
    # JSON swallowed to prev_bytes==0) has no verified byte baseline. Without it the post-append
    # accounting check below (new_bytes == base_bytes + length(seg), base_bytes==0) can only fire
    # AFTER CSV.write has already appended the segment, so every retention retry re-appends the same
    # rows (duplicating archived evidence) and throws again while retention never completes. Refuse
    # before serializing (like the schema-drift guard) so retention aborts and the rows stay safely in
    # the hot log until the archive/manifest is repaired.
    existed && prev_bytes == 0 && filesize(archive_path) > 0 && error(
        "cold archive manifest missing or unreadable at $manifest_path beside a non-empty archive " *
        "($(filesize(archive_path)) bytes) at $archive_path; refusing to append without a verified " *
        "byte baseline")

    # Schema-drift guard: CSV.write appends positionally and never re-reads the target header, so a
    # pruned frame whose columns differ from the existing archive header would silently write
    # misaligned rows under the wrong header. Refuse before touching the file so retention aborts and
    # the rows stay safely in the hot log for the next attempt.
    if existed
        archive_header = try
            names(CSV.read(archive_path, DataFrame; limit=0))
        catch e
            e isa InterruptException && rethrow()
            error("cold archive header unreadable at $archive_path: $(sprint(showerror, e))")
        end
        names(pruned) == archive_header || error(
            "cold archive header mismatch at $archive_path: archive columns $archive_header " *
            "!= pruned columns $(names(pruned)); refusing to append misaligned rows")
    end

    # Serialize the segment once (header only when creating), hash it, append, flush.
    buf = IOBuffer()
    CSV.write(buf, pruned; append=existed, header=!existed)
    seg = take!(buf)
    seg_sha = bytes2hex(sha256(seg))
    open(archive_path, "a") do io
        write(io, seg)
        flush(io)
    end
    base_bytes = existed ? prev_bytes : 0
    new_bytes = filesize(archive_path)
    new_bytes == base_bytes + length(seg) || error(
        "cold archive append incomplete: expected $(base_bytes + length(seg)) bytes, " *
        "found $new_bytes at $archive_path")

    total_rows = prev_rows + nrow(pruned)
    tmp = string(manifest_path, ".tmp")
    open(tmp, "w") do io
        JSON3.write(io, (archived_rows = total_rows,
                         archive_bytes = new_bytes,
                         last_segment_rows = nrow(pruned),
                         last_segment_sha256 = seg_sha,
                         updated_utc = stamp()))
    end
    mv(tmp, manifest_path; force=true)
    return nrow(pruned)
end

# Run one body step, reporting but never propagating failures. The issuance path counts these
# call results for diagnostics, then validates the completed log cycle independently.
# Every step logs its wall-clock duration: a 2026-07-30 audit found a 3.43 h issuance stall
# whose cause could not be attributed because the log had no per-step timings — with these
# lines, any future stall names the step that consumed the time. (All network fetches in the
# cycle are already bounded by connect_timeout=15/readtimeout=30 at their three call sites.)
function guarded(label, f)
    t0 = time()
    try
        f()
        logln("step ", label, " ok in ", round(time() - t0; digits=1), " s")
        return true
    catch e
        e isa InterruptException && rethrow()
        logln("WARN ", label, " failed after ", round(time() - t0; digits=1), " s: ",
              sprint(showerror, e))
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
        "Source: daemon issuance dead-man\n",
        "Detected UTC: ", stamp(), "\n",
        "Summary: issuance incomplete for ", consecutive,
        " consecutive cycle(s); newest forecast age ", age_txt, "\n",
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

function clear_outage_sentinel(path::AbstractString=OUTAGE_SENTINEL)
    isfile(path) || return false
    body = try
        read(path, String)
    catch e
        e isa InterruptException && rethrow()
        return false
    end
    occursin("Source: daemon issuance dead-man", body) || return false
    rm(path; force=true)
    return true
end

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
        # Durability: cold-archive the oldest rows about to be dropped BEFORE the hot log is
        # rewritten, so the locked-live record survives the FIFO cap. A failed/short archive throws
        # here (retention aborts, rows stay in the hot log) rather than silently destroying evidence.
        # The archive lives beside the log it protects (a non-default log path archives to its own
        # sibling directory, never the module-const production archive).
        archive_path = joinpath(dirname(path), "archive", "live_forecast_log_archive.csv")
        _archive_pruned_rows!(df[1:(n - max_rows), :];
                              archive_path=archive_path,
                              manifest_path=string(archive_path, ".manifest.json"))
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
        write_live_comparison_report(
            cfg.log_path,
            cfg.report_path;
            df=df,
            empty_identity=:v2_1,
        )
        pend = count(ismissing, df.observation_dst_nt)
        logln("log rows=", nrow(df), " pending=", pend)
    end)
    return issuance
end

function main_live_monitor()
    # Bounded diagnostics must be ready before the first line: ensure the log directory exists and
    # rotate the launchd console-capture files for this (re)start so out-of-band crash output from the
    # previous generation is preserved once and both streams stay bounded.
    isdir(LOG_DIR) || mkpath(LOG_DIR)
    _rotate_launchd_stream!(LAUNCHD_OUT)
    _rotate_launchd_stream!(LAUNCHD_ERR)
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

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_live_monitor()
