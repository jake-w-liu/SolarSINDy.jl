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

# Cold-archive segment naming. Segment 0 is the base archive; a hot-log schema change that
# post-dates the archive's creation rolls forward to `<stem>.<n>.csv` with its own
# `<stem>.<n>.csv.manifest.json`. Retention is the only writer, and `_resolve_archive_segment` only
# ever appends to the highest-index segment, so segment order is append order and the full record is
# the concatenation of the segments in index order.
const FORECAST_ARCHIVE_MAX_SEGMENTS = 1000

function _archive_segment_paths(archive_path::AbstractString, manifest_path::AbstractString,
                                index::Integer)
    index == 0 && return (String(archive_path), String(manifest_path))
    stem, ext = splitext(String(archive_path))
    segment = string(stem, '.', Int(index), ext)
    return (segment, string(segment, ".manifest.json"))
end

# Pick the archive segment that can accept a frame with these `columns`. Only the HIGHEST-index
# non-empty segment is a candidate: if its header matches, rows are appended to it; if it differs,
# it is left byte-untouched and a new segment is opened one index further on. A header mismatch
# therefore never produces a misaligned append AND never blocks retention permanently.
#
# Restricting the candidate to the last segment is what makes the documented archive contract true.
# Searching from index 0 for the FIRST header match reuses an earlier segment whenever the hot-log
# schema returns to an older shape — a rollback, a reverted column, an operator running the previous
# release — and rows written later would then sit in a lower-index file than rows written earlier.
# Segment order would no longer be append order, and reading the archive as the concatenation of its
# segments in index order would silently interleave the record out of chronological sequence. The
# cost of the invariant is one extra segment file after a schema revert; the alternative is an
# archive whose ordering contract cannot be stated.
#
# Before the rollover a single column addition — routine in this package, where the `v2_2_*`, `v23_*`
# and `v24_*` column families were each added within weeks — made every subsequent
# `_retain_live_forecast_log!` throw "cold archive header mismatch". Rows were then never pruned
# again and the hot log grew past LIVE_MONITOR_MAX_LOG_ROWS without bound, which in turn makes every
# failed dashboard log parse a multi-second serialised stall.
function _resolve_archive_segment(archive_path::AbstractString, manifest_path::AbstractString,
                                  columns::Vector{String})
    # Scan the whole index range rather than stopping at the first hole: a manually removed middle
    # segment must not make a later, populated segment invisible and send appends backwards.
    last_index = -1
    for index in 0:FORECAST_ARCHIVE_MAX_SEGMENTS
        path, _ = _archive_segment_paths(archive_path, manifest_path, index)
        isfile(path) && filesize(path) > 0 && (last_index = index)
    end
    if last_index < 0
        path, manifest = _archive_segment_paths(archive_path, manifest_path, 0)
        return (path=path, manifest=manifest, index=0, existed=false)
    end

    path, manifest = _archive_segment_paths(archive_path, manifest_path, last_index)
    header = try
        names(CSV.read(path, DataFrame; limit=0))
    catch e
        e isa InterruptException && rethrow()
        error("cold archive header unreadable at $path: $(sprint(showerror, e))")
    end
    header == columns && return (path=path, manifest=manifest, index=last_index, existed=true)

    next_index = last_index + 1
    next_index <= FORECAST_ARCHIVE_MAX_SEGMENTS || error(
        "cold archive segment limit $FORECAST_ARCHIVE_MAX_SEGMENTS reached at $archive_path; " *
        "refusing to append rows whose columns match no segment header")
    next_path, next_manifest = _archive_segment_paths(archive_path, manifest_path, next_index)
    return (path=next_path, manifest=next_manifest, index=next_index, existed=false)
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
    # Schema-drift handling: CSV.write appends positionally and never re-reads the target header, so
    # a pruned frame whose columns differ from a segment's header must never be appended to it. The
    # resolver keeps that guarantee by appending only to the highest-index segment when its header
    # matches exactly, and opening the next index when it does not.
    segment = _resolve_archive_segment(archive_path, manifest_path, names(pruned))
    segment_path = segment.path
    segment_manifest = segment.manifest
    existed = segment.existed
    if segment.index > 0 && !existed
        logln("cold archive schema drift: hot-log columns match no existing segment header; ",
              "rolling to segment ", segment.index, " at ", segment_path,
              " (earlier segments left untouched)")
    end

    prev_rows = 0
    prev_bytes = 0
    if isfile(segment_manifest)
        m = try
            JSON3.read(read(segment_manifest, String))
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
    existed && prev_bytes != 0 && filesize(segment_path) != prev_bytes && error(
        "cold archive size changed outside retention: expected $prev_bytes bytes, " *
        "found $(filesize(segment_path)) at $segment_path")

    # Manifest-completeness guard: a non-empty archive with no readable manifest (missing, or corrupt
    # JSON swallowed to prev_bytes==0) has no verified byte baseline. Without it the post-append
    # accounting check below (new_bytes == base_bytes + length(seg), base_bytes==0) can only fire
    # AFTER CSV.write has already appended the segment, so every retention retry re-appends the same
    # rows (duplicating archived evidence) and throws again while retention never completes. Refuse
    # before serializing so retention aborts and the rows stay safely in the hot log until the
    # archive/manifest is repaired.
    existed && prev_bytes == 0 && filesize(segment_path) > 0 && error(
        "cold archive manifest missing or unreadable at $segment_manifest beside a non-empty " *
        "archive ($(filesize(segment_path)) bytes) at $segment_path; refusing to append without a " *
        "verified byte baseline")

    # Serialize the segment once (header only when creating), hash it, append, flush.
    buf = IOBuffer()
    CSV.write(buf, pruned; append=existed, header=!existed)
    seg = take!(buf)
    seg_sha = bytes2hex(sha256(seg))
    open(segment_path, "a") do io
        write(io, seg)
        flush(io)
    end
    base_bytes = existed ? prev_bytes : 0
    new_bytes = filesize(segment_path)
    new_bytes == base_bytes + length(seg) || error(
        "cold archive append incomplete: expected $(base_bytes + length(seg)) bytes, " *
        "found $new_bytes at $segment_path")

    total_rows = prev_rows + nrow(pruned)
    tmp = string(segment_manifest, ".tmp")
    open(tmp, "w") do io
        JSON3.write(io, (archived_rows = total_rows,
                         archive_bytes = new_bytes,
                         segment_index = segment.index,
                         last_segment_rows = nrow(pruned),
                         last_segment_sha256 = seg_sha,
                         updated_utc = stamp()))
    end
    mv(tmp, segment_manifest; force=true)
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
    # A V2.4e-served row publishes the study's split-conformal band under `V2_4_INTERVAL_SOURCE`
    # whatever the batch's fallback interval policy is (the aci/static policy governs the shifted
    # frozen-tail and adaptive bands of the fallback stages only). That source is therefore valid
    # under either policy, but only on rows the V2.4 stage actually served: a cycle carrying the
    # conformal source on rows whose V2.4 status is not `ok` is incoherent and stays incomplete.
    # Without this clause every V2.4e cycle failed the aci-policy check, the issuance dead-man
    # tripped after six cycles and the supervisor restarted the daemon (2026-08-17/18).
    if source == V2_4_INTERVAL_SOURCE
        hasproperty(cycle, :v24_status) || return false
        return all(status -> _v2_4_served_acted(status), cycle.v24_status)
    end
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

# Verified-residual counts behind one ACI readiness decision, read from the checkpoint the readiness
# probe has just written beside the log. Returns `nothing` when no checkpoint exists for the stream
# (e.g. a stubbed readiness function, or a log missing the required columns). Diagnostic only: it
# never influences the policy, so a read failure degrades to an unannotated stream name.
function _monitor_aci_stream_counts(log_path::AbstractString, model_steps::Integer,
                                    pred_col::Symbol)
    try
        isfile(log_path) || return nothing
        state = _read_live_state(String(log_path))
        state === nothing && return nothing
        streams = get(state, "aci_streams", nothing)
        streams isa AbstractDict || return nothing
        key = _aci_stream_key(pred_col, model_steps, _ACI_ACTIVITY_THRESHOLD_NT,
                              _ACI_TARGET_COVERAGE, _ACI_GAMMA, _ACI_WARMUP,
                              _ACI_HISTORY_WINDOW)
        entry = get(streams, key, nothing)
        entry isa AbstractDict || return nothing
        counts = Int[]
        for pool in ("all", "quiet", "disturbed")
            snapshot = get(entry, pool, nothing)
            snapshot isa AbstractDict || return nothing
            push!(counts, Int(snapshot["count"]))
        end
        return (all=counts[1], quiet=counts[2], disturbed=counts[3])
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
end

# Choose one interval policy before any horizon is written. Both the baseline-center and served-
# center residual streams must be mature for every required model-step lead before the batch may use
# ACI; otherwise every horizon uses its shared static/conformal fallback.
#
# Which model-step leads are "required" is set by the cadence phase: the Kyoto Dst anchor lags the
# issue hour by 0 or 1 h depending on the minute the daemon happens to run, so the same product
# horizons {1,2,3,6} map to model steps {1,2,3,6} at lag 0 and {2,3,4,7} at lag 1. The two step sets
# have separate residual streams with separate maturities, which is why a restart at a different
# minute of the hour can flip this policy — and with it the fallback-row and frozen-tail band widths
# by roughly a factor of four. That dependence was previously invisible: only the resulting `:static`
# or `:aci` symbol was recorded. The diagnostic below names the anchor lag, the required step set and
# every immature stream with its verified-residual counts, so the flip is attributable from the log
# alone. `immature`, when supplied, receives the same `(model_steps, pred_col)` pairs.
function _monitor_interval_policy(inputs;
                                  log_path::AbstractString=LOG,
                                  horizons=HORIZONS,
                                  readiness_fn::Function=_monitor_aci_ready,
                                  immature::Union{Nothing,AbstractVector}=nothing)
    issue_time = inputs.issue_time
    dst_times, dst_vals = inputs.dst
    dst_idx = _latest_causal_index(dst_times, issue_time, "Kyoto Dst")
    latest_dst_time = dst_times[dst_idx]
    latest_dst = Float64(dst_vals[dst_idx])
    required_steps = Int[]
    not_ready = Tuple{Int,Symbol}[]
    for horizon in horizons
        target = _next_hourly_target(issue_time, horizon, latest_dst_time)
        model_steps = Int((target - latest_dst_time) / Hour(1))
        push!(required_steps, model_steps)
        for pred_col in (:v2_pred_dst_nt, :served_pred_dst_nt)
            readiness_fn(log_path, model_steps, latest_dst, pred_col) ||
                push!(not_ready, (model_steps, pred_col))
        end
    end
    immature === nothing || append!(immature, not_ready)
    isempty(not_ready) && return :aci
    anchor_lag_hours = (floor(issue_time, Hour) - latest_dst_time) / Hour(1)
    detail = join(map(not_ready) do (model_steps, pred_col)
        counts = _monitor_aci_stream_counts(log_path, model_steps, pred_col)
        counts === nothing ? string(pred_col, "@ms=", model_steps) :
            string(pred_col, "@ms=", model_steps, " n=", counts.all, "(all)/",
                   counts.quiet, "(quiet)/", counts.disturbed, "(disturbed)")
    end, ", ")
    logln("interval policy :static — Kyoto anchor ", latest_dst_time,
          " lags issue hour ", floor(issue_time, Hour), " by ", anchor_lag_hours,
          " h, so this cadence phase needs model steps ", join(required_steps, "/"),
          "; immature ACI residual streams (need ", _ACI_WARMUP + _ACI_POOL_MARGIN,
          " verified rows): ", detail)
    return :static
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
#
# The issuance computation (solar-wind + Dst feeds, interval-policy preflight, four horizon
# appends) is one guarded block; every step after it is an OBSERVATION-side step that does not
# depend on the solar-wind feed and therefore runs on every cycle:
#   * Kyoto verification (`refresh_observations!` / `verify_pending!`) closes out targets whose
#     hour has already been published,
#   * hot-log retention keeps the row cap and the cold archive current,
#   * the prospective external Dst snapshot is an independent hourly scientific record — the
#     hour it misses is lost, it cannot be backfilled,
#   * the comparison report is the operator-facing summary.
# Before this split a solar-wind feed outage returned early and skipped all four (observed in
# production on 2026-07-29, cycles 15/16, ECONNRESET), so an L1 outage silently stalled Dst
# verification and destroyed external-snapshot hours. The returned issuance status is unchanged
# and still drives dead-man accounting only.
#
# Every collaborator is injectable so the skip-nothing contract is testable without the network.
function cycle!(; prepare_fn::Function=prepare_issue_inputs,
                  policy_fn::Function=_monitor_interval_policy,
                  issue_cycle_fn::Function=_issue_horizon_cycle!,
                  dst_fn::Function=_fetch_dst,
                  refresh_fn::Function=refresh_observations!,
                  verify_fn::Function=verify_pending!,
                  retention_fn::Function=_retain_live_forecast_log!,
                  snapshot_fn::Function=capture_and_score_external_dst_snapshot!,
                  report_fn::Function=write_live_comparison_report,
                  log_path::AbstractString=LOG,
                  report_path::AbstractString=REPORT,
                  calibration_path::AbstractString=V2_CALIB,
                  external_cfg=EXTERNAL_DST_CFG,
                  max_log_rows::Integer=MAX_LOG_ROWS)
    base_cfg = LiveVerifyConfig(; mode=:issue, model=:v2, horizon_hours=first(HORIZONS),
                                log_path=String(log_path),
                                v2_calibration_path=String(calibration_path))
    inputs = nothing
    try
        inputs = prepare_fn(base_cfg)
    catch e
        e isa InterruptException && rethrow()
        logln("WARN prepare issuance inputs failed: ", sprint(showerror, e))
    end
    interval_policy = nothing
    if inputs !== nothing
        try
            interval_policy = policy_fn(inputs)
        catch e
            e isa InterruptException && rethrow()
            logln("WARN select coherent interval policy failed: ", sprint(showerror, e))
        end
    end
    issuance = if interval_policy === nothing
        logln("WARN issuance skipped this cycle; verification, retention, the external Dst ",
              "snapshot and the comparison report still run")
        (succeeded=0, complete=false)
    else
        logln("forecast interval policy: ", interval_policy)
        issue_cycle_fn(inputs; log_path=String(log_path),
                       calibration_path=String(calibration_path),
                       interval_policy=interval_policy)
    end

    # Observation feed for the remaining steps. Reuse the issuance fetch when there is one;
    # otherwise fetch Kyoto Dst once for all three consumers. If that fetch also fails, pass
    # `nothing` and let each step use its own built-in fetch fallback rather than skipping.
    dst = inputs === nothing ? nothing : inputs.dst
    if dst === nothing
        guarded("observation_dst_fetch", () -> begin
            dst = dst_fn()
            nothing
        end)
    end
    dst_times = dst === nothing ? nothing : first(dst)
    dst_vals = dst === nothing ? nothing : last(dst)

    cfg = LiveVerifyConfig(; log_path=String(log_path), report_path=String(report_path))
    refreshed = guarded(
        "refresh_observations",
        () -> refresh_fn(cfg; dst_times=dst_times, dst_vals=dst_vals),
    )
    refreshed || guarded(
        "verify_pending_fallback",
        () -> verify_fn(cfg; dst_times=dst_times, dst_vals=dst_vals),
    )
    guarded("forecast_log_retention",
            () -> retention_fn(String(log_path), Int(max_log_rows)))
    observations = dst === nothing ? nothing : DataFrame(
        observed_time_utc=DateTime.(dst_times),
        observed_dst_nt=Float64.(dst_vals),
    )
    guarded(
        "external_dst_snapshot",
        () -> snapshot_fn(external_cfg; observations=observations),
    )
    guarded("comparison_report_and_summary", () -> begin
        df = CSV.read(cfg.log_path, DataFrame)
        report_fn(
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
