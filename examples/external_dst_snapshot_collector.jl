#!/usr/bin/env julia
# Prospective external Dst forecast snapshot collector.
#
# Captures public same-unit Dst forecast/nowcast products as issue-time snapshots, stores raw
# response hashes, and scores rows later against the SWPC-served Kyoto Dst product. Driven by the
# live monitor, and runnable standalone.
#
# Output locations are parameterized so the collector writes into the monitor directory. The
# default is the package runtime directory (`var/monitor`).
#
# Env:
#   SOLARSINDY_EXTERNAL_DST_DIR  output directory for the external Dst log/report/raw snapshots
#                                (default SOLARSINDY_MONITOR_DIR, else <package>/var/monitor)
#   SOLARSINDY_EXTERNAL_DST_MAX_ROWS
#                                maximum retained forecast rows (default 50000)
#   SOLARSINDY_EXTERNAL_DST_MAX_RAW
#                                maximum retained raw response snapshots (default 10000)

using CSV
using DataFrames
using Dates
using HTTP
using JSON3
using Printf
using SHA
using Statistics
using FileWatching: Pidfile

const EXTERNAL_DST_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_EXTERNAL_DST_DIR = joinpath(EXTERNAL_DST_PACKAGE_ROOT, "var", "monitor")
const EXTERNAL_DST_DIR = get(ENV, "SOLARSINDY_EXTERNAL_DST_DIR",
                             get(ENV, "SOLARSINDY_MONITOR_DIR", DEFAULT_EXTERNAL_DST_DIR))
const EXTERNAL_DST_LOG = joinpath(EXTERNAL_DST_DIR, "external_dst_forecast_log.csv")
const EXTERNAL_DST_REPORT = joinpath(EXTERNAL_DST_DIR, "external_dst_forecast_report.md")
const EXTERNAL_DST_RAW_DIR = joinpath(EXTERNAL_DST_DIR, "source_cache", "external_dst_snapshots")
const EXTERNAL_DST_REPO_ROOT = normpath(joinpath(EXTERNAL_DST_DIR, ".."))
const EXTERNAL_DST_MAX_OBS_GAP_MIN = 30.0
const EXTERNAL_DST_MAX_LOG_ROWS = 50_000
const EXTERNAL_DST_MAX_RAW_SNAPSHOTS = 10_000
const EXTERNAL_DST_RAW_LOCK_NAME = ".external_dst_raw_store.lock"
const EXTERNAL_DST_RAW_OWNER_NAME = ".external_dst_raw_store.owner"
const EXTERNAL_DST_RAW_OWNER_MAGIC = "SolarSINDy external Dst raw-store owner v1"
const EXTERNAL_DST_LOG_OWNER_SUFFIX = ".external_dst_raw_store.owner"
const EXTERNAL_DST_LOG_OWNER_MAGIC = "SolarSINDy external Dst log owner v1"
const EXTERNAL_DST_SOURCES = [
    (; name = "swpc_geospace_dst_1_hour",
       url = "https://services.swpc.noaa.gov/json/geospace/geospace_dst_1_hour.json",
       kind = "swpc_geospace_json"),
    (; name = "temerin_li_dst_last_96h",
       url = "https://lasp.colorado.edu/space_weather/dsttemerin/dst_last_96_hrs.txt",
       kind = "temerin_li_ascii",
       run_url = "https://lasp.colorado.edu/space_weather/dsttemerin/dsttemerin.html"),
]
const EXTERNAL_DST_OBS_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"
const EXTERNAL_DST_SCHEMA = [
    :source, :issue_utc, :fetched_utc, :target_utc, :lead_h, :forecast_dst_nt,
    :forecast_cadence_min, :issue_basis, :source_url, :raw_sha256, :raw_path,
    :source_run_utc, :source_last_modified_utc, :source_max_target_utc, :row_role,
    :observed_dst_nt, :observed_time_utc, :observed_gap_min, :abs_error_nt, :scored_utc,
]

Base.@kwdef struct ExternalDstCollectorConfig
    log_path::String = EXTERNAL_DST_LOG
    report_path::String = EXTERNAL_DST_REPORT
    raw_dir::String = EXTERNAL_DST_RAW_DIR
    repo_root::String = EXTERNAL_DST_REPO_ROOT
    sources::Vector = EXTERNAL_DST_SOURCES
    obs_url::String = EXTERNAL_DST_OBS_URL
    max_obs_gap_min::Float64 = EXTERNAL_DST_MAX_OBS_GAP_MIN
    max_log_rows::Int = parse(Int, get(ENV, "SOLARSINDY_EXTERNAL_DST_MAX_ROWS",
                                       string(EXTERNAL_DST_MAX_LOG_ROWS)))
    max_raw_snapshots::Int = parse(Int, get(ENV, "SOLARSINDY_EXTERNAL_DST_MAX_RAW",
                                            string(EXTERNAL_DST_MAX_RAW_SNAPSHOTS)))
end

function _external_empty_log()
    return DataFrame(
        source = String[],
        issue_utc = String[],
        fetched_utc = String[],
        target_utc = String[],
        lead_h = Float64[],
        forecast_dst_nt = Float64[],
        forecast_cadence_min = Float64[],
        issue_basis = String[],
        source_url = String[],
        raw_sha256 = String[],
        raw_path = String[],
        source_run_utc = Union{Missing, String}[],
        source_last_modified_utc = Union{Missing, String}[],
        source_max_target_utc = Union{Missing, String}[],
        row_role = String[],
        observed_dst_nt = Union{Missing, Float64}[],
        observed_time_utc = Union{Missing, String}[],
        observed_gap_min = Union{Missing, Float64}[],
        abs_error_nt = Union{Missing, Float64}[],
        scored_utc = Union{Missing, String}[],
    )
end

_fmt_utc(t::DateTime) = Dates.format(t, dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
_slug(s::AbstractString) = replace(lowercase(String(s)), r"[^a-z0-9]+" => "_")

function _parse_external_time(x)
    (x === missing || x === nothing) && return missing
    s = strip(String(string(x)))
    isempty(s) && return missing
    s = replace(s, " " => "T")
    s = replace(s, r"Z$" => "")
    s = split(s, '.')[1]
    for fmt in (dateformat"yyyy-mm-ddTHH:MM:SS", dateformat"yyyy/mm/dd-HH:MM:SS")
        t = tryparse(DateTime, s, fmt)
        t !== nothing && return t
    end
    return missing
end

function _parse_http_last_modified(headers)
    month_map = Dict("Jan" => 1, "Feb" => 2, "Mar" => 3, "Apr" => 4,
                     "May" => 5, "Jun" => 6, "Jul" => 7, "Aug" => 8,
                     "Sep" => 9, "Oct" => 10, "Nov" => 11, "Dec" => 12)
    for (k, v) in headers
        lowercase(String(k)) == "last-modified" || continue
        raw = String(v)
        m = match(r"^\w{3},\s+(\d{1,2})\s+(\w{3})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\s+GMT$", raw)
        if m !== nothing && haskey(month_map, m.captures[2])
            return DateTime(parse(Int, m.captures[3]), month_map[m.captures[2]],
                            parse(Int, m.captures[1]), parse(Int, m.captures[4]),
                            parse(Int, m.captures[5]), parse(Int, m.captures[6]))
        end
    end
    return missing
end

_sha256_hex(text::AbstractString) = bytes2hex(sha256(codeunits(String(text))))

function _external_file_sha256_hex(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

function _external_durable_flush(io::IO)
    flush(io)
    rc = Sys.iswindows() ?
         ccall(:_commit, Cint, (Cint,), fd(io)) :
         ccall(:fsync, Cint, (Cint,), fd(io))
    systemerror("durable external-Dst flush", rc != 0)
    return nothing
end

function _external_sync_parent(path::AbstractString)
    Sys.iswindows() && return nothing
    parent = dirname(path)
    isempty(parent) && (parent = ".")
    open(parent, "r") do io
        rc = ccall(:fsync, Cint, (Cint,), fd(io))
        systemerror("durable external-Dst directory flush", rc != 0)
    end
    return nothing
end

function _external_require_regular_target(path::AbstractString)
    islink(path) && throw(ArgumentError(
        "external Dst output target must not be a symbolic link: $path",
    ))
    ispath(path) && !isfile(path) && throw(ArgumentError(
        "external Dst output target exists but is not a regular file: $path",
    ))
    return path
end

# Prepare a complete, durable same-directory replacement without changing the
# target.  File-set transactions can stage every member before the first rename.
function _external_stage_file(writer::Function, path::AbstractString)
    parent = dirname(path)
    isempty(parent) && (parent = ".")
    mkpath(parent)
    _external_require_regular_target(path)
    tmp, io = mktemp(parent; cleanup=false)
    try
        writer(io)
        _external_durable_flush(io)
        close(io)
    catch
        isopen(io) && close(io)
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    return tmp
end

# Same-directory temporary + fsync + rename: a failed writer leaves the old
# complete target in place, and readers never observe a partially-written file.
function _external_atomic_file(writer::Function, path::AbstractString)
    tmp = _external_stage_file(writer, path)
    try
        Base.Filesystem.rename(tmp, path)
        _external_sync_parent(path)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    return path
end

function _external_target_snapshot(path::AbstractString)
    _external_require_regular_target(path)
    return isfile(path) ? (exists=true, bytes=read(path)) :
                          (exists=false, bytes=UInt8[])
end

function _restore_external_target!(path::AbstractString, snapshot)
    if snapshot.exists
        _external_atomic_file(path) do io
            write(io, snapshot.bytes)
        end
    elseif isfile(path)
        _external_require_regular_target(path)
        rm(path)
        _external_sync_parent(path)
    end
    return nothing
end

function _external_canonical_destination(path::AbstractString)
    absolute = abspath(path)
    parent = dirname(absolute)
    mkpath(parent)
    return normpath(joinpath(realpath(parent), basename(absolute)))
end

function _external_targets_alias(first_path::AbstractString,
                                 second_path::AbstractString)
    first_destination = _external_canonical_destination(first_path)
    second_destination = _external_canonical_destination(second_path)
    first_destination == second_destination && return true
    # Fail closed for case-only basename distinctions in the same resolved
    # directory. They are aliases on common case-insensitive filesystems and
    # are unsuitable as separate operational outputs even when a filesystem
    # happens to permit both names.
    dirname(first_destination) == dirname(second_destination) &&
        lowercase(basename(first_destination)) ==
            lowercase(basename(second_destination)) && return true
    return ispath(first_path) && ispath(second_path) &&
           Base.Filesystem.samefile(first_path, second_path)
end

_external_atomic_text(path::AbstractString, text::AbstractString) =
    _external_atomic_file(path) do io
        write(io, text)
    end

_external_atomic_csv(path::AbstractString, table) =
    _external_atomic_file(path) do io
        CSV.write(io, table)
    end

function _with_external_dst_pidlock(f::Function, lock_path::AbstractString;
                                    resource::AbstractString,
                                    timeout_sec::Real=30.0,
                                    stale_after_sec::Real=900.0,
                                    poll_sec::Real=0.05)
    isfinite(timeout_sec) && timeout_sec >= 0 ||
        throw(ArgumentError("timeout_sec must be finite and nonnegative"))
    isfinite(stale_after_sec) && stale_after_sec >= 0 ||
        throw(ArgumentError("stale_after_sec must be finite and nonnegative"))
    isfinite(poll_sec) && poll_sec > 0 ||
        throw(ArgumentError("poll_sec must be finite and positive"))
    parent = dirname(lock_path)
    isempty(parent) || mkpath(parent)
    deadline = time() + Float64(timeout_sec)
    owner = false
    while owner === false
        if !isdir(lock_path) && !islink(lock_path)
            owner = Pidfile.trymkpidlock(
                lock_path; stale_age=Float64(stale_after_sec),
                refresh=stale_after_sec == 0 ? 0.0 : Float64(stale_after_sec) / 2,
            )
        end
        owner === false || break
        time() < deadline || error(
            "timed out waiting for external Dst $resource lock: $lock_path",
        )
        sleep(min(Float64(poll_sec), max(deadline - time(), 0.0)))
    end
    try
        return f()
    finally
        close(owner)
    end
end

function _with_external_dst_lock(f::Function, log_path::AbstractString;
                                 timeout_sec::Real=30.0,
                                 stale_after_sec::Real=900.0,
                                 poll_sec::Real=0.05)
    return _with_external_dst_pidlock(
        f, string(log_path, ".lock"); resource="log", timeout_sec,
        stale_after_sec, poll_sec,
    )
end


function _external_raw_store_lock_path(raw_dir::AbstractString)
    mkpath(raw_dir)
    return joinpath(realpath(raw_dir), EXTERNAL_DST_RAW_LOCK_NAME)
end


function _external_canonical_raw_dir(raw_dir::AbstractString)
    mkpath(raw_dir)
    return normpath(realpath(raw_dir))
end


function _external_canonical_log_destination(log_path::AbstractString)
    destination = _external_canonical_destination(log_path)
    return ispath(destination) ? normpath(realpath(destination)) : destination
end


function _external_raw_store_owner_path(raw_dir::AbstractString)
    return joinpath(_external_canonical_raw_dir(raw_dir), EXTERNAL_DST_RAW_OWNER_NAME)
end


function _external_log_raw_store_owner_path(log_path::AbstractString)
    return string(_external_canonical_log_destination(log_path),
                  EXTERNAL_DST_LOG_OWNER_SUFFIX)
end


function _external_canonical_repo_root(repo_root::AbstractString)
    absolute = abspath(repo_root)
    isdir(absolute) || throw(ArgumentError(
        "external Dst repo_root must be an existing directory: $absolute",
    ))
    return normpath(realpath(absolute))
end


function _external_raw_store_owner_record(log_path::AbstractString,
                                          repo_root::AbstractString)
    canonical_log = _external_canonical_log_destination(log_path)
    canonical_root = _external_canonical_repo_root(repo_root)
    return string(
        EXTERNAL_DST_RAW_OWNER_MAGIC, '\n',
        "log_hex=", bytes2hex(codeunits(canonical_log)), '\n',
        "repo_root_hex=", bytes2hex(codeunits(canonical_root)), '\n',
    )
end


function _external_log_raw_store_owner_record(raw_dir::AbstractString,
                                              repo_root::AbstractString)
    canonical_raw = _external_canonical_raw_dir(raw_dir)
    canonical_root = _external_canonical_repo_root(repo_root)
    return string(
        EXTERNAL_DST_LOG_OWNER_MAGIC, '\n',
        "raw_dir_hex=", bytes2hex(codeunits(canonical_raw)), '\n',
        "repo_root_hex=", bytes2hex(codeunits(canonical_root)), '\n',
    )
end


function _external_owner_marker_exists(marker::AbstractString,
                                       record::AbstractString,
                                       identity::AbstractString)
    expected = Vector{UInt8}(codeunits(record))
    _external_require_regular_target(marker)
    isfile(marker) || return false
    filesize(marker) == length(expected) && read(marker) == expected ||
        throw(ArgumentError(
            "external Dst $identity marker is corrupt or belongs to a different " *
            "canonical storage identity: $marker",
        ))
    return true
end


function _install_external_owner_marker!(marker::AbstractString,
                                         record::AbstractString,
                                         identity::AbstractString)
    _external_atomic_text(marker, record)
    _external_owner_marker_exists(marker, record, identity) || error(
        "external Dst $identity marker failed verification: $marker",
    )
    return marker
end


function _bind_external_collector_store!(raw_dir::AbstractString,
                                         log_path::AbstractString,
                                         repo_root::AbstractString;
                                         has_provenance_rows::Bool=false)
    raw_marker = _external_raw_store_owner_path(raw_dir)
    log_marker = _external_log_raw_store_owner_path(log_path)
    !_external_targets_alias(raw_marker, log_marker) || throw(ArgumentError(
        "external Dst raw-store and log ownership markers must differ",
    ))
    raw_record = _external_raw_store_owner_record(log_path, repo_root)
    log_record = _external_log_raw_store_owner_record(raw_dir, repo_root)

    # Validate both identities before creating either marker. This prevents a conflict on one
    # side from leaving a new partial binding on the other side.
    raw_exists = _external_owner_marker_exists(
        raw_marker, raw_record, "raw-store ownership",
    )
    log_exists = _external_owner_marker_exists(
        log_marker, log_record, "log ownership",
    )
    if !raw_exists && !log_exists && !has_provenance_rows
        unowned_entries = filter(readdir(raw_dir)) do name
            name != EXTERNAL_DST_RAW_LOCK_NAME &&
                name != EXTERNAL_DST_RAW_OWNER_NAME
        end
        isempty(unowned_entries) || throw(ArgumentError(
            "external Dst unmarked raw store is nonempty but the log has no " *
            "provenance rows: $raw_dir",
        ))
    end
    created = Tuple{String, String}[]
    try
        if !raw_exists
            push!(created, (raw_marker, raw_record))
            _install_external_owner_marker!(
                raw_marker, raw_record, "raw-store ownership",
            )
        end
        if !log_exists
            push!(created, (log_marker, log_record))
            _install_external_owner_marker!(
                log_marker, log_record, "log ownership",
            )
        end
    catch failure
        rollback_errors = String[]
        for (marker, record) in Iterators.reverse(created)
            try
                expected = Vector{UInt8}(codeunits(record))
                if isfile(marker) && !islink(marker) &&
                   filesize(marker) == length(expected) && read(marker) == expected
                    rm(marker)
                    _external_sync_parent(marker)
                end
            catch rollback_error
                push!(rollback_errors, sprint(showerror, rollback_error))
            end
        end
        isempty(rollback_errors) || error(
            "external Dst storage binding failed ($(sprint(showerror, failure))) " *
            "and rollback was incomplete: $(join(rollback_errors, "; "))",
        )
        rethrow()
    end
    return (; raw_marker, log_marker)
end


function _external_canonical_raw_reference(raw_path::AbstractString,
                                           repo_root::AbstractString)
    canonical_root = _external_canonical_repo_root(repo_root)
    absolute = isabspath(raw_path) ? abspath(raw_path) :
               abspath(joinpath(canonical_root, raw_path))
    parent = dirname(absolute)
    canonical_parent = isdir(parent) ? normpath(realpath(parent)) : normpath(parent)
    return relpath(joinpath(canonical_parent, basename(absolute)), canonical_root)
end


function _canonicalize_external_raw_references!(df::DataFrame,
                                                repo_root::AbstractString)
    df.raw_path .= [
        _external_canonical_raw_reference(String(path), repo_root)
        for path in df.raw_path
    ]
    return df
end


function _validate_external_log_raw_store(df::DataFrame,
                                          raw_dir::AbstractString,
                                          repo_root::AbstractString)
    canonical_raw = _external_canonical_raw_dir(raw_dir)
    canonical_root = _external_canonical_repo_root(repo_root)
    for relative_path in unique(df.raw_path)
        destination = normpath(joinpath(canonical_root, String(relative_path)))
        dirname(destination) == canonical_raw || throw(ArgumentError(
            "external Dst log references a different canonical raw store: " *
            String(relative_path),
        ))
    end
    return true
end


function _with_external_dst_collector_locks(f::Function,
                                            cfg::ExternalDstCollectorConfig)
    raw_lock = _external_raw_store_lock_path(cfg.raw_dir)
    log_lock = string(_external_canonical_log_destination(cfg.log_path), ".lock")
    if _external_targets_alias(raw_lock, log_lock)
        return _with_external_dst_pidlock(
            f, raw_lock; resource="raw-store/log",
        )
    end
    return _with_external_dst_pidlock(raw_lock; resource="raw-store") do
        _with_external_dst_pidlock(f, log_lock; resource="log")
    end
end

function _http_text(url::AbstractString; http_get::Function = HTTP.get)
    resp = http_get(String(url); connect_timeout = 15, readtimeout = 30,
                    retries = 1, status_exception = true)
    body = String(getproperty(resp, :body))
    last_modified = _parse_http_last_modified(getproperty(resp, :headers))
    return body, last_modified
end

function _write_raw_snapshot(raw_dir::AbstractString, source::AbstractString,
                             fetched_utc::DateTime, sha::AbstractString,
                             body::AbstractString, repo_root::AbstractString)
    mkpath(raw_dir)
    path = _external_raw_snapshot_path(raw_dir, source, fetched_utc, sha)
    expected = String(sha)
    _sha256_hex(body) == expected || throw(ArgumentError(
        "raw snapshot SHA-256 does not match its response body",
    ))
    if isfile(path) && !islink(path)
        _external_file_sha256_hex(path) == expected && return relpath(path, repo_root)
    end
    _external_atomic_text(path, body)
    _external_file_sha256_hex(path) == expected || error(
        "raw snapshot failed post-write SHA-256 verification: $path",
    )
    return relpath(path, repo_root)
end

_external_raw_snapshot_path(raw_dir::AbstractString, source::AbstractString,
                            fetched_utc::DateTime, sha::AbstractString) =
    joinpath(raw_dir, string(Dates.format(fetched_utc, dateformat"yyyymmddTHHMMSS"),
                             "Z_", _slug(source), "_", first(String(sha), 12), ".raw"))

function _external_raw_retention_plan(raw_dir::AbstractString,
                                      repo_root::AbstractString,
                                      max_files::Int,
                                      referenced_paths::AbstractSet{<:AbstractString})
    max_files >= 1 || throw(ArgumentError("max_raw_snapshots must be at least 1"))
    isdir(raw_dir) || return (Set{String}(), String[])
    files = [joinpath(raw_dir, name) for name in readdir(raw_dir)
             if endswith(name, ".raw") &&
                isfile(joinpath(raw_dir, name)) && !islink(joinpath(raw_dir, name))]
    referenced = Set(normpath(String(path)) for path in referenced_paths)
    relative(path) = normpath(relpath(path, repo_root))
    referenced_files = [path for path in files if relative(path) in referenced]
    unreferenced_files = [path for path in files if !(relative(path) in referenced)]
    sort!(referenced_files; by=path -> (mtime(path), basename(path)))
    first_kept = max(1, length(referenced_files) - max_files + 1)
    kept = isempty(referenced_files) ? String[] : referenced_files[first_kept:end]
    rotated = first_kept == 1 ? String[] : referenced_files[1:(first_kept - 1)]
    removed = vcat(unreferenced_files, rotated)
    relative_kept = Set(normpath(relpath(path, repo_root)) for path in kept)
    return relative_kept, removed
end

function _verify_retained_external_raw_snapshots!(df::DataFrame,
                                                   repo_root::AbstractString)
    expected_by_path = Dict{String,String}()
    for row in eachrow(df)
        relative_path = normpath(String(row.raw_path))
        expected = String(row.raw_sha256)
        prior = get(expected_by_path, relative_path, expected)
        prior == expected || error(
            "retained raw snapshot has conflicting logged SHA-256 digests: $relative_path",
        )
        expected_by_path[relative_path] = expected
    end
    for (relative_path, expected) in expected_by_path
        path = normpath(joinpath(repo_root, relative_path))
        isfile(path) && !islink(path) || error(
            "retained raw snapshot is missing or not a regular file: $path",
        )
        actual = _external_file_sha256_hex(path)
        actual == expected || error(
            "retained raw snapshot SHA-256 mismatch: $path",
        )
    end
    return true
end

function _delete_external_raw_snapshots!(paths::AbstractVector{<:AbstractString})
    for path in paths
        rm(path)
    end
    return length(paths)
end

function _retain_external_rows(df::DataFrame, max_rows::Int)
    max_rows >= 1 || throw(ArgumentError("max_log_rows must be at least 1"))
    nrow(df) <= max_rows && return df
    return df[(nrow(df) - max_rows + 1):nrow(df), :]
end

function _valid_temerin_timestamp_parts(year::Int, day_of_year::Int,
                                         hour::Int, minute::Int, second::Int)
    return 1 <= year <= 9999 && 1 <= day_of_year <= daysinyear(year) &&
           0 <= hour <= 23 && 0 <= minute <= 59 && 0 <= second <= 59
end

function _parse_temerin_model_run(html::AbstractString)
    m = match(r"Time of model run:\s+(\d{4})/(\d{1,3})-(\d{2}):(\d{2}):(\d{2})", html)
    m === nothing && return missing
    yr = parse(Int, m.captures[1])
    doy = parse(Int, m.captures[2])
    hh = parse(Int, m.captures[3])
    mm = parse(Int, m.captures[4])
    ss = parse(Int, m.captures[5])
    _valid_temerin_timestamp_parts(yr, doy, hh, mm, ss) || return missing
    return DateTime(yr, 1, 1) + Day(doy - 1) + Hour(hh) + Minute(mm) + Second(ss)
end

function _parse_temerin_ascii(text::AbstractString)
    rows = DataFrame(target_utc = DateTime[], forecast_dst_nt = Float64[])
    pat = r"^\s*(\d{4})/(\d{1,3})-(\d{2}):(\d{2}):(\d{2})\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    for line in split(text, '\n')
        m = match(pat, line)
        m === nothing && continue
        yr = parse(Int, m.captures[1])
        doy = parse(Int, m.captures[2])
        hh = parse(Int, m.captures[3])
        mm = parse(Int, m.captures[4])
        ss = parse(Int, m.captures[5])
        val = parse(Float64, m.captures[6])
        _valid_temerin_timestamp_parts(yr, doy, hh, mm, ss) || continue
        push!(rows, (DateTime(yr, 1, 1) + Day(doy - 1) + Hour(hh) + Minute(mm) + Second(ss), val))
    end
    sort!(rows, :target_utc)
    return rows
end

function _parse_swpc_geospace_json(text::AbstractString)
    raw = JSON3.read(text)
    rows = DataFrame(target_utc = DateTime[], forecast_dst_nt = Float64[])
    for r in raw
        tt = get(r, :time_tag, get(r, "time_tag", nothing))
        dv = get(r, :dst, get(r, "dst", nothing))
        t = _parse_external_time(tt)
        t === missing && continue
        val = try
            Float64(dv)
        catch e
            e isa InterruptException && rethrow()
            NaN
        end
        isfinite(val) && push!(rows, (t, val))
    end
    sort!(rows, :target_utc)
    return rows
end

function _median_cadence_min(times::Vector{DateTime})
    length(times) < 2 && return NaN
    gaps = [Dates.value(times[i] - times[i - 1]) / 60000 for i in 2:length(times)]
    all(>(0), gaps) || return NaN
    return Float64(median(gaps))
end

function _staged_future_rows_for_source(source; fetched_utc::DateTime = now(UTC),
                                        http_get::Function = HTTP.get)
    body, last_modified = _http_text(source.url; http_get = http_get)
    sha = _sha256_hex(body)
    source_run = missing
    issue_basis = "http_last_modified"
    if source.kind == "temerin_li_ascii"
        run_html, _ = _http_text(source.run_url; http_get = http_get)
        source_run = _parse_temerin_model_run(run_html)
        source_run === missing || (issue_basis = "source_model_run")
        forecast = _parse_temerin_ascii(body)
    elseif source.kind == "swpc_geospace_json"
        forecast = _parse_swpc_geospace_json(body)
    else
        error("unknown external Dst source kind: $(source.kind)")
    end
    isempty(forecast) && error("$(source.name) produced no parseable Dst rows")
    issue = source_run !== missing ? source_run :
            last_modified !== missing ? last_modified : fetched_utc
    issue_basis = source_run !== missing ? issue_basis :
                  last_modified !== missing ? "http_last_modified" : "fetch_time"
    source_max = maximum(forecast.target_utc)
    cadence = _median_cadence_min(DateTime.(forecast.target_utc))
    isfinite(cadence) && cadence > 0 || error(
        "$(source.name) requires at least two distinct, strictly increasing " *
        "forecast timestamps to infer cadence",
    )

    out = _external_empty_log()
    for r in eachrow(forecast)
        target = DateTime(r.target_utc)
        target > issue || continue
        lead_h = Dates.value(target - issue) / 3_600_000
        push!(out, (
            String(source.name), _fmt_utc(issue), _fmt_utc(fetched_utc),
            _fmt_utc(target), Float64(lead_h), Float64(r.forecast_dst_nt),
            cadence, issue_basis, String(source.url), String(sha), "",
            source_run === missing ? missing : _fmt_utc(source_run),
            last_modified === missing ? missing : _fmt_utc(last_modified),
            _fmt_utc(source_max), "future_forecast",
            missing, missing, missing, missing, missing,
        ))
    end
    return (; rows=out, source=String(source.name), fetched_utc,
            sha=String(sha), body=String(body))
end

# Compatibility wrapper for callers that request one source directly. The long-running
# collector uses the staged helper so no raw file is installed before every fetch succeeds.
function _future_rows_for_source(source; fetched_utc::DateTime = now(UTC),
                                 http_get::Function = HTTP.get,
                                 raw_dir::AbstractString = EXTERNAL_DST_RAW_DIR,
                                 repo_root::AbstractString = EXTERNAL_DST_REPO_ROOT)
    staged = _staged_future_rows_for_source(
        source; fetched_utc=fetched_utc, http_get=http_get,
    )
    raw_path = _write_raw_snapshot(
        raw_dir, staged.source, staged.fetched_utc, staged.sha, staged.body, repo_root,
    )
    staged.rows.raw_path .= raw_path
    return staged.rows
end

function _load_external_log(path::AbstractString)
    isfile(path) || return _external_empty_log()
    df = CSV.read(path, DataFrame; missingstring = "")
    for col in EXTERNAL_DST_SCHEMA
        col in propertynames(df) || error("external Dst log missing column $col")
    end
    return df
end

function _dedupe_external_log(df::DataFrame)
    seen = Set{String}()
    keep = trues(nrow(df))
    for i in 1:nrow(df)
        issue = _parse_external_time(df.issue_utc[i])
        target = _parse_external_time(df.target_utc[i])
        issue_key = issue === missing ? string(df.issue_utc[i]) : _fmt_utc(issue)
        target_key = target === missing ? string(df.target_utc[i]) : _fmt_utc(target)
        key = join((string(df.source[i]), issue_key, target_key, string(df.raw_sha256[i])), "|")
        if key in seen
            keep[i] = false
        else
            push!(seen, key)
        end
    end
    return df[keep, :]
end

function _parse_observed_dst_json(text::AbstractString)
    raw = JSON3.read(text)
    rows = DataFrame(observed_time_utc = DateTime[], observed_dst_nt = Float64[])
    for r in raw
        tt = get(r, :time_tag, get(r, "time_tag", nothing))
        dv = get(r, :dst, get(r, "dst", nothing))
        t = _parse_external_time(tt)
        t === missing && continue
        val = try
            Float64(dv)
        catch e
            e isa InterruptException && rethrow()
            NaN
        end
        isfinite(val) && push!(rows, (t, val))
    end
    sort!(rows, :observed_time_utc)
    return rows
end

function _nearest_observation(obs::DataFrame, target::DateTime;
                              not_after::Union{Nothing,DateTime}=nothing)
    isempty(obs) && return nothing, Inf
    times = DateTime.(obs.observed_time_utc)
    idx = searchsortedfirst(times, target)
    best_i = 0
    best_gap = Inf
    for j in (idx - 1):(idx + 1)
        1 <= j <= length(times) || continue
        not_after !== nothing && times[j] > not_after && continue
        gap = abs(Dates.value(times[j] - target)) / 60000
        if gap < best_gap
            best_i = j
            best_gap = gap
        end
    end
    best_i == 0 && return nothing, Inf
    return best_i, best_gap
end

function score_external_dst_rows!(df::DataFrame, obs::DataFrame;
                                  max_obs_gap_min::Real = EXTERNAL_DST_MAX_OBS_GAP_MIN,
                                  scored_utc::DateTime = now(UTC))
    isfinite(max_obs_gap_min) && max_obs_gap_min >= 0 || throw(ArgumentError(
        "max_obs_gap_min must be finite and nonnegative",
    ))
    isempty(df) && return 0
    sort!(obs, :observed_time_utc)
    scored = 0
    for i in 1:nrow(df)
        ismissing(df.observed_dst_nt[i]) || continue
        target = _parse_external_time(df.target_utc[i])
        target === missing && continue
        target <= scored_utc || continue
        idx, gap = _nearest_observation(obs, target; not_after=scored_utc)
        idx === nothing && continue
        gap <= max_obs_gap_min || continue
        obs_val = Float64(obs.observed_dst_nt[idx])
        df.observed_dst_nt[i] = obs_val
        df.observed_time_utc[i] = _fmt_utc(DateTime(obs.observed_time_utc[idx]))
        df.observed_gap_min[i] = Float64(gap)
        df.abs_error_nt[i] = abs(Float64(df.forecast_dst_nt[i]) - obs_val)
        df.scored_utc[i] = _fmt_utc(scored_utc)
        scored += 1
    end
    return scored
end

function _fetch_observations(url::AbstractString; http_get::Function = HTTP.get)
    body, _ = _http_text(url; http_get = http_get)
    return _parse_observed_dst_json(body)
end

function _validate_external_dst_log(
    df::DataFrame;
    max_obs_gap_min::Real=EXTERNAL_DST_MAX_OBS_GAP_MIN,
)
    isfinite(max_obs_gap_min) && max_obs_gap_min >= 0 || throw(ArgumentError(
        "max_obs_gap_min must be finite and nonnegative",
    ))
    for col in EXTERNAL_DST_SCHEMA
        col in propertynames(df) || error("external Dst log missing column $col")
    end
    for i in 1:nrow(df)
        issue = _parse_external_time(df.issue_utc[i])
        fetched = _parse_external_time(df.fetched_utc[i])
        target = _parse_external_time(df.target_utc[i])
        issue === missing && error("external Dst row $i has unparsable issue_utc")
        fetched === missing && error("external Dst row $i has unparsable fetched_utc")
        target === missing && error("external Dst row $i has unparsable target_utc")
        target > issue || error("external Dst row $i is not a future forecast row")
        lead = Float64(df.lead_h[i])
        expected_lead = Dates.value(target - issue) / 3_600_000
        isfinite(lead) && lead > 0 &&
            isapprox(lead, expected_lead; rtol=32eps(Float64), atol=1e-9) ||
            error("external Dst row $i has invalid lead_h")
        isfinite(Float64(df.forecast_dst_nt[i])) ||
            error("external Dst row $i has invalid forecast_dst_nt")
        isfinite(Float64(df.forecast_cadence_min[i])) &&
            Float64(df.forecast_cadence_min[i]) > 0 ||
            error("external Dst row $i has invalid forecast_cadence_min")
        length(String(df.raw_sha256[i])) == 64 ||
            error("external Dst row $i has invalid raw_sha256")
        score_fields = (
            df.observed_dst_nt[i], df.observed_time_utc[i],
            df.observed_gap_min[i], df.abs_error_nt[i], df.scored_utc[i],
        )
        score_present = map(value -> !ismissing(value), score_fields)
        all(score_present) || !any(score_present) || error(
            "external Dst row $i has partially populated score fields",
        )
        if all(score_present)
            observed_time = _parse_external_time(df.observed_time_utc[i])
            scored_time = _parse_external_time(df.scored_utc[i])
            observed_time === missing && error(
                "external Dst row $i has invalid observed_time_utc",
            )
            scored_time === missing && error(
                "external Dst row $i has invalid scored_utc",
            )
            target <= scored_time && observed_time <= scored_time || error(
                "external Dst row $i was scored before its target or observation matured",
            )
            isfinite(Float64(df.observed_dst_nt[i])) ||
                error("external Dst row $i has invalid observed_dst_nt")
            isfinite(Float64(df.abs_error_nt[i])) ||
                error("external Dst row $i has invalid abs_error_nt")
            observed_gap = Float64(df.observed_gap_min[i])
            expected_gap = abs(Dates.value(observed_time - target)) / 60_000
            isfinite(observed_gap) && observed_gap >= 0 &&
                observed_gap <= max_obs_gap_min + 1e-9 &&
                isapprox(observed_gap, expected_gap; rtol=0.0, atol=1e-9) ||
                error("external Dst row $i has invalid observed_gap_min")
            observed_dst = Float64(df.observed_dst_nt[i])
            expected_error = abs(Float64(df.forecast_dst_nt[i]) - observed_dst)
            isapprox(Float64(df.abs_error_nt[i]), expected_error;
                     rtol=32eps(Float64), atol=1e-9) || error(
                "external Dst row $i has inconsistent abs_error_nt",
            )
        end
    end
    return true
end

function _external_stable_rmse(values::AbstractVector{<:Real})
    isempty(values) && return missing
    scale = maximum(abs, values)
    isfinite(scale) || throw(ArgumentError("external Dst residuals must be finite"))
    scale == 0 && return 0.0
    normalized = Float64.(values) ./ Float64(scale)
    result = Float64(scale) * sqrt(mean(abs2, normalized))
    isfinite(result) || throw(ArgumentError(
        "external Dst RMSE exceeds the supported Float64 range",
    ))
    return result
end

function external_dst_summary(df::DataFrame)
    out = DataFrame(source = String[], n_rows = Int[], n_scored = Int[],
                    n_issues = Int[], max_lead_h = Float64[],
                    rmse_nt = Union{Missing, Float64}[], mae_nt = Union{Missing, Float64}[])
    for source in sort(unique(String.(df.source)))
        sub = df[String.(df.source) .== source, :]
        scored_mask = .!ismissing.(sub.observed_dst_nt)
        if any(scored_mask)
            absolute_errors = abs.(
                Float64.(sub.forecast_dst_nt[scored_mask]) .-
                Float64.(sub.observed_dst_nt[scored_mask]),
            )
            all(isfinite, absolute_errors) || throw(ArgumentError(
                "external Dst residual exceeds the supported Float64 range",
            ))
            rmse_val = _external_stable_rmse(absolute_errors)
            mae_val = mean(absolute_errors)
        else
            rmse_val = missing
            mae_val = missing
        end
        push!(out, (source, nrow(sub), count(scored_mask), length(unique(String.(sub.issue_utc))),
                    maximum(Float64.(sub.lead_h)), rmse_val, mae_val))
    end
    return out
end

function _emit_external_dst_report(
    io::IO,
    df::DataFrame;
    max_obs_gap_min::Real=EXTERNAL_DST_MAX_OBS_GAP_MIN,
)
    isfinite(max_obs_gap_min) && max_obs_gap_min >= 0 || throw(ArgumentError(
        "max_obs_gap_min must be finite and nonnegative",
    ))
    summary = external_dst_summary(df)
    println(io, "# Prospective external Dst forecast snapshots\n")
    println(io, "This log captures public same-unit Dst forecast/nowcast products as issue-time snapshots. Rows are written only when the product target time is after the inferred issue time. The collector stores raw-response SHA-256 hashes and scores rows later against the SWPC-served Kyoto Dst product within $(Float64(max_obs_gap_min)) min.\n")
    println(io, "| Source | Rows | Scored | Issues | Max lead [h] | RMSE [nT] | MAE [nT] |")
    println(io, "|---|---:|---:|---:|---:|---:|---:|")
    for r in eachrow(summary)
        rmse_s = ismissing(r.rmse_nt) ? "pending" : @sprintf("%.2f", r.rmse_nt)
        mae_s = ismissing(r.mae_nt) ? "pending" : @sprintf("%.2f", r.mae_nt)
        @printf(io, "| %s | %d | %d | %d | %.3f | %s | %s |\n",
                r.source, r.n_rows, r.n_scored, r.n_issues, r.max_lead_h,
                rmse_s, mae_s)
    end
    println(io, "\nBoundary: this starts a prospective issue-time-resolved external Dst archive. It does not backfill missing historical issue snapshots, and current public products may provide sub-hour future Dst rows rather than the full 1--6 h V2 lead set.")
    return nothing
end

function _write_external_dst_report(
    path::AbstractString,
    df::DataFrame;
    max_obs_gap_min::Real=EXTERNAL_DST_MAX_OBS_GAP_MIN,
)
    _external_atomic_file(path) do io
        _emit_external_dst_report(io, df; max_obs_gap_min=max_obs_gap_min)
    end
    return path
end

# Stage the CSV and its derived report before committing either target. If any
# rename, directory sync, or injected post-log step fails, restore both exact
# pre-transaction byte sequences before propagating the failure.
function _external_transactional_log_report!(
    log_path::AbstractString,
    report_path::AbstractString,
    df::DataFrame;
    max_obs_gap_min::Real=EXTERNAL_DST_MAX_OBS_GAP_MIN,
    after_log_commit::Function=() -> nothing,
)
    !_external_targets_alias(log_path, report_path) ||
        throw(ArgumentError("external Dst log and report paths must differ"))
    log_snapshot = _external_target_snapshot(log_path)
    report_snapshot = _external_target_snapshot(report_path)
    log_stage = nothing
    report_stage = nothing
    commit_started = false
    try
        log_stage = _external_stage_file(log_path) do io
            CSV.write(io, df)
        end
        report_stage = _external_stage_file(report_path) do io
            _emit_external_dst_report(io, df; max_obs_gap_min=max_obs_gap_min)
        end
        commit_started = true
        Base.Filesystem.rename(log_stage, log_path)
        log_stage = nothing
        _external_sync_parent(log_path)
        after_log_commit()
        Base.Filesystem.rename(report_stage, report_path)
        report_stage = nothing
        _external_sync_parent(report_path)
    catch failure
        log_stage isa AbstractString && isfile(log_stage) && rm(log_stage; force=true)
        report_stage isa AbstractString && isfile(report_stage) && rm(report_stage; force=true)
        if commit_started
            rollback_errors = String[]
            for (path, snapshot) in ((log_path, log_snapshot),
                                     (report_path, report_snapshot))
                try
                    _restore_external_target!(path, snapshot)
                catch rollback_failure
                    push!(rollback_errors, string(path, ": ",
                        sprint(showerror, rollback_failure)))
                end
            end
            isempty(rollback_errors) || error(
                "external Dst file-set transaction failed ($(sprint(showerror, failure))) " *
                "and rollback was incomplete: $(join(rollback_errors, "; "))",
            )
        end
        rethrow()
    end
    return nothing
end

function capture_and_score_external_dst_snapshot!(cfg::ExternalDstCollectorConfig = ExternalDstCollectorConfig();
                                                  fetched_utc::DateTime = now(UTC),
                                                  http_get::Function = HTTP.get,
                                                  observations=nothing)
    isfinite(cfg.max_obs_gap_min) && cfg.max_obs_gap_min >= 0 ||
        throw(ArgumentError("max_obs_gap_min must be finite and nonnegative"))
    cfg.max_log_rows >= 1 || throw(ArgumentError("max_log_rows must be at least 1"))
    cfg.max_raw_snapshots >= 1 ||
        throw(ArgumentError("max_raw_snapshots must be at least 1"))

    # Fetch outside the log lock. Raw responses stay in memory until every source and the
    # observation feed succeed; otherwise a partial upstream failure could leave one uniquely
    # timestamped raw file per failed cycle and bypass the success-path retention policy.
    staged_sources = [
        _staged_future_rows_for_source(
            source; fetched_utc=fetched_utc, http_get=http_get,
        ) for source in cfg.sources
    ]
    obs = observations === nothing ?
          _fetch_observations(cfg.obs_url; http_get = http_get) : observations

    return _with_external_dst_collector_locks(cfg) do
        _external_require_regular_target(cfg.log_path)
        _external_require_regular_target(cfg.report_path)
        !_external_targets_alias(cfg.log_path, cfg.report_path) ||
            throw(ArgumentError("external Dst log and report paths must differ"))
        canonical_raw_dir = _external_canonical_raw_dir(cfg.raw_dir)
        canonical_repo_root = _external_canonical_repo_root(cfg.repo_root)
        # Load before installing any raw response. Existing unmarked logs are accepted only
        # when every row already belongs to this canonical store, which makes marker migration
        # fail closed for the inverse same-log/different-raw configuration.
        current = _load_external_log(cfg.log_path)
        _canonicalize_external_raw_references!(current, canonical_repo_root)
        _validate_external_log_raw_store(
            current, canonical_raw_dir, canonical_repo_root,
        )
        # Validate every existing provenance edge before retention can remove anything. A
        # missing, linked, or modified raw must fail closed rather than silently deleting its row.
        _verify_retained_external_raw_snapshots!(current, canonical_repo_root)
        # Persistent markers make ownership bidirectional: one raw store per log and one log
        # per raw store, with repo_root included because raw_path values are relative to it.
        _bind_external_collector_store!(
            canonical_raw_dir, cfg.log_path, canonical_repo_root;
            has_provenance_rows=nrow(current) > 0,
        )
        created_raw_paths = String[]
        committed = false
        try
            new_rows = _external_empty_log()
            for staged in staged_sources
                isempty(staged.rows) && continue
                issue = String(first(staged.rows.issue_utc))
                prior_row = findfirst(eachrow(current)) do row
                    String(row.source) == staged.source &&
                        String(row.issue_utc) == issue &&
                        String(row.raw_sha256) == staged.sha
                end
                relative_path = if prior_row === nothing
                    raw_path = _external_raw_snapshot_path(
                        canonical_raw_dir, staged.source, staged.fetched_utc, staged.sha,
                    )
                    # Preserve any pre-existing filesystem entry, including a rejected symlink or
                    # non-regular target. Cleanup owns only paths that were absent at this point.
                    existed = ispath(raw_path) || islink(raw_path)
                    existed || push!(created_raw_paths, raw_path)
                    _write_raw_snapshot(
                        canonical_raw_dir, staged.source, staged.fetched_utc,
                        staged.sha, staged.body, canonical_repo_root,
                    )
                else
                    String(current.raw_path[prior_row])
                end
                staged.rows.raw_path .= relative_path
                append!(new_rows, staged.rows; cols=:union)
            end

            combined = _dedupe_external_log(vcat(current, new_rows; cols = :union))
            rows_added = max(0, nrow(combined) - nrow(current))
            _validate_external_dst_log(
                combined; max_obs_gap_min=cfg.max_obs_gap_min,
            )
            before_retention = nrow(combined)
            retained_raw_paths, raw_to_remove = _external_raw_retention_plan(
                canonical_raw_dir, canonical_repo_root, cfg.max_raw_snapshots,
                Set(normpath(String(path)) for path in combined.raw_path),
            )
            # Raw response retention and row retention are one policy: never retain a
            # metadata row whose claimed raw payload has been rotated away.
            raw_mask = [normpath(String(path)) in retained_raw_paths
                        for path in combined.raw_path]
            combined = combined[raw_mask, :]
            _verify_retained_external_raw_snapshots!(combined, canonical_repo_root)
            combined = _retain_external_rows(combined, cfg.max_log_rows)
            rows_dropped = before_retention - nrow(combined)
            n_scored = score_external_dst_rows!(
                combined, obs; max_obs_gap_min=cfg.max_obs_gap_min,
                scored_utc=fetched_utc,
            )
            _validate_external_dst_log(
                combined; max_obs_gap_min=cfg.max_obs_gap_min,
            )
            _external_transactional_log_report!(
                cfg.log_path, cfg.report_path, combined;
                max_obs_gap_min=cfg.max_obs_gap_min,
            )
            committed = true
            raw_pruned = _delete_external_raw_snapshots!(raw_to_remove)
            return (; rows_added, rows_dropped, raw_pruned,
                    rows_total=nrow(combined), rows_scored_now=n_scored,
                    summary=external_dst_summary(combined))
        catch
            if !committed
                # A failed validation/transaction must not leak newly installed raws. Re-read
                # the authoritative log first and retain anything it actually references; if
                # that read fails, leave the files in place rather than risk a dangling row.
                referenced = try
                    current = _load_external_log(cfg.log_path)
                    _canonicalize_external_raw_references!(
                        current, canonical_repo_root,
                    )
                    Set(normpath(String(path)) for path in current.raw_path)
                catch cleanup_error
                    cleanup_error isa InterruptException && rethrow()
                    nothing
                end
                if referenced !== nothing
                    for path in created_raw_paths
                        relative = normpath(relpath(path, canonical_repo_root))
                        relative in referenced || rm(path; force=true)
                    end
                end
            end
            rethrow()
        end
    end
end

function _mock_response(body::AbstractString; last_modified::Union{Nothing, String} = nothing)
    headers = last_modified === nothing ? Pair{String, String}[] : ["Last-Modified" => last_modified]
    return (; status = 200, body = Vector{UInt8}(codeunits(String(body))), headers)
end

function _selftest_external_dst_collector()
    swpc = """[
      {"time_tag":"2026-06-27T05:00:00","dst":-20.0},
      {"time_tag":"2026-06-27T05:20:00","dst":-25.0},
      {"time_tag":"2026-06-27T05:40:00","dst":-30.0}
    ]"""
    temerin = """
          Time          Predicted Dst
    2026/178-05:01:00      -19.0
    2026/178-05:11:00      -21.0
    2026/178-05:31:00      -27.0
    """
    html = """
    <pre>
    Time of model run:     2026/178-05:05:44
    </pre>
    """
    obs = """[
      {"time_tag":"2026-06-27T05:00:00","dst":-18},
      {"time_tag":"2026-06-27T06:00:00","dst":-29}
    ]"""
    function fake_get(url; kwargs...)
        u = String(url)
        if occursin("geospace_dst_1_hour", u)
            return _mock_response(swpc; last_modified = "Sat, 27 Jun 2026 05:10:00 GMT")
        elseif occursin("dst_last_96_hrs", u)
            return _mock_response(temerin)
        elseif occursin("dsttemerin.html", u)
            return _mock_response(html)
        elseif occursin("kyoto-dst", u)
            return _mock_response(obs)
        end
        error("unexpected URL $u")
    end
    mktempdir() do dir
        cfg = ExternalDstCollectorConfig(;
            log_path = joinpath(dir, "external_dst_forecast_log.csv"),
            report_path = joinpath(dir, "external_dst_forecast_report.md"),
            raw_dir = joinpath(dir, "raw"),
            repo_root = dir,
        )
        result = capture_and_score_external_dst_snapshot!(cfg;
            fetched_utc = DateTime(2026, 6, 27, 5, 12),
            http_get = fake_get,
        )
        @assert result.rows_total == 4 "future-row filtering should keep 4 rows"
        df = CSV.read(cfg.log_path, DataFrame)
        @assert _validate_external_dst_log(df)
        @assert all(DateTime.(replace.(df.target_utc, "Z" => "")) .>
                    DateTime.(replace.(df.issue_utc, "Z" => "")))
        @assert all(length.(String.(df.raw_sha256)) .== 64)
        @assert all(!isabspath(String(p)) for p in df.raw_path)
        @assert count(.!ismissing.(df.observed_dst_nt)) == 1
        @assert isfile(cfg.report_path)
        result2 = capture_and_score_external_dst_snapshot!(cfg;
            fetched_utc = DateTime(2026, 6, 27, 5, 13),
            http_get = fake_get,
        )
        @assert result2.rows_total == 4 "dedupe should not append duplicate source issue/target/hash rows"
    end
    println("  ✓ external Dst snapshot collector self-test: future rows, raw hashes, scoring CRC")
    return true
end

function main_external_dst_collector(args = ARGS)
    if "--self-test" in args
        return _selftest_external_dst_collector()
    end
    cfg = ExternalDstCollectorConfig()
    result = capture_and_score_external_dst_snapshot!(cfg)
    println("External Dst snapshot collector: rows_total=", result.rows_total,
            ", rows_added=", result.rows_added,
            ", rows_scored_now=", result.rows_scored_now)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_external_dst_collector()
end
