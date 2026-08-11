#!/usr/bin/env julia

# Immutable receipt-time capture for the future V2.2-M2 L1 data contract.
# This collector archives raw SWPC responses and a per-source hash chain. It
# does not issue forecasts and is not started by package or service code.

using Dates
using FileWatching: Pidfile
using HTTP
using JSON3
using SHA

const V22_L1_RECEIPT_SCHEMA_VERSION = "v2_2_l1_receipt_v2"
const V22_L1_RECEIPT_SOURCES = (
    (
        name="swpc_rtsw_mag_1m",
        url="https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    ),
    (
        name="swpc_rtsw_wind_1m",
        url="https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
    ),
)
const V22_L1_RECEIPT_DEFAULT_INTERVAL_SEC = 60.0
const V22_L1_RECEIPT_ZERO_SHA256 = repeat("0", 64)

_v22_l1_sha256(bytes) = bytes2hex(sha256(bytes))
_v22_l1_utc(time::DateTime) = Dates.format(time, dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z"

function _v22_l1_parse_utc(value)
    text = String(value)
    endswith(text, "Z") || throw(ArgumentError(
        "V2.2 L1 UTC timestamp must end in Z",
    ))
    parsed = DateTime(text[1:(end - 1)])
    _v22_l1_utc(parsed) == text || throw(ArgumentError(
        "V2.2 L1 UTC timestamp is not canonical",
    ))
    return parsed
end

function _v22_l1_validate_root(root::AbstractString; create::Bool)
    path = abspath(String(root))
    islink(path) && throw(ArgumentError(
        "V2.2 L1 receipt root must not be a symbolic link: $path",
    ))
    if ispath(path)
        isdir(path) || throw(ArgumentError(
            "V2.2 L1 receipt root exists but is not a directory: $path",
        ))
    elseif create
        mkpath(path)
    else
        throw(ArgumentError("V2.2 L1 receipt root does not exist: $path"))
    end
    return realpath(path)
end

function _v22_l1_validate_source(source)
    name = String(source.name)
    url = String(source.url)
    occursin(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", name) || throw(ArgumentError(
        "V2.2 L1 source name is not safe for archive paths: $name",
    ))
    startswith(url, "https://") || throw(ArgumentError(
        "V2.2 L1 source URL must use HTTPS: $url",
    ))
    return (name=name, url=url)
end

function _v22_l1_resolve_relative(root::AbstractString,
                                  relative::AbstractString,
                                  field::AbstractString)
    value = String(relative)
    isempty(value) && throw(ArgumentError("V2.2 L1 $field must not be empty"))
    isabspath(value) && throw(ArgumentError("V2.2 L1 $field must be relative"))
    normalized = normpath(value)
    normalized == "." && throw(ArgumentError("V2.2 L1 $field is not a file path"))
    value == normalized || throw(ArgumentError(
        "V2.2 L1 $field is not canonical",
    ))
    components = splitpath(normalized)
    any(==(".."), components) && throw(ArgumentError(
        "V2.2 L1 $field escapes its storage root",
    ))
    path = normpath(joinpath(root, normalized))
    relpath(path, root) == normalized || throw(ArgumentError(
        "V2.2 L1 $field is not canonical",
    ))
    current = root
    for component in components
        current = joinpath(current, component)
        islink(current) && throw(ArgumentError(
            "V2.2 L1 $field contains a symbolic-link component: $current",
        ))
    end
    return path
end

function _v22_l1_mkpath_relative(root::AbstractString,
                                 relative::AbstractString,
                                 field::AbstractString)
    normalized = normpath(String(relative))
    normalized == "." && return root
    isabspath(normalized) && throw(ArgumentError("V2.2 L1 $field must be relative"))
    components = splitpath(normalized)
    any(==(".."), components) && throw(ArgumentError(
        "V2.2 L1 $field escapes its storage root",
    ))
    current = root
    for component in components
        current = joinpath(current, component)
        islink(current) && throw(ArgumentError(
            "V2.2 L1 $field contains a symbolic-link component: $current",
        ))
        if ispath(current)
            isdir(current) || throw(ArgumentError(
                "V2.2 L1 $field component is not a directory: $current",
            ))
        else
            mkdir(current)
            _v22_l1_sync_parent(current)
        end
    end
    return current
end

function _v22_l1_require_regular_target(path::AbstractString)
    islink(path) && throw(ArgumentError(
        "V2.2 L1 receipt target must not be a symbolic link: $path",
    ))
    ispath(path) && !isfile(path) && throw(ArgumentError(
        "V2.2 L1 receipt target exists but is not a regular file: $path",
    ))
    return path
end

function _v22_l1_durable_flush(io::IO)
    flush(io)
    rc = Sys.iswindows() ?
         ccall(:_commit, Cint, (Cint,), fd(io)) :
         ccall(:fsync, Cint, (Cint,), fd(io))
    systemerror("durable V2.2 L1 receipt flush", rc != 0)
    return nothing
end

function _v22_l1_sync_parent(path::AbstractString)
    Sys.iswindows() && return nothing
    parent = dirname(path)
    isempty(parent) && (parent = ".")
    open(parent, "r") do io
        rc = ccall(:fsync, Cint, (Cint,), fd(io))
        systemerror("durable V2.2 L1 receipt directory flush", rc != 0)
    end
    return nothing
end

function _v22_l1_atomic_file(writer::Function,
                             root::AbstractString,
                             relative::AbstractString;
                             replace::Bool=false)
    normalized = normpath(String(relative))
    target = _v22_l1_resolve_relative(root, normalized, "output path")
    parent_relative = dirname(normalized)
    parent = _v22_l1_mkpath_relative(root, parent_relative, "output directory")
    target == joinpath(parent, basename(normalized)) || throw(ArgumentError(
        "V2.2 L1 output path is inconsistent with its parent",
    ))
    _v22_l1_require_regular_target(target)
    !replace && ispath(target) && throw(ArgumentError(
        "V2.2 L1 immutable target already exists: $target",
    ))
    temporary, io = mktemp(parent; cleanup=false)
    try
        writer(io)
        _v22_l1_durable_flush(io)
        close(io)
        _v22_l1_resolve_relative(root, normalized, "output path") == target ||
            throw(ArgumentError("V2.2 L1 output path changed during installation"))
        _v22_l1_require_regular_target(target)
        !replace && ispath(target) && throw(ArgumentError(
            "V2.2 L1 immutable target appeared during installation: $target",
        ))
        Base.Filesystem.rename(temporary, target)
        _v22_l1_sync_parent(target)
    catch
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary; force=true)
        rethrow()
    end
    return target
end

_v22_l1_atomic_bytes(root::AbstractString, relative::AbstractString,
                     bytes::AbstractVector{UInt8}) =
    _v22_l1_atomic_file(root, relative) do io
        write(io, bytes)
    end

_v22_l1_atomic_text(root::AbstractString, relative::AbstractString,
                    value::AbstractString; replace::Bool=false) =
    _v22_l1_atomic_file(root, relative; replace=replace) do io
        write(io, value)
    end

function _v22_l1_header(response, requested::AbstractString)
    lowercase_requested = lowercase(String(requested))
    for pair in getproperty(response, :headers)
        key, value = pair isa Pair ? (first(pair), last(pair)) : (pair[1], pair[2])
        lowercase(String(key)) == lowercase_requested && return String(value)
    end
    return ""
end

function _v22_l1_response_body(response)
    body = getproperty(response, :body)
    body isa AbstractVector{UInt8} && return Vector{UInt8}(body)
    return Vector{UInt8}(codeunits(String(body)))
end

function _v22_l1_body_diagnostics(body::Vector{UInt8})
    try
        # `String(::Vector{UInt8})` may take ownership of and empty its input.
        # Diagnostics must never mutate the raw response whose size is recorded.
        parsed = JSON3.read(String(copy(body)))
        parsed isa AbstractVector || return (
            json_valid=true, array_valid=false, row_count=0,
            minimum_time_tag="", maximum_time_tag="",
        )
        times = String[]
        for row in parsed
            raw = try
                get(row, :time_tag, get(row, "time_tag", nothing))
            catch error
                error isa InterruptException && rethrow()
                nothing
            end
            raw === nothing || push!(times, String(raw))
        end
        sort!(times)
        return (
            json_valid=true,
            array_valid=true,
            row_count=length(parsed),
            minimum_time_tag=isempty(times) ? "" : first(times),
            maximum_time_tag=isempty(times) ? "" : last(times),
        )
    catch error
        error isa InterruptException && rethrow()
        return (
            json_valid=false, array_valid=false, row_count=0,
            minimum_time_tag="", maximum_time_tag="",
        )
    end
end

function _v22_l1_record_payload(record)
    return (
        schema_version=String(record.schema_version),
        source_name=String(record.source_name),
        source_url=String(record.source_url),
        request_started_utc=String(record.request_started_utc),
        receipt_completed_utc=String(record.receipt_completed_utc),
        monotonic_started_ns=Int(record.monotonic_started_ns),
        monotonic_completed_ns=Int(record.monotonic_completed_ns),
        capture_outcome=String(record.capture_outcome),
        transport_error_type=String(record.transport_error_type),
        transport_error_message=String(record.transport_error_message),
        http_status=Int(record.http_status),
        http_date=String(record.http_date),
        http_etag=String(record.http_etag),
        http_last_modified=String(record.http_last_modified),
        body_bytes=Int(record.body_bytes),
        body_sha256=String(record.body_sha256),
        raw_relative_path=String(record.raw_relative_path),
        json_valid=Bool(record.json_valid),
        array_valid=Bool(record.array_valid),
        row_count=Int(record.row_count),
        minimum_time_tag=String(record.minimum_time_tag),
        maximum_time_tag=String(record.maximum_time_tag),
        sequence=Int(record.sequence),
        previous_record_relative_path=String(record.previous_record_relative_path),
        previous_record_sha256=String(record.previous_record_sha256),
    )
end

function _v22_l1_record_sha256(record)
    payload = JSON3.write(_v22_l1_record_payload(record))
    return _v22_l1_sha256(codeunits(payload))
end

_v22_l1_latest_relative(source_name::AbstractString) =
    joinpath("latest", String(source_name) * ".json")

function _v22_l1_latest_path(root::AbstractString, source_name::AbstractString)
    return _v22_l1_resolve_relative(
        root, _v22_l1_latest_relative(source_name), "latest pointer path",
    )
end

function _v22_l1_read_latest(root::AbstractString, source_name::AbstractString)
    path = _v22_l1_latest_path(root, source_name)
    isfile(path) || return (
        sequence=0,
        record_relative_path="",
        record_sha256=V22_L1_RECEIPT_ZERO_SHA256,
    )
    islink(path) && throw(ArgumentError(
        "V2.2 L1 latest pointer must not be a symbolic link: $path",
    ))
    value = JSON3.read(read(path, String))
    String(value.schema_version) == V22_L1_RECEIPT_SCHEMA_VERSION ||
        throw(ArgumentError("V2.2 L1 latest pointer has the wrong schema"))
    String(value.source_name) == String(source_name) || throw(ArgumentError(
        "V2.2 L1 latest pointer changed source identity",
    ))
    sequence = Int(value.sequence)
    sequence >= 1 || throw(ArgumentError(
        "V2.2 L1 latest pointer has an invalid sequence",
    ))
    relative = String(value.record_relative_path)
    _v22_l1_resolve_relative(root, relative, "latest record path")
    checksum = String(value.record_sha256)
    occursin(r"^[0-9a-f]{64}$", checksum) || throw(ArgumentError(
        "V2.2 L1 latest pointer has a malformed checksum",
    ))
    return (
        sequence=sequence,
        record_relative_path=relative,
        record_sha256=checksum,
    )
end

function _v22_l1_store_raw!(root::AbstractString, body::Vector{UInt8})
    checksum = _v22_l1_sha256(body)
    relative = joinpath("raw", checksum[1:2], checksum * ".raw")
    path = _v22_l1_resolve_relative(root, relative, "raw object path")
    if isfile(path)
        islink(path) && throw(ArgumentError(
            "V2.2 L1 raw object must not be a symbolic link: $path",
        ))
        _v22_l1_sha256(read(path)) == checksum || error(
            "V2.2 L1 content-addressed raw object is corrupt: $path",
        )
    else
        _v22_l1_atomic_bytes(root, relative, body)
        _v22_l1_sha256(read(path)) == checksum || error(
            "V2.2 L1 raw object failed post-write verification: $path",
        )
    end
    return relative, checksum
end

function _v22_l1_record_relative(source_name::AbstractString,
                                 sequence::Integer)
    sequence >= 1 || throw(ArgumentError(
        "V2.2 L1 record sequence must be positive",
    ))
    token = lpad(string(sequence), 20, '0')
    return joinpath("records", String(source_name), token * ".json")
end

function _v22_l1_validate_predecessor(root::AbstractString, source,
                                      receipt_completed::DateTime)
    prior = _v22_l1_read_latest(root, source.name)
    if prior.sequence == 0
        relative_directory = joinpath("records", String(source.name))
        directory = _v22_l1_resolve_relative(
            root, relative_directory, "source record directory",
        )
        if ispath(directory)
            isdir(directory) || error(
                "V2.2 L1 source record path is not a directory: $directory",
            )
            isempty(readdir(directory)) || error(
                "V2.2 L1 source has orphan records without a latest pointer",
            )
        end
        return prior
    end

    verified = _v22_l1_verify_source(
        root, source; require_nonempty=true, head_only=true,
    )
    verified.latest_record_sha256 == prior.record_sha256 || error(
        "V2.2 L1 predecessor changed during validation",
    )
    receipt_completed >= verified.latest_receipt_completed_utc ||
        throw(ArgumentError(
            "V2.2 L1 receipt UTC regressed behind its predecessor",
        ))
    next_relative = _v22_l1_record_relative(source.name, prior.sequence + 1)
    next_path = _v22_l1_resolve_relative(
        root, next_relative, "next record path",
    )
    ispath(next_path) && error(
        "V2.2 L1 next sequence path already exists outside the current chain",
    )
    return prior
end

function _v22_l1_commit_record!(root::AbstractString, source,
                                request_started::DateTime,
                                receipt_completed::DateTime,
                                monotonic_started_ns::Int,
                                monotonic_completed_ns::Int,
                                evidence)
    receipt_completed >= request_started || throw(ArgumentError(
        "V2.2 L1 receipt completion precedes request start",
    ))
    monotonic_completed_ns >= monotonic_started_ns || throw(ArgumentError(
        "V2.2 L1 monotonic completion precedes request start",
    ))
    prior = _v22_l1_validate_predecessor(root, source, receipt_completed)
    base_record = (
        schema_version=V22_L1_RECEIPT_SCHEMA_VERSION,
        source_name=String(source.name),
        source_url=String(source.url),
        request_started_utc=_v22_l1_utc(request_started),
        receipt_completed_utc=_v22_l1_utc(receipt_completed),
        monotonic_started_ns=monotonic_started_ns,
        monotonic_completed_ns=monotonic_completed_ns,
        capture_outcome=String(evidence.capture_outcome),
        transport_error_type=String(evidence.transport_error_type),
        transport_error_message=String(evidence.transport_error_message),
        http_status=Int(evidence.http_status),
        http_date=String(evidence.http_date),
        http_etag=String(evidence.http_etag),
        http_last_modified=String(evidence.http_last_modified),
        body_bytes=Int(evidence.body_bytes),
        body_sha256=String(evidence.body_sha256),
        raw_relative_path=String(evidence.raw_relative_path),
        json_valid=Bool(evidence.json_valid),
        array_valid=Bool(evidence.array_valid),
        row_count=Int(evidence.row_count),
        minimum_time_tag=String(evidence.minimum_time_tag),
        maximum_time_tag=String(evidence.maximum_time_tag),
        sequence=prior.sequence + 1,
        previous_record_relative_path=prior.record_relative_path,
        previous_record_sha256=prior.record_sha256,
    )
    record_sha256 = _v22_l1_record_sha256(base_record)
    relative = _v22_l1_record_relative(source.name, base_record.sequence)
    path = _v22_l1_resolve_relative(root, relative, "record path")
    record = merge(base_record, (record_sha256=record_sha256,))
    ispath(path) && error("V2.2 L1 receipt record identity collision: $path")
    _v22_l1_atomic_text(root, relative, JSON3.write(record))
    _v22_l1_atomic_text(
        root,
        _v22_l1_latest_relative(source.name),
        JSON3.write((
            schema_version=V22_L1_RECEIPT_SCHEMA_VERSION,
            source_name=String(source.name),
            sequence=record.sequence,
            record_relative_path=relative,
            record_sha256=record_sha256,
        )),
        replace=true,
    )
    return merge(record, (record_relative_path=relative,))
end

function _v22_l1_install_record!(root::AbstractString, source,
                                 response, request_started::DateTime,
                                 receipt_completed::DateTime,
                                 monotonic_started_ns::Int,
                                 monotonic_completed_ns::Int)
    status = Int(getproperty(response, :status))
    100 <= status <= 599 || throw(ArgumentError(
        "V2.2 L1 HTTP status must lie between 100 and 599",
    ))
    body = _v22_l1_response_body(response)
    diagnostic = _v22_l1_body_diagnostics(body)
    prior = _v22_l1_validate_predecessor(root, source, receipt_completed)
    raw_relative, body_sha256 = _v22_l1_store_raw!(root, body)
    evidence = (
        capture_outcome="http_response",
        transport_error_type="",
        transport_error_message="",
        http_status=status,
        http_date=_v22_l1_header(response, "date"),
        http_etag=_v22_l1_header(response, "etag"),
        http_last_modified=_v22_l1_header(response, "last-modified"),
        body_bytes=length(body),
        body_sha256=body_sha256,
        raw_relative_path=raw_relative,
        json_valid=diagnostic.json_valid,
        array_valid=diagnostic.array_valid,
        row_count=diagnostic.row_count,
        minimum_time_tag=diagnostic.minimum_time_tag,
        maximum_time_tag=diagnostic.maximum_time_tag,
    )
    # Revalidation inside commit closes the gap between raw installation and
    # advancing the chain head.
    prior == _v22_l1_read_latest(root, source.name) || error(
        "V2.2 L1 predecessor changed before record commit",
    )
    return _v22_l1_commit_record!(
        root, source, request_started, receipt_completed,
        monotonic_started_ns, monotonic_completed_ns, evidence,
    )
end

function _v22_l1_install_transport_error!(root::AbstractString, source,
                                          error_value,
                                          request_started::DateTime,
                                          receipt_completed::DateTime,
                                          monotonic_started_ns::Int,
                                          monotonic_completed_ns::Int)
    evidence = (
        capture_outcome="transport_error",
        transport_error_type=string(typeof(error_value)),
        transport_error_message=sprint(showerror, error_value),
        http_status=0,
        http_date="",
        http_etag="",
        http_last_modified="",
        body_bytes=0,
        body_sha256=V22_L1_RECEIPT_ZERO_SHA256,
        raw_relative_path="",
        json_valid=false,
        array_valid=false,
        row_count=0,
        minimum_time_tag="",
        maximum_time_tag="",
    )
    return _v22_l1_commit_record!(
        root, source, request_started, receipt_completed,
        monotonic_started_ns, monotonic_completed_ns, evidence,
    )
end

function _v22_l1_with_lock(work::Function, root::AbstractString;
                           timeout_sec::Real=30.0)
    timeout = Float64(timeout_sec)
    isfinite(timeout) && timeout >= 0.0 || throw(ArgumentError(
        "V2.2 L1 lock timeout must be finite and nonnegative",
    ))
    lock_path = joinpath(root, ".collector.lock")
    deadline = time() + timeout
    owner = false
    while owner === false
        !islink(lock_path) || throw(ArgumentError(
            "V2.2 L1 collector lock must not be a symbolic link",
        ))
        owner = Pidfile.trymkpidlock(
            lock_path; stale_age=900.0, refresh=300.0,
        )
        owner === false || break
        time() < deadline || error("timed out waiting for V2.2 L1 collector lock")
        sleep(0.05)
    end
    try
        return work()
    finally
        close(owner)
    end
end

function capture_v2_2_l1_receipts!(root::AbstractString;
        sources=V22_L1_RECEIPT_SOURCES,
        http_get::Function=HTTP.get,
        utc_clock::Function=() -> now(UTC),
        monotonic_clock::Function=time_ns,
        lock_timeout_sec::Real=30.0,
        verify_existing::Bool=true)
    isempty(sources) && throw(ArgumentError("V2.2 L1 sources must not be empty"))
    validated_sources = map(_v22_l1_validate_source, sources)
    length(unique(lowercase(source.name) for source in validated_sources)) ==
        length(validated_sources) || throw(ArgumentError(
            "V2.2 L1 source names must be unique ignoring case",
        ))
    storage = _v22_l1_validate_root(root; create=true)
    return _v22_l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        if verify_existing
            for source in validated_sources
                _v22_l1_verify_source(
                    storage, source; require_nonempty=false,
                )
            end
        end
        records = NamedTuple[]
        for source in validated_sources
            started_utc = utc_clock()
            started_ns = Int(monotonic_clock())
            transport_error = nothing
            response = try
                http_get(
                    String(source.url);
                    connect_timeout=15,
                    readtimeout=30,
                    retries=0,
                    status_exception=false,
                )
            catch error_value
                error_value isa InterruptException && rethrow()
                transport_error = error_value
                nothing
            end
            completed_ns = Int(monotonic_clock())
            completed_utc = utc_clock()
            record = if transport_error !== nothing
                _v22_l1_install_transport_error!(
                    storage, source, transport_error, started_utc, completed_utc,
                    started_ns, completed_ns,
                )
            else
                _v22_l1_install_record!(
                    storage, source, response, started_utc, completed_utc,
                    started_ns, completed_ns,
                )
            end
            push!(records, record)
        end
        return records
    end
end

function _v22_l1_source_record_files(root::AbstractString,
                                     source_name::AbstractString)
    relative_directory = joinpath("records", String(source_name))
    directory = _v22_l1_resolve_relative(
        root, relative_directory, "source record directory",
    )
    ispath(directory) || return Set{String}()
    isdir(directory) && !islink(directory) || error(
        "V2.2 L1 source record directory is not regular: $directory",
    )
    files = Set{String}()
    for entry in readdir(directory)
        relative = joinpath(relative_directory, entry)
        path = _v22_l1_resolve_relative(root, relative, "source record entry")
        isfile(path) && !islink(path) || error(
            "V2.2 L1 source record entry is not a regular file: $path",
        )
        push!(files, relative)
    end
    return files
end

function _v22_l1_verify_response_evidence(root::AbstractString, record,
                                          raw_paths::Set{String})
    isempty(String(record.transport_error_type)) &&
        isempty(String(record.transport_error_message)) || error(
            "V2.2 L1 HTTP response contains transport-error metadata",
        )
    status = Int(record.http_status)
    100 <= status <= 599 || error("V2.2 L1 HTTP response has an invalid status")
    body_sha256 = String(record.body_sha256)
    occursin(r"^[0-9a-f]{64}$", body_sha256) || error(
        "V2.2 L1 raw response checksum is malformed",
    )
    expected_relative = joinpath(
        "raw", body_sha256[1:2], body_sha256 * ".raw",
    )
    raw_relative = normpath(String(record.raw_relative_path))
    raw_relative == expected_relative || error(
        "V2.2 L1 raw response path is not content addressed",
    )
    raw_path = _v22_l1_resolve_relative(root, raw_relative, "raw path")
    isfile(raw_path) && !islink(raw_path) || error(
        "V2.2 L1 raw response is missing or not regular",
    )
    bytes = read(raw_path)
    _v22_l1_sha256(bytes) == body_sha256 || error(
        "V2.2 L1 raw response checksum mismatch",
    )
    length(bytes) == Int(record.body_bytes) || error(
        "V2.2 L1 raw response size mismatch",
    )
    diagnostic = _v22_l1_body_diagnostics(bytes)
    diagnostic.json_valid == Bool(record.json_valid) &&
        diagnostic.array_valid == Bool(record.array_valid) &&
        diagnostic.row_count == Int(record.row_count) &&
        diagnostic.minimum_time_tag == String(record.minimum_time_tag) &&
        diagnostic.maximum_time_tag == String(record.maximum_time_tag) || error(
            "V2.2 L1 raw-response diagnostics changed",
        )
    push!(raw_paths, raw_relative)
    return nothing
end

function _v22_l1_verify_transport_error(record)
    isempty(String(record.transport_error_type)) && error(
        "V2.2 L1 transport-error record omits its exception type",
    )
    Int(record.http_status) == 0 &&
        isempty(String(record.http_date)) &&
        isempty(String(record.http_etag)) &&
        isempty(String(record.http_last_modified)) &&
        Int(record.body_bytes) == 0 &&
        String(record.body_sha256) == V22_L1_RECEIPT_ZERO_SHA256 &&
        isempty(String(record.raw_relative_path)) &&
        !Bool(record.json_valid) &&
        !Bool(record.array_valid) &&
        Int(record.row_count) == 0 &&
        isempty(String(record.minimum_time_tag)) &&
        isempty(String(record.maximum_time_tag)) || error(
            "V2.2 L1 transport-error evidence fields are inconsistent",
        )
    return nothing
end

function _v22_l1_verify_source(root::AbstractString, source;
                               require_nonempty::Bool,
                               head_only::Bool=false)
    current = _v22_l1_read_latest(root, source.name)
    if current.sequence == 0
        require_nonempty && error(
            "V2.2 L1 source has no receipt records: $(source.name)",
        )
        isempty(_v22_l1_source_record_files(root, source.name)) || error(
            "V2.2 L1 source has orphan records without a latest pointer",
        )
        return (
            source_name=String(source.name), records=0,
            latest_record_sha256=V22_L1_RECEIPT_ZERO_SHA256,
            latest_receipt_completed_utc=nothing,
            record_relative_paths=Set{String}(),
            raw_relative_paths=Set{String}(),
        )
    end

    latest = current
    records = 0
    seen = Set{String}()
    raw_paths = Set{String}()
    expected_sequence = current.sequence
    newer_receipt = nothing
    latest_receipt = nothing
    while !isempty(current.record_relative_path)
        relative = normpath(current.record_relative_path)
        relative in seen && error("V2.2 L1 receipt chain contains a cycle")
        push!(seen, relative)
        path = _v22_l1_resolve_relative(root, relative, "record path")
        isfile(path) && !islink(path) || error(
            "V2.2 L1 receipt record is missing or not regular: $path",
        )
        record = JSON3.read(read(path, String))
        String(record.schema_version) == V22_L1_RECEIPT_SCHEMA_VERSION || error(
            "V2.2 L1 receipt record has the wrong schema",
        )
        String(record.source_name) == String(source.name) || error(
            "V2.2 L1 receipt source chain changed identity",
        )
        String(record.source_url) == String(source.url) || error(
            "V2.2 L1 receipt source URL changed identity",
        )
        Int(record.sequence) == expected_sequence || error(
            "V2.2 L1 receipt sequence is discontinuous",
        )
        checksum = _v22_l1_record_sha256(record)
        checksum == String(record.record_sha256) == current.record_sha256 || error(
            "V2.2 L1 receipt record checksum mismatch",
        )
        relative == _v22_l1_record_relative(source.name, expected_sequence) ||
            error("V2.2 L1 record path is not sequence addressed")

        outcome = String(record.capture_outcome)
        if outcome == "http_response"
            _v22_l1_verify_response_evidence(root, record, raw_paths)
        elseif outcome == "transport_error"
            _v22_l1_verify_transport_error(record)
        else
            error("V2.2 L1 receipt record has an unknown capture outcome")
        end

        request_started = _v22_l1_parse_utc(record.request_started_utc)
        receipt_completed = _v22_l1_parse_utc(record.receipt_completed_utc)
        receipt_completed >= request_started || error(
            "V2.2 L1 receipt completion precedes request start",
        )
        if newer_receipt !== nothing
            newer_receipt >= receipt_completed || error(
                "V2.2 L1 receipt UTC regresses across sequence order",
            )
        else
            latest_receipt = receipt_completed
        end
        newer_receipt = receipt_completed
        Int(record.monotonic_completed_ns) >= Int(record.monotonic_started_ns) ||
            error("V2.2 L1 monotonic completion precedes request start")

        records += 1
        previous_path = String(record.previous_record_relative_path)
        previous_sha = String(record.previous_record_sha256)
        if isempty(previous_path)
            previous_sha == V22_L1_RECEIPT_ZERO_SHA256 || error(
                "V2.2 L1 chain origin checksum is invalid",
            )
        else
            occursin(r"^[0-9a-f]{64}$", previous_sha) || error(
                "V2.2 L1 predecessor checksum is malformed",
            )
        end
        if head_only
            return (
                source_name=String(source.name),
                records=latest.sequence,
                latest_record_sha256=latest.record_sha256,
                latest_receipt_completed_utc=latest_receipt,
                record_relative_paths=seen,
                raw_relative_paths=raw_paths,
            )
        end
        current = (
            sequence=expected_sequence - 1,
            record_relative_path=previous_path,
            record_sha256=previous_sha,
        )
        expected_sequence -= 1
    end
    expected_sequence == 0 || error("V2.2 L1 receipt chain ended early")
    _v22_l1_source_record_files(root, source.name) == seen || error(
        "V2.2 L1 source contains orphan or unchained record files",
    )
    return (
        source_name=String(source.name),
        records=records,
        latest_record_sha256=latest.record_sha256,
        latest_receipt_completed_utc=latest_receipt,
        record_relative_paths=seen,
        raw_relative_paths=raw_paths,
    )
end

function _v22_l1_shallow_entries(root::AbstractString,
                                 relative::AbstractString,
                                 field::AbstractString)
    directory = _v22_l1_resolve_relative(root, relative, field)
    isdir(directory) && !islink(directory) || error(
        "V2.2 L1 $field is missing or not a regular directory",
    )
    return readdir(directory)
end

function _v22_l1_tree_files(root::AbstractString,
                            relative::AbstractString;
                            allow_missing::Bool)
    directory = _v22_l1_resolve_relative(root, relative, "$relative directory")
    if !ispath(directory)
        allow_missing && return Set{String}()
        error("V2.2 L1 $relative directory is missing")
    end
    isdir(directory) && !islink(directory) || error(
        "V2.2 L1 $relative path is not a regular directory",
    )
    files = Set{String}()
    for (current, directories, names) in walkdir(directory; follow_symlinks=false)
        for name in directories
            path = joinpath(current, name)
            islink(path) && error(
                "V2.2 L1 $relative tree contains a symbolic-link directory: $path",
            )
        end
        for name in names
            path = joinpath(current, name)
            isfile(path) && !islink(path) || error(
                "V2.2 L1 $relative tree contains a non-regular file: $path",
            )
            relative_path = normpath(relpath(path, root))
            _v22_l1_resolve_relative(root, relative_path, "$relative file")
            push!(files, relative_path)
        end
    end
    return files
end

function verify_v2_2_l1_receipts(root::AbstractString;
                                 sources=V22_L1_RECEIPT_SOURCES)
    isempty(sources) && throw(ArgumentError(
        "V2.2 L1 verification sources must not be empty",
    ))
    storage = _v22_l1_validate_root(root; create=false)
    validated_sources = map(_v22_l1_validate_source, sources)
    length(unique(lowercase(source.name) for source in validated_sources)) ==
        length(validated_sources) || throw(ArgumentError(
            "V2.2 L1 source names must be unique ignoring case",
        ))

    expected_names = Set(source.name for source in validated_sources)
    Set(_v22_l1_shallow_entries(storage, "records", "records directory")) ==
        expected_names || error(
            "V2.2 L1 records directory does not match the configured sources",
        )
    expected_latest = Set(source.name * ".json" for source in validated_sources)
    Set(_v22_l1_shallow_entries(storage, "latest", "latest directory")) ==
        expected_latest || error(
            "V2.2 L1 latest directory does not match the configured sources",
        )

    result = NamedTuple[]
    expected_raw = Set{String}()
    for source in validated_sources
        verified = _v22_l1_verify_source(storage, source; require_nonempty=true)
        union!(expected_raw, verified.raw_relative_paths)
        push!(result, (
            source_name=verified.source_name,
            records=verified.records,
            latest_record_sha256=verified.latest_record_sha256,
        ))
    end
    actual_raw = _v22_l1_tree_files(
        storage, "raw"; allow_missing=isempty(expected_raw),
    )
    actual_raw == expected_raw || error(
        "V2.2 L1 raw store contains missing or orphan content",
    )
    return result
end

function _v22_l1_argument(args, prefix, fallback)
    match = findfirst(argument -> startswith(argument, prefix), args)
    match === nothing && return fallback
    return split(args[match], '='; limit=2)[2]
end

function main_v2_2_l1_receipt_collector(args=ARGS)
    root = _v22_l1_argument(
        args, "--root=", joinpath(@__DIR__, "..", "var", "v2_2_l1_receipts"),
    )
    interval = parse(Float64, _v22_l1_argument(
        args, "--interval-sec=", string(V22_L1_RECEIPT_DEFAULT_INTERVAL_SEC),
    ))
    isfinite(interval) && interval > 0.0 || error(
        "--interval-sec must be finite and positive",
    )
    once = "--once" in args
    first_cycle = true
    while true
        records = capture_v2_2_l1_receipts!(
            root; verify_existing=first_cycle,
        )
        first_cycle = false
        println("captured V2.2 L1 receipts: ", join(
            (record.source_name * "=" * record.record_sha256[1:12]
             for record in records), ", ",
        ))
        once && break
        sleep(interval)
    end
    verify_v2_2_l1_receipts(root)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_v2_2_l1_receipt_collector()
end
