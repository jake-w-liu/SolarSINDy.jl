#!/usr/bin/env julia

# Immutable receipt-time capture for the future V2.2-M2 L1 data contract.
# This collector archives raw SWPC responses and a per-source hash chain. It
# does not issue forecasts and is not started by package or service code.

using Dates
using FileWatching: Pidfile
using HTTP
using JSON3
using SHA

const V22_L1_RECEIPT_SCHEMA_VERSION = "v2_2_l1_receipt_v4"
const V22_L1_METADATA_CONTRACT_VERSION = "noaa_rtsw_metadata_v2"
const V22_L1_NOAA_METADATA_AUTHORITY_URL =
    "https://www.weather.gov/media/notification/pdf_2026/" *
    "scn26-21_Data_Format_Changes_Impacting_SWPC_Products.pdf"
const V22_L1_NOAA_QUALITY_AUTHORITY_URL =
    "https://www.swpc.noaa.gov/products/real-time-solar-wind"
const V22_L1_NOAA_MAG_QUALITY_SCHEMA_URL =
    "https://data.noaa.gov/waf/NOAA/NESDIS/NGDC/STP/Space_Weather/" *
    "iso/xml/dscovr_m1m.xml"
const V22_L1_NOAA_WIND_QUALITY_SCHEMA_URL =
    "https://data.noaa.gov/waf/NOAA/NESDIS/NGDC/STP/Space_Weather/" *
    "iso/xml/dscovr_f1m.xml"
const V22_L1_NASA_QUALITY_AUTHORITY_URL =
    "https://cdaweb.gsfc.nasa.gov/misc/NotesD.html"
const V22_L1_NOAA_EPHEMERIS_URL =
    "https://services.swpc.noaa.gov/json/rtsw/rtsw_ephemerides_1h.json"
const V22_L1_NOAA_EPHEMERIS_SCHEMA_URL =
    "https://data.noaa.gov/waf/NOAA/NESDIS/NGDC/STP/Space_Weather/" *
    "iso/xml/dscovr_pop.xml"
const V22_L1_NOAA_EPHEMERIS_ARCHIVE_CONTRACT_URL =
    "https://www.ncei.noaa.gov/archive/atrac/export/" *
    "2015-06-15T18-40-09.pdf?id=24749"
const V22_L1_NASA_EPHEMERIS_AUTHORITY_URL = "https://sscweb.gsfc.nasa.gov/"
const V22_L1_SOURCE_FIELD_SEMANTICS =
    "source identifies the satellite from which the data originated"
const V22_L1_ACTIVE_FIELD_SEMANTICS =
    "active indicates whether SWPC forecasters considered the satellite active at the time"
const V22_L1_QUALITY_SEMANTICS =
    "NOAA DSCOVR overall_quality: 0 normal, 1 suspect, 2 error"
const V22_L1_ARCHIVE_QUALITY_SEMANTICS =
    "NASA CDAWeb DSCOVR quality flags have dataset-specific definitions"
const V22_L1_EPHEMERIS_INTERPOLATION_RULE =
    "same-source active GSE exact match or bracketed linear interpolation; " *
    "maximum one-hour bracket; no extrapolation"
const V22_L1_EPHEMERIS_POSITION_FRAME = "GSE"
const V22_L1_EPHEMERIS_POSITION_UNITS = "km"
const V22_L1_EPHEMERIS_MISSING_VALUE = -99999.0
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

function _v22_l1_parse_source_utc(value)
    text = String(value)
    endswith(text, "Z") || throw(ArgumentError(
        "V2.2 L1 source timestamp must end in Z",
    ))
    parsed = DateTime(text[1:(end - 1)])
    canonical = _v22_l1_utc(parsed)
    (text == canonical ||
     (endswith(canonical, ".000Z") && text == canonical[1:(end - 5)] * "Z")) ||
        throw(ArgumentError("V2.2 L1 source timestamp is not canonical"))
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

function _v22_l1_normalized_source_url(url::AbstractString)
    uri = try
        HTTP.URI(String(url))
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError("V2.2 L1 source URL is malformed: $url"))
    end
    lowercase(String(uri.scheme)) == "https" && !isempty(String(uri.host)) ||
        throw(ArgumentError("V2.2 L1 source URL is malformed: $url"))
    port = isempty(String(uri.port)) || String(uri.port) == "443" ? "" :
           ":" * String(uri.port)
    path = isempty(String(uri.path)) ? "/" : String(uri.path)
    query = isempty(String(uri.query)) ? "" : "?" * String(uri.query)
    userinfo = isempty(String(uri.userinfo)) ? "" : String(uri.userinfo) * "@"
    return "https://" * userinfo * lowercase(String(uri.host)) * port * path * query
end

function _v22_l1_reject_reused_source_urls(sources)
    identities = _v22_l1_normalized_source_url.(getproperty.(sources, :url))
    length(unique(identities)) == length(identities) || throw(ArgumentError(
        "V2.2 L1 source URLs must be unique after normalization",
    ))
    return sources
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

function _v22_l1_object_field(row, key::Symbol)
    (row isa AbstractDict || row isa JSON3.Object) || return nothing
    haskey(row, key) && return row[key]
    text = String(key)
    haskey(row, text) && return row[text]
    return nothing
end

function _v22_l1_ephemeris_capture_unavailable(status::AbstractString)
    return (
        capture_outcome=String(status),
        transport_error_type="",
        transport_error_message="",
        request_started_utc="",
        receipt_completed_utc="",
        monotonic_started_ns=0,
        monotonic_completed_ns=0,
        http_status=0,
        http_date="",
        http_etag="",
        http_last_modified="",
        body=UInt8[],
        body_bytes=0,
        body_sha256=V22_L1_RECEIPT_ZERO_SHA256,
        raw_relative_path="",
    )
end

function _v22_l1_capture_ephemeris(http_get::Function,
                                   utc_clock::Function,
                                   monotonic_clock::Function)
    started_utc = utc_clock()
    started_ns = Int(monotonic_clock())
    transport_error = nothing
    response = try
        http_get(
            V22_L1_NOAA_EPHEMERIS_URL;
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
    completed_utc >= started_utc || throw(ArgumentError(
        "V2.2 L1 ephemeris receipt completion precedes request start",
    ))
    completed_ns >= started_ns || throw(ArgumentError(
        "V2.2 L1 ephemeris monotonic completion precedes request start",
    ))
    if transport_error !== nothing
        return merge(_v22_l1_ephemeris_capture_unavailable("transport_error"), (
            transport_error_type=string(typeof(transport_error)),
            transport_error_message=sprint(showerror, transport_error),
            request_started_utc=_v22_l1_utc(started_utc),
            receipt_completed_utc=_v22_l1_utc(completed_utc),
            monotonic_started_ns=started_ns,
            monotonic_completed_ns=completed_ns,
        ))
    end

    status = Int(getproperty(response, :status))
    100 <= status <= 599 || throw(ArgumentError(
        "V2.2 L1 ephemeris HTTP status must lie between 100 and 599",
    ))
    body = _v22_l1_response_body(response)
    return (
        capture_outcome="http_response",
        transport_error_type="",
        transport_error_message="",
        request_started_utc=_v22_l1_utc(started_utc),
        receipt_completed_utc=_v22_l1_utc(completed_utc),
        monotonic_started_ns=started_ns,
        monotonic_completed_ns=completed_ns,
        http_status=status,
        http_date=_v22_l1_header(response, "date"),
        http_etag=_v22_l1_header(response, "etag"),
        http_last_modified=_v22_l1_header(response, "last-modified"),
        body=body,
        body_bytes=length(body),
        body_sha256=_v22_l1_sha256(body),
        raw_relative_path="",
    )
end

function _v22_l1_store_ephemeris!(root::AbstractString, capture)
    capture.capture_outcome == "http_response" || return capture
    relative, checksum = _v22_l1_store_raw!(root, capture.body)
    checksum == capture.body_sha256 || error(
        "V2.2 L1 ephemeris source-object checksum changed before storage",
    )
    return merge(capture, (raw_relative_path=relative,))
end

function _v22_l1_valid_source_token(value)
    value isa AbstractString || return nothing
    token = String(value)
    return !isempty(token) && strip(token) == token ? token : nothing
end

function _v22_l1_finite_number(value)
    value isa Real && !(value isa Bool) || return nothing
    number = try
        Float64(value)
    catch error
        error isa InterruptException && rethrow()
        return nothing
    end
    return isfinite(number) && number != V22_L1_EPHEMERIS_MISSING_VALUE ?
           number : nothing
end

function _v22_l1_measurement_target(parsed)
    candidates = NamedTuple[]
    for row in parsed
        raw_time = _v22_l1_object_field(row, :time_tag)
        raw_time isa AbstractString || continue
        timestamp = try
            _v22_l1_parse_source_utc(raw_time)
        catch error
            error isa InterruptException && rethrow()
            continue
        end
        push!(candidates, (
            timestamp=timestamp,
            source=_v22_l1_valid_source_token(_v22_l1_object_field(row, :source)),
            active=_v22_l1_object_field(row, :active),
            row=row,
        ))
    end
    isempty(candidates) && return (status="missing_measurement_time", target=nothing)
    latest_time = maximum(candidate.timestamp for candidate in candidates)
    latest = filter(candidate -> candidate.timestamp == latest_time, candidates)
    length(latest) == 1 || return (
        status="ambiguous_latest_measurement_time", target=nothing,
    )
    target = only(latest)
    target.source === nothing && return (
        status="latest_measurement_source_unbound", target=nothing,
    )
    target.active isa Bool || return (
        status="latest_measurement_active_unbound", target=nothing,
    )
    target.active || return (
        status="latest_measurement_source_not_active", target=nothing,
    )
    return (status="bound", target=target)
end

function _v22_l1_bounded_measurement(row, key::Symbol,
                                     lower::Float64, upper::Float64)
    value = _v22_l1_finite_number(_v22_l1_object_field(row, key))
    value === nothing && return nothing
    return lower <= value <= upper ? value : nothing
end

function _v22_l1_quality_binding(source_url::AbstractString,
                                 measurement_target)
    default = (
        authority_url=V22_L1_NOAA_QUALITY_AUTHORITY_URL,
        source_product="",
        row_timestamp_utc="",
        row_source="",
        value=-1,
        binding_status="missing_documented_per_row_quality",
        required_fields_status="unverified",
        decision="reject_missing_authoritative_per_row_quality",
        accepted=false,
    )
    measurement_target === nothing && return default
    target = merge(default, (
        row_timestamp_utc=_v22_l1_utc(measurement_target.timestamp),
        row_source=String(measurement_target.source),
    ))

    url = String(source_url)
    if url == V22_L1_RECEIPT_SOURCES[1].url
        authority_url = V22_L1_NOAA_MAG_QUALITY_SCHEMA_URL
        product = "dscovr_m1m"
        fields_valid = all(key -> _v22_l1_bounded_measurement(
            measurement_target.row, key, -1.0e3, 1.0e3,
        ) !== nothing, (:bx_gsm, :by_gsm, :bz_gsm))
        fields_status = fields_valid ? "bound_required_bx_by_bz_gsm" :
                        "missing_or_invalid_required_bx_by_bz_gsm"
    elseif url == V22_L1_RECEIPT_SOURCES[2].url
        authority_url = V22_L1_NOAA_WIND_QUALITY_SCHEMA_URL
        product = "dscovr_f1m"
        speed_valid = _v22_l1_bounded_measurement(
            measurement_target.row, :proton_speed, 50.0, 5.0e3,
        ) !== nothing
        density_valid = _v22_l1_bounded_measurement(
            measurement_target.row, :proton_density, 0.0, 1.0e3,
        ) !== nothing
        vx_gse_valid = _v22_l1_bounded_measurement(
            measurement_target.row, :proton_vx_gse, -5.0e3, 5.0e3,
        ) !== nothing
        fields_valid = speed_valid && density_valid && vx_gse_valid
        fields_status = fields_valid ? "bound_required_speed_density_vx_gse" :
                        "missing_or_invalid_required_speed_density_vx_gse"
    else
        return merge(target, (
            binding_status="untrusted_endpoint_quality_semantics",
            decision="reject_untrusted_endpoint_quality_semantics",
        ))
    end
    source_binding = merge(target, (
        authority_url=authority_url,
        source_product=product,
        required_fields_status=fields_status,
    ))
    measurement_target.source == "DSCOVR" || return merge(source_binding, (
        binding_status="unverified_non_dscovr_quality_semantics",
        decision="reject_unverified_source_quality_semantics",
    ))

    value = _v22_l1_object_field(measurement_target.row, :overall_quality)
    number = _v22_l1_finite_number(value)
    if number === nothing || !(0.0 <= number <= 2.0) || !isinteger(number)
        missing = value === nothing
        return merge(source_binding, (
            binding_status=value === nothing ?
                           "missing_noaa_overall_quality" :
                           "invalid_noaa_overall_quality",
            decision=missing ? "reject_missing_noaa_overall_quality" :
                               "reject_invalid_noaa_overall_quality",
        ))
    end
    quality = Int(number)

    decision = if quality == 1
        "reject_suspect_overall_quality"
    elseif quality == 2
        "reject_error_overall_quality"
    elseif !fields_valid
        "reject_missing_or_invalid_required_forecast_fields"
    else
        "accept_normal_overall_quality"
    end
    return merge(source_binding, (
        value=quality,
        binding_status="bound_noaa_dscovr_overall_quality",
        decision=decision,
        accepted=quality == 0 && fields_valid,
    ))
end

function _v22_l1_ephemeris_position(capture, measurement_target,
                                    measurement_request_started::DateTime,
                                    measurement_receipt_completed::DateTime,
                                    measurement_monotonic_started_ns::Int)
    capture.capture_outcome == "http_response" || return (
        status="ephemeris_$(capture.capture_outcome)", record=nothing,
        available_before_issue=false,
    )
    capture.http_status == 200 || return (
        status="ephemeris_http_status_not_200", record=nothing,
        available_before_issue=false,
    )
    ephemeris_receipt = _v22_l1_parse_utc(capture.receipt_completed_utc)
    available = ephemeris_receipt <= measurement_request_started &&
                capture.monotonic_completed_ns <= measurement_monotonic_started_ns
    available || return (
        status="ephemeris_not_available_before_issue", record=nothing,
        available_before_issue=false,
    )
    measurement_target.timestamp <= measurement_receipt_completed || return (
        status="measurement_time_after_issue", record=nothing,
        available_before_issue=true,
    )

    parsed = try
        JSON3.read(String(copy(capture.body)))
    catch error
        error isa InterruptException && rethrow()
        nothing
    end
    parsed isa AbstractVector || return (
        status="ephemeris_source_object_not_json_array", record=nothing,
        available_before_issue=true,
    )
    rows = NamedTuple[]
    for row in parsed
        _v22_l1_valid_source_token(_v22_l1_object_field(row, :source)) ==
            measurement_target.source || continue
        _v22_l1_object_field(row, :active) === true || continue
        raw_time = _v22_l1_object_field(row, :time_tag)
        raw_time isa AbstractString || continue
        timestamp = try
            _v22_l1_parse_source_utc(raw_time)
        catch error
            error isa InterruptException && rethrow()
            continue
        end
        x = _v22_l1_finite_number(_v22_l1_object_field(row, :x_gse))
        y = _v22_l1_finite_number(_v22_l1_object_field(row, :y_gse))
        z = _v22_l1_finite_number(_v22_l1_object_field(row, :z_gse))
        any(isnothing, (x, y, z)) && continue
        push!(rows, (timestamp=timestamp, x=x, y=y, z=z))
    end
    isempty(rows) && return (
        status="missing_same_source_active_gse_ephemeris", record=nothing,
        available_before_issue=true,
    )
    sort!(rows; by=row -> row.timestamp)
    length(unique(row.timestamp for row in rows)) == length(rows) || return (
        status="duplicate_same_source_ephemeris_time", record=nothing,
        available_before_issue=true,
    )

    target_time = measurement_target.timestamp
    exact = findfirst(row -> row.timestamp == target_time, rows)
    if exact !== nothing
        lower = upper = rows[exact]
        fraction = 0.0
        method = "exact"
    else
        lower_index = findlast(row -> row.timestamp < target_time, rows)
        upper_index = findfirst(row -> row.timestamp > target_time, rows)
        (lower_index !== nothing && upper_index !== nothing) || return (
            status="ephemeris_extrapolation_required", record=nothing,
            available_before_issue=true,
        )
        lower = rows[lower_index]
        upper = rows[upper_index]
        span_ms = Dates.value(upper.timestamp - lower.timestamp)
        0 < span_ms <= Dates.value(Millisecond(Hour(1))) || return (
            status="ephemeris_bracket_exceeds_one_hour", record=nothing,
            available_before_issue=true,
        )
        fraction = Dates.value(target_time - lower.timestamp) / span_ms
        0.0 < fraction < 1.0 || error(
            "V2.2 L1 internal ephemeris interpolation fraction is invalid",
        )
        method = "linear"
    end
    interpolate(a, b) = a + fraction * (b - a)
    record = (
        source=measurement_target.source,
        position_time_utc=_v22_l1_utc(target_time),
        position_frame=V22_L1_EPHEMERIS_POSITION_FRAME,
        position_units=V22_L1_EPHEMERIS_POSITION_UNITS,
        method=method,
        lower_time_utc=_v22_l1_utc(lower.timestamp),
        upper_time_utc=_v22_l1_utc(upper.timestamp),
        interpolation_fraction=fraction,
        x_gse=interpolate(lower.x, upper.x),
        y_gse=interpolate(lower.y, upper.y),
        z_gse=interpolate(lower.z, upper.z),
    )
    return (
        status="bound_issue_causal_swpc_ephemeris",
        record=record,
        available_before_issue=true,
    )
end

function _v22_l1_metadata_payload(metadata)
    return (
        metadata_contract_version=String(metadata.metadata_contract_version),
        identity_authority_url=String(metadata.identity_authority_url),
        source_field_semantics=String(metadata.source_field_semantics),
        source_tokens=String.(metadata.source_tokens),
        source_rows=Int(metadata.source_rows),
        identity_status=String(metadata.identity_status),
        active_field_semantics=String(metadata.active_field_semantics),
        active_source_tokens=String.(metadata.active_source_tokens),
        active_boolean_rows=Int(metadata.active_boolean_rows),
        active_status=String(metadata.active_status),
        quality_authority_url=String(metadata.quality_authority_url),
        quality_operational_context_url=String(
            metadata.quality_operational_context_url,
        ),
        quality_semantics=String(metadata.quality_semantics),
        quality_source_product=String(metadata.quality_source_product),
        quality_row_timestamp_utc=String(metadata.quality_row_timestamp_utc),
        quality_row_source=String(metadata.quality_row_source),
        quality_value=Int(metadata.quality_value),
        quality_required_fields_status=String(
            metadata.quality_required_fields_status,
        ),
        archive_quality_authority_url=String(metadata.archive_quality_authority_url),
        archive_quality_semantics=String(metadata.archive_quality_semantics),
        quality_binding_status=String(metadata.quality_binding_status),
        quality_decision=String(metadata.quality_decision),
        archive_quality_transfer_status=String(metadata.archive_quality_transfer_status),
        ephemeris_authority_url=String(metadata.ephemeris_authority_url),
        ephemeris_schema_authority_url=String(
            metadata.ephemeris_schema_authority_url,
        ),
        ephemeris_archive_contract_url=String(
            metadata.ephemeris_archive_contract_url,
        ),
        ephemeris_independent_authority_url=String(
            metadata.ephemeris_independent_authority_url,
        ),
        ephemeris_interpolation_rule=String(metadata.ephemeris_interpolation_rule),
        ephemeris_position_frame=String(metadata.ephemeris_position_frame),
        ephemeris_position_units=String(metadata.ephemeris_position_units),
        ephemeris_position_timestamp_utc=String(
            metadata.ephemeris_position_timestamp_utc,
        ),
        ephemeris_capture_outcome=String(metadata.ephemeris_capture_outcome),
        ephemeris_transport_error_type=String(
            metadata.ephemeris_transport_error_type,
        ),
        ephemeris_transport_error_message=String(
            metadata.ephemeris_transport_error_message,
        ),
        ephemeris_request_started_utc=String(
            metadata.ephemeris_request_started_utc,
        ),
        ephemeris_receipt_completed_utc=String(
            metadata.ephemeris_receipt_completed_utc,
        ),
        ephemeris_monotonic_started_ns=Int(
            metadata.ephemeris_monotonic_started_ns,
        ),
        ephemeris_monotonic_completed_ns=Int(
            metadata.ephemeris_monotonic_completed_ns,
        ),
        ephemeris_http_status=Int(metadata.ephemeris_http_status),
        ephemeris_http_date=String(metadata.ephemeris_http_date),
        ephemeris_http_etag=String(metadata.ephemeris_http_etag),
        ephemeris_http_last_modified=String(
            metadata.ephemeris_http_last_modified,
        ),
        ephemeris_source_object_bytes=Int(metadata.ephemeris_source_object_bytes),
        ephemeris_source_object_sha256=String(
            metadata.ephemeris_source_object_sha256,
        ),
        ephemeris_source_object_raw_relative_path=String(
            metadata.ephemeris_source_object_raw_relative_path,
        ),
        ephemeris_source_available_before_issue=Bool(
            metadata.ephemeris_source_available_before_issue,
        ),
        ephemeris_binding_status=String(metadata.ephemeris_binding_status),
        ephemeris_record_sha256=String(metadata.ephemeris_record_sha256),
        ephemeris_record_json=String(metadata.ephemeris_record_json),
        rows_admissible=Bool(metadata.rows_admissible),
        admissibility_blockers=String.(metadata.admissibility_blockers),
    )
end

function _v22_l1_metadata_record(source_tokens, source_rows::Integer,
                                 identity_status::AbstractString,
                                 active_source_tokens,
                                 active_boolean_rows::Integer,
                                 active_status::AbstractString;
                                 quality_binding=nothing,
                                 ephemeris_capture=nothing,
                                 ephemeris_binding=nothing)
    quality = quality_binding === nothing ?
              _v22_l1_quality_binding("", nothing) : quality_binding
    capture = ephemeris_capture === nothing ?
              _v22_l1_ephemeris_capture_unavailable("not_requested") :
              ephemeris_capture
    binding = ephemeris_binding === nothing ? (
        status="missing_bound_ephemeris_record",
        record=nothing,
        available_before_issue=false,
    ) : ephemeris_binding
    record_json = binding.record === nothing ? "" : JSON3.write(binding.record)
    record_sha256 = isempty(record_json) ? V22_L1_RECEIPT_ZERO_SHA256 :
                    _v22_l1_sha256(codeunits(record_json))
    blockers = String[]
    identity_status == "bound_noaa_source_field" ||
        push!(blockers, "spacecraft_identity_not_fully_bound")
    active_status == "bound_noaa_active_field" ||
        push!(blockers, "active_designation_not_fully_bound")
    quality.accepted || push!(blockers, "missing_or_non_normal_row_quality")
    binding.status == "bound_issue_causal_swpc_ephemeris" ||
        push!(blockers, "missing_bound_ephemeris_record")
    admissible = identity_status == "bound_noaa_source_field" &&
                 active_status == "bound_noaa_active_field" &&
                 quality.accepted &&
                 binding.status == "bound_issue_causal_swpc_ephemeris"
    return (
        metadata_contract_version=V22_L1_METADATA_CONTRACT_VERSION,
        identity_authority_url=V22_L1_NOAA_METADATA_AUTHORITY_URL,
        source_field_semantics=V22_L1_SOURCE_FIELD_SEMANTICS,
        source_tokens=sort!(unique!(String.(collect(source_tokens)))),
        source_rows=Int(source_rows),
        identity_status=String(identity_status),
        active_field_semantics=V22_L1_ACTIVE_FIELD_SEMANTICS,
        active_source_tokens=sort!(unique!(String.(collect(active_source_tokens)))),
        active_boolean_rows=Int(active_boolean_rows),
        active_status=String(active_status),
        quality_authority_url=String(quality.authority_url),
        quality_operational_context_url=V22_L1_NOAA_QUALITY_AUTHORITY_URL,
        quality_semantics=V22_L1_QUALITY_SEMANTICS,
        quality_source_product=String(quality.source_product),
        quality_row_timestamp_utc=String(quality.row_timestamp_utc),
        quality_row_source=String(quality.row_source),
        quality_value=Int(quality.value),
        quality_required_fields_status=String(quality.required_fields_status),
        archive_quality_authority_url=V22_L1_NASA_QUALITY_AUTHORITY_URL,
        archive_quality_semantics=V22_L1_ARCHIVE_QUALITY_SEMANTICS,
        quality_binding_status=String(quality.binding_status),
        quality_decision=String(quality.decision),
        archive_quality_transfer_status="not_bound_to_swpc_rows",
        ephemeris_authority_url=V22_L1_NOAA_EPHEMERIS_URL,
        ephemeris_schema_authority_url=V22_L1_NOAA_EPHEMERIS_SCHEMA_URL,
        ephemeris_archive_contract_url=V22_L1_NOAA_EPHEMERIS_ARCHIVE_CONTRACT_URL,
        ephemeris_independent_authority_url=V22_L1_NASA_EPHEMERIS_AUTHORITY_URL,
        ephemeris_interpolation_rule=V22_L1_EPHEMERIS_INTERPOLATION_RULE,
        ephemeris_position_frame=V22_L1_EPHEMERIS_POSITION_FRAME,
        ephemeris_position_units=V22_L1_EPHEMERIS_POSITION_UNITS,
        ephemeris_position_timestamp_utc=binding.record === nothing ? "" :
                                         binding.record.position_time_utc,
        ephemeris_capture_outcome=String(capture.capture_outcome),
        ephemeris_transport_error_type=String(capture.transport_error_type),
        ephemeris_transport_error_message=String(capture.transport_error_message),
        ephemeris_request_started_utc=String(capture.request_started_utc),
        ephemeris_receipt_completed_utc=String(capture.receipt_completed_utc),
        ephemeris_monotonic_started_ns=Int(capture.monotonic_started_ns),
        ephemeris_monotonic_completed_ns=Int(capture.monotonic_completed_ns),
        ephemeris_http_status=Int(capture.http_status),
        ephemeris_http_date=String(capture.http_date),
        ephemeris_http_etag=String(capture.http_etag),
        ephemeris_http_last_modified=String(capture.http_last_modified),
        ephemeris_source_object_bytes=Int(capture.body_bytes),
        ephemeris_source_object_sha256=String(capture.body_sha256),
        ephemeris_source_object_raw_relative_path=String(capture.raw_relative_path),
        ephemeris_source_available_before_issue=Bool(
            binding.available_before_issue,
        ),
        ephemeris_binding_status=String(binding.status),
        ephemeris_record_sha256=record_sha256,
        ephemeris_record_json=record_json,
        rows_admissible=admissible,
        admissibility_blockers=blockers,
    )
end

function _v22_l1_authoritative_rtsw_url(url::AbstractString)
    value = String(url)
    return value == V22_L1_NOAA_EPHEMERIS_URL ||
           any(source -> value == source.url, V22_L1_RECEIPT_SOURCES)
end

function _v22_l1_metadata_diagnostics(body::Vector{UInt8},
                                      source_url::AbstractString;
                                      ephemeris_capture=nothing,
                                      request_started=nothing,
                                      receipt_completed=nothing,
                                      monotonic_started_ns::Integer=0)
    parsed = try
        JSON3.read(String(copy(body)))
    catch error
        error isa InterruptException && rethrow()
        nothing
    end
    parsed isa AbstractVector || return _v22_l1_metadata_record(
        String[], 0, "missing_noaa_source_field",
        String[], 0, "missing_noaa_active_field";
        ephemeris_capture=ephemeris_capture,
    )

    source_tokens = String[]
    active_source_tokens = String[]
    source_rows = 0
    active_boolean_rows = 0
    for row in parsed
        raw_source = _v22_l1_object_field(row, :source)
        source = if raw_source isa AbstractString
            token = String(raw_source)
            !isempty(token) && strip(token) == token ? token : nothing
        else
            nothing
        end
        if source !== nothing
            source_rows += 1
            push!(source_tokens, source)
        end

        active = _v22_l1_object_field(row, :active)
        if active isa Bool
            active_boolean_rows += 1
            active && source !== nothing && push!(active_source_tokens, source)
        end
    end
    row_count = length(parsed)
    identity_status = if row_count > 0 && source_rows == row_count
        "bound_noaa_source_field"
    elseif source_rows > 0
        "partial_noaa_source_field"
    else
        "missing_noaa_source_field"
    end
    active_status = if row_count > 0 && active_boolean_rows == row_count
        "bound_noaa_active_field"
    elseif active_boolean_rows > 0
        "partial_noaa_active_field"
    else
        "missing_noaa_active_field"
    end
    target = _v22_l1_measurement_target(parsed)
    quality_binding = _v22_l1_quality_binding(
        source_url, target.target,
    )
    ephemeris_binding = nothing
    if ephemeris_capture !== nothing
        request_started isa DateTime && receipt_completed isa DateTime || error(
            "V2.2 L1 ephemeris binding requires measurement receipt clocks",
        )
        ephemeris_binding = if target.target === nothing
            (
                status=target.status,
                record=nothing,
                available_before_issue=false,
            )
        else
            _v22_l1_ephemeris_position(
                ephemeris_capture, target.target,
                request_started, receipt_completed,
                Int(monotonic_started_ns),
            )
        end
    end
    metadata = _v22_l1_metadata_record(
        source_tokens, source_rows, identity_status,
        active_source_tokens, active_boolean_rows, active_status;
        quality_binding=quality_binding,
        ephemeris_capture=ephemeris_capture,
        ephemeris_binding=ephemeris_binding,
    )
    _v22_l1_authoritative_rtsw_url(source_url) && return metadata
    return _v22_l1_metadata_record(
        metadata.source_tokens, metadata.source_rows,
        "untrusted_non_noaa_rtsw_endpoint",
        metadata.active_source_tokens, metadata.active_boolean_rows,
        "untrusted_non_noaa_rtsw_endpoint";
        quality_binding=_v22_l1_quality_binding("", target.target),
        ephemeris_capture=ephemeris_capture,
        ephemeris_binding=ephemeris_binding,
    )
end

function _v22_l1_no_response_metadata(ephemeris_capture=nothing)
    ephemeris_binding = ephemeris_capture === nothing ? nothing : (
        status="unavailable_no_measurement_response",
        record=nothing,
        available_before_issue=false,
    )
    return _v22_l1_metadata_record(
        String[], 0, "unavailable_no_http_response",
        String[], 0, "unavailable_no_http_response";
        ephemeris_capture=ephemeris_capture,
        ephemeris_binding=ephemeris_binding,
    )
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
        metadata_provenance=_v22_l1_metadata_payload(record.metadata_provenance),
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

function _v22_l1_reject_existing_source_url_reuse(root::AbstractString,
                                                   sources)
    names_by_url = Dict{String,String}(
        _v22_l1_normalized_source_url(source.url) => String(source.name)
        for source in sources
    )
    latest_directory = _v22_l1_resolve_relative(
        root, "latest", "latest directory",
    )
    ispath(latest_directory) || return nothing
    isdir(latest_directory) && !islink(latest_directory) || error(
        "V2.2 L1 latest path is not a regular directory",
    )
    for entry in readdir(latest_directory)
        endswith(entry, ".json") || error(
            "V2.2 L1 latest directory contains an unexpected entry",
        )
        source_name = entry[1:(end - 5)]
        occursin(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", source_name) || error(
            "V2.2 L1 latest pointer has an unsafe source name",
        )
        latest = _v22_l1_read_latest(root, source_name)
        record_path = _v22_l1_resolve_relative(
            root, latest.record_relative_path, "latest record path",
        )
        isfile(record_path) && !islink(record_path) || error(
            "V2.2 L1 latest record is missing or not regular",
        )
        record = JSON3.read(read(record_path, String))
        String(record.source_name) == source_name || error(
            "V2.2 L1 latest record changed source identity",
        )
        _v22_l1_record_sha256(record) == String(record.record_sha256) ==
            latest.record_sha256 || error(
            "V2.2 L1 latest record checksum mismatch",
        )
        identity = _v22_l1_normalized_source_url(String(record.source_url))
        prior_name = get(names_by_url, identity, source_name)
        prior_name == source_name || throw(ArgumentError(
            "V2.2 L1 source URL is already bound to $prior_name",
        ))
        names_by_url[identity] = source_name
    end
    return nothing
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
        metadata_provenance=_v22_l1_metadata_payload(evidence.metadata_provenance),
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
                                 monotonic_completed_ns::Int;
                                 ephemeris_capture=nothing)
    status = Int(getproperty(response, :status))
    100 <= status <= 599 || throw(ArgumentError(
        "V2.2 L1 HTTP status must lie between 100 and 599",
    ))
    body = _v22_l1_response_body(response)
    diagnostic = _v22_l1_body_diagnostics(body)
    _v22_l1_metadata_diagnostics(
        body, source.url;
        ephemeris_capture=ephemeris_capture,
        request_started=request_started,
        receipt_completed=receipt_completed,
        monotonic_started_ns=monotonic_started_ns,
    )
    prior = _v22_l1_validate_predecessor(root, source, receipt_completed)
    stored_ephemeris = ephemeris_capture === nothing ? nothing :
                       _v22_l1_store_ephemeris!(root, ephemeris_capture)
    metadata = _v22_l1_metadata_diagnostics(
        body, source.url;
        ephemeris_capture=stored_ephemeris,
        request_started=request_started,
        receipt_completed=receipt_completed,
        monotonic_started_ns=monotonic_started_ns,
    )
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
        metadata_provenance=metadata,
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
                                          monotonic_completed_ns::Int;
                                          ephemeris_capture=nothing)
    prior = _v22_l1_validate_predecessor(root, source, receipt_completed)
    stored_ephemeris = ephemeris_capture === nothing ? nothing :
                       _v22_l1_store_ephemeris!(root, ephemeris_capture)
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
        metadata_provenance=_v22_l1_no_response_metadata(stored_ephemeris),
    )
    prior == _v22_l1_read_latest(root, source.name) || error(
        "V2.2 L1 predecessor changed before transport-error record commit",
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
        ephemeris_http_get::Union{Nothing,Function}=nothing,
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
    _v22_l1_reject_reused_source_urls(validated_sources)
    storage = _v22_l1_validate_root(root; create=true)
    return _v22_l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        _v22_l1_reject_existing_source_url_reuse(
            storage, validated_sources,
        )
        if verify_existing
            for source in validated_sources
                _v22_l1_verify_source(
                    storage, source; require_nonempty=false,
                )
            end
        end
        ephemeris_capture = ephemeris_http_get === nothing ? nothing :
                            _v22_l1_capture_ephemeris(
                                ephemeris_http_get,
                                utc_clock,
                                monotonic_clock,
                            )
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
                    started_ns, completed_ns;
                    ephemeris_capture=ephemeris_capture,
                )
            else
                _v22_l1_install_record!(
                    storage, source, response, started_utc, completed_utc,
                    started_ns, completed_ns;
                    ephemeris_capture=ephemeris_capture,
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

function _v22_l1_ephemeris_capture_from_metadata(root::AbstractString,
                                                  metadata,
                                                  raw_paths::Set{String})
    outcome = String(metadata.ephemeris_capture_outcome)
    outcome == "not_requested" && return nothing
    request_started = _v22_l1_parse_utc(metadata.ephemeris_request_started_utc)
    receipt_completed = _v22_l1_parse_utc(metadata.ephemeris_receipt_completed_utc)
    receipt_completed >= request_started || error(
        "V2.2 L1 ephemeris receipt completion precedes request start",
    )
    monotonic_started = Int(metadata.ephemeris_monotonic_started_ns)
    monotonic_completed = Int(metadata.ephemeris_monotonic_completed_ns)
    monotonic_completed >= monotonic_started || error(
        "V2.2 L1 ephemeris monotonic completion precedes request start",
    )

    if outcome == "transport_error"
        isempty(String(metadata.ephemeris_transport_error_type)) && error(
            "V2.2 L1 ephemeris transport error omits its exception type",
        )
        Int(metadata.ephemeris_http_status) == 0 &&
            isempty(String(metadata.ephemeris_http_date)) &&
            isempty(String(metadata.ephemeris_http_etag)) &&
            isempty(String(metadata.ephemeris_http_last_modified)) &&
            Int(metadata.ephemeris_source_object_bytes) == 0 &&
            String(metadata.ephemeris_source_object_sha256) ==
                V22_L1_RECEIPT_ZERO_SHA256 &&
            isempty(String(metadata.ephemeris_source_object_raw_relative_path)) ||
            error("V2.2 L1 ephemeris transport-error evidence is inconsistent")
        return (
            capture_outcome=outcome,
            transport_error_type=String(metadata.ephemeris_transport_error_type),
            transport_error_message=String(
                metadata.ephemeris_transport_error_message,
            ),
            request_started_utc=String(metadata.ephemeris_request_started_utc),
            receipt_completed_utc=String(metadata.ephemeris_receipt_completed_utc),
            monotonic_started_ns=monotonic_started,
            monotonic_completed_ns=monotonic_completed,
            http_status=0,
            http_date=String(metadata.ephemeris_http_date),
            http_etag=String(metadata.ephemeris_http_etag),
            http_last_modified=String(metadata.ephemeris_http_last_modified),
            body=UInt8[],
            body_bytes=0,
            body_sha256=V22_L1_RECEIPT_ZERO_SHA256,
            raw_relative_path="",
        )
    end
    outcome == "http_response" || error(
        "V2.2 L1 ephemeris capture has an unknown outcome",
    )
    isempty(String(metadata.ephemeris_transport_error_type)) &&
        isempty(String(metadata.ephemeris_transport_error_message)) || error(
            "V2.2 L1 ephemeris HTTP response contains transport-error metadata",
        )
    status = Int(metadata.ephemeris_http_status)
    100 <= status <= 599 || error(
        "V2.2 L1 ephemeris HTTP response has an invalid status",
    )
    checksum = String(metadata.ephemeris_source_object_sha256)
    occursin(r"^[0-9a-f]{64}$", checksum) || error(
        "V2.2 L1 ephemeris source-object checksum is malformed",
    )
    expected_relative = joinpath("raw", checksum[1:2], checksum * ".raw")
    relative = normpath(String(
        metadata.ephemeris_source_object_raw_relative_path,
    ))
    relative == expected_relative || error(
        "V2.2 L1 ephemeris source-object path is not content addressed",
    )
    path = _v22_l1_resolve_relative(root, relative, "ephemeris raw path")
    isfile(path) && !islink(path) || error(
        "V2.2 L1 ephemeris source object is missing or not regular",
    )
    body = read(path)
    _v22_l1_sha256(body) == checksum || error(
        "V2.2 L1 ephemeris source-object checksum mismatch",
    )
    length(body) == Int(metadata.ephemeris_source_object_bytes) || error(
        "V2.2 L1 ephemeris source-object size mismatch",
    )
    push!(raw_paths, relative)
    return (
        capture_outcome=outcome,
        transport_error_type="",
        transport_error_message="",
        request_started_utc=String(metadata.ephemeris_request_started_utc),
        receipt_completed_utc=String(metadata.ephemeris_receipt_completed_utc),
        monotonic_started_ns=monotonic_started,
        monotonic_completed_ns=monotonic_completed,
        http_status=status,
        http_date=String(metadata.ephemeris_http_date),
        http_etag=String(metadata.ephemeris_http_etag),
        http_last_modified=String(metadata.ephemeris_http_last_modified),
        body=body,
        body_bytes=length(body),
        body_sha256=checksum,
        raw_relative_path=relative,
    )
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
    ephemeris_capture = _v22_l1_ephemeris_capture_from_metadata(
        root, record.metadata_provenance, raw_paths,
    )
    _v22_l1_metadata_payload(record.metadata_provenance) ==
        _v22_l1_metadata_diagnostics(
            bytes, record.source_url;
            ephemeris_capture=ephemeris_capture,
            request_started=_v22_l1_parse_utc(record.request_started_utc),
            receipt_completed=_v22_l1_parse_utc(record.receipt_completed_utc),
            monotonic_started_ns=Int(record.monotonic_started_ns),
        ) || error(
            "V2.2 L1 raw-response metadata provenance changed",
        )
    push!(raw_paths, raw_relative)
    return nothing
end

function _v22_l1_verify_transport_error(root::AbstractString, record,
                                        raw_paths::Set{String})
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
    ephemeris_capture = _v22_l1_ephemeris_capture_from_metadata(
        root, record.metadata_provenance, raw_paths,
    )
    _v22_l1_metadata_payload(record.metadata_provenance) ==
        _v22_l1_no_response_metadata(ephemeris_capture) || error(
            "V2.2 L1 transport-error metadata provenance is inconsistent",
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
            _v22_l1_verify_transport_error(root, record, raw_paths)
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
    _v22_l1_reject_reused_source_urls(validated_sources)

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
            root;
            ephemeris_http_get=HTTP.get,
            verify_existing=first_cycle,
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
