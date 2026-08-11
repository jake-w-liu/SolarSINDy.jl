#!/usr/bin/env julia

module V22ProspectiveIssueCapture

using Dates
using HTTP
using JSON3
using SHA

include(joinpath(@__DIR__, "v2_2_l1_receipt_pairing.jl"))
const L1 = V22L1ReceiptPairing

export V22_DST_SOURCE, capture_v2_2_dst_receipt!,
       capture_v2_2_prospective_inputs!,
       capture_v2_2_research_issue!, verify_v2_2_research_issue,
       verify_v2_2_research_issues, next_v2_2_issue_time,
       run_v2_2_research_capture_scheduler!

const V22_DST_RECEIPT_SCHEMA_VERSION = "v2_2_dst_receipt_v1"
const V22_DST_PARSER_VERSION = "swpc_kyoto_dst_json_v1"
const V22_RESEARCH_ISSUE_SCHEMA_VERSION = "v2_2_research_issue_v1"
const V22_DST_SOURCE = (
    name="swpc_kyoto_dst_realtime",
    product_id="kyoto-dst",
    url="https://services.swpc.noaa.gov/products/kyoto-dst.json",
)
const V22_RESEARCH_LEADS_HOURS = (1, 2, 3, 4, 6, 7)
const _ZERO_SHA256 = repeat("0", 64)

_sha256(bytes) = bytes2hex(sha256(bytes))
_utc(time::DateTime) = L1._v22_l1_utc(time)
_parse_utc(value) = L1._v22_l1_parse_utc(value)

function _require_sha(value, field::AbstractString; zero_allowed::Bool=false)
    text = String(value)
    occursin(r"^[0-9a-f]{64}$", text) || throw(ArgumentError(
        "V2.2 $field is not a lowercase SHA-256 digest",
    ))
    !zero_allowed && text == _ZERO_SHA256 && throw(ArgumentError(
        "V2.2 $field must not use the zero digest",
    ))
    return text
end

function _require_grid_issue(issue_time_utc::DateTime)
    Dates.minute(issue_time_utc) in (0, 30) &&
        Dates.second(issue_time_utc) == 0 &&
        Dates.millisecond(issue_time_utc) == 0 || throw(ArgumentError(
            "V2.2 issue time must be an exact UTC half-hour grid point",
        ))
    return issue_time_utc
end

"Return the first exact half-hour issue at or after a scheduler clock reading."
function next_v2_2_issue_time(time_utc::DateTime)
    floored_minute = Dates.minute(time_utc) < 30 ? 0 : 30
    floored = DateTime(
        Dates.year(time_utc), Dates.month(time_utc), Dates.day(time_utc),
        Dates.hour(time_utc), floored_minute,
    )
    return floored == time_utc ? floored : floored + Minute(30)
end

function _headers(response)
    return [(
        name=String(pair isa Pair ? first(pair) : pair[1]),
        value=String(pair isa Pair ? last(pair) : pair[2]),
    ) for pair in getproperty(response, :headers)]
end

_headers_sha256(headers) = _sha256(codeunits(JSON3.write([(
    name=String(row.name), value=String(row.value),
) for row in headers])))

function _dst_raw_relative(checksum::AbstractString)
    digest = _require_sha(checksum, "Dst body checksum")
    return joinpath("dst", "raw", "sha256", digest[1:2], digest * ".bin")
end

function _dst_record_relative(sequence::Integer)
    sequence >= 1 || throw(ArgumentError(
        "V2.2 Dst receipt sequence must be positive",
    ))
    return joinpath(
        "dst", "records", lpad(string(sequence), 20, '0') * ".json",
    )
end

const _DST_PARSER_FIELDS = (
    :parser_decision,
    :parser_rejections,
    :json_valid,
    :array_valid,
    :row_count,
    :provider_observation_time_utc,
    :dst_nt_text,
    :provider_row_sha256,
)

function _dst_parser_payload(parsed)
    return (
        parser_decision=String(parsed.parser_decision),
        parser_rejections=String.(collect(parsed.parser_rejections)),
        json_valid=Bool(parsed.json_valid),
        array_valid=Bool(parsed.array_valid),
        row_count=Int(parsed.row_count),
        provider_observation_time_utc=String(
            parsed.provider_observation_time_utc,
        ),
        dst_nt_text=String(parsed.dst_nt_text),
        provider_row_sha256=String(parsed.provider_row_sha256),
    )
end

function _dst_rejection(index::Integer, reason::AbstractString)
    return "row[$index]:" * String(reason)
end

function _parse_dst_source_utc(value)
    text = String(value)
    endswith(text, "Z") && return L1._v22_l1_parse_source_utc(text)
    parsed = tryparse(DateTime, text)
    parsed === nothing && throw(ArgumentError(
        "V2.2 Dst source timestamp is invalid",
    ))
    second_precision = Dates.format(parsed, dateformat"yyyy-mm-ddTHH:MM:SS")
    millisecond_precision = Dates.format(
        parsed, dateformat"yyyy-mm-ddTHH:MM:SS.sss",
    )
    text in (second_precision, millisecond_precision) || throw(ArgumentError(
        "V2.2 Dst source timestamp is not canonical",
    ))
    return parsed
end

function _parse_dst_body(body::Vector{UInt8}, status::Integer,
                         receipt_completed::DateTime)
    default = (
        parser_decision="reject_no_http_200_response",
        parser_rejections=String[],
        json_valid=false,
        array_valid=false,
        row_count=0,
        provider_observation_time_utc="",
        dst_nt_text="",
        provider_row_sha256=_ZERO_SHA256,
    )
    status == 200 || return default
    parsed = try
        JSON3.read(String(copy(body)))
    catch error
        error isa InterruptException && rethrow()
        return merge(default, (parser_decision="reject_invalid_json",))
    end
    parsed isa AbstractVector || return merge(default, (
        parser_decision="reject_non_array_json", json_valid=true,
    ))

    rejections = String[]
    candidates = NamedTuple[]
    for (index, row) in enumerate(parsed)
        raw_time = L1._v22_l1_object_field(row, :time_tag)
        if !(raw_time isa AbstractString)
            push!(rejections, _dst_rejection(index, "missing_time_tag"))
            continue
        end
        timestamp = try
            _parse_dst_source_utc(raw_time)
        catch error
            error isa InterruptException && rethrow()
            push!(rejections, _dst_rejection(index, "invalid_time_tag"))
            continue
        end
        if timestamp > receipt_completed
            push!(rejections, _dst_rejection(index, "post_receipt_time_tag"))
            continue
        end
        raw_dst = L1._v22_l1_object_field(row, :dst)
        if !(raw_dst isa Real) || raw_dst isa Bool
            push!(rejections, _dst_rejection(index, "missing_numeric_dst"))
            continue
        end
        dst = try
            Float64(raw_dst)
        catch error
            error isa InterruptException && rethrow()
            push!(rejections, _dst_rejection(index, "unrepresentable_dst"))
            continue
        end
        if !isfinite(dst) || !(-1_000.0 <= dst <= 1_000.0)
            push!(rejections, _dst_rejection(index, "out_of_range_dst"))
            continue
        end
        push!(candidates, (
            timestamp=timestamp,
            dst=dst,
            row_sha256=_sha256(codeunits(JSON3.write(row))),
        ))
    end
    isempty(candidates) && return merge(default, (
        parser_decision="reject_no_admissible_dst_row",
        parser_rejections=rejections,
        json_valid=true,
        array_valid=true,
        row_count=length(parsed),
    ))
    latest_time = maximum(row.timestamp for row in candidates)
    latest = filter(row -> row.timestamp == latest_time, candidates)
    length(latest) == 1 || return merge(default, (
        parser_decision="reject_ambiguous_latest_dst_row",
        parser_rejections=rejections,
        json_valid=true,
        array_valid=true,
        row_count=length(parsed),
    ))
    selected = only(latest)
    return (
        parser_decision="accept_latest_unique_dst_row",
        parser_rejections=rejections,
        json_valid=true,
        array_valid=true,
        row_count=length(parsed),
        provider_observation_time_utc=_utc(selected.timestamp),
        dst_nt_text=repr(selected.dst),
        provider_row_sha256=selected.row_sha256,
    )
end

const _DST_RECORD_FIELDS = (
    :schema_version,
    :source_name,
    :source_url,
    :product_id,
    :parser_version,
    :request_started_utc,
    :receipt_completed_utc,
    :monotonic_started_ns,
    :monotonic_completed_ns,
    :capture_outcome,
    :transport_error_type,
    :transport_error_message,
    :http_status,
    :response_headers,
    :response_headers_sha256,
    :body_bytes,
    :body_sha256,
    :raw_relative_path,
    _DST_PARSER_FIELDS...,
    :sequence,
    :previous_record_relative_path,
    :previous_record_sha256,
    :first_body_receipt_completed_utc,
    :first_body_monotonic_completed_ns,
    :revision_status,
    :revision_ordinal,
    :revision_of_record_relative_path,
    :revision_of_record_sha256,
)

function _dst_record_payload(record)
    headers = [(
        name=String(row.name), value=String(row.value),
    ) for row in record.response_headers]
    return (
        schema_version=String(record.schema_version),
        source_name=String(record.source_name),
        source_url=String(record.source_url),
        product_id=String(record.product_id),
        parser_version=String(record.parser_version),
        request_started_utc=String(record.request_started_utc),
        receipt_completed_utc=String(record.receipt_completed_utc),
        monotonic_started_ns=Int(record.monotonic_started_ns),
        monotonic_completed_ns=Int(record.monotonic_completed_ns),
        capture_outcome=String(record.capture_outcome),
        transport_error_type=String(record.transport_error_type),
        transport_error_message=String(record.transport_error_message),
        http_status=Int(record.http_status),
        response_headers=headers,
        response_headers_sha256=String(record.response_headers_sha256),
        body_bytes=Int(record.body_bytes),
        body_sha256=String(record.body_sha256),
        raw_relative_path=String(record.raw_relative_path),
        _dst_parser_payload(record)...,
        sequence=Int(record.sequence),
        previous_record_relative_path=String(
            record.previous_record_relative_path,
        ),
        previous_record_sha256=String(record.previous_record_sha256),
        first_body_receipt_completed_utc=String(
            record.first_body_receipt_completed_utc,
        ),
        first_body_monotonic_completed_ns=Int(
            record.first_body_monotonic_completed_ns,
        ),
        revision_status=String(record.revision_status),
        revision_ordinal=Int(record.revision_ordinal),
        revision_of_record_relative_path=String(
            record.revision_of_record_relative_path,
        ),
        revision_of_record_sha256=String(record.revision_of_record_sha256),
    )
end

_dst_record_sha256(record) =
    _sha256(codeunits(JSON3.write(_dst_record_payload(record))))

function _read_regular_json(storage::AbstractString,
                            relative::AbstractString,
                            label::AbstractString)
    return JSON3.read(String(_read_regular_bytes(storage, relative, label)))
end

function _same_file_identity(left, right)
    return left.device == right.device && left.inode == right.inode &&
           left.mode == right.mode
end

function _read_regular_bytes(storage::AbstractString,
                             relative::AbstractString,
                             label::AbstractString)
    path = L1._v22_l1_resolve_relative(storage, relative, label)
    isfile(path) && !islink(path) || throw(ArgumentError(
        "V2.2 $label must be a regular non-symlink file",
    ))
    before = lstat(path)
    bytes = open(path, "r") do io
        opened = stat(io)
        _same_file_identity(before, opened) || throw(ArgumentError(
            "V2.2 $label changed while it was opened",
        ))
        value = read(io)
        _same_file_identity(opened, stat(io)) || throw(ArgumentError(
            "V2.2 $label changed while it was read",
        ))
        value
    end
    after = lstat(path)
    _same_file_identity(before, after) &&
        L1._v22_l1_resolve_relative(storage, relative, label) == path ||
        throw(ArgumentError("V2.2 $label changed after it was read"))
    return bytes
end

function _dst_latest(storage::AbstractString)
    relative = joinpath("dst", "latest.json")
    path = L1._v22_l1_resolve_relative(storage, relative, "Dst latest pointer")
    if !ispath(path)
        return (
            sequence=0, record_relative_path="", record_sha256=_ZERO_SHA256,
        )
    end
    latest = _read_regular_json(storage, relative, "Dst latest pointer")
    Set(propertynames(latest)) == Set((
        :schema_version, :source_name, :source_url, :product_id, :sequence,
        :record_relative_path, :record_sha256,
    )) || throw(ArgumentError("V2.2 Dst latest-pointer fields changed"))
    String(latest.schema_version) == V22_DST_RECEIPT_SCHEMA_VERSION ||
        throw(ArgumentError("V2.2 Dst latest-pointer schema changed"))
    String(latest.source_name) == V22_DST_SOURCE.name &&
        String(latest.source_url) == V22_DST_SOURCE.url &&
        String(latest.product_id) == V22_DST_SOURCE.product_id ||
        throw(ArgumentError("V2.2 Dst latest-pointer identity changed"))
    sequence = Int(latest.sequence)
    sequence >= 1 || throw(ArgumentError(
        "V2.2 Dst latest-pointer sequence must be positive",
    ))
    relative_record = String(latest.record_relative_path)
    relative_record == _dst_record_relative(sequence) || throw(ArgumentError(
        "V2.2 Dst latest-pointer path changed",
    ))
    checksum = _require_sha(latest.record_sha256, "Dst latest record checksum")
    return (
        sequence=sequence,
        record_relative_path=relative_record,
        record_sha256=checksum,
    )
end

function _dst_latest_payload(record)
    return (
        schema_version=V22_DST_RECEIPT_SCHEMA_VERSION,
        source_name=V22_DST_SOURCE.name,
        source_url=V22_DST_SOURCE.url,
        product_id=V22_DST_SOURCE.product_id,
        sequence=Int(record.sequence),
        record_relative_path=_dst_record_relative(Int(record.sequence)),
        record_sha256=String(record.record_sha256),
    )
end

function _write_dst_latest!(storage::AbstractString, record)
    L1._v22_l1_atomic_text(
        storage, joinpath("dst", "latest.json"),
        JSON3.write(_dst_latest_payload(record)); replace=true,
    )
    return nothing
end

function _record_names(storage::AbstractString, relative::AbstractString)
    directory = L1._v22_l1_resolve_relative(storage, relative, "record directory")
    !ispath(directory) && return String[]
    isdir(directory) && !islink(directory) || throw(ArgumentError(
        "V2.2 record directory must be a regular directory",
    ))
    names = readdir(directory)
    for name in names
        path = joinpath(directory, name)
        isfile(path) && !islink(path) || throw(ArgumentError(
            "V2.2 record directory contains a non-regular entry",
        ))
    end
    return sort!(names)
end

function _verify_dst_prefix(storage::AbstractString, head_sequence::Integer,
                            head_sha256::AbstractString;
                            require_complete::Bool=false)
    count = Int(head_sequence)
    count >= 0 || throw(ArgumentError("V2.2 Dst cutoff sequence is negative"))
    if count == 0
        String(head_sha256) == _ZERO_SHA256 || throw(ArgumentError(
            "V2.2 empty Dst prefix has a nonzero checksum",
        ))
        if require_complete
            isempty(_record_names(storage, joinpath("dst", "records"))) ||
                throw(ArgumentError("V2.2 Dst archive has orphan receipt records"))
        end
        return (
            records=NamedTuple[],
            latest_by_observation=Dict{String,Any}(),
            first_by_body=Dict{String,Tuple{String,Int}}(),
        )
    end
    _require_sha(head_sha256, "Dst cutoff checksum")
    records = NamedTuple[]
    first_body = Dict{String,Tuple{String,Int}}()
    latest_by_observation = Dict{String,Any}()
    expected_previous_path = ""
    expected_previous_sha = _ZERO_SHA256
    prior_receipt = typemin(DateTime)
    for sequence in 1:count
        relative = _dst_record_relative(sequence)
        record = _read_regular_json(storage, relative, "Dst receipt record")
        Set(propertynames(record)) == Set((_DST_RECORD_FIELDS..., :record_sha256)) ||
            throw(ArgumentError("V2.2 Dst receipt fields changed"))
        String(record.schema_version) == V22_DST_RECEIPT_SCHEMA_VERSION &&
            String(record.source_name) == V22_DST_SOURCE.name &&
            String(record.source_url) == V22_DST_SOURCE.url &&
            String(record.product_id) == V22_DST_SOURCE.product_id &&
            String(record.parser_version) == V22_DST_PARSER_VERSION ||
            throw(ArgumentError("V2.2 Dst receipt identity changed"))
        Int(record.sequence) == sequence || throw(ArgumentError(
            "V2.2 Dst receipt sequence changed",
        ))
        String(record.previous_record_relative_path) == expected_previous_path &&
            String(record.previous_record_sha256) == expected_previous_sha ||
            throw(ArgumentError("V2.2 Dst receipt chain changed"))
        checksum = _require_sha(record.record_sha256, "Dst record checksum")
        _dst_record_sha256(record) == checksum || throw(ArgumentError(
            "V2.2 Dst record checksum mismatch",
        ))
        started = _parse_utc(record.request_started_utc)
        completed = _parse_utc(record.receipt_completed_utc)
        completed >= started && completed >= prior_receipt || throw(ArgumentError(
            "V2.2 Dst receipt clocks regress",
        ))
        Int(record.monotonic_completed_ns) >= Int(record.monotonic_started_ns) ||
            throw(ArgumentError("V2.2 Dst monotonic clock regresses"))
        headers = [(
            begin
                Set(propertynames(row)) == Set((:name, :value)) ||
                    throw(ArgumentError("V2.2 Dst response-header fields changed"))
                (name=String(row.name), value=String(row.value))
            end
        ) for row in record.response_headers]
        _headers_sha256(headers) == String(record.response_headers_sha256) ||
            throw(ArgumentError("V2.2 Dst response-header checksum mismatch"))

        outcome = String(record.capture_outcome)
        if outcome == "http_response"
            isempty(String(record.transport_error_type)) &&
                isempty(String(record.transport_error_message)) ||
                throw(ArgumentError("V2.2 Dst HTTP response has transport error data"))
            status = Int(record.http_status)
            100 <= status <= 599 || throw(ArgumentError(
                "V2.2 Dst HTTP status is invalid",
            ))
            raw_relative = String(record.raw_relative_path)
            body_sha = _require_sha(record.body_sha256, "Dst body checksum")
            raw_relative == _dst_raw_relative(body_sha) || throw(ArgumentError(
                "V2.2 Dst raw-object path changed",
            ))
            body = _read_regular_bytes(storage, raw_relative, "Dst raw object")
            length(body) == Int(record.body_bytes) && _sha256(body) == body_sha ||
                throw(ArgumentError("V2.2 Dst raw object changed"))
            parsed = _parse_dst_body(body, status, completed)
            _dst_parser_payload(parsed) == _dst_parser_payload(record) ||
                throw(ArgumentError("V2.2 Dst parser decision changed"))
            if haskey(first_body, body_sha)
                first_receipt, first_monotonic = first_body[body_sha]
            else
                first_receipt = _utc(completed)
                first_monotonic = Int(record.monotonic_completed_ns)
                first_body[body_sha] = (first_receipt, first_monotonic)
            end
            String(record.first_body_receipt_completed_utc) == first_receipt &&
                Int(record.first_body_monotonic_completed_ns) == first_monotonic ||
                throw(ArgumentError("V2.2 Dst first-body receipt changed"))
        elseif outcome == "transport_error"
            expected_parser = (
                parser_decision="reject_transport_error",
                parser_rejections=String[],
                json_valid=false,
                array_valid=false,
                row_count=0,
                provider_observation_time_utc="",
                dst_nt_text="",
                provider_row_sha256=_ZERO_SHA256,
            )
            Int(record.http_status) == 0 && isempty(headers) &&
                !isempty(String(record.transport_error_type)) &&
                Int(record.body_bytes) == 0 &&
                String(record.body_sha256) == _ZERO_SHA256 &&
                isempty(String(record.raw_relative_path)) &&
                String(record.response_headers_sha256) ==
                    _headers_sha256(NamedTuple[]) &&
                _dst_parser_payload(record) ==
                    _dst_parser_payload(expected_parser) &&
                isempty(String(record.first_body_receipt_completed_utc)) &&
                Int(record.first_body_monotonic_completed_ns) == 0 ||
                throw(ArgumentError("V2.2 Dst transport-error record changed"))
        else
            throw(ArgumentError("V2.2 Dst capture outcome is unknown"))
        end

        observation = String(record.provider_observation_time_utc)
        if String(record.parser_decision) == "accept_latest_unique_dst_row"
            _parse_utc(observation) <= completed || throw(ArgumentError(
                "V2.2 Dst observation is later than receipt",
            ))
            dst = tryparse(Float64, String(record.dst_nt_text))
            dst !== nothing && isfinite(dst) || throw(ArgumentError(
                "V2.2 admitted Dst value is invalid",
            ))
            prior = get(latest_by_observation, observation, nothing)
            if prior === nothing
                expected_status = "first_observation"
                expected_ordinal = 1
                revision_path = ""
                revision_sha = _ZERO_SHA256
            else
                expected_status = prior.dst_nt_text == String(record.dst_nt_text) ?
                                  "unchanged_repeat" : "revised_observation"
                expected_ordinal = prior.revision_ordinal + 1
                revision_path = prior.record_relative_path
                revision_sha = prior.record_sha256
            end
            String(record.revision_status) == expected_status &&
                Int(record.revision_ordinal) == expected_ordinal &&
                String(record.revision_of_record_relative_path) == revision_path &&
                String(record.revision_of_record_sha256) == revision_sha ||
                throw(ArgumentError("V2.2 Dst revision lineage changed"))
            latest_by_observation[observation] = (
                sequence=sequence,
                record_relative_path=relative,
                record_sha256=checksum,
                receipt_completed_utc=_utc(completed),
                provider_observation_time_utc=observation,
                dst_nt_text=String(record.dst_nt_text),
                revision_ordinal=expected_ordinal,
            )
        else
            isempty(observation) && isempty(String(record.dst_nt_text)) &&
                String(record.provider_row_sha256) == _ZERO_SHA256 &&
                String(record.revision_status) == "no_admitted_anchor" &&
                Int(record.revision_ordinal) == 0 &&
                isempty(String(record.revision_of_record_relative_path)) &&
                String(record.revision_of_record_sha256) == _ZERO_SHA256 ||
                throw(ArgumentError("V2.2 rejected Dst lineage changed"))
        end
        push!(records, (
            sequence=sequence,
            record_relative_path=relative,
            record_sha256=checksum,
            receipt_completed_utc=_utc(completed),
            parser_decision=String(record.parser_decision),
            provider_observation_time_utc=observation,
            dst_nt_text=String(record.dst_nt_text),
            revision_ordinal=Int(record.revision_ordinal),
        ))
        expected_previous_path = relative
        expected_previous_sha = checksum
        prior_receipt = completed
    end
    last(records).record_sha256 == String(head_sha256) || throw(ArgumentError(
        "V2.2 Dst cutoff checksum does not match its prefix",
    ))
    if require_complete
        expected_names = [basename(_dst_record_relative(sequence)) for sequence in 1:count]
        _record_names(storage, joinpath("dst", "records")) == expected_names ||
            throw(ArgumentError("V2.2 Dst archive has orphan receipt records"))
    end
    return (
        records=records,
        latest_by_observation=latest_by_observation,
        first_by_body=first_body,
    )
end

const _DST_VERIFICATION_CACHE = Dict{String,Any}()

function _latest_dst_anchor(latest_by_observation)
    selected = nothing
    selected_key = nothing
    for row in values(latest_by_observation)
        key = (_parse_utc(row.provider_observation_time_utc), row.sequence)
        if selected_key === nothing || key > selected_key
            selected = row
            selected_key = key
        end
    end
    return selected
end

function _verify_complete_dst(storage::AbstractString)
    latest = _dst_latest(storage)
    verified = _verify_dst_prefix(
        storage, latest.sequence, latest.record_sha256; require_complete=true,
    )
    if latest.sequence > 0
        last(verified.records).record_relative_path == latest.record_relative_path ||
            throw(ArgumentError("V2.2 Dst latest pointer changed"))
    end
    result = merge(verified, (
        latest=latest,
        latest_anchor=_latest_dst_anchor(verified.latest_by_observation),
    ))
    _DST_VERIFICATION_CACHE[String(storage)] = result
    return result
end

function _verify_complete_dst_cached(storage::AbstractString)
    latest = _dst_latest(storage)
    cached = get(_DST_VERIFICATION_CACHE, String(storage), nothing)
    if cached !== nothing && cached.latest == latest
        if latest.sequence > 0
            head = _verify_dst_record_standalone(
                storage, latest.sequence, latest.record_sha256,
            )
            head.record_relative_path == latest.record_relative_path ||
                throw(ArgumentError("V2.2 cached Dst head path changed"))
        end
        return cached
    end
    return _verify_complete_dst(storage)
end

function _recover_dst_orphan_unlocked(storage::AbstractString)
    latest = _dst_latest(storage)
    orphan_sequence = latest.sequence + 1
    relative = _dst_record_relative(orphan_sequence)
    path = L1._v22_l1_resolve_relative(storage, relative, "next Dst record")
    ispath(path) || return nothing
    orphan = _read_regular_json(storage, relative, "orphan Dst receipt record")
    checksum = _require_sha(orphan.record_sha256, "orphan Dst checksum")
    _verify_dst_prefix(
        storage, orphan_sequence, checksum; require_complete=true,
    )
    _write_dst_latest!(storage, orphan)
    _verify_complete_dst(storage)
    return orphan
end

function _store_dst_raw!(storage::AbstractString, body::Vector{UInt8})
    checksum = _sha256(body)
    relative = _dst_raw_relative(checksum)
    path = L1._v22_l1_resolve_relative(storage, relative, "Dst raw object")
    if ispath(path)
        _sha256(_read_regular_bytes(storage, relative, "Dst raw object")) ==
            checksum ||
            throw(ArgumentError("V2.2 Dst content-addressed object changed"))
    else
        L1._v22_l1_atomic_bytes(storage, relative, body)
    end
    return relative, checksum
end

function _capture_dst_unlocked(storage::AbstractString;
                               http_get::Function,
                               utc_clock::Function,
                               monotonic_clock::Function,
                               crash_hook::Function)
    verified = _verify_complete_dst_cached(storage)
    started_utc = utc_clock()
    started_ns = Int(monotonic_clock())
    response = nothing
    transport_error = nothing
    try
        response = http_get(
            V22_DST_SOURCE.url;
            connect_timeout=15,
            readtimeout=30,
            retries=0,
            status_exception=false,
        )
    catch error
        error isa InterruptException && rethrow()
        transport_error = error
    end
    completed_ns = Int(monotonic_clock())
    completed_utc = utc_clock()
    completed_utc >= started_utc && completed_ns >= started_ns ||
        throw(ArgumentError("V2.2 Dst receipt clocks regress"))

    if transport_error === nothing
        status = Int(getproperty(response, :status))
        100 <= status <= 599 || throw(ArgumentError(
            "V2.2 Dst HTTP status must lie between 100 and 599",
        ))
        headers = _headers(response)
        body = L1._v22_l1_response_body(response)
        raw_relative, body_sha = _store_dst_raw!(storage, body)
        parsed = _parse_dst_body(body, status, completed_utc)
        capture = (
            capture_outcome="http_response",
            transport_error_type="",
            transport_error_message="",
            http_status=status,
            response_headers=headers,
            response_headers_sha256=_headers_sha256(headers),
            body_bytes=length(body),
            body_sha256=body_sha,
            raw_relative_path=raw_relative,
            parsed=parsed,
        )
    else
        capture = (
            capture_outcome="transport_error",
            transport_error_type=string(typeof(transport_error)),
            transport_error_message=sprint(showerror, transport_error),
            http_status=0,
            response_headers=NamedTuple[],
            response_headers_sha256=_headers_sha256(NamedTuple[]),
            body_bytes=0,
            body_sha256=_ZERO_SHA256,
            raw_relative_path="",
            parsed=(
                parser_decision="reject_transport_error",
                parser_rejections=String[],
                json_valid=false,
                array_valid=false,
                row_count=0,
                provider_observation_time_utc="",
                dst_nt_text="",
                provider_row_sha256=_ZERO_SHA256,
            ),
        )
    end

    if capture.capture_outcome == "http_response"
        first = get(verified.first_by_body, capture.body_sha256, nothing)
        if first === nothing
            first_receipt = _utc(completed_utc)
            first_monotonic = completed_ns
        else
            first_receipt, first_monotonic = first
        end
    else
        first_receipt = ""
        first_monotonic = 0
    end

    observation = String(capture.parsed.provider_observation_time_utc)
    prior_revision = isempty(observation) ? nothing :
                     get(verified.latest_by_observation, observation, nothing)
    if isempty(observation)
        revision_status = "no_admitted_anchor"
        revision_ordinal = 0
        revision_path = ""
        revision_sha = _ZERO_SHA256
    elseif prior_revision === nothing
        revision_status = "first_observation"
        revision_ordinal = 1
        revision_path = ""
        revision_sha = _ZERO_SHA256
    else
        revision_status = prior_revision.dst_nt_text == capture.parsed.dst_nt_text ?
                          "unchanged_repeat" : "revised_observation"
        revision_ordinal = prior_revision.revision_ordinal + 1
        revision_path = prior_revision.record_relative_path
        revision_sha = prior_revision.record_sha256
    end

    previous = verified.latest
    sequence = previous.sequence + 1
    relative = _dst_record_relative(sequence)
    payload = (
        schema_version=V22_DST_RECEIPT_SCHEMA_VERSION,
        source_name=V22_DST_SOURCE.name,
        source_url=V22_DST_SOURCE.url,
        product_id=V22_DST_SOURCE.product_id,
        parser_version=V22_DST_PARSER_VERSION,
        request_started_utc=_utc(started_utc),
        receipt_completed_utc=_utc(completed_utc),
        monotonic_started_ns=started_ns,
        monotonic_completed_ns=completed_ns,
        capture_outcome=capture.capture_outcome,
        transport_error_type=capture.transport_error_type,
        transport_error_message=capture.transport_error_message,
        http_status=capture.http_status,
        response_headers=capture.response_headers,
        response_headers_sha256=capture.response_headers_sha256,
        body_bytes=capture.body_bytes,
        body_sha256=capture.body_sha256,
        raw_relative_path=capture.raw_relative_path,
        _dst_parser_payload(capture.parsed)...,
        sequence=sequence,
        previous_record_relative_path=previous.record_relative_path,
        previous_record_sha256=previous.record_sha256,
        first_body_receipt_completed_utc=first_receipt,
        first_body_monotonic_completed_ns=first_monotonic,
        revision_status=revision_status,
        revision_ordinal=revision_ordinal,
        revision_of_record_relative_path=revision_path,
        revision_of_record_sha256=revision_sha,
    )
    checksum = _dst_record_sha256(payload)
    record = merge(payload, (record_sha256=checksum,))
    L1._v22_l1_atomic_text(storage, relative, JSON3.write(record))
    crash_hook(:after_dst_record)
    _write_dst_latest!(storage, record)
    stored = _read_regular_json(storage, relative, "Dst receipt record")
    _dst_record_sha256(stored) == checksum || throw(ArgumentError(
        "V2.2 Dst record changed immediately after commit",
    ))
    _verify_dst_record_standalone(storage, sequence, checksum)
    push!(verified.records, (
        sequence=sequence,
        record_relative_path=relative,
        record_sha256=checksum,
        receipt_completed_utc=_utc(completed_utc),
        parser_decision=String(capture.parsed.parser_decision),
        provider_observation_time_utc=observation,
        dst_nt_text=String(capture.parsed.dst_nt_text),
        revision_ordinal=revision_ordinal,
    ))
    if capture.capture_outcome == "http_response" &&
       !haskey(verified.first_by_body, capture.body_sha256)
        verified.first_by_body[capture.body_sha256] = (
            first_receipt, first_monotonic,
        )
    end
    accepted_summary = nothing
    if !isempty(observation)
        accepted_summary = (
            sequence=sequence,
            record_relative_path=relative,
            record_sha256=checksum,
            receipt_completed_utc=_utc(completed_utc),
            provider_observation_time_utc=observation,
            dst_nt_text=String(capture.parsed.dst_nt_text),
            revision_ordinal=revision_ordinal,
        )
        verified.latest_by_observation[observation] = accepted_summary
    end
    latest_anchor = verified.latest_anchor
    if accepted_summary !== nothing &&
       (latest_anchor === nothing ||
        (_parse_utc(accepted_summary.provider_observation_time_utc), sequence) >=
        (_parse_utc(latest_anchor.provider_observation_time_utc),
         latest_anchor.sequence))
        latest_anchor = accepted_summary
    end
    updated = (
        records=verified.records,
        latest_by_observation=verified.latest_by_observation,
        first_by_body=verified.first_by_body,
        latest_anchor=latest_anchor,
        latest=(
            sequence=sequence,
            record_relative_path=relative,
            record_sha256=checksum,
        ),
    )
    _DST_VERIFICATION_CACHE[String(storage)] = updated
    return stored
end

"Capture one raw Kyoto-Dst response. This function never runs automatically."
function capture_v2_2_dst_receipt!(root::AbstractString;
        http_get::Function=HTTP.get,
        utc_clock::Function=() -> now(UTC),
        monotonic_clock::Function=time_ns,
        lock_timeout_sec::Real=30.0,
        crash_hook::Function=stage -> nothing)
    storage = L1._v22_l1_validate_root(root; create=true)
    return L1._v22_l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        _assert_cohort_valid(storage)
        recovered = _recover_dst_orphan_unlocked(storage)
        recovered === nothing || return recovered
        _capture_dst_unlocked(
            storage; http_get=http_get, utc_clock=utc_clock,
            monotonic_clock=monotonic_clock, crash_hook=crash_hook,
        )
    end
end

"Capture one L1 and one Kyoto-Dst polling cycle; never called on module load."
function capture_v2_2_prospective_inputs!(root::AbstractString;
        l1_capture!::Function=storage -> L1.capture_v2_2_l1_receipts!(
            storage; http_get=HTTP.get, ephemeris_http_get=HTTP.get,
        ),
        dst_capture!::Function=storage -> capture_v2_2_dst_receipt!(
            storage; http_get=HTTP.get,
        ))
    storage = L1._v22_l1_validate_root(root; create=true)
    L1._v22_l1_with_lock(storage; timeout_sec=30.0) do
        _assert_cohort_valid(storage)
    end
    l1_result = l1_capture!(storage)
    L1._v22_l1_with_lock(storage; timeout_sec=30.0) do
        _assert_cohort_valid(storage)
    end
    dst_result = dst_capture!(storage)
    return (l1=l1_result, dst=dst_result)
end

function _issue_token(issue_time_utc::DateTime)
    _require_grid_issue(issue_time_utc)
    return Dates.format(issue_time_utc, dateformat"yyyymmddTHHMMSS") * "Z"
end

_issue_record_relative(issue_time_utc::DateTime) =
    joinpath("research_issues", "records", _issue_token(issue_time_utc) * ".json")

function _verify_l1_cutoff_heads(storage::AbstractString,
                                 issue::DateTime,
                                 relative::AbstractString)
    expected_relative = L1._v22_l1_issue_cutoff_relative(issue)
    String(relative) == expected_relative || throw(ArgumentError(
        "V2.2 L1 cutoff path does not match its issue",
    ))
    cutoff = _read_regular_json(storage, relative, "L1 issue cutoff")
    Set(propertynames(cutoff)) == Set((
        L1._V22_L1_ISSUE_CUTOFF_FIELDS..., :cutoff_sha256,
    )) || throw(ArgumentError("V2.2 L1 issue-cutoff fields changed"))
    String(cutoff.schema_version) == L1.V22_L1_ISSUE_CUTOFF_SCHEMA_VERSION &&
        String(cutoff.receipt_schema_version) == L1.V22_L1_RECEIPT_SCHEMA_VERSION &&
        _parse_utc(cutoff.issue_time_utc) == issue &&
        String(cutoff.cutoff_relative_path) == expected_relative ||
        throw(ArgumentError("V2.2 L1 issue-cutoff identity changed"))
    checksum = _require_sha(cutoff.cutoff_sha256, "L1 issue-cutoff checksum")
    L1._v22_l1_issue_cutoff_sha256(cutoff) == checksum ||
        throw(ArgumentError("V2.2 L1 issue-cutoff checksum mismatch"))
    length(cutoff.sources) == length(L1.V22_L1_RECEIPT_SOURCES) ||
        throw(ArgumentError("V2.2 L1 issue-cutoff source count changed"))
    prefixes = NamedTuple[]
    for (source, row) in zip(L1.V22_L1_RECEIPT_SOURCES, cutoff.sources)
        Set(propertynames(row)) == Set(L1._V22_L1_ISSUE_CUTOFF_SOURCE_FIELDS) ||
            throw(ArgumentError("V2.2 L1 issue-cutoff source fields changed"))
        String(row.source_name) == source.name &&
            String(row.source_url) == source.url || throw(ArgumentError(
                "V2.2 L1 issue-cutoff source identity changed",
            ))
        sequence = Int(row.sequence)
        sequence >= 1 || throw(ArgumentError(
            "V2.2 L1 issue-cutoff sequence must be positive",
        ))
        record_relative = String(row.record_relative_path)
        record_relative == L1._v22_l1_record_relative(source.name, sequence) ||
            throw(ArgumentError("V2.2 L1 issue-cutoff head path changed"))
        record_sha = _require_sha(
            row.record_sha256, "L1 issue-cutoff head checksum",
        )
        receipt = _parse_utc(row.latest_receipt_completed_utc)
        receipt <= issue || throw(ArgumentError(
            "V2.2 L1 issue-cutoff contains a post-issue receipt",
        ))
        verified = L1._v22_l1_verify_source(
            storage, source;
            require_nonempty=true,
            head_only=true,
            fixed_head=(
                sequence=sequence,
                record_relative_path=record_relative,
                record_sha256=record_sha,
            ),
            require_complete_tree=false,
        )
        verified.latest_receipt_completed_utc == receipt ||
            throw(ArgumentError("V2.2 L1 issue-cutoff head receipt changed"))
        push!(prefixes, (
            source_name=source.name,
            records=sequence,
            latest_record_sha256=record_sha,
            latest_receipt_completed_utc=_utc(receipt),
        ))
    end
    return (
        schema_version=L1.V22_L1_ISSUE_CUTOFF_SCHEMA_VERSION,
        issue_time_utc=_utc(issue),
        cutoff_relative_path=expected_relative,
        cutoff_sha256=checksum,
        prefixes=Tuple(prefixes),
    )
end

function _capture_l1_cutoff_heads_unlocked(storage::AbstractString,
                                           issue::DateTime)
    relative = L1._v22_l1_issue_cutoff_relative(issue)
    path = L1._v22_l1_resolve_relative(storage, relative, "L1 issue cutoff")
    ispath(path) && return _verify_l1_cutoff_heads(storage, issue, relative)
    rows = NamedTuple[]
    for source in L1.V22_L1_RECEIPT_SOURCES
        latest = L1._v22_l1_read_latest(storage, source.name)
        verified = L1._v22_l1_verify_source(
            storage, source;
            require_nonempty=true,
            head_only=true,
            fixed_head=latest,
            require_complete_tree=false,
        )
        receipt = verified.latest_receipt_completed_utc
        receipt <= issue || throw(ArgumentError(
            "V2.2 L1 scheduled cutoff observed a post-issue receipt",
        ))
        push!(rows, (
            source_name=source.name,
            source_url=source.url,
            sequence=Int(latest.sequence),
            record_relative_path=String(latest.record_relative_path),
            record_sha256=String(latest.record_sha256),
            latest_receipt_completed_utc=_utc(receipt),
        ))
    end
    payload = (
        schema_version=L1.V22_L1_ISSUE_CUTOFF_SCHEMA_VERSION,
        receipt_schema_version=L1.V22_L1_RECEIPT_SCHEMA_VERSION,
        issue_time_utc=_utc(issue),
        cutoff_relative_path=relative,
        sources=rows,
    )
    cutoff = merge(payload, (
        cutoff_sha256=L1._v22_l1_issue_cutoff_sha256(payload),
    ))
    L1._v22_l1_atomic_text(storage, relative, JSON3.write(cutoff))
    return _verify_l1_cutoff_heads(storage, issue, relative)
end

const _ISSUE_RECORD_FIELDS = (
    :schema_version,
    :issue_time_utc,
    :issue_sequence,
    :issue_record_relative_path,
    :record_prepared_utc,
    :record_prepared_monotonic_ns,
    :commit_deadline_utc,
    :commit_witness_status,
    :capture_mode,
    :scheduler_pending_record_sha256,
    :scheduler_completion_status,
    :target_times_utc,
    :l1_cutoff_relative_path,
    :l1_cutoff_sha256,
    :l1_pair_schema_version,
    :l1_pair_status,
    :l1_pair_contract_sha256,
    :l1_pair_measurement_time_utc,
    :l1_pair_source,
    :dst_source_name,
    :dst_source_url,
    :dst_product_id,
    :dst_issue_cutoff_relative_path,
    :dst_issue_cutoff_sha256,
    :dst_cutoff_sequence,
    :dst_cutoff_record_relative_path,
    :dst_cutoff_record_sha256,
    :dst_anchor_status,
    :dst_anchor_record_relative_path,
    :dst_anchor_record_sha256,
    :dst_anchor_time_utc,
    :dst_anchor_age_seconds,
    :model_component_status,
    :issuance_status,
    :numeric_forecast_status,
    :previous_issue_record_relative_path,
    :previous_issue_record_sha256,
)

function _issue_payload(record)
    return (
        schema_version=String(record.schema_version),
        issue_time_utc=String(record.issue_time_utc),
        issue_sequence=Int(record.issue_sequence),
        issue_record_relative_path=String(record.issue_record_relative_path),
        record_prepared_utc=String(record.record_prepared_utc),
        record_prepared_monotonic_ns=Int(record.record_prepared_monotonic_ns),
        commit_deadline_utc=String(record.commit_deadline_utc),
        commit_witness_status=String(record.commit_witness_status),
        capture_mode=String(record.capture_mode),
        scheduler_pending_record_sha256=String(
            record.scheduler_pending_record_sha256,
        ),
        scheduler_completion_status=String(
            record.scheduler_completion_status,
        ),
        target_times_utc=String.(collect(record.target_times_utc)),
        l1_cutoff_relative_path=String(record.l1_cutoff_relative_path),
        l1_cutoff_sha256=String(record.l1_cutoff_sha256),
        l1_pair_schema_version=String(record.l1_pair_schema_version),
        l1_pair_status=String(record.l1_pair_status),
        l1_pair_contract_sha256=String(record.l1_pair_contract_sha256),
        l1_pair_measurement_time_utc=String(
            record.l1_pair_measurement_time_utc,
        ),
        l1_pair_source=String(record.l1_pair_source),
        dst_source_name=String(record.dst_source_name),
        dst_source_url=String(record.dst_source_url),
        dst_product_id=String(record.dst_product_id),
        dst_issue_cutoff_relative_path=String(
            record.dst_issue_cutoff_relative_path,
        ),
        dst_issue_cutoff_sha256=String(record.dst_issue_cutoff_sha256),
        dst_cutoff_sequence=Int(record.dst_cutoff_sequence),
        dst_cutoff_record_relative_path=String(
            record.dst_cutoff_record_relative_path,
        ),
        dst_cutoff_record_sha256=String(record.dst_cutoff_record_sha256),
        dst_anchor_status=String(record.dst_anchor_status),
        dst_anchor_record_relative_path=String(
            record.dst_anchor_record_relative_path,
        ),
        dst_anchor_record_sha256=String(record.dst_anchor_record_sha256),
        dst_anchor_time_utc=String(record.dst_anchor_time_utc),
        dst_anchor_age_seconds=Int(record.dst_anchor_age_seconds),
        model_component_status=String(record.model_component_status),
        issuance_status=String(record.issuance_status),
        numeric_forecast_status=String(record.numeric_forecast_status),
        previous_issue_record_relative_path=String(
            record.previous_issue_record_relative_path,
        ),
        previous_issue_record_sha256=String(
            record.previous_issue_record_sha256,
        ),
    )
end

_issue_record_sha256(record) =
    _sha256(codeunits(JSON3.write(_issue_payload(record))))

const V22_INVALID_COHORT_SCHEMA_VERSION = "v2_2_invalid_cohort_v1"
const V22_PENDING_ISSUE_SCHEMA_VERSION = "v2_2_pending_issue_v1"
const V22_ISSUE_COMPLETION_SCHEMA_VERSION = "v2_2_issue_completion_v1"
const _INVALID_COHORT_RELATIVE = joinpath(
    "research_issues", "cohort_invalid.json",
)
const _PENDING_ISSUE_RELATIVE = joinpath(
    "research_issues", "commit_pending.json",
)
const _INVALID_COHORT_FIELDS = (
    :schema_version, :invalid_issue_time_utc,
    :invalid_issue_record_relative_path, :invalid_issue_record_sha256,
    :detected_utc, :reason,
)
const _PENDING_ISSUE_FIELDS = (
    :schema_version, :issue_time_utc, :issue_sequence,
    :pending_record_relative_path, :started_utc, :started_monotonic_ns,
    :previous_issue_record_relative_path, :previous_issue_record_sha256,
)
const _ISSUE_COMPLETION_FIELDS = (
    :schema_version, :issue_time_utc,
    :pending_record_relative_path, :pending_record_sha256,
    :issue_record_relative_path, :issue_record_sha256,
    :completed_utc, :status,
)

function _invalid_cohort_payload(record)
    return NamedTuple{_INVALID_COHORT_FIELDS}(Tuple(
        String(getproperty(record, name)) for name in _INVALID_COHORT_FIELDS
    ))
end

_invalid_cohort_sha256(record) =
    _sha256(codeunits(JSON3.write(_invalid_cohort_payload(record))))

function _pending_issue_payload(record)
    return NamedTuple{_PENDING_ISSUE_FIELDS}(Tuple(
        getproperty(record, name) for name in _PENDING_ISSUE_FIELDS
    ))
end

_pending_issue_sha256(record) =
    _sha256(codeunits(JSON3.write(_pending_issue_payload(record))))

function _issue_completion_payload(record)
    return NamedTuple{_ISSUE_COMPLETION_FIELDS}(Tuple(
        getproperty(record, name) for name in _ISSUE_COMPLETION_FIELDS
    ))
end

_issue_completion_sha256(record) =
    _sha256(codeunits(JSON3.write(_issue_completion_payload(record))))

_issue_completion_relative(issue::DateTime) = joinpath(
    "research_issues", "commit_guards", "completions",
    _issue_token(issue) * ".json",
)

function _pending_archive_relative(issue::DateTime,
                                   checksum::AbstractString)
    digest = _require_sha(checksum, "pending-issue checksum")
    return joinpath(
        "research_issues", "commit_guards", "pending_archive",
        _issue_token(issue) * "." * digest[1:16] * ".json",
    )
end

function _read_pending_issue(storage::AbstractString)
    path = L1._v22_l1_resolve_relative(
        storage, _PENDING_ISSUE_RELATIVE, "pending-issue guard",
    )
    ispath(path) || return nothing
    pending = _read_regular_json(
        storage, _PENDING_ISSUE_RELATIVE, "pending-issue guard",
    )
    Set(propertynames(pending)) == Set((
        _PENDING_ISSUE_FIELDS..., :pending_record_sha256,
    )) || throw(ArgumentError("V2.2 pending-issue guard fields changed"))
    String(pending.schema_version) == V22_PENDING_ISSUE_SCHEMA_VERSION ||
        throw(ArgumentError("V2.2 pending-issue guard schema changed"))
    issue = _require_grid_issue(_parse_utc(pending.issue_time_utc))
    String(pending.pending_record_relative_path) == _PENDING_ISSUE_RELATIVE ||
        throw(ArgumentError("V2.2 pending-issue guard path changed"))
    checksum = _require_sha(
        pending.pending_record_sha256, "pending-issue guard checksum",
    )
    _pending_issue_sha256(pending) == checksum || throw(ArgumentError(
        "V2.2 pending-issue guard checksum changed",
    ))
    Int(pending.issue_sequence) >= 1 || throw(ArgumentError(
        "V2.2 pending-issue sequence must be positive",
    ))
    started = _parse_utc(pending.started_utc)
    issue <= started <= issue + Minute(5) || throw(ArgumentError(
        "V2.2 pending-issue start clock is outside the commitment window",
    ))
    Int(pending.started_monotonic_ns) >= 0 || throw(ArgumentError(
        "V2.2 pending-issue monotonic clock is negative",
    ))
    previous_relative = String(pending.previous_issue_record_relative_path)
    previous_sha = String(pending.previous_issue_record_sha256)
    if Int(pending.issue_sequence) == 1
        isempty(previous_relative) && previous_sha == _ZERO_SHA256 ||
            throw(ArgumentError("V2.2 first pending issue has a predecessor"))
    else
        isempty(previous_relative) && throw(ArgumentError(
            "V2.2 pending issue is missing its predecessor path",
        ))
        _require_sha(previous_sha, "pending-issue predecessor checksum")
    end
    return pending
end

function _read_invalid_cohort(storage::AbstractString)
    path = L1._v22_l1_resolve_relative(
        storage, _INVALID_COHORT_RELATIVE, "invalid-cohort marker",
    )
    ispath(path) || return nothing
    marker = _read_regular_json(
        storage, _INVALID_COHORT_RELATIVE, "invalid-cohort marker",
    )
    Set(propertynames(marker)) == Set((
        _INVALID_COHORT_FIELDS..., :marker_sha256,
    )) || throw(ArgumentError("V2.2 invalid-cohort marker fields changed"))
    String(marker.schema_version) == V22_INVALID_COHORT_SCHEMA_VERSION ||
        throw(ArgumentError("V2.2 invalid-cohort marker schema changed"))
    checksum = _require_sha(marker.marker_sha256,
                            "invalid-cohort marker checksum")
    _invalid_cohort_sha256(marker) == checksum || throw(ArgumentError(
        "V2.2 invalid-cohort marker checksum changed",
    ))
    issue = _require_grid_issue(_parse_utc(marker.invalid_issue_time_utc))
    detected = _parse_utc(marker.detected_utc)
    detected > issue + Minute(5) || throw(ArgumentError(
        "V2.2 invalid-cohort detection is not after the commitment window",
    ))
    String(marker.reason) == "durable_issue_after_five_minute_window" ||
        throw(ArgumentError("V2.2 invalid-cohort reason changed"))
    relative = String(marker.invalid_issue_record_relative_path)
    relative == _issue_record_relative(issue) || throw(ArgumentError(
        "V2.2 invalid-cohort issue path changed",
    ))
    issue_record = _read_regular_json(
        storage, relative, "invalid-cohort bound issue record",
    )
    bound_checksum = _require_sha(
        marker.invalid_issue_record_sha256,
        "invalid-cohort bound issue checksum",
    )
    String(issue_record.issue_time_utc) == _utc(issue) &&
        String(issue_record.issue_record_relative_path) == relative &&
        _require_sha(
            issue_record.issue_record_sha256,
            "invalid-cohort issue record checksum",
        ) == bound_checksum &&
        _issue_record_sha256(issue_record) == bound_checksum ||
        throw(ArgumentError("V2.2 invalid-cohort issue binding changed"))
    return marker
end

function _assert_cohort_valid(storage::AbstractString;
                              allowed_pending_issue::Union{Nothing,DateTime}=nothing,
                              allowed_pending_sha256::Union{Nothing,String}=nothing)
    marker = _read_invalid_cohort(storage)
    marker === nothing || throw(ArgumentError(
        "V2.2 prospective cohort is permanently invalid from issue " *
        String(marker.invalid_issue_time_utc) * "; use a new storage root",
    ))
    pending = _read_pending_issue(storage)
    if pending !== nothing
        allowed = allowed_pending_issue !== nothing &&
            allowed_pending_sha256 !== nothing &&
            _parse_utc(pending.issue_time_utc) == allowed_pending_issue &&
            String(pending.pending_record_sha256) ==
                String(allowed_pending_sha256)
        allowed || throw(ArgumentError(
            "V2.2 prospective cohort has an uncertain pending issue " *
            String(pending.issue_time_utc) * "; use a new storage root",
        ))
    end
    return nothing
end

function _record_invalid_cohort!(storage::AbstractString, issue::DateTime,
                                 issue_record, detected::DateTime,
                                 pending_sha256::AbstractString;
                                 crash_hook::Function=stage -> nothing)
    return L1._v22_l1_with_lock(storage; timeout_sec=30.0) do
        existing = _read_invalid_cohort(storage)
        existing === nothing || return existing
        pending = _read_pending_issue(storage)
        pending === nothing && throw(ArgumentError(
            "V2.2 cannot invalidate a cohort without its pending guard",
        ))
        _parse_utc(pending.issue_time_utc) == issue &&
            String(pending.pending_record_sha256) == pending_sha256 ||
            throw(ArgumentError("V2.2 invalidation pending guard changed"))
        detected > issue + Minute(5) || throw(ArgumentError(
            "V2.2 invalidation clock is not after the commitment window",
        ))
        String(issue_record.issue_time_utc) == _utc(issue) &&
            String(issue_record.issue_record_relative_path) ==
                _issue_record_relative(issue) &&
            _issue_record_sha256(issue_record) ==
                String(issue_record.issue_record_sha256) ||
            throw(ArgumentError("V2.2 invalidation issue record changed"))
        payload = (
            schema_version=V22_INVALID_COHORT_SCHEMA_VERSION,
            invalid_issue_time_utc=_utc(issue),
            invalid_issue_record_relative_path=String(
                issue_record.issue_record_relative_path,
            ),
            invalid_issue_record_sha256=String(
                issue_record.issue_record_sha256,
            ),
            detected_utc=_utc(detected),
            reason="durable_issue_after_five_minute_window",
        )
        marker = merge(payload, (
            marker_sha256=_invalid_cohort_sha256(payload),
        ))
        crash_hook(:before_invalid_cohort_marker)
        L1._v22_l1_atomic_text(
            storage, _INVALID_COHORT_RELATIVE, JSON3.write(marker),
        )
        return _read_invalid_cohort(storage)
    end
end

function _issue_latest(storage::AbstractString)
    relative = joinpath("research_issues", "latest.json")
    path = L1._v22_l1_resolve_relative(storage, relative, "issue latest pointer")
    if !ispath(path)
        return (
            sequence=0, issue_time_utc="", record_relative_path="",
            record_sha256=_ZERO_SHA256,
        )
    end
    latest = _read_regular_json(storage, relative, "issue latest pointer")
    Set(propertynames(latest)) == Set((
        :schema_version, :issue_time_utc, :sequence,
        :record_relative_path, :record_sha256,
    )) || throw(ArgumentError("V2.2 issue latest-pointer fields changed"))
    String(latest.schema_version) == V22_RESEARCH_ISSUE_SCHEMA_VERSION ||
        throw(ArgumentError("V2.2 issue latest-pointer schema changed"))
    sequence = Int(latest.sequence)
    sequence >= 1 || throw(ArgumentError(
        "V2.2 issue latest-pointer sequence must be positive",
    ))
    issue = _parse_utc(latest.issue_time_utc)
    _require_grid_issue(issue)
    relative_record = String(latest.record_relative_path)
    relative_record == _issue_record_relative(issue) || throw(ArgumentError(
        "V2.2 issue latest-pointer path changed",
    ))
    return (
        sequence=sequence,
        issue_time_utc=_utc(issue),
        record_relative_path=relative_record,
        record_sha256=_require_sha(
            latest.record_sha256, "issue latest record checksum",
        ),
    )
end

function _write_issue_latest!(storage::AbstractString, record)
    payload = (
        schema_version=V22_RESEARCH_ISSUE_SCHEMA_VERSION,
        issue_time_utc=String(record.issue_time_utc),
        sequence=Int(record.issue_sequence),
        record_relative_path=String(record.issue_record_relative_path),
        record_sha256=String(record.issue_record_sha256),
    )
    L1._v22_l1_atomic_text(
        storage, joinpath("research_issues", "latest.json"),
        JSON3.write(payload); replace=true,
    )
    return nothing
end

function _archive_pending_issue_unlocked(storage::AbstractString, pending)
    current = _read_pending_issue(storage)
    current === nothing && throw(ArgumentError(
        "V2.2 pending-issue guard disappeared before archival",
    ))
    checksum = String(pending.pending_record_sha256)
    String(current.pending_record_sha256) == checksum || throw(ArgumentError(
        "V2.2 pending-issue guard changed before archival",
    ))
    issue = _parse_utc(current.issue_time_utc)
    destination_relative = _pending_archive_relative(issue, checksum)
    destination = L1._v22_l1_resolve_relative(
        storage, destination_relative, "pending-issue archive",
    )
    source = L1._v22_l1_resolve_relative(
        storage, _PENDING_ISSUE_RELATIVE, "pending-issue guard",
    )
    isfile(source) && !islink(source) || throw(ArgumentError(
        "V2.2 pending-issue guard is not a regular file",
    ))
    before = lstat(source)
    source_bytes = _read_regular_bytes(
        storage, _PENDING_ISSUE_RELATIVE, "pending-issue guard",
    )
    _same_file_identity(before, lstat(source)) || throw(ArgumentError(
        "V2.2 pending-issue guard changed before archival",
    ))
    if ispath(destination)
        _read_regular_bytes(
            storage, destination_relative, "existing pending-issue archive",
        ) == source_bytes || throw(ArgumentError(
            "V2.2 pending-issue archive conflicts with the active guard",
        ))
    else
        L1._v22_l1_atomic_bytes(
            storage, destination_relative, source_bytes,
        )
    end
    _read_regular_bytes(
        storage, destination_relative, "pending-issue archive",
    ) == source_bytes || throw(ArgumentError(
        "V2.2 pending-issue archive changed before activation",
    ))
    L1._v22_l1_resolve_relative(
        storage, _PENDING_ISSUE_RELATIVE, "pending-issue guard",
    ) == source && _same_file_identity(before, lstat(source)) ||
        throw(ArgumentError(
            "V2.2 pending-issue guard changed before removal",
        ))
    Base.Filesystem.unlink(source)
    L1._v22_l1_sync_parent(source)
    return destination_relative
end

function _begin_pending_issue!(storage::AbstractString, issue::DateTime,
                               started::DateTime, monotonic_ns::Integer)
    return L1._v22_l1_with_lock(storage; timeout_sec=30.0) do
        _assert_cohort_valid(storage)
        issue = _require_grid_issue(issue)
        issue <= started <= issue + Minute(5) || throw(ArgumentError(
            "V2.2 pending issue did not start inside its commitment window",
        ))
        Int(monotonic_ns) >= 0 || throw(ArgumentError(
            "V2.2 pending-issue monotonic clock is negative",
        ))
        issue_path = L1._v22_l1_resolve_relative(
            storage, _issue_record_relative(issue), "scheduled issue record",
        )
        !ispath(issue_path) && !islink(issue_path) || throw(ArgumentError(
            "V2.2 scheduler refuses an issue that is already present",
        ))
        latest = _issue_latest(storage)
        if latest.sequence > 0
            latest_issue = _parse_utc(latest.issue_time_utc)
            latest_issue + Minute(30) == issue || throw(ArgumentError(
                "V2.2 pending issue is not the next exact half-hour",
            ))
            latest_record = _verify_issue_header(storage, latest_issue)
            String(latest_record.issue_record_sha256) == latest.record_sha256 ||
                throw(ArgumentError(
                    "V2.2 pending-issue predecessor changed",
                ))
        end
        payload = (
            schema_version=V22_PENDING_ISSUE_SCHEMA_VERSION,
            issue_time_utc=_utc(issue),
            issue_sequence=latest.sequence + 1,
            pending_record_relative_path=_PENDING_ISSUE_RELATIVE,
            started_utc=_utc(started),
            started_monotonic_ns=Int(monotonic_ns),
            previous_issue_record_relative_path=latest.record_relative_path,
            previous_issue_record_sha256=latest.record_sha256,
        )
        pending = merge(payload, (
            pending_record_sha256=_pending_issue_sha256(payload),
        ))
        L1._v22_l1_atomic_text(
            storage, _PENDING_ISSUE_RELATIVE, JSON3.write(pending),
        )
        return _read_pending_issue(storage)
    end
end

function _cancel_uncommitted_pending_issue!(storage::AbstractString,
                                            issue::DateTime,
                                            pending_sha256::AbstractString)
    return L1._v22_l1_with_lock(storage; timeout_sec=30.0) do
        pending = _read_pending_issue(storage)
        pending === nothing && throw(ArgumentError(
            "V2.2 pending-issue guard disappeared during cancellation",
        ))
        _parse_utc(pending.issue_time_utc) == issue &&
            String(pending.pending_record_sha256) == pending_sha256 ||
            throw(ArgumentError("V2.2 canceled pending issue changed"))
        issue_path = L1._v22_l1_resolve_relative(
            storage, _issue_record_relative(issue), "scheduled issue record",
        )
        !ispath(issue_path) && !islink(issue_path) || throw(ArgumentError(
            "V2.2 cannot cancel a pending issue after its record appeared",
        ))
        return _archive_pending_issue_unlocked(storage, pending)
    end
end

function _complete_pending_issue!(storage::AbstractString, issue::DateTime,
                                  pending_sha256::AbstractString,
                                  issue_record, completed::DateTime)
    return L1._v22_l1_with_lock(storage; timeout_sec=30.0) do
        pending = _read_pending_issue(storage)
        pending === nothing && throw(ArgumentError(
            "V2.2 pending-issue guard disappeared before completion",
        ))
        _parse_utc(pending.issue_time_utc) == issue &&
            String(pending.pending_record_sha256) == pending_sha256 ||
            throw(ArgumentError("V2.2 completed pending issue changed"))
        started = _parse_utc(pending.started_utc)
        started <= completed <= issue + Minute(5) || throw(ArgumentError(
            "V2.2 issue completion is outside its commitment window",
        ))
        verified_issue = _verify_issue_header(storage, issue)
        String(verified_issue.issue_record_sha256) ==
            String(issue_record.issue_record_sha256) &&
            String(verified_issue.capture_mode) ==
                "scheduled_fail_closed_guard" &&
            String(verified_issue.scheduler_pending_record_sha256) ==
                pending_sha256 &&
            String(verified_issue.scheduler_completion_status) == "required" ||
            throw(ArgumentError(
                "V2.2 completed issue record changed",
            ))
        relative = _issue_completion_relative(issue)
        payload = (
            schema_version=V22_ISSUE_COMPLETION_SCHEMA_VERSION,
            issue_time_utc=_utc(issue),
            pending_record_relative_path=_PENDING_ISSUE_RELATIVE,
            pending_record_sha256=String(pending.pending_record_sha256),
            issue_record_relative_path=String(
                verified_issue.issue_record_relative_path,
            ),
            issue_record_sha256=String(verified_issue.issue_record_sha256),
            completed_utc=_utc(completed),
            status="durable_within_five_minute_window",
        )
        completion = merge(payload, (
            completion_record_sha256=_issue_completion_sha256(payload),
        ))
        L1._v22_l1_atomic_text(storage, relative, JSON3.write(completion))
        _archive_pending_issue_unlocked(storage, pending)
        return completion
    end
end

function _pair_unavailable(error)
    return error isa ErrorException &&
           occursin(
               "has no exact admitted mag/wind timestamp in the requested causal window",
               sprint(showerror, error),
           )
end

function _select_pair(storage::AbstractString, issue::DateTime, cutoff)
    try
        pair = L1._select_v2_2_l1_issue_pair_unlocked(
            storage, issue, cutoff.cutoff_relative_path,
        )
        return (
            status="available_verified_cutoff_pair",
            checksum=String(pair.pair_contract_sha256),
            measurement_time=String(pair.measurement_time_utc),
            source=String(pair.source),
        )
    catch error
        error isa InterruptException && rethrow()
        _pair_unavailable(error) || rethrow()
        return (
            status="unavailable_no_exact_admitted_pair",
            checksum=_ZERO_SHA256,
            measurement_time="",
            source="",
        )
    end
end

function _dst_cutoff(verified, issue::DateTime)
    selected = nothing
    for index in reverse(eachindex(verified.records))
        row = verified.records[index]
        if _parse_utc(row.receipt_completed_utc) <= issue
            selected = row
            break
        end
    end
    selected === nothing && return (
        sequence=0, record_relative_path="", record_sha256=_ZERO_SHA256,
    )
    return (
        sequence=selected.sequence,
        record_relative_path=selected.record_relative_path,
        record_sha256=selected.record_sha256,
    )
end

function _select_dst_anchor(prefix, issue::DateTime)
    if hasproperty(prefix, :latest) && hasproperty(prefix, :latest_anchor) &&
       prefix.latest.sequence > 0 && prefix.latest_anchor !== nothing &&
       _parse_utc(last(prefix.records).receipt_completed_utc) <= issue
        selected = prefix.latest_anchor
        age_ms = Dates.value(
            issue - _parse_utc(selected.provider_observation_time_utc),
        )
        age_ms >= 0 && age_ms % 1_000 == 0 || throw(ArgumentError(
            "V2.2 Dst anchor age is invalid",
        ))
        return (
            status="available_receipt_causal_anchor",
            record_relative_path=selected.record_relative_path,
            record_sha256=selected.record_sha256,
            anchor_time_utc=selected.provider_observation_time_utc,
            anchor_age_seconds=age_ms ÷ 1_000,
        )
    end
    selected = nothing
    selected_key = nothing
    for row in prefix.records
        row.parser_decision == "accept_latest_unique_dst_row" || continue
        receipt = _parse_utc(row.receipt_completed_utc)
        observation = _parse_utc(row.provider_observation_time_utc)
        receipt <= issue && observation <= issue || continue
        key = (observation, row.sequence)
        if selected_key === nothing || key > selected_key
            selected = row
            selected_key = key
        end
    end
    selected === nothing && return (
        status="unavailable_no_preissue_dst_anchor",
        record_relative_path="",
        record_sha256=_ZERO_SHA256,
        anchor_time_utc="",
        anchor_age_seconds=-1,
    )
    age_ms = Dates.value(issue - _parse_utc(selected.provider_observation_time_utc))
    age_ms >= 0 || throw(ArgumentError("V2.2 Dst anchor is post-issue"))
    age_ms % 1_000 == 0 || throw(ArgumentError(
        "V2.2 Dst anchor age is not an integral second",
    ))
    return (
        status="available_receipt_causal_anchor",
        record_relative_path=selected.record_relative_path,
        record_sha256=selected.record_sha256,
        anchor_time_utc=selected.provider_observation_time_utc,
        anchor_age_seconds=age_ms ÷ 1_000,
    )
end

function _verify_dst_record_standalone(storage::AbstractString,
                                       sequence::Integer,
                                       checksum::AbstractString)
    relative = _dst_record_relative(sequence)
    record = _read_regular_json(storage, relative, "bound Dst receipt record")
    Set(propertynames(record)) == Set((_DST_RECORD_FIELDS..., :record_sha256)) ||
        throw(ArgumentError("V2.2 bound Dst receipt fields changed"))
    String(record.schema_version) == V22_DST_RECEIPT_SCHEMA_VERSION &&
        String(record.source_name) == V22_DST_SOURCE.name &&
        String(record.source_url) == V22_DST_SOURCE.url &&
        String(record.product_id) == V22_DST_SOURCE.product_id &&
        String(record.parser_version) == V22_DST_PARSER_VERSION &&
        Int(record.sequence) == Int(sequence) || throw(ArgumentError(
            "V2.2 bound Dst receipt identity changed",
        ))
    expected_sha = _require_sha(checksum, "bound Dst receipt checksum")
    String(record.record_sha256) == expected_sha &&
        _dst_record_sha256(record) == expected_sha || throw(ArgumentError(
            "V2.2 bound Dst receipt checksum changed",
        ))
    started = _parse_utc(record.request_started_utc)
    receipt = _parse_utc(record.receipt_completed_utc)
    receipt >= started &&
        Int(record.monotonic_completed_ns) >= Int(record.monotonic_started_ns) ||
        throw(ArgumentError("V2.2 bound Dst receipt clocks changed"))
    headers = [begin
        Set(propertynames(row)) == Set((:name, :value)) ||
            throw(ArgumentError("V2.2 bound Dst header fields changed"))
        (name=String(row.name), value=String(row.value))
    end for row in record.response_headers]
    _headers_sha256(headers) == String(record.response_headers_sha256) ||
        throw(ArgumentError("V2.2 bound Dst header checksum changed"))
    outcome = String(record.capture_outcome)
    if outcome == "http_response"
        body = _read_regular_bytes(
            storage, String(record.raw_relative_path), "bound Dst raw object",
        )
        body_sha = _require_sha(record.body_sha256, "bound Dst body checksum")
        String(record.raw_relative_path) == _dst_raw_relative(body_sha) &&
            length(body) == Int(record.body_bytes) && _sha256(body) == body_sha ||
            throw(ArgumentError("V2.2 bound Dst raw object changed"))
        parsed = _parse_dst_body(body, Int(record.http_status), receipt)
        _dst_parser_payload(parsed) == _dst_parser_payload(record) ||
            throw(ArgumentError("V2.2 bound Dst parser decision changed"))
    elseif outcome == "transport_error"
        String(record.parser_decision) == "reject_transport_error" &&
            Int(record.http_status) == 0 &&
            String(record.body_sha256) == _ZERO_SHA256 &&
            isempty(String(record.raw_relative_path)) || throw(ArgumentError(
                "V2.2 bound Dst transport evidence changed",
            ))
    else
        throw(ArgumentError("V2.2 bound Dst outcome changed"))
    end
    return (
        sequence=Int(sequence),
        record_relative_path=relative,
        record_sha256=expected_sha,
        receipt_completed_utc=_utc(receipt),
        previous_record_relative_path=String(
            record.previous_record_relative_path,
        ),
        previous_record_sha256=String(record.previous_record_sha256),
        parser_decision=String(record.parser_decision),
        provider_observation_time_utc=String(
            record.provider_observation_time_utc,
        ),
        revision_ordinal=Int(record.revision_ordinal),
    )
end

const V22_DST_ISSUE_CUTOFF_SCHEMA_VERSION = "v2_2_dst_issue_cutoff_v1"

_dst_issue_cutoff_relative(issue::DateTime) = joinpath(
    "dst", "issue_cutoffs", _issue_token(issue) * ".json",
)

const _DST_ISSUE_CUTOFF_FIELDS = (
    :schema_version, :issue_time_utc, :cutoff_relative_path,
    :source_name, :source_url, :product_id,
    :eligible_sequence, :eligible_record_relative_path,
    :eligible_record_sha256,
    :first_excluded_sequence, :first_excluded_record_relative_path,
    :first_excluded_record_sha256, :first_excluded_receipt_completed_utc,
    :anchor_status, :anchor_record_relative_path, :anchor_record_sha256,
    :anchor_time_utc, :anchor_age_seconds,
)

function _dst_issue_cutoff_payload(cutoff)
    return NamedTuple{_DST_ISSUE_CUTOFF_FIELDS}(Tuple(
        name in (:eligible_sequence, :first_excluded_sequence,
                 :anchor_age_seconds) ? Int(getproperty(cutoff, name)) :
        String(getproperty(cutoff, name))
        for name in _DST_ISSUE_CUTOFF_FIELDS
    ))
end

_dst_issue_cutoff_sha256(cutoff) =
    _sha256(codeunits(JSON3.write(_dst_issue_cutoff_payload(cutoff))))

function _verify_dst_issue_cutoff(storage::AbstractString,
                                  issue::DateTime,
                                  relative::AbstractString)
    expected_relative = _dst_issue_cutoff_relative(issue)
    String(relative) == expected_relative || throw(ArgumentError(
        "V2.2 Dst issue-cutoff path changed",
    ))
    cutoff = _read_regular_json(storage, relative, "Dst issue cutoff")
    Set(propertynames(cutoff)) == Set((
        _DST_ISSUE_CUTOFF_FIELDS..., :cutoff_sha256,
    )) || throw(ArgumentError("V2.2 Dst issue-cutoff fields changed"))
    String(cutoff.schema_version) == V22_DST_ISSUE_CUTOFF_SCHEMA_VERSION &&
        _parse_utc(cutoff.issue_time_utc) == issue &&
        String(cutoff.cutoff_relative_path) == expected_relative &&
        String(cutoff.source_name) == V22_DST_SOURCE.name &&
        String(cutoff.source_url) == V22_DST_SOURCE.url &&
        String(cutoff.product_id) == V22_DST_SOURCE.product_id ||
        throw(ArgumentError("V2.2 Dst issue-cutoff identity changed"))
    checksum = _require_sha(cutoff.cutoff_sha256, "Dst issue-cutoff checksum")
    _dst_issue_cutoff_sha256(cutoff) == checksum || throw(ArgumentError(
        "V2.2 Dst issue-cutoff checksum mismatch",
    ))
    eligible_sequence = Int(cutoff.eligible_sequence)
    if eligible_sequence == 0
        isempty(String(cutoff.eligible_record_relative_path)) &&
            String(cutoff.eligible_record_sha256) == _ZERO_SHA256 ||
            throw(ArgumentError("V2.2 empty Dst eligible cutoff changed"))
        eligible = nothing
    else
        eligible = _verify_dst_record_standalone(
            storage, eligible_sequence, String(cutoff.eligible_record_sha256),
        )
        eligible.record_relative_path ==
            String(cutoff.eligible_record_relative_path) &&
            _parse_utc(eligible.receipt_completed_utc) <= issue ||
            throw(ArgumentError("V2.2 Dst eligible cutoff is noncausal"))
    end
    excluded_sequence = Int(cutoff.first_excluded_sequence)
    if excluded_sequence == 0
        isempty(String(cutoff.first_excluded_record_relative_path)) &&
            String(cutoff.first_excluded_record_sha256) == _ZERO_SHA256 &&
            isempty(String(cutoff.first_excluded_receipt_completed_utc)) ||
            throw(ArgumentError("V2.2 empty Dst exclusion proof changed"))
    else
        excluded_sequence == eligible_sequence + 1 || throw(ArgumentError(
            "V2.2 Dst exclusion proof is not the next sequence",
        ))
        excluded = _verify_dst_record_standalone(
            storage, excluded_sequence,
            String(cutoff.first_excluded_record_sha256),
        )
        excluded.record_relative_path ==
            String(cutoff.first_excluded_record_relative_path) &&
            excluded.receipt_completed_utc ==
                String(cutoff.first_excluded_receipt_completed_utc) &&
            _parse_utc(excluded.receipt_completed_utc) > issue ||
            throw(ArgumentError("V2.2 Dst exclusion proof is not post-issue"))
        if eligible_sequence > 0
            excluded.previous_record_relative_path ==
                String(cutoff.eligible_record_relative_path) &&
                excluded.previous_record_sha256 ==
                    String(cutoff.eligible_record_sha256) ||
                throw(ArgumentError("V2.2 Dst exclusion chain changed"))
        else
            isempty(excluded.previous_record_relative_path) &&
                excluded.previous_record_sha256 == _ZERO_SHA256 ||
                throw(ArgumentError("V2.2 first Dst exclusion origin changed"))
        end
    end
    anchor_status = String(cutoff.anchor_status)
    if anchor_status == "available_receipt_causal_anchor"
        anchor_path = String(cutoff.anchor_record_relative_path)
        path_match = Base.match(r"([0-9]{20})\.json$", anchor_path)
        path_match === nothing && throw(ArgumentError(
            "V2.2 Dst anchor path is not sequence addressed",
        ))
        anchor_sequence = parse(Int, only(path_match.captures))
        anchor_sequence <= eligible_sequence || throw(ArgumentError(
            "V2.2 Dst anchor lies beyond its eligible cutoff",
        ))
        anchor = _verify_dst_record_standalone(
            storage, anchor_sequence, String(cutoff.anchor_record_sha256),
        )
        anchor.record_relative_path == anchor_path &&
            anchor.provider_observation_time_utc == String(cutoff.anchor_time_utc) &&
            anchor.parser_decision == "accept_latest_unique_dst_row" &&
            _parse_utc(anchor.receipt_completed_utc) <= issue ||
            throw(ArgumentError("V2.2 Dst anchor proof changed"))
        age_ms = Dates.value(issue - _parse_utc(cutoff.anchor_time_utc))
        age_ms >= 0 && age_ms % 1_000 == 0 &&
            age_ms ÷ 1_000 == Int(cutoff.anchor_age_seconds) ||
            throw(ArgumentError("V2.2 Dst anchor age changed"))
    elseif anchor_status == "unavailable_no_preissue_dst_anchor"
        isempty(String(cutoff.anchor_record_relative_path)) &&
            String(cutoff.anchor_record_sha256) == _ZERO_SHA256 &&
            isempty(String(cutoff.anchor_time_utc)) &&
            Int(cutoff.anchor_age_seconds) == -1 || throw(ArgumentError(
                "V2.2 unavailable Dst anchor proof changed",
            ))
    else
        throw(ArgumentError("V2.2 Dst anchor status changed"))
    end
    return (
        cutoff_relative_path=expected_relative,
        cutoff_sha256=checksum,
        eligible=(
            sequence=eligible_sequence,
            record_relative_path=String(cutoff.eligible_record_relative_path),
            record_sha256=String(cutoff.eligible_record_sha256),
        ),
        anchor=(
            status=anchor_status,
            record_relative_path=String(cutoff.anchor_record_relative_path),
            record_sha256=String(cutoff.anchor_record_sha256),
            anchor_time_utc=String(cutoff.anchor_time_utc),
            anchor_age_seconds=Int(cutoff.anchor_age_seconds),
        ),
    )
end

function _capture_dst_issue_cutoff_unlocked(storage::AbstractString,
                                            issue::DateTime,
                                            verified)
    relative = _dst_issue_cutoff_relative(issue)
    path = L1._v22_l1_resolve_relative(storage, relative, "Dst issue cutoff")
    ispath(path) && return _verify_dst_issue_cutoff(storage, issue, relative)
    eligible = _dst_cutoff(verified, issue)
    excluded = eligible.sequence < length(verified.records) ?
               verified.records[eligible.sequence + 1] : nothing
    anchor = _select_dst_anchor(verified, issue)
    payload = (
        schema_version=V22_DST_ISSUE_CUTOFF_SCHEMA_VERSION,
        issue_time_utc=_utc(issue),
        cutoff_relative_path=relative,
        source_name=V22_DST_SOURCE.name,
        source_url=V22_DST_SOURCE.url,
        product_id=V22_DST_SOURCE.product_id,
        eligible_sequence=eligible.sequence,
        eligible_record_relative_path=eligible.record_relative_path,
        eligible_record_sha256=eligible.record_sha256,
        first_excluded_sequence=excluded === nothing ? 0 : excluded.sequence,
        first_excluded_record_relative_path=
            excluded === nothing ? "" : excluded.record_relative_path,
        first_excluded_record_sha256=
            excluded === nothing ? _ZERO_SHA256 : excluded.record_sha256,
        first_excluded_receipt_completed_utc=
            excluded === nothing ? "" : excluded.receipt_completed_utc,
        anchor_status=anchor.status,
        anchor_record_relative_path=anchor.record_relative_path,
        anchor_record_sha256=anchor.record_sha256,
        anchor_time_utc=anchor.anchor_time_utc,
        anchor_age_seconds=anchor.anchor_age_seconds,
    )
    record = merge(payload, (
        cutoff_sha256=_dst_issue_cutoff_sha256(payload),
    ))
    L1._v22_l1_atomic_text(storage, relative, JSON3.write(record))
    return _verify_dst_issue_cutoff(storage, issue, relative)
end

function _verify_issue_header(storage::AbstractString,
                              issue_time_utc::DateTime)
    issue = _require_grid_issue(issue_time_utc)
    relative = _issue_record_relative(issue)
    record = _read_regular_json(storage, relative, "research issue record")
    Set(propertynames(record)) == Set((_ISSUE_RECORD_FIELDS...,
                                      :issue_record_sha256)) ||
        throw(ArgumentError("V2.2 research-issue fields changed"))
    String(record.schema_version) == V22_RESEARCH_ISSUE_SCHEMA_VERSION ||
        throw(ArgumentError("V2.2 research-issue schema changed"))
    _parse_utc(record.issue_time_utc) == issue &&
        String(record.issue_record_relative_path) == relative ||
        throw(ArgumentError("V2.2 research-issue identity changed"))
    checksum = _require_sha(record.issue_record_sha256,
                            "research-issue checksum")
    _issue_record_sha256(record) == checksum || throw(ArgumentError(
        "V2.2 research-issue checksum mismatch",
    ))
    return record
end

function _verify_scheduled_issue_terminal(storage::AbstractString,
                                          issue::DateTime, record)
    pending_sha = _require_sha(
        record.scheduler_pending_record_sha256,
        "scheduled pending-record checksum",
    )
    archive_relative = _pending_archive_relative(issue, pending_sha)
    pending = _read_regular_json(
        storage, archive_relative, "scheduled pending-record archive",
    )
    Set(propertynames(pending)) == Set((
        _PENDING_ISSUE_FIELDS..., :pending_record_sha256,
    )) || throw(ArgumentError(
        "V2.2 scheduled pending-record archive fields changed",
    ))
    String(pending.schema_version) == V22_PENDING_ISSUE_SCHEMA_VERSION &&
        _parse_utc(pending.issue_time_utc) == issue &&
        String(pending.pending_record_relative_path) ==
            _PENDING_ISSUE_RELATIVE &&
        String(pending.pending_record_sha256) == pending_sha &&
        _pending_issue_sha256(pending) == pending_sha ||
        throw(ArgumentError(
            "V2.2 scheduled pending-record archive binding changed",
        ))
    Int(pending.issue_sequence) == Int(record.issue_sequence) &&
        String(pending.previous_issue_record_relative_path) ==
            String(record.previous_issue_record_relative_path) &&
        String(pending.previous_issue_record_sha256) ==
            String(record.previous_issue_record_sha256) ||
        throw(ArgumentError(
            "V2.2 scheduled pending-record predecessor changed",
        ))
    started = _parse_utc(pending.started_utc)
    prepared = _parse_utc(record.record_prepared_utc)
    started == prepared || throw(ArgumentError(
        "V2.2 scheduled pending and preparation clocks disagree",
    ))
    pending_monotonic = Int(pending.started_monotonic_ns)
    issue_monotonic = Int(record.record_prepared_monotonic_ns)
    0 <= pending_monotonic <= issue_monotonic || throw(ArgumentError(
        "V2.2 scheduled monotonic clock regressed",
    ))

    completion_relative = _issue_completion_relative(issue)
    completion = _read_regular_json(
        storage, completion_relative, "scheduled issue completion",
    )
    Set(propertynames(completion)) == Set((
        _ISSUE_COMPLETION_FIELDS..., :completion_record_sha256,
    )) || throw(ArgumentError(
        "V2.2 scheduled issue-completion fields changed",
    ))
    String(completion.schema_version) ==
        V22_ISSUE_COMPLETION_SCHEMA_VERSION &&
        _parse_utc(completion.issue_time_utc) == issue &&
        String(completion.pending_record_relative_path) ==
            _PENDING_ISSUE_RELATIVE &&
        String(completion.pending_record_sha256) == pending_sha &&
        String(completion.issue_record_relative_path) ==
            String(record.issue_record_relative_path) &&
        String(completion.issue_record_sha256) ==
            String(record.issue_record_sha256) &&
        String(completion.status) ==
            "durable_within_five_minute_window" ||
        throw(ArgumentError(
            "V2.2 scheduled issue-completion binding changed",
        ))
    completion_sha = _require_sha(
        completion.completion_record_sha256,
        "scheduled issue-completion checksum",
    )
    _issue_completion_sha256(completion) == completion_sha ||
        throw(ArgumentError(
            "V2.2 scheduled issue-completion checksum changed",
        ))
    completed = _parse_utc(completion.completed_utc)
    prepared <= completed <= issue + Minute(5) || throw(ArgumentError(
        "V2.2 scheduled issue completion clock changed",
    ))
    return (pending=pending, completion=completion)
end

function _verify_issue_record(storage::AbstractString,
                              issue_time_utc::DateTime;
                              verify_predecessors::Bool=false,
                              require_scheduled_terminal::Bool=true)
    issue = _require_grid_issue(issue_time_utc)
    record = _verify_issue_header(storage, issue)
    prepared = _parse_utc(record.record_prepared_utc)
    deadline = _parse_utc(record.commit_deadline_utc)
    deadline == issue + Minute(5) && issue <= prepared <= deadline ||
        throw(ArgumentError(
        "V2.2 research-issue preparation clocks changed",
        ))
    String(record.commit_witness_status) ==
        "unavailable_research_capture_only" || throw(ArgumentError(
            "V2.2 research-issue commitment witness status changed",
        ))
    prepared_monotonic = Int(record.record_prepared_monotonic_ns)
    prepared_monotonic >= 0 || throw(ArgumentError(
        "V2.2 research-issue monotonic clock is negative",
    ))
    capture_mode = String(record.capture_mode)
    if capture_mode == "scheduled_fail_closed_guard"
        String(record.scheduler_completion_status) == "required" ||
            throw(ArgumentError(
                "V2.2 scheduled completion status changed",
            ))
        _require_sha(
            record.scheduler_pending_record_sha256,
            "scheduled pending-record checksum",
        )
        require_scheduled_terminal && _verify_scheduled_issue_terminal(
            storage, issue, record,
        )
    elseif capture_mode == "manual_research_capture"
        String(record.scheduler_pending_record_sha256) == _ZERO_SHA256 &&
            String(record.scheduler_completion_status) == "not_applicable" ||
            throw(ArgumentError("V2.2 manual capture status changed"))
    else
        throw(ArgumentError("V2.2 research-issue capture mode changed"))
    end
    String.(collect(record.target_times_utc)) ==
        [_utc(issue + Hour(lead)) for lead in V22_RESEARCH_LEADS_HOURS] ||
        throw(ArgumentError("V2.2 research-issue target clock changed"))
    String(record.dst_source_name) == V22_DST_SOURCE.name &&
        String(record.dst_source_url) == V22_DST_SOURCE.url &&
        String(record.dst_product_id) == V22_DST_SOURCE.product_id ||
        throw(ArgumentError("V2.2 research-issue Dst identity changed"))
    String(record.model_component_status) ==
        "unavailable_no_fitted_gated_v2_2" &&
        String(record.issuance_status) ==
            "research_capture_only_unavailable" &&
        String(record.numeric_forecast_status) == "not_emitted" ||
        throw(ArgumentError("V2.2 unavailable issuance status changed"))

    cutoff = _verify_l1_cutoff_heads(
        storage, issue, String(record.l1_cutoff_relative_path),
    )
    cutoff.cutoff_sha256 == String(record.l1_cutoff_sha256) ||
        throw(ArgumentError("V2.2 research-issue cutoff binding changed"))
    pair = _select_pair(storage, issue, cutoff)
    String(record.l1_pair_schema_version) ==
        L1.V22_L1_ISSUE_PAIR_SCHEMA_VERSION &&
        String(record.l1_pair_status) == pair.status &&
        String(record.l1_pair_contract_sha256) == pair.checksum &&
        String(record.l1_pair_measurement_time_utc) == pair.measurement_time &&
        String(record.l1_pair_source) == pair.source || throw(ArgumentError(
            "V2.2 L1 pair binding changed",
        ))

    dst_cutoff = _verify_dst_issue_cutoff(
        storage, issue, String(record.dst_issue_cutoff_relative_path),
    )
    dst_cutoff.cutoff_sha256 == String(record.dst_issue_cutoff_sha256) ||
        throw(ArgumentError("V2.2 Dst issue-cutoff binding changed"))
    dst_sequence = Int(record.dst_cutoff_sequence)
    dst_checksum = String(record.dst_cutoff_record_sha256)
    dst_cutoff.eligible.sequence == dst_sequence &&
        dst_cutoff.eligible.record_relative_path ==
            String(record.dst_cutoff_record_relative_path) &&
        dst_cutoff.eligible.record_sha256 == dst_checksum ||
        throw(ArgumentError("V2.2 Dst eligible-head binding changed"))
    if dst_sequence == 0
        isempty(String(record.dst_cutoff_record_relative_path)) ||
            throw(ArgumentError("V2.2 empty Dst cutoff has a record path"))
    else
        String(record.dst_cutoff_record_relative_path) ==
            _dst_record_relative(dst_sequence) || throw(ArgumentError(
                "V2.2 Dst cutoff path changed",
            ))
    end
    anchor = dst_cutoff.anchor
    anchor.status == String(record.dst_anchor_status) &&
        anchor.record_relative_path ==
            String(record.dst_anchor_record_relative_path) &&
        anchor.record_sha256 == String(record.dst_anchor_record_sha256) &&
        anchor.anchor_time_utc == String(record.dst_anchor_time_utc) &&
        anchor.anchor_age_seconds == Int(record.dst_anchor_age_seconds) ||
        throw(ArgumentError("V2.2 Dst anchor binding changed"))

    sequence = Int(record.issue_sequence)
    sequence >= 1 || throw(ArgumentError(
        "V2.2 research-issue sequence must be positive",
    ))
    previous_relative = String(record.previous_issue_record_relative_path)
    previous_sha = String(record.previous_issue_record_sha256)
    if sequence == 1
        isempty(previous_relative) && previous_sha == _ZERO_SHA256 ||
            throw(ArgumentError("V2.2 first issue has a predecessor"))
    else
        isempty(previous_relative) && throw(ArgumentError(
            "V2.2 research-issue predecessor is missing",
        ))
        previous_raw = _read_regular_json(
            storage, previous_relative, "previous research issue record",
        )
        previous_issue = _parse_utc(previous_raw.issue_time_utc)
        previous = _verify_issue_header(storage, previous_issue)
        Int(previous.issue_sequence) == sequence - 1 &&
            String(previous.issue_record_sha256) == previous_sha &&
            String(previous.issue_record_relative_path) == previous_relative ||
            throw(ArgumentError("V2.2 research-issue predecessor changed"))
        previous_issue + Minute(30) == issue &&
            previous_relative == _issue_record_relative(previous_issue) ||
            throw(ArgumentError("V2.2 research-issue half-hour chain changed"))
        verify_predecessors && _verify_issue_record(
            storage, previous_issue; verify_predecessors=true,
        )
    end
    return record
end

"Verify one issue against only its saved input prefixes and predecessor chain."
function verify_v2_2_research_issue(root::AbstractString,
                                    issue_time_utc::DateTime;
                                    lock_timeout_sec::Real=30.0)
    storage = L1._v22_l1_validate_root(root; create=false)
    return L1._v22_l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        _assert_cohort_valid(storage)
        _verify_issue_record(storage, issue_time_utc)
    end
end

function verify_v2_2_research_issues(root::AbstractString;
                                     lock_timeout_sec::Real=30.0)
    storage = L1._v22_l1_validate_root(root; create=false)
    return L1._v22_l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        _assert_cohort_valid(storage)
        latest = _issue_latest(storage)
        if latest.sequence == 0
            isempty(_record_names(
                storage, joinpath("research_issues", "records"),
            )) || throw(ArgumentError(
                "V2.2 issue archive has records without a latest pointer",
            ))
            return NamedTuple[]
        end
        head = _verify_issue_record(
            storage, _parse_utc(latest.issue_time_utc),
        )
        Int(head.issue_sequence) == latest.sequence &&
            String(head.issue_record_relative_path) == latest.record_relative_path &&
            String(head.issue_record_sha256) == latest.record_sha256 ||
            throw(ArgumentError("V2.2 issue latest pointer changed"))
        names = _record_names(
            storage, joinpath("research_issues", "records"),
        )
        length(names) == latest.sequence || throw(ArgumentError(
            "V2.2 issue archive has orphan records",
        ))
        records = Any[]
        current_issue = _parse_utc(head.issue_time_utc)
        while true
            current = _verify_issue_record(
                storage, current_issue; verify_predecessors=false,
            )
            push!(records, current)
            Int(current.issue_sequence) == 1 && break
            previous = _read_regular_json(
                storage, String(current.previous_issue_record_relative_path),
                "previous research issue record",
            )
            current_issue = _parse_utc(previous.issue_time_utc)
        end
        reverse!(records)
        return records
    end
end

function _existing_issue_unlocked(storage::AbstractString, issue::DateTime)
    relative = _issue_record_relative(issue)
    path = L1._v22_l1_resolve_relative(storage, relative, "research issue record")
    ispath(path) || return nothing
    record = _verify_issue_record(storage, issue)
    latest = _issue_latest(storage)
    if latest.sequence < Int(record.issue_sequence)
        latest.sequence == Int(record.issue_sequence) - 1 &&
            latest.record_relative_path ==
                String(record.previous_issue_record_relative_path) &&
            latest.record_sha256 ==
                String(record.previous_issue_record_sha256) ||
            throw(ArgumentError("V2.2 orphan issue cannot extend latest chain"))
        _write_issue_latest!(storage, record)
    elseif latest.sequence == Int(record.issue_sequence)
        latest.record_relative_path == relative &&
            latest.record_sha256 == String(record.issue_record_sha256) ||
            throw(ArgumentError("V2.2 duplicate issue conflicts with latest"))
    end
    return record
end

"Write one immutable, unavailable research-capture issue on the UTC half-hour."
function capture_v2_2_research_issue!(root::AbstractString,
        issue_time_utc::DateTime;
        preparation_utc_clock::Function=() -> now(UTC),
        monotonic_clock::Function=time_ns,
        lock_timeout_sec::Real=30.0,
        crash_hook::Function=stage -> nothing,
        _pending_issue_sha256::Union{Nothing,String}=nothing)
    issue = _require_grid_issue(issue_time_utc)
    storage = L1._v22_l1_validate_root(root; create=false)
    return L1._v22_l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        _assert_cohort_valid(
            storage;
            allowed_pending_issue=
                _pending_issue_sha256 === nothing ? nothing : issue,
            allowed_pending_sha256=_pending_issue_sha256,
        )
        existing = _existing_issue_unlocked(storage, issue)
        existing === nothing || return existing
        latest = _issue_latest(storage)
        if latest.sequence > 0
            latest_record = _verify_issue_header(
                storage, _parse_utc(latest.issue_time_utc),
            )
            _parse_utc(latest_record.issue_time_utc) + Minute(30) == issue ||
                throw(ArgumentError(
                    "V2.2 research issue is not the next half-hour; " *
                    "gaps and backfills require a new cohort",
                ))
        end
        cutoff = _capture_l1_cutoff_heads_unlocked(storage, issue)
        l1_pair = _select_pair(storage, issue, cutoff)

        dst_verified = _verify_complete_dst_cached(storage)
        dst_issue_cutoff = _capture_dst_issue_cutoff_unlocked(
            storage, issue, dst_verified,
        )
        dst_cutoff = dst_issue_cutoff.eligible
        dst_anchor = dst_issue_cutoff.anchor

        prepared = preparation_utc_clock()
        prepared isa DateTime || throw(ArgumentError(
            "V2.2 preparation clock must return DateTime",
        ))
        prepared >= issue || throw(ArgumentError(
            "V2.2 research issue cannot be prepared before its issue time",
        ))
        deadline = issue + Minute(5)
        prepared <= deadline || throw(ArgumentError(
            "V2.2 research issue cannot be prepared after its commitment window",
        ))
        if latest.sequence > 0
            previous = _read_regular_json(
                storage, latest.record_relative_path,
                "previous research issue record",
            )
            prepared >= _parse_utc(previous.record_prepared_utc) ||
                throw(ArgumentError(
                    "V2.2 research-issue preparation clock regressed",
                ))
        end
        prepared_monotonic = Int(monotonic_clock())
        prepared_monotonic >= 0 || throw(ArgumentError(
            "V2.2 research-issue monotonic clock is negative",
        ))
        sequence = latest.sequence + 1
        if _pending_issue_sha256 !== nothing
            pending = _read_pending_issue(storage)
            pending !== nothing &&
                String(pending.pending_record_sha256) ==
                    _pending_issue_sha256 &&
                _parse_utc(pending.issue_time_utc) == issue &&
                Int(pending.issue_sequence) == sequence &&
                String(pending.previous_issue_record_relative_path) ==
                    latest.record_relative_path &&
                String(pending.previous_issue_record_sha256) ==
                    latest.record_sha256 || throw(ArgumentError(
                        "V2.2 scheduled pending-issue binding changed",
                    ))
            prepared_monotonic >= Int(pending.started_monotonic_ns) ||
                throw(ArgumentError(
                    "V2.2 scheduled monotonic clock regressed",
                ))
        end
        relative = _issue_record_relative(issue)
        payload = (
            schema_version=V22_RESEARCH_ISSUE_SCHEMA_VERSION,
            issue_time_utc=_utc(issue),
            issue_sequence=sequence,
            issue_record_relative_path=relative,
            record_prepared_utc=_utc(prepared),
            record_prepared_monotonic_ns=prepared_monotonic,
            commit_deadline_utc=_utc(deadline),
            commit_witness_status="unavailable_research_capture_only",
            capture_mode=_pending_issue_sha256 === nothing ?
                "manual_research_capture" : "scheduled_fail_closed_guard",
            scheduler_pending_record_sha256=
                _pending_issue_sha256 === nothing ? _ZERO_SHA256 :
                _pending_issue_sha256,
            scheduler_completion_status=
                _pending_issue_sha256 === nothing ? "not_applicable" :
                "required",
            target_times_utc=[
                _utc(issue + Hour(lead)) for lead in V22_RESEARCH_LEADS_HOURS
            ],
            l1_cutoff_relative_path=cutoff.cutoff_relative_path,
            l1_cutoff_sha256=cutoff.cutoff_sha256,
            l1_pair_schema_version=L1.V22_L1_ISSUE_PAIR_SCHEMA_VERSION,
            l1_pair_status=l1_pair.status,
            l1_pair_contract_sha256=l1_pair.checksum,
            l1_pair_measurement_time_utc=l1_pair.measurement_time,
            l1_pair_source=l1_pair.source,
            dst_source_name=V22_DST_SOURCE.name,
            dst_source_url=V22_DST_SOURCE.url,
            dst_product_id=V22_DST_SOURCE.product_id,
            dst_issue_cutoff_relative_path=
                dst_issue_cutoff.cutoff_relative_path,
            dst_issue_cutoff_sha256=dst_issue_cutoff.cutoff_sha256,
            dst_cutoff_sequence=dst_cutoff.sequence,
            dst_cutoff_record_relative_path=dst_cutoff.record_relative_path,
            dst_cutoff_record_sha256=dst_cutoff.record_sha256,
            dst_anchor_status=dst_anchor.status,
            dst_anchor_record_relative_path=dst_anchor.record_relative_path,
            dst_anchor_record_sha256=dst_anchor.record_sha256,
            dst_anchor_time_utc=dst_anchor.anchor_time_utc,
            dst_anchor_age_seconds=dst_anchor.anchor_age_seconds,
            model_component_status="unavailable_no_fitted_gated_v2_2",
            issuance_status="research_capture_only_unavailable",
            numeric_forecast_status="not_emitted",
            previous_issue_record_relative_path=latest.record_relative_path,
            previous_issue_record_sha256=latest.record_sha256,
        )
        checksum = _issue_record_sha256(payload)
        record = merge(payload, (issue_record_sha256=checksum,))
        L1._v22_l1_atomic_text(storage, relative, JSON3.write(record))
        crash_hook(:after_issue_record)
        _write_issue_latest!(storage, record)
        return _verify_issue_record(
            storage, issue;
            require_scheduled_terminal=_pending_issue_sha256 === nothing,
        )
    end
end

"""
Run the explicit prospective input/issue scheduler.

The first issue is supplied as an exact UTC half-hour. Input polling stops for
`input_blackout_sec` before each issue, the immutable issue is written before
post-issue polling resumes, and a missed five-minute commitment window fails
closed. The scheduler is never started by package or deployment code.
"""
function run_v2_2_research_capture_scheduler!(root::AbstractString,
        first_issue_time_utc::DateTime;
        wall_clock::Function=() -> now(UTC),
        monotonic_clock::Function=time_ns,
        input_capture!::Function=capture_v2_2_prospective_inputs!,
        sleeper::Function=sleep,
        poll_interval_sec::Real=60.0,
        input_blackout_sec::Integer=300,
        issue_crash_hook::Function=stage -> nothing,
        invalidation_crash_hook::Function=stage -> nothing,
        max_iterations::Union{Nothing,Integer}=nothing)
    next_issue = _require_grid_issue(first_issue_time_utc)
    interval = Float64(poll_interval_sec)
    isfinite(interval) && interval > 0.0 || throw(ArgumentError(
        "V2.2 scheduler poll interval must be finite and positive",
    ))
    blackout = Int(input_blackout_sec)
    180 <= blackout < 1_800 || throw(ArgumentError(
        "V2.2 scheduler input blackout must be 180--1799 seconds",
    ))
    if max_iterations !== nothing
        Int(max_iterations) >= 1 || throw(ArgumentError(
            "V2.2 scheduler max iterations must be positive",
        ))
    end
    storage = L1._v22_l1_validate_root(root; create=true)
    L1._v22_l1_with_lock(storage; timeout_sec=30.0) do
        _assert_cohort_valid(storage)
    end
    iterations = 0
    while true
        L1._v22_l1_with_lock(storage; timeout_sec=30.0) do
            _assert_cohort_valid(storage)
        end
        current = wall_clock()
        current isa DateTime || throw(ArgumentError(
            "V2.2 scheduler wall clock must return DateTime",
        ))
        if current >= next_issue
            current <= next_issue + Minute(5) || throw(ArgumentError(
                "V2.2 scheduler missed an issue commitment window; " *
                "the cohort must restart without shifting or backfilling",
            ))
            pending = _begin_pending_issue!(
                storage, next_issue, current, Int(monotonic_clock()),
            )
            pending_sha256 = String(pending.pending_record_sha256)
            issued = try
                capture_v2_2_research_issue!(
                    storage, next_issue;
                    preparation_utc_clock=() -> current,
                    monotonic_clock=monotonic_clock,
                    crash_hook=issue_crash_hook,
                    _pending_issue_sha256=pending_sha256,
                )
            catch capture_error
                issue_path = L1._v22_l1_resolve_relative(
                    storage, _issue_record_relative(next_issue),
                    "scheduled issue record",
                )
                if !ispath(issue_path) && !islink(issue_path)
                    cancellation_error = try
                        _cancel_uncommitted_pending_issue!(
                            storage, next_issue, pending_sha256,
                        )
                        nothing
                    catch error
                        error
                    end
                    cancellation_error === nothing || throw(ErrorException(
                        "V2.2 issue capture failed and its uncommitted " *
                        "pending guard could not be archived: " *
                        sprint(showerror, cancellation_error),
                    ))
                end
                rethrow(capture_error)
            end
            completed = wall_clock()
            completed isa DateTime && completed >= current ||
                throw(ArgumentError(
                    "V2.2 scheduler wall clock regressed after issue commit",
                ))
            if completed > next_issue + Minute(5)
                _record_invalid_cohort!(
                    storage, next_issue, issued, completed, pending_sha256;
                    crash_hook=invalidation_crash_hook,
                )
                throw(ArgumentError(
                    "V2.2 issue became durable after its commitment window; " *
                    "the cohort must restart",
                ))
            end
            _complete_pending_issue!(
                storage, next_issue, pending_sha256, issued, completed,
            )
            next_issue += Minute(30)
        elseif current < next_issue - Second(blackout)
            input_capture!(storage)
        end
        iterations += 1
        if max_iterations !== nothing && iterations >= Int(max_iterations)
            return (
                iterations=iterations,
                next_issue_time_utc=_utc(next_issue),
            )
        end
        sleeper(interval)
    end
end

function _argument(args, prefix, fallback=nothing)
    match = findfirst(argument -> startswith(argument, prefix), args)
    match === nothing && return fallback
    return split(args[match], '='; limit=2)[2]
end

function main_v2_2_prospective_issue_capture(args=ARGS)
    capture_once = "--capture-once" in args
    run_scheduler = "--run-scheduler" in args
    xor(capture_once, run_scheduler) || error(
        "V2.2 prospective capture is off by default; pass exactly one of " *
        "--capture-once or --run-scheduler",
    )
    root = _argument(args, "--root=")
    issue_text = _argument(args, "--issue-utc=")
    root === nothing && error("--root is required")
    issue_text === nothing && error("--issue-utc is required")
    issue = _parse_utc(issue_text)
    if run_scheduler
        return run_v2_2_research_capture_scheduler!(root, issue)
    end
    record = capture_v2_2_research_issue!(root, issue)
    println(
        "captured unavailable V2.2 research issue ", record.issue_time_utc,
        " ", record.issue_record_sha256,
    )
    return record
end

end # module V22ProspectiveIssueCapture

if abspath(PROGRAM_FILE) == @__FILE__
    V22ProspectiveIssueCapture.main_v2_2_prospective_issue_capture()
end
