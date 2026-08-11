module V22L1ReceiptPairing

# Pure, read-only pairing over an already captured V2.2 L1 v4 archive.
# Including the collector supplies its archive verifier and schema oracles; its
# guarded main entry point is not executed.
include(joinpath(@__DIR__, "v2_2_l1_receipt_collector.jl"))

export V22_L1_ISSUE_CUTOFF_SCHEMA_VERSION,
       V22_L1_ISSUE_PAIR_SCHEMA_VERSION,
       capture_v2_2_l1_issue_cutoff!, verify_v2_2_l1_issue_cutoff,
       select_v2_2_l1_issue_pair, select_v2_2_l1_issue_pairs

const V22_L1_ISSUE_PAIR_SCHEMA_VERSION = "v2_2_l1_issue_pair_v2"

function _v22_l1_pair_position(metadata)
    text = String(metadata.ephemeris_record_json)
    isempty(text) && error("V2.2 L1 admitted row has no ephemeris record")
    position = JSON3.read(text)
    source = _v22_l1_valid_source_token(_v22_l1_object_field(position, :source))
    source == "DSCOVR" || error(
        "V2.2 L1 paired ephemeris source is not DSCOVR",
    )
    timestamp = String(_v22_l1_object_field(position, :position_time_utc))
    _v22_l1_parse_utc(timestamp)
    frame = String(_v22_l1_object_field(position, :position_frame))
    units = String(_v22_l1_object_field(position, :position_units))
    frame == V22_L1_EPHEMERIS_POSITION_FRAME || error(
        "V2.2 L1 paired ephemeris frame changed",
    )
    units == V22_L1_EPHEMERIS_POSITION_UNITS || error(
        "V2.2 L1 paired ephemeris units changed",
    )
    x = _v22_l1_finite_number(_v22_l1_object_field(position, :x_gse))
    y = _v22_l1_finite_number(_v22_l1_object_field(position, :y_gse))
    z = _v22_l1_finite_number(_v22_l1_object_field(position, :z_gse))
    any(isnothing, (x, y, z)) && error(
        "V2.2 L1 paired ephemeris contains invalid GSE coordinates",
    )
    method = String(_v22_l1_object_field(position, :method))
    method in ("exact", "linear") || error(
        "V2.2 L1 paired ephemeris interpolation method changed",
    )
    fraction = _v22_l1_finite_number(
        _v22_l1_object_field(position, :interpolation_fraction),
    )
    fraction !== nothing && 0.0 <= fraction <= 1.0 || error(
        "V2.2 L1 paired ephemeris interpolation fraction is invalid",
    )
    return (
        source=source,
        timestamp_utc=timestamp,
        frame=frame,
        units=units,
        method=method,
        lower_time_utc=String(_v22_l1_object_field(position, :lower_time_utc)),
        upper_time_utc=String(_v22_l1_object_field(position, :upper_time_utc)),
        interpolation_fraction=fraction,
        x_gse=x,
        y_gse=y,
        z_gse=z,
    )
end

function _v22_l1_row_observations(parsed)
    rows_by_time = Dict{DateTime,Vector{Any}}()
    for row in parsed
        raw_time = _v22_l1_object_field(row, :time_tag)
        raw_time isa AbstractString || continue
        timestamp = try
            _v22_l1_parse_source_utc(raw_time)
        catch error
            error isa InterruptException && rethrow()
            continue
        end
        push!(get!(rows_by_time, timestamp, Any[]), row)
    end
    observations = NamedTuple[]
    for (timestamp, rows) in rows_by_time
        if length(rows) != 1
            push!(observations, (
                timestamp=timestamp,
                source="",
                target_row_sha256="ambiguous",
                status="ambiguous_measurement_time",
            ))
            continue
        end
        row = only(rows)
        source = _v22_l1_valid_source_token(_v22_l1_object_field(row, :source))
        push!(observations, (
            timestamp=timestamp,
            source=source === nothing ? "" : source,
            target_row_sha256=_v22_l1_sha256(codeunits(JSON3.write(row))),
            status="unique_measurement_row",
        ))
    end
    return observations
end

function _v22_l1_pair_candidate(root::AbstractString, source, record,
                                kind::Symbol, issue_time::DateTime)
    receipt = _v22_l1_parse_utc(record.receipt_completed_utc)
    receipt <= issue_time || return (candidate=nothing, observations=NamedTuple[])
    String(record.capture_outcome) == "http_response" ||
        return (candidate=nothing, observations=NamedTuple[])
    Int(record.http_status) == 200 ||
        return (candidate=nothing, observations=NamedTuple[])
    metadata = record.metadata_provenance

    raw_relative = String(record.raw_relative_path)
    raw_path = _v22_l1_resolve_relative(root, raw_relative, "paired raw path")
    raw = read(raw_path)
    raw_sha256 = _v22_l1_sha256(raw)
    raw_sha256 == String(record.body_sha256) || error(
        "V2.2 L1 paired raw response checksum changed",
    )
    parsed = JSON3.read(String(copy(raw)))
    parsed isa AbstractVector || error(
        "V2.2 L1 admitted raw response is not an array",
    )
    observations = _v22_l1_row_observations(parsed)
    Bool(metadata.rows_admissible) || return (
        candidate=nothing, observations=observations,
    )
    selected = _v22_l1_measurement_target(parsed)
    selected.status == "bound" || error(
        "V2.2 L1 admitted response has no unique active latest row",
    )
    target = selected.target
    target.timestamp <= receipt || error(
        "V2.2 L1 admitted measurement time is later than its receipt",
    )
    target.source == "DSCOVR" || error(
        "V2.2 L1 admitted response does not target DSCOVR",
    )
    timestamp_utc = _v22_l1_utc(target.timestamp)
    timestamp_utc == String(metadata.quality_row_timestamp_utc) || error(
        "V2.2 L1 quality timestamp does not match its admitted row",
    )
    String(metadata.quality_row_source) == target.source || error(
        "V2.2 L1 quality source does not match its admitted row",
    )
    String(metadata.quality_binding_status) ==
        "bound_noaa_dscovr_overall_quality" || error(
        "V2.2 L1 admitted row lacks bound DSCOVR quality",
    )
    Int(metadata.quality_value) == 0 || error(
        "V2.2 L1 admitted row is not normal quality",
    )
    String(metadata.quality_decision) == "accept_normal_overall_quality" ||
        error("V2.2 L1 admitted row has a non-accepting quality decision")

    if kind == :mag
        String(source.name) == V22_L1_RECEIPT_SOURCES[1].name &&
            String(source.url) == V22_L1_RECEIPT_SOURCES[1].url || error(
            "V2.2 L1 magnetometer source is not canonical",
        )
        String(metadata.quality_source_product) == "dscovr_m1m" || error(
            "V2.2 L1 magnetometer quality product changed",
        )
        String(metadata.quality_required_fields_status) ==
            "bound_required_bx_by_bz_gsm" || error(
            "V2.2 L1 admitted magnetometer row lacks required components",
        )
        drivers = (
            bx_gsm=_v22_l1_bounded_measurement(
                target.row, :bx_gsm, -1.0e3, 1.0e3,
            ),
            by_gsm=_v22_l1_bounded_measurement(
                target.row, :by_gsm, -1.0e3, 1.0e3,
            ),
            bz_gsm=_v22_l1_bounded_measurement(
                target.row, :bz_gsm, -1.0e3, 1.0e3,
            ),
        )
    elseif kind == :wind
        String(source.name) == V22_L1_RECEIPT_SOURCES[2].name &&
            String(source.url) == V22_L1_RECEIPT_SOURCES[2].url || error(
            "V2.2 L1 wind source is not canonical",
        )
        String(metadata.quality_source_product) == "dscovr_f1m" || error(
            "V2.2 L1 wind quality product changed",
        )
        String(metadata.quality_required_fields_status) ==
            "bound_required_speed_density_vx_gse" || error(
            "V2.2 L1 admitted wind row lacks required transport fields",
        )
        drivers = (
            proton_speed=_v22_l1_bounded_measurement(
                target.row, :proton_speed, 50.0, 5.0e3,
            ),
            proton_density=_v22_l1_bounded_measurement(
                target.row, :proton_density, 0.0, 1.0e3,
            ),
            proton_vx_gse=_v22_l1_bounded_measurement(
                target.row, :proton_vx_gse, -5.0e3, 5.0e3,
            ),
        )
    else
        throw(ArgumentError("V2.2 L1 pair candidate has an unknown kind"))
    end
    any(isnothing, values(drivers)) && error(
        "V2.2 L1 admitted row contains an invalid required driver",
    )

    position = _v22_l1_pair_position(metadata)
    position.timestamp_utc == timestamp_utc || error(
        "V2.2 L1 ephemeris timestamp does not match its admitted row",
    )
    String(metadata.ephemeris_record_sha256) ==
        _v22_l1_sha256(codeunits(String(metadata.ephemeris_record_json))) || error(
        "V2.2 L1 paired ephemeris record checksum changed",
    )
    row_sha256 = _v22_l1_sha256(codeunits(JSON3.write(target.row)))
    candidate = (
        kind=kind,
        timestamp=target.timestamp,
        timestamp_utc=timestamp_utc,
        source=target.source,
        receipt=receipt,
        receipt_completed_utc=String(record.receipt_completed_utc),
        sequence=Int(record.sequence),
        record_relative_path=_v22_l1_record_relative(
            source.name, Int(record.sequence),
        ),
        record_sha256=String(record.record_sha256),
        raw_relative_path=raw_relative,
        raw_sha256=raw_sha256,
        target_row_sha256=row_sha256,
        ephemeris_source_object_raw_relative_path=String(
            metadata.ephemeris_source_object_raw_relative_path,
        ),
        ephemeris_source_object_sha256=String(
            metadata.ephemeris_source_object_sha256,
        ),
        ephemeris_record_sha256=String(metadata.ephemeris_record_sha256),
        source_product_id=String(source.name),
        quality_source_product=String(metadata.quality_source_product),
        quality_value=Int(metadata.quality_value),
        quality_binding_status=String(metadata.quality_binding_status),
        quality_decision=String(metadata.quality_decision),
        quality_required_fields_status=String(
            metadata.quality_required_fields_status,
        ),
        drivers=drivers,
        position=position,
    )
    target_observations = filter(
        item -> item.timestamp == candidate.timestamp, observations,
    )
    length(target_observations) == 1 &&
        only(target_observations).status == "unique_measurement_row" &&
        only(target_observations).target_row_sha256 ==
            candidate.target_row_sha256 || error(
        "V2.2 L1 admitted target disagrees with its latest raw row",
    )
    return (candidate=candidate, observations=observations)
end

function _v22_l1_pair_candidates(root::AbstractString, source, count::Integer,
                                 kind::Symbol, issue_time::DateTime)
    candidates = NamedTuple[]
    observations = NamedTuple[]
    for sequence in 1:Int(count)
        relative = _v22_l1_record_relative(source.name, sequence)
        path = _v22_l1_resolve_relative(root, relative, "paired record path")
        record = JSON3.read(read(path, String))
        _v22_l1_record_sha256(record) == String(record.record_sha256) || error(
            "V2.2 L1 paired record checksum changed after verification",
        )
        result = _v22_l1_pair_candidate(
            root, source, record, kind, issue_time,
        )
        result.candidate === nothing || push!(candidates, result.candidate)
        append!(observations, result.observations)
    end
    return (admitted=candidates, observations=observations)
end

function _v22_l1_stable_candidate(candidates, observations,
                                  timestamp::DateTime,
                                  label::AbstractString)
    matches = filter(candidate -> candidate.timestamp == timestamp, candidates)
    isempty(matches) && error("V2.2 L1 $label pair candidate is missing")
    fingerprints = unique((
        candidate.target_row_sha256,
        candidate.quality_value,
        candidate.quality_decision,
        candidate.quality_required_fields_status,
        candidate.ephemeris_record_sha256,
    ) for candidate in matches)
    length(fingerprints) == 1 || error(
        "V2.2 L1 $label row has conflicting pre-issue revisions",
    )
    observed = filter(item -> item.timestamp == timestamp, observations)
    isempty(observed) && error(
        "V2.2 L1 $label row lacks raw revision evidence",
    )
    raw_fingerprints = unique((
        item.status, item.source, item.target_row_sha256,
    ) for item in observed)
    length(raw_fingerprints) == 1 || error(
        "V2.2 L1 $label row has conflicting admitted or rejected revisions",
    )
    raw_fingerprint = only(raw_fingerprints)
    raw_fingerprint[1] == "unique_measurement_row" &&
        raw_fingerprint[2] == "DSCOVR" &&
        raw_fingerprint[3] == first(matches).target_row_sha256 || error(
        "V2.2 L1 $label row revision no longer matches its admitted evidence",
    )
    sort!(matches; by=candidate -> (candidate.receipt, candidate.sequence))
    return last(matches)
end

function _v22_l1_assert_candidate_unchanged(root::AbstractString, candidate)
    record_path = _v22_l1_resolve_relative(
        root, candidate.record_relative_path, "selected paired record path",
    )
    record = JSON3.read(read(record_path, String))
    _v22_l1_record_sha256(record) == candidate.record_sha256 || error(
        "V2.2 L1 selected record changed during pairing",
    )
    raw_path = _v22_l1_resolve_relative(
        root, candidate.raw_relative_path, "selected paired raw path",
    )
    _v22_l1_sha256(read(raw_path)) == candidate.raw_sha256 || error(
        "V2.2 L1 selected raw response changed during pairing",
    )
    ephemeris_path = _v22_l1_resolve_relative(
        root, candidate.ephemeris_source_object_raw_relative_path,
        "selected paired ephemeris path",
    )
    _v22_l1_sha256(read(ephemeris_path)) ==
        candidate.ephemeris_source_object_sha256 || error(
        "V2.2 L1 selected ephemeris source object changed during pairing",
    )
    return nothing
end

"""
    select_v2_2_l1_issue_pair(root, issue_time_utc::DateTime; lock_timeout_sec=30.0)

Acquire the collector lock, fully verify the live receipt archive, save its
immutable issue cutoff, and return the latest common, individually admitted
DSCOVR magnetometer/wind row received no later than `issue_time_utc`. The
three-argument overload requires an existing cutoff and reads only its exact
source prefixes. Neither overload interpolates measurement values or performs
network access.
"""
function _v22_l1_pair_floor_30(time::DateTime)
    minute = Dates.minute(time) < 30 ? 0 : 30
    return DateTime(
        Dates.year(time), Dates.month(time), Dates.day(time), Dates.hour(time),
        minute,
    )
end

function _v22_l1_pair_ceil_30(time::DateTime)
    floored = _v22_l1_pair_floor_30(time)
    return floored == time ? floored : floored + Minute(30)
end

function _v22_l1_issue_pair_output(mag_selected, wind_selected,
                                   issue_time_utc::DateTime,
                                   cutoff)
    mag_selected.source == wind_selected.source == "DSCOVR" || error(
        "V2.2 L1 paired records changed spacecraft identity",
    )
    mag_selected.timestamp_utc == wind_selected.timestamp_utc || error(
        "V2.2 L1 paired records changed measurement timestamp",
    )
    mag_selected.ephemeris_record_sha256 ==
        wind_selected.ephemeris_record_sha256 || error(
        "V2.2 L1 paired records disagree on the bound GSE position",
    )
    position = mag_selected.position
    first_eligible = _v22_l1_pair_ceil_30(max(
        mag_selected.receipt, wind_selected.receipt,
    ))
    payload = (
        schema_version=V22_L1_ISSUE_PAIR_SCHEMA_VERSION,
        issue_time_utc=_v22_l1_utc(issue_time_utc),
        first_eligible_issue_time_utc=_v22_l1_utc(first_eligible),
        issue_cutoff_relative_path=cutoff.cutoff_relative_path,
        issue_cutoff_sha256=cutoff.cutoff_sha256,
        measurement_time_utc=mag_selected.timestamp_utc,
        source="DSCOVR",
        mag_source_product_id=mag_selected.source_product_id,
        wind_source_product_id=wind_selected.source_product_id,
        magnetic_component_frame="GSM",
        magnetic_component_units="nT",
        proton_speed_units="km/s",
        proton_density_units="cm^-3",
        proton_vx_frame="GSE",
        proton_vx_units="km/s",
        bx_gsm=mag_selected.drivers.bx_gsm,
        by_gsm=mag_selected.drivers.by_gsm,
        bz_gsm=mag_selected.drivers.bz_gsm,
        proton_speed=wind_selected.drivers.proton_speed,
        proton_density=wind_selected.drivers.proton_density,
        proton_vx_gse=wind_selected.drivers.proton_vx_gse,
        position_x_gse=position.x_gse,
        position_y_gse=position.y_gse,
        position_z_gse=position.z_gse,
        position_frame=position.frame,
        position_units=position.units,
        position_method=position.method,
        position_lower_time_utc=position.lower_time_utc,
        position_upper_time_utc=position.upper_time_utc,
        position_interpolation_fraction=position.interpolation_fraction,
        ephemeris_record_sha256=mag_selected.ephemeris_record_sha256,
        mag_quality_source_product=mag_selected.quality_source_product,
        mag_quality_value=mag_selected.quality_value,
        mag_quality_binding_status=mag_selected.quality_binding_status,
        mag_quality_decision=mag_selected.quality_decision,
        mag_quality_required_fields_status=
            mag_selected.quality_required_fields_status,
        mag_sequence=mag_selected.sequence,
        mag_receipt_completed_utc=mag_selected.receipt_completed_utc,
        mag_record_sha256=mag_selected.record_sha256,
        mag_raw_sha256=mag_selected.raw_sha256,
        mag_ephemeris_source_object_sha256=
            mag_selected.ephemeris_source_object_sha256,
        wind_quality_source_product=wind_selected.quality_source_product,
        wind_quality_value=wind_selected.quality_value,
        wind_quality_binding_status=wind_selected.quality_binding_status,
        wind_quality_decision=wind_selected.quality_decision,
        wind_quality_required_fields_status=
            wind_selected.quality_required_fields_status,
        wind_sequence=wind_selected.sequence,
        wind_receipt_completed_utc=wind_selected.receipt_completed_utc,
        wind_record_sha256=wind_selected.record_sha256,
        wind_raw_sha256=wind_selected.raw_sha256,
        wind_ephemeris_source_object_sha256=
            wind_selected.ephemeris_source_object_sha256,
    )
    return merge(payload, (
        pair_contract_sha256=_v22_l1_sha256(
            codeunits(JSON3.write(payload)),
        ),
    ))
end

function _select_v2_2_l1_issue_pairs_unlocked(
        storage::AbstractString,
        issue_time_utc::DateTime,
        measurement_start_utc::DateTime,
        cutoff_relative_path::AbstractString;
        latest_only::Bool=false)
    measurement_start_utc <= issue_time_utc || throw(ArgumentError(
        "V2.2 L1 measurement window starts after its issue time",
    ))
    sources = V22_L1_RECEIPT_SOURCES
    before = _v22_l1_verify_issue_cutoff(
        storage, issue_time_utc, cutoff_relative_path, sources,
    )
    counts = Dict(result.source_name => result.records for result in before.prefixes)
    mag = _v22_l1_pair_candidates(
        storage, sources[1], counts[sources[1].name], :mag, issue_time_utc,
    )
    wind = _v22_l1_pair_candidates(
        storage, sources[2], counts[sources[2].name], :wind, issue_time_utc,
    )
    common = intersect(
        Set(candidate.timestamp for candidate in mag.admitted),
        Set(candidate.timestamp for candidate in wind.admitted),
    )
    timestamps = sort!(collect(filter(
        timestamp -> timestamp >= measurement_start_utc, common,
    )))
    isempty(timestamps) && error(
        "V2.2 L1 archive has no exact admitted mag/wind timestamp in the requested causal window",
    )
    latest_only && (timestamps = [last(timestamps)])
    selected = Tuple((
        mag=_v22_l1_stable_candidate(
            mag.admitted, mag.observations, timestamp, "magnetometer",
        ),
        wind=_v22_l1_stable_candidate(
            wind.admitted, wind.observations, timestamp, "wind",
        ),
    ) for timestamp in timestamps)

    after = _v22_l1_verify_issue_cutoff(
        storage, issue_time_utc, cutoff_relative_path, sources,
    )
    before == after || error("V2.2 L1 archive changed during issue pairing")
    for pair in selected
        _v22_l1_assert_candidate_unchanged(storage, pair.mag)
        _v22_l1_assert_candidate_unchanged(storage, pair.wind)
    end
    return Tuple(
        _v22_l1_issue_pair_output(
            pair.mag, pair.wind, issue_time_utc, before,
        )
        for pair in selected
    )
end

function _select_v2_2_l1_issue_pair_unlocked(storage::AbstractString,
                                             issue_time_utc::DateTime,
                                             cutoff_relative_path::AbstractString)
    pairs = _select_v2_2_l1_issue_pairs_unlocked(
        storage, issue_time_utc, typemin(DateTime), cutoff_relative_path;
        latest_only=true,
    )
    return last(pairs)
end

function select_v2_2_l1_issue_pair(root::AbstractString,
                                   issue_time_utc::DateTime;
                                   lock_timeout_sec::Real=30.0)
    storage = _v22_l1_validate_root(root; create=false)
    return _v22_l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        sources = V22_L1_RECEIPT_SOURCES
        cutoff = _v22_l1_capture_issue_cutoff_unlocked(
            storage, issue_time_utc, sources,
        )
        _select_v2_2_l1_issue_pair_unlocked(
            storage, issue_time_utc, cutoff.cutoff_relative_path,
        )
    end
end

function select_v2_2_l1_issue_pair(
        root::AbstractString,
        issue_time_utc::DateTime,
        cutoff_relative_path::AbstractString;
        lock_timeout_sec::Real=30.0)
    storage = _v22_l1_validate_root(root; create=false)
    return _v22_l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        _select_v2_2_l1_issue_pair_unlocked(
            storage, issue_time_utc, cutoff_relative_path,
        )
    end
end


"""
    select_v2_2_l1_issue_pairs(root, issue_time_utc::DateTime;
                               measurement_start_utc, lock_timeout_sec=30.0)

Under one collector lock, build the verified magnetometer and wind candidate
sets once per call and return every stable exact DSCOVR minute in the inclusive
measurement-time window. The result is chronological and contains only records
received by the issue. The live overload first saves a fully verified issue
cutoff; the cutoff-bound overload verifies and reads only those saved prefixes.
"""
function select_v2_2_l1_issue_pairs(
        root::AbstractString,
        issue_time_utc::DateTime;
        measurement_start_utc::DateTime,
        lock_timeout_sec::Real=30.0)
    storage = _v22_l1_validate_root(root; create=false)
    return _v22_l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        cutoff = _v22_l1_capture_issue_cutoff_unlocked(
            storage, issue_time_utc, V22_L1_RECEIPT_SOURCES,
        )
        _select_v2_2_l1_issue_pairs_unlocked(
            storage, issue_time_utc, measurement_start_utc,
            cutoff.cutoff_relative_path,
        )
    end
end


function select_v2_2_l1_issue_pairs(
        root::AbstractString,
        issue_time_utc::DateTime,
        cutoff_relative_path::AbstractString;
        measurement_start_utc::DateTime,
        lock_timeout_sec::Real=30.0)
    storage = _v22_l1_validate_root(root; create=false)
    return _v22_l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        _select_v2_2_l1_issue_pairs_unlocked(
            storage, issue_time_utc, measurement_start_utc,
            cutoff_relative_path,
        )
    end
end

end # module V22L1ReceiptPairing
