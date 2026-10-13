module StaticStackL1ReceiptPairing

# Pure, read-only pairing over an already captured static stack L1 v4 archive.
# Including the collector supplies its archive verifier and schema oracles; its
# guarded main entry point is not executed.
include(joinpath(@__DIR__, "l1_receipt_collector.jl"))

export L1_ISSUE_CUTOFF_SCHEMA_VERSION,
       L1_ISSUE_PAIR_SCHEMA_VERSION,
       capture_l1_issue_cutoff!, verify_l1_issue_cutoff,
       select_l1_issue_pair, select_l1_issue_pairs

const L1_ISSUE_PAIR_SCHEMA_VERSION = "l1_issue_pair_operational"
# Latest-only selection scans only records that can contribute a measurement
# minute within this lookback of the issue (G11-06/G11-08).
const L1_PAIR_LOOKBACK = Hour(2)

function _l1_pair_position(metadata)
    text = String(metadata.ephemeris_record_json)
    isempty(text) && error("static stack L1 admitted row has no ephemeris record")
    position = JSON3.read(text)
    source = _l1_valid_source_token(_l1_object_field(position, :source))
    source !== nothing && source in L1_ADMISSIBLE_SOURCES || error(
        "static stack L1 paired ephemeris source is not an admitted source",
    )
    timestamp = String(_l1_object_field(position, :position_time_utc))
    position_time = _l1_parse_utc(timestamp)
    frame = String(_l1_object_field(position, :position_frame))
    units = String(_l1_object_field(position, :position_units))
    frame == L1_EPHEMERIS_POSITION_FRAME || error(
        "static stack L1 paired ephemeris frame changed",
    )
    units == L1_EPHEMERIS_POSITION_UNITS || error(
        "static stack L1 paired ephemeris units changed",
    )
    x = _l1_finite_number(_l1_object_field(position, :x_gse))
    y = _l1_finite_number(_l1_object_field(position, :y_gse))
    z = _l1_finite_number(_l1_object_field(position, :z_gse))
    any(isnothing, (x, y, z)) && error(
        "static stack L1 paired ephemeris contains invalid GSE coordinates",
    )
    method = String(_l1_object_field(position, :method))
    method in ("exact", "linear", "hold") || error(
        "static stack L1 paired ephemeris interpolation method changed",
    )
    fraction = _l1_finite_number(
        _l1_object_field(position, :interpolation_fraction),
    )
    fraction !== nothing && 0.0 <= fraction <= 1.0 || error(
        "static stack L1 paired ephemeris interpolation fraction is invalid",
    )
    lower_time_utc = String(_l1_object_field(position, :lower_time_utc))
    upper_time_utc = String(_l1_object_field(position, :upper_time_utc))
    if method == "hold"
        # v3 bounded causal hold: one earlier row, zero fraction, recorded age
        # equal to the measurement-minus-row interval and at most 90 minutes.
        held = _l1_parse_utc(lower_time_utc)
        age = _l1_finite_number(
            _l1_object_field(position, :hold_age_seconds),
        )
        lower_time_utc == upper_time_utc && fraction == 0.0 &&
            held < position_time &&
            position_time - held <= L1_EPHEMERIS_MAX_HOLD &&
            age !== nothing &&
            age == Dates.value(position_time - held) / 1000.0 || error(
            "static stack L1 paired ephemeris hold is inconsistent or stale",
        )
    end
    return (
        source=source,
        timestamp_utc=timestamp,
        frame=frame,
        units=units,
        method=method,
        lower_time_utc=lower_time_utc,
        upper_time_utc=upper_time_utc,
        interpolation_fraction=fraction,
        x_gse=x,
        y_gse=y,
        z_gse=z,
    )
end

function _l1_row_observations(parsed)
    # Keyed by (timestamp, source): live feeds report the same minute from
    # several spacecraft, which is not a same-stream ambiguity.
    rows_by_time = Dict{Tuple{DateTime,String},Vector{Any}}()
    for row in parsed
        raw_time = _l1_object_field(row, :time_tag)
        raw_time isa AbstractString || continue
        timestamp = try
            _l1_parse_source_utc(raw_time)
        catch error
            error isa InterruptException && rethrow()
            continue
        end
        row_source = _l1_valid_source_token(
            _l1_object_field(row, :source),
        )
        key = (timestamp, row_source === nothing ? "" : row_source)
        push!(get!(rows_by_time, key, Any[]), row)
    end
    observations = NamedTuple[]
    for ((timestamp, source), rows) in rows_by_time
        if length(rows) != 1
            push!(observations, (
                timestamp=timestamp,
                source=source,
                target_row_sha256="ambiguous",
                status="ambiguous_measurement_time",
            ))
            continue
        end
        row = only(rows)
        push!(observations, (
            timestamp=timestamp,
            source=source,
            target_row_sha256=_l1_sha256(codeunits(JSON3.write(row))),
            status="unique_measurement_row",
        ))
    end
    return observations
end

function _l1_pair_candidate(root::AbstractString, source, record,
                                kind::Symbol, issue_time::DateTime)
    receipt = _l1_parse_utc(record.receipt_completed_utc)
    receipt <= issue_time || return (candidate=nothing, observations=NamedTuple[])
    String(record.capture_outcome) == "http_response" ||
        return (candidate=nothing, observations=NamedTuple[])
    Int(record.http_status) == 200 ||
        return (candidate=nothing, observations=NamedTuple[])
    metadata = record.metadata_provenance

    raw_relative = String(record.raw_relative_path)
    raw_path = _l1_resolve_relative(root, raw_relative, "paired raw path")
    _l1_count_read!(:raw)
    raw = read(raw_path)
    raw_sha256 = _l1_sha256(raw)
    raw_sha256 == String(record.body_sha256) || error(
        "static stack L1 paired raw response checksum changed",
    )
    parsed = JSON3.read(String(copy(raw)))
    parsed isa AbstractVector || error(
        "static stack L1 admitted raw response is not an array",
    )
    observations = _l1_row_observations(parsed)
    Bool(metadata.rows_admissible) || return (
        candidate=nothing, observations=observations,
    )
    selected = _l1_measurement_target(parsed)
    selected.status == "bound" || error(
        "static stack L1 admitted response has no unique active latest row",
    )
    target = selected.target
    target.timestamp <= receipt || error(
        "static stack L1 admitted measurement time is later than its receipt",
    )
    target.source in L1_ADMISSIBLE_SOURCES || error(
        "static stack L1 admitted response does not target an admitted source",
    )
    timestamp_utc = _l1_utc(target.timestamp)
    timestamp_utc == String(metadata.quality_row_timestamp_utc) || error(
        "static stack L1 quality timestamp does not match its admitted row",
    )
    String(metadata.quality_row_source) == target.source || error(
        "static stack L1 quality source does not match its admitted row",
    )
    _l1_bound_quality_status(
        String(metadata.quality_binding_status), target.source,
    ) || error(
        "static stack L1 admitted row lacks bound NOAA RTSW quality",
    )
    Int(metadata.quality_value) == 0 || error(
        "static stack L1 admitted row is not normal quality",
    )
    String(metadata.quality_decision) == "accept_normal_overall_quality" ||
        error("static stack L1 admitted row has a non-accepting quality decision")

    if kind == :mag
        String(source.name) == L1_RECEIPT_SOURCES[1].name &&
            String(source.url) == L1_RECEIPT_SOURCES[1].url || error(
            "static stack L1 magnetometer source is not canonical",
        )
        String(metadata.quality_source_product) == "dscovr_m1m" || error(
            "static stack L1 magnetometer quality product changed",
        )
        String(metadata.quality_required_fields_status) ==
            "bound_required_bx_by_bz_gsm" || error(
            "static stack L1 admitted magnetometer row lacks required components",
        )
        drivers = (
            bx_gsm=_l1_bounded_measurement(
                target.row, :bx_gsm, -1.0e3, 1.0e3,
            ),
            by_gsm=_l1_bounded_measurement(
                target.row, :by_gsm, -1.0e3, 1.0e3,
            ),
            bz_gsm=_l1_bounded_measurement(
                target.row, :bz_gsm, -1.0e3, 1.0e3,
            ),
        )
    elseif kind == :wind
        String(source.name) == L1_RECEIPT_SOURCES[2].name &&
            String(source.url) == L1_RECEIPT_SOURCES[2].url || error(
            "static stack L1 wind source is not canonical",
        )
        String(metadata.quality_source_product) == "dscovr_f1m" || error(
            "static stack L1 wind quality product changed",
        )
        String(metadata.quality_required_fields_status) ==
            "bound_required_speed_density_vx_gse" || error(
            "static stack L1 admitted wind row lacks required transport fields",
        )
        drivers = (
            proton_speed=_l1_bounded_measurement(
                target.row, :proton_speed, 50.0, 5.0e3,
            ),
            proton_density=_l1_bounded_measurement(
                target.row, :proton_density, 0.0, 1.0e3,
            ),
            proton_vx_gse=_l1_bounded_measurement(
                target.row, :proton_vx_gse, -5.0e3, 5.0e3,
            ),
        )
    else
        throw(ArgumentError("static stack L1 pair candidate has an unknown kind"))
    end
    any(isnothing, values(drivers)) && error(
        "static stack L1 admitted row contains an invalid required driver",
    )

    position = _l1_pair_position(metadata)
    position.source == target.source || error(
        "static stack L1 ephemeris source does not match its admitted row",
    )
    position.timestamp_utc == timestamp_utc || error(
        "static stack L1 ephemeris timestamp does not match its admitted row",
    )
    String(metadata.ephemeris_record_sha256) ==
        _l1_sha256(codeunits(String(metadata.ephemeris_record_json))) || error(
        "static stack L1 paired ephemeris record checksum changed",
    )
    row_sha256 = _l1_sha256(codeunits(JSON3.write(target.row)))
    candidate = (
        kind=kind,
        timestamp=target.timestamp,
        timestamp_utc=timestamp_utc,
        source=target.source,
        receipt=receipt,
        receipt_completed_utc=String(record.receipt_completed_utc),
        sequence=Int(record.sequence),
        record_relative_path=_l1_record_relative(
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
        item -> item.timestamp == candidate.timestamp &&
                item.source == candidate.source,
        observations,
    )
    length(target_observations) == 1 &&
        only(target_observations).status == "unique_measurement_row" &&
        only(target_observations).target_row_sha256 ==
            candidate.target_row_sha256 || error(
        "static stack L1 admitted target disagrees with its latest raw row",
    )
    return (candidate=candidate, observations=observations)
end

# True when a verified record cannot contain a row stamped at or after
# `scan_start`: received before it and with an earlier maximum time_tag.
# Unparseable maximum time tags are treated as possibly recent.
function _l1_record_precedes_scan(record, scan_start::DateTime)
    _l1_parse_utc(record.receipt_completed_utc) < scan_start || return false
    text = String(record.maximum_time_tag)
    isempty(text) && return true
    maximum_time = try
        _l1_parse_source_utc(text)
    catch error
        error isa InterruptException && rethrow()
        return false
    end
    return maximum_time < scan_start
end

function _l1_pair_candidates(root::AbstractString, source, count::Integer,
                                 kind::Symbol, issue_time::DateTime;
                                 scan_start::DateTime=typemin(DateTime))
    candidates = NamedTuple[]
    observations = NamedTuple[]
    # Walk backward from the verified cutoff head (receipts are non-decreasing
    # along the chain) and stop at the first record that can contribute neither
    # a measurement minute nor revision evidence at or after `scan_start`.
    for sequence in Int(count):-1:1
        relative = _l1_record_relative(source.name, sequence)
        path = _l1_resolve_relative(root, relative, "paired record path")
        _l1_count_read!(:record)
        record = JSON3.read(read(path, String))
        _l1_record_sha256(record) == String(record.record_sha256) || error(
            "static stack L1 paired record checksum changed after verification",
        )
        _l1_record_precedes_scan(record, scan_start) && break
        result = _l1_pair_candidate(
            root, source, record, kind, issue_time,
        )
        result.candidate === nothing || push!(candidates, result.candidate)
        append!(observations, result.observations)
    end
    return (admitted=reverse!(candidates), observations=observations)
end

function _l1_stable_candidate(candidates, observations,
                                  timestamp::DateTime,
                                  source::AbstractString,
                                  label::AbstractString)
    matches = filter(
        candidate -> candidate.timestamp == timestamp &&
                     candidate.source == source,
        candidates,
    )
    isempty(matches) && error("static stack L1 $label pair candidate is missing")
    fingerprints = unique((
        candidate.target_row_sha256,
        candidate.quality_value,
        candidate.quality_decision,
        candidate.quality_required_fields_status,
        candidate.ephemeris_record_sha256,
    ) for candidate in matches)
    length(fingerprints) == 1 || error(
        "static stack L1 $label row has conflicting pre-issue revisions",
    )
    observed = filter(
        item -> item.timestamp == timestamp && item.source == source,
        observations,
    )
    isempty(observed) && error(
        "static stack L1 $label row lacks raw revision evidence",
    )
    raw_fingerprints = unique((
        item.status, item.source, item.target_row_sha256,
    ) for item in observed)
    length(raw_fingerprints) == 1 || error(
        "static stack L1 $label row has conflicting admitted or rejected revisions",
    )
    raw_fingerprint = only(raw_fingerprints)
    raw_fingerprint[1] == "unique_measurement_row" &&
        raw_fingerprint[2] == source &&
        raw_fingerprint[3] == first(matches).target_row_sha256 || error(
        "static stack L1 $label row revision no longer matches its admitted evidence",
    )
    sort!(matches; by=candidate -> (candidate.receipt, candidate.sequence))
    return last(matches)
end

function _l1_assert_candidate_unchanged(root::AbstractString, candidate)
    record_path = _l1_resolve_relative(
        root, candidate.record_relative_path, "selected paired record path",
    )
    record = JSON3.read(read(record_path, String))
    _l1_record_sha256(record) == candidate.record_sha256 || error(
        "static stack L1 selected record changed during pairing",
    )
    raw_path = _l1_resolve_relative(
        root, candidate.raw_relative_path, "selected paired raw path",
    )
    _l1_sha256(read(raw_path)) == candidate.raw_sha256 || error(
        "static stack L1 selected raw response changed during pairing",
    )
    ephemeris_path = _l1_resolve_relative(
        root, candidate.ephemeris_source_object_raw_relative_path,
        "selected paired ephemeris path",
    )
    _l1_sha256(read(ephemeris_path)) ==
        candidate.ephemeris_source_object_sha256 || error(
        "static stack L1 selected ephemeris source object changed during pairing",
    )
    return nothing
end

function _l1_pair_floor_30(time::DateTime)
    minute = Dates.minute(time) < 30 ? 0 : 30
    return DateTime(
        Dates.year(time), Dates.month(time), Dates.day(time), Dates.hour(time),
        minute,
    )
end

function _l1_pair_ceil_30(time::DateTime)
    floored = _l1_pair_floor_30(time)
    return floored == time ? floored : floored + Minute(30)
end

function _l1_issue_pair_output(mag_selected, wind_selected,
                                   issue_time_utc::DateTime,
                                   cutoff)
    mag_selected.source == wind_selected.source &&
        mag_selected.source in L1_ADMISSIBLE_SOURCES || error(
        "static stack L1 paired records changed spacecraft identity",
    )
    mag_selected.timestamp_utc == wind_selected.timestamp_utc || error(
        "static stack L1 paired records changed measurement timestamp",
    )
    mag_selected.ephemeris_record_sha256 ==
        wind_selected.ephemeris_record_sha256 || error(
        "static stack L1 paired records disagree on the bound GSE position",
    )
    position = mag_selected.position
    first_eligible = _l1_pair_ceil_30(max(
        mag_selected.receipt, wind_selected.receipt,
    ))
    payload = (
        schema_version=L1_ISSUE_PAIR_SCHEMA_VERSION,
        issue_time_utc=_l1_utc(issue_time_utc),
        first_eligible_issue_time_utc=_l1_utc(first_eligible),
        issue_cutoff_relative_path=cutoff.cutoff_relative_path,
        issue_cutoff_sha256=cutoff.cutoff_sha256,
        measurement_time_utc=mag_selected.timestamp_utc,
        source=mag_selected.source,
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
        pair_contract_sha256=_l1_sha256(
            codeunits(JSON3.write(payload)),
        ),
    ))
end

function _select_l1_issue_pairs_unlocked(
        storage::AbstractString,
        issue_time_utc::DateTime,
        measurement_start_utc::DateTime,
        cutoff_relative_path::AbstractString;
        latest_only::Bool=false)
    measurement_start_utc <= issue_time_utc || throw(ArgumentError(
        "static stack L1 measurement window starts after its issue time",
    ))
    sources = L1_RECEIPT_SOURCES
    # Prefix verification reuses the in-process verified-prefix memo; a
    # missing, stale, or contradicted memo falls back to full verification.
    before = _l1_verify_issue_cutoff(
        storage, issue_time_utc, cutoff_relative_path, sources; memo=true,
    )
    counts = Dict(result.source_name => result.records for result in before.prefixes)
    # Latest-only selection considers measurement minutes within the documented
    # lookback of the issue; windowed selection is bounded by its window start.
    window_start = latest_only ?
        max(measurement_start_utc, issue_time_utc - L1_PAIR_LOOKBACK) :
        measurement_start_utc
    mag = _l1_pair_candidates(
        storage, sources[1], counts[sources[1].name], :mag, issue_time_utc;
        scan_start=window_start,
    )
    wind = _l1_pair_candidates(
        storage, sources[2], counts[sources[2].name], :wind, issue_time_utc;
        scan_start=window_start,
    )
    # Magnetometer and wind rows of a pair must come from the same admitted
    # source; one minute admitted from two sources fails closed.
    common = intersect(
        Set((candidate.timestamp, candidate.source) for candidate in mag.admitted),
        Set((candidate.timestamp, candidate.source) for candidate in wind.admitted),
    )
    pair_keys = sort!(collect(filter(key -> key[1] >= window_start, common)))
    isempty(pair_keys) && error(
        "static stack L1 archive has no exact admitted mag/wind timestamp in the requested causal window",
    )
    if latest_only
        latest_time = last(pair_keys)[1]
        pair_keys = filter(key -> key[1] == latest_time, pair_keys)
    end
    length(unique(first.(pair_keys))) == length(pair_keys) || error(
        "static stack L1 measurement minute is admitted from more than one source",
    )
    selected = Tuple((
        mag=_l1_stable_candidate(
            mag.admitted, mag.observations, timestamp, source, "magnetometer",
        ),
        wind=_l1_stable_candidate(
            wind.admitted, wind.observations, timestamp, source, "wind",
        ),
    ) for (timestamp, source) in pair_keys)

    after = _l1_verify_issue_cutoff(
        storage, issue_time_utc, cutoff_relative_path, sources; memo=true,
    )
    before == after || error("static stack L1 archive changed during issue pairing")
    for pair in selected
        _l1_assert_candidate_unchanged(storage, pair.mag)
        _l1_assert_candidate_unchanged(storage, pair.wind)
    end
    return Tuple(
        _l1_issue_pair_output(
            pair.mag, pair.wind, issue_time_utc, before,
        )
        for pair in selected
    )
end

function _select_l1_issue_pair_unlocked(storage::AbstractString,
                                             issue_time_utc::DateTime,
                                             cutoff_relative_path::AbstractString)
    pairs = _select_l1_issue_pairs_unlocked(
        storage, issue_time_utc, typemin(DateTime), cutoff_relative_path;
        latest_only=true,
    )
    return last(pairs)
end

"""
    select_l1_issue_pair(root, issue_time_utc::DateTime; lock_timeout_sec=30.0)

Acquire the collector lock, verify the live receipt archive (incrementally from
the in-process verified-prefix memo when it is valid, otherwise in full), save
its immutable issue cutoff, and return the latest common, individually admitted
magnetometer/wind row from one allow-listed source (`DSCOVR` or `SOLAR1`)
received no later than `issue_time_utc`, with a measurement minute within
`L1_PAIR_LOOKBACK` (2 h) of the issue. The three-argument overload requires
an existing cutoff and reads only its exact source prefixes. Neither overload
interpolates measurement values or performs network access.
"""
function select_l1_issue_pair(root::AbstractString,
                                   issue_time_utc::DateTime;
                                   lock_timeout_sec::Real=30.0)
    storage = _l1_validate_root(root; create=false)
    return _l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        sources = L1_RECEIPT_SOURCES
        cutoff = _l1_capture_issue_cutoff_unlocked(
            storage, issue_time_utc, sources,
        )
        _select_l1_issue_pair_unlocked(
            storage, issue_time_utc, cutoff.cutoff_relative_path,
        )
    end
end

function select_l1_issue_pair(
        root::AbstractString,
        issue_time_utc::DateTime,
        cutoff_relative_path::AbstractString;
        lock_timeout_sec::Real=30.0)
    storage = _l1_validate_root(root; create=false)
    return _l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        _select_l1_issue_pair_unlocked(
            storage, issue_time_utc, cutoff_relative_path,
        )
    end
end


"""
    select_l1_issue_pairs(root, issue_time_utc::DateTime;
                               measurement_start_utc, lock_timeout_sec=30.0)

Under one collector lock, build the verified magnetometer and wind candidate
sets once per call and return every stable exact same-source minute (`DSCOVR`
or `SOLAR1`) in the inclusive measurement-time window. The result is chronological and contains only records
received by the issue. The live overload first saves a verified issue
cutoff; the cutoff-bound overload verifies and reads only those saved prefixes.
"""
function select_l1_issue_pairs(
        root::AbstractString,
        issue_time_utc::DateTime;
        measurement_start_utc::DateTime,
        lock_timeout_sec::Real=30.0)
    storage = _l1_validate_root(root; create=false)
    return _l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        cutoff = _l1_capture_issue_cutoff_unlocked(
            storage, issue_time_utc, L1_RECEIPT_SOURCES,
        )
        _select_l1_issue_pairs_unlocked(
            storage, issue_time_utc, measurement_start_utc,
            cutoff.cutoff_relative_path,
        )
    end
end


function select_l1_issue_pairs(
        root::AbstractString,
        issue_time_utc::DateTime,
        cutoff_relative_path::AbstractString;
        measurement_start_utc::DateTime,
        lock_timeout_sec::Real=30.0)
    storage = _l1_validate_root(root; create=false)
    return _l1_with_lock(storage; timeout_sec=lock_timeout_sec) do
        _select_l1_issue_pairs_unlocked(
            storage, issue_time_utc, measurement_start_utc,
            cutoff_relative_path,
        )
    end
end

end # module StaticStackL1ReceiptPairing
