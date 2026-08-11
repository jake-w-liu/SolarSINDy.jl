# Receipt-causal L1 transport and arrival-queue construction for V2.2-M2.

import SHA
import JSON3

"Frozen receipt-pair contract accepted by the V2.2-M2 arrival queue."
const OPERATIONAL_V22_ARRIVAL_PAIR_SCHEMA_VERSION = "v2_2_l1_issue_pair_v2"
const OPERATIONAL_V22_ARRIVAL_SCHEMA_VERSION = "operational_v2_2_m2_arrival_v2"
const OPERATIONAL_V22_ARRIVAL_PATH_SCHEMA_VERSION =
    "operational_v2_2_m2_arrival_candidate_path_v2"
const OPERATIONAL_V22_ARRIVAL_PATH_GATE_STATUS = :ungated_candidate
const OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES = 30
const OPERATIONAL_V22_ARRIVAL_TRAILING_MINUTES = 15
const OPERATIONAL_V22_ARRIVAL_HISTORY_ROWS = 25
const OPERATIONAL_V22_ARRIVAL_PATH_STEPS = 14
const OPERATIONAL_V22_ARRIVAL_MIN_DELAY_MINUTES = 20
const OPERATIONAL_V22_ARRIVAL_MAX_DELAY_MINUTES = 120
const OPERATIONAL_V22_ARRIVAL_MAX_FRESHNESS_MINUTES = 90
const OPERATIONAL_V22_ARRIVAL_X_REF_GSE_KM = 0.0
const OPERATIONAL_V22_ARRIVAL_V21_COMPATIBILITY_DISTANCE_KM = 1.5e6

const _OPERATIONAL_V22_ARRIVAL_PAIR_PAYLOAD_FIELDS = (
    :schema_version,
    :issue_time_utc,
    :first_eligible_issue_time_utc,
    :issue_cutoff_relative_path,
    :issue_cutoff_sha256,
    :measurement_time_utc,
    :source,
    :mag_source_product_id,
    :wind_source_product_id,
    :magnetic_component_frame,
    :magnetic_component_units,
    :proton_speed_units,
    :proton_density_units,
    :proton_vx_frame,
    :proton_vx_units,
    :bx_gsm,
    :by_gsm,
    :bz_gsm,
    :proton_speed,
    :proton_density,
    :proton_vx_gse,
    :position_x_gse,
    :position_y_gse,
    :position_z_gse,
    :position_frame,
    :position_units,
    :position_method,
    :position_lower_time_utc,
    :position_upper_time_utc,
    :position_interpolation_fraction,
    :ephemeris_record_sha256,
    :mag_quality_source_product,
    :mag_quality_value,
    :mag_quality_binding_status,
    :mag_quality_decision,
    :mag_quality_required_fields_status,
    :mag_sequence,
    :mag_receipt_completed_utc,
    :mag_record_sha256,
    :mag_raw_sha256,
    :mag_ephemeris_source_object_sha256,
    :wind_quality_source_product,
    :wind_quality_value,
    :wind_quality_binding_status,
    :wind_quality_decision,
    :wind_quality_required_fields_status,
    :wind_sequence,
    :wind_receipt_completed_utc,
    :wind_record_sha256,
    :wind_raw_sha256,
    :wind_ephemeris_source_object_sha256,
)
const _OPERATIONAL_V22_ARRIVAL_PAIR_FIELDS = (
    _OPERATIONAL_V22_ARRIVAL_PAIR_PAYLOAD_FIELDS...,
    :pair_contract_sha256,
)

"Immutable physical state assigned to one half-open UTC arrival bin."
struct OperationalV22ArrivalBin
    start_utc::DateTime
    end_utc::DateTime
    bx_gsm::Float64
    by_gsm::Float64
    bz_gsm::Float64
    proton_speed::Float64
    proton_density::Float64
    observed::Bool
    filled_from_start_utc::Union{Nothing,DateTime}
    contributing_pair_sha256::Tuple
end

"Immutable receipt-causal transport result and sparse-driver seed queue."
struct OperationalV22ArrivalQueue
    schema_version::String
    issue_time_utc::DateTime
    status::Symbol
    fallback_reason::Symbol
    x_ref_gse_km::Float64
    v21_compatibility_distance_km::Float64
    receipt_pairs::Tuple
    transported_pairs::Tuple
    arrival_bins::Tuple
    history_bins::Tuple
    future_bins::Tuple
    composite_sha256::String
end

"One physical half-hour state in the M2 queue-plus-sparse output path."
struct OperationalV22ArrivalPathStep
    start_utc::DateTime
    end_utc::DateTime
    bx_gsm::Float64
    by_gsm::Float64
    bz_gsm::Float64
    proton_speed::Float64
    proton_density::Float64
    origin::Symbol
    contributing_pair_sha256::Tuple
end

"Immutable fourteen-step Stage-A/Stage-B handoff bound to queue and artifact."
struct OperationalV22ArrivalPath
    schema_version::String
    gate_status::Symbol
    issue_time_utc::DateTime
    status::Symbol
    fallback_reason::Symbol
    queue_sha256::String
    artifact_sha256::String
    steps::Tuple
    composite_sha256::String
end

_operational_v22_arrival_utc(time::DateTime) =
    Dates.format(time, dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z"

function _operational_v22_arrival_parse_utc(value, field::AbstractString)
    value isa AbstractString || throw(ArgumentError(
        "V2.2-M2 $field must be a canonical UTC string",
    ))
    text = String(value)
    endswith(text, "Z") || throw(ArgumentError(
        "V2.2-M2 $field must end in Z",
    ))
    parsed = try
        DateTime(chop(text; tail=1))
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError("V2.2-M2 $field is not a valid UTC timestamp"))
    end
    _operational_v22_arrival_utc(parsed) == text || throw(ArgumentError(
        "V2.2-M2 $field is not canonical to millisecond precision",
    ))
    return parsed
end

function _operational_v22_arrival_float(value, field::AbstractString;
                                        finite::Bool=true)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "V2.2-M2 $field must be a real number",
    ))
    converted = try
        Float64(value)
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError("V2.2-M2 $field cannot be represented as Float64"))
    end
    (!finite || isfinite(converted)) || throw(ArgumentError(
        "V2.2-M2 $field must be finite",
    ))
    return converted
end

function _operational_v22_arrival_positive_int(value, field::AbstractString)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "V2.2-M2 $field must be a positive integer",
    ))
    converted = try
        Int(value)
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError("V2.2-M2 $field is outside the Int range"))
    end
    converted > 0 || throw(ArgumentError(
        "V2.2-M2 $field must be positive",
    ))
    return converted
end

function _operational_v22_arrival_nonnegative_int(value,
                                                   field::AbstractString)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "V2.2-M2 $field must be a nonnegative integer",
    ))
    converted = try
        Int(value)
    catch error
        error isa InterruptException && rethrow()
        throw(ArgumentError("V2.2-M2 $field is outside the Int range"))
    end
    converted >= 0 || throw(ArgumentError(
        "V2.2-M2 $field must be nonnegative",
    ))
    return converted
end

function _operational_v22_arrival_string(value, field::AbstractString)
    value isa AbstractString || throw(ArgumentError(
        "V2.2-M2 $field must be a string",
    ))
    return String(value)
end

function _operational_v22_arrival_hash(value, field::AbstractString)
    text = _operational_v22_arrival_string(value, field)
    occursin(r"^[0-9a-f]{64}$", text) || throw(ArgumentError(
        "V2.2-M2 $field must be a lowercase SHA-256 digest",
    ))
    return text
end

function _operational_v22_arrival_require_pair_fields(pair)
    pair isa NamedTuple || throw(ArgumentError(
        "V2.2-M2 receipt pairs must be NamedTuples returned by the v2 selector",
    ))
    names = keys(pair)
    length(names) == length(_OPERATIONAL_V22_ARRIVAL_PAIR_FIELDS) &&
        all(name -> name in names, _OPERATIONAL_V22_ARRIVAL_PAIR_FIELDS) ||
        throw(ArgumentError("V2.2-M2 receipt-pair field contract changed"))
    return nothing
end

function _operational_v22_arrival_pair_payload(pair)
    return NamedTuple{_OPERATIONAL_V22_ARRIVAL_PAIR_PAYLOAD_FIELDS}(
        Tuple(getproperty(pair, name)
              for name in _OPERATIONAL_V22_ARRIVAL_PAIR_PAYLOAD_FIELDS),
    )
end

function _operational_v22_arrival_pair_contract_sha256(pair)
    return bytes2hex(SHA.sha256(
        codeunits(JSON3.write(_operational_v22_arrival_pair_payload(pair))),
    ))
end

function _operational_v22_arrival_cutoff_relative(issue_time::DateTime)
    token = Dates.format(issue_time, dateformat"yyyymmddTHHMMSSsss") * "Z"
    return joinpath("issue_cutoffs", token * ".json")
end

function _operational_v22_arrival_hash_value(io::IO, value)
    if value === nothing
        print(io, "nothing|")
    elseif value isa Float64
        print(io, "f64:", bitstring(value), '|')
    elseif value isa DateTime
        print(io, "datetime:", Dates.value(value), '|')
    elseif value isa Symbol
        text = String(value)
        print(io, "symbol:", ncodeunits(text), ':', text, '|')
    elseif value isa AbstractString
        text = String(value)
        print(io, "string:", ncodeunits(text), ':', text, '|')
    elseif value isa Integer && !(value isa Bool)
        print(io, "int:", value, '|')
    elseif value isa Bool
        print(io, value ? "bool:1|" : "bool:0|")
    elseif value isa NamedTuple
        print(io, "named:", length(value), '|')
        for name in keys(value)
            _operational_v22_arrival_hash_value(io, name)
            _operational_v22_arrival_hash_value(io, getproperty(value, name))
        end
    elseif value isa Tuple
        print(io, "tuple:", length(value), '|')
        for item in value
            _operational_v22_arrival_hash_value(io, item)
        end
    elseif value isa OperationalV22ArrivalBin
        _operational_v22_arrival_hash_value(io, (
            start_utc=value.start_utc,
            end_utc=value.end_utc,
            bx_gsm=value.bx_gsm,
            by_gsm=value.by_gsm,
            bz_gsm=value.bz_gsm,
            proton_speed=value.proton_speed,
            proton_density=value.proton_density,
            observed=value.observed,
            filled_from_start_utc=value.filled_from_start_utc,
            contributing_pair_sha256=value.contributing_pair_sha256,
        ))
    elseif value isa OperationalV22ArrivalPathStep
        _operational_v22_arrival_hash_value(io, (
            start_utc=value.start_utc,
            end_utc=value.end_utc,
            bx_gsm=value.bx_gsm,
            by_gsm=value.by_gsm,
            bz_gsm=value.bz_gsm,
            proton_speed=value.proton_speed,
            proton_density=value.proton_density,
            origin=value.origin,
            contributing_pair_sha256=value.contributing_pair_sha256,
        ))
    else
        throw(ArgumentError(
            "V2.2-M2 checksum received unsupported value type $(typeof(value))",
        ))
    end
    return nothing
end

function _operational_v22_arrival_digest(values...)
    io = IOBuffer()
    for value in values
        _operational_v22_arrival_hash_value(io, value)
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _operational_v22_arrival_pair_sha256(pair)
    payload = NamedTuple{_OPERATIONAL_V22_ARRIVAL_PAIR_FIELDS}(
        Tuple(getproperty(pair, name) for name in _OPERATIONAL_V22_ARRIVAL_PAIR_FIELDS),
    )
    return _operational_v22_arrival_digest(
        OPERATIONAL_V22_ARRIVAL_PAIR_SCHEMA_VERSION, payload,
    )
end

function _operational_v22_arrival_normalize_pair(pair)
    _operational_v22_arrival_require_pair_fields(pair)
    stored_pair_contract_sha = _operational_v22_arrival_hash(
        pair.pair_contract_sha256, "pair_contract_sha256",
    )
    _operational_v22_arrival_pair_contract_sha256(pair) ==
        stored_pair_contract_sha || throw(ArgumentError(
        "V2.2-M2 receipt-pair contract checksum mismatch",
    ))
    schema = _operational_v22_arrival_string(pair.schema_version, "schema_version")
    schema == OPERATIONAL_V22_ARRIVAL_PAIR_SCHEMA_VERSION || throw(ArgumentError(
        "V2.2-M2 receipt-pair schema is unsupported",
    ))
    pair_issue = _operational_v22_arrival_parse_utc(
        pair.issue_time_utc, "pair issue_time_utc",
    )
    _operational_v22_arrival_floor(pair_issue) == pair_issue ||
        throw(ArgumentError(
            "V2.2-M2 pair issue time must lie on the 30-minute UTC grid",
        ))
    first_eligible = _operational_v22_arrival_parse_utc(
        pair.first_eligible_issue_time_utc,
        "first_eligible_issue_time_utc",
    )
    cutoff_relative = _operational_v22_arrival_string(
        pair.issue_cutoff_relative_path, "issue_cutoff_relative_path",
    )
    cutoff_relative == normpath(cutoff_relative) &&
        cutoff_relative == _operational_v22_arrival_cutoff_relative(pair_issue) ||
        throw(ArgumentError(
        "V2.2-M2 receipt-pair cutoff path does not match its issue",
    ))
    cutoff_sha = _operational_v22_arrival_hash(
        pair.issue_cutoff_sha256, "issue_cutoff_sha256",
    )
    measurement = _operational_v22_arrival_parse_utc(
        pair.measurement_time_utc, "measurement_time_utc",
    )
    mag_receipt = _operational_v22_arrival_parse_utc(
        pair.mag_receipt_completed_utc, "mag_receipt_completed_utc",
    )
    wind_receipt = _operational_v22_arrival_parse_utc(
        pair.wind_receipt_completed_utc, "wind_receipt_completed_utc",
    )
    mag_receipt <= pair_issue && wind_receipt <= pair_issue || throw(ArgumentError(
        "V2.2-M2 selected record was not received by its pair issue time",
    ))
    measurement <= mag_receipt && measurement <= wind_receipt || throw(ArgumentError(
        "V2.2-M2 measurement timestamp is later than a selected receipt",
    ))
    first_eligible == _operational_v22_arrival_ceil_30(max(
        mag_receipt, wind_receipt,
    )) || throw(ArgumentError(
        "V2.2-M2 first eligible issue time is inconsistent with receipts",
    ))
    first_eligible <= pair_issue || throw(ArgumentError(
        "V2.2-M2 receipt pair predates its first eligible issue",
    ))

    source = _operational_v22_arrival_string(pair.source, "source")
    source == "DSCOVR" || throw(ArgumentError(
        "V2.2-M2 receipt pair is not from DSCOVR",
    ))
    mag_source_product = _operational_v22_arrival_string(
        pair.mag_source_product_id, "mag_source_product_id",
    )
    mag_source_product == "swpc_rtsw_mag_1m" || throw(ArgumentError(
        "V2.2-M2 magnetometer source product changed",
    ))
    wind_source_product = _operational_v22_arrival_string(
        pair.wind_source_product_id, "wind_source_product_id",
    )
    wind_source_product == "swpc_rtsw_wind_1m" || throw(ArgumentError(
        "V2.2-M2 wind source product changed",
    ))
    magnetic_frame = _operational_v22_arrival_string(
        pair.magnetic_component_frame, "magnetic_component_frame",
    )
    magnetic_frame == "GSM" || throw(ArgumentError(
        "V2.2-M2 magnetic-component frame must be GSM",
    ))
    magnetic_units = _operational_v22_arrival_string(
        pair.magnetic_component_units, "magnetic_component_units",
    )
    magnetic_units == "nT" || throw(ArgumentError(
        "V2.2-M2 magnetic-component units must be nT",
    ))
    speed_units = _operational_v22_arrival_string(
        pair.proton_speed_units, "proton_speed_units",
    )
    speed_units == "km/s" || throw(ArgumentError(
        "V2.2-M2 proton-speed units must be km/s",
    ))
    density_units = _operational_v22_arrival_string(
        pair.proton_density_units, "proton_density_units",
    )
    density_units == "cm^-3" || throw(ArgumentError(
        "V2.2-M2 proton-density units must be cm^-3",
    ))
    vx_frame = _operational_v22_arrival_string(
        pair.proton_vx_frame, "proton_vx_frame",
    )
    vx_frame == "GSE" || throw(ArgumentError(
        "V2.2-M2 proton Vx frame must be GSE",
    ))
    vx_units = _operational_v22_arrival_string(
        pair.proton_vx_units, "proton_vx_units",
    )
    vx_units == "km/s" || throw(ArgumentError(
        "V2.2-M2 proton Vx units must be km/s",
    ))
    frame = _operational_v22_arrival_string(pair.position_frame, "position_frame")
    frame == "GSE" || throw(ArgumentError(
        "V2.2-M2 position frame must be exactly GSE",
    ))
    units = _operational_v22_arrival_string(pair.position_units, "position_units")
    units == "km" || throw(ArgumentError(
        "V2.2-M2 position units must be exactly km",
    ))
    method = _operational_v22_arrival_string(pair.position_method, "position_method")
    method in ("exact", "linear") || throw(ArgumentError(
        "V2.2-M2 position interpolation method changed",
    ))
    lower = _operational_v22_arrival_parse_utc(
        pair.position_lower_time_utc, "position_lower_time_utc",
    )
    upper = _operational_v22_arrival_parse_utc(
        pair.position_upper_time_utc, "position_upper_time_utc",
    )
    fraction = _operational_v22_arrival_float(
        pair.position_interpolation_fraction,
        "position_interpolation_fraction",
    )
    lower <= measurement <= upper || throw(ArgumentError(
        "V2.2-M2 position bracket does not contain the measurement time",
    ))
    upper - lower <= Hour(1) || throw(ArgumentError(
        "V2.2-M2 position bracket exceeds one hour",
    ))
    if method == "exact"
        lower == measurement == upper && fraction == 0.0 || throw(ArgumentError(
            "V2.2-M2 exact position metadata is inconsistent",
        ))
    else
        lower < measurement < upper || throw(ArgumentError(
            "V2.2-M2 linear position metadata does not strictly bracket the row",
        ))
        expected = Dates.value(measurement - lower) / Dates.value(upper - lower)
        isapprox(fraction, expected; atol=0.0, rtol=8eps(Float64)) ||
            throw(ArgumentError(
                "V2.2-M2 position interpolation fraction is inconsistent",
            ))
    end

    mag_quality_product = _operational_v22_arrival_string(
        pair.mag_quality_source_product, "mag_quality_source_product",
    )
    mag_quality_product == "dscovr_m1m" || throw(ArgumentError(
        "V2.2-M2 magnetometer quality product changed",
    ))
    wind_quality_product = _operational_v22_arrival_string(
        pair.wind_quality_source_product, "wind_quality_source_product",
    )
    wind_quality_product == "dscovr_f1m" || throw(ArgumentError(
        "V2.2-M2 wind quality product changed",
    ))
    mag_quality_value = _operational_v22_arrival_nonnegative_int(
        pair.mag_quality_value, "mag_quality_value",
    )
    wind_quality_value = _operational_v22_arrival_nonnegative_int(
        pair.wind_quality_value, "wind_quality_value",
    )
    mag_quality_value == 0 && wind_quality_value == 0 || throw(ArgumentError(
        "V2.2-M2 receipt pair is not normal quality",
    ))
    mag_quality_binding = _operational_v22_arrival_string(
        pair.mag_quality_binding_status, "mag_quality_binding_status",
    )
    wind_quality_binding = _operational_v22_arrival_string(
        pair.wind_quality_binding_status, "wind_quality_binding_status",
    )
    mag_quality_binding == wind_quality_binding ==
        "bound_noaa_dscovr_overall_quality" || throw(ArgumentError(
        "V2.2-M2 receipt pair lacks bound NOAA DSCOVR quality",
    ))
    mag_quality_decision = _operational_v22_arrival_string(
        pair.mag_quality_decision, "mag_quality_decision",
    )
    wind_quality_decision = _operational_v22_arrival_string(
        pair.wind_quality_decision, "wind_quality_decision",
    )
    mag_quality_decision == wind_quality_decision ==
        "accept_normal_overall_quality" || throw(ArgumentError(
        "V2.2-M2 receipt-pair quality decision is not accepting",
    ))
    mag_required_status = _operational_v22_arrival_string(
        pair.mag_quality_required_fields_status,
        "mag_quality_required_fields_status",
    )
    mag_required_status == "bound_required_bx_by_bz_gsm" ||
        throw(ArgumentError(
            "V2.2-M2 magnetometer required-field status changed",
        ))
    wind_required_status = _operational_v22_arrival_string(
        pair.wind_quality_required_fields_status,
        "wind_quality_required_fields_status",
    )
    wind_required_status == "bound_required_speed_density_vx_gse" ||
        throw(ArgumentError("V2.2-M2 wind required-field status changed"))

    normalized = (
        schema_version=schema,
        issue_time_utc=pair_issue,
        first_eligible_issue_time_utc=first_eligible,
        issue_cutoff_relative_path=cutoff_relative,
        issue_cutoff_sha256=cutoff_sha,
        measurement_time_utc=measurement,
        source=source,
        mag_source_product_id=mag_source_product,
        wind_source_product_id=wind_source_product,
        magnetic_component_frame=magnetic_frame,
        magnetic_component_units=magnetic_units,
        proton_speed_units=speed_units,
        proton_density_units=density_units,
        proton_vx_frame=vx_frame,
        proton_vx_units=vx_units,
        bx_gsm=_operational_v22_arrival_float(pair.bx_gsm, "bx_gsm"),
        by_gsm=_operational_v22_arrival_float(pair.by_gsm, "by_gsm"),
        bz_gsm=_operational_v22_arrival_float(pair.bz_gsm, "bz_gsm"),
        proton_speed=_operational_v22_arrival_float(
            pair.proton_speed, "proton_speed"; finite=false,
        ),
        proton_density=_operational_v22_arrival_float(
            pair.proton_density, "proton_density"; finite=false,
        ),
        proton_vx_gse=_operational_v22_arrival_float(
            pair.proton_vx_gse, "proton_vx_gse"; finite=false,
        ),
        position_x_gse=_operational_v22_arrival_float(
            pair.position_x_gse, "position_x_gse",
        ),
        position_y_gse=_operational_v22_arrival_float(
            pair.position_y_gse, "position_y_gse",
        ),
        position_z_gse=_operational_v22_arrival_float(
            pair.position_z_gse, "position_z_gse",
        ),
        position_frame=frame,
        position_units=units,
        position_method=method,
        position_lower_time_utc=lower,
        position_upper_time_utc=upper,
        position_interpolation_fraction=fraction,
        ephemeris_record_sha256=_operational_v22_arrival_hash(
            pair.ephemeris_record_sha256, "ephemeris_record_sha256",
        ),
        mag_quality_source_product=mag_quality_product,
        mag_quality_value=mag_quality_value,
        mag_quality_binding_status=mag_quality_binding,
        mag_quality_decision=mag_quality_decision,
        mag_quality_required_fields_status=mag_required_status,
        mag_sequence=_operational_v22_arrival_positive_int(
            pair.mag_sequence, "mag_sequence",
        ),
        mag_receipt_completed_utc=mag_receipt,
        mag_record_sha256=_operational_v22_arrival_hash(
            pair.mag_record_sha256, "mag_record_sha256",
        ),
        mag_raw_sha256=_operational_v22_arrival_hash(
            pair.mag_raw_sha256, "mag_raw_sha256",
        ),
        mag_ephemeris_source_object_sha256=_operational_v22_arrival_hash(
            pair.mag_ephemeris_source_object_sha256,
            "mag_ephemeris_source_object_sha256",
        ),
        wind_quality_source_product=wind_quality_product,
        wind_quality_value=wind_quality_value,
        wind_quality_binding_status=wind_quality_binding,
        wind_quality_decision=wind_quality_decision,
        wind_quality_required_fields_status=wind_required_status,
        wind_sequence=_operational_v22_arrival_positive_int(
            pair.wind_sequence, "wind_sequence",
        ),
        wind_receipt_completed_utc=wind_receipt,
        wind_record_sha256=_operational_v22_arrival_hash(
            pair.wind_record_sha256, "wind_record_sha256",
        ),
        wind_raw_sha256=_operational_v22_arrival_hash(
            pair.wind_raw_sha256, "wind_raw_sha256",
        ),
        wind_ephemeris_source_object_sha256=_operational_v22_arrival_hash(
            pair.wind_ephemeris_source_object_sha256,
            "wind_ephemeris_source_object_sha256",
        ),
        pair_contract_sha256=stored_pair_contract_sha,
    )
    canonical_contract = _operational_v22_arrival_pair_contract(normalized)
    _operational_v22_arrival_pair_contract_sha256(canonical_contract) ==
        stored_pair_contract_sha || throw(ArgumentError(
        "V2.2-M2 receipt-pair contract is not canonically typed",
    ))
    return merge(normalized, (
        pair_sha256=_operational_v22_arrival_pair_sha256(normalized),
    ))
end

function _operational_v22_arrival_pair_contract(pair)
    date_fields = (
        :issue_time_utc,
        :first_eligible_issue_time_utc,
        :measurement_time_utc,
        :position_lower_time_utc,
        :position_upper_time_utc,
        :mag_receipt_completed_utc,
        :wind_receipt_completed_utc,
    )
    payload = NamedTuple{_OPERATIONAL_V22_ARRIVAL_PAIR_PAYLOAD_FIELDS}(
        Tuple(name in date_fields ?
              _operational_v22_arrival_utc(getproperty(pair, name)) :
              getproperty(pair, name)
              for name in _OPERATIONAL_V22_ARRIVAL_PAIR_PAYLOAD_FIELDS),
    )
    return merge(payload, (pair_contract_sha256=pair.pair_contract_sha256,))
end

function _operational_v22_arrival_pair_fingerprint(pair)
    return (
        pair.schema_version,
        pair.source,
        pair.mag_source_product_id,
        pair.wind_source_product_id,
        pair.magnetic_component_frame,
        pair.magnetic_component_units,
        pair.proton_speed_units,
        pair.proton_density_units,
        pair.proton_vx_frame,
        pair.proton_vx_units,
        pair.bx_gsm,
        pair.by_gsm,
        pair.bz_gsm,
        pair.proton_speed,
        pair.proton_density,
        pair.proton_vx_gse,
        pair.position_x_gse,
        pair.position_y_gse,
        pair.position_z_gse,
        pair.position_frame,
        pair.position_units,
        pair.position_method,
        pair.position_lower_time_utc,
        pair.position_upper_time_utc,
        pair.position_interpolation_fraction,
        pair.ephemeris_record_sha256,
        pair.mag_quality_source_product,
        pair.mag_quality_value,
        pair.mag_quality_binding_status,
        pair.mag_quality_decision,
        pair.mag_quality_required_fields_status,
        pair.mag_sequence,
        pair.mag_receipt_completed_utc,
        pair.mag_record_sha256,
        pair.mag_raw_sha256,
        pair.mag_ephemeris_source_object_sha256,
        pair.wind_quality_source_product,
        pair.wind_quality_value,
        pair.wind_quality_binding_status,
        pair.wind_quality_decision,
        pair.wind_quality_required_fields_status,
        pair.wind_sequence,
        pair.wind_receipt_completed_utc,
        pair.wind_record_sha256,
        pair.wind_raw_sha256,
        pair.wind_ephemeris_source_object_sha256,
    )
end

function _operational_v22_arrival_stable_pairs(pairs)
    grouped = Dict{DateTime,Vector{Any}}()
    for pair in pairs
        push!(get!(grouped, pair.measurement_time_utc, Any[]), pair)
    end
    selected = NamedTuple[]
    for timestamp in sort!(collect(keys(grouped)))
        matches = grouped[timestamp]
        fingerprints = unique(_operational_v22_arrival_pair_fingerprint.(matches))
        length(fingerprints) == 1 || throw(ArgumentError(
            "V2.2-M2 receipt pairs contain a conflicting pre-issue revision",
        ))
        sort!(matches; by=pair -> (
            pair.issue_time_utc,
            max(pair.mag_receipt_completed_utc, pair.wind_receipt_completed_utc),
            pair.mag_sequence,
            pair.wind_sequence,
            pair.pair_sha256,
        ))
        push!(selected, last(matches))
    end
    return selected
end

function _operational_v22_arrival_floor(time::DateTime)
    minute = Dates.minute(time) < 30 ? 0 : 30
    return DateTime(Dates.year(time), Dates.month(time), Dates.day(time),
                    Dates.hour(time), minute)
end

function _operational_v22_arrival_ceil_30(time::DateTime)
    floored = _operational_v22_arrival_floor(time)
    return floored == time ? floored :
        floored + Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES)
end

function _operational_v22_arrival_state(bin::OperationalV22ArrivalBin)
    bin.proton_speed > 0.0 && bin.proton_density > 0.0 || throw(ArgumentError(
        "V2.2-M2 arrival-bin plasma values must be positive",
    ))
    return (
        bin.bx_gsm,
        bin.by_gsm,
        bin.bz_gsm,
        log(bin.proton_speed),
        log(bin.proton_density),
    )
end

function _operational_v22_arrival_bin(start::DateTime, pairs)
    values(field) = [getproperty(pair, field) for pair in pairs]
    hashes = Tuple(sort([pair.pair_sha256 for pair in pairs]))
    return OperationalV22ArrivalBin(
        start,
        start + Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES),
        median(values(:bx_gsm)),
        median(values(:by_gsm)),
        median(values(:bz_gsm)),
        median(values(:proton_speed)),
        median(values(:proton_density)),
        true,
        nothing,
        hashes,
    )
end

function _operational_v22_arrival_bins(transported)
    grouped = Dict{DateTime,Vector{Any}}()
    for pair in transported
        start = _operational_v22_arrival_floor(pair.arrival_time_utc)
        push!(get!(grouped, start, Any[]), pair)
    end
    return Tuple(
        _operational_v22_arrival_bin(start, grouped[start])
        for start in sort!(collect(keys(grouped)))
    )
end

function _operational_v22_arrival_queue_hash_payload(queue)
    return (
        queue.schema_version,
        queue.issue_time_utc,
        queue.status,
        queue.fallback_reason,
        queue.x_ref_gse_km,
        queue.v21_compatibility_distance_km,
        queue.receipt_pairs,
        queue.transported_pairs,
        queue.arrival_bins,
        queue.history_bins,
        queue.future_bins,
    )
end

"Return the composite SHA-256 identity of an arrival queue."
function operational_v22_arrival_sha256(queue::OperationalV22ArrivalQueue)
    return _operational_v22_arrival_digest(
        _operational_v22_arrival_queue_hash_payload(queue)...,
    )
end

function _operational_v22_arrival_queue(issue_time, status, reason,
                                        receipt_pairs, transported_pairs,
                                        arrival_bins, history_bins, future_bins)
    provisional = OperationalV22ArrivalQueue(
        OPERATIONAL_V22_ARRIVAL_SCHEMA_VERSION,
        issue_time,
        status,
        reason,
        OPERATIONAL_V22_ARRIVAL_X_REF_GSE_KM,
        OPERATIONAL_V22_ARRIVAL_V21_COMPATIBILITY_DISTANCE_KM,
        Tuple(receipt_pairs),
        Tuple(transported_pairs),
        Tuple(arrival_bins),
        Tuple(history_bins),
        Tuple(future_bins),
        repeat("0", 64),
    )
    return OperationalV22ArrivalQueue(
        provisional.schema_version,
        provisional.issue_time_utc,
        provisional.status,
        provisional.fallback_reason,
        provisional.x_ref_gse_km,
        provisional.v21_compatibility_distance_km,
        provisional.receipt_pairs,
        provisional.transported_pairs,
        provisional.arrival_bins,
        provisional.history_bins,
        provisional.future_bins,
        operational_v22_arrival_sha256(provisional),
    )
end

function _operational_v22_arrival_fallback(issue_time, reason, receipt_pairs;
                                           transported_pairs=(),
                                           arrival_bins=(), history_bins=(),
                                           future_bins=())
    return _operational_v22_arrival_queue(
        issue_time, :fallback, reason, receipt_pairs, transported_pairs,
        arrival_bins, history_bins, future_bins,
    )
end

function _operational_v22_arrival_transport(pairs, issue_time)
    transported = NamedTuple[]
    fallback_reason = :none
    prior_maximum = nothing
    for pair in pairs
        if !(isfinite(pair.proton_speed) && pair.proton_speed > 0.0 &&
             isfinite(pair.proton_density) && pair.proton_density > 0.0)
            fallback_reason == :none && (fallback_reason = :invalid_plasma)
            continue
        end
        if !(isfinite(pair.proton_vx_gse) && pair.proton_vx_gse < 0.0)
            fallback_reason == :none && (fallback_reason = :invalid_vx)
            continue
        end
        window_start = pair.measurement_time_utc -
            Minute(OPERATIONAL_V22_ARRIVAL_TRAILING_MINUTES)
        window = filter(candidate ->
            window_start < candidate.measurement_time_utc <=
                pair.measurement_time_utc &&
                isfinite(candidate.proton_speed) &&
                candidate.proton_speed > 0.0 &&
                isfinite(candidate.proton_density) &&
                candidate.proton_density > 0.0 &&
                isfinite(candidate.proton_vx_gse) &&
                candidate.proton_vx_gse < 0.0,
            pairs,
        )
        trailing_vx = median([
            candidate.proton_vx_gse for candidate in window
        ])
        if !(isfinite(trailing_vx) && trailing_vx < 0.0)
            fallback_reason == :none && (fallback_reason = :invalid_vx)
            continue
        end
        delay_seconds = (pair.position_x_gse -
                         OPERATIONAL_V22_ARRIVAL_X_REF_GSE_KM) / (-trailing_vx)
        compatibility_delay_seconds =
            OPERATIONAL_V22_ARRIVAL_V21_COMPATIBILITY_DISTANCE_KM / (-trailing_vx)
        minimum = 60.0 * OPERATIONAL_V22_ARRIVAL_MIN_DELAY_MINUTES
        maximum = 60.0 * OPERATIONAL_V22_ARRIVAL_MAX_DELAY_MINUTES
        if !(isfinite(delay_seconds) && minimum <= delay_seconds <= maximum)
            fallback_reason == :none &&
                (fallback_reason = :delay_out_of_bounds)
            continue
        end
        delay_milliseconds = try
            round(Int, 1000.0 * delay_seconds)
        catch error
            error isa InterruptException && rethrow()
            fallback_reason == :none &&
                (fallback_reason = :delay_out_of_bounds)
            continue
        end
        candidate = merge(pair, (
            trailing_vx_gse=trailing_vx,
            delay_seconds=delay_seconds,
            v21_compatibility_delay_seconds=compatibility_delay_seconds,
            arrival_time_utc=pair.measurement_time_utc +
                Millisecond(delay_milliseconds),
        ))
        if prior_maximum !== nothing &&
                prior_maximum - candidate.arrival_time_utc >
                    Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES)
            fallback_reason == :none &&
                (fallback_reason = :overtaking_exceeds_one_bin)
            break
        end
        push!(transported, candidate)
        prior_maximum = prior_maximum === nothing ? candidate.arrival_time_utc :
            max(prior_maximum, candidate.arrival_time_utc)
    end
    return (
        status=fallback_reason == :none ? :ready : :fallback,
        reason=fallback_reason,
        transported=transported,
    )
end

function _operational_v22_arrival_history(bins, issue_time)
    complete = filter(bin -> bin.end_utc <= issue_time, bins)
    isempty(complete) && return (
        status=:fallback, reason=:no_complete_arrival_bin, history=(),
    )
    latest = last(complete)
    issue_time - latest.end_utc <=
        Minute(OPERATIONAL_V22_ARRIVAL_MAX_FRESHNESS_MINUTES) || return (
            status=:fallback, reason=:stale_history, history=(),
        )
    by_start = Dict(bin.start_utc => bin for bin in complete)
    first_start = latest.start_utc - Minute(
        OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES *
        (OPERATIONAL_V22_ARRIVAL_HISTORY_ROWS - 1),
    )
    history = OperationalV22ArrivalBin[]
    missing = 0
    for index in 1:OPERATIONAL_V22_ARRIVAL_HISTORY_ROWS
        start = first_start + Minute(
            OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES * (index - 1),
        )
        if haskey(by_start, start)
            push!(history, by_start[start])
            continue
        end
        missing += 1
        missing <= 1 && !isempty(history) || return (
            status=:fallback, reason=:incomplete_history, history=Tuple(history),
        )
        previous = last(history)
        push!(history, OperationalV22ArrivalBin(
            start,
            start + Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES),
            previous.bx_gsm,
            previous.by_gsm,
            previous.bz_gsm,
            previous.proton_speed,
            previous.proton_density,
            false,
            previous.start_utc,
            (),
        ))
    end
    return (status=:ready, reason=:none, history=Tuple(history))
end

function _operational_v22_arrival_future(bins, issue_time)
    issue_floor = _operational_v22_arrival_floor(issue_time)
    return Tuple(filter(
        bin -> bin.start_utc >= issue_floor && bin.start_utc >= issue_time,
        bins,
    ))
end

"""
    build_operational_v22_arrival_queue(pairs, issue_time_utc::DateTime)

Build a pure receipt-causal L1 arrival queue from selector-v2 pair outputs.
Each ballistic speed is the median GSE Vx over `(s-15 min, s]`; arrival bins
are half-open UTC intervals. The Earth driver boundary is `x_ref_gse_km=0`.
The V2.1 1.5-million-km distance is retained only as a diagnostic and never
changes the ephemeris-based arrival time.
"""
function build_operational_v22_arrival_queue(pairs,
                                             issue_time_utc::DateTime)
    _operational_v22_arrival_floor(issue_time_utc) == issue_time_utc ||
        throw(ArgumentError(
            "V2.2-M2 issue time must lie on the 30-minute UTC grid",
        ))
    candidates = collect(pairs)
    eligible = NamedTuple[]
    for candidate in candidates
        candidate isa NamedTuple && hasproperty(candidate, :issue_time_utc) ||
            throw(ArgumentError(
                "V2.2-M2 receipt candidate must expose issue_time_utc",
            ))
        pair_issue = _operational_v22_arrival_parse_utc(
            candidate.issue_time_utc, "pair issue_time_utc",
        )
        pair_issue > issue_time_utc && continue
        normalized = _operational_v22_arrival_normalize_pair(candidate)
        normalized.mag_receipt_completed_utc <= issue_time_utc &&
            normalized.wind_receipt_completed_utc <= issue_time_utc ||
            throw(ArgumentError(
                "V2.2-M2 receipt cutoff disagrees with pair issue causality",
            ))
        push!(eligible, normalized)
    end
    stable = _operational_v22_arrival_stable_pairs(eligible)
    isempty(stable) && return _operational_v22_arrival_fallback(
        issue_time_utc, :no_receipt_eligible_pairs, stable,
    )
    snapshots = unique((
        pair.issue_time_utc,
        pair.issue_cutoff_relative_path,
        pair.issue_cutoff_sha256,
    ) for pair in stable)
    length(snapshots) == 1 || throw(ArgumentError(
        "V2.2-M2 receipt pairs mix distinct issue-cutoff snapshots",
    ))

    transport = _operational_v22_arrival_transport(stable, issue_time_utc)
    bins = _operational_v22_arrival_bins(transport.transported)
    history = _operational_v22_arrival_history(bins, issue_time_utc)
    future = _operational_v22_arrival_future(bins, issue_time_utc)
    if transport.status != :ready
        return _operational_v22_arrival_fallback(
            issue_time_utc, transport.reason, stable;
            transported_pairs=transport.transported,
            arrival_bins=bins,
            history_bins=history.history,
            future_bins=future,
        )
    end
    if history.status != :ready
        return _operational_v22_arrival_fallback(
            issue_time_utc, history.reason, stable;
            transported_pairs=transport.transported,
            arrival_bins=bins,
            history_bins=history.history,
            future_bins=future,
        )
    end
    return _operational_v22_arrival_queue(
        issue_time_utc, :ready, :none, stable, transport.transported,
        bins, history.history, future,
    )
end

"Return the exact 25-by-5 `(Bx, By, Bz, logV, logn)` sparse-driver seed."
function operational_v22_arrival_history(queue::OperationalV22ArrivalQueue)
    queue.status == :ready || throw(ArgumentError(
        "V2.2-M2 arrival history is unavailable: $(queue.fallback_reason)",
    ))
    length(queue.history_bins) == OPERATIONAL_V22_ARRIVAL_HISTORY_ROWS ||
        throw(ArgumentError("V2.2-M2 arrival history has the wrong length"))
    result = Matrix{Float64}(undef, OPERATIONAL_V22_ARRIVAL_HISTORY_ROWS, 5)
    for (row, bin) in pairs(queue.history_bins)
        result[row, :] .= _operational_v22_arrival_state(bin)
    end
    return result
end

"Recheck the queue checksum and all actionable structural invariants."
function verify_operational_v22_arrival_queue(queue::OperationalV22ArrivalQueue)
    queue.schema_version == OPERATIONAL_V22_ARRIVAL_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported V2.2-M2 arrival-queue schema"))
    _operational_v22_arrival_floor(queue.issue_time_utc) ==
        queue.issue_time_utc || throw(ArgumentError(
        "V2.2-M2 queue issue time is not on the 30-minute UTC grid",
    ))
    queue.x_ref_gse_km == OPERATIONAL_V22_ARRIVAL_X_REF_GSE_KM ||
        throw(ArgumentError("V2.2-M2 arrival boundary changed"))
    queue.v21_compatibility_distance_km ==
        OPERATIONAL_V22_ARRIVAL_V21_COMPATIBILITY_DISTANCE_KM ||
        throw(ArgumentError("V2.2-M2 compatibility distance changed"))
    queue.status in (:ready, :fallback) || throw(ArgumentError(
        "V2.2-M2 arrival-queue status is invalid",
    ))
    (queue.status == :ready) == (queue.fallback_reason == :none) ||
        throw(ArgumentError("V2.2-M2 arrival-queue status/reason mismatch"))
    occursin(r"^[0-9a-f]{64}$", queue.composite_sha256) &&
        operational_v22_arrival_sha256(queue) == queue.composite_sha256 ||
        throw(ArgumentError("V2.2-M2 arrival-queue checksum mismatch"))
    for pair in queue.receipt_pairs
        pair.pair_sha256 == _operational_v22_arrival_pair_sha256(pair) ||
            throw(ArgumentError("V2.2-M2 receipt-pair checksum mismatch"))
        pair.issue_time_utc <= queue.issue_time_utc &&
            pair.first_eligible_issue_time_utc <= queue.issue_time_utc &&
            pair.mag_receipt_completed_utc <= queue.issue_time_utc &&
            pair.wind_receipt_completed_utc <= queue.issue_time_utc ||
            throw(ArgumentError("V2.2-M2 queue contains post-issue provenance"))
    end
    starts = [bin.start_utc for bin in queue.arrival_bins]
    issorted(starts) && allunique(starts) || throw(ArgumentError(
        "V2.2-M2 arrival bins are not uniquely chronological",
    ))
    for bin in queue.arrival_bins
        bin.end_utc - bin.start_utc ==
            Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES) ||
            throw(ArgumentError("V2.2-M2 arrival-bin cadence changed"))
        bin.observed && bin.filled_from_start_utc === nothing ||
            throw(ArgumentError("V2.2-M2 observed arrival-bin metadata changed"))
        all(isfinite, (bin.bx_gsm, bin.by_gsm, bin.bz_gsm,
                       bin.proton_speed, bin.proton_density)) &&
            bin.proton_speed > 0.0 && bin.proton_density > 0.0 ||
            throw(ArgumentError("V2.2-M2 arrival bin is outside its domain"))
    end
    if queue.status == :ready
        length(queue.history_bins) == OPERATIONAL_V22_ARRIVAL_HISTORY_ROWS ||
            throw(ArgumentError("V2.2-M2 ready queue lacks its 25-row seed"))
        for index in 2:length(queue.history_bins)
            queue.history_bins[index].start_utc -
                queue.history_bins[index - 1].start_utc ==
                Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES) ||
                throw(ArgumentError("V2.2-M2 history is not contiguous"))
        end
        count(bin -> !bin.observed, queue.history_bins) <= 1 ||
            throw(ArgumentError("V2.2-M2 history exceeds its one-bin fill"))
        all(bin -> bin.end_utc <= queue.issue_time_utc, queue.history_bins) ||
            throw(ArgumentError("V2.2-M2 history contains an incomplete bin"))
        all(bin -> bin.start_utc >= queue.issue_time_utc,
            queue.future_bins) || throw(ArgumentError(
                "V2.2-M2 future queue contains a begun bin",
            ))
    end
    rebuilt = build_operational_v22_arrival_queue(
        Tuple(_operational_v22_arrival_pair_contract(pair)
              for pair in queue.receipt_pairs),
        queue.issue_time_utc,
    )
    rebuilt.composite_sha256 == queue.composite_sha256 || throw(ArgumentError(
        "V2.2-M2 queue derivation disagrees with its receipt-pair provenance",
    ))
    return true
end

function _operational_v22_arrival_path_step(bin::OperationalV22ArrivalBin,
                                            origin::Symbol)
    return OperationalV22ArrivalPathStep(
        bin.start_utc, bin.end_utc, bin.bx_gsm, bin.by_gsm, bin.bz_gsm,
        bin.proton_speed, bin.proton_density, origin,
        bin.contributing_pair_sha256,
    )
end

function _operational_v22_arrival_path_step(start::DateTime, state,
                                            origin::Symbol)
    all(isfinite, state) || throw(ArgumentError(
        "V2.2-M2 sparse state is non-finite",
    ))
    speed = exp(state[4])
    density = exp(state[5])
    all(isfinite, (speed, density)) && speed > 0.0 && density > 0.0 ||
        throw(ArgumentError("V2.2-M2 sparse plasma state left its domain"))
    return OperationalV22ArrivalPathStep(
        start,
        start + Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES),
        state[1], state[2], state[3], speed, density, origin, (),
    )
end

function _operational_v22_arrival_path_hash_payload(path)
    return (
        path.schema_version,
        path.gate_status,
        path.issue_time_utc,
        path.status,
        path.fallback_reason,
        path.queue_sha256,
        path.artifact_sha256,
        path.steps,
    )
end

"Return the composite SHA-256 identity of an arrival path."
function operational_v22_arrival_path_sha256(path::OperationalV22ArrivalPath)
    return _operational_v22_arrival_digest(
        _operational_v22_arrival_path_hash_payload(path)...,
    )
end

function _operational_v22_arrival_path(issue, status, reason, queue_sha,
                                       artifact_sha, steps)
    provisional = OperationalV22ArrivalPath(
        OPERATIONAL_V22_ARRIVAL_PATH_SCHEMA_VERSION,
        OPERATIONAL_V22_ARRIVAL_PATH_GATE_STATUS,
        issue,
        status,
        reason,
        queue_sha,
        artifact_sha,
        Tuple(steps),
        repeat("0", 64),
    )
    return OperationalV22ArrivalPath(
        provisional.schema_version,
        provisional.gate_status,
        provisional.issue_time_utc,
        provisional.status,
        provisional.fallback_reason,
        provisional.queue_sha256,
        provisional.artifact_sha256,
        provisional.steps,
        operational_v22_arrival_path_sha256(provisional),
    )
end

function _operational_v22_arrival_target_start(issue::DateTime)
    floored = _operational_v22_arrival_floor(issue)
    return floored == issue ? issue :
        floored + Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES)
end

function _operational_v22_arrival_latest_seed(queue)
    candidates = filter(bin -> bin.end_utc <= queue.issue_time_utc,
                        queue.arrival_bins)
    return isempty(candidates) ? nothing : last(candidates)
end

function _operational_v22_arrival_persistence_path(queue, artifact_sha, reason)
    start = _operational_v22_arrival_target_start(queue.issue_time_utc)
    seed = _operational_v22_arrival_latest_seed(queue)
    seed === nothing && return _operational_v22_arrival_path(
        queue.issue_time_utc, :fallback, reason,
        queue.composite_sha256, artifact_sha, (),
    )
    future = Dict(bin.start_utc => bin for bin in queue.future_bins)
    steps = OperationalV22ArrivalPathStep[]
    current = seed
    still_prefix = true
    for index in 0:(OPERATIONAL_V22_ARRIVAL_PATH_STEPS - 1)
        timestamp = start + Minute(
            OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES * index,
        )
        if still_prefix && haskey(future, timestamp)
            current = future[timestamp]
            push!(steps, _operational_v22_arrival_path_step(current, :queued))
        else
            still_prefix = false
            push!(steps, OperationalV22ArrivalPathStep(
                timestamp,
                timestamp + Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES),
                current.bx_gsm,
                current.by_gsm,
                current.bz_gsm,
                current.proton_speed,
                current.proton_density,
                :persistence,
                current.contributing_pair_sha256,
            ))
        end
    end
    return _operational_v22_arrival_path(
        queue.issue_time_utc, :fallback, reason, queue.composite_sha256,
        artifact_sha, steps,
    )
end

function _operational_v22_arrival_append_state(history::Matrix{Float64}, state)
    size(history) == (OPERATIONAL_V22_ARRIVAL_HISTORY_ROWS, 5) ||
        throw(DimensionMismatch("V2.2-M2 sparse seed must be 25 by 5"))
    return vcat(@view(history[2:end, :]), transpose(collect(state)))
end

function _operational_v22_arrival_one_step(artifact, history)
    return vec(operational_v22_driver_rollout(artifact, history)[1, :])
end

"""
    build_operational_v22_arrival_path(queue, artifact)

Copy the contiguous prefix of already queued future bins exactly, then apply
the frozen fourteen-step sparse driver recursion only beyond the last known
bin. Any queue, prefix, artifact, or numerical-domain failure returns the same
queued prefix followed by transported persistence and an explicit reason.
"""
function build_operational_v22_arrival_path(
        queue::OperationalV22ArrivalQueue,
        artifact::OperationalV22DriverArtifact)
    verify_operational_v22_arrival_queue(queue)
    artifact_sha = operational_v22_driver_sha256(artifact)
    queue.status == :ready || return _operational_v22_arrival_persistence_path(
        queue, artifact_sha, queue.fallback_reason,
    )

    start = _operational_v22_arrival_target_start(queue.issue_time_utc)
    future = Dict(bin.start_utc => bin for bin in queue.future_bins)
    last_known_index = 0
    seen_gap = false
    for index in 1:OPERATIONAL_V22_ARRIVAL_PATH_STEPS
        timestamp = start + Minute(
            OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES * (index - 1),
        )
        if haskey(future, timestamp)
            seen_gap && return _operational_v22_arrival_persistence_path(
                queue, artifact_sha, :future_queue_not_contiguous,
            )
            last_known_index = index
        else
            seen_gap = true
        end
    end

    history = try
        operational_v22_arrival_history(queue)
    catch error
        error isa InterruptException && rethrow()
        return _operational_v22_arrival_persistence_path(
            queue, artifact_sha, :invalid_sparse_seed,
        )
    end
    history_end = last(queue.history_bins).end_utc
    history_end <= start || return _operational_v22_arrival_persistence_path(
        queue, artifact_sha, :history_after_path_start,
    )

    # Advance any stale-but-admissible complete seed to the first product bin.
    cursor = history_end
    while cursor < start
        state = try
            _operational_v22_arrival_one_step(artifact, history)
        catch error
            error isa InterruptException && rethrow()
            return _operational_v22_arrival_persistence_path(
                queue, artifact_sha, :artifact_rollout_failed,
            )
        end
        all(isfinite, state) || return _operational_v22_arrival_persistence_path(
            queue, artifact_sha, :sparse_tail_out_of_domain,
        )
        history = _operational_v22_arrival_append_state(history, state)
        cursor += Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES)
    end

    steps = OperationalV22ArrivalPathStep[]
    for index in 1:OPERATIONAL_V22_ARRIVAL_PATH_STEPS
        timestamp = start + Minute(
            OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES * (index - 1),
        )
        if index <= last_known_index
            bin = future[timestamp]
            push!(steps, _operational_v22_arrival_path_step(bin, :queued))
            history = _operational_v22_arrival_append_state(
                history, _operational_v22_arrival_state(bin),
            )
            continue
        end
        state = try
            _operational_v22_arrival_one_step(artifact, history)
        catch error
            error isa InterruptException && rethrow()
            return _operational_v22_arrival_persistence_path(
                queue, artifact_sha, :artifact_rollout_failed,
            )
        end
        step = try
            _operational_v22_arrival_path_step(timestamp, state, :sparse)
        catch error
            error isa InterruptException && rethrow()
            return _operational_v22_arrival_persistence_path(
                queue, artifact_sha, :sparse_tail_out_of_domain,
            )
        end
        push!(steps, step)
        history = _operational_v22_arrival_append_state(history, state)
    end
    return _operational_v22_arrival_path(
        queue.issue_time_utc, :ready, :none, queue.composite_sha256,
        artifact_sha, steps,
    )
end

"Recheck candidate-path identity, time grid, origin contract, and domain."
function verify_operational_v22_arrival_path(path::OperationalV22ArrivalPath)
    path.schema_version == OPERATIONAL_V22_ARRIVAL_PATH_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported V2.2-M2 arrival-path schema"))
    path.gate_status == OPERATIONAL_V22_ARRIVAL_PATH_GATE_STATUS ||
        throw(ArgumentError(
            "V2.2-M2 arrival path lacks the frozen support/gate artifact",
        ))
    _operational_v22_arrival_floor(path.issue_time_utc) ==
        path.issue_time_utc || throw(ArgumentError(
        "V2.2-M2 arrival-path issue is not on the 30-minute UTC grid",
    ))
    path.status in (:ready, :fallback) || throw(ArgumentError(
        "V2.2-M2 arrival-path status is invalid",
    ))
    (path.status == :ready) == (path.fallback_reason == :none) ||
        throw(ArgumentError("V2.2-M2 arrival-path status/reason mismatch"))
    occursin(r"^[0-9a-f]{64}$", path.queue_sha256) &&
        occursin(r"^[0-9a-f]{64}$", path.artifact_sha256) ||
        throw(ArgumentError("V2.2-M2 arrival-path dependency hash is malformed"))
    operational_v22_arrival_path_sha256(path) == path.composite_sha256 ||
        throw(ArgumentError("V2.2-M2 arrival-path checksum mismatch"))
    length(path.steps) in (0, OPERATIONAL_V22_ARRIVAL_PATH_STEPS) ||
        throw(ArgumentError("V2.2-M2 arrival path has the wrong length"))
    path.status == :ready &&
        length(path.steps) != OPERATIONAL_V22_ARRIVAL_PATH_STEPS &&
        throw(ArgumentError("V2.2-M2 ready path must contain fourteen states"))
    seen_tail = false
    for (index, step) in pairs(path.steps)
        step.start_utc == _operational_v22_arrival_target_start(
            path.issue_time_utc,
        ) + Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES * (index - 1)) ||
            throw(ArgumentError("V2.2-M2 arrival-path grid changed"))
        step.end_utc - step.start_utc ==
            Minute(OPERATIONAL_V22_ARRIVAL_CADENCE_MINUTES) ||
            throw(ArgumentError("V2.2-M2 arrival-path cadence changed"))
        step.origin in (:queued, :sparse, :persistence) || throw(ArgumentError(
            "V2.2-M2 arrival-path origin is invalid",
        ))
        if path.status == :ready
            step.origin == :persistence && throw(ArgumentError(
                "V2.2-M2 ready path contains a persistence fallback",
            ))
            step.origin == :sparse && (seen_tail = true)
            seen_tail && step.origin == :queued && throw(ArgumentError(
                "V2.2-M2 ready path returns to its queue after recursion",
            ))
        else
            step.origin == :sparse && throw(ArgumentError(
                "V2.2-M2 fallback path contains a sparse state",
            ))
            step.origin == :persistence && (seen_tail = true)
            seen_tail && step.origin == :queued && throw(ArgumentError(
                "V2.2-M2 fallback path returns to its queue after persistence",
            ))
        end
        all(isfinite, (step.bx_gsm, step.by_gsm, step.bz_gsm,
                       step.proton_speed, step.proton_density)) &&
            step.proton_speed > 0.0 && step.proton_density > 0.0 ||
            throw(ArgumentError("V2.2-M2 arrival-path state left its domain"))
    end
    return true
end

"Rebuild a path from its queue and artifact to verify dependency binding."
function verify_operational_v22_arrival_path(
        path::OperationalV22ArrivalPath,
        queue::OperationalV22ArrivalQueue,
        artifact::OperationalV22DriverArtifact)
    verify_operational_v22_arrival_path(path)
    verify_operational_v22_arrival_queue(queue)
    path.queue_sha256 == queue.composite_sha256 || throw(ArgumentError(
        "V2.2-M2 arrival path is bound to a different queue",
    ))
    path.artifact_sha256 == operational_v22_driver_sha256(artifact) ||
        throw(ArgumentError(
            "V2.2-M2 arrival path is bound to a different driver artifact",
        ))
    rebuilt = build_operational_v22_arrival_path(queue, artifact)
    rebuilt.composite_sha256 == path.composite_sha256 || throw(ArgumentError(
        "V2.2-M2 arrival path disagrees with its bound queue and artifact",
    ))
    return true
end

"Return the low-level research candidate as `(Bx, By, Bz, logV, logn)`."
function operational_v22_arrival_path_matrix(path::OperationalV22ArrivalPath)
    verify_operational_v22_arrival_path(path)
    length(path.steps) == OPERATIONAL_V22_ARRIVAL_PATH_STEPS ||
        throw(ArgumentError("V2.2-M2 arrival path has no usable fallback state"))
    result = Matrix{Float64}(undef, OPERATIONAL_V22_ARRIVAL_PATH_STEPS, 5)
    for (row, step) in pairs(path.steps)
        result[row, :] .= (
            step.bx_gsm,
            step.by_gsm,
            step.bz_gsm,
            log(step.proton_speed),
            log(step.proton_density),
        )
    end
    all(isfinite, result) || throw(ArgumentError(
        "V2.2-M2 arrival path cannot be represented in driver-state space",
    ))
    return result
end
