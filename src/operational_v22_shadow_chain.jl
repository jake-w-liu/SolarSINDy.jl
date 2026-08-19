# Checksum-bound, offline-only contract for the Operational V2.2 shadow chain.

import SHA

const OPERATIONAL_V22_SHADOW_SCHEMA_VERSION = "operational_v2_2_shadow_chain_v1"
const OPERATIONAL_V22_SHADOW_PRODUCT_VERSION = "v2.2-shadow"
const OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H = OPERATIONAL_V22_MODEL_STEPS
const OPERATIONAL_V22_SHADOW_DEFAULT_FEATURE_SCHEMA = ntuple(
    index -> "matured_h1_innovation_lag_" *
             string(OPERATIONAL_V22_ERROR_LAGS_H[index]) * "h",
    length(OPERATIONAL_V22_ERROR_LAGS_H),
)

function _operational_v22_shadow_sha(value, field::AbstractString)
    text = String(value)
    occursin(r"^[0-9a-f]{64}$", text) || throw(ArgumentError(
        "V2.2 shadow-chain $field must be lowercase SHA-256",
    ))
    return text
end

function _operational_v22_shadow_float(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "V2.2 shadow-chain $field must be a real number",
    ))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError(
        "V2.2 shadow-chain $field must be finite",
    ))
    return converted
end

function _operational_v22_shadow_int(value, field::AbstractString)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "V2.2 shadow-chain $field must be an integer",
    ))
    typemin(Int) <= value <= typemax(Int) || throw(ArgumentError(
        "V2.2 shadow-chain $field exceeds the supported integer range",
    ))
    return Int(value)
end

function _operational_v22_shadow_feature_schema(values)
    schema = Tuple(String(value) for value in values)
    isempty(schema) && throw(ArgumentError(
        "V2.2 shadow-chain feature schema must not be empty",
    ))
    all(value -> !isempty(strip(value)) &&
                 !occursin(';', value) &&
                 !any(iscntrl, value), schema) || throw(ArgumentError(
        "V2.2 shadow-chain feature names must be nonempty and delimiter-safe",
    ))
    length(unique(schema)) == length(schema) || throw(ArgumentError(
        "V2.2 shadow-chain feature names must be unique",
    ))
    return schema
end

function _operational_v22_shadow_hash_token(io::IO, value)
    text = value isa Float64 ? bitstring(value) : string(value)
    kind = string(typeof(value))
    print(io, ncodeunits(kind), ':', kind, ':', ncodeunits(text), ':', text, '|')
    return nothing
end

"Hash a regular non-symlink file for use as an external provenance token."
function operational_v22_regular_file_sha256(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "V2.2 shadow-chain provenance source must be a regular non-symlink file: $source",
    ))
    return open(source, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function _operational_v22_validate_core(core::OperationalCore)
    artifacts = core.artifacts
    artifacts.version == OPERATIONAL_V2_1_MODEL_VERSION || throw(ArgumentError(
        "V2.2 shadow chain requires the frozen V2.1 point core",
    ))
    artifacts.candidate_count == 20 && artifacts.active_count == 11 ||
        throw(ArgumentError(
            "V2.2 shadow-chain core must retain the 20-candidate/11-active contract",
        ))
    terms = get_term_names(core.library)
    canonical_library = _operational_library(OPERATIONAL_V2_1_MODEL_VERSION)
    expected_terms = get_term_names(canonical_library)
    terms == expected_terms || throw(ArgumentError(
        "V2.2 shadow-chain core library is not the frozen V2.1 library",
    ))
    canonical_codes = canonical_library._contract_term_codes
    all(!iszero, canonical_codes) || throw(ErrorException(
        "frozen V2.1 core contains an untrusted fallback term code",
    ))
    core.library._contract_term_codes == canonical_codes ||
        throw(ArgumentError(
            "V2.2 shadow-chain core executable term semantics do not match V2.1",
        ))
    length(core.coefficients) == length(terms) || throw(DimensionMismatch(
        "V2.2 shadow-chain core coefficients do not match its library",
    ))
    all(isfinite, core.coefficients) || throw(ArgumentError(
        "V2.2 shadow-chain core coefficients must be finite",
    ))
    count(!iszero, core.coefficients) == artifacts.active_count ||
        throw(ArgumentError(
            "V2.2 shadow-chain core does not have exactly 11 active terms",
        ))
    "n*V^2" in terms && throw(ArgumentError(
        "V2.2 shadow-chain core contains the retired redundant pressure term",
    ))
    for required in ("Pdyn", "Pdyn*Bs")
        index = findfirst(==(required), terms)
        index === nothing && throw(ArgumentError(
            "V2.2 shadow-chain core omits required term $required",
        ))
        core.coefficients[index] != 0.0 || throw(ArgumentError(
            "V2.2 shadow-chain core has inactive required term $required",
        ))
    end
    return terms, canonical_codes
end

"Return the deterministic semantic identity of the frozen V2.1 point core."
function operational_v22_core_sha256(core::OperationalCore)
    terms, term_codes = _operational_v22_validate_core(core)
    io = IOBuffer()
    for value in (
            "operational_v2_2_frozen_core_identity_v1",
            core.artifacts.version,
            core.artifacts.candidate_count,
            core.artifacts.active_count,
        )
        _operational_v22_shadow_hash_token(io, value)
    end
    _operational_v22_shadow_hash_token(io, length(terms))
    for index in eachindex(terms)
        _operational_v22_shadow_hash_token(io, terms[index])
        _operational_v22_shadow_hash_token(io, term_codes[index])
        _operational_v22_shadow_hash_token(io, core.coefficients[index])
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function _operational_v22_validate_stratum(
        stratum::ConformalStratum,
        expected_key::Symbol,
        coverage::Float64)
    stratum.key == expected_key || throw(ArgumentError(
        "V2.2 shadow-chain conformal stratum key is inconsistent",
    ))
    stratum.n >= 1 || throw(ArgumentError(
        "V2.2 shadow-chain conformal strata require at least one row",
    ))
    isfinite(stratum.half_width) && stratum.half_width >= 0.0 ||
        throw(ArgumentError(
            "V2.2 shadow-chain conformal half-widths must be finite and nonnegative",
        ))
    expected_rank = clamp(ceil(Int, (stratum.n + 1) * coverage), 1, stratum.n)
    expected_floor = expected_rank / (stratum.n + 1)
    stratum.coverage_floor == expected_floor || throw(ArgumentError(
        "V2.2 shadow-chain conformal coverage floor is inconsistent with n and coverage",
    ))
    return nothing
end

function _operational_v22_validate_conformal(calibration::ConformalCalibration)
    isfinite(calibration.coverage) && 0.0 < calibration.coverage < 1.0 ||
        throw(ArgumentError(
            "V2.2 shadow-chain conformal coverage must lie in (0, 1)",
        ))
    _validate_conformal_edges(calibration.horizon_edges)
    isfinite(calibration.activity_threshold_nt) || throw(ArgumentError(
        "V2.2 shadow-chain conformal activity threshold must be finite",
    ))
    calibration.min_stratum_n >= 1 || throw(ArgumentError(
        "V2.2 shadow-chain conformal minimum stratum size must be positive",
    ))
    isfinite(calibration.max_horizon) && calibration.max_horizon >= 0.0 ||
        throw(ArgumentError(
            "V2.2 shadow-chain conformal maximum horizon must be finite and nonnegative",
        ))
    _operational_v22_validate_stratum(
        calibration.global_stratum, :global, calibration.coverage,
    )
    haskey(calibration.strata, :global) && throw(ArgumentError(
        "V2.2 shadow-chain conformal global stratum must not be duplicated",
    ))
    valid_keys = Set(
        _stratum_key(bin, regime)
        for bin in 1:(length(calibration.horizon_edges) - 1)
        for regime in (:quiet, :disturbed)
    )
    for (key, stratum) in calibration.strata
        key in valid_keys || throw(ArgumentError(
            "V2.2 shadow-chain conformal stratum key is outside the schema: $key",
        ))
        _operational_v22_validate_stratum(stratum, key, calibration.coverage)
    end
    sum(stratum.n for stratum in values(calibration.strata)) ==
        calibration.global_stratum.n || throw(ArgumentError(
            "V2.2 shadow-chain conformal stratum counts do not match the global count",
        ))
    return nothing
end

"Return a deterministic semantic hash of every interval-significant field."
function operational_v22_conformal_sha256(calibration::ConformalCalibration)
    _operational_v22_validate_conformal(calibration)
    io = IOBuffer()
    for value in (
            "operational_v2_2_conformal_identity_v1",
            calibration.coverage,
            calibration.activity_threshold_nt,
            calibration.min_stratum_n,
            calibration.max_horizon,
        )
        _operational_v22_shadow_hash_token(io, value)
    end
    _operational_v22_shadow_hash_token(io, length(calibration.horizon_edges))
    for edge in calibration.horizon_edges
        _operational_v22_shadow_hash_token(io, Float64(edge))
    end
    keys_in_order = sort!(collect(keys(calibration.strata)); by=String)
    _operational_v22_shadow_hash_token(io, length(keys_in_order) + 1)
    for stratum in (calibration.global_stratum,
                    (calibration.strata[key] for key in keys_in_order)...)
        for value in (
                stratum.key,
                stratum.n,
                stratum.half_width,
                stratum.coverage_floor,
            )
            _operational_v22_shadow_hash_token(io, value)
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

"External immutable identities that the numerical component objects cannot prove."
struct OperationalV22ShadowBindings
    product_version::String
    receipt_pair_contract_sha256::String
    transport_support_sha256::String
    anchor_pressure_contract_sha256::String
    conformal_sidecar_sha256::String
    point_calibration_sha256::String
    feature_schema::Tuple{Vararg{String}}

    function OperationalV22ShadowBindings(
            product_version::String,
            receipt_pair_contract_sha256::String,
            transport_support_sha256::String,
            anchor_pressure_contract_sha256::String,
            conformal_sidecar_sha256::String,
            point_calibration_sha256::String,
            feature_schema::Tuple{Vararg{String}},
            ::Val{:validated})
        product_version == OPERATIONAL_V22_SHADOW_PRODUCT_VERSION ||
            throw(ArgumentError(
                "V2.2 shadow-chain product version must be " *
                OPERATIONAL_V22_SHADOW_PRODUCT_VERSION,
            ))
        return new(
            product_version,
            _operational_v22_shadow_sha(
                receipt_pair_contract_sha256, "receipt-pair contract checksum",
            ),
            _operational_v22_shadow_sha(
                transport_support_sha256, "transport/support checksum",
            ),
            _operational_v22_shadow_sha(
                anchor_pressure_contract_sha256,
                "anchor-pressure contract checksum",
            ),
            _operational_v22_shadow_sha(
                conformal_sidecar_sha256, "conformal sidecar checksum",
            ),
            _operational_v22_shadow_sha(
                point_calibration_sha256, "point-calibration checksum",
            ),
            begin
                normalized = _operational_v22_shadow_feature_schema(feature_schema)
                # `operational_v22_error_exogenous.jl` is included before this file, so the
                # exogenous schema is always defined here; the guarded lookup it replaced could
                # silently accept only the default schema if that ever stopped being true.
                exogenous_schema = Tuple(String.(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES))
                normalized in (
                    OPERATIONAL_V22_SHADOW_DEFAULT_FEATURE_SCHEMA,
                    exogenous_schema,
                ) ||
                    throw(ArgumentError(
                        "V2.2 shadow-chain feature schema is not a frozen M3 schema",
                    ))
                normalized
            end,
        )
    end
end

function OperationalV22ShadowBindings(;
        receipt_pair_contract_sha256::AbstractString,
        transport_support_sha256::AbstractString,
        anchor_pressure_contract_sha256::AbstractString,
        conformal_sidecar_sha256::AbstractString,
        point_calibration_sha256::AbstractString,
        feature_schema=OPERATIONAL_V22_SHADOW_DEFAULT_FEATURE_SCHEMA,
        product_version::AbstractString=OPERATIONAL_V22_SHADOW_PRODUCT_VERSION)
    return OperationalV22ShadowBindings(
        String(product_version),
        String(receipt_pair_contract_sha256),
        String(transport_support_sha256),
        String(anchor_pressure_contract_sha256),
        String(conformal_sidecar_sha256),
        String(point_calibration_sha256),
        _operational_v22_shadow_feature_schema(feature_schema),
        Val(:validated),
    )
end

function _operational_v22_shadow_bindings_match(
        left::OperationalV22ShadowBindings,
        right::OperationalV22ShadowBindings)
    return left.product_version == right.product_version &&
           left.receipt_pair_contract_sha256 ==
               right.receipt_pair_contract_sha256 &&
           left.transport_support_sha256 == right.transport_support_sha256 &&
           left.anchor_pressure_contract_sha256 ==
               right.anchor_pressure_contract_sha256 &&
           left.conformal_sidecar_sha256 == right.conformal_sidecar_sha256 &&
           left.point_calibration_sha256 == right.point_calibration_sha256 &&
           left.feature_schema == right.feature_schema
end

"Identity of the complete M2-driven frozen-core center contract."
function operational_v22_base_center_sha256(
        bindings::OperationalV22ShadowBindings,
        driver::OperationalV22DriverArtifact,
        core::OperationalCore;
        anchor_lag_hours::Integer=0)
    lag = _operational_v22_shadow_int(anchor_lag_hours, "anchor lag")
    lag == 0 || throw(ArgumentError(
        "V2.2 shadow-chain center currently requires a same-hour anchor",
    ))
    io = IOBuffer()
    for value in (
            "operational_v2_2_m2_plus_frozen_core_v1",
            bindings.receipt_pair_contract_sha256,
            bindings.transport_support_sha256,
            bindings.anchor_pressure_contract_sha256,
            operational_v22_driver_sha256(driver),
            OPERATIONAL_V22_CORE_PATH_SCHEMA_VERSION,
            OPERATIONAL_V22_CORE_PATH_HOURS,
            OPERATIONAL_V22_CORE_PATH_SUBSTEPS_PER_HOUR,
            "physical_v_n_pairwise_hourly_mean",
            "proton_pdyn_recomputed_after_hourly_aggregation",
            "sequential_one_hour_frozen_v2_1_core_steps",
            "target_hour_pdyn_dst_star_inversion",
            operational_v22_core_sha256(core),
            lag,
        )
        _operational_v22_shadow_hash_token(io, value)
    end
    for horizon in OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H
        _operational_v22_shadow_hash_token(io, horizon)
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

"Research-only point center tied to its issue, anchor, target, horizon, and model."
struct OperationalV22BaseCenterForecast
    issue_time::DateTime
    anchor_time::DateTime
    target_time::DateTime
    horizon_hours::Int
    base_center_sha256::String
    prediction_dst_nt::Float64

    function OperationalV22BaseCenterForecast(
            issue_time::DateTime,
            anchor_time::DateTime,
            horizon_hours::Int,
            base_center_sha256::String,
            prediction_dst_nt::Float64,
            ::Val{:research_only})
        horizon_hours >= 1 || throw(ArgumentError(
            "V2.2 base-center horizon must be positive",
        ))
        anchor_time <= issue_time || throw(ArgumentError(
            "V2.2 base-center anchor cannot follow its issue time",
        ))
        return new(
            issue_time,
            anchor_time,
            issue_time + Hour(horizon_hours),
            horizon_hours,
            _operational_v22_shadow_sha(
                base_center_sha256, "base-center checksum",
            ),
            _operational_v22_shadow_float(
                prediction_dst_nt, "base-center prediction",
            ),
        )
    end
end

function _operational_v22_research_base_center(
        issue_time::DateTime,
        anchor_time::DateTime,
        horizon_hours::Integer,
        base_center_sha256::AbstractString,
        prediction_dst_nt::Real)
    return OperationalV22BaseCenterForecast(
        issue_time,
        anchor_time,
        _operational_v22_shadow_int(horizon_hours, "base-center horizon"),
        String(base_center_sha256),
        _operational_v22_shadow_float(
            prediction_dst_nt, "base-center prediction",
        ),
        Val(:research_only),
    )
end

"Construct an explicitly synthetic numeric center for offline research only."
function OperationalV22BaseCenterForecast(
        issue_time::DateTime,
        anchor_time::DateTime,
        horizon_hours::Integer,
        base_center_sha256::AbstractString,
        prediction_dst_nt::Real;
        execution_scope)
    execution_scope === :synthetic_research_only || throw(ArgumentError(
        "numeric V2.2 base centers require execution_scope=:synthetic_research_only",
    ))
    return _operational_v22_research_base_center(
        issue_time,
        anchor_time,
        horizon_hours,
        base_center_sha256,
        prediction_dst_nt,
    )
end

"Construct a research-only center from an explicitly low-level core-path output."
function OperationalV22BaseCenterForecast(
        issue_time::DateTime,
        anchor_time::DateTime,
        horizon_hours::Integer,
        base_center_sha256::AbstractString,
        core_path_result::NamedTuple)
    required = (
        :schema_version,
        :internal_step_hours,
        :supported_model_steps,
        :pred_dst_nt,
        :execution_scope,
    )
    all(name -> hasproperty(core_path_result, name), required) ||
        throw(ArgumentError(
            "V2.2 base-center core-path result is missing required fields",
        ))
    core_path_result.execution_scope === :low_level_research_only ||
        throw(ArgumentError(
            "V2.2 core-path centers require " *
            "execution_scope=:low_level_research_only",
        ))
    core_path_result.schema_version == OPERATIONAL_V22_CORE_PATH_SCHEMA_VERSION ||
        throw(ArgumentError("V2.2 base-center core-path schema mismatch"))
    Tuple(core_path_result.internal_step_hours) ==
        ntuple(identity, OPERATIONAL_V22_CORE_PATH_HOURS) ||
        throw(ArgumentError("V2.2 base-center internal-step schema mismatch"))
    Tuple(core_path_result.supported_model_steps) ==
        OPERATIONAL_V22_CORE_PATH_SUPPORTED_MODEL_STEPS ||
        throw(ArgumentError("V2.2 base-center supported-step schema mismatch"))
    length(core_path_result.pred_dst_nt) == OPERATIONAL_V22_CORE_PATH_HOURS ||
        throw(DimensionMismatch(
            "V2.2 base-center core path must contain seven Dst predictions",
        ))
    horizon = _operational_v22_shadow_int(
        horizon_hours, "base-center horizon",
    )
    anchor_time == issue_time || throw(ArgumentError(
        "V2.2 base-center core path currently requires a same-hour anchor",
    ))
    horizon in OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H ||
        throw(ArgumentError("unsupported V2.2 issue-relative horizon=$horizon"))
    index = findfirst(==(horizon), core_path_result.internal_step_hours)
    index === nothing && throw(ArgumentError(
        "V2.2 base-center horizon is absent from the core path",
    ))
    return _operational_v22_research_base_center(
        issue_time,
        anchor_time,
        horizon,
        base_center_sha256,
        core_path_result.pred_dst_nt[index],
    )
end

"Immutable identity manifest for one complete V2.2 shadow chain."
struct OperationalV22ShadowChainArtifact
    label::String
    bindings::OperationalV22ShadowBindings
    driver_sha256::String
    core_sha256::String
    base_center_sha256::String
    m3_kind::Symbol
    m3_sha256_by_horizon::NTuple{6,String}
    conformal_sha256::String
    supported_horizons_hours::NTuple{6,Int}
    anchor_lag_hours::Int

    function OperationalV22ShadowChainArtifact(
            label::String,
            bindings::OperationalV22ShadowBindings,
            driver_sha256::String,
            core_sha256::String,
            base_center_sha256::String,
            m3_kind::Symbol,
            m3_sha256_by_horizon::NTuple{6,String},
            conformal_sha256::String,
            supported_horizons_hours::NTuple{6,Int},
            anchor_lag_hours::Int,
            ::Val{:validated})
        isempty(strip(label)) && throw(ArgumentError(
            "V2.2 shadow-chain artifact label must not be empty",
        ))
        supported_horizons_hours == OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H ||
            throw(ArgumentError(
                "V2.2 shadow-chain issue-relative horizon schema is not frozen",
            ))
        anchor_lag_hours == 0 || throw(ArgumentError(
            "V2.2 shadow-chain currently requires a same-hour anchor",
        ))
        m3_kind in (:ar_control, :exogenous) || throw(ArgumentError(
            "V2.2 shadow-chain M3 kind is unsupported: $m3_kind",
        ))
        return new(
            label,
            bindings,
            _operational_v22_shadow_sha(driver_sha256, "M2 driver checksum"),
            _operational_v22_shadow_sha(core_sha256, "V2.1 core checksum"),
            _operational_v22_shadow_sha(
                base_center_sha256, "base-center checksum",
            ),
            m3_kind,
            ntuple(
                index -> _operational_v22_shadow_sha(
                    m3_sha256_by_horizon[index], "M3 checksum",
                ),
                length(m3_sha256_by_horizon),
            ),
            _operational_v22_shadow_sha(
                conformal_sha256, "conformal semantic checksum",
            ),
            supported_horizons_hours,
            anchor_lag_hours,
        )
    end
end

function OperationalV22ShadowChainArtifact(
        bindings::OperationalV22ShadowBindings,
        driver::OperationalV22DriverArtifact,
        core::OperationalCore,
        error_state::OperationalV22ErrorStateArtifact,
        conformal::ConformalCalibration;
        anchor_lag_hours::Integer=0,
        label::AbstractString="operational-v2.2-shadow-chain")
    lag = _operational_v22_shadow_int(anchor_lag_hours, "anchor lag")
    driver_hash = operational_v22_driver_sha256(driver)
    core_hash = operational_v22_core_sha256(core)
    base_hash = operational_v22_base_center_sha256(
        bindings, driver, core; anchor_lag_hours=lag,
    )
    error_state.base_center_sha256 == base_hash || throw(ArgumentError(
        "V2.2-M3 artifact is not bound to this M2-plus-core center",
    ))
    all(horizon -> horizon in OPERATIONAL_V22_ERROR_SUPPORTED_MODEL_STEPS,
        OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H) || throw(ErrorException(
        "V2.2 shadow-chain and M3 horizon schemas disagree",
    ))
    conformal.max_horizon >= maximum(OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H) ||
        throw(ArgumentError(
            "V2.2 conformal calibration does not cover every supported horizon",
        ))
    error_hash = operational_v22_error_state_sha256(error_state)
    bindings.feature_schema == OPERATIONAL_V22_SHADOW_DEFAULT_FEATURE_SCHEMA ||
        throw(ArgumentError(
            "V2.2 AR-control chain requires the frozen nine-lag feature schema",
        ))
    return OperationalV22ShadowChainArtifact(
        String(label),
        bindings,
        driver_hash,
        core_hash,
        base_hash,
        :ar_control,
        ntuple(_ -> error_hash,
               length(OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H)),
        operational_v22_conformal_sha256(conformal),
        OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H,
        lag,
        Val(:validated),
    )
end

const _OperationalV22ExogenousArtifactCollection = Union{
    AbstractVector{<:OperationalV22ErrorExogenousArtifact},
    NTuple{6,OperationalV22ErrorExogenousArtifact},
}

function _operational_v22_shadow_ordered_exogenous(
        artifacts::_OperationalV22ExogenousArtifactCollection)
    length(artifacts) == length(OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H) ||
        throw(DimensionMismatch(
            "V2.2 exogenous shadow chain requires six lead artifacts",
        ))
    by_lead = Dict{Int,OperationalV22ErrorExogenousArtifact}()
    for artifact in artifacts
        haskey(by_lead, artifact.model_step_hours) && throw(ArgumentError(
            "V2.2 exogenous shadow chain has a duplicate lead artifact",
        ))
        by_lead[artifact.model_step_hours] = artifact
    end
    all(haskey(by_lead, horizon)
        for horizon in OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H) ||
        throw(ArgumentError(
            "V2.2 exogenous shadow chain does not cover the frozen horizons",
        ))
    return ntuple(
        index -> by_lead[OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H[index]],
        length(OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H),
    )
end

"Construct a full-M3 chain from one exact exogenous artifact per issued lead."
function OperationalV22ShadowChainArtifact(
        bindings::OperationalV22ShadowBindings,
        driver::OperationalV22DriverArtifact,
        core::OperationalCore,
        error_states::_OperationalV22ExogenousArtifactCollection,
        conformal::ConformalCalibration;
        anchor_lag_hours::Integer=0,
        label::AbstractString="operational-v2.2-shadow-chain")
    lag = _operational_v22_shadow_int(anchor_lag_hours, "anchor lag")
    ordered = _operational_v22_shadow_ordered_exogenous(error_states)
    expected_features = Tuple(String.(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES))
    bindings.feature_schema == expected_features || throw(ArgumentError(
        "V2.2 exogenous chain requires the frozen 73-feature schema",
    ))
    driver_hash = operational_v22_driver_sha256(driver)
    core_hash = operational_v22_core_sha256(core)
    base_hash = operational_v22_base_center_sha256(
        bindings, driver, core; anchor_lag_hours=lag,
    )
    all(error_state -> error_state.base_center_sha256 == base_hash, ordered) ||
        throw(ArgumentError(
            "V2.2 exogenous M3 artifacts are not bound to this center",
        ))
    conformal.max_horizon >= maximum(OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H) ||
        throw(ArgumentError(
            "V2.2 conformal calibration does not cover every supported horizon",
        ))
    return OperationalV22ShadowChainArtifact(
        String(label),
        bindings,
        driver_hash,
        core_hash,
        base_hash,
        :exogenous,
        ntuple(
            index -> operational_v22_error_exogenous_sha256(ordered[index]),
            length(ordered),
        ),
        operational_v22_conformal_sha256(conformal),
        OPERATIONAL_V22_SHADOW_SUPPORTED_HORIZONS_H,
        lag,
        Val(:validated),
    )
end

function _operational_v22_validate_shadow_base(
        artifact::OperationalV22ShadowChainArtifact,
        bindings::OperationalV22ShadowBindings,
        driver::OperationalV22DriverArtifact,
        core::OperationalCore,
        conformal::ConformalCalibration)
    _operational_v22_shadow_bindings_match(bindings, artifact.bindings) ||
        throw(ArgumentError(
        "V2.2 shadow-chain external identity bindings do not match",
        ))
    operational_v22_driver_sha256(driver) == artifact.driver_sha256 ||
        throw(ArgumentError("V2.2 shadow-chain M2 driver identity mismatch"))
    operational_v22_core_sha256(core) == artifact.core_sha256 ||
        throw(ArgumentError("V2.2 shadow-chain V2.1 core identity mismatch"))
    operational_v22_base_center_sha256(
        bindings,
        driver,
        core;
        anchor_lag_hours=artifact.anchor_lag_hours,
    ) ==
        artifact.base_center_sha256 || throw(ArgumentError(
            "V2.2 shadow-chain base-center identity mismatch",
        ))
    operational_v22_conformal_sha256(conformal) == artifact.conformal_sha256 ||
        throw(ArgumentError(
            "V2.2 shadow-chain conformal semantic identity mismatch",
        ))
    conformal.max_horizon >= maximum(artifact.supported_horizons_hours) ||
        throw(ArgumentError(
            "V2.2 shadow-chain conformal calibration is out of horizon support",
        ))
    return artifact
end

function _operational_v22_validate_shadow_components(
        artifact::OperationalV22ShadowChainArtifact,
        bindings::OperationalV22ShadowBindings,
        driver::OperationalV22DriverArtifact,
        core::OperationalCore,
        error_state::OperationalV22ErrorStateArtifact,
        conformal::ConformalCalibration)
    _operational_v22_validate_shadow_base(
        artifact, bindings, driver, core, conformal,
    )
    error_state.base_center_sha256 == artifact.base_center_sha256 ||
        throw(ArgumentError(
            "V2.2 shadow-chain M3 base-center identity mismatch",
        ))
    artifact.m3_kind == :ar_control || throw(ArgumentError(
        "V2.2 shadow-chain manifest does not select the AR-control M3",
    ))
    error_hash = operational_v22_error_state_sha256(error_state)
    all(==(error_hash), artifact.m3_sha256_by_horizon) || throw(ArgumentError(
        "V2.2 shadow-chain M3 artifact identity mismatch",
    ))
    return artifact
end

"Validate all mutable and external identities without producing a forecast."
function validate_operational_v22_shadow_chain(
        artifact::OperationalV22ShadowChainArtifact,
        bindings::OperationalV22ShadowBindings,
        driver::OperationalV22DriverArtifact,
        core::OperationalCore,
        error_state::OperationalV22ErrorStateArtifact,
        conformal::ConformalCalibration)
    return _operational_v22_validate_shadow_components(
        artifact, bindings, driver, core, error_state, conformal,
    )
end

function _operational_v22_validate_shadow_components(
        artifact::OperationalV22ShadowChainArtifact,
        bindings::OperationalV22ShadowBindings,
        driver::OperationalV22DriverArtifact,
        core::OperationalCore,
        error_states::_OperationalV22ExogenousArtifactCollection,
        conformal::ConformalCalibration)
    _operational_v22_validate_shadow_base(
        artifact, bindings, driver, core, conformal,
    )
    artifact.m3_kind == :exogenous || throw(ArgumentError(
        "V2.2 shadow-chain manifest does not select the exogenous M3",
    ))
    bindings.feature_schema ==
        Tuple(String.(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES)) ||
        throw(ArgumentError(
            "V2.2 exogenous shadow-chain feature schema mismatch",
        ))
    ordered = _operational_v22_shadow_ordered_exogenous(error_states)
    all(error_state ->
            error_state.base_center_sha256 == artifact.base_center_sha256,
        ordered) || throw(ArgumentError(
            "V2.2 exogenous M3 base-center identity mismatch",
        ))
    current_hashes = ntuple(
        index -> operational_v22_error_exogenous_sha256(ordered[index]),
        length(ordered),
    )
    current_hashes == artifact.m3_sha256_by_horizon || throw(ArgumentError(
        "V2.2 exogenous M3 artifact-set identity mismatch",
    ))
    return artifact
end

function validate_operational_v22_shadow_chain(
        artifact::OperationalV22ShadowChainArtifact,
        bindings::OperationalV22ShadowBindings,
        driver::OperationalV22DriverArtifact,
        core::OperationalCore,
        error_states::_OperationalV22ExogenousArtifactCollection,
        conformal::ConformalCalibration)
    return _operational_v22_validate_shadow_components(
        artifact, bindings, driver, core, error_states, conformal,
    )
end

function _operational_v22_shadow_validate_base_record(
        artifact::OperationalV22ShadowChainArtifact,
        base_center::OperationalV22BaseCenterForecast)
    base_center.horizon_hours in artifact.supported_horizons_hours ||
        throw(ArgumentError(
            "unsupported V2.2 issue-relative horizon=$(base_center.horizon_hours)",
        ))
    expected_anchor = base_center.issue_time - Hour(artifact.anchor_lag_hours)
    base_center.anchor_time == expected_anchor || throw(ArgumentError(
        "V2.2 shadow-chain anchor does not match the artifact anchor policy",
    ))
    base_center.target_time ==
        base_center.issue_time + Hour(base_center.horizon_hours) ||
        throw(ArgumentError("V2.2 shadow-chain target is not issue-relative"))
    base_center.base_center_sha256 == artifact.base_center_sha256 ||
        throw(ArgumentError(
            "V2.2 shadow-chain supplied center identity does not match",
        ))
    return base_center
end

"Research-only AR-control shadow arithmetic without an issued-path gate."
function operational_v22_shadow_research_predict(
        artifact::OperationalV22ShadowChainArtifact,
        bindings::OperationalV22ShadowBindings,
        driver::OperationalV22DriverArtifact,
        core::OperationalCore,
        error_state::OperationalV22ErrorStateArtifact,
        conformal::ConformalCalibration,
        base_center::OperationalV22BaseCenterForecast,
        latest_dst_nt::Real,
        innovations::AbstractVector{<:OperationalV22H1Innovation})
    _operational_v22_validate_shadow_components(
        artifact, bindings, driver, core, error_state, conformal,
    )
    _operational_v22_shadow_validate_base_record(artifact, base_center)
    latest = _operational_v22_shadow_float(latest_dst_nt, "latest Dst")
    corrected = operational_v22_error_state_predict(
        error_state,
        base_center.issue_time,
        base_center.horizon_hours,
        artifact.base_center_sha256,
        base_center.prediction_dst_nt,
        innovations,
    )
    corrected.fallback_reason == :base_center_mismatch && throw(ArgumentError(
        "V2.2 shadow-chain innovation history belongs to a different center",
    ))
    lo, hi = conformal_interval(
        conformal,
        corrected.pred_dst_nt,
        base_center.horizon_hours,
        latest,
    )
    return merge(corrected, (
        lo_dst_nt=lo,
        hi_dst_nt=hi,
        issue_time=base_center.issue_time,
        anchor_time=base_center.anchor_time,
        target_time=base_center.target_time,
        issue_relative_horizon_hours=base_center.horizon_hours,
        base_center_sha256=artifact.base_center_sha256,
        chain_sha256=operational_v22_shadow_chain_sha256(artifact),
        product_version=artifact.bindings.product_version,
    ))
end

"Research-only full-M3 shadow arithmetic without an issued-path gate."
function operational_v22_shadow_research_predict(
        artifact::OperationalV22ShadowChainArtifact,
        bindings::OperationalV22ShadowBindings,
        driver::OperationalV22DriverArtifact,
        core::OperationalCore,
        error_states::_OperationalV22ExogenousArtifactCollection,
        conformal::ConformalCalibration,
        base_center::OperationalV22BaseCenterForecast,
        latest_dst_nt::Real,
        issue::OperationalV22ErrorExogenousIssue,
        issue_history::AbstractVector{<:OperationalV22ErrorExogenousIssue},
        innovations::AbstractVector{<:OperationalV22H1Innovation})
    _operational_v22_validate_shadow_components(
        artifact, bindings, driver, core, error_states, conformal,
    )
    _operational_v22_shadow_validate_base_record(artifact, base_center)
    issue.issue_time == base_center.issue_time || throw(ArgumentError(
        "V2.2 exogenous issue snapshot does not match the base-center issue",
    ))
    issue.base_center_sha256 == artifact.base_center_sha256 ||
        throw(ArgumentError(
            "V2.2 exogenous issue snapshot belongs to a different center",
        ))
    ordered = _operational_v22_shadow_ordered_exogenous(error_states)
    horizon_index = findfirst(
        ==(base_center.horizon_hours), artifact.supported_horizons_hours,
    )
    horizon_index === nothing && throw(ErrorException(
        "V2.2 exogenous horizon selection failed after validation",
    ))
    selected = ordered[horizon_index]
    latest = _operational_v22_shadow_float(latest_dst_nt, "latest Dst")
    corrected = operational_v22_error_exogenous_predict(
        selected,
        issue,
        base_center.horizon_hours,
        artifact.base_center_sha256,
        base_center.prediction_dst_nt,
        issue_history,
        innovations,
    )
    corrected.fallback_reason == :base_center_mismatch && throw(ArgumentError(
        "V2.2 exogenous history belongs to a different center",
    ))
    lo, hi = conformal_interval(
        conformal,
        corrected.pred_dst_nt,
        base_center.horizon_hours,
        latest,
    )
    return merge(corrected, (
        lo_dst_nt=lo,
        hi_dst_nt=hi,
        issue_time=base_center.issue_time,
        anchor_time=base_center.anchor_time,
        target_time=base_center.target_time,
        issue_relative_horizon_hours=base_center.horizon_hours,
        base_center_sha256=artifact.base_center_sha256,
        chain_sha256=operational_v22_shadow_chain_sha256(artifact),
        product_version=artifact.bindings.product_version,
    ))
end

"Fail closed until a frozen issued-path gate artifact and proof are defined."
function operational_v22_shadow_predict(args...; kwargs...)
    throw(ArgumentError(
        "operational V2.2 shadow prediction is disabled until a frozen " *
        "issued-path gate artifact/proof exists; use " *
        "operational_v22_shadow_research_predict only for offline research",
    ))
end

"Return the checksum of every field in the composite identity manifest."
function operational_v22_shadow_chain_sha256(
        artifact::OperationalV22ShadowChainArtifact)
    io = IOBuffer()
    for value in (
            OPERATIONAL_V22_SHADOW_SCHEMA_VERSION,
            artifact.label,
            artifact.bindings.product_version,
            artifact.bindings.receipt_pair_contract_sha256,
            artifact.bindings.transport_support_sha256,
            artifact.bindings.anchor_pressure_contract_sha256,
            artifact.bindings.conformal_sidecar_sha256,
            artifact.bindings.point_calibration_sha256,
            artifact.driver_sha256,
            artifact.core_sha256,
            artifact.base_center_sha256,
            artifact.m3_kind,
            artifact.conformal_sha256,
            artifact.anchor_lag_hours,
        )
        _operational_v22_shadow_hash_token(io, value)
    end
    for values in (
            artifact.bindings.feature_schema,
            artifact.supported_horizons_hours,
            artifact.m3_sha256_by_horizon,
        )
        _operational_v22_shadow_hash_token(io, length(values))
        for value in values
            _operational_v22_shadow_hash_token(io, value)
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

const _OPERATIONAL_V22_SHADOW_CSV_COLUMNS = (
    :schema_version,
    :artifact_sha256,
    :label,
    :product_version,
    :receipt_pair_contract_sha256,
    :transport_support_sha256,
    :anchor_pressure_contract_sha256,
    :conformal_sidecar_sha256,
    :point_calibration_sha256,
    :driver_sha256,
    :core_sha256,
    :base_center_sha256,
    :m3_kind,
    :m3_sha256_by_horizon,
    :conformal_sha256,
    :feature_schema,
    :supported_horizons_hours,
    :anchor_lag_hours,
)

"Atomically write a one-row checksummed V2.2 shadow-chain manifest."
function write_operational_v22_shadow_chain(
        path::AbstractString,
        artifact::OperationalV22ShadowChainArtifact)
    row = (
        schema_version=OPERATIONAL_V22_SHADOW_SCHEMA_VERSION,
        artifact_sha256=operational_v22_shadow_chain_sha256(artifact),
        label=artifact.label,
        product_version=artifact.bindings.product_version,
        receipt_pair_contract_sha256=
            artifact.bindings.receipt_pair_contract_sha256,
        transport_support_sha256=artifact.bindings.transport_support_sha256,
        anchor_pressure_contract_sha256=
            artifact.bindings.anchor_pressure_contract_sha256,
        conformal_sidecar_sha256=artifact.bindings.conformal_sidecar_sha256,
        point_calibration_sha256=artifact.bindings.point_calibration_sha256,
        driver_sha256=artifact.driver_sha256,
        core_sha256=artifact.core_sha256,
        base_center_sha256=artifact.base_center_sha256,
        m3_kind=String(artifact.m3_kind),
        m3_sha256_by_horizon=join(artifact.m3_sha256_by_horizon, ';'),
        conformal_sha256=artifact.conformal_sha256,
        feature_schema=join(artifact.bindings.feature_schema, ';'),
        supported_horizons_hours=join(artifact.supported_horizons_hours, ';'),
        anchor_lag_hours=artifact.anchor_lag_hours,
    )
    target = String(path)
    mkpath(dirname(abspath(target)))
    return _write_selection_csv(target, [row])
end

function _operational_v22_shadow_csv_int(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "V2.2 shadow-chain artifact $field must be an integer",
    ))
    numeric = Float64(value)
    isfinite(numeric) && isinteger(numeric) &&
        typemin(Int) <= numeric <= typemax(Int) || throw(ArgumentError(
            "V2.2 shadow-chain artifact $field must be an integer",
        ))
    return Int(numeric)
end

"Read and validate a checksummed V2.2 shadow-chain manifest."
function read_operational_v22_shadow_chain(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "V2.2 shadow-chain artifact must be a regular non-symlink file: $source",
    ))
    string_columns = Dict(
        column => String
        for column in (
            :schema_version,
            :artifact_sha256,
            :label,
            :product_version,
            :receipt_pair_contract_sha256,
            :transport_support_sha256,
            :anchor_pressure_contract_sha256,
            :conformal_sidecar_sha256,
            :point_calibration_sha256,
            :driver_sha256,
            :core_sha256,
            :base_center_sha256,
            :m3_kind,
            :m3_sha256_by_horizon,
            :conformal_sha256,
            :feature_schema,
            :supported_horizons_hours,
        )
    )
    df = CSV.read(source, DataFrame; types=string_columns)
    names(df) == collect(String.(_OPERATIONAL_V22_SHADOW_CSV_COLUMNS)) ||
        throw(ArgumentError("V2.2 shadow-chain artifact CSV schema is invalid"))
    nrow(df) == 1 || throw(ArgumentError(
        "V2.2 shadow-chain artifact must contain exactly one row",
    ))
    for column in _OPERATIONAL_V22_SHADOW_CSV_COLUMNS
        ismissing(df[1, column]) && throw(ArgumentError(
            "V2.2 shadow-chain artifact contains missing $column",
        ))
    end
    String(df.schema_version[1]) == OPERATIONAL_V22_SHADOW_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported V2.2 shadow-chain artifact schema"))
    checksum = _operational_v22_shadow_sha(
        String(df.artifact_sha256[1]), "artifact checksum",
    )
    bindings = OperationalV22ShadowBindings(
        receipt_pair_contract_sha256=String(df.receipt_pair_contract_sha256[1]),
        transport_support_sha256=String(df.transport_support_sha256[1]),
        anchor_pressure_contract_sha256=
            String(df.anchor_pressure_contract_sha256[1]),
        conformal_sidecar_sha256=String(df.conformal_sidecar_sha256[1]),
        point_calibration_sha256=String(df.point_calibration_sha256[1]),
        feature_schema=split(String(df.feature_schema[1]), ';'),
        product_version=String(df.product_version[1]),
    )
    horizons = Tuple(
        parse(Int, value)
        for value in split(String(df.supported_horizons_hours[1]), ';')
    )
    length(horizons) == 6 || throw(ArgumentError(
        "V2.2 shadow-chain artifact horizon schema has the wrong length",
    ))
    m3_hashes = Tuple(
        String(value)
        for value in split(String(df.m3_sha256_by_horizon[1]), ';')
    )
    length(m3_hashes) == 6 || throw(ArgumentError(
        "V2.2 shadow-chain artifact M3 checksum schema has the wrong length",
    ))
    artifact = OperationalV22ShadowChainArtifact(
        String(df.label[1]),
        bindings,
        String(df.driver_sha256[1]),
        String(df.core_sha256[1]),
        String(df.base_center_sha256[1]),
        Symbol(String(df.m3_kind[1])),
        convert(NTuple{6,String}, m3_hashes),
        String(df.conformal_sha256[1]),
        convert(NTuple{6,Int}, horizons),
        _operational_v22_shadow_csv_int(df.anchor_lag_hours[1], "anchor_lag_hours"),
        Val(:validated),
    )
    operational_v22_shadow_chain_sha256(artifact) == checksum ||
        throw(ArgumentError("V2.2 shadow-chain artifact checksum mismatch"))
    return artifact
end
