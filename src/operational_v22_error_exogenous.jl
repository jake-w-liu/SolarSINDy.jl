# Stable group-sparse exogenous error correction for Operational V2.2-M3.

import SHA

"Frozen issue offsets used by every lagged exogenous family."
const OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H =
    OPERATIONAL_V22_ERROR_LAGS_H

"Frozen lagged scalar families, ordered as in the preregistered temporal probe."
const OPERATIONAL_V22_ERROR_EXOGENOUS_TEMPORAL_VARIABLES = (
    :latest_dst_nt,
    :Bz_nt,
    :VBsouth_mvm,
    :sqrt_Pdyn_npa,
    :h1_innovation_nt,
)

"Minimal causal M2 trajectory block: the five states at the requested lead."
const OPERATIONAL_V22_ERROR_EXOGENOUS_M2_FEATURES = (
    :m2_endpoint_Bx_nt,
    :m2_endpoint_By_nt,
    :m2_endpoint_Bz_nt,
    :m2_endpoint_logV,
    :m2_endpoint_logn,
)

"Exact 73-feature order used by both fitting and prediction."
const OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES = let
    names = Symbol[OPERATIONAL_V22_RESIDUAL_FEATURES...]
    for variable in OPERATIONAL_V22_ERROR_EXOGENOUS_TEMPORAL_VARIABLES,
            lag in OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H
        push!(names, Symbol(variable, "_lag_", lag, "h"))
    end
    append!(names, OPERATIONAL_V22_ERROR_EXOGENOUS_M2_FEATURES)
    push!(names, :m2_core_center_dst_nt)
    Tuple(names)
end

"Frozen group order: 22 singletons, five lag families, one M2 block, one center."
const OPERATIONAL_V22_ERROR_EXOGENOUS_GROUPS = (
    OPERATIONAL_V22_RESIDUAL_FEATURES...,
    :latest_dst_nt_lag_family,
    :Bz_nt_lag_family,
    :VBsouth_mvm_lag_family,
    :sqrt_Pdyn_npa_lag_family,
    :h1_innovation_nt_lag_family,
    :m2_endpoint_state,
    :m2_core_center_dst_nt,
)

const OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS = let
    groups = collect(1:length(OPERATIONAL_V22_RESIDUAL_FEATURES))
    first_lag_group = length(OPERATIONAL_V22_RESIDUAL_FEATURES) + 1
    for variable_index in eachindex(
            OPERATIONAL_V22_ERROR_EXOGENOUS_TEMPORAL_VARIABLES)
        append!(groups, fill(
            first_lag_group + variable_index - 1,
            length(OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H),
        ))
    end
    append!(groups, fill(length(OPERATIONAL_V22_ERROR_EXOGENOUS_GROUPS) - 1,
                         length(OPERATIONAL_V22_ERROR_EXOGENOUS_M2_FEATURES)))
    push!(groups, length(OPERATIONAL_V22_ERROR_EXOGENOUS_GROUPS))
    Tuple(groups)
end

const OPERATIONAL_V22_ERROR_EXOGENOUS_SUPPORTED_MODEL_STEPS =
    OPERATIONAL_V22_ERROR_SUPPORTED_MODEL_STEPS
const OPERATIONAL_V22_ERROR_EXOGENOUS_RIDGE_GRID =
    OPERATIONAL_V22_DRIVER_RIDGE_GRID
const OPERATIONAL_V22_ERROR_EXOGENOUS_THRESHOLD_GRID =
    OPERATIONAL_V22_DRIVER_THRESHOLD_GRID
const OPERATIONAL_V22_ERROR_EXOGENOUS_MAX_THRESHOLD_ITERATIONS = 20
const OPERATIONAL_V22_ERROR_EXOGENOUS_MAX_SPECTRAL_RADIUS =
    OPERATIONAL_V22_ERROR_MAX_SPECTRAL_RADIUS
const OPERATIONAL_V22_ERROR_EXOGENOUS_SCHEMA_VERSION =
    "operational_v2_2_m3_error_exogenous_v1"
const OPERATIONAL_V22_ERROR_EXOGENOUS_PACKAGE_VERSION = "SolarSINDy-0.2.1"

"""
Positions in `OPERATIONAL_V22_RESIDUAL_FEATURES` of the lagged families whose history is read from
the issue-time feature vector.

Every temporal family except `h1_innovation_nt` — which comes from the matured innovation records
rather than from the issue features — is one of the predeclared residual features, so the positions
are derived from the two schemas instead of being written out as `(1, 5, 7, 11)`. Reordering
`OPERATIONAL_V22_RESIDUAL_FEATURES` would otherwise have silently populated the lag families from the
wrong columns while every length check still passed.
"""
const _OPERATIONAL_V22_ERROR_EXOGENOUS_ISSUE_INDICES = let
    residual = collect(OPERATIONAL_V22_RESIDUAL_FEATURES)
    indices = Int[]
    unresolved = Symbol[]
    for variable in OPERATIONAL_V22_ERROR_EXOGENOUS_TEMPORAL_VARIABLES
        position = findfirst(==(variable), residual)
        position === nothing ? push!(unresolved, variable) : push!(indices, position)
    end
    unresolved == [:h1_innovation_nt] || error(
        "V2.2-M3 exogenous temporal families outside the issue-feature schema changed: $unresolved",
    )
    Tuple(indices)
end

const _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE =
    length(OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES)
const _OPERATIONAL_V22_ERROR_EXOGENOUS_NGROUP =
    length(OPERATIONAL_V22_ERROR_EXOGENOUS_GROUPS)
const _OPERATIONAL_V22_ERROR_EXOGENOUS_M2_STEPS =
    OPERATIONAL_V22_DRIVER_ROLLOUT_STEPS
const _OPERATIONAL_V22_ERROR_EXOGENOUS_FALLBACKS = (
    :none,
    :missing_issue_history,
    :duplicate_issue_record,
    :missing_innovation_history,
    :duplicate_innovation_record,
    :observation_not_mature,
    :base_center_mismatch,
)

function _operational_v22_error_exogenous_lead(value)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "V2.2-M3 exogenous model_step_hours must be an integer",
    ))
    lead = Int(value)
    lead in OPERATIONAL_V22_ERROR_EXOGENOUS_SUPPORTED_MODEL_STEPS ||
        throw(ArgumentError(
            "unsupported V2.2-M3 exogenous model_step_hours=$lead",
        ))
    return lead
end

function _operational_v22_error_exogenous_names(values, expected, field)
    length(values) == length(expected) || throw(DimensionMismatch(
        "V2.2-M3 exogenous $field has the wrong length",
    ))
    converted = Tuple(Symbol(value) for value in values)
    converted == expected || throw(ArgumentError(
        "V2.2-M3 exogenous $field order is not frozen",
    ))
    return converted
end

"Immutable horizon row; lag extraction uses only canonical H1 history rows."
struct OperationalV22ErrorExogenousIssue
    issue_time::DateTime
    available_at::DateTime
    base_center_sha256::String
    model_step_hours::Int
    feature_names::NTuple{length(OPERATIONAL_V22_RESIDUAL_FEATURES),Symbol}
    issue_features::NTuple{length(OPERATIONAL_V22_RESIDUAL_FEATURES),Float64}
    m2_state_names::NTuple{length(OPERATIONAL_V22_DRIVER_STATES),Symbol}
    m2_trajectory::NTuple{
        _OPERATIONAL_V22_ERROR_EXOGENOUS_M2_STEPS,
        NTuple{length(OPERATIONAL_V22_DRIVER_STATES),Float64},
    }

    function OperationalV22ErrorExogenousIssue(
            issue_time::DateTime,
            available_at::DateTime,
            base_center_sha256::String,
            model_step_hours::Int,
            feature_names::NTuple{
                length(OPERATIONAL_V22_RESIDUAL_FEATURES),Symbol},
            issue_features::NTuple{
                length(OPERATIONAL_V22_RESIDUAL_FEATURES),Float64},
            m2_state_names::NTuple{
                length(OPERATIONAL_V22_DRIVER_STATES),Symbol},
            m2_trajectory::NTuple{
                _OPERATIONAL_V22_ERROR_EXOGENOUS_M2_STEPS,
                NTuple{length(OPERATIONAL_V22_DRIVER_STATES),Float64},
            })
        available_at <= issue_time || throw(ArgumentError(
            "V2.2-M3 exogenous issue snapshot was not available at issue time",
        ))
        feature_names == OPERATIONAL_V22_RESIDUAL_FEATURES ||
            throw(ArgumentError(
                "V2.2-M3 exogenous issue-feature schema order is not frozen",
            ))
        m2_state_names == OPERATIONAL_V22_DRIVER_STATES ||
            throw(ArgumentError(
                "V2.2-M3 exogenous M2 state schema order is not frozen",
            ))
        all(isfinite, issue_features) || throw(ArgumentError(
            "V2.2-M3 exogenous issue features must be finite",
        ))
        all(row -> all(isfinite, row), m2_trajectory) ||
            throw(ArgumentError(
                "V2.2-M3 exogenous M2 trajectory states must be finite",
            ))
        return new(
            issue_time,
            available_at,
            _operational_v22_error_sha(
                base_center_sha256, "exogenous base-center checksum",
            ),
            _operational_v22_error_exogenous_lead(model_step_hours),
            feature_names,
            issue_features,
            m2_state_names,
            m2_trajectory,
        )
    end
end

function OperationalV22ErrorExogenousIssue(
        issue_time::DateTime,
        available_at::DateTime,
        base_center_sha256::AbstractString,
        issue_features::AbstractVector,
        m2_trajectory::AbstractMatrix;
        model_step_hours::Integer=1,
        feature_names=OPERATIONAL_V22_RESIDUAL_FEATURES,
        m2_state_names=OPERATIONAL_V22_DRIVER_STATES)
    length(feature_names) == length(OPERATIONAL_V22_RESIDUAL_FEATURES) ||
        throw(DimensionMismatch(
            "V2.2-M3 exogenous issue-feature schema has the wrong length",
        ))
    length(m2_state_names) == length(OPERATIONAL_V22_DRIVER_STATES) ||
        throw(DimensionMismatch(
            "V2.2-M3 exogenous M2 state schema has the wrong length",
        ))
    frozen_feature_names = Tuple(Symbol(value) for value in feature_names)
    frozen_state_names = Tuple(Symbol(value) for value in m2_state_names)
    length(issue_features) == length(OPERATIONAL_V22_RESIDUAL_FEATURES) ||
        throw(DimensionMismatch(
            "V2.2-M3 exogenous issue snapshot requires 22 features",
        ))
    size(m2_trajectory) == (
        _OPERATIONAL_V22_ERROR_EXOGENOUS_M2_STEPS,
        length(OPERATIONAL_V22_DRIVER_STATES),
    ) || throw(DimensionMismatch(
        "V2.2-M3 exogenous M2 trajectory must have size (14, 5)",
    ))
    features = ntuple(
        index -> _operational_v22_error_float(
            issue_features[index], "exogenous issue feature",
        ),
        length(OPERATIONAL_V22_RESIDUAL_FEATURES),
    )
    trajectory = ntuple(
        row -> ntuple(
            column -> _operational_v22_error_float(
                m2_trajectory[row, column], "exogenous M2 trajectory state",
            ),
            length(OPERATIONAL_V22_DRIVER_STATES),
        ),
        _OPERATIONAL_V22_ERROR_EXOGENOUS_M2_STEPS,
    )
    return OperationalV22ErrorExogenousIssue(
        issue_time,
        available_at,
        String(base_center_sha256),
        _operational_v22_error_exogenous_lead(model_step_hours),
        frozen_feature_names,
        features,
        frozen_state_names,
        trajectory,
    )
end

"Causally assembled model row, with an explicit fail-closed status."
struct OperationalV22ErrorExogenousFeatures
    issue_time::DateTime
    model_step_hours::Int
    base_center_sha256::String
    base_prediction_dst_nt::Float64
    feature_names::NTuple{_OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Symbol}
    values::NTuple{_OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Float64}
    ready::Bool
    fallback_reason::Symbol

    function OperationalV22ErrorExogenousFeatures(
            issue_time::DateTime,
            model_step_hours::Int,
            base_center_sha256::String,
            base_prediction_dst_nt::Float64,
            feature_names::NTuple{
                _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Symbol},
            values::NTuple{
                _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Float64},
            ready::Bool,
            fallback_reason::Symbol)
        _operational_v22_error_exogenous_lead(model_step_hours)
        feature_names == OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES ||
            throw(ArgumentError(
                "V2.2-M3 exogenous feature order is not frozen",
            ))
        fallback_reason in _OPERATIONAL_V22_ERROR_EXOGENOUS_FALLBACKS ||
            throw(ArgumentError(
                "unknown V2.2-M3 exogenous fallback reason: $fallback_reason",
            ))
        ready == (fallback_reason == :none) || throw(ArgumentError(
            "V2.2-M3 exogenous readiness and fallback reason disagree",
        ))
        all(isfinite, values) || throw(ArgumentError(
            "V2.2-M3 exogenous features must be finite",
        ))
        ready && values[end] != base_prediction_dst_nt && throw(ArgumentError(
            "V2.2-M3 exogenous center feature does not match the base prediction",
        ))
        return new(
            issue_time,
            model_step_hours,
            _operational_v22_error_sha(
                base_center_sha256, "exogenous base-center checksum",
            ),
            _operational_v22_error_float(
                base_prediction_dst_nt, "exogenous base prediction",
            ),
            feature_names,
            values,
            ready,
            fallback_reason,
        )
    end
end

function _operational_v22_error_exogenous_fallback(
        issue_time::DateTime,
        lead::Int,
        center_hash::String,
        base_prediction::Float64,
        reason::Symbol)
    return OperationalV22ErrorExogenousFeatures(
        issue_time,
        lead,
        center_hash,
        base_prediction,
        OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES,
        ntuple(_ -> 0.0, _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE),
        false,
        reason,
    )
end

function _operational_v22_error_exogenous_issue_index(
        issues::AbstractVector{<:OperationalV22ErrorExogenousIssue},
        current::OperationalV22ErrorExogenousIssue)
    required = Set(current.issue_time - Hour(lag)
                   for lag in OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H)
    issue_at = Dict{DateTime,OperationalV22ErrorExogenousIssue}()
    duplicates = Set{DateTime}()
    for issue in issues
        issue.model_step_hours == 1 || continue
        issue.issue_time in required || continue
        if haskey(issue_at, issue.issue_time)
            push!(duplicates, issue.issue_time)
        else
            issue_at[issue.issue_time] = issue
        end
    end
    return issue_at, duplicates
end

function _operational_v22_error_exogenous_innovation_index(
        records::AbstractVector{<:OperationalV22H1Innovation},
        issue_time::DateTime)
    required = Set(issue_time - Hour(lag)
                   for lag in OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H)
    record_at = Dict{DateTime,OperationalV22H1Innovation}()
    duplicates = Set{DateTime}()
    for record in records
        record.observation_available_at <= issue_time || continue
        record.issued_at in required || continue
        if haskey(record_at, record.issued_at)
            push!(duplicates, record.issued_at)
        else
            record_at[record.issued_at] = record
        end
    end
    return record_at, duplicates
end

"Build the exact horizon row from current-lead and canonical H1 issue snapshots."
function operational_v22_error_exogenous_features(
        issue::OperationalV22ErrorExogenousIssue,
        issue_history::AbstractVector{<:OperationalV22ErrorExogenousIssue},
        innovation_records::AbstractVector{<:OperationalV22H1Innovation},
        model_step_hours::Integer,
        base_center_sha256::AbstractString,
        base_prediction_dst_nt::Real)
    lead = _operational_v22_error_exogenous_lead(model_step_hours)
    issue.model_step_hours == lead || throw(ArgumentError(
        "V2.2-M3 exogenous current issue snapshot lead does not match prediction lead",
    ))
    center_hash = _operational_v22_error_sha(
        base_center_sha256, "current exogenous base-center checksum",
    )
    center_hash == issue.base_center_sha256 || throw(ArgumentError(
        "V2.2-M3 exogenous current base-center checksum does not match the issue snapshot",
    ))
    base = _operational_v22_error_float(
        base_prediction_dst_nt, "exogenous base prediction",
    )

    issue_at, duplicate_issues =
        _operational_v22_error_exogenous_issue_index(issue_history, issue)
    if !isempty(duplicate_issues)
        return _operational_v22_error_exogenous_fallback(
            issue.issue_time, lead, center_hash, base, :duplicate_issue_record,
        )
    end
    required_times = Tuple(
        issue.issue_time - Hour(lag)
        for lag in OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H
    )
    if !all(haskey(issue_at, time) for time in required_times)
        return _operational_v22_error_exogenous_fallback(
            issue.issue_time, lead, center_hash, base, :missing_issue_history,
        )
    end
    if !all(issue_at[time].base_center_sha256 == center_hash
            for time in required_times)
        return _operational_v22_error_exogenous_fallback(
            issue.issue_time, lead, center_hash, base, :base_center_mismatch,
        )
    end

    record_at, duplicate_records =
        _operational_v22_error_exogenous_innovation_index(
            innovation_records, issue.issue_time,
        )
    if !isempty(duplicate_records)
        return _operational_v22_error_exogenous_fallback(
            issue.issue_time, lead, center_hash, base,
            :duplicate_innovation_record,
        )
    end
    if !all(haskey(record_at, time) for time in required_times)
        return _operational_v22_error_exogenous_fallback(
            issue.issue_time, lead, center_hash, base,
            :missing_innovation_history,
        )
    end
    if !all(record_at[time].base_center_sha256 == center_hash
            for time in required_times)
        return _operational_v22_error_exogenous_fallback(
            issue.issue_time, lead, center_hash, base, :base_center_mismatch,
        )
    end
    values = Float64[issue.issue_features...]
    for feature_index in _OPERATIONAL_V22_ERROR_EXOGENOUS_ISSUE_INDICES
        for time in required_times
            push!(values, issue_at[time].issue_features[feature_index])
        end
    end
    for time in required_times
        push!(values, operational_v22_h1_innovation(record_at[time]))
    end
    endpoint = issue.m2_trajectory[2 * lead]
    append!(values, endpoint)
    push!(values, base)
    length(values) == _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE ||
        throw(ErrorException(
            "V2.2-M3 exogenous feature construction is incomplete",
        ))
    return OperationalV22ErrorExogenousFeatures(
        issue.issue_time,
        lead,
        center_hash,
        base,
        OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES,
        Tuple(values),
        true,
        :none,
    )
end

"One causally built feature row paired with a later-maturing verification target."
struct OperationalV22ErrorExogenousFitRow
    features::OperationalV22ErrorExogenousFeatures
    target_time::DateTime
    observation_available_at::DateTime
    observation_dst_nt::Float64

    function OperationalV22ErrorExogenousFitRow(
            features::OperationalV22ErrorExogenousFeatures,
            target_time::DateTime,
            observation_available_at::DateTime,
            observation_dst_nt::Float64)
        target_time == features.issue_time + Hour(features.model_step_hours) ||
            throw(ArgumentError(
                "V2.2-M3 exogenous target time does not match the forecast lead",
            ))
        observation_available_at >= target_time || throw(ArgumentError(
            "V2.2-M3 exogenous observation cannot be available before target time",
        ))
        return new(
            features,
            target_time,
            observation_available_at,
            _operational_v22_error_float(
                observation_dst_nt, "exogenous target observation",
            ),
        )
    end
end

function OperationalV22ErrorExogenousFitRow(
        features::OperationalV22ErrorExogenousFeatures,
        target_time::DateTime,
        observation_available_at::DateTime,
        observation_dst_nt::Real)
    return OperationalV22ErrorExogenousFitRow(
        features,
        target_time,
        observation_available_at,
        _operational_v22_error_float(
            observation_dst_nt, "exogenous target observation",
        ),
    )
end

function _operational_v22_error_exogenous_raw_ar_coefficients(
        coefficients,
        scale)
    first_innovation = length(OPERATIONAL_V22_RESIDUAL_FEATURES) +
        (length(OPERATIONAL_V22_ERROR_EXOGENOUS_TEMPORAL_VARIABLES) - 1) *
        length(OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H) + 1
    return ntuple(
        index -> coefficients[first_innovation + index - 1] /
                 scale[first_innovation + index - 1],
        length(OPERATIONAL_V22_ERROR_EXOGENOUS_LAGS_H),
    )
end

"Immutable lead-specific full M3 artifact, bound to one M2-plus-core center."
struct OperationalV22ErrorExogenousArtifact
    label::String
    base_center_sha256::String
    model_step_hours::Int
    feature_names::NTuple{_OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Symbol}
    feature_groups::NTuple{_OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Int}
    group_names::NTuple{_OPERATIONAL_V22_ERROR_EXOGENOUS_NGROUP,Symbol}
    support_mask::NTuple{_OPERATIONAL_V22_ERROR_EXOGENOUS_NGROUP,Bool}
    feature_center::NTuple{_OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Float64}
    feature_scale::NTuple{_OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Float64}
    intercept_nt::Float64
    coefficients::NTuple{_OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Float64}
    spectral_radius::Float64
    stability_limit::Float64
    ridge::Float64
    threshold::Float64
    fit_rows::Int
    threshold_iterations::Int

    function OperationalV22ErrorExogenousArtifact(
            label::String,
            base_center_sha256::String,
            model_step_hours::Int,
            feature_names::NTuple{
                _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Symbol},
            feature_groups::NTuple{
                _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Int},
            group_names::NTuple{
                _OPERATIONAL_V22_ERROR_EXOGENOUS_NGROUP,Symbol},
            support_mask::NTuple{
                _OPERATIONAL_V22_ERROR_EXOGENOUS_NGROUP,Bool},
            feature_center::NTuple{
                _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Float64},
            feature_scale::NTuple{
                _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Float64},
            intercept_nt::Float64,
            coefficients::NTuple{
                _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,Float64},
            ridge::Float64,
            threshold::Float64,
            fit_rows::Int,
            threshold_iterations::Int,
            ::Val{:validated})
        isempty(strip(label)) && throw(ArgumentError(
            "V2.2-M3 exogenous artifact label must not be empty",
        ))
        _operational_v22_error_exogenous_lead(model_step_hours)
        feature_names == OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES ||
            throw(ArgumentError(
                "V2.2-M3 exogenous artifact feature order is not frozen",
            ))
        feature_groups == OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS ||
            throw(ArgumentError(
                "V2.2-M3 exogenous artifact feature groups are not frozen",
            ))
        group_names == OPERATIONAL_V22_ERROR_EXOGENOUS_GROUPS ||
            throw(ArgumentError(
                "V2.2-M3 exogenous artifact group order is not frozen",
            ))
        all(isfinite, feature_center) && all(isfinite, feature_scale) &&
            all(>(0.0), feature_scale) || throw(ArgumentError(
                "V2.2-M3 exogenous normalization must be finite with positive scales",
            ))
        isfinite(intercept_nt) && all(isfinite, coefficients) ||
            throw(ArgumentError(
                "V2.2-M3 exogenous coefficients must be finite",
            ))
        ridge in OPERATIONAL_V22_ERROR_EXOGENOUS_RIDGE_GRID ||
            throw(ArgumentError(
                "V2.2-M3 exogenous ridge is not on the frozen grid",
            ))
        threshold in OPERATIONAL_V22_ERROR_EXOGENOUS_THRESHOLD_GRID ||
            throw(ArgumentError(
                "V2.2-M3 exogenous threshold is not on the frozen grid",
            ))
        fit_rows >= 2 || throw(ArgumentError(
            "V2.2-M3 exogenous artifact requires at least two fit rows",
        ))
        1 <= threshold_iterations <=
            OPERATIONAL_V22_ERROR_EXOGENOUS_MAX_THRESHOLD_ITERATIONS ||
            throw(ArgumentError(
                "V2.2-M3 exogenous threshold iterations must lie in 1:20",
            ))
        for group in 1:_OPERATIONAL_V22_ERROR_EXOGENOUS_NGROUP
            indices = findall(==(group), feature_groups)
            group_coefficients = coefficients[indices]
            if support_mask[group]
                norm(group_coefficients) >= threshold || throw(ArgumentError(
                    "V2.2-M3 exogenous selected groups must meet the threshold",
                ))
            else
                all(iszero, group_coefficients) || throw(ArgumentError(
                    "V2.2-M3 exogenous excluded groups must have exact zero coefficients",
                ))
            end
        end
        raw_ar = _operational_v22_error_exogenous_raw_ar_coefficients(
            coefficients, feature_scale,
        )
        radius = _operational_v22_error_spectral_radius(raw_ar)
        isfinite(radius) &&
            radius <= OPERATIONAL_V22_ERROR_EXOGENOUS_MAX_SPECTRAL_RADIUS ||
            throw(ArgumentError(
                "V2.2-M3 exogenous innovation recurrence is unstable: spectral radius=$radius",
            ))
        return new(
            label,
            _operational_v22_error_sha(
                base_center_sha256, "exogenous base-center checksum",
            ),
            model_step_hours,
            feature_names,
            feature_groups,
            group_names,
            support_mask,
            feature_center,
            feature_scale,
            intercept_nt,
            coefficients,
            radius,
            OPERATIONAL_V22_ERROR_EXOGENOUS_MAX_SPECTRAL_RADIUS,
            ridge,
            threshold,
            fit_rows,
            threshold_iterations,
        )
    end
end

function OperationalV22ErrorExogenousArtifact(
        base_center_sha256::AbstractString,
        model_step_hours::Integer,
        intercept_nt::Real,
        coefficients::AbstractVector{<:Real};
        feature_names=OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES,
        feature_groups=OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS,
        group_names=OPERATIONAL_V22_ERROR_EXOGENOUS_GROUPS,
        support_mask=nothing,
        feature_center=zeros(_OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE),
        feature_scale=ones(_OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE),
        ridge::Real=first(OPERATIONAL_V22_ERROR_EXOGENOUS_RIDGE_GRID),
        threshold::Real=first(OPERATIONAL_V22_ERROR_EXOGENOUS_THRESHOLD_GRID),
        fit_rows::Integer=2,
        threshold_iterations::Integer=1,
        label::AbstractString="operational-v2.2-m3-error-exogenous")
    lead = _operational_v22_error_exogenous_lead(model_step_hours)
    frozen_feature_names = _operational_v22_error_exogenous_names(
        feature_names, OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES,
        "artifact feature schema",
    )
    frozen_group_names = _operational_v22_error_exogenous_names(
        group_names, OPERATIONAL_V22_ERROR_EXOGENOUS_GROUPS,
        "artifact group schema",
    )
    length(feature_groups) == _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE ||
        throw(DimensionMismatch(
            "V2.2-M3 exogenous artifact feature-group map has the wrong length",
        ))
    all(value -> value isa Integer && !(value isa Bool), feature_groups) ||
        throw(ArgumentError(
            "V2.2-M3 exogenous feature-group indices must be integers",
        ))
    converted_groups = Tuple(Int(value) for value in feature_groups)
    converted_groups == OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS ||
        throw(ArgumentError(
            "V2.2-M3 exogenous artifact feature-group map is not frozen",
        ))
    length(coefficients) == _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE ||
        throw(DimensionMismatch(
            "V2.2-M3 exogenous artifact requires 73 coefficients",
        ))
    length(feature_center) == _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE ||
        throw(DimensionMismatch(
            "V2.2-M3 exogenous artifact requires 73 feature centers",
        ))
    length(feature_scale) == _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE ||
        throw(DimensionMismatch(
            "V2.2-M3 exogenous artifact requires 73 feature scales",
        ))
    converted_coefficients = ntuple(
        index -> _operational_v22_error_float(
            coefficients[index], "exogenous coefficient",
        ),
        _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,
    )
    converted_center = ntuple(
        index -> _operational_v22_error_float(
            feature_center[index], "exogenous feature center",
        ),
        _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,
    )
    converted_scale = ntuple(
        index -> _operational_v22_error_float(
            feature_scale[index], "exogenous feature scale",
        ),
        _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE,
    )
    if support_mask === nothing
        converted_support = ntuple(
            group -> any(
                index -> converted_groups[index] == group &&
                         !iszero(converted_coefficients[index]),
                eachindex(converted_coefficients),
            ),
            _OPERATIONAL_V22_ERROR_EXOGENOUS_NGROUP,
        )
    else
        length(support_mask) == _OPERATIONAL_V22_ERROR_EXOGENOUS_NGROUP ||
            throw(DimensionMismatch(
                "V2.2-M3 exogenous artifact requires 29 support flags",
            ))
        all(value -> value isa Bool, support_mask) || throw(ArgumentError(
            "V2.2-M3 exogenous support flags must be Boolean",
        ))
        converted_support = Tuple(support_mask)
    end
    threshold_iterations isa Bool && throw(ArgumentError(
        "V2.2-M3 exogenous threshold_iterations must be an integer",
    ))
    return OperationalV22ErrorExogenousArtifact(
        String(label),
        String(base_center_sha256),
        lead,
        frozen_feature_names,
        converted_groups,
        frozen_group_names,
        converted_support,
        converted_center,
        converted_scale,
        _operational_v22_error_float(intercept_nt, "exogenous intercept"),
        converted_coefficients,
        _operational_v22_error_float(ridge, "exogenous ridge"),
        _operational_v22_error_float(threshold, "exogenous threshold"),
        Int(fit_rows),
        Int(threshold_iterations),
        Val(:validated),
    )
end

function _operational_v22_error_exogenous_ridge_fit(
        design::Matrix{Float64},
        target::Vector{Float64},
        active_groups::BitVector,
        ridge::Float64)
    coefficients = zeros(Float64, size(design, 2))
    target_mean = mean(target)
    selected = findall(
        index -> active_groups[
            OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS[index]],
        axes(design, 2),
    )
    isempty(selected) && return target_mean, coefficients
    selected_design = design[:, selected]
    predictor_mean = vec(mean(selected_design; dims=1))
    centered_design = selected_design .- transpose(predictor_mean)
    centered_target = target .- target_mean
    gram = transpose(centered_design) * centered_design
    gram[diagind(gram)] .+= ridge
    fitted = cholesky(Symmetric(gram); check=true) \
             (transpose(centered_design) * centered_target)
    all(isfinite, fitted) || throw(ArgumentError(
        "V2.2-M3 exogenous ridge coefficients are non-finite",
    ))
    coefficients[selected] .= fitted
    intercept = target_mean - dot(fitted, predictor_mean)
    isfinite(intercept) || throw(ArgumentError(
        "V2.2-M3 exogenous ridge intercept is non-finite",
    ))
    return intercept, coefficients
end

function _operational_v22_error_exogenous_threshold_fit(
        design::Matrix{Float64},
        target::Vector{Float64},
        ridge::Float64,
        threshold::Float64)
    active = trues(_OPERATIONAL_V22_ERROR_EXOGENOUS_NGROUP)
    intercept = 0.0
    coefficients = zeros(Float64, size(design, 2))
    for iteration in 1:OPERATIONAL_V22_ERROR_EXOGENOUS_MAX_THRESHOLD_ITERATIONS
        intercept, coefficients = _operational_v22_error_exogenous_ridge_fit(
            design, target, active, ridge,
        )
        next_active = copy(active)
        for group in eachindex(active)
            active[group] || continue
            indices = findall(
                ==(group), OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS,
            )
            next_active[group] = norm(coefficients[indices]) >= threshold
        end
        next_active == active &&
            return intercept, coefficients, active, iteration
        active = next_active
    end
    intercept, coefficients = _operational_v22_error_exogenous_ridge_fit(
        design, target, active, ridge,
    )
    all(group -> !active[group] || norm(coefficients[
            findall(==(group), OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS),
        ]) >= threshold, eachindex(active)) || throw(ErrorException(
            "V2.2-M3 exogenous group thresholding did not converge in 20 iterations",
        ))
    return intercept, coefficients, active,
           OPERATIONAL_V22_ERROR_EXOGENOUS_MAX_THRESHOLD_ITERATIONS
end

"Fit one deterministic lead-specific sparse model from causally built rows."
function fit_operational_v22_error_exogenous(
        rows::AbstractVector{<:OperationalV22ErrorExogenousFitRow};
        base_center_sha256::AbstractString,
        model_step_hours::Integer,
        fit_as_of::DateTime,
        ridge::Real,
        threshold::Real,
        minimum_rows::Integer=32,
        label::AbstractString="operational-v2.2-m3-error-exogenous")
    center_hash = _operational_v22_error_sha(
        base_center_sha256, "exogenous base-center checksum",
    )
    lead = _operational_v22_error_exogenous_lead(model_step_hours)
    ridge_value = _operational_v22_error_float(ridge, "exogenous ridge")
    threshold_value = _operational_v22_error_float(
        threshold, "exogenous threshold",
    )
    ridge_value in OPERATIONAL_V22_ERROR_EXOGENOUS_RIDGE_GRID ||
        throw(ArgumentError(
            "V2.2-M3 exogenous ridge is not on the frozen grid",
        ))
    threshold_value in OPERATIONAL_V22_ERROR_EXOGENOUS_THRESHOLD_GRID ||
        throw(ArgumentError(
            "V2.2-M3 exogenous threshold is not on the frozen grid",
        ))
    minimum_rows >= 2 || throw(ArgumentError(
        "V2.2-M3 exogenous minimum_rows must be at least two",
    ))
    eligible = [
        row for row in rows
        if row.features.ready &&
           row.features.model_step_hours == lead &&
           row.features.base_center_sha256 == center_hash &&
           row.features.issue_time < fit_as_of &&
           row.observation_available_at <= fit_as_of
    ]
    sort!(eligible; by=row -> row.features.issue_time)
    length(unique(row.features.issue_time for row in eligible)) ==
        length(eligible) || throw(ArgumentError(
            "V2.2-M3 exogenous fitting rows contain duplicate issue times",
        ))
    length(eligible) >= minimum_rows || throw(ArgumentError(
        "V2.2-M3 exogenous fitting has $(length(eligible)) complete rows; " *
        "minimum_rows=$minimum_rows",
    ))
    design = Matrix{Float64}(undef, length(eligible),
                             _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE)
    target = Vector{Float64}(undef, length(eligible))
    for (row_index, row) in pairs(eligible)
        design[row_index, :] .= row.features.values
        target[row_index] = row.observation_dst_nt -
                            row.features.base_prediction_dst_nt
    end
    all(isfinite, design) && all(isfinite, target) || throw(ArgumentError(
        "V2.2-M3 exogenous fitting inputs must be finite",
    ))
    feature_center = vec(mean(design; dims=1))
    feature_scale = vec(std(design; dims=1, corrected=false))
    for index in eachindex(feature_scale)
        iszero(feature_scale[index]) && (feature_scale[index] = 1.0)
    end
    all(isfinite, feature_center) &&
        all(value -> isfinite(value) && value > 0.0, feature_scale) ||
        throw(ArgumentError(
            "V2.2-M3 exogenous normalization exceeds the supported range",
        ))
    standardized = (design .- transpose(feature_center)) ./
                   transpose(feature_scale)
    intercept, coefficients, support, iterations =
        _operational_v22_error_exogenous_threshold_fit(
            standardized, target, ridge_value, threshold_value,
        )
    artifact = OperationalV22ErrorExogenousArtifact(
        center_hash,
        lead,
        intercept,
        coefficients;
        support_mask=Tuple(support),
        feature_center=feature_center,
        feature_scale=feature_scale,
        ridge=ridge_value,
        threshold=threshold_value,
        fit_rows=length(eligible),
        threshold_iterations=iterations,
        label=label,
    )
    all(row -> isfinite(_operational_v22_error_exogenous_raw_predict(
            artifact, row.features.values,
        )), eligible) || throw(ErrorException(
            "V2.2-M3 exogenous fitted predictions are non-finite",
        ))
    return artifact
end

function _operational_v22_error_exogenous_raw_predict(
        artifact::OperationalV22ErrorExogenousArtifact,
        values)
    length(values) == _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE ||
        throw(DimensionMismatch(
            "V2.2-M3 exogenous prediction requires 73 features",
        ))
    correction = artifact.intercept_nt
    for index in eachindex(artifact.coefficients)
        correction += artifact.coefficients[index] *
            (_operational_v22_error_float(
                values[index], "exogenous prediction feature",
            ) - artifact.feature_center[index]) /
            artifact.feature_scale[index]
    end
    isfinite(correction) || throw(ErrorException(
        "V2.2-M3 exogenous correction became non-finite",
    ))
    return correction
end

"Predict a bounded correction; incomplete issue-time features leave the center unchanged."
function operational_v22_error_exogenous_predict(
        artifact::OperationalV22ErrorExogenousArtifact,
        issue::OperationalV22ErrorExogenousIssue,
        model_step_hours::Integer,
        base_center_sha256::AbstractString,
        base_prediction_dst_nt::Real,
        issue_history::AbstractVector{<:OperationalV22ErrorExogenousIssue},
        innovation_records::AbstractVector{<:OperationalV22H1Innovation})
    lead = _operational_v22_error_exogenous_lead(model_step_hours)
    lead == artifact.model_step_hours || throw(ArgumentError(
        "V2.2-M3 exogenous prediction lead does not match the artifact",
    ))
    center_hash = _operational_v22_error_sha(
        base_center_sha256, "current exogenous base-center checksum",
    )
    center_hash == artifact.base_center_sha256 || throw(ArgumentError(
        "V2.2-M3 exogenous current base-center checksum does not match the artifact",
    ))
    base = _operational_v22_error_float(
        base_prediction_dst_nt, "exogenous base prediction",
    )
    features = operational_v22_error_exogenous_features(
        issue,
        issue_history,
        innovation_records,
        lead,
        center_hash,
        base,
    )
    cap = _operational_v22_error_cap(lead)
    checksum = operational_v22_error_exogenous_sha256(artifact)
    if !features.ready
        return (
            pred_dst_nt=base,
            raw_correction_nt=0.0,
            correction_nt=0.0,
            correction_cap_nt=cap,
            correction_was_capped=false,
            correction_applied=false,
            fallback_reason=features.fallback_reason,
            artifact_sha256=checksum,
        )
    end
    raw = _operational_v22_error_exogenous_raw_predict(
        artifact, features.values,
    )
    correction = clamp(raw, -cap, cap)
    prediction = base + correction
    isfinite(prediction) || throw(ErrorException(
        "V2.2-M3 exogenous corrected forecast became non-finite",
    ))
    return (
        pred_dst_nt=prediction,
        raw_correction_nt=raw,
        correction_nt=correction,
        correction_cap_nt=cap,
        correction_was_capped=correction != raw,
        correction_applied=true,
        fallback_reason=:none,
        artifact_sha256=checksum,
    )
end

function _operational_v22_error_exogenous_hash_token(io::IO, value)
    text = value isa Float64 ? bitstring(value) : string(value)
    kind = string(typeof(value))
    print(io, ncodeunits(kind), ':', kind, ':', ncodeunits(text), ':', text, '|')
    return nothing
end

"Return the byte-significant SHA-256 identity of a full M3 artifact."
function operational_v22_error_exogenous_sha256(
        artifact::OperationalV22ErrorExogenousArtifact)
    io = IOBuffer()
    for value in (
            OPERATIONAL_V22_ERROR_EXOGENOUS_SCHEMA_VERSION,
            OPERATIONAL_V22_ERROR_EXOGENOUS_PACKAGE_VERSION,
            artifact.label,
            artifact.base_center_sha256,
            artifact.model_step_hours,
            artifact.intercept_nt,
            _operational_v22_hashable_spectral_radius(artifact.spectral_radius),
            artifact.stability_limit,
            artifact.ridge,
            artifact.threshold,
            artifact.fit_rows,
            artifact.threshold_iterations,
        )
        _operational_v22_error_exogenous_hash_token(io, value)
    end
    for values in (
            artifact.feature_names,
            artifact.feature_groups,
            artifact.group_names,
            artifact.support_mask,
            artifact.feature_center,
            artifact.feature_scale,
            artifact.coefficients,
        )
        _operational_v22_error_exogenous_hash_token(io, length(values))
        for value in values
            _operational_v22_error_exogenous_hash_token(io, value)
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

const _OPERATIONAL_V22_ERROR_EXOGENOUS_CSV_COLUMNS = (
    :schema_version,
    :package_version,
    :artifact_sha256,
    :label,
    :base_center_sha256,
    :model_step_hours,
    :intercept_nt,
    :spectral_radius,
    :stability_limit,
    :ridge,
    :threshold,
    :fit_rows,
    :threshold_iterations,
    :feature_index,
    :feature_name,
    :group_index,
    :group_name,
    :group_selected,
    :feature_center,
    :feature_scale,
    :coefficient,
)

function _operational_v22_error_exogenous_atomic_csv(path::String, rows)
    target = abspath(path)
    mkpath(dirname(target))
    _write_selection_csv(target, rows)
    return path
end

"Atomically write a strictly versioned, checksummed full M3 artifact."
function write_operational_v22_error_exogenous(
        path::AbstractString,
        artifact::OperationalV22ErrorExogenousArtifact)
    checksum = operational_v22_error_exogenous_sha256(artifact)
    rows = [(
        schema_version=OPERATIONAL_V22_ERROR_EXOGENOUS_SCHEMA_VERSION,
        package_version=OPERATIONAL_V22_ERROR_EXOGENOUS_PACKAGE_VERSION,
        artifact_sha256=checksum,
        label=artifact.label,
        base_center_sha256=artifact.base_center_sha256,
        model_step_hours=artifact.model_step_hours,
        intercept_nt=artifact.intercept_nt,
        spectral_radius=artifact.spectral_radius,
        stability_limit=artifact.stability_limit,
        ridge=artifact.ridge,
        threshold=artifact.threshold,
        fit_rows=artifact.fit_rows,
        threshold_iterations=artifact.threshold_iterations,
        feature_index=index,
        feature_name=String(artifact.feature_names[index]),
        group_index=artifact.feature_groups[index],
        group_name=String(artifact.group_names[artifact.feature_groups[index]]),
        group_selected=artifact.support_mask[artifact.feature_groups[index]],
        feature_center=artifact.feature_center[index],
        feature_scale=artifact.feature_scale[index],
        coefficient=artifact.coefficients[index],
    ) for index in eachindex(artifact.feature_names)]
    return _operational_v22_error_exogenous_atomic_csv(String(path), rows)
end

function _operational_v22_error_exogenous_consistent(
        df::DataFrame, column::Symbol)
    values = df[!, column]
    any(ismissing, values) && throw(ArgumentError(
        "V2.2-M3 exogenous artifact $column contains missing values",
    ))
    value = first(values)
    all(isequal(value), values) || throw(ArgumentError(
        "V2.2-M3 exogenous artifact $column is inconsistent",
    ))
    return value
end

"Read and fully validate a checksummed full M3 artifact."
function read_operational_v22_error_exogenous(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "V2.2-M3 exogenous artifact must be a regular non-symlink file: $source",
    ))
    df = CSV.read(source, DataFrame)
    names(df) == collect(String.(_OPERATIONAL_V22_ERROR_EXOGENOUS_CSV_COLUMNS)) ||
        throw(ArgumentError(
            "V2.2-M3 exogenous artifact CSV schema is invalid",
        ))
    nrow(df) == _OPERATIONAL_V22_ERROR_EXOGENOUS_NFEATURE ||
        throw(ArgumentError(
            "V2.2-M3 exogenous artifact must have one row per frozen feature",
        ))
    for row in 1:nrow(df), column in _OPERATIONAL_V22_ERROR_EXOGENOUS_CSV_COLUMNS
        ismissing(df[row, column]) && throw(ArgumentError(
            "V2.2-M3 exogenous artifact contains missing data",
        ))
    end
    string(_operational_v22_error_exogenous_consistent(df, :schema_version)) ==
        OPERATIONAL_V22_ERROR_EXOGENOUS_SCHEMA_VERSION || throw(ArgumentError(
            "unsupported V2.2-M3 exogenous artifact schema",
        ))
    string(_operational_v22_error_exogenous_consistent(df, :package_version)) ==
        OPERATIONAL_V22_ERROR_EXOGENOUS_PACKAGE_VERSION || throw(ArgumentError(
            "unsupported V2.2-M3 exogenous package version",
        ))
    checksum = string(_operational_v22_error_exogenous_consistent(
        df, :artifact_sha256,
    ))
    occursin(r"^[0-9a-f]{64}$", checksum) || throw(ArgumentError(
        "V2.2-M3 exogenous artifact checksum is malformed",
    ))
    for row in 1:nrow(df)
        _operational_v22_error_int(df[row, :feature_index], "feature_index") == row ||
            throw(ArgumentError(
                "V2.2-M3 exogenous feature indices are not sequential",
            ))
        Symbol(df[row, :feature_name]) ==
            OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURES[row] ||
            throw(ArgumentError(
                "V2.2-M3 exogenous feature order is invalid",
            ))
        expected_group = OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS[row]
        _operational_v22_error_int(df[row, :group_index], "group_index") ==
            expected_group || throw(ArgumentError(
                "V2.2-M3 exogenous feature-group map is invalid",
            ))
        Symbol(df[row, :group_name]) ==
            OPERATIONAL_V22_ERROR_EXOGENOUS_GROUPS[expected_group] ||
            throw(ArgumentError(
                "V2.2-M3 exogenous group order is invalid",
            ))
    end
    support = ntuple(
        group -> begin
            rows_in_group = findall(
                ==(group), OPERATIONAL_V22_ERROR_EXOGENOUS_FEATURE_GROUPS,
            )
            flags = df[rows_in_group, :group_selected]
            all(value -> value isa Bool, flags) || throw(ArgumentError(
                "V2.2-M3 exogenous selected flags must be Boolean",
            ))
            all(==(first(flags)), flags) || throw(ArgumentError(
                "V2.2-M3 exogenous selected flags are inconsistent within a group",
            ))
            first(flags)
        end,
        _OPERATIONAL_V22_ERROR_EXOGENOUS_NGROUP,
    )
    artifact = OperationalV22ErrorExogenousArtifact(
        string(_operational_v22_error_exogenous_consistent(
            df, :base_center_sha256,
        )),
        _operational_v22_error_int(
            _operational_v22_error_exogenous_consistent(
                df, :model_step_hours,
            ),
            "model_step_hours",
        ),
        _operational_v22_error_float(
            _operational_v22_error_exogenous_consistent(df, :intercept_nt),
            "exogenous artifact intercept",
        ),
        [_operational_v22_error_float(
            df[row, :coefficient], "exogenous artifact coefficient",
        ) for row in 1:nrow(df)];
        support_mask=support,
        feature_center=[_operational_v22_error_float(
            df[row, :feature_center], "exogenous artifact feature center",
        ) for row in 1:nrow(df)],
        feature_scale=[_operational_v22_error_float(
            df[row, :feature_scale], "exogenous artifact feature scale",
        ) for row in 1:nrow(df)],
        ridge=_operational_v22_error_float(
            _operational_v22_error_exogenous_consistent(df, :ridge),
            "exogenous artifact ridge",
        ),
        threshold=_operational_v22_error_float(
            _operational_v22_error_exogenous_consistent(df, :threshold),
            "exogenous artifact threshold",
        ),
        fit_rows=_operational_v22_error_int(
            _operational_v22_error_exogenous_consistent(df, :fit_rows),
            "fit_rows",
        ),
        threshold_iterations=_operational_v22_error_int(
            _operational_v22_error_exogenous_consistent(
                df, :threshold_iterations,
            ),
            "threshold_iterations",
        ),
        label=string(_operational_v22_error_exogenous_consistent(df, :label)),
    )
    stored_radius = _operational_v22_error_float(
        _operational_v22_error_exogenous_consistent(df, :spectral_radius),
        "exogenous artifact spectral radius",
    )
    _operational_v22_spectral_radius_agrees(stored_radius, artifact.spectral_radius) ||
        throw(ArgumentError(
            "V2.2-M3 exogenous spectral radius is inconsistent",
        ))
    stored_limit = _operational_v22_error_float(
        _operational_v22_error_exogenous_consistent(df, :stability_limit),
        "exogenous artifact stability limit",
    )
    stored_limit == artifact.stability_limit || throw(ArgumentError(
        "V2.2-M3 exogenous stability limit is inconsistent",
    ))
    operational_v22_error_exogenous_sha256(artifact) == checksum ||
        throw(ArgumentError(
            "V2.2-M3 exogenous artifact checksum mismatch",
        ))
    return artifact
end
