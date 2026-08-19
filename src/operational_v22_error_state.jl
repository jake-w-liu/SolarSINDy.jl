# Pure, receipt-causal scalar error-state dynamics for Operational V2.2-M3.

using CSV
using DataFrames
using Dates
using LinearAlgebra
import SHA

"Frozen issue offsets of the matured one-hour innovations used by V2.2-M3."
const OPERATIONAL_V22_ERROR_LAGS_H = (1, 2, 3, 4, 6, 9, 12, 18, 24)
const OPERATIONAL_V22_ERROR_SUPPORTED_MODEL_STEPS = OPERATIONAL_V22_MODEL_STEPS
const OPERATIONAL_V22_ERROR_SCHEMA_VERSION = "operational_v2_2_m3_error_v1"
const OPERATIONAL_V22_ERROR_PACKAGE_VERSION = "SolarSINDy-0.2.1"
const OPERATIONAL_V22_ERROR_MAX_SPECTRAL_RADIUS = 0.98
const _OPERATIONAL_V22_ERROR_BUFFER_H = maximum(OPERATIONAL_V22_ERROR_LAGS_H)
const _OPERATIONAL_V22_ERROR_FALLBACKS = (
    :none,
    :missing_history,
    :duplicate_issue_record,
    :observation_not_mature,
    :base_center_mismatch,
)

function _operational_v22_error_float(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "V2.2-M3 error-state $field must be a real number",
    ))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError(
        "V2.2-M3 error-state $field must be finite",
    ))
    return converted
end

function _operational_v22_error_sha(value, field::AbstractString)
    text = String(value)
    occursin(r"^[0-9a-f]{64}$", text) || throw(ArgumentError(
        "V2.2-M3 error-state $field must be lowercase SHA-256",
    ))
    return text
end

"An immutable one-hour forecast/observation pair, indexed by forecast issue."
struct OperationalV22H1Innovation
    issued_at::DateTime
    target_at::DateTime
    observation_available_at::DateTime
    base_center_sha256::String
    base_prediction_dst_nt::Float64
    observation_dst_nt::Float64

    function OperationalV22H1Innovation(
            issued_at::DateTime,
            target_at::DateTime,
            observation_available_at::DateTime,
            base_center_sha256::String,
            base_prediction_dst_nt::Float64,
            observation_dst_nt::Float64)
        target_at == issued_at + Hour(1) || throw(ArgumentError(
            "V2.2-M3 innovation target must be exactly one hour after issue",
        ))
        observation_available_at >= target_at || throw(ArgumentError(
            "V2.2-M3 innovation observation cannot be available before target time",
        ))
        return new(
            issued_at,
            target_at,
            observation_available_at,
            _operational_v22_error_sha(base_center_sha256, "base-center checksum"),
            _operational_v22_error_float(
                base_prediction_dst_nt, "base prediction",
            ),
            _operational_v22_error_float(observation_dst_nt, "observation"),
        )
    end
end

function OperationalV22H1Innovation(
        issued_at::DateTime,
        target_at::DateTime,
        observation_available_at::DateTime,
        base_center_sha256::AbstractString,
        base_prediction_dst_nt::Real,
        observation_dst_nt::Real)
    return OperationalV22H1Innovation(
        issued_at,
        target_at,
        observation_available_at,
        String(base_center_sha256),
        _operational_v22_error_float(base_prediction_dst_nt, "base prediction"),
        _operational_v22_error_float(observation_dst_nt, "observation"),
    )
end

operational_v22_h1_innovation(record::OperationalV22H1Innovation) =
    record.observation_dst_nt - record.base_prediction_dst_nt

"Causal innovation state at one issue, with an explicit fail-closed status."
struct OperationalV22ErrorHistory
    issue_time::DateTime
    base_center_sha256::String
    innovation_buffer_nt::NTuple{_OPERATIONAL_V22_ERROR_BUFFER_H,Float64}
    lagged_innovations_nt::NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Float64}
    ready::Bool
    fallback_reason::Symbol

    function OperationalV22ErrorHistory(
            issue_time::DateTime,
            base_center_sha256::String,
            innovation_buffer_nt::NTuple{_OPERATIONAL_V22_ERROR_BUFFER_H,Float64},
            lagged_innovations_nt::NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Float64},
            ready::Bool,
            fallback_reason::Symbol)
        fallback_reason in _OPERATIONAL_V22_ERROR_FALLBACKS || throw(ArgumentError(
            "unknown V2.2-M3 error-state fallback reason: $fallback_reason",
        ))
        ready == (fallback_reason == :none) || throw(ArgumentError(
            "V2.2-M3 error-state readiness and fallback reason disagree",
        ))
        all(isfinite, innovation_buffer_nt) && all(isfinite, lagged_innovations_nt) ||
            throw(ArgumentError("V2.2-M3 error-state history must be finite"))
        return new(
            issue_time,
            _operational_v22_error_sha(base_center_sha256, "base-center checksum"),
            innovation_buffer_nt,
            lagged_innovations_nt,
            ready,
            fallback_reason,
        )
    end
end

function _operational_v22_error_fallback(
        issue_time::DateTime,
        base_center_sha256::String,
        reason::Symbol)
    zeros24 = ntuple(_ -> 0.0, _OPERATIONAL_V22_ERROR_BUFFER_H)
    zeros9 = ntuple(_ -> 0.0, length(OPERATIONAL_V22_ERROR_LAGS_H))
    return OperationalV22ErrorHistory(
        issue_time, base_center_sha256, zeros24, zeros9, false, reason,
    )
end

"""
Classify the 24 h innovation buffer visible at `issue_time` from an index of records.

`immature_issues` names the issue hours a caller found inside the window but dropped because their
observation had not matured by `issue_time`. Without it every such hour is indistinguishable from an
hour that was never produced, and the fallback reports `:missing_history` for a record the caller is
holding — the two are different operational states and only one of them resolves by waiting.
"""
function _operational_v22_error_history_from_index(
        record_at::AbstractDict{DateTime,OperationalV22H1Innovation},
        duplicate_issues::AbstractSet{DateTime},
        issue_time::DateTime,
        center_hash::String;
        immature_issues::AbstractSet{DateTime} = Set{DateTime}())
    any(issue_time - Hour(offset) in duplicate_issues
        for offset in 1:_OPERATIONAL_V22_ERROR_BUFFER_H) &&
        return _operational_v22_error_fallback(
            issue_time, center_hash, :duplicate_issue_record,
        )
    values = Vector{Float64}(undef, _OPERATIONAL_V22_ERROR_BUFFER_H)
    for offset in 1:_OPERATIONAL_V22_ERROR_BUFFER_H
        issued_at = issue_time - Hour(offset)
        haskey(record_at, issued_at) || return _operational_v22_error_fallback(
            issue_time, center_hash,
            issued_at in immature_issues ? :observation_not_mature : :missing_history,
        )
        record = record_at[issued_at]
        record.base_center_sha256 == center_hash ||
            return _operational_v22_error_fallback(
                issue_time, center_hash, :base_center_mismatch,
            )
        record.observation_available_at <= issue_time ||
            return _operational_v22_error_fallback(
                issue_time, center_hash, :observation_not_mature,
            )
        values[offset] = operational_v22_h1_innovation(record)
    end
    buffer = Tuple(values)
    lagged = ntuple(
        index -> buffer[OPERATIONAL_V22_ERROR_LAGS_H[index]],
        length(OPERATIONAL_V22_ERROR_LAGS_H),
    )
    return OperationalV22ErrorHistory(
        issue_time, center_hash, buffer, lagged, true, :none,
    )
end

"Build the exact history visible at `issue_time`; records issued then or later are ignored."
function operational_v22_matured_h1_history(
        records::AbstractVector{<:OperationalV22H1Innovation},
        issue_time::DateTime,
        base_center_sha256::AbstractString)
    center_hash = _operational_v22_error_sha(
        base_center_sha256, "base-center checksum",
    )
    relevant = Dict{DateTime,OperationalV22H1Innovation}()
    duplicates = Set{DateTime}()
    immature = Set{DateTime}()
    earliest = issue_time - Hour(_OPERATIONAL_V22_ERROR_BUFFER_H)
    for record in records
        earliest <= record.issued_at < issue_time || continue
        # An in-window record whose observation has not matured is remembered rather than silently
        # dropped, so the fallback can say which of the two states the buffer is in.
        if record.observation_available_at > issue_time
            push!(immature, record.issued_at)
            continue
        end
        if haskey(relevant, record.issued_at)
            push!(duplicates, record.issued_at)
        else
            relevant[record.issued_at] = record
        end
    end
    return _operational_v22_error_history_from_index(
        relevant, duplicates, issue_time, center_hash; immature_issues = immature,
    )
end

function _operational_v22_error_spectral_radius(
        coefficients::NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Float64})
    companion = zeros(Float64, _OPERATIONAL_V22_ERROR_BUFFER_H,
                      _OPERATIONAL_V22_ERROR_BUFFER_H)
    for (index, lag) in enumerate(OPERATIONAL_V22_ERROR_LAGS_H)
        companion[1, lag] = coefficients[index]
    end
    for row in 2:_OPERATIONAL_V22_ERROR_BUFFER_H
        companion[row, row - 1] = 1.0
    end
    return maximum(abs, eigvals(companion))
end

"Immutable sparse delay-recurrence artifact bound to one M2-plus-core center."
struct OperationalV22ErrorStateArtifact
    label::String
    base_center_sha256::String
    issue_lags_hours::NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Int}
    support_mask::NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Bool}
    intercept_nt::Float64
    coefficients::NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Float64}
    spectral_radius::Float64
    stability_limit::Float64
    ridge::Float64
    fit_rows::Int
    selection_score::Float64

    function OperationalV22ErrorStateArtifact(
            label::String,
            base_center_sha256::String,
            issue_lags_hours::NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Int},
            support_mask::NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Bool},
            intercept_nt::Float64,
            coefficients::NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Float64},
            stability_limit::Float64,
            ridge::Float64,
            fit_rows::Int,
            selection_score::Float64,
            ::Val{:validated})
        isempty(strip(label)) && throw(ArgumentError(
            "V2.2-M3 error-state artifact label must not be empty",
        ))
        issue_lags_hours == OPERATIONAL_V22_ERROR_LAGS_H || throw(ArgumentError(
            "V2.2-M3 error-state lag schema is not frozen",
        ))
        all(index -> support_mask[index] || coefficients[index] == 0.0,
            eachindex(coefficients)) || throw(ArgumentError(
                "V2.2-M3 excluded delay terms must have exact zero coefficients",
            ))
        isfinite(intercept_nt) && all(isfinite, coefficients) || throw(ArgumentError(
            "V2.2-M3 error-state coefficients must be finite",
        ))
        stability_limit == OPERATIONAL_V22_ERROR_MAX_SPECTRAL_RADIUS ||
            throw(ArgumentError("V2.2-M3 stability limit is not frozen"))
        radius = _operational_v22_error_spectral_radius(coefficients)
        isfinite(radius) && radius <= stability_limit || throw(ArgumentError(
            "V2.2-M3 error-state recurrence is unstable: spectral radius=$radius",
        ))
        isfinite(ridge) && ridge >= 0.0 || throw(ArgumentError(
            "V2.2-M3 error-state ridge must be finite and nonnegative",
        ))
        fit_rows >= 2 || throw(ArgumentError(
            "V2.2-M3 error-state artifact requires at least two fit rows",
        ))
        isfinite(selection_score) || throw(ArgumentError(
            "V2.2-M3 error-state selection score must be finite",
        ))
        return new(
            label,
            _operational_v22_error_sha(base_center_sha256, "base-center checksum"),
            issue_lags_hours,
            support_mask,
            intercept_nt,
            coefficients,
            radius,
            stability_limit,
            ridge,
            fit_rows,
            selection_score,
        )
    end
end

function OperationalV22ErrorStateArtifact(
        base_center_sha256::AbstractString,
        intercept_nt::Real,
        coefficients::AbstractVector{<:Real};
        support_mask::NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Bool}=
            ntuple(_ -> true, length(OPERATIONAL_V22_ERROR_LAGS_H)),
        ridge::Real=0.0,
        fit_rows::Integer,
        selection_score::Real=0.0,
        label::AbstractString="operational_v2_2_m3_error")
    length(coefficients) == length(OPERATIONAL_V22_ERROR_LAGS_H) ||
        throw(DimensionMismatch(
            "V2.2-M3 error-state artifact requires nine delay coefficients",
        ))
    converted = ntuple(
        index -> _operational_v22_error_float(
            coefficients[index], "coefficient at lag " *
            string(OPERATIONAL_V22_ERROR_LAGS_H[index]),
        ),
        length(OPERATIONAL_V22_ERROR_LAGS_H),
    )
    return OperationalV22ErrorStateArtifact(
        String(label),
        String(base_center_sha256),
        OPERATIONAL_V22_ERROR_LAGS_H,
        support_mask,
        _operational_v22_error_float(intercept_nt, "intercept"),
        converted,
        OPERATIONAL_V22_ERROR_MAX_SPECTRAL_RADIUS,
        _operational_v22_error_float(ridge, "ridge"),
        Int(fit_rows),
        _operational_v22_error_float(selection_score, "selection score"),
        Val(:validated),
    )
end

function _operational_v22_error_fit_coefficients(
        features::Matrix{Float64},
        target::Vector{Float64},
        selected::Vector{Int},
        ridge::Float64)
    design = ones(Float64, size(features, 1), length(selected) + 1)
    isempty(selected) || (design[:, 2:end] .= features[:, selected])
    if iszero(ridge)
        rank(design) == size(design, 2) || return nothing
        coefficients = design \ target
    else
        gram = design' * design
        for index in 2:size(gram, 1)
            gram[index, index] += ridge
        end
        coefficients = cholesky(Symmetric(gram); check=true) \ (design' * target)
    end
    all(isfinite, coefficients) || return nothing
    return coefficients, design
end

"Fit a deterministic BIC-selected sparse stable delay recurrence from matured records."
function fit_operational_v22_error_state(
        records::AbstractVector{<:OperationalV22H1Innovation};
        base_center_sha256::AbstractString,
        fit_as_of::DateTime,
        ridge::Real=1.0e-6,
        max_terms::Integer=3,
        minimum_rows::Integer=32,
        label::AbstractString="operational_v2_2_m3_error")
    center_hash = _operational_v22_error_sha(
        base_center_sha256, "base-center checksum",
    )
    ridge_value = _operational_v22_error_float(ridge, "ridge")
    ridge_value >= 0.0 || throw(ArgumentError(
        "V2.2-M3 error-state ridge must be nonnegative",
    ))
    0 <= max_terms <= length(OPERATIONAL_V22_ERROR_LAGS_H) ||
        throw(ArgumentError("V2.2-M3 max_terms is outside the lag schema"))
    minimum_rows >= 2 || throw(ArgumentError(
        "V2.2-M3 minimum_rows must be at least two",
    ))

    eligible = [
        record for record in records
        if record.issued_at < fit_as_of &&
           record.observation_available_at <= fit_as_of &&
           record.base_center_sha256 == center_hash
    ]
    length(unique(record.issued_at for record in eligible)) == length(eligible) ||
        throw(ArgumentError("V2.2-M3 fitting records contain duplicate issue times"))
    sort!(eligible; by=record -> record.issued_at)
    record_at = Dict(record.issued_at => record for record in eligible)
    duplicate_issues = Set{DateTime}()

    feature_rows = NTuple{length(OPERATIONAL_V22_ERROR_LAGS_H),Float64}[]
    target = Float64[]
    for record in eligible
        history = _operational_v22_error_history_from_index(
            record_at, duplicate_issues, record.issued_at, center_hash,
        )
        history.ready || continue
        push!(feature_rows, history.lagged_innovations_nt)
        push!(target, operational_v22_h1_innovation(record))
    end
    length(target) >= minimum_rows || throw(ArgumentError(
        "V2.2-M3 fitting has $(length(target)) complete rows; " *
        "minimum_rows=$minimum_rows",
    ))
    features = reduce(vcat, permutedims(collect(row)) for row in feature_rows)
    all(isfinite, features) && all(isfinite, target) || throw(ArgumentError(
        "V2.2-M3 fitting inputs must be finite",
    ))

    best = nothing
    n_lags = length(OPERATIONAL_V22_ERROR_LAGS_H)
    for bits in 0:(2^n_lags - 1)
        count_ones(bits) <= max_terms || continue
        selected = [index for index in 1:n_lags if !iszero(bits & (1 << (index - 1)))]
        fitted = _operational_v22_error_fit_coefficients(
            features, target, selected, ridge_value,
        )
        fitted === nothing && continue
        reduced_coefficients, design = fitted
        slopes = zeros(Float64, n_lags)
        isempty(selected) || (slopes[selected] .= reduced_coefficients[2:end])
        coefficient_tuple = Tuple(slopes)
        radius = _operational_v22_error_spectral_radius(coefficient_tuple)
        radius <= OPERATIONAL_V22_ERROR_MAX_SPECTRAL_RADIUS || continue
        residual = target - design * reduced_coefficients
        rss = sum(abs2, residual)
        rss_floor = max(floatmin(Float64), eps(Float64) * max(sum(abs2, target), 1.0))
        parameter_count = length(selected) + 1
        score = length(target) * log(max(rss, rss_floor) / length(target)) +
                parameter_count * log(length(target))
        candidate = (
            bits=bits,
            support=ntuple(index -> index in selected, n_lags),
            intercept=reduced_coefficients[1],
            coefficients=coefficient_tuple,
            score=score,
            terms=length(selected),
        )
        if best === nothing ||
           candidate.score < best.score - 1.0e-12 ||
           (isapprox(candidate.score, best.score; atol=1.0e-12, rtol=0.0) &&
            (candidate.terms < best.terms ||
             (candidate.terms == best.terms && candidate.bits < best.bits)))
            best = candidate
        end
    end
    best === nothing && throw(ErrorException(
        "V2.2-M3 sparse search found no stable full-rank candidate",
    ))
    return OperationalV22ErrorStateArtifact(
        center_hash,
        best.intercept,
        collect(best.coefficients);
        support_mask=best.support,
        ridge=ridge_value,
        fit_rows=length(target),
        selection_score=best.score,
        label=label,
    )
end

_operational_v22_error_cap(model_step_hours::Integer) =
    operational_v22_correction_cap_nt(model_step_hours)

"Predict a bounded correction; incomplete history returns the center unchanged."
function operational_v22_error_state_predict(
        artifact::OperationalV22ErrorStateArtifact,
        issue_time::DateTime,
        model_step_hours::Integer,
        base_center_sha256::AbstractString,
        base_prediction_dst_nt::Real,
        records::AbstractVector{<:OperationalV22H1Innovation})
    model_step_hours isa Bool && throw(ArgumentError(
        "V2.2-M3 model_step_hours must be an integer",
    ))
    lead = Int(model_step_hours)
    lead in OPERATIONAL_V22_ERROR_SUPPORTED_MODEL_STEPS || throw(ArgumentError(
        "unsupported V2.2-M3 model_step_hours=$lead",
    ))
    center_hash = _operational_v22_error_sha(
        base_center_sha256, "current base-center checksum",
    )
    center_hash == artifact.base_center_sha256 || throw(ArgumentError(
        "V2.2-M3 current base-center checksum does not match the artifact",
    ))
    base = _operational_v22_error_float(base_prediction_dst_nt, "base prediction")
    cap = _operational_v22_error_cap(lead)
    history = operational_v22_matured_h1_history(
        records, issue_time, artifact.base_center_sha256,
    )
    checksum = operational_v22_error_state_sha256(artifact)
    if !history.ready
        return (
            pred_dst_nt=base,
            raw_correction_nt=0.0,
            correction_nt=0.0,
            correction_cap_nt=cap,
            correction_was_capped=false,
            correction_applied=false,
            fallback_reason=history.fallback_reason,
            artifact_sha256=checksum,
        )
    end

    state = Dict{DateTime,Float64}(
        issue_time - Hour(offset) => history.innovation_buffer_nt[offset]
        for offset in 1:_OPERATIONAL_V22_ERROR_BUFFER_H
    )
    raw = 0.0
    for step in 0:(lead - 1)
        forecast_issue = issue_time + Hour(step)
        raw = artifact.intercept_nt
        for (index, lag) in enumerate(artifact.issue_lags_hours)
            artifact.support_mask[index] || continue
            raw += artifact.coefficients[index] * state[forecast_issue - Hour(lag)]
        end
        isfinite(raw) || throw(ErrorException(
            "V2.2-M3 error-state rollout became non-finite",
        ))
        state[forecast_issue] = raw
    end
    correction = clamp(raw, -cap, cap)
    predicted = base + correction
    isfinite(predicted) || throw(ErrorException(
        "V2.2-M3 corrected forecast became non-finite",
    ))
    return (
        pred_dst_nt=predicted,
        raw_correction_nt=raw,
        correction_nt=correction,
        correction_cap_nt=cap,
        correction_was_capped=correction != raw,
        correction_applied=true,
        fallback_reason=:none,
        artifact_sha256=checksum,
    )
end

function _operational_v22_error_hash_token(io::IO, value)
    text = value isa Float64 ? bitstring(value) : string(value)
    kind = string(typeof(value))
    print(io, ncodeunits(kind), ':', kind, ':', ncodeunits(text), ':', text, '|')
    return nothing
end

"Return the byte-significant SHA-256 identity of an error-state artifact."
function operational_v22_error_state_sha256(
        artifact::OperationalV22ErrorStateArtifact)
    io = IOBuffer()
    for value in (
            OPERATIONAL_V22_ERROR_SCHEMA_VERSION,
            OPERATIONAL_V22_ERROR_PACKAGE_VERSION,
            artifact.label,
            artifact.base_center_sha256,
            artifact.intercept_nt,
            _operational_v22_hashable_spectral_radius(artifact.spectral_radius),
            artifact.stability_limit,
            artifact.ridge,
            artifact.fit_rows,
            artifact.selection_score,
        )
        _operational_v22_error_hash_token(io, value)
    end
    for values in (
            artifact.issue_lags_hours,
            artifact.support_mask,
            artifact.coefficients,
        )
        _operational_v22_error_hash_token(io, length(values))
        for value in values
            _operational_v22_error_hash_token(io, value)
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

const _OPERATIONAL_V22_ERROR_CSV_COLUMNS = (
    :schema_version,
    :package_version,
    :artifact_sha256,
    :label,
    :base_center_sha256,
    :intercept_nt,
    :spectral_radius,
    :stability_limit,
    :ridge,
    :fit_rows,
    :selection_score,
    :lag_index,
    :lag_hours,
    :selected,
    :coefficient,
)

function _operational_v22_error_atomic_csv(path::String, rows)
    target = abspath(path)
    mkpath(dirname(target))
    _write_selection_csv(target, rows)
    return path
end

"Atomically write a strictly versioned, checksummed error-state artifact."
function write_operational_v22_error_state(
        path::AbstractString,
        artifact::OperationalV22ErrorStateArtifact)
    checksum = operational_v22_error_state_sha256(artifact)
    rows = [(
        schema_version=OPERATIONAL_V22_ERROR_SCHEMA_VERSION,
        package_version=OPERATIONAL_V22_ERROR_PACKAGE_VERSION,
        artifact_sha256=checksum,
        label=artifact.label,
        base_center_sha256=artifact.base_center_sha256,
        intercept_nt=artifact.intercept_nt,
        spectral_radius=artifact.spectral_radius,
        stability_limit=artifact.stability_limit,
        ridge=artifact.ridge,
        fit_rows=artifact.fit_rows,
        selection_score=artifact.selection_score,
        lag_index=index,
        lag_hours=artifact.issue_lags_hours[index],
        selected=artifact.support_mask[index],
        coefficient=artifact.coefficients[index],
    ) for index in eachindex(artifact.issue_lags_hours)]
    return _operational_v22_error_atomic_csv(String(path), rows)
end

function _operational_v22_error_consistent(df::DataFrame, column::Symbol)
    values = df[!, column]
    any(ismissing, values) && throw(ArgumentError(
        "V2.2-M3 error-state artifact $column contains missing values",
    ))
    value = first(values)
    all(isequal(value), values) || throw(ArgumentError(
        "V2.2-M3 error-state artifact $column is inconsistent",
    ))
    return value
end

function _operational_v22_error_int(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "V2.2-M3 error-state artifact $field must be an integer",
    ))
    numeric = Float64(value)
    isfinite(numeric) && isinteger(numeric) &&
        typemin(Int) <= numeric <= typemax(Int) || throw(ArgumentError(
            "V2.2-M3 error-state artifact $field must be an integer",
        ))
    return Int(numeric)
end

"Read and fully validate a checksummed error-state artifact."
function read_operational_v22_error_state(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "V2.2-M3 error-state artifact must be a regular non-symlink file: $source",
    ))
    df = CSV.read(source, DataFrame)
    names(df) == collect(String.(_OPERATIONAL_V22_ERROR_CSV_COLUMNS)) ||
        throw(ArgumentError("V2.2-M3 error-state artifact CSV schema is invalid"))
    nrow(df) == length(OPERATIONAL_V22_ERROR_LAGS_H) || throw(ArgumentError(
        "V2.2-M3 error-state artifact must have one row per frozen lag",
    ))
    for row in 1:nrow(df), column in _OPERATIONAL_V22_ERROR_CSV_COLUMNS
        ismissing(df[row, column]) && throw(ArgumentError(
            "V2.2-M3 error-state artifact contains missing data",
        ))
    end
    string(_operational_v22_error_consistent(df, :schema_version)) ==
        OPERATIONAL_V22_ERROR_SCHEMA_VERSION || throw(ArgumentError(
            "unsupported V2.2-M3 error-state artifact schema",
        ))
    string(_operational_v22_error_consistent(df, :package_version)) ==
        OPERATIONAL_V22_ERROR_PACKAGE_VERSION || throw(ArgumentError(
            "unsupported V2.2-M3 error-state package version",
        ))
    checksum = string(_operational_v22_error_consistent(df, :artifact_sha256))
    occursin(r"^[0-9a-f]{64}$", checksum) || throw(ArgumentError(
        "V2.2-M3 error-state artifact checksum is malformed",
    ))
    [_operational_v22_error_int(df[row, :lag_index], "lag_index")
     for row in 1:nrow(df)] == collect(eachindex(OPERATIONAL_V22_ERROR_LAGS_H)) ||
        throw(ArgumentError("V2.2-M3 error-state lag indices are invalid"))
    Tuple(_operational_v22_error_int(df[row, :lag_hours], "lag_hours")
          for row in 1:nrow(df)) == OPERATIONAL_V22_ERROR_LAGS_H ||
        throw(ArgumentError("V2.2-M3 error-state lag order is invalid"))
    support = Tuple(begin
        value = df[row, :selected]
        value isa Bool || throw(ArgumentError(
            "V2.2-M3 error-state selected flags must be Boolean",
        ))
        value
    end for row in 1:nrow(df))
    artifact = OperationalV22ErrorStateArtifact(
        string(_operational_v22_error_consistent(df, :base_center_sha256)),
        _operational_v22_error_float(
            _operational_v22_error_consistent(df, :intercept_nt),
            "artifact intercept",
        ),
        [_operational_v22_error_float(df[row, :coefficient], "artifact coefficient")
         for row in 1:nrow(df)];
        support_mask=support,
        ridge=_operational_v22_error_float(
            _operational_v22_error_consistent(df, :ridge), "artifact ridge",
        ),
        fit_rows=_operational_v22_error_int(
            _operational_v22_error_consistent(df, :fit_rows), "fit_rows",
        ),
        selection_score=_operational_v22_error_float(
            _operational_v22_error_consistent(df, :selection_score),
            "selection_score",
        ),
        label=string(_operational_v22_error_consistent(df, :label)),
    )
    stored_radius = _operational_v22_error_float(
        _operational_v22_error_consistent(df, :spectral_radius),
        "artifact spectral radius",
    )
    _operational_v22_spectral_radius_agrees(stored_radius, artifact.spectral_radius) ||
        throw(ArgumentError(
            "V2.2-M3 error-state spectral radius is inconsistent",
        ))
    stored_limit = _operational_v22_error_float(
        _operational_v22_error_consistent(df, :stability_limit),
        "artifact stability limit",
    )
    stored_limit == artifact.stability_limit || throw(ArgumentError(
        "V2.2-M3 error-state stability limit is inconsistent",
    ))
    operational_v22_error_state_sha256(artifact) == checksum || throw(ArgumentError(
        "V2.2-M3 error-state artifact checksum mismatch",
    ))
    return artifact
end
