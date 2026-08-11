# Stable group-sparse continuation of the frozen V2.2-M2 five-state driver.

import SHA

"Frozen V2.2-M2 state order: magnetic field followed by log speed and density."
const OPERATIONAL_V22_DRIVER_STATES = (:Bx, :By, :Bz, :logV, :logn)

"Frozen V2.2-M2 delay coordinates in 30-minute samples."
const OPERATIONAL_V22_DRIVER_LAGS = (0, 1, 2, 6, 12, 24)

const OPERATIONAL_V22_DRIVER_CADENCE_MINUTES = 30
const OPERATIONAL_V22_DRIVER_ROLLOUT_STEPS = 14
const OPERATIONAL_V22_DRIVER_STABILITY_TOLERANCE = 1.0e-8
const OPERATIONAL_V22_DRIVER_RIDGE_GRID = Tuple(10.0 .^ (-6:2))
const OPERATIONAL_V22_DRIVER_THRESHOLD_GRID =
    (0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1)
const OPERATIONAL_V22_DRIVER_MAX_THRESHOLD_ITERATIONS = 20
const OPERATIONAL_V22_DRIVER_SCHEMA_VERSION = "operational_v2_2_m2_driver_v1"
const OPERATIONAL_V22_DRIVER_PACKAGE_VERSION = "SolarSINDy-0.2.1"

const _OPERATIONAL_V22_DRIVER_NSTATE = length(OPERATIONAL_V22_DRIVER_STATES)
const _OPERATIONAL_V22_DRIVER_NLAG = length(OPERATIONAL_V22_DRIVER_LAGS)
const _OPERATIONAL_V22_DRIVER_HISTORY_ROWS = maximum(OPERATIONAL_V22_DRIVER_LAGS) + 1

function _operational_v22_driver_float(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "V2.2-M2 $field must be a real number",
    ))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError(
        "V2.2-M2 $field must be finite",
    ))
    return converted
end

function _operational_v22_driver_coefficients(values)
    size(values) == (
        _OPERATIONAL_V22_DRIVER_NSTATE,
        _OPERATIONAL_V22_DRIVER_NSTATE,
        _OPERATIONAL_V22_DRIVER_NLAG,
    ) || throw(DimensionMismatch(
        "V2.2-M2 coefficients must have size (5, 5, 6)",
    ))
    return ntuple(
        lag_index -> ntuple(
            predictor -> ntuple(
                output -> _operational_v22_driver_float(
                    values[output, predictor, lag_index],
                    "coefficient",
                ),
                _OPERATIONAL_V22_DRIVER_NSTATE,
            ),
            _OPERATIONAL_V22_DRIVER_NSTATE,
        ),
        _OPERATIONAL_V22_DRIVER_NLAG,
    )
end

function _operational_v22_driver_support(values)
    size(values) == (
        _OPERATIONAL_V22_DRIVER_NSTATE,
        _OPERATIONAL_V22_DRIVER_NLAG,
    ) || throw(DimensionMismatch(
        "V2.2-M2 support must have size (5, 6)",
    ))
    all(value -> value isa Bool, values) || throw(ArgumentError(
        "V2.2-M2 support entries must be Boolean",
    ))
    return ntuple(
        lag_index -> ntuple(
            predictor -> values[predictor, lag_index],
            _OPERATIONAL_V22_DRIVER_NSTATE,
        ),
        _OPERATIONAL_V22_DRIVER_NLAG,
    )
end

function _operational_v22_driver_coefficient_array(coefficients)
    values = Array{Float64}(undef,
        _OPERATIONAL_V22_DRIVER_NSTATE,
        _OPERATIONAL_V22_DRIVER_NSTATE,
        _OPERATIONAL_V22_DRIVER_NLAG,
    )
    for lag_index in 1:_OPERATIONAL_V22_DRIVER_NLAG,
            predictor in 1:_OPERATIONAL_V22_DRIVER_NSTATE,
            output in 1:_OPERATIONAL_V22_DRIVER_NSTATE
        values[output, predictor, lag_index] =
            coefficients[lag_index][predictor][output]
    end
    return values
end

function _operational_v22_driver_support_array(support)
    values = Matrix{Bool}(undef,
        _OPERATIONAL_V22_DRIVER_NSTATE,
        _OPERATIONAL_V22_DRIVER_NLAG,
    )
    for lag_index in 1:_OPERATIONAL_V22_DRIVER_NLAG,
            predictor in 1:_OPERATIONAL_V22_DRIVER_NSTATE
        values[predictor, lag_index] = support[lag_index][predictor]
    end
    return values
end

function _operational_v22_driver_companion(coefficients)
    coefficient_array = _operational_v22_driver_coefficient_array(coefficients)
    dimension = _OPERATIONAL_V22_DRIVER_NSTATE * _OPERATIONAL_V22_DRIVER_HISTORY_ROWS
    companion = zeros(Float64, dimension, dimension)
    rows = 1:_OPERATIONAL_V22_DRIVER_NSTATE
    for (lag_index, lag) in pairs(OPERATIONAL_V22_DRIVER_LAGS)
        columns = ((lag * _OPERATIONAL_V22_DRIVER_NSTATE + 1):
                   ((lag + 1) * _OPERATIONAL_V22_DRIVER_NSTATE))
        companion[rows, columns] .= @view coefficient_array[:, :, lag_index]
    end
    for block in 2:_OPERATIONAL_V22_DRIVER_HISTORY_ROWS
        destination = (((block - 1) * _OPERATIONAL_V22_DRIVER_NSTATE + 1):
                       (block * _OPERATIONAL_V22_DRIVER_NSTATE))
        source = (((block - 2) * _OPERATIONAL_V22_DRIVER_NSTATE + 1):
                  ((block - 1) * _OPERATIONAL_V22_DRIVER_NSTATE))
        companion[destination, source] .=
            Matrix{Float64}(I, _OPERATIONAL_V22_DRIVER_NSTATE,
                            _OPERATIONAL_V22_DRIVER_NSTATE)
    end
    return companion
end

function _operational_v22_driver_spectral_radius(coefficients)
    eigenvalues = eigvals(_operational_v22_driver_companion(coefficients))
    radius = maximum(abs, eigenvalues)
    isfinite(radius) || throw(ArgumentError(
        "V2.2-M2 companion spectral radius is non-finite",
    ))
    return radius
end

"Immutable, checksummed V2.2-M2 group-sparse driver-continuation artifact."
struct OperationalV22DriverArtifact
    label::String
    state_names::NTuple{5,Symbol}
    lags::NTuple{6,Int}
    center::NTuple{5,Float64}
    scale::NTuple{5,Float64}
    intercept::NTuple{5,Float64}
    coefficients::NTuple{6,NTuple{5,NTuple{5,Float64}}}
    support_mask::NTuple{6,NTuple{5,Bool}}
    ridge::Float64
    threshold::Float64
    fit_rows::Int
    threshold_iterations::Int
    spectral_radius::Float64

    function OperationalV22DriverArtifact(
            label::String,
            state_names::NTuple{5,Symbol},
            lags::NTuple{6,Int},
            center::NTuple{5,Float64},
            scale::NTuple{5,Float64},
            intercept::NTuple{5,Float64},
            coefficients::NTuple{6,NTuple{5,NTuple{5,Float64}}},
            support_mask::NTuple{6,NTuple{5,Bool}},
            ridge::Float64,
            threshold::Float64,
            fit_rows::Int,
            threshold_iterations::Int,
            ::Val{:validated})
        isempty(strip(label)) && throw(ArgumentError(
            "V2.2-M2 artifact label must not be empty",
        ))
        state_names == OPERATIONAL_V22_DRIVER_STATES || throw(ArgumentError(
            "V2.2-M2 state order must be exactly (Bx, By, Bz, logV, logn)",
        ))
        lags == OPERATIONAL_V22_DRIVER_LAGS || throw(ArgumentError(
            "V2.2-M2 lag order must be exactly (0, 1, 2, 6, 12, 24)",
        ))
        all(isfinite, center) && all(isfinite, scale) && all(isfinite, intercept) ||
            throw(ArgumentError("V2.2-M2 normalization values must be finite"))
        all(>(0.0), scale) || throw(ArgumentError(
            "V2.2-M2 state scales must be positive",
        ))
        ridge in OPERATIONAL_V22_DRIVER_RIDGE_GRID || throw(ArgumentError(
            "V2.2-M2 ridge must belong to the frozen search grid",
        ))
        threshold in OPERATIONAL_V22_DRIVER_THRESHOLD_GRID || throw(ArgumentError(
            "V2.2-M2 threshold must belong to the frozen search grid",
        ))
        fit_rows >= 1 || throw(ArgumentError(
            "V2.2-M2 artifact requires at least one fit row",
        ))
        1 <= threshold_iterations <= OPERATIONAL_V22_DRIVER_MAX_THRESHOLD_ITERATIONS ||
            throw(ArgumentError(
                "V2.2-M2 threshold iterations must lie in 1:20",
            ))
        for lag_index in 1:_OPERATIONAL_V22_DRIVER_NLAG,
                predictor in 1:_OPERATIONAL_V22_DRIVER_NSTATE
            group = coefficients[lag_index][predictor]
            all(isfinite, group) || throw(ArgumentError(
                "V2.2-M2 coefficients must be finite",
            ))
            if support_mask[lag_index][predictor]
                norm(group) >= threshold || throw(ArgumentError(
                    "V2.2-M2 selected groups must meet the stored threshold",
                ))
            else
                all(iszero, group) || throw(ArgumentError(
                    "V2.2-M2 excluded groups must have exact zero coefficients",
                ))
            end
        end
        radius = _operational_v22_driver_spectral_radius(coefficients)
        radius <= 1.0 + OPERATIONAL_V22_DRIVER_STABILITY_TOLERANCE ||
            throw(ArgumentError(
                "V2.2-M2 candidate is unstable (spectral radius=$radius)",
            ))
        return new(
            label, state_names, lags, center, scale, intercept, coefficients,
            support_mask, ridge, threshold, fit_rows, threshold_iterations,
            radius,
        )
    end
end

"Construct and validate a stable V2.2-M2 artifact from a `(5, 5, 6)` map."
function OperationalV22DriverArtifact(
        coefficients::AbstractArray{<:Real,3};
        center=zeros(_OPERATIONAL_V22_DRIVER_NSTATE),
        scale=ones(_OPERATIONAL_V22_DRIVER_NSTATE),
        intercept=zeros(_OPERATIONAL_V22_DRIVER_NSTATE),
        support_mask=dropdims(any(!iszero, coefficients; dims=1); dims=1),
        ridge::Real=first(OPERATIONAL_V22_DRIVER_RIDGE_GRID),
        threshold::Real=first(OPERATIONAL_V22_DRIVER_THRESHOLD_GRID),
        fit_rows::Integer=1,
        threshold_iterations::Integer=1,
        label::AbstractString="operational-v2.2-m2-driver")
    length(center) == _OPERATIONAL_V22_DRIVER_NSTATE || throw(DimensionMismatch(
        "V2.2-M2 center must contain exactly five values",
    ))
    length(scale) == _OPERATIONAL_V22_DRIVER_NSTATE || throw(DimensionMismatch(
        "V2.2-M2 scale must contain exactly five values",
    ))
    length(intercept) == _OPERATIONAL_V22_DRIVER_NSTATE || throw(DimensionMismatch(
        "V2.2-M2 intercept must contain exactly five values",
    ))
    converted_center = ntuple(
        index -> _operational_v22_driver_float(center[index], "center"),
        _OPERATIONAL_V22_DRIVER_NSTATE,
    )
    converted_scale = ntuple(
        index -> _operational_v22_driver_float(scale[index], "scale"),
        _OPERATIONAL_V22_DRIVER_NSTATE,
    )
    converted_intercept = ntuple(
        index -> _operational_v22_driver_float(intercept[index], "intercept"),
        _OPERATIONAL_V22_DRIVER_NSTATE,
    )
    return OperationalV22DriverArtifact(
        String(label), OPERATIONAL_V22_DRIVER_STATES, OPERATIONAL_V22_DRIVER_LAGS,
        converted_center, converted_scale, converted_intercept,
        _operational_v22_driver_coefficients(coefficients),
        _operational_v22_driver_support(support_mask),
        _operational_v22_driver_float(ridge, "ridge"),
        _operational_v22_driver_float(threshold, "threshold"),
        Int(fit_rows), Int(threshold_iterations), Val(:validated),
    )
end

"Return a mutable `(output, predictor, lag)` copy of the frozen coefficients."
operational_v22_driver_coefficients(artifact::OperationalV22DriverArtifact) =
    _operational_v22_driver_coefficient_array(artifact.coefficients)

"Return a mutable `(predictor, lag)` copy of the selected group support."
operational_v22_driver_support(artifact::OperationalV22DriverArtifact) =
    _operational_v22_driver_support_array(artifact.support_mask)

"Construct the 125-by-125 companion matrix of a V2.2-M2 artifact."
operational_v22_driver_companion(artifact::OperationalV22DriverArtifact) =
    _operational_v22_driver_companion(artifact.coefficients)

"Return the verified companion spectral radius of a V2.2-M2 artifact."
operational_v22_driver_spectral_radius(artifact::OperationalV22DriverArtifact) =
    artifact.spectral_radius

function _operational_v22_driver_matrix(states::AbstractMatrix)
    size(states, 2) == _OPERATIONAL_V22_DRIVER_NSTATE || throw(DimensionMismatch(
        "V2.2-M2 state matrix must have exactly five columns",
    ))
    all(value -> value isa Real && !(value isa Bool), states) || throw(ArgumentError(
        "V2.2-M2 states must be real numbers",
    ))
    converted = Matrix{Float64}(states)
    all(isfinite, converted) || throw(ArgumentError(
        "V2.2-M2 states must be finite",
    ))
    return converted
end

function _operational_v22_driver_ridge_fit(
        design::Matrix{Float64},
        targets::Matrix{Float64},
        active::BitVector,
        ridge::Float64)
    coefficients = zeros(Float64, size(design, 2), size(targets, 2))
    target_mean = vec(mean(targets; dims=1))
    selected = findall(active)
    isempty(selected) && return target_mean, coefficients

    selected_design = design[:, selected]
    predictor_mean = vec(mean(selected_design; dims=1))
    centered_design = selected_design .- transpose(predictor_mean)
    centered_targets = targets .- transpose(target_mean)
    gram = transpose(centered_design) * centered_design
    gram[diagind(gram)] .+= ridge
    fitted = gram \ (transpose(centered_design) * centered_targets)
    all(isfinite, fitted) || throw(ArgumentError(
        "V2.2-M2 ridge coefficients are non-finite",
    ))
    coefficients[selected, :] .= fitted
    intercept = target_mean - transpose(fitted) * predictor_mean
    all(isfinite, intercept) || throw(ArgumentError(
        "V2.2-M2 ridge intercept is non-finite",
    ))
    return intercept, coefficients
end

function _operational_v22_driver_threshold_fit(
        design::Matrix{Float64},
        targets::Matrix{Float64},
        ridge::Float64,
        threshold::Float64;
        max_iterations::Int=OPERATIONAL_V22_DRIVER_MAX_THRESHOLD_ITERATIONS)
    max_iterations >= 1 || throw(ArgumentError(
        "V2.2-M2 threshold fitting requires at least one iteration",
    ))
    active = trues(size(design, 2))
    intercept = zeros(Float64, size(targets, 2))
    fitted = zeros(Float64, size(design, 2), size(targets, 2))
    iterations = 0
    support_changed = false
    for iteration in 1:max_iterations
        iterations = iteration
        intercept, fitted = _operational_v22_driver_ridge_fit(
            design, targets, active, ridge,
        )
        next_active = copy(active)
        for group in eachindex(active)
            active[group] || continue
            next_active[group] = norm(@view fitted[group, :]) >= threshold
        end
        support_changed = next_active != active
        active = next_active
        support_changed || break
    end
    if support_changed
        intercept, fitted = _operational_v22_driver_ridge_fit(
            design, targets, active, ridge,
        )
        all(group -> !active[group] ||
                     norm(@view(fitted[group, :])) >= threshold,
            eachindex(active)) || throw(ErrorException(
                "V2.2-M2 group thresholding did not converge within " *
                "$max_iterations iterations",
            ))
    end
    return intercept, fitted, active, iterations
end

function _operational_v22_driver_design(standardized::Matrix{Float64})
    first_anchor = maximum(OPERATIONAL_V22_DRIVER_LAGS) + 1
    anchors = first_anchor:(size(standardized, 1) - 1)
    design = Matrix{Float64}(undef, length(anchors),
        _OPERATIONAL_V22_DRIVER_NSTATE * _OPERATIONAL_V22_DRIVER_NLAG)
    targets = Matrix{Float64}(undef, length(anchors), _OPERATIONAL_V22_DRIVER_NSTATE)
    for (row, anchor) in pairs(anchors)
        column = 1
        for lag in OPERATIONAL_V22_DRIVER_LAGS,
                predictor in 1:_OPERATIONAL_V22_DRIVER_NSTATE
            design[row, column] = standardized[anchor - lag, predictor]
            column += 1
        end
        targets[row, :] .= @view standardized[anchor + 1, :]
    end
    return design, targets
end

function _operational_v22_driver_rollout_standardized(
        artifact::OperationalV22DriverArtifact,
        standardized_history::Matrix{Float64})
    size(standardized_history) == (
        _OPERATIONAL_V22_DRIVER_HISTORY_ROWS,
        _OPERATIONAL_V22_DRIVER_NSTATE,
    ) || throw(DimensionMismatch(
        "V2.2-M2 rollout history must have size (25, 5)",
    ))
    all(isfinite, standardized_history) || throw(ArgumentError(
        "V2.2-M2 rollout history must be finite",
    ))
    trajectory = Matrix{Float64}(undef,
        _OPERATIONAL_V22_DRIVER_HISTORY_ROWS + OPERATIONAL_V22_DRIVER_ROLLOUT_STEPS,
        _OPERATIONAL_V22_DRIVER_NSTATE,
    )
    trajectory[1:_OPERATIONAL_V22_DRIVER_HISTORY_ROWS, :] .= standardized_history
    for step in 1:OPERATIONAL_V22_DRIVER_ROLLOUT_STEPS
        anchor = _OPERATIONAL_V22_DRIVER_HISTORY_ROWS + step - 1
        for output in 1:_OPERATIONAL_V22_DRIVER_NSTATE
            value = artifact.intercept[output]
            for (lag_index, lag) in pairs(OPERATIONAL_V22_DRIVER_LAGS),
                    predictor in 1:_OPERATIONAL_V22_DRIVER_NSTATE
                value += artifact.coefficients[lag_index][predictor][output] *
                         trajectory[anchor - lag, predictor]
            end
            isfinite(value) || throw(ArgumentError(
                "V2.2-M2 produced a non-finite recursive state",
            ))
            trajectory[anchor + 1, output] = value
        end
    end
    return trajectory[(_OPERATIONAL_V22_DRIVER_HISTORY_ROWS + 1):end, :]
end

"Roll the stable continuation exactly fourteen 30-minute steps."
function operational_v22_driver_rollout(
        artifact::OperationalV22DriverArtifact,
        history::AbstractMatrix)
    converted = _operational_v22_driver_matrix(history)
    size(converted, 1) == _OPERATIONAL_V22_DRIVER_HISTORY_ROWS ||
        throw(DimensionMismatch(
            "V2.2-M2 rollout requires exactly 25 chronological state rows",
        ))
    center = collect(artifact.center)
    scale = collect(artifact.scale)
    standardized = (converted .- transpose(center)) ./ transpose(scale)
    rolled = _operational_v22_driver_rollout_standardized(artifact, standardized)
    result = rolled .* transpose(scale) .+ transpose(center)
    all(isfinite, result) || throw(ArgumentError(
        "V2.2-M2 rollout exceeds the supported Float64 range",
    ))
    return result
end

"Fit the frozen sequential group-thresholded ridge continuation."
function fit_operational_v22_driver(
        states::AbstractMatrix;
        ridge::Real,
        threshold::Real,
        label::AbstractString="operational-v2.2-m2-driver")
    converted = _operational_v22_driver_matrix(states)
    size(converted, 1) >= _OPERATIONAL_V22_DRIVER_HISTORY_ROWS + 1 ||
        throw(ArgumentError(
            "V2.2-M2 fitting requires at least 26 chronological state rows",
        ))
    ridge64 = _operational_v22_driver_float(ridge, "ridge")
    threshold64 = _operational_v22_driver_float(threshold, "threshold")
    ridge64 in OPERATIONAL_V22_DRIVER_RIDGE_GRID || throw(ArgumentError(
        "V2.2-M2 ridge must belong to the frozen search grid",
    ))
    threshold64 in OPERATIONAL_V22_DRIVER_THRESHOLD_GRID || throw(ArgumentError(
        "V2.2-M2 threshold must belong to the frozen search grid",
    ))

    center_vector = vec(mean(converted; dims=1))
    scale_vector = vec(std(converted; dims=1, corrected=false))
    all(isfinite, center_vector) || throw(ArgumentError(
        "V2.2-M2 state centers exceed the supported Float64 range",
    ))
    all(value -> isfinite(value) && value > 0.0, scale_vector) ||
        throw(ArgumentError(
            "V2.2-M2 fitting requires finite nonconstant state columns",
        ))
    standardized = (converted .- transpose(center_vector)) ./ transpose(scale_vector)
    design, targets = _operational_v22_driver_design(standardized)

    intercept, fitted, active, iterations =
        _operational_v22_driver_threshold_fit(
            design, targets, ridge64, threshold64,
        )

    coefficients = zeros(Float64,
        _OPERATIONAL_V22_DRIVER_NSTATE,
        _OPERATIONAL_V22_DRIVER_NSTATE,
        _OPERATIONAL_V22_DRIVER_NLAG,
    )
    support = falses(_OPERATIONAL_V22_DRIVER_NSTATE, _OPERATIONAL_V22_DRIVER_NLAG)
    group = 1
    for lag_index in 1:_OPERATIONAL_V22_DRIVER_NLAG,
            predictor in 1:_OPERATIONAL_V22_DRIVER_NSTATE
        support[predictor, lag_index] = active[group]
        coefficients[:, predictor, lag_index] .= @view fitted[group, :]
        group += 1
    end
    artifact = OperationalV22DriverArtifact(
        coefficients;
        center=center_vector,
        scale=scale_vector,
        intercept=intercept,
        support_mask=support,
        ridge=ridge64,
        threshold=threshold64,
        fit_rows=size(design, 1),
        threshold_iterations=iterations,
        label=label,
    )

    for anchor in _OPERATIONAL_V22_DRIVER_HISTORY_ROWS:size(standardized, 1)
        first_row = anchor - maximum(OPERATIONAL_V22_DRIVER_LAGS)
        history = Matrix(@view converted[first_row:anchor, :])
        operational_v22_driver_rollout(artifact, history)
    end
    return artifact
end

function _operational_v22_driver_hash_token(io::IO, value)
    text = value isa Float64 ? bitstring(value) : string(value)
    type_text = string(typeof(value))
    print(io, ncodeunits(type_text), ':', type_text, ':', ncodeunits(text), ':', text, '|')
    return nothing
end

"Return the portable SHA-256 identity of a V2.2-M2 driver artifact."
function operational_v22_driver_sha256(artifact::OperationalV22DriverArtifact)
    io = IOBuffer()
    for value in (
            OPERATIONAL_V22_DRIVER_SCHEMA_VERSION,
            OPERATIONAL_V22_DRIVER_PACKAGE_VERSION,
            artifact.label,
            OPERATIONAL_V22_DRIVER_CADENCE_MINUTES,
            OPERATIONAL_V22_DRIVER_ROLLOUT_STEPS,
            artifact.ridge,
            artifact.threshold,
            artifact.fit_rows,
            artifact.threshold_iterations,
            artifact.spectral_radius,
        )
        _operational_v22_driver_hash_token(io, value)
    end
    for values in (
            artifact.state_names,
            artifact.lags,
            artifact.center,
            artifact.scale,
            artifact.intercept,
        )
        _operational_v22_driver_hash_token(io, length(values))
        for value in values
            _operational_v22_driver_hash_token(io, value)
        end
    end
    for lag_index in 1:_OPERATIONAL_V22_DRIVER_NLAG,
            predictor in 1:_OPERATIONAL_V22_DRIVER_NSTATE
        _operational_v22_driver_hash_token(
            io, artifact.support_mask[lag_index][predictor],
        )
        for output in 1:_OPERATIONAL_V22_DRIVER_NSTATE
            _operational_v22_driver_hash_token(
                io, artifact.coefficients[lag_index][predictor][output],
            )
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

const _OPERATIONAL_V22_DRIVER_CSV_COLUMNS = (
    :schema_version,
    :package_version,
    :artifact_sha256,
    :label,
    :cadence_minutes,
    :rollout_steps,
    :state_schema,
    :lag_schema,
    :ridge,
    :threshold,
    :fit_rows,
    :threshold_iterations,
    :spectral_radius,
    :center_bx,
    :center_by,
    :center_bz,
    :center_logv,
    :center_logn,
    :scale_bx,
    :scale_by,
    :scale_bz,
    :scale_logv,
    :scale_logn,
    :intercept_bx,
    :intercept_by,
    :intercept_bz,
    :intercept_logv,
    :intercept_logn,
    :lag_index,
    :lag_samples,
    :predictor_index,
    :predictor,
    :selected,
    :coefficient_bx,
    :coefficient_by,
    :coefficient_bz,
    :coefficient_logv,
    :coefficient_logn,
)

"Atomically write a checksummed V2.2-M2 driver artifact."
function write_operational_v22_driver(
        path::AbstractString,
        artifact::OperationalV22DriverArtifact)
    target = String(path)
    mkpath(dirname(abspath(target)))
    checksum = operational_v22_driver_sha256(artifact)
    rows = NamedTuple[]
    for lag_index in 1:_OPERATIONAL_V22_DRIVER_NLAG,
            predictor in 1:_OPERATIONAL_V22_DRIVER_NSTATE
        coefficients = artifact.coefficients[lag_index][predictor]
        push!(rows, (
            schema_version=OPERATIONAL_V22_DRIVER_SCHEMA_VERSION,
            package_version=OPERATIONAL_V22_DRIVER_PACKAGE_VERSION,
            artifact_sha256=checksum,
            label=artifact.label,
            cadence_minutes=OPERATIONAL_V22_DRIVER_CADENCE_MINUTES,
            rollout_steps=OPERATIONAL_V22_DRIVER_ROLLOUT_STEPS,
            state_schema=join(String.(artifact.state_names), ";"),
            lag_schema=join(artifact.lags, ";"),
            ridge=artifact.ridge,
            threshold=artifact.threshold,
            fit_rows=artifact.fit_rows,
            threshold_iterations=artifact.threshold_iterations,
            spectral_radius=artifact.spectral_radius,
            center_bx=artifact.center[1],
            center_by=artifact.center[2],
            center_bz=artifact.center[3],
            center_logv=artifact.center[4],
            center_logn=artifact.center[5],
            scale_bx=artifact.scale[1],
            scale_by=artifact.scale[2],
            scale_bz=artifact.scale[3],
            scale_logv=artifact.scale[4],
            scale_logn=artifact.scale[5],
            intercept_bx=artifact.intercept[1],
            intercept_by=artifact.intercept[2],
            intercept_bz=artifact.intercept[3],
            intercept_logv=artifact.intercept[4],
            intercept_logn=artifact.intercept[5],
            lag_index=lag_index,
            lag_samples=artifact.lags[lag_index],
            predictor_index=predictor,
            predictor=String(artifact.state_names[predictor]),
            selected=artifact.support_mask[lag_index][predictor],
            coefficient_bx=coefficients[1],
            coefficient_by=coefficients[2],
            coefficient_bz=coefficients[3],
            coefficient_logv=coefficients[4],
            coefficient_logn=coefficients[5],
        ))
    end
    _write_selection_csv(target, rows)
    return target
end

function _operational_v22_driver_consistent(df::DataFrame, column::Symbol)
    values = df[!, column]
    any(ismissing, values) && throw(ArgumentError(
        "V2.2-M2 artifact metadata $column contains missing values",
    ))
    first_value = first(values)
    all(isequal(first_value), values) || throw(ArgumentError(
        "V2.2-M2 artifact metadata $column is inconsistent",
    ))
    return first_value
end

function _operational_v22_driver_int(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "V2.2-M2 artifact $field must be an integer",
    ))
    converted = Float64(value)
    isfinite(converted) && isinteger(converted) &&
        typemin(Int) <= converted <= typemax(Int) || throw(ArgumentError(
            "V2.2-M2 artifact $field must be an integer",
        ))
    return Int(converted)
end

function _operational_v22_driver_bool(value, field::AbstractString)
    value isa Bool || throw(ArgumentError(
        "V2.2-M2 artifact $field must be Boolean",
    ))
    return value
end

"Read and validate every serving-significant field of a V2.2-M2 artifact."
function read_operational_v22_driver(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "V2.2-M2 artifact must be a regular non-symlink file: $source",
    ))
    df = CSV.read(source, DataFrame)
    names(df) == collect(String.(_OPERATIONAL_V22_DRIVER_CSV_COLUMNS)) ||
        throw(ArgumentError("V2.2-M2 artifact CSV schema is invalid"))
    expected_rows = _OPERATIONAL_V22_DRIVER_NSTATE * _OPERATIONAL_V22_DRIVER_NLAG
    nrow(df) == expected_rows || throw(ArgumentError(
        "V2.2-M2 artifact must contain exactly 30 group rows",
    ))
    for row in 1:nrow(df), column in _OPERATIONAL_V22_DRIVER_CSV_COLUMNS
        ismissing(df[row, column]) && throw(ArgumentError(
            "V2.2-M2 artifact contains missing at row $row column $column",
        ))
    end

    schema = string(_operational_v22_driver_consistent(df, :schema_version))
    schema == OPERATIONAL_V22_DRIVER_SCHEMA_VERSION || throw(ArgumentError(
        "unsupported V2.2-M2 artifact schema: $schema",
    ))
    package_version = string(_operational_v22_driver_consistent(df, :package_version))
    package_version == OPERATIONAL_V22_DRIVER_PACKAGE_VERSION || throw(ArgumentError(
        "unsupported V2.2-M2 package version: $package_version",
    ))
    checksum = string(_operational_v22_driver_consistent(df, :artifact_sha256))
    occursin(r"^[0-9a-f]{64}$", checksum) || throw(ArgumentError(
        "V2.2-M2 artifact checksum is malformed",
    ))
    _operational_v22_driver_int(
        _operational_v22_driver_consistent(df, :cadence_minutes),
        "cadence_minutes",
    ) == OPERATIONAL_V22_DRIVER_CADENCE_MINUTES || throw(ArgumentError(
        "V2.2-M2 artifact cadence is not 30 minutes",
    ))
    _operational_v22_driver_int(
        _operational_v22_driver_consistent(df, :rollout_steps),
        "rollout_steps",
    ) == OPERATIONAL_V22_DRIVER_ROLLOUT_STEPS || throw(ArgumentError(
        "V2.2-M2 artifact rollout is not fourteen steps",
    ))
    string(_operational_v22_driver_consistent(df, :state_schema)) ==
        join(String.(OPERATIONAL_V22_DRIVER_STATES), ";") || throw(ArgumentError(
            "V2.2-M2 artifact state schema is invalid",
        ))
    string(_operational_v22_driver_consistent(df, :lag_schema)) ==
        join(OPERATIONAL_V22_DRIVER_LAGS, ";") || throw(ArgumentError(
            "V2.2-M2 artifact lag schema is invalid",
        ))

    coefficients = zeros(Float64,
        _OPERATIONAL_V22_DRIVER_NSTATE,
        _OPERATIONAL_V22_DRIVER_NSTATE,
        _OPERATIONAL_V22_DRIVER_NLAG,
    )
    support = falses(_OPERATIONAL_V22_DRIVER_NSTATE, _OPERATIONAL_V22_DRIVER_NLAG)
    coefficient_columns = (
        :coefficient_bx, :coefficient_by, :coefficient_bz,
        :coefficient_logv, :coefficient_logn,
    )
    row = 1
    for lag_index in 1:_OPERATIONAL_V22_DRIVER_NLAG,
            predictor in 1:_OPERATIONAL_V22_DRIVER_NSTATE
        _operational_v22_driver_int(df[row, :lag_index], "lag_index") == lag_index ||
            throw(ArgumentError("V2.2-M2 artifact lag indices are not sequential"))
        _operational_v22_driver_int(df[row, :lag_samples], "lag_samples") ==
            OPERATIONAL_V22_DRIVER_LAGS[lag_index] || throw(ArgumentError(
                "V2.2-M2 artifact lag value is invalid",
            ))
        _operational_v22_driver_int(df[row, :predictor_index], "predictor_index") ==
            predictor || throw(ArgumentError(
                "V2.2-M2 artifact predictor indices are not sequential",
            ))
        string(df[row, :predictor]) == String(OPERATIONAL_V22_DRIVER_STATES[predictor]) ||
            throw(ArgumentError("V2.2-M2 artifact predictor order is invalid"))
        support[predictor, lag_index] =
            _operational_v22_driver_bool(df[row, :selected], "selected")
        for output in 1:_OPERATIONAL_V22_DRIVER_NSTATE
            coefficients[output, predictor, lag_index] =
                _operational_v22_driver_float(
                    df[row, coefficient_columns[output]], "artifact coefficient",
                )
        end
        row += 1
    end
    center_columns = (:center_bx, :center_by, :center_bz, :center_logv, :center_logn)
    scale_columns = (:scale_bx, :scale_by, :scale_bz, :scale_logv, :scale_logn)
    intercept_columns = (
        :intercept_bx, :intercept_by, :intercept_bz,
        :intercept_logv, :intercept_logn,
    )
    center = [_operational_v22_driver_float(
        _operational_v22_driver_consistent(df, column), "artifact center",
    ) for column in center_columns]
    scale = [_operational_v22_driver_float(
        _operational_v22_driver_consistent(df, column), "artifact scale",
    ) for column in scale_columns]
    intercept = [_operational_v22_driver_float(
        _operational_v22_driver_consistent(df, column), "artifact intercept",
    ) for column in intercept_columns]
    artifact = OperationalV22DriverArtifact(
        coefficients;
        center=center,
        scale=scale,
        intercept=intercept,
        support_mask=support,
        ridge=_operational_v22_driver_consistent(df, :ridge),
        threshold=_operational_v22_driver_consistent(df, :threshold),
        fit_rows=_operational_v22_driver_int(
            _operational_v22_driver_consistent(df, :fit_rows), "fit_rows",
        ),
        threshold_iterations=_operational_v22_driver_int(
            _operational_v22_driver_consistent(df, :threshold_iterations),
            "threshold_iterations",
        ),
        label=string(_operational_v22_driver_consistent(df, :label)),
    )
    stored_radius = _operational_v22_driver_float(
        _operational_v22_driver_consistent(df, :spectral_radius),
        "spectral_radius",
    )
    stored_radius == artifact.spectral_radius || throw(ArgumentError(
        "V2.2-M2 artifact spectral radius is inconsistent",
    ))
    operational_v22_driver_sha256(artifact) == checksum || throw(ArgumentError(
        "V2.2-M2 artifact checksum mismatch",
    ))
    return artifact
end
