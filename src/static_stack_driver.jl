# Stable group-sparse continuation of the frozen static stack-M2 five-state driver.

import SHA

"Frozen static stack-M2 state order: magnetic field followed by log speed and density."
const STATIC_STACK_DRIVER_STATES = (:Bx, :By, :Bz, :logV, :logn)

"Frozen static stack-M2 delay coordinates in 30-minute samples."
const STATIC_STACK_DRIVER_LAGS = (0, 1, 2, 6, 12, 24)

"Time between sparse-driver state samples, in minutes."
const STATIC_STACK_DRIVER_CADENCE_MINUTES = 30
"Number of half-hour steps in the sparse-driver rollout."
const STATIC_STACK_DRIVER_ROLLOUT_STEPS = 14
"Numerical allowance above unit companion spectral radius in driver validation."
const STATIC_STACK_DRIVER_STABILITY_TOLERANCE = 1.0e-8
"Candidate ridge penalties for sparse-driver fitting."
const STATIC_STACK_DRIVER_RIDGE_GRID = Tuple(10.0 .^ (-6:2))
"Candidate group thresholds for sparse-driver fitting."
const STATIC_STACK_DRIVER_THRESHOLD_GRID =
    (0.0, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 3.0e-1)
const STATIC_STACK_DRIVER_MAX_THRESHOLD_ITERATIONS = 20
const STATIC_STACK_DRIVER_SCHEMA_VERSION = "static_stack_driver_driver_v1"
const STATIC_STACK_DRIVER_PACKAGE_VERSION = "SolarSINDy-0.2.1"

const _STATIC_STACK_DRIVER_NSTATE = length(STATIC_STACK_DRIVER_STATES)
const _STATIC_STACK_DRIVER_NLAG = length(STATIC_STACK_DRIVER_LAGS)
const _STATIC_STACK_DRIVER_HISTORY_ROWS = maximum(STATIC_STACK_DRIVER_LAGS) + 1

function _static_stack_driver_float(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "static stack-M2 $field must be a real number",
    ))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError(
        "static stack-M2 $field must be finite",
    ))
    return converted
end

function _static_stack_driver_coefficients(values)
    size(values) == (
        _STATIC_STACK_DRIVER_NSTATE,
        _STATIC_STACK_DRIVER_NSTATE,
        _STATIC_STACK_DRIVER_NLAG,
    ) || throw(DimensionMismatch(
        "static stack-M2 coefficients must have size (5, 5, 6)",
    ))
    return ntuple(
        lag_index -> ntuple(
            predictor -> ntuple(
                output -> _static_stack_driver_float(
                    values[output, predictor, lag_index],
                    "coefficient",
                ),
                _STATIC_STACK_DRIVER_NSTATE,
            ),
            _STATIC_STACK_DRIVER_NSTATE,
        ),
        _STATIC_STACK_DRIVER_NLAG,
    )
end

function _static_stack_driver_support(values)
    size(values) == (
        _STATIC_STACK_DRIVER_NSTATE,
        _STATIC_STACK_DRIVER_NLAG,
    ) || throw(DimensionMismatch(
        "static stack-M2 support must have size (5, 6)",
    ))
    all(value -> value isa Bool, values) || throw(ArgumentError(
        "static stack-M2 support entries must be Boolean",
    ))
    return ntuple(
        lag_index -> ntuple(
            predictor -> values[predictor, lag_index],
            _STATIC_STACK_DRIVER_NSTATE,
        ),
        _STATIC_STACK_DRIVER_NLAG,
    )
end

function _static_stack_driver_coefficient_array(coefficients)
    values = Array{Float64}(undef,
        _STATIC_STACK_DRIVER_NSTATE,
        _STATIC_STACK_DRIVER_NSTATE,
        _STATIC_STACK_DRIVER_NLAG,
    )
    for lag_index in 1:_STATIC_STACK_DRIVER_NLAG,
            predictor in 1:_STATIC_STACK_DRIVER_NSTATE,
            output in 1:_STATIC_STACK_DRIVER_NSTATE
        values[output, predictor, lag_index] =
            coefficients[lag_index][predictor][output]
    end
    return values
end

function _static_stack_driver_support_array(support)
    values = Matrix{Bool}(undef,
        _STATIC_STACK_DRIVER_NSTATE,
        _STATIC_STACK_DRIVER_NLAG,
    )
    for lag_index in 1:_STATIC_STACK_DRIVER_NLAG,
            predictor in 1:_STATIC_STACK_DRIVER_NSTATE
        values[predictor, lag_index] = support[lag_index][predictor]
    end
    return values
end

function _static_stack_driver_companion(coefficients)
    coefficient_array = _static_stack_driver_coefficient_array(coefficients)
    dimension = _STATIC_STACK_DRIVER_NSTATE * _STATIC_STACK_DRIVER_HISTORY_ROWS
    companion = zeros(Float64, dimension, dimension)
    rows = 1:_STATIC_STACK_DRIVER_NSTATE
    for (lag_index, lag) in pairs(STATIC_STACK_DRIVER_LAGS)
        columns = ((lag * _STATIC_STACK_DRIVER_NSTATE + 1):
                   ((lag + 1) * _STATIC_STACK_DRIVER_NSTATE))
        companion[rows, columns] .= @view coefficient_array[:, :, lag_index]
    end
    for block in 2:_STATIC_STACK_DRIVER_HISTORY_ROWS
        destination = (((block - 1) * _STATIC_STACK_DRIVER_NSTATE + 1):
                       (block * _STATIC_STACK_DRIVER_NSTATE))
        source = (((block - 2) * _STATIC_STACK_DRIVER_NSTATE + 1):
                  ((block - 1) * _STATIC_STACK_DRIVER_NSTATE))
        companion[destination, source] .=
            Matrix{Float64}(I, _STATIC_STACK_DRIVER_NSTATE,
                            _STATIC_STACK_DRIVER_NSTATE)
    end
    return companion
end

function _static_stack_driver_spectral_radius(coefficients)
    eigenvalues = eigvals(_static_stack_driver_companion(coefficients))
    radius = maximum(abs, eigenvalues)
    isfinite(radius) || throw(ArgumentError(
        "static stack-M2 companion spectral radius is non-finite",
    ))
    return radius
end

"Immutable, checksummed static stack-M2 group-sparse driver-continuation artifact."
struct StaticStackDriverArtifact
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

    function StaticStackDriverArtifact(
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
            "static stack-M2 artifact label must not be empty",
        ))
        state_names == STATIC_STACK_DRIVER_STATES || throw(ArgumentError(
            "static stack-M2 state order must be exactly (Bx, By, Bz, logV, logn)",
        ))
        lags == STATIC_STACK_DRIVER_LAGS || throw(ArgumentError(
            "static stack-M2 lag order must be exactly (0, 1, 2, 6, 12, 24)",
        ))
        all(isfinite, center) && all(isfinite, scale) && all(isfinite, intercept) ||
            throw(ArgumentError("static stack-M2 normalization values must be finite"))
        all(>(0.0), scale) || throw(ArgumentError(
            "static stack-M2 state scales must be positive",
        ))
        ridge in STATIC_STACK_DRIVER_RIDGE_GRID || throw(ArgumentError(
            "static stack-M2 ridge must belong to the frozen search grid",
        ))
        threshold in STATIC_STACK_DRIVER_THRESHOLD_GRID || throw(ArgumentError(
            "static stack-M2 threshold must belong to the frozen search grid",
        ))
        fit_rows >= 1 || throw(ArgumentError(
            "static stack-M2 artifact requires at least one fit row",
        ))
        1 <= threshold_iterations <= STATIC_STACK_DRIVER_MAX_THRESHOLD_ITERATIONS ||
            throw(ArgumentError(
                "static stack-M2 threshold iterations must lie in 1:20",
            ))
        for lag_index in 1:_STATIC_STACK_DRIVER_NLAG,
                predictor in 1:_STATIC_STACK_DRIVER_NSTATE
            group = coefficients[lag_index][predictor]
            all(isfinite, group) || throw(ArgumentError(
                "static stack-M2 coefficients must be finite",
            ))
            if support_mask[lag_index][predictor]
                norm(group) >= threshold || throw(ArgumentError(
                    "static stack-M2 selected groups must meet the stored threshold",
                ))
            else
                all(iszero, group) || throw(ArgumentError(
                    "static stack-M2 excluded groups must have exact zero coefficients",
                ))
            end
        end
        radius = _static_stack_driver_spectral_radius(coefficients)
        radius <= 1.0 + STATIC_STACK_DRIVER_STABILITY_TOLERANCE ||
            throw(ArgumentError(
                "static stack-M2 candidate is unstable (spectral radius=$radius)",
            ))
        return new(
            label, state_names, lags, center, scale, intercept, coefficients,
            support_mask, ridge, threshold, fit_rows, threshold_iterations,
            radius,
        )
    end
end

"Construct and validate a stable static stack-M2 artifact from a `(5, 5, 6)` map."
function StaticStackDriverArtifact(
        coefficients::AbstractArray{<:Real,3};
        center=zeros(_STATIC_STACK_DRIVER_NSTATE),
        scale=ones(_STATIC_STACK_DRIVER_NSTATE),
        intercept=zeros(_STATIC_STACK_DRIVER_NSTATE),
        support_mask=dropdims(any(!iszero, coefficients; dims=1); dims=1),
        ridge::Real=first(STATIC_STACK_DRIVER_RIDGE_GRID),
        threshold::Real=first(STATIC_STACK_DRIVER_THRESHOLD_GRID),
        fit_rows::Integer=1,
        threshold_iterations::Integer=1,
        label::AbstractString="operational-static-stack-m2-driver")
    length(center) == _STATIC_STACK_DRIVER_NSTATE || throw(DimensionMismatch(
        "static stack-M2 center must contain exactly five values",
    ))
    length(scale) == _STATIC_STACK_DRIVER_NSTATE || throw(DimensionMismatch(
        "static stack-M2 scale must contain exactly five values",
    ))
    length(intercept) == _STATIC_STACK_DRIVER_NSTATE || throw(DimensionMismatch(
        "static stack-M2 intercept must contain exactly five values",
    ))
    converted_center = ntuple(
        index -> _static_stack_driver_float(center[index], "center"),
        _STATIC_STACK_DRIVER_NSTATE,
    )
    converted_scale = ntuple(
        index -> _static_stack_driver_float(scale[index], "scale"),
        _STATIC_STACK_DRIVER_NSTATE,
    )
    converted_intercept = ntuple(
        index -> _static_stack_driver_float(intercept[index], "intercept"),
        _STATIC_STACK_DRIVER_NSTATE,
    )
    return StaticStackDriverArtifact(
        String(label), STATIC_STACK_DRIVER_STATES, STATIC_STACK_DRIVER_LAGS,
        converted_center, converted_scale, converted_intercept,
        _static_stack_driver_coefficients(coefficients),
        _static_stack_driver_support(support_mask),
        _static_stack_driver_float(ridge, "ridge"),
        _static_stack_driver_float(threshold, "threshold"),
        Int(fit_rows), Int(threshold_iterations), Val(:validated),
    )
end

"Return a mutable `(output, predictor, lag)` copy of the frozen coefficients."
static_stack_driver_coefficients(artifact::StaticStackDriverArtifact) =
    _static_stack_driver_coefficient_array(artifact.coefficients)

"Return a mutable `(predictor, lag)` copy of the selected group support."
static_stack_driver_support(artifact::StaticStackDriverArtifact) =
    _static_stack_driver_support_array(artifact.support_mask)

"Construct the 125-by-125 companion matrix of a static stack-M2 artifact."
static_stack_driver_companion(artifact::StaticStackDriverArtifact) =
    _static_stack_driver_companion(artifact.coefficients)

"Return the verified companion spectral radius of a static stack-M2 artifact."
static_stack_driver_spectral_radius(artifact::StaticStackDriverArtifact) =
    artifact.spectral_radius

function _static_stack_driver_matrix(states::AbstractMatrix)
    size(states, 2) == _STATIC_STACK_DRIVER_NSTATE || throw(DimensionMismatch(
        "static stack-M2 state matrix must have exactly five columns",
    ))
    all(value -> value isa Real && !(value isa Bool), states) || throw(ArgumentError(
        "static stack-M2 states must be real numbers",
    ))
    converted = Matrix{Float64}(states)
    all(isfinite, converted) || throw(ArgumentError(
        "static stack-M2 states must be finite",
    ))
    return converted
end

function _static_stack_driver_ridge_fit(
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
        "static stack-M2 ridge coefficients are non-finite",
    ))
    coefficients[selected, :] .= fitted
    intercept = target_mean - transpose(fitted) * predictor_mean
    all(isfinite, intercept) || throw(ArgumentError(
        "static stack-M2 ridge intercept is non-finite",
    ))
    return intercept, coefficients
end

function _static_stack_driver_threshold_fit(
        design::Matrix{Float64},
        targets::Matrix{Float64},
        ridge::Float64,
        threshold::Float64;
        max_iterations::Int=STATIC_STACK_DRIVER_MAX_THRESHOLD_ITERATIONS)
    max_iterations >= 1 || throw(ArgumentError(
        "static stack-M2 threshold fitting requires at least one iteration",
    ))
    active = trues(size(design, 2))
    intercept = zeros(Float64, size(targets, 2))
    fitted = zeros(Float64, size(design, 2), size(targets, 2))
    iterations = 0
    support_changed = false
    for iteration in 1:max_iterations
        iterations = iteration
        intercept, fitted = _static_stack_driver_ridge_fit(
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
        intercept, fitted = _static_stack_driver_ridge_fit(
            design, targets, active, ridge,
        )
        all(group -> !active[group] ||
                     norm(@view(fitted[group, :])) >= threshold,
            eachindex(active)) || throw(ErrorException(
                "static stack-M2 group thresholding did not converge within " *
                "$max_iterations iterations",
            ))
    end
    return intercept, fitted, active, iterations
end

function _static_stack_driver_design(standardized::Matrix{Float64})
    first_anchor = maximum(STATIC_STACK_DRIVER_LAGS) + 1
    anchors = first_anchor:(size(standardized, 1) - 1)
    design = Matrix{Float64}(undef, length(anchors),
        _STATIC_STACK_DRIVER_NSTATE * _STATIC_STACK_DRIVER_NLAG)
    targets = Matrix{Float64}(undef, length(anchors), _STATIC_STACK_DRIVER_NSTATE)
    for (row, anchor) in pairs(anchors)
        column = 1
        for lag in STATIC_STACK_DRIVER_LAGS,
                predictor in 1:_STATIC_STACK_DRIVER_NSTATE
            design[row, column] = standardized[anchor - lag, predictor]
            column += 1
        end
        targets[row, :] .= @view standardized[anchor + 1, :]
    end
    return design, targets
end

function _static_stack_driver_rollout_standardized(
        artifact::StaticStackDriverArtifact,
        standardized_history::Matrix{Float64})
    size(standardized_history) == (
        _STATIC_STACK_DRIVER_HISTORY_ROWS,
        _STATIC_STACK_DRIVER_NSTATE,
    ) || throw(DimensionMismatch(
        "static stack-M2 rollout history must have size (25, 5)",
    ))
    all(isfinite, standardized_history) || throw(ArgumentError(
        "static stack-M2 rollout history must be finite",
    ))
    trajectory = Matrix{Float64}(undef,
        _STATIC_STACK_DRIVER_HISTORY_ROWS + STATIC_STACK_DRIVER_ROLLOUT_STEPS,
        _STATIC_STACK_DRIVER_NSTATE,
    )
    trajectory[1:_STATIC_STACK_DRIVER_HISTORY_ROWS, :] .= standardized_history
    for step in 1:STATIC_STACK_DRIVER_ROLLOUT_STEPS
        anchor = _STATIC_STACK_DRIVER_HISTORY_ROWS + step - 1
        for output in 1:_STATIC_STACK_DRIVER_NSTATE
            value = artifact.intercept[output]
            for (lag_index, lag) in pairs(STATIC_STACK_DRIVER_LAGS),
                    predictor in 1:_STATIC_STACK_DRIVER_NSTATE
                value += artifact.coefficients[lag_index][predictor][output] *
                         trajectory[anchor - lag, predictor]
            end
            isfinite(value) || throw(ArgumentError(
                "static stack-M2 produced a non-finite recursive state",
            ))
            trajectory[anchor + 1, output] = value
        end
    end
    return trajectory[(_STATIC_STACK_DRIVER_HISTORY_ROWS + 1):end, :]
end

"Roll the stable continuation exactly fourteen 30-minute steps."
function static_stack_driver_rollout(
        artifact::StaticStackDriverArtifact,
        history::AbstractMatrix)
    converted = _static_stack_driver_matrix(history)
    size(converted, 1) == _STATIC_STACK_DRIVER_HISTORY_ROWS ||
        throw(DimensionMismatch(
            "static stack-M2 rollout requires exactly 25 chronological state rows",
        ))
    center = collect(artifact.center)
    scale = collect(artifact.scale)
    standardized = (converted .- transpose(center)) ./ transpose(scale)
    rolled = _static_stack_driver_rollout_standardized(artifact, standardized)
    result = rolled .* transpose(scale) .+ transpose(center)
    all(isfinite, result) || throw(ArgumentError(
        "static stack-M2 rollout exceeds the supported Float64 range",
    ))
    return result
end

function _static_stack_driver_fit_segments(
        converted_segments::Vector{Matrix{Float64}},
        ridge64::Float64,
        threshold64::Float64,
        label::String;
        verify_training_rollouts::Bool=true)
    all(segment -> size(segment, 1) >=
                   _STATIC_STACK_DRIVER_HISTORY_ROWS + 1,
        converted_segments) || throw(ArgumentError(
            "each static stack-M2 fitting segment requires at least 26 chronological state rows",
        ))
    combined = reduce(vcat, converted_segments)
    center_vector = vec(mean(combined; dims=1))
    scale_vector = vec(std(combined; dims=1, corrected=false))
    all(isfinite, center_vector) || throw(ArgumentError(
        "static stack-M2 state centers exceed the supported Float64 range",
    ))
    all(value -> isfinite(value) && value > 0.0, scale_vector) ||
        throw(ArgumentError(
            "static stack-M2 fitting requires finite nonconstant state columns",
        ))
    standardized_segments = [
        (segment .- transpose(center_vector)) ./ transpose(scale_vector)
        for segment in converted_segments
    ]
    design_targets = map(_static_stack_driver_design, standardized_segments)
    design = reduce(vcat, first.(design_targets))
    targets = reduce(vcat, last.(design_targets))

    intercept, fitted, active, iterations =
        _static_stack_driver_threshold_fit(
            design, targets, ridge64, threshold64,
        )

    coefficients = zeros(Float64,
        _STATIC_STACK_DRIVER_NSTATE,
        _STATIC_STACK_DRIVER_NSTATE,
        _STATIC_STACK_DRIVER_NLAG,
    )
    support = falses(_STATIC_STACK_DRIVER_NSTATE, _STATIC_STACK_DRIVER_NLAG)
    group = 1
    for lag_index in 1:_STATIC_STACK_DRIVER_NLAG,
            predictor in 1:_STATIC_STACK_DRIVER_NSTATE
        support[predictor, lag_index] = active[group]
        coefficients[:, predictor, lag_index] .= @view fitted[group, :]
        group += 1
    end
    artifact = StaticStackDriverArtifact(
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

    if verify_training_rollouts
        for segment in converted_segments
            for anchor in _STATIC_STACK_DRIVER_HISTORY_ROWS:size(segment, 1)
                first_row = anchor - maximum(STATIC_STACK_DRIVER_LAGS)
                history = Matrix(@view segment[first_row:anchor, :])
                static_stack_driver_rollout(artifact, history)
            end
        end
    end
    return artifact
end

function _static_stack_driver_fit_hyperparameters(ridge, threshold)
    ridge64 = _static_stack_driver_float(ridge, "ridge")
    threshold64 = _static_stack_driver_float(threshold, "threshold")
    ridge64 in STATIC_STACK_DRIVER_RIDGE_GRID || throw(ArgumentError(
        "static stack-M2 ridge must belong to the frozen search grid",
    ))
    threshold64 in STATIC_STACK_DRIVER_THRESHOLD_GRID || throw(ArgumentError(
        "static stack-M2 threshold must belong to the frozen search grid",
    ))
    return ridge64, threshold64
end

"Fit the frozen continuation to one contiguous 30-minute state sequence."
function fit_static_stack_driver(
        states::AbstractMatrix;
        ridge::Real,
        threshold::Real,
        label::AbstractString="operational-static-stack-m2-driver")
    converted = _static_stack_driver_matrix(states)
    ridge64, threshold64 =
        _static_stack_driver_fit_hyperparameters(ridge, threshold)
    return _static_stack_driver_fit_segments(
        [converted], ridge64, threshold64, String(label),
    )
end

"Fit without creating artificial lag or target rows across observed data gaps."
function fit_static_stack_driver(
        segments::AbstractVector;
        ridge::Real,
        threshold::Real,
        label::AbstractString="operational-static-stack-m2-driver")
    isempty(segments) && throw(ArgumentError(
        "static stack-M2 segmented fitting requires at least one segment",
    ))
    all(segment -> segment isa AbstractMatrix, segments) || throw(ArgumentError(
        "static stack-M2 fitting segments must be state matrices",
    ))
    converted = [_static_stack_driver_matrix(segment) for segment in segments]
    ridge64, threshold64 =
        _static_stack_driver_fit_hyperparameters(ridge, threshold)
    return _static_stack_driver_fit_segments(
        converted, ridge64, threshold64, String(label),
    )
end

function _static_stack_driver_hash_token(io::IO, value)
    text = value isa Float64 ? bitstring(value) : string(value)
    type_text = string(typeof(value))
    print(io, ncodeunits(type_text), ':', type_text, ':', ncodeunits(text), ':', text, '|')
    return nothing
end

"Return the portable SHA-256 identity of a static stack-M2 driver artifact."
function static_stack_driver_sha256(artifact::StaticStackDriverArtifact)
    io = IOBuffer()
    for value in (
            STATIC_STACK_DRIVER_SCHEMA_VERSION,
            STATIC_STACK_DRIVER_PACKAGE_VERSION,
            artifact.label,
            STATIC_STACK_DRIVER_CADENCE_MINUTES,
            STATIC_STACK_DRIVER_ROLLOUT_STEPS,
            artifact.ridge,
            artifact.threshold,
            artifact.fit_rows,
            artifact.threshold_iterations,
            # Rounded: an exact radius would make this "portable" identity depend on the
            # eigen-solver path (see STATIC_STACK_SPECTRAL_RADIUS_DIGITS).
            _static_stack_hashable_spectral_radius(artifact.spectral_radius),
        )
        _static_stack_driver_hash_token(io, value)
    end
    for values in (
            artifact.state_names,
            artifact.lags,
            artifact.center,
            artifact.scale,
            artifact.intercept,
        )
        _static_stack_driver_hash_token(io, length(values))
        for value in values
            _static_stack_driver_hash_token(io, value)
        end
    end
    for lag_index in 1:_STATIC_STACK_DRIVER_NLAG,
            predictor in 1:_STATIC_STACK_DRIVER_NSTATE
        _static_stack_driver_hash_token(
            io, artifact.support_mask[lag_index][predictor],
        )
        for output in 1:_STATIC_STACK_DRIVER_NSTATE
            _static_stack_driver_hash_token(
                io, artifact.coefficients[lag_index][predictor][output],
            )
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

const _STATIC_STACK_DRIVER_CSV_COLUMNS = (
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

"Atomically write a checksummed static stack-M2 driver artifact."
function write_static_stack_driver(
        path::AbstractString,
        artifact::StaticStackDriverArtifact)
    target = String(path)
    mkpath(dirname(abspath(target)))
    checksum = static_stack_driver_sha256(artifact)
    rows = NamedTuple[]
    for lag_index in 1:_STATIC_STACK_DRIVER_NLAG,
            predictor in 1:_STATIC_STACK_DRIVER_NSTATE
        coefficients = artifact.coefficients[lag_index][predictor]
        push!(rows, (
            schema_version=STATIC_STACK_DRIVER_SCHEMA_VERSION,
            package_version=STATIC_STACK_DRIVER_PACKAGE_VERSION,
            artifact_sha256=checksum,
            label=artifact.label,
            cadence_minutes=STATIC_STACK_DRIVER_CADENCE_MINUTES,
            rollout_steps=STATIC_STACK_DRIVER_ROLLOUT_STEPS,
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

function _static_stack_driver_consistent(df::DataFrame, column::Symbol)
    values = df[!, column]
    any(ismissing, values) && throw(ArgumentError(
        "static stack-M2 artifact metadata $column contains missing values",
    ))
    first_value = first(values)
    all(isequal(first_value), values) || throw(ArgumentError(
        "static stack-M2 artifact metadata $column is inconsistent",
    ))
    return first_value
end

function _static_stack_driver_int(value, field::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "static stack-M2 artifact $field must be an integer",
    ))
    converted = Float64(value)
    isfinite(converted) && isinteger(converted) &&
        typemin(Int) <= converted <= typemax(Int) || throw(ArgumentError(
            "static stack-M2 artifact $field must be an integer",
        ))
    return Int(converted)
end

function _static_stack_driver_bool(value, field::AbstractString)
    value isa Bool || throw(ArgumentError(
        "static stack-M2 artifact $field must be Boolean",
    ))
    return value
end

"Read and validate every serving-significant field of a static stack-M2 artifact."
function read_static_stack_driver(path::AbstractString)
    source = String(path)
    isfile(source) && !islink(source) || throw(ArgumentError(
        "static stack-M2 artifact must be a regular non-symlink file: $source",
    ))
    # Text-valued columns are read as text: a numeric-looking label ("007", "1e3") would otherwise
    # parse as a number and change the hashed string. `validate = false` keeps a missing column
    # reporting through the schema check below rather than through the parser.
    df = CSV.read(IOBuffer(read(source)), DataFrame;
                  types = Dict(:schema_version => String, :package_version => String,
                               :artifact_sha256 => String, :label => String,
                               :state_schema => String, :lag_schema => String,
                               :predictor => String),
                  validate = false)
    names(df) == collect(String.(_STATIC_STACK_DRIVER_CSV_COLUMNS)) ||
        throw(ArgumentError("static stack-M2 artifact CSV schema is invalid"))
    expected_rows = _STATIC_STACK_DRIVER_NSTATE * _STATIC_STACK_DRIVER_NLAG
    nrow(df) == expected_rows || throw(ArgumentError(
        "static stack-M2 artifact must contain exactly 30 group rows",
    ))
    for row in 1:nrow(df), column in _STATIC_STACK_DRIVER_CSV_COLUMNS
        ismissing(df[row, column]) && throw(ArgumentError(
            "static stack-M2 artifact contains missing at row $row column $column",
        ))
    end

    schema = string(_static_stack_driver_consistent(df, :schema_version))
    schema == STATIC_STACK_DRIVER_SCHEMA_VERSION || throw(ArgumentError(
        "unsupported static stack-M2 artifact schema: $schema",
    ))
    package_version = string(_static_stack_driver_consistent(df, :package_version))
    package_version == STATIC_STACK_DRIVER_PACKAGE_VERSION || throw(ArgumentError(
        "unsupported static stack-M2 package version: $package_version",
    ))
    checksum = string(_static_stack_driver_consistent(df, :artifact_sha256))
    occursin(r"^[0-9a-f]{64}$", checksum) || throw(ArgumentError(
        "static stack-M2 artifact checksum is malformed",
    ))
    _static_stack_driver_int(
        _static_stack_driver_consistent(df, :cadence_minutes),
        "cadence_minutes",
    ) == STATIC_STACK_DRIVER_CADENCE_MINUTES || throw(ArgumentError(
        "static stack-M2 artifact cadence is not 30 minutes",
    ))
    _static_stack_driver_int(
        _static_stack_driver_consistent(df, :rollout_steps),
        "rollout_steps",
    ) == STATIC_STACK_DRIVER_ROLLOUT_STEPS || throw(ArgumentError(
        "static stack-M2 artifact rollout is not fourteen steps",
    ))
    string(_static_stack_driver_consistent(df, :state_schema)) ==
        join(String.(STATIC_STACK_DRIVER_STATES), ";") || throw(ArgumentError(
            "static stack-M2 artifact state schema is invalid",
        ))
    string(_static_stack_driver_consistent(df, :lag_schema)) ==
        join(STATIC_STACK_DRIVER_LAGS, ";") || throw(ArgumentError(
            "static stack-M2 artifact lag schema is invalid",
        ))

    coefficients = zeros(Float64,
        _STATIC_STACK_DRIVER_NSTATE,
        _STATIC_STACK_DRIVER_NSTATE,
        _STATIC_STACK_DRIVER_NLAG,
    )
    support = falses(_STATIC_STACK_DRIVER_NSTATE, _STATIC_STACK_DRIVER_NLAG)
    coefficient_columns = (
        :coefficient_bx, :coefficient_by, :coefficient_bz,
        :coefficient_logv, :coefficient_logn,
    )
    row = 1
    for lag_index in 1:_STATIC_STACK_DRIVER_NLAG,
            predictor in 1:_STATIC_STACK_DRIVER_NSTATE
        _static_stack_driver_int(df[row, :lag_index], "lag_index") == lag_index ||
            throw(ArgumentError("static stack-M2 artifact lag indices are not sequential"))
        _static_stack_driver_int(df[row, :lag_samples], "lag_samples") ==
            STATIC_STACK_DRIVER_LAGS[lag_index] || throw(ArgumentError(
                "static stack-M2 artifact lag value is invalid",
            ))
        _static_stack_driver_int(df[row, :predictor_index], "predictor_index") ==
            predictor || throw(ArgumentError(
                "static stack-M2 artifact predictor indices are not sequential",
            ))
        string(df[row, :predictor]) == String(STATIC_STACK_DRIVER_STATES[predictor]) ||
            throw(ArgumentError("static stack-M2 artifact predictor order is invalid"))
        support[predictor, lag_index] =
            _static_stack_driver_bool(df[row, :selected], "selected")
        for output in 1:_STATIC_STACK_DRIVER_NSTATE
            coefficients[output, predictor, lag_index] =
                _static_stack_driver_float(
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
    center = [_static_stack_driver_float(
        _static_stack_driver_consistent(df, column), "artifact center",
    ) for column in center_columns]
    scale = [_static_stack_driver_float(
        _static_stack_driver_consistent(df, column), "artifact scale",
    ) for column in scale_columns]
    intercept = [_static_stack_driver_float(
        _static_stack_driver_consistent(df, column), "artifact intercept",
    ) for column in intercept_columns]
    artifact = StaticStackDriverArtifact(
        coefficients;
        center=center,
        scale=scale,
        intercept=intercept,
        support_mask=support,
        ridge=_static_stack_driver_consistent(df, :ridge),
        threshold=_static_stack_driver_consistent(df, :threshold),
        fit_rows=_static_stack_driver_int(
            _static_stack_driver_consistent(df, :fit_rows), "fit_rows",
        ),
        threshold_iterations=_static_stack_driver_int(
            _static_stack_driver_consistent(df, :threshold_iterations),
            "threshold_iterations",
        ),
        label=string(_static_stack_driver_consistent(df, :label)),
    )
    stored_radius = _static_stack_driver_float(
        _static_stack_driver_consistent(df, :spectral_radius),
        "spectral_radius",
    )
    _static_stack_spectral_radius_agrees(stored_radius, artifact.spectral_radius) ||
        throw(ArgumentError(
            "static stack-M2 artifact spectral radius is inconsistent",
        ))
    static_stack_driver_sha256(artifact) == checksum || throw(ArgumentError(
        "static stack-M2 artifact checksum mismatch",
    ))
    return artifact
end
