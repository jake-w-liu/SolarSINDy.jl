# Evaluation metrics for model comparison

using SpecialFunctions: erfc

function _validate_metric_vectors(name::AbstractString, vectors::AbstractVector...)
    isempty(vectors) && throw(ArgumentError("$name requires paired inputs"))
    n = length(first(vectors))
    n >= 1 || throw(ArgumentError("$name requires at least one paired sample"))
    all(length(v) == n for v in vectors) ||
        throw(DimensionMismatch("$name inputs must have equal length"))
    all(v -> all(x -> x isa Real && !(x isa Bool) && isfinite(x), v), vectors) ||
        throw(ArgumentError("$name inputs must contain only finite real values"))
    return n
end

function _metric_float_vectors(name::AbstractString, vectors::AbstractVector...)
    _validate_metric_vectors(name, vectors...)
    converted = map(vector -> Float64.(vector), vectors)
    all(vector -> all(isfinite, vector), converted) || throw(ArgumentError(
        "$name inputs must be representable as finite Float64 values",
    ))
    return converted
end

_metric_difference(left::Integer, right::Integer) =
    Float64(BigInt(left) - BigInt(right))
_metric_difference(left::Real, right::Real) = Float64(left - right)

function _metric_differences(name::AbstractString, left::AbstractVector,
                             right::AbstractVector)
    differences = map(_metric_difference, left, right)
    all(isfinite, differences) || throw(ArgumentError(
        "$name paired differences exceed the supported Float64 range",
    ))
    return differences
end

function _metric_centered(name::AbstractString, values::AbstractVector)
    anchor = first(values)
    offsets = map(value -> _metric_difference(value, anchor), values)
    all(isfinite, offsets) || throw(ArgumentError(
        "$name centered values exceed the supported Float64 range",
    ))
    centered = offsets .- _stable_mean(offsets)
    all(isfinite, centered) || throw(ArgumentError(
        "$name centered values exceed the supported Float64 range",
    ))
    return centered
end

function _stable_root_mean_square(values::AbstractVector{<:Real})
    scale = maximum(abs, values)
    scale == 0 && return zero(scale)
    normalized_mean = sum(value -> abs2(value / scale), values) / length(values)
    return scale * sqrt(clamp(normalized_mean,
                              zero(normalized_mean), one(normalized_mean)))
end

function _stable_mean_absolute(values::AbstractVector{Float64})
    scale = maximum(abs, values)
    scale == 0 && return 0.0
    normalized_mean = sum(value -> abs(value / scale), values) / length(values)
    return scale * clamp(normalized_mean, 0.0, 1.0)
end

function _stable_mean(values::AbstractVector{Float64})
    scale = maximum(abs, values)
    scale == 0 && return 0.0
    return (sum(value -> value / scale, values) / length(values)) * scale
end

function _finite_efficiency(name::AbstractString, numerator::Float64,
                            denominator::Float64)
    denominator == 0 && return NaN
    ratio = numerator / denominator
    isfinite(ratio) || throw(ArgumentError("$name error ratio exceeds Float64 range"))
    squared = ratio * ratio
    isfinite(squared) || throw(ArgumentError("$name score exceeds Float64 range"))
    score = 1.0 - squared
    isfinite(score) || throw(ArgumentError("$name score exceeds Float64 range"))
    return score
end

"""
    rmse(predicted, observed)

Root Mean Square Error.
"""
function rmse(predicted::AbstractVector, observed::AbstractVector)
    _metric_float_vectors("rmse", predicted, observed)
    differences = _metric_differences("rmse", predicted, observed)
    return _stable_root_mean_square(differences)
end

"""
    mae(predicted, observed)

Mean Absolute Error, evaluated with scaled accumulation to avoid integer and
floating-point overflow for representable inputs.
"""
function mae(predicted::AbstractVector, observed::AbstractVector)
    _metric_float_vectors("mae", predicted, observed)
    differences = _metric_differences("mae", predicted, observed)
    return _stable_mean_absolute(differences)
end

"""
    correlation(predicted, observed)

Pearson correlation coefficient. Returns `NaN` for a degenerate (zero-variance)
input rather than letting `cor` emit a warning/`NaN` implicitly, so callers can
filter undefined correlations explicitly.
"""
function correlation(predicted::AbstractVector, observed::AbstractVector)
    _metric_float_vectors("correlation", predicted, observed)
    predicted_centered = _metric_centered("correlation", predicted)
    observed_centered = _metric_centered("correlation", observed)
    predicted_scale = maximum(abs, predicted_centered)
    observed_scale = maximum(abs, observed_centered)
    (predicted_scale == 0 || observed_scale == 0) && return NaN
    predicted_normalized = predicted_centered ./ predicted_scale
    observed_normalized = observed_centered ./ observed_scale
    denominator = sqrt(sum(abs2, predicted_normalized) *
                       sum(abs2, observed_normalized))
    denominator == 0 && return NaN
    value = dot(predicted_normalized, observed_normalized) / denominator
    return clamp(value, -1.0, 1.0)
end

"""
    skill_score(predicted, observed, reference)

Skill score relative to reference model:
  SS = 1 - MSE(predicted) / MSE(reference)
SS > 0 means predicted is better than reference.
"""
function skill_score(predicted::AbstractVector, observed::AbstractVector,
                     reference::AbstractVector)
    _metric_float_vectors("skill_score", predicted, observed, reference)
    predicted_error = _metric_differences(
        "skill_score", predicted, observed,
    )
    reference_error = _metric_differences(
        "skill_score", reference, observed,
    )
    rmse_pred = _stable_root_mean_square(predicted_error)
    rmse_ref = _stable_root_mean_square(reference_error)
    # A zero-variance reference (reference == observed) leaves the skill score
    # undefined; flag it as NaN rather than dividing by a 1e-20 floor that would
    # return huge finite garbage. Matches the explicit NaN guard in correlation().
    return _finite_efficiency("skill_score", rmse_pred, rmse_ref)
end

"""
    prediction_efficiency(predicted, observed)

Nash-Sutcliffe prediction efficiency (PE):
  PE = 1 - Σ(pred - obs)² / Σ(obs - mean(obs))²
PE = 1 is perfect, PE = 0 is no better than mean, PE < 0 is worse.
"""
function prediction_efficiency(predicted::AbstractVector,
                               observed::AbstractVector)
    _metric_float_vectors("prediction_efficiency", predicted, observed)
    residual = _metric_differences(
        "prediction_efficiency", predicted, observed,
    )
    centered = _metric_centered("prediction_efficiency", observed)
    residual_rmse = _stable_root_mean_square(residual)
    centered_rmse = _stable_root_mean_square(centered)
    # A zero-variance observation (constant target) leaves PE undefined; flag it
    # as NaN rather than dividing by a 1e-20 floor that would return huge finite
    # garbage. Matches the explicit NaN guard in correlation().
    return _finite_efficiency(
        "prediction_efficiency", residual_rmse, centered_rmse,
    )
end

"""
    metrics_summary(predicted, observed; name="Model")

Compute all metrics, return as NamedTuple.
"""
function metrics_summary(predicted::AbstractVector, observed::AbstractVector;
                         name::String="Model")
    return (
        name = name,
        rmse = rmse(predicted, observed),
        mae = mae(predicted, observed),
        corr = correlation(predicted, observed),
        pe = prediction_efficiency(predicted, observed),
        n_points = length(predicted)
    )
end

"""
    wilcoxon_signed_rank_p(differences)

Two-sided paired Wilcoxon signed-rank test via the normal approximation.
`differences` are the paired differences `x .- y`. Zero differences are dropped;
tied magnitudes receive average ranks with the standard tie correction to the
variance; a continuity correction of 0.5 is applied. Returns a NamedTuple
`(p, z, w, n)` where `w = min(W⁺, W⁻)` and `n` is the count of nonzero
differences. Returns `p = NaN` when no nonzero differences remain.

This makes the significance statistics reported in the manuscript reproducible
from the per-storm RMSE artifacts.
"""
function wilcoxon_signed_rank_p(differences::AbstractVector{<:Real})
    all(value -> !(value isa Bool) && isfinite(value), differences) ||
        throw(ArgumentError("Wilcoxon differences must contain only finite values"))
    d = Float64[x for x in differences if x != 0.0]
    all(isfinite, d) ||
        throw(ArgumentError("Wilcoxon differences exceed the supported Float64 range"))
    n = length(d)
    n == 0 && return (p = NaN, z = NaN, w = NaN, n = 0)

    mag = abs.(d)
    order = sortperm(mag)
    ranks = zeros(n)
    i = 1
    while i <= n
        j = i
        while j < n && mag[order[j + 1]] == mag[order[i]]
            j += 1
        end
        avg = (i + j) / 2          # average rank for ties
        for k in i:j
            ranks[order[k]] = avg
        end
        i = j + 1
    end

    w_plus = sum(ranks[d .> 0])
    w_minus = sum(ranks[d .< 0])
    w = min(w_plus, w_minus)

    counts = Dict{Float64,Int}()
    for x in mag
        counts[x] = get(counts, x, 0) + 1
    end
    μ, σ2 = _wilcoxon_null_moments(n, values(counts))
    σ2 <= 0 && return (p = NaN, z = NaN, w = w, n = n)

    # `w = min(W⁺, W⁻)` cannot exceed μ. Clamp the continuity-corrected
    # statistic at zero for the exactly balanced case, then evaluate the
    # two-sided lower tail through erfc. The algebraically equivalent
    # `1 + erf(z/√2)` catastrophically cancels to zero for large negative z.
    z = min((w - μ + 0.5) / sqrt(σ2), 0.0)
    p = erfc(-z / sqrt(2.0))
    return (p = clamp(p, 0.0, 1.0), z = z, w = w, n = n)
end

function _wilcoxon_null_moments(n::Int, tie_counts)
    n >= 1 || throw(ArgumentError("Wilcoxon moment count must be positive"))
    counts = collect(tie_counts)
    all(count -> count isa Integer && !(count isa Bool) && count >= 1, counts) ||
        throw(ArgumentError("Wilcoxon tie counts must be positive integers"))
    sum(Int128, counts) == Int128(n) || throw(ArgumentError(
        "Wilcoxon tie counts must sum to the nonzero sample count",
    ))

    # Form the common-denominator numerator exactly before the single Float64
    # conversion.  The prior subtraction of two rounded O(n³) Float64 values
    # lost a unit in the last place for large tied samples. Int128 covers every
    # practically allocatable Julia vector; retain a BigInt fallback for the
    # complete Int API domain.
    function variance_numerator(::Type{T}) where {T<:Integer}
        n_exact = T(n)
        base = Base.Checked.checked_mul(
            Base.Checked.checked_mul(
                Base.Checked.checked_mul(T(2), n_exact), n_exact + T(1),
            ),
            Base.Checked.checked_add(
                Base.Checked.checked_mul(T(2), n_exact), T(1),
            ),
        )
        correction = zero(T)
        for count in counts
            count == 1 && continue
            value = T(count)
            term = Base.Checked.checked_mul(
                Base.Checked.checked_mul(value, value - T(1)), value + T(1),
            )
            correction = Base.Checked.checked_add(correction, term)
        end
        return Base.Checked.checked_sub(base, correction)
    end

    numerator = try
        variance_numerator(Int128)
    catch error
        error isa OverflowError || rethrow()
        n_big = BigInt(n)
        2n_big * (n_big + 1) * (2n_big + 1) -
            sum((BigInt(count)^3 - BigInt(count) for count in counts); init=BigInt(0))
    end
    mean_numerator = Int128(n) * (Int128(n) + 1)
    μ = Float64(mean_numerator) / 4.0
    σ2 = Float64(numerator) / 48.0
    isfinite(μ) && isfinite(σ2) || throw(ArgumentError(
        "Wilcoxon null moments exceeded the supported range",
    ))
    return μ, σ2
end
