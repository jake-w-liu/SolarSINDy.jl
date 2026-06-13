# Evaluation metrics for model comparison

using SpecialFunctions: erf

"""
    rmse(predicted, observed)

Root Mean Square Error.
"""
function rmse(predicted::AbstractVector, observed::AbstractVector)
    return sqrt(mean((predicted .- observed).^2))
end

"""
    correlation(predicted, observed)

Pearson correlation coefficient. Returns `NaN` for a degenerate (zero-variance)
input rather than letting `cor` emit a warning/`NaN` implicitly, so callers can
filter undefined correlations explicitly.
"""
function correlation(predicted::AbstractVector, observed::AbstractVector)
    (std(predicted) == 0 || std(observed) == 0) && return NaN
    return cor(predicted, observed)
end

"""
    skill_score(predicted, observed, reference)

Skill score relative to reference model:
  SS = 1 - MSE(predicted) / MSE(reference)
SS > 0 means predicted is better than reference.
"""
function skill_score(predicted::AbstractVector, observed::AbstractVector,
                     reference::AbstractVector)
    mse_pred = mean((predicted .- observed).^2)
    mse_ref = mean((reference .- observed).^2)
    return 1.0 - mse_pred / max(mse_ref, 1e-20)
end

"""
    prediction_efficiency(predicted, observed)

Nash-Sutcliffe prediction efficiency (PE):
  PE = 1 - Σ(pred - obs)² / Σ(obs - mean(obs))²
PE = 1 is perfect, PE = 0 is no better than mean, PE < 0 is worse.
"""
function prediction_efficiency(predicted::AbstractVector,
                               observed::AbstractVector)
    ss_res = sum((predicted .- observed).^2)
    ss_tot = sum((observed .- mean(observed)).^2)
    return 1.0 - ss_res / max(ss_tot, 1e-20)
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
    d = Float64[x for x in differences if isfinite(x) && x != 0.0]
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

    μ = n * (n + 1) / 4
    σ2 = n * (n + 1) * (2n + 1) / 24
    # Tie correction: subtract Σ(tⱼ³ - tⱼ)/48 over tied groups.
    counts = Dict{Float64,Int}()
    for x in mag
        counts[x] = get(counts, x, 0) + 1
    end
    σ2 -= sum(t^3 - t for t in values(counts)) / 48
    σ2 <= 0 && return (p = NaN, z = NaN, w = w, n = n)

    z = (w - μ + 0.5) / sqrt(σ2)     # continuity correction; w=min ⇒ z ≤ 0
    p = 2.0 * 0.5 * (1.0 + erf(z / sqrt(2.0)))   # two-sided
    return (p = clamp(p, 0.0, 1.0), z = z, w = w, n = n)
end
