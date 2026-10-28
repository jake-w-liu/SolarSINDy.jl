module DBDTUniversalPortfolio
export UniversalPortfolio, radius, update!

"""Uncorrected publisher Algorithm 1, Liu/Dobriban/Orabona, ICML 2026.

Sequential nonnegative scores only. Delay/interleaving is the caller's contract.
The log-wealth representation avoids arithmetic overflow in long streams.
"""
mutable struct UniversalPortfolio
    nominal::Float64
    resolved::Int
    misses::Int
    log_wealth::Float64
end

function UniversalPortfolio(; nominal=0.1)
    alpha = Float64(nominal)
    0 < alpha < 1 || throw(ArgumentError("nominal must remain in (0,1) in Float64"))
    return UniversalPortfolio(alpha, 0, 0, 0.0)
end

function radius(state::UniversalPortfolio)
    weight = (state.misses + 0.5) / (state.resolved + 1)
    weight <= state.nominal && return 0.0
    return exp(state.log_wealth + log(weight - state.nominal) -
               log(state.nominal) - log1p(-state.nominal))
end

function update!(state::UniversalPortfolio, score::Real)
    value = Float64(score)
    isfinite(value) && value >= 0 || throw(ArgumentError("score must be finite and nonnegative"))
    weight = (state.misses + 0.5) / (state.resolved + 1)
    miss = value > radius(state)
    increment = miss ? log(weight) - log(state.nominal) :
                       log1p(-weight) - log1p(-state.nominal)
    updated = state.log_wealth + increment
    isfinite(updated) || throw(ArgumentError("nonfinite portfolio wealth"))
    state.resolved += 1
    state.misses += miss
    state.log_wealth = updated
    return miss
end
end
