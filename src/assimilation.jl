# Online data assimilation for the discovered sparse Dst* ODE (N2).
#
# The discovered governing equation dDst*/dt = Θ(x)·ξ is interpretable but its
# coefficients are fixed at fit time, so it cannot track regime or solar-cycle
# drift operationally. This module runs an Extended Kalman Filter over an
# augmented state [Dst*; ξ_adapt], where ξ_adapt is a SMALL, physically-motivated
# subset of coefficients (e.g. the injection scale and the decay rate). The
# sparse term structure is never changed — only the magnitudes of the chosen
# coefficients adapt — so interpretability is preserved.
#
# State-space model (forward-Euler step, dt hours):
#   Dst*_{k+1} = Dst*_k + dt · Θ(x_k, Dst*_k)·ξ(θ_k) + process noise
#   θ_{k+1}    = θ_k + random walk           (slow coefficient drift)
#   y_k        = Dst*_k + measurement noise  (observed pressure-corrected Dst)
#
# The dynamics are bilinear in (state, coefficients), hence the EKF: the
# coefficient–state coupling is exact (dDst is linear in ξ, so ∂/∂ξ_j = Θ_j),
# and the state self-derivative ∂(Θξ)/∂Dst* is obtained by a central difference
# of the library evaluation (robust to the term set, no per-term hand Jacobian).
#
# OPERATIONAL STATUS: retired from the live forecast path. The EKF predict/update math
# remains useful research infrastructure, but both deployment candidates failed the later
# promotion gate recorded in live_forecasts/EKF_V3_DECISION.md:
#   * decay-only constrained EKF: no lead/regime beat the stronger of {v2, persistence}
#     with a positive 95% CI on the causal G4/G5 storm replay;
#   * injection-adaptive EKF: worse than decay-only and robustly not promotable across
#     the tested injection-walk variance sweep.
# Keep this module for reproducible negative evidence and future assimilation experiments,
# but do not wire EKF output into dashboard, daemon, alerting, or served forecast columns
# without a new promotion report that passes the industrial gate.

"""
    AssimilationFilter

Mutable EKF state for online coefficient/state assimilation of the sparse Dst*
ODE. `mean` is the augmented state `[Dst*; ξ_adapt]` (length `1 + m`); `cov` is
its covariance; `Q` the process-noise covariance; `R` the scalar observation
variance; `dt` the step in hours. `xi_base` holds the full discovered coefficient
vector with the adapted entries overwritten by `mean[2:end]` at each step.
"""
mutable struct AssimilationFilter
    lib::CandidateLibrary
    xi_base::Vector{Float64}
    adapt_idx::Vector{Int}
    mean::Vector{Float64}
    cov::Matrix{Float64}
    Q::Matrix{Float64}
    R::Float64
    dt::Float64
    bounds::Vector{Tuple{Float64,Float64}}   # per-adapted-coeff (lo,hi) box; (-Inf,Inf)=unconstrained
end

"""
    init_assimilation(lib, xi_full, adapt_idx, dst0; kwargs...)

Build an [`AssimilationFilter`](@ref). `adapt_idx` are the indices of `xi_full`
to adapt online. Keyword arguments set the initial uncertainties and noise:
`dst_var0` (initial Dst* variance), `coeff_var0` (initial adapted-coefficient
variance), `q_dst` (per-step Dst* process variance), `q_coeff` (per-step
coefficient random-walk variance), `R` (observation variance), `dt`.
"""
function init_assimilation(lib::CandidateLibrary, xi_full::AbstractVector{<:Real},
                           adapt_idx::AbstractVector{<:Integer}, dst0::Real;
                           dst_var0::Real=25.0, coeff_var0::Real=1.0e-2,
                           q_dst::Real=1.0, q_coeff::Real=1.0e-6,
                           R::Real=4.0, dt::Real=1.0, coeff_bounds=nothing)
    length(xi_full) == length(lib) ||
        throw(DimensionMismatch("xi_full length $(length(xi_full)) != library length $(length(lib))"))
    all(i -> 1 <= i <= length(xi_full), adapt_idx) ||
        throw(ArgumentError("adapt_idx out of range"))
    length(unique(adapt_idx)) == length(adapt_idx) ||
        throw(ArgumentError("adapt_idx must be unique"))
    R > 0 || throw(ArgumentError("R must be positive"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    (dst_var0 >= 0 && coeff_var0 >= 0 && q_dst >= 0 && q_coeff >= 0) ||
        throw(ArgumentError("variances must be nonnegative"))

    idx = collect(Int, adapt_idx)
    m = length(idx)
    # Optional inequality constraints on the adapted coefficients (stability box). Default
    # unconstrained, so existing behaviour is byte-for-byte unchanged. When supplied,
    # `coeff_bounds[j]=(lo,hi)` constrains adapted coefficient j (matching adapt_idx order); the
    # mean is projected into the box after every measurement update (a projected/constrained KF).
    bounds = if coeff_bounds === nothing
        [(-Inf, Inf) for _ in 1:m]
    else
        length(coeff_bounds) == m ||
            throw(ArgumentError("coeff_bounds length $(length(coeff_bounds)) != adapted-coefficient count $m"))
        all(b -> length(b) == 2, coeff_bounds) ||
            throw(ArgumentError("each coeff_bound must be a 2-tuple (lo, hi)"))
        [(Float64(b[1]), Float64(b[2])) for b in coeff_bounds]
    end
    all(b -> b[1] <= b[2], bounds) || throw(ArgumentError("each coeff_bound must satisfy lo <= hi"))

    mean = vcat(Float64(dst0), Float64[xi_full[i] for i in idx])
    for j in 1:m                                  # respect the box at initialisation too
        mean[j + 1] = clamp(mean[j + 1], bounds[j][1], bounds[j][2])
    end
    cov = Matrix{Float64}(I, m + 1, m + 1)
    cov[1, 1] = dst_var0
    for j in 1:m
        cov[j + 1, j + 1] = coeff_var0
    end
    Q = zeros(m + 1, m + 1)
    Q[1, 1] = q_dst
    for j in 1:m
        Q[j + 1, j + 1] = q_coeff
    end
    return AssimilationFilter(lib, collect(Float64, xi_full), idx, mean, cov,
                              Q, Float64(R), Float64(dt), bounds)
end

current_dst(f::AssimilationFilter) = f.mean[1]
current_coeffs(f::AssimilationFilter) = f.mean[2:end]
dst_variance(f::AssimilationFilter) = f.cov[1, 1]

# Full coefficient vector with the adapted entries set from the filter mean.
function _xi_with_adapted(f::AssimilationFilter)
    ξ = copy(f.xi_base)
    for (j, i) in enumerate(f.adapt_idx)
        ξ[i] = f.mean[j + 1]
    end
    return ξ
end

# dDst*/dt at a given Dst* and drivers, using a supplied coefficient vector.
function _ddst(f::AssimilationFilter, dst_star::Float64, ξ::Vector{Float64}, drivers)
    θ = Vector{Float64}(undef, length(f.lib))
    _evaluate_point_vector!(θ, f.lib, dst_star, drivers.V, drivers.Bz, drivers.By,
                            drivers.n, drivers.Pdyn)
    return dot(θ, ξ), θ
end

"""
    assimilation_predict!(f, drivers)

EKF time update: advance the augmented state one `dt` step under the discovered
ODE with the current coefficients held by the filter. `drivers` is a NamedTuple
`(V, Bz, By, n, Pdyn)`. Coefficients follow a random walk (identity mean).
"""
function assimilation_predict!(f::AssimilationFilter, drivers)
    m = length(f.adapt_idx)
    dst = f.mean[1]
    ξ = _xi_with_adapted(f)

    # Non-finite-driver guard (mirrors the NaN-observation gap policy in
    # assimilation_update!): a single bad driver would otherwise propagate
    # through ddst and the Jacobian and permanently corrupt the EKF state.
    # Policy: predict-only coast — hold the mean and inflate the covariance by
    # the process noise Q only, exactly as a missing observation grows variance.
    drivers_finite = isfinite(drivers.V) && isfinite(drivers.Bz) &&
                     isfinite(drivers.By) && isfinite(drivers.n) &&
                     isfinite(drivers.Pdyn)
    if drivers_finite
        ddst, θ = _ddst(f, dst, ξ, drivers)
    else
        ddst = NaN
        θ = Float64[]
    end
    if !(drivers_finite && isfinite(ddst))
        f.cov = f.cov + f.Q          # coast: mean unchanged, covariance grows by Q
        _symmetrize!(f.cov)
        return f
    end

    # Mean propagation.
    f.mean[1] = dst + f.dt * ddst
    # (coefficient means unchanged)

    # Jacobian F (size (m+1)×(m+1)).
    F = Matrix{Float64}(I, m + 1, m + 1)
    # ∂Dst*_{k+1}/∂Dst*_k via central difference of dDst(Dst*).
    h = 1.0e-4 * max(abs(dst), 1.0)
    dplus, _ = _ddst(f, dst + h, ξ, drivers)
    dminus, _ = _ddst(f, dst - h, ξ, drivers)
    F[1, 1] = 1.0 + f.dt * (dplus - dminus) / (2h)
    # ∂Dst*_{k+1}/∂ξ_adapt_j = dt · Θ_j  (dDst linear in ξ).
    for j in 1:m
        F[1, j + 1] = f.dt * θ[f.adapt_idx[j]]
    end

    f.cov = F * f.cov * F' + f.Q
    _symmetrize!(f.cov)
    return f
end

"""
    assimilation_update!(f, observed_dst_star)

EKF measurement update from an observed pressure-corrected Dst (observes the
state's Dst* component only). No-op-safe: pass `NaN` to skip the update (a
predict-only step when no observation is available).
"""
function assimilation_update!(f::AssimilationFilter, observed_dst_star::Real)
    isfinite(observed_dst_star) || return f
    n = length(f.mean)
    # H = [1, 0, ..., 0]; innovation S = P11 + R.
    S = f.cov[1, 1] + f.R
    K = f.cov[:, 1] ./ S                      # Kalman gain (n-vector)
    innovation = Float64(observed_dst_star) - f.mean[1]
    f.mean .+= K .* innovation
    # Project the adapted coefficients into their constraint box (no-op when unconstrained). This is a
    # projected/constrained Kalman update: it keeps each adapted coefficient inside its stability range
    # (e.g. the decay rate held < 0), preventing the filter from drifting into a dynamically unstable ODE
    # that would diverge under free-running multi-step rollout.
    @inbounds for j in 1:length(f.adapt_idx)
        lo, hi = f.bounds[j]
        (isfinite(lo) || isfinite(hi)) && (f.mean[j + 1] = clamp(f.mean[j + 1], lo, hi))
    end
    # Joseph-free covariance update P = (I - K Hᵀ) P, then symmetrize.
    f.cov .-= K * f.cov[1, :]'
    _symmetrize!(f.cov)
    return f
end

function _symmetrize!(P::Matrix{Float64})
    @inbounds for i in 1:size(P, 1), j in 1:i-1
        avg = (P[i, j] + P[j, i]) / 2
        P[i, j] = avg
        P[j, i] = avg
    end
    return P
end

"""
    run_assimilation(f, drivers_seq, obs_seq)

Run the filter over a sequence: for each step, predict with `drivers_seq[k]` then
update with `obs_seq[k]` (use `NaN` for a gap). Returns a NamedTuple of vectors:
`dst` (filtered Dst*), `dst_var` (its variance), and `coeffs` (a Vector of the
adapted-coefficient estimates per step). The filter is advanced in place.
"""
function run_assimilation(f::AssimilationFilter, drivers_seq, obs_seq)
    length(drivers_seq) == length(obs_seq) ||
        throw(DimensionMismatch("drivers_seq and obs_seq must have equal length"))
    n = length(drivers_seq)
    dst = Vector{Float64}(undef, n)
    dst_var = Vector{Float64}(undef, n)
    coeffs = Vector{Vector{Float64}}(undef, n)
    for k in 1:n
        assimilation_predict!(f, drivers_seq[k])
        assimilation_update!(f, obs_seq[k])
        dst[k] = current_dst(f)
        dst_var[k] = dst_variance(f)
        coeffs[k] = current_coeffs(f)
    end
    return (; dst, dst_var, coeffs)
end
