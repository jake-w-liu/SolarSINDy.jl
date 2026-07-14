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
# but do not wire EKF output into dashboard, daemon, alerting, or V2 forecast columns
# without a new promotion report that passes the V2 readiness gate.

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
    all(isfinite, xi_full) || throw(ArgumentError("xi_full must contain only finite values"))
    isfinite(dst0) || throw(ArgumentError("dst0 must be finite"))
    isfinite(R) && R > 0 || throw(ArgumentError("R must be finite and positive"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    all(x -> isfinite(x) && x >= 0, (dst_var0, coeff_var0, q_dst, q_coeff)) ||
        throw(ArgumentError("variances must be finite and nonnegative"))

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
    all(b -> !isnan(b[1]) && !isnan(b[2]) && b[1] <= b[2] &&
             b[1] < Inf && b[2] > -Inf, bounds) ||
        throw(ArgumentError(
            "each coeff_bound must be ordered, non-NaN, and contain a finite value",
        ))

    xi_base = collect(Float64, xi_full)
    all(isfinite, xi_base) ||
        throw(ArgumentError("xi_full exceeds the supported Float64 range"))
    dst_initial = Float64(dst0)
    isfinite(dst_initial) ||
        throw(ArgumentError("dst0 exceeds the supported Float64 range"))
    variance_values = Float64.((dst_var0, coeff_var0, q_dst, q_coeff, R, dt))
    all(isfinite, variance_values) ||
        throw(ArgumentError("assimilation parameters exceed the supported Float64 range"))
    dst_var, coeff_var, dst_noise, coeff_noise, observation_var, step = variance_values
    mean = vcat(dst_initial, Float64[xi_base[i] for i in idx])
    for j in 1:m                                  # respect the box at initialisation too
        mean[j + 1] = clamp(mean[j + 1], bounds[j][1], bounds[j][2])
    end
    all(isfinite, mean) || throw(ArgumentError(
        "initial assimilation mean must remain finite after coefficient projection",
    ))
    cov = Matrix{Float64}(I, m + 1, m + 1)
    cov[1, 1] = dst_var
    for j in 1:m
        cov[j + 1, j + 1] = coeff_var
    end
    Q = zeros(m + 1, m + 1)
    Q[1, 1] = dst_noise
    for j in 1:m
        Q[j + 1, j + 1] = coeff_noise
    end
    return AssimilationFilter(lib, xi_base, idx, mean, cov,
                              Q, observation_var, step, bounds)
end

current_dst(f::AssimilationFilter) = f.mean[1]
current_coeffs(f::AssimilationFilter) = f.mean[2:end]
dst_variance(f::AssimilationFilter) = f.cov[1, 1]

# `AssimilationFilter` is intentionally mutable so long-running experiments can
# checkpoint and restore its numerical state.  That also means callers can
# invalidate relationships that the EKF kernels rely on (several hot loops use
# `@inbounds`).  Validate the complete structural/numerical contract at every
# public mutation boundary so a damaged checkpoint fails closed instead of
# indexing unrelated memory or silently advancing an incoherent filter.
function _validate_assimilation_filter(f::AssimilationFilter)
    n_terms = _validate_candidate_library(f.lib)

    length(f.xi_base) == n_terms || throw(DimensionMismatch(
        "assimilation base-coefficient length $(length(f.xi_base)) != library length $n_terms",
    ))
    all(isfinite, f.xi_base) || throw(ArgumentError(
        "assimilation base coefficients must be finite",
    ))

    m = length(f.adapt_idx)
    all(i -> 1 <= i <= n_terms, f.adapt_idx) || throw(ArgumentError(
        "assimilation adapted-coefficient index is out of range",
    ))
    length(unique(f.adapt_idx)) == m || throw(ArgumentError(
        "assimilation adapted-coefficient indices must be unique",
    ))
    length(f.bounds) == m || throw(DimensionMismatch(
        "assimilation coefficient-bound count $(length(f.bounds)) != adapted-coefficient count $m",
    ))

    state_dim = m + 1
    length(f.mean) == state_dim || throw(DimensionMismatch(
        "assimilation mean length $(length(f.mean)) != expected state dimension $state_dim",
    ))
    all(isfinite, f.mean) || throw(ArgumentError(
        "assimilation mean must be finite",
    ))
    size(f.cov) == (state_dim, state_dim) || throw(DimensionMismatch(
        "assimilation covariance size $(size(f.cov)) != ($state_dim, $state_dim)",
    ))
    size(f.Q) == (state_dim, state_dim) || throw(DimensionMismatch(
        "assimilation process-noise size $(size(f.Q)) != ($state_dim, $state_dim)",
    ))
    all(isfinite, f.cov) || throw(ArgumentError(
        "assimilation covariance must be finite",
    ))
    all(isfinite, f.Q) || throw(ArgumentError(
        "assimilation process noise must be finite",
    ))
    isfinite(f.R) && f.R > 0 || throw(ArgumentError(
        "assimilation observation variance must be finite and positive",
    ))
    isfinite(f.dt) && f.dt > 0 || throw(ArgumentError(
        "assimilation time step must be finite and positive",
    ))

    for j in 1:m
        lo, hi = f.bounds[j]
        !isnan(lo) && !isnan(hi) && lo <= hi && lo < Inf && hi > -Inf ||
            throw(ArgumentError(
                "each assimilation coefficient bound must be ordered, non-NaN, and contain a finite value",
            ))
        lo <= f.mean[j + 1] <= hi || throw(ArgumentError(
            "adapted assimilation coefficient $j lies outside its configured bound",
        ))
    end
    return f
end

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
    _evaluate_point_vector_unchecked!(
        θ, f.lib, dst_star, drivers.V, drivers.Bz, drivers.By,
        drivers.n, drivers.Pdyn,
    )
    return dot(θ, ξ), θ
end

"""
    assimilation_predict!(f, drivers)

EKF time update: advance the augmented state one `dt` step under the discovered
ODE with the current coefficients held by the filter. `drivers` is a NamedTuple
`(V, Bz, By, n, Pdyn)`. Coefficients follow a random walk (identity mean).
"""
function assimilation_predict!(f::AssimilationFilter, drivers)
    _validate_assimilation_filter(f)
    m = length(f.adapt_idx)
    dst = f.mean[1]
    ξ = _xi_with_adapted(f)

    # Normalize the public `Real` driver contract to the Float64 point evaluator.
    # Non-finite inputs retain the documented coast policy; finite but physically
    # impossible speed, density, or pressure fail closed like the forecast path.
    raw_drivers = (drivers.V, drivers.Bz, drivers.By, drivers.n, drivers.Pdyn)
    drivers64 = (
        V=Float64(drivers.V),
        Bz=Float64(drivers.Bz),
        By=Float64(drivers.By),
        n=Float64(drivers.n),
        Pdyn=Float64(drivers.Pdyn),
    )
    all(isfinite, raw_drivers) && !all(isfinite, drivers64) &&
        throw(ArgumentError("finite drivers exceed the supported Float64 range"))

    # Non-finite-driver guard (mirrors the NaN-observation gap policy in
    # assimilation_update!): a single bad driver would otherwise propagate
    # through ddst and the Jacobian and permanently corrupt the EKF state.
    # Policy: predict-only coast — hold the mean and inflate the covariance by
    # the process noise Q only, exactly as a missing observation grows variance.
    drivers_finite = all(isfinite, drivers64)
    if drivers_finite && !(drivers64.V >= 0 && drivers64.n >= 0 && drivers64.Pdyn >= 0)
        throw(ArgumentError("V, density, and dynamic pressure must be nonnegative"))
    end
    if !drivers_finite
        new_cov = f.cov + f.Q        # coast: mean unchanged, covariance grows by Q
        all(isfinite, new_cov) ||
            throw(ArgumentError("assimilation covariance became non-finite while coasting"))
        _symmetrize!(new_cov)
        f.cov = new_cov
        return f
    end
    ddst, θ = _ddst(f, dst, ξ, drivers64)
    isfinite(ddst) ||
        throw(ArgumentError("assimilation derivative became non-finite"))

    # Compute the full update before mutating the filter, so a numerical failure
    # does not leave a half-updated state.
    new_dst = dst + f.dt * ddst
    isfinite(new_dst) || throw(ArgumentError("assimilation state became non-finite"))
    # (coefficient means unchanged)

    # Jacobian F (size (m+1)×(m+1)).
    F = Matrix{Float64}(I, m + 1, m + 1)
    # ∂Dst*_{k+1}/∂Dst*_k via central difference of dDst(Dst*).
    h = 1.0e-4 * max(abs(dst), 1.0)
    dplus, _ = _ddst(f, dst + h, ξ, drivers64)
    dminus, _ = _ddst(f, dst - h, ξ, drivers64)
    F[1, 1] = 1.0 + f.dt * (dplus - dminus) / (2h)
    # ∂Dst*_{k+1}/∂ξ_adapt_j = dt · Θ_j  (dDst linear in ξ).
    for j in 1:m
        F[1, j + 1] = f.dt * θ[f.adapt_idx[j]]
    end

    new_cov = F * f.cov * F' + f.Q
    all(isfinite, new_cov) ||
        throw(ArgumentError("assimilation covariance became non-finite"))
    _symmetrize!(new_cov)
    f.mean[1] = new_dst
    f.cov = new_cov
    return f
end

"""
    assimilation_update!(f, observed_dst_star)

EKF measurement update from an observed pressure-corrected Dst (observes the
state's Dst* component only). No-op-safe: pass `NaN` to skip the update (a
predict-only step when no observation is available).
"""
function assimilation_update!(f::AssimilationFilter, observed_dst_star::Real)
    _validate_assimilation_filter(f)
    isnan(observed_dst_star) && return f
    isfinite(observed_dst_star) ||
        throw(ArgumentError("observed Dst* must be finite or NaN"))
    observation = Float64(observed_dst_star)
    isfinite(observation) ||
        throw(ArgumentError("observed Dst* exceeds the supported Float64 range"))
    n = length(f.mean)
    # H = [1, 0, ..., 0]; innovation S = P11 + R.
    S = f.cov[1, 1] + f.R
    isfinite(S) && S > 0 ||
        throw(ArgumentError("assimilation innovation variance must be finite and positive"))
    K = f.cov[:, 1] ./ S                      # Kalman gain (n-vector)
    innovation = observation - f.mean[1]
    new_mean = f.mean .+ K .* innovation
    all(isfinite, new_mean) ||
        throw(ArgumentError("assimilation mean became non-finite during update"))
    # Project the adapted coefficients into their constraint box (no-op when unconstrained). This is a
    # projected/constrained Kalman update: it keeps each adapted coefficient inside its stability range
    # (e.g. the decay rate held < 0), preventing the filter from drifting into a dynamically unstable ODE
    # that would diverge under free-running multi-step rollout.
    @inbounds for j in 1:length(f.adapt_idx)
        lo, hi = f.bounds[j]
        (isfinite(lo) || isfinite(hi)) &&
            (new_mean[j + 1] = clamp(new_mean[j + 1], lo, hi))
    end
    # Joseph form preserves symmetry and positive semidefiniteness substantially
    # better than the former one-sided `P -= K*P[1,:]` update.
    update = Matrix{Float64}(I, n, n)
    update[:, 1] .-= K
    new_cov = update * f.cov * update' + f.R .* (K * K')
    all(isfinite, new_cov) ||
        throw(ArgumentError("assimilation covariance became non-finite during update"))
    _symmetrize!(new_cov)
    f.mean = new_mean
    f.cov = new_cov
    return f
end

function _symmetrize!(P::Matrix{Float64})
    size(P, 1) == size(P, 2) || throw(DimensionMismatch(
        "covariance must be square before symmetrization",
    ))
    all(isfinite, P) || throw(ArgumentError(
        "covariance must be finite before symmetrization",
    ))
    @inbounds for i in 1:size(P, 1), j in 1:i-1
        left = P[i, j]
        right = P[j, i]
        overflow_risk = signbit(left) == signbit(right) &&
                        abs(left) > floatmax(Float64) - abs(right)
        avg = overflow_risk ? left / 2 + right / 2 : (left + right) / 2
        P[i, j] = avg
        P[j, i] = avg
    end
    all(isfinite, P) || throw(ArgumentError(
        "covariance became non-finite during symmetrization",
    ))
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
