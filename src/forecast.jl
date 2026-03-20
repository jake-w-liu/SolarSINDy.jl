# Rolling forecaster with ensemble uncertainty quantification

"""
    ForecastResult

Single-step forecast output with ensemble statistics.
"""
struct ForecastResult
    t::DateTime
    dst_predicted::Float64       # primary model prediction [nT]
    dst_median::Float64          # ensemble median [nT]
    dst_ci_05::Float64           # 5th percentile (worst case) [nT]
    dst_ci_95::Float64           # 95th percentile (best case) [nT]
    dst_observed::Float64        # observed Dst* (NaN if unavailable) [nT]
end

"""
    ForecastState

Mutable state for rolling one-step-ahead forecasting.
"""
mutable struct ForecastState
    t_current::DateTime
    dst_current::Float64
    lib::CandidateLibrary
    ξ_primary::Vector{Float64}
    ξ_ensemble::Matrix{Float64}  # n_ensemble × n_terms
    dt::Float64
    history::Vector{ForecastResult}
end

"""
    init_forecast(; coefficients_csv, ensemble_csv, t0, dst0, dt=1.0)

Initialise a ForecastState from saved discovery results.

- `coefficients_csv`: path to real_sindy_discovery_coefficients.csv
- `ensemble_csv`: path to real_ensemble_inclusion.csv
- `t0`: initial DateTime (UTC)
- `dst0`: initial Dst* value [nT]
"""
function init_forecast(; coefficients_csv::String,
                         ensemble_csv::String,
                         t0::DateTime,
                         dst0::Float64,
                         dt::Float64=1.0)
    # Load primary coefficients
    coef_df = CSV.read(coefficients_csv, DataFrame)
    lib = build_solar_wind_library()
    term_names = get_term_names(lib)

    # Map CSV terms to library indices
    ξ_primary = zeros(length(lib))
    for row in eachrow(coef_df)
        idx = findfirst(==(row.term), term_names)
        if idx !== nothing
            ξ_primary[idx] = row.coefficient
        end
    end

    # Load ensemble coefficients — build matrix from median_coef for each ensemble member
    # The ensemble CSV has per-term statistics; we reconstruct ensemble spread
    # by sampling from [ci_025, ci_975] for terms with π > 0.9
    ens_df = CSV.read(ensemble_csv, DataFrame)
    n_ensemble = 500
    n_terms = length(lib)
    ξ_ensemble = zeros(n_ensemble, n_terms)
    rng = MersenneTwister(42)

    for row in eachrow(ens_df)
        idx = findfirst(==(row.term), term_names)
        idx === nothing && continue
        if row.inclusion_prob >= 0.9
            # Sample uniformly within CI for ensemble diversity
            lo, hi = row.ci_025, row.ci_975
            for i in 1:n_ensemble
                ξ_ensemble[i, idx] = lo + (hi - lo) * rand(rng)
            end
        end
    end

    return ForecastState(t0, dst0, lib, ξ_primary, ξ_ensemble, dt,
                         ForecastResult[])
end

"""
    _evaluate_point(lib, dst_star, V, Bz, By, n, Pdyn)

Evaluate library at a single space-time point. Returns 1×p matrix.
"""
function _evaluate_point(lib::CandidateLibrary,
                         dst_star::Float64, V::Float64,
                         Bz::Float64, By::Float64,
                         n_density::Float64, Pdyn::Float64)
    Bs = max(-Bz, 0.0)
    theta_c = atan(abs(By), Bz)
    BT = sqrt(By^2 + Bz^2)
    data = Dict{String,Vector{Float64}}(
        "V"        => [V],
        "Bs"       => [Bs],
        "n"        => [n_density],
        "Pdyn"     => [Pdyn],
        "Dst_star" => [dst_star],
        "theta_c"  => [theta_c],
        "BT"       => [BT],
        "By"       => [By],
        "Bz"       => [Bz],
    )
    return evaluate_library(lib, data)  # 1 × n_terms
end

"""
    step_forecast!(state, t, V, Bz, By, n, Pdyn; dst_observed=NaN)

Advance forecast by one time step using solar wind inputs.
Updates state.dst_current and appends to state.history.

Returns ForecastResult.
"""
function step_forecast!(state::ForecastState,
                        t::DateTime,
                        V::Float64, Bz::Float64, By::Float64,
                        n_density::Float64, Pdyn::Float64;
                        dst_observed::Float64=NaN)
    Θ_k = _evaluate_point(state.lib, state.dst_current,
                           V, Bz, By, n_density, Pdyn)

    # Primary prediction
    dDst = clamp((Θ_k * state.ξ_primary)[1], -200.0, 200.0)
    dst_next = clamp(state.dst_current + state.dt * dDst, -2000.0, 50.0)

    # Ensemble predictions
    n_ens = size(state.ξ_ensemble, 1)
    dst_ens = Vector{Float64}(undef, n_ens)
    for i in 1:n_ens
        dDst_i = clamp((Θ_k * @view(state.ξ_ensemble[i, :]))[1], -200.0, 200.0)
        dst_ens[i] = clamp(state.dst_current + state.dt * dDst_i, -2000.0, 50.0)
    end
    sort!(dst_ens)

    result = ForecastResult(
        t,
        dst_next,
        dst_ens[div(n_ens, 2)],           # median
        dst_ens[max(1, div(n_ens, 20))],   # 5th percentile
        dst_ens[min(n_ens, n_ens - div(n_ens, 20) + 1)],  # 95th percentile
        dst_observed,
    )

    # Update state
    if !isnan(dst_observed)
        state.dst_current = dst_observed   # anchor to observation when available
    else
        state.dst_current = dst_next
    end
    state.t_current = t
    push!(state.history, result)

    return result
end

"""
    forecast_ahead(state, V, Bz, By, n, Pdyn, n_hours)

Multi-hour forecast assuming persistence of current solar wind conditions.
Does NOT modify state. Returns Vector{ForecastResult}.
"""
function forecast_ahead(state::ForecastState,
                        V::Float64, Bz::Float64, By::Float64,
                        n_density::Float64, Pdyn::Float64,
                        n_hours::Int)
    results = ForecastResult[]
    dst_curr = state.dst_current
    t_curr = state.t_current

    n_ens = size(state.ξ_ensemble, 1)
    dst_ens_curr = fill(dst_curr, n_ens)

    for h in 1:n_hours
        t_next = t_curr + Hour(1)

        Θ_k = _evaluate_point(state.lib, dst_curr, V, Bz, By, n_density, Pdyn)

        # Primary
        dDst = clamp((Θ_k * state.ξ_primary)[1], -200.0, 200.0)
        dst_next = clamp(dst_curr + state.dt * dDst, -2000.0, 50.0)

        # Ensemble (each starts from its own previous prediction)
        for i in 1:n_ens
            Θ_i = _evaluate_point(state.lib, dst_ens_curr[i],
                                   V, Bz, By, n_density, Pdyn)
            dDst_i = clamp((Θ_i * @view(state.ξ_ensemble[i, :]))[1], -200.0, 200.0)
            dst_ens_curr[i] = clamp(dst_ens_curr[i] + state.dt * dDst_i, -2000.0, 50.0)
        end
        sorted_ens = sort(dst_ens_curr)

        push!(results, ForecastResult(
            t_next, dst_next,
            sorted_ens[div(n_ens, 2)],
            sorted_ens[max(1, div(n_ens, 20))],
            sorted_ens[min(n_ens, n_ens - div(n_ens, 20) + 1)],
            NaN,
        ))

        dst_curr = dst_next
        t_curr = t_next
    end

    return results
end
