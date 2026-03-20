# SINDy algorithm: Sequential Thresholded Least Squares (STLSQ)

"""
    stlsq(Θ, dx; λ=0.1, max_iter=25, normalize=true)

Sequential Thresholded Least Squares (STLSQ) for sparse regression.

Given library matrix Θ (n × p) and target derivative dx (n,),
find sparse coefficient vector ξ such that dx ≈ Θξ.

Algorithm (Brunton et al., 2016):
1. ξ = Θ \\ dx  (least squares)
2. Set |ξ_j| < λ to zero
3. Re-solve least squares on remaining terms
4. Repeat until convergence

Returns coefficient vector ξ.
"""
function stlsq(Θ::AbstractMatrix, dx::AbstractVector;
                λ::Real=0.1, max_iter::Int=25, normalize::Bool=true)
    n, p = size(Θ)
    @assert length(dx) == n "Dimension mismatch: Θ has $n rows, dx has $(length(dx))"

    # Optional column normalization for better conditioning
    if normalize
        col_norms = [norm(Θ[:, j]) for j in 1:p]
        col_norms[col_norms .== 0] .= 1.0
        Θn = Θ ./ col_norms'
    else
        Θn = Θ
        col_norms = ones(p)
    end

    # Initial least squares
    ξ = Θn \ dx

    for iter in 1:max_iter
        # Threshold small coefficients
        small_inds = abs.(ξ) .< λ
        ξ[small_inds] .= 0.0

        # Find active (non-zero) indices
        active = findall(.!small_inds)
        if isempty(active)
            break
        end

        # Re-solve on active terms only
        ξ_new = zeros(p)
        ξ_new[active] = Θn[:, active] \ dx

        # Check convergence
        if norm(ξ_new - ξ) / max(norm(ξ), 1e-10) < 1e-8
            ξ = ξ_new
            break
        end
        ξ = ξ_new
    end

    # Undo normalization
    if normalize
        ξ ./= col_norms
    end

    return ξ
end

"""
    sindy_discover(data::Dict, lib::CandidateLibrary, target_derivative::Vector;
                   λ=0.1, normalize=true)

Full SINDy discovery pipeline.
Returns (ξ, active_terms, Θ) where active_terms maps indices to names.
"""
function sindy_discover(data::Dict{String,Vector{Float64}},
                        lib::CandidateLibrary,
                        target_derivative::AbstractVector;
                        λ::Real=0.1, normalize::Bool=true)
    Θ = evaluate_library(lib, data)
    ξ = stlsq(Θ, target_derivative; λ=λ, normalize=normalize)

    # Extract active terms
    active_idx = findall(abs.(ξ) .> 0)
    active_terms = Dict{String,Float64}()
    for idx in active_idx
        active_terms[lib.names[idx]] = ξ[idx]
    end

    return ξ, active_terms, Θ
end

"""
    ensemble_sindy(data, lib, target; λ=0.1, n_models=100,
                   subsample_frac=0.8, seed=42)

Ensemble SINDy via bagging: subsample data, run STLSQ, aggregate.
Returns median coefficients and inclusion probabilities.
"""
function ensemble_sindy(data::Dict{String,Vector{Float64}},
                        lib::CandidateLibrary,
                        target::AbstractVector;
                        λ::Real=0.1, n_models::Int=500,
                        subsample_frac::Real=0.8, seed::Int=42)
    rng = MersenneTwister(seed)
    Θ = evaluate_library(lib, data)
    n, p = size(Θ)
    n_sub = round(Int, n * subsample_frac)

    all_ξ = zeros(p, n_models)
    for m in 1:n_models
        idx = sort(randperm(rng, n)[1:n_sub])
        all_ξ[:, m] = stlsq(Θ[idx, :], target[idx]; λ=λ, normalize=true)
    end

    # Inclusion probability: fraction of models where coefficient is nonzero
    inclusion_prob = vec(mean(all_ξ .!= 0, dims=2))

    # Median coefficient (over nonzero values)
    median_ξ = zeros(p)
    for j in 1:p
        nonzero = all_ξ[j, all_ξ[j, :] .!= 0]
        if !isempty(nonzero)
            median_ξ[j] = median(nonzero)
        end
    end

    return median_ξ, inclusion_prob, all_ξ
end

"""
    sindy_predict(ξ, lib, data)

Predict dDst/dt using discovered coefficients ξ and library.
"""
function sindy_predict(ξ::AbstractVector, lib::CandidateLibrary,
                       data::Dict{String,Vector{Float64}})
    Θ = evaluate_library(lib, data)
    return Θ * ξ
end

"""
    simulate_sindy(ξ, lib, swd, dt; Dst0=swd.Dst_star[1])

Forward-integrate the SINDy-discovered ODE using Forward Euler
with solar wind drivers at each time step.

Applies identical clamping as `simulate_burton` and `simulate_obrien`:
- Derivative clamping: |dDst*/dt| ≤ 200 nT/hr
- State bounds: -2000 ≤ Dst* ≤ 50 nT

Arguments:
- `ξ`: coefficient vector from SINDy discovery
- `lib`: CandidateLibrary used in discovery
- `swd`: SolarWindData with solar wind drivers
- `dt`: time step in hours
- `Dst0`: initial Dst* value (defaults to first value in swd)

Returns Dst* time series (Vector{Float64}).
"""
function simulate_sindy(ξ::AbstractVector, lib::CandidateLibrary,
                        swd::SolarWindData, dt::Real;
                        Dst0::Real=swd.Dst_star[1])
    n_pts = length(swd.t)
    Dst_pred = zeros(n_pts)
    Dst_pred[1] = Dst0

    for k in 1:(n_pts - 1)
        Bs_k = max(-swd.Bz[k], 0.0)
        theta_c_k = atan(abs(swd.By[k]), swd.Bz[k])
        BT_k = sqrt(swd.By[k]^2 + swd.Bz[k]^2)

        point_data = Dict{String,Vector{Float64}}(
            "V"        => [swd.V[k]],
            "Bs"       => [Bs_k],
            "n"        => [swd.n[k]],
            "Pdyn"     => [swd.Pdyn[k]],
            "Dst_star" => [Dst_pred[k]],
            "theta_c"  => [theta_c_k],
            "BT"       => [BT_k],
            "By"       => [swd.By[k]],
            "Bz"       => [swd.Bz[k]]
        )

        Θ_k = evaluate_library(lib, point_data)  # 1 × n_terms
        dDst = (Θ_k * ξ)[1]

        # Clamp to prevent numerical blow-up (same bounds as baseline models)
        dDst = clamp(dDst, -200.0, 200.0)
        Dst_pred[k+1] = clamp(Dst_pred[k] + dt * dDst, -2000.0, 50.0)
    end

    return Dst_pred
end

"""
    sweep_lambda(Θ, dx, lambdas; normalize=true)

Sweep sparsity threshold λ, return (n_terms, RMSE) for each.
Useful for Pareto front of parsimony vs. accuracy.
"""
function sweep_lambda(Θ::AbstractMatrix, dx::AbstractVector,
                      lambdas::AbstractVector; normalize::Bool=true)
    results = Vector{NamedTuple{(:λ, :n_terms, :rmse, :ξ),
                                 Tuple{Float64, Int, Float64, Vector{Float64}}}}()
    for λ in lambdas
        ξ = stlsq(Θ, dx; λ=λ, normalize=normalize)
        n_terms = count(abs.(ξ) .> 0)
        pred = Θ * ξ
        err = sqrt(mean((pred .- dx).^2))
        push!(results, (λ=λ, n_terms=n_terms, rmse=err, ξ=ξ))
    end
    return results
end
