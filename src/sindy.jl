# SINDy algorithm: Sequential Thresholded Least Squares (STLSQ)

# Unit-normalize design columns without silently zeroing a finite column whose
# Euclidean norm exceeds the floating-point range. Ordinary columns retain the
# previous `column / norm(column)` evaluation exactly. Overflowing norms use a
# max-abs scale followed by the norm of the scaled column; the two divisors stay
# separate so their product never has to be representable.
function _normalize_sindy_columns(Θ::AbstractMatrix)
    p = size(Θ, 2)
    divisors = [norm(@view Θ[:, j]) for j in 1:p]
    secondary_divisors = similar(divisors)
    fill!(secondary_divisors, one(eltype(secondary_divisors)))
    divisors[divisors .== 0] .= one(eltype(divisors))
    Θn = Θ ./ divisors'

    for j in 1:p
        isfinite(divisors[j]) && continue
        scale = maximum(abs, @view Θ[:, j])
        isfinite(scale) && scale > 0 || throw(ArgumentError(
            "SINDy column $j cannot be normalized within the supported range",
        ))
        scaled = Θ[:, j] ./ scale
        shape_norm = norm(scaled)
        isfinite(shape_norm) && shape_norm > 0 || throw(ArgumentError(
            "SINDy column $j scaled norm is invalid",
        ))
        Θn[:, j] = scaled ./ shape_norm
        divisors[j] = scale
        secondary_divisors[j] = shape_norm
    end
    all(isfinite, Θn) || throw(ArgumentError(
        "normalized SINDy design contains a non-finite value",
    ))
    return Θn, divisors, secondary_divisors
end

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

Threshold semantics: when `normalize=true` (the default), the columns of Θ are
scaled to unit norm before the fit, so the threshold `λ` is applied to the
*normalized* coefficient `ξ̃_j = ξ_j · ‖Θ_j‖` — i.e. each term's contribution
magnitude — not to the physical coefficient `ξ_j`. The returned ξ is rescaled
back to physical units. To reproduce a sparsity pattern exactly, the column
norms ‖Θ_j‖ (or λ expressed in normalized units) must be reported alongside λ.
With `normalize=false`, λ thresholds the physical coefficient directly.

Returns coefficient vector ξ (in physical units).
"""
function stlsq(Θ::AbstractMatrix, dx::AbstractVector;
                λ::Real=0.1, max_iter::Int=25, normalize::Bool=true)
    n, p = size(Θ)
    length(dx) == n ||
        throw(DimensionMismatch("Θ has $n rows, dx has $(length(dx)) entries"))
    n > 0 && p > 0 || throw(ArgumentError("Θ must be nonempty, got size $(size(Θ))"))
    isfinite(λ) && λ >= 0 ||
        throw(ArgumentError("λ must be finite and nonnegative, got $λ"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be at least 1, got $max_iter"))
    all(isfinite, Θ) || throw(ArgumentError("Θ must contain only finite values"))
    all(isfinite, dx) || throw(ArgumentError("dx must contain only finite values"))

    # Optional column normalization for better conditioning
    if normalize
        Θn, col_norms, col_norm_shapes = _normalize_sindy_columns(Θ)
    else
        Θn = Θ
        col_norms = ones(p)
        col_norm_shapes = ones(p)
    end

    # Initial least squares
    ξ = Θn \ dx
    all(isfinite, ξ) || throw(ArgumentError("initial least-squares solve was non-finite"))

    for iter in 1:max_iter
        # Threshold small coefficients. NOTE: each (normalized) coefficient is
        # thresholded independently, with no collinearity or joint-contribution
        # guard, so STLSQ can retain a block of near-collinear columns whose large
        # opposite-sign coefficients cancel to a near-null net contribution (an
        # individually non-identifiable, numerically fragile combination). Use
        # `collinearity_diagnostics` on the returned fit to surface such a block.
        small_inds = abs.(ξ) .< λ
        ξ[small_inds] .= 0.0

        # Find active (non-zero) indices
        active = findall(.!small_inds)
        if isempty(active)
            break
        end

        # Re-solve on active terms only
        ξ_new = similar(ξ, p)
        fill!(ξ_new, zero(eltype(ξ_new)))
        ξ_new[active] = Θn[:, active] \ dx
        all(isfinite, ξ_new) ||
            throw(ArgumentError("active-set least-squares solve was non-finite"))

        # Check convergence
        if norm(ξ_new - ξ) / max(norm(ξ), 1e-10) < 1e-8
            ξ = ξ_new
            break
        end
        ξ = ξ_new
    end

    # Final thresholding fixed point. A single threshold+resolve is NOT a fixed
    # point: the closing least-squares refit can push a surviving coefficient back
    # below λ (reachable on the max_iter-exhaustion path, e.g. libraries with more
    # than max_iter columns). Repeat threshold-then-refit until no active
    # coefficient is sub-λ, so the returned support ALWAYS satisfies the STLSQ
    # contract (every returned nonzero normalized |ξ_j| ≥ λ) regardless of how the
    # main loop exited. The active set is monotone non-increasing — each pass drops
    # at least one term or stops — so this terminates in at most p passes; on an
    # already-stable support (the converged case, and every in-repo call where
    # p < max_iter) it exits without an extra solve, reproducing the prior result.
    while true
        active_final = findall(!=(0.0), ξ)
        isempty(active_final) && break
        below = active_final[abs.(ξ[active_final]) .< λ]
        isempty(below) && break
        ξ[below] .= 0.0
        active_final = findall(!=(0.0), ξ)
        isempty(active_final) && break
        ξ_new = similar(ξ, p)
        fill!(ξ_new, zero(eltype(ξ_new)))
        ξ_new[active_final] = Θn[:, active_final] \ dx
        all(isfinite, ξ_new) ||
            throw(ArgumentError("final active-set least-squares solve was non-finite"))
        ξ = ξ_new
    end

    # Undo normalization
    if normalize
        for j in eachindex(ξ)
            normalized_coefficient = ξ[j]
            ξ[j] = (normalized_coefficient / col_norms[j]) / col_norm_shapes[j]
            iszero(ξ[j]) && !iszero(normalized_coefficient) && throw(ArgumentError(
                "physical SINDy coefficient underflowed the supported range",
            ))
        end
        all(isfinite, ξ) || throw(ArgumentError(
            "physical SINDy coefficients exceed the supported range",
        ))
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
    collinearity_diagnostics(Θ, ξ; groups=nothing, cond_warn=100.0)

Post-STLSQ conditioning / joint-contribution diagnostics for a discovered sparse
model `dx ≈ Θ ξ`. STLSQ thresholds each (normalized) coefficient independently, so
it can keep a block of near-collinear library columns whose large opposite-sign
coefficients cancel to a small net contribution — a numerically fragile,
individually non-identifiable combination. This routine surfaces that situation
WITHOUT altering the fit (`cond_warn` only sets the `warn` flag).

`Θ` is the evaluated library matrix (`n × p`, e.g. from [`evaluate_library`](@ref))
and `ξ` the discovered coefficients (`length p`). Condition numbers are computed on
UNIT-NORMALIZED columns so scale differences between terms do not masquerade as
collinearity; net/gross contributions are in the physical units of `dx`, evaluated
over the supplied rows of `Θ`.

Returns a NamedTuple with
- `active`       — indices of nonzero coefficients;
- `block_cond`   — condition number of the unit-normalized active-column block
  (`κ ≫ 1` signals collinearity among retained terms; `Inf` if singular);
- `net_range`    — `(min, max)` of the active net contribution `Θ_active ξ_active`;
- `net_absmax`   — maximum absolute net contribution;
- `gross_absmax` — `max_i Σ_j |Θ_ij ξ_j|` over active terms (sum of per-term
  magnitudes); `gross_absmax ≫ net_absmax` means the retained terms largely cancel;
- `cancellation` — `gross_absmax / max(net_absmax, eps())`, a cancellation ratio;
- `warn`         — `block_cond > cond_warn`;
- `groups`       — per-group NamedTuples (`indices`, `cond`, `net_range`,
  `net_absmax`, `gross_absmax`, `cancellation`) when `groups` is supplied.

`groups` is an optional vector of index vectors naming candidate collinear blocks
(e.g. a trig/clock-angle block); each is reported separately.
"""
function collinearity_diagnostics(Θ::AbstractMatrix, ξ::AbstractVector;
                                  groups::Union{Nothing,AbstractVector}=nothing,
                                  cond_warn::Real=100.0)
    n, p = size(Θ)
    n >= 1 && p >= 1 ||
        throw(ArgumentError("collinearity diagnostics require a nonempty design matrix"))
    length(ξ) == p ||
        throw(DimensionMismatch("ξ length $(length(ξ)) != $(p) library columns"))
    all(isfinite, Θ) || throw(ArgumentError("design matrix must be finite"))
    all(isfinite, ξ) || throw(ArgumentError("coefficients must be finite"))
    isfinite(cond_warn) && cond_warn > 0 ||
        throw(ArgumentError("cond_warn must be finite and positive"))
    if groups !== nothing
        for group in groups
            idx = collect(Int, group)
            all(i -> 1 <= i <= p, idx) ||
                throw(ArgumentError("collinearity group index out of range"))
            length(unique(idx)) == length(idx) ||
                throw(ArgumentError("collinearity groups must not repeat an index"))
        end
    end

    _block_cond(idx) = begin
        isempty(idx) && return 1.0
        # `svdvals` returns only min(n, p) singular values. For a wide block it
        # can therefore look perfectly conditioned even though its columns are
        # necessarily linearly dependent and individual coefficients are not
        # identifiable.
        length(idx) > n && return Inf
        B = Θ[:, idx]
        normalized, _, _ = _normalize_sindy_columns(B)
        s = svdvals(normalized)
        smin = minimum(s)
        smin == 0 ? Inf : maximum(s) / smin
    end
    _net(idx) = isempty(idx) ? Float64[] : Θ[:, idx] * ξ[idx]
    _gross_absmax(idx) = isempty(idx) ? 0.0 :
        maximum(vec(sum(abs.(Θ[:, idx] .* reshape(ξ[idx], 1, :)), dims=2)))
    _absmax(v) = isempty(v) ? 0.0 : maximum(abs, v)
    _range(v) = isempty(v) ? (0.0, 0.0) : (minimum(v), maximum(v))

    active = findall(!=(0.0), ξ)
    net = _net(active)
    net_absmax = _absmax(net)
    gross_absmax = _gross_absmax(active)

    group_reports = NamedTuple[]
    if groups !== nothing
        for g in groups
            gv = collect(g)
            gnet = _net(gv)
            g_absmax = _absmax(gnet)
            g_gross = _gross_absmax(gv)
            push!(group_reports, (
                indices = gv,
                cond = _block_cond(gv),
                net_range = _range(gnet),
                net_absmax = g_absmax,
                gross_absmax = g_gross,
                cancellation = g_gross / max(g_absmax, eps()),
            ))
        end
    end

    block_cond = _block_cond(active)
    return (
        active = active,
        block_cond = block_cond,
        net_range = _range(net),
        net_absmax = net_absmax,
        gross_absmax = gross_absmax,
        cancellation = gross_absmax / max(net_absmax, eps()),
        warn = block_cond > cond_warn,
        groups = group_reports,
    )
end

"""
    ensemble_sindy(data, lib, target; λ=0.1, n_models=500,
                   subsample_frac=0.8, seed=42, bootstrap=false)

Ensemble SINDy: repeatedly resample the rows, run STLSQ on each resample, and
aggregate. Returns `(median_ξ, inclusion_prob, all_ξ)` where `inclusion_prob[j]` is
the fraction of resamples that selected term `j`, `median_ξ[j]` is the median over
the resamples in which term `j` was selected (nonzero), and `all_ξ` is the full
`p × n_models` draw matrix.

Resampling mode:
- `bootstrap=false` (default): m-out-of-n subsampling WITHOUT replacement, with
  `m = round(subsample_frac · n)` (subagging). This is the shipped default and the
  mode that produced the persisted inclusion/CI artifacts. Quantiles of m-out-of-n
  subsample coefficients are NOT bootstrap coefficient-σ estimates: the per-draw
  spread scales like `√((n-m)/(n·m))` rather than the full-sample `√(1/n)`, so a CI
  half-width read as `±z·σ` (see `init_forecast`) understates the full-sample
  coefficient σ and should not be relabeled a bootstrap CI.
- `bootstrap=true`: classical n-out-of-n resampling WITH replacement (bagging),
  matching the ensemble-SINDy bootstrap literature; use this when a bootstrap
  coefficient-σ interpretation is required. `subsample_frac` is then ignored.
"""
function ensemble_sindy(data::Dict{String,Vector{Float64}},
                        lib::CandidateLibrary,
                        target::AbstractVector;
                        λ::Real=0.1, n_models::Int=500,
                        subsample_frac::Real=0.8, seed::Int=42,
                        bootstrap::Bool=false)
    rng = MersenneTwister(seed)
    Θ = evaluate_library(lib, data)
    n, p = size(Θ)
    length(target) == n ||
        throw(DimensionMismatch("target length $(length(target)) != $n data rows"))
    all(isfinite, target) || throw(ArgumentError("target must contain only finite values"))
    n_models >= 1 || throw(ArgumentError("n_models must be at least 1, got $n_models"))
    isfinite(λ) && λ >= 0 || throw(ArgumentError("λ must be finite and nonnegative, got $λ"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative, got $seed"))
    if bootstrap
        n_sub = n
    else
        isfinite(subsample_frac) && 0 < subsample_frac <= 1 ||
            throw(ArgumentError("subsample_frac must lie in (0, 1], got $subsample_frac"))
        n_sub = round(Int, n * subsample_frac)
        n_sub >= 1 || throw(ArgumentError("subsample_frac selects zero rows"))
    end

    all_ξ = zeros(p, n_models)
    for m in 1:n_models
        # Default: subsample m<n rows without replacement (subagging). Optional:
        # true n-out-of-n bootstrap (with replacement). Keeping the default path
        # byte-identical preserves the shipped inclusion/CI artifacts.
        idx = bootstrap ? rand(rng, 1:n, n) : sort(randperm(rng, n)[1:n_sub])
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
    length(ξ) == length(lib) ||
        throw(DimensionMismatch("ξ length $(length(ξ)) != library length $(length(lib))"))
    all(isfinite, ξ) || throw(ArgumentError("ξ must contain only finite values"))
    Θ = evaluate_library(lib, data)
    prediction = Θ * ξ
    all(isfinite, prediction) || throw(ArgumentError(
        "SINDy prediction exceeded the supported numeric range",
    ))
    return prediction
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
    _validate_candidate_library(lib)
    n_pts = length(swd.t)
    n_pts >= 1 || throw(ArgumentError("simulate_sindy requires at least one sample"))
    all(length(v) == n_pts for v in
        (swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn, swd.Dst, swd.Dst_star)) ||
        throw(DimensionMismatch("SolarWindData fields must have equal length"))
    length(ξ) == length(lib) ||
        throw(DimensionMismatch("ξ length $(length(ξ)) != library length $(length(lib))"))
    all(isfinite, ξ) || throw(ArgumentError("ξ must contain only finite values"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive, got $dt"))
    isfinite(Dst0) || throw(ArgumentError("Dst0 must be finite, got $Dst0"))
    ξ_float = Float64.(ξ)
    all(isfinite, ξ_float) || throw(ArgumentError(
        "ξ values must be representable as finite Float64 values",
    ))
    dt_float = Float64(dt)
    isfinite(dt_float) || throw(ArgumentError(
        "dt must be representable as a finite Float64 value",
    ))
    Dst0_float = Float64(Dst0)
    isfinite(Dst0_float) || throw(ArgumentError(
        "Dst0 must be representable as a finite Float64 value",
    ))
    all(isfinite, swd.t) || throw(ArgumentError("SolarWindData times must be finite"))
    all(v -> all(isfinite, v), (swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn)) ||
        throw(ArgumentError("SolarWindData drivers must contain only finite values"))
    all(>=(0.0), swd.V) && all(>=(0.0), swd.n) && all(>=(0.0), swd.Pdyn) ||
        throw(ArgumentError("speed, density, and dynamic pressure must be nonnegative"))
    Dst_pred = zeros(n_pts)
    Dst_pred[1] = Dst0_float
    θ = Vector{Float64}(undef, length(lib))

    for k in 1:(n_pts - 1)
        _evaluate_point_vector_unchecked!(θ, lib, Dst_pred[k], Float64(swd.V[k]),
                                          Float64(swd.Bz[k]), Float64(swd.By[k]),
                                          Float64(swd.n[k]), Float64(swd.Pdyn[k]))
        dDst = dot(θ, ξ_float)
        isfinite(dDst) || throw(ArgumentError("SINDy derivative became non-finite at sample $k"))

        # Clamp to prevent numerical blow-up (same bounds as baseline models)
        dDst = clamp(dDst, -200.0, 200.0)
        increment = dt_float * dDst
        isfinite(increment) || throw(ArgumentError(
            "SINDy integration increment became non-finite at sample $k",
        ))
        next_state = Dst_pred[k] + increment
        isfinite(next_state) || throw(ArgumentError(
            "SINDy state became non-finite at sample $(k + 1)",
        ))
        Dst_pred[k+1] = clamp(next_state, -2000.0, 50.0)
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
    isempty(lambdas) && throw(ArgumentError("lambda sweep must not be empty"))
    all(λ -> isfinite(λ) && λ >= 0, lambdas) ||
        throw(ArgumentError("lambda sweep values must be finite and nonnegative"))
    return map(lambdas) do λ
        ξ = stlsq(Θ, dx; λ=λ, normalize=normalize)
        n_terms = count(abs.(ξ) .> 0)
        pred = Θ * ξ
        all(isfinite, pred) || throw(ArgumentError(
            "lambda-sweep prediction exceeded the supported range",
        ))
        differences = pred .- dx
        all(isfinite, differences) || throw(ArgumentError(
            "lambda-sweep residual exceeded the supported range",
        ))
        err = _stable_root_mean_square(differences)
        (λ=λ, n_terms=n_terms, rmse=err, ξ=ξ)
    end
end
