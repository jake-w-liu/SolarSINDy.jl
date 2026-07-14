#!/usr/bin/env julia
# Canonical coupled Dst*-AE discovery and matched-cohort outer evaluation.

using SolarSINDy
using CSV, DataFrames, Dates, Statistics, Random, LinearAlgebra, SHA

isdefined(@__MODULE__, :validation_output_paths) ||
    include(joinpath(@__DIR__, "output_paths.jl"))
isdefined(@__MODULE__, :write_output_manifest) ||
    include(joinpath(@__DIR__, "canonical_provenance.jl"))
isdefined(@__MODULE__, :DiscoveryObservationPolicy) ||
    include(joinpath(@__DIR__, "real_discovery_helpers.jl"))

const COUPLED_AE_TERMS = ("AE", "AE^2", "Dst_star*AE", "V*AE", "Bs*AE")
const COUPLED_ENSEMBLE_SEED = 42
const COUPLED_ENSEMBLE_DRAWS = 500
const COUPLED_SUBSAMPLE_FRACTION = 0.8
const COUPLED_CONDITION_WARNING = 100.0

"""Return the canonical 20-term library augmented by five explicit AE terms."""
function build_coupled_library(base_lib::CandidateLibrary=build_solar_wind_library())
    base_names = get_term_names(base_lib)
    length(base_names) == 20 || throw(ArgumentError(
        "the coupled model requires the canonical identifiable 20-term base library",
    ))
    "n*V^2" in base_names && throw(ArgumentError(
        "the coupled base library must not contain the redundant n*V^2 proxy",
    ))
    names = vcat(base_names, collect(COUPLED_AE_TERMS))
    funcs = vcat(copy(base_lib.funcs), Function[
        data -> data["AE"],
        data -> data["AE"].^2,
        data -> data["Dst_star"] .* data["AE"],
        data -> data["V"] .* data["AE"],
        data -> data["Bs"] .* data["AE"],
    ])
    return CandidateLibrary(names, funcs)
end

function _validate_coupled_layout(base_lib::CandidateLibrary,
                                  coupled_lib::CandidateLibrary)
    names = get_term_names(coupled_lib)
    p = length(base_lib)
    length(names) == p + length(COUPLED_AE_TERMS) ||
        throw(DimensionMismatch("coupled library has an unexpected number of terms"))
    names[1:p] == get_term_names(base_lib) || throw(ArgumentError(
        "coupled-library base terms do not match the single-equation library",
    ))
    Tuple(names[(p + 1):end]) == COUPLED_AE_TERMS || throw(ArgumentError(
        "coupled-library AE terms do not match the declared augmented basis",
    ))
    return nothing
end

"""Forward-integrate the fitted coupled equations from one shared observed anchor."""
function simulate_coupled(ξ_dst::AbstractVector, ξ_ae::AbstractVector,
                          base_lib::CandidateLibrary, swd::SolarWindData,
                          ae0::Real, dt::Real; dst0::Real=swd.Dst_star[1])
    SolarSINDy._validate_candidate_library(base_lib)
    p = length(base_lib)
    q = p + length(COUPLED_AE_TERMS)
    length(ξ_dst) == q && length(ξ_ae) == q || throw(DimensionMismatch(
        "both coupled coefficient vectors must have $q entries",
    ))
    all(isfinite, ξ_dst) && all(isfinite, ξ_ae) ||
        throw(ArgumentError("coupled coefficients must be finite"))
    isfinite(ae0) && ae0 >= 0 ||
        throw(ArgumentError("the AE anchor must be finite and nonnegative"))
    isfinite(dst0) || throw(ArgumentError("the Dst* anchor must be finite"))
    isfinite(dt) && dt > 0 ||
        throw(ArgumentError("dt must be finite and positive"))
    n = length(swd.t)
    n >= 1 || throw(ArgumentError("coupled simulation requires at least one sample"))
    all(length(values) == n for values in
        (swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn, swd.Dst, swd.Dst_star)) ||
        throw(DimensionMismatch("SolarWindData fields must have equal length"))
    all(values -> all(isfinite, values),
        (swd.t, swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn)) ||
        throw(ArgumentError("coupled simulation inputs must be finite"))
    all(>=(0), swd.V) && all(>=(0), swd.n) && all(>=(0), swd.Pdyn) ||
        throw(ArgumentError("speed, density, and dynamic pressure must be nonnegative"))

    dst = zeros(n)
    ae = zeros(n)
    dst[1] = dst0
    ae[1] = ae0
    θ = Vector{Float64}(undef, p)
    dst_base = @view ξ_dst[1:p]
    ae_base = @view ξ_ae[1:p]
    @inbounds for k in 1:(n - 1)
        V = Float64(swd.V[k])
        Bz = Float64(swd.Bz[k])
        By = Float64(swd.By[k])
        density = Float64(swd.n[k])
        pressure = Float64(swd.Pdyn[k])
        SolarSINDy._evaluate_point_vector_unchecked!(
            θ, base_lib, dst[k], V, Bz, By, density, pressure,
        )
        bs = max(-Bz, 0.0)
        augmented = (ae[k], ae[k]^2, dst[k] * ae[k], V * ae[k], bs * ae[k])
        d_dst = dot(θ, dst_base)
        d_ae = dot(θ, ae_base)
        for j in eachindex(augmented)
            d_dst += ξ_dst[p + j] * augmented[j]
            d_ae += ξ_ae[p + j] * augmented[j]
        end
        isfinite(d_dst) && isfinite(d_ae) || throw(ArgumentError(
            "coupled derivative became non-finite at sample $k",
        ))
        dst[k + 1] = clamp(dst[k] + dt * clamp(d_dst, -200.0, 200.0),
                           -2000.0, 50.0)
        ae[k + 1] = clamp(ae[k] + dt * clamp(d_ae, -2000.0, 2000.0),
                          0.0, 5000.0)
    end
    return dst, ae
end

function _coupled_scoring_state(df::DataFrame, entry::StormCatalogEntry,
                                shared_record, policy::DiscoveryObservationPolicy)
    if !shared_record.eligible
        return (
            coupled_eligible = false,
            coupled_exclusion_reason = shared_record.exclusion_reason,
            common_anchor_idx = 0,
            post_anchor_dst_star_rows = 0,
            post_anchor_dst_star_fraction = 0.0,
            post_anchor_ae_rows = 0,
            post_anchor_ae_fraction = 0.0,
        )
    end
    first_row = shared_record.scoring_start_idx
    last_row = shared_record.scoring_end_idx
    rows = collect(first_row:last_row)
    dst_original = Bool[
        Bool(df[row, :Dst_observed]) && Bool(df[row, :V_observed]) &&
        Bool(df[row, :n_observed]) && isfinite(df[row, :Dst_star]) for row in rows
    ]
    ae_original = Bool[
        Bool(df[row, :AE_observed]) && isfinite(df[row, :AE]) for row in rows
    ]
    anchor = findfirst(dst_original .& ae_original)
    anchor === nothing && return (
        coupled_eligible = false,
        coupled_exclusion_reason = "no_common_original_dst_star_ae_anchor",
        common_anchor_idx = 0,
        post_anchor_dst_star_rows = 0,
        post_anchor_dst_star_fraction = 0.0,
        post_anchor_ae_rows = 0,
        post_anchor_ae_fraction = 0.0,
    )
    opportunities = length(rows) - anchor
    n_dst = opportunities == 0 ? 0 : count(@view dst_original[(anchor + 1):end])
    n_ae = opportunities == 0 ? 0 : count(@view ae_original[(anchor + 1):end])
    dst_fraction = opportunities == 0 ? 0.0 : n_dst / opportunities
    ae_fraction = opportunities == 0 ? 0.0 : n_ae / opportunities
    reason = n_dst < policy.min_scoring_rows ?
        "insufficient_post_anchor_original_dst_star_rows" :
        dst_fraction < policy.min_scoring_fraction ?
        "insufficient_post_anchor_original_dst_star_fraction" :
        n_ae < policy.min_scoring_rows ?
        "insufficient_post_anchor_original_ae_rows" :
        ae_fraction < policy.min_scoring_fraction ?
        "insufficient_post_anchor_original_ae_fraction" : ""
    return (
        coupled_eligible = isempty(reason),
        coupled_exclusion_reason = reason,
        common_anchor_idx = first_row + anchor - 1,
        post_anchor_dst_star_rows = n_dst,
        post_anchor_dst_star_fraction = dst_fraction,
        post_anchor_ae_rows = n_ae,
        post_anchor_ae_fraction = ae_fraction,
    )
end

function _summarize_coupled_observations(records)
    isempty(records) && return NamedTuple[]
    return [begin
        subset = filter(record -> record.solar_cycle == cycle, records)
        (
            solar_cycle = cycle,
            catalog_storms = length(subset),
            shared_policy_eligible_storms = count(record -> record.eligible, subset),
            coupled_eligible_storms = count(record -> record.coupled_eligible, subset),
            excluded_storms = count(record -> !record.coupled_eligible, subset),
            regression_rows = sum(record.regression_rows for record in subset),
            dst_star_scoring_rows = sum(record.post_anchor_dst_star_rows for record in subset),
            ae_scoring_rows = sum(record.post_anchor_ae_rows for record in subset),
            exclusion_reasons = join(sort!(unique(filter(!isempty,
                String[record.coupled_exclusion_reason for record in subset]))), ";"),
        )
    end for cycle in sort!(unique(record.solar_cycle for record in records))]
end

function _audit_coupled_observations(df::DataFrame,
                                     catalog::Vector{StormCatalogEntry},
                                     policy::DiscoveryObservationPolicy)
    policy.require_ae || throw(ArgumentError(
        "coupled discovery requires DiscoveryObservationPolicy(require_ae=true)",
    ))
    shared = _audit_discovery_observations(df, catalog; policy)
    entries = Dict(entry.storm_id => entry for entry in catalog)
    records = [merge(record,
        _coupled_scoring_state(df, entries[record.storm_id], record, policy))
        for record in shared.storm_records]
    eligible_ids = Set(record.storm_id for record in records if record.coupled_eligible)
    eligible = [entry for entry in shared.eligible_entries if entry.storm_id in eligible_ids]
    return (
        eligible_entries = eligible,
        storm_records = records,
        cycle_records = _summarize_coupled_observations(records),
        policy = policy,
    )
end

function _prepare_coupled_storm(df::DataFrame, entry::StormCatalogEntry,
                                base_lib::CandidateLibrary,
                                coupled_lib::CandidateLibrary,
                                policy::DiscoveryObservationPolicy)
    _validate_coupled_layout(base_lib, coupled_lib)
    shared = _prepare_discovery_storm(df, entry, base_lib; policy)
    state = _coupled_scoring_state(df, entry, shared.observation_record, policy)
    state.coupled_eligible || throw(ArgumentError(
        "storm $(entry.storm_id) is ineligible for coupled scoring: " *
        state.coupled_exclusion_reason,
    ))

    rows = entry.onset_idx:entry.end_idx
    ae_smoothed = smooth_moving_average(Float64.(df.AE[rows]), policy.smooth_window)
    dae = numerical_derivative(ae_smoothed, 1.0)
    ae_target = dae[shared.regression_mask]
    ae_regression = ae_smoothed[shared.regression_mask]
    all(isfinite, ae_target) && all(isfinite, ae_regression) || throw(ArgumentError(
        "storm $(entry.storm_id) has non-finite AE values on audited regression rows",
    ))
    data = copy(shared.data)
    data["AE"] = ae_regression
    theta_coupled = evaluate_library(coupled_lib, data)

    first_row = shared.observation_record.scoring_start_idx
    last_row = shared.observation_record.scoring_end_idx
    length(shared.swd.t) == last_row - first_row + 1 || throw(DimensionMismatch(
        "storm $(entry.storm_id) scoring segment disagrees with the observation audit",
    ))
    dst_scoring = copy(shared.scoring_observations)
    ae_scoring = fill(NaN, length(dst_scoring))
    for (local_index, row) in enumerate(first_row:last_row)
        Bool(df[row, :AE_observed]) && isfinite(df[row, :AE]) &&
            (ae_scoring[local_index] = Float64(df[row, :AE]))
    end
    anchor = state.common_anchor_idx - first_row + 1
    dst_scoring[1:(anchor - 1)] .= NaN
    ae_scoring[1:(anchor - 1)] .= NaN
    isfinite(dst_scoring[anchor]) && isfinite(ae_scoring[anchor]) ||
        error("audited common Dst*/AE anchor is not original and finite")

    return (
        storm_id = entry.storm_id,
        onset_time = entry.onset_time,
        entry = entry,
        swd = shared.swd,
        data = shared.data,
        theta = shared.theta,
        target = shared.target,
        theta_single = shared.theta,
        theta_coupled = theta_coupled,
        target_dst = shared.target,
        target_ae = ae_target,
        scoring_observations = dst_scoring,
        scoring_dst = dst_scoring,
        scoring_ae = ae_scoring,
        score_start_idx = first_row,
        score_end_idx = last_row,
        common_anchor_idx = state.common_anchor_idx,
        observation_record = merge(shared.observation_record, state),
        library_terms = Tuple(get_term_names(base_lib)),
        coupled_library_terms = Tuple(get_term_names(coupled_lib)),
    )
end

function _prepare_coupled_storms(df::DataFrame,
                                 entries::Vector{StormCatalogEntry},
                                 base_lib::CandidateLibrary,
                                 coupled_lib::CandidateLibrary,
                                 policy::DiscoveryObservationPolicy)
    return [_prepare_coupled_storm(df, entry, base_lib, coupled_lib, policy)
            for entry in entries]
end

function _cached_coupled_design!(cache::AbstractDict, storms::AbstractVector)
    isempty(storms) && throw(ArgumentError("cannot fit an empty coupled storm set"))
    ordered = sort(collect(storms); by=storm -> (storm.onset_time, storm.storm_id))
    ids = Tuple(string(storm.storm_id) for storm in ordered)
    length(unique(ids)) == length(ids) ||
        throw(ArgumentError("coupled-fit storm identifiers must be unique"))
    terms = ordered[1].coupled_library_terms
    all(storm -> storm.coupled_library_terms == terms, ordered) ||
        throw(ArgumentError("coupled-fit storms use different libraries"))
    key = (terms, ids)
    return get!(cache, key) do
        theta = reduce(vcat, (storm.theta_coupled for storm in ordered))
        dst = reduce(vcat, (storm.target_dst for storm in ordered))
        ae = reduce(vcat, (storm.target_ae for storm in ordered))
        size(theta, 1) == length(dst) == length(ae) ||
            throw(DimensionMismatch("coupled design and target lengths differ"))
        (theta=theta, dst=dst, ae=ae, storm_ids=ids)
    end
end

function _coupled_ae_score(model, storm, base_lib)
    anchor = findfirst(isfinite, storm.scoring_dst)
    anchor === nothing && error("storm $(storm.storm_id) has no common anchor")
    isfinite(storm.scoring_ae[anchor]) ||
        error("storm $(storm.storm_id) common anchor has no original AE")
    swd = _slice_solar_wind(storm.swd, anchor)
    _, prediction = simulate_coupled(model.dst, model.ae, base_lib, swd,
                                     storm.scoring_ae[anchor], 1.0;
                                     dst0=storm.scoring_dst[anchor])
    observed = storm.scoring_ae[anchor:end]
    indices = findall(isfinite, @view observed[2:end]) .+ 1
    isempty(indices) && error("storm $(storm.storm_id) has no original AE score rows")
    return (rmse_nt=rmse(prediction[indices], observed[indices]),
            n_scored=length(indices))
end

"""Select one shared lambda for both coupled equations using training storms only."""
function _select_coupled_lambda(training_storms::AbstractVector,
                                base_lib::CandidateLibrary,
                                coupled_lib::CandidateLibrary,
                                design_cache::AbstractDict=Dict{Any,Any}())
    _validate_coupled_layout(base_lib, coupled_lib)
    terms = get_term_names(coupled_lib)
    model_cache = Dict{Tuple{Tuple{Vararg{String}},Float64},Any}()
    fit = function (storms, lambda)
        design = _cached_coupled_design!(design_cache, storms)
        model = (
            dst = stlsq(design.theta, design.dst; λ=lambda, normalize=true),
            ae = stlsq(design.theta, design.ae; λ=lambda, normalize=true),
        )
        model_cache[(design.storm_ids, Float64(lambda))] = model
        support = vcat(
            ["dDst_star/dt::$(terms[i])" for i in findall(!=(0.0), model.dst)],
            ["dAE/dt::$(terms[i])" for i in findall(!=(0.0), model.ae)],
        )
        return (model=model, support=support)
    end
    integrate = function (model, storm, anchor, dst0)
        isfinite(storm.scoring_ae[anchor]) ||
            error("coupled selector received a Dst* anchor without original AE")
        swd = _slice_solar_wind(storm.swd, anchor)
        dst, _ = simulate_coupled(model.dst, model.ae, base_lib, swd,
                                  storm.scoring_ae[anchor], 1.0; dst0)
        return dst
    end
    selection = select_storm_lambda(training_storms;
        fit,
        integrate,
        observations=storm -> storm.scoring_dst,
    )

    inner_ids = Tuple(row.storm_id for row in selection.split_records
                      if row.inner_split == "train")
    inner_validation = [storm for storm in training_storms
        if string(storm.storm_id) in Set(row.storm_id for row in selection.split_records
                                        if row.inner_split == "validation")]
    secondary_errors = NamedTuple[]
    secondary_candidates = NamedTuple[]
    fit_records = NamedTuple[]
    for candidate in selection.candidate_records
        model = model_cache[(inner_ids, Float64(candidate.lambda))]
        errors = Float64[]
        for storm in sort(inner_validation; by=storm -> (storm.onset_time, storm.storm_id))
            score = _coupled_ae_score(model, storm, base_lib)
            push!(errors, score.rmse_nt)
            push!(secondary_errors, (
                candidate_index = candidate.candidate_index,
                lambda = candidate.lambda,
                storm_id = string(storm.storm_id),
                ae_rmse_nt = score.rmse_nt,
                n_scored = score.n_scored,
                used_for_lambda_selection = false,
            ))
        end
        push!(secondary_candidates, (
            candidate_index = candidate.candidate_index,
            lambda = candidate.lambda,
            mean_storm_ae_rmse_nt = mean(errors),
            standard_error_nt = length(errors) == 1 ? 0.0 :
                std(errors) / sqrt(length(errors)),
            used_for_lambda_selection = false,
            dst_selected = candidate.selected,
        ))
        for equation in (:dst, :ae)
            coefficients = getproperty(model, equation)
            push!(fit_records, (
                stage = "inner_candidate",
                candidate_index = candidate.candidate_index,
                lambda = candidate.lambda,
                equation = equation == :dst ? "dDst_star/dt" : "dAE/dt",
                n_fit_storms = length(inner_ids),
                n_active_terms = count(!=(0.0), coefficients),
                shared_lambda = true,
            ))
        end
    end
    full_ids = Tuple(row.storm_id for row in selection.split_records)
    for equation in (:dst, :ae)
        coefficients = getproperty(selection.model, equation)
        push!(fit_records, (
            stage = "full_refit",
            candidate_index = selection.decision_record.selected_candidate_index,
            lambda = selection.selected_lambda,
            equation = equation == :dst ? "dDst_star/dt" : "dAE/dt",
            n_fit_storms = length(full_ids),
            n_active_terms = count(!=(0.0), coefficients),
            shared_lambda = true,
        ))
    end
    return merge(selection, (
        ae_candidate_records = secondary_candidates,
        ae_error_records = secondary_errors,
        equation_fit_records = fit_records,
    ))
end

function _assert_matched_inner_split(coupled_selection, single_selection)
    coupled = [(row.storm_id, row.onset_time, row.inner_split)
               for row in coupled_selection.split_records]
    single = [(row.storm_id, row.onset_time, row.inner_split)
              for row in single_selection.split_records]
    coupled == single || error(
        "coupled and single-equation controls used different inner storm splits",
    )
    return [(
        storm_id = coupled[i][1],
        onset_time = coupled[i][2],
        coupled_inner_split = coupled[i][3],
        single_inner_split = single[i][3],
        exact_match = true,
    ) for i in eachindex(coupled)]
end

function _finish_coupled_stlsq(theta::AbstractMatrix,
                               target::AbstractVector,
                               initial::AbstractVector,
                               column_norms::AbstractVector,
                               column_norm_shapes::AbstractVector,
                               lambda::Real,
                               max_iter::Int)
    # This is the threshold/refit fixed point from `stlsq`; the paired caller
    # injects a shared full-design solve and otherwise preserves its semantics.
    p = size(theta, 2)
    coefficients = copy(initial)
    for _ in 1:max_iter
        small = abs.(coefficients) .< lambda
        coefficients[small] .= 0.0
        active = findall(.!small)
        isempty(active) && break
        updated = similar(coefficients, p)
        fill!(updated, zero(eltype(updated)))
        updated[active] = theta[:, active] \ target
        all(isfinite, updated) ||
            throw(ArgumentError("active-set least-squares solve was non-finite"))
        if norm(updated - coefficients) / max(norm(coefficients), 1e-10) < 1e-8
            coefficients = updated
            break
        end
        coefficients = updated
    end
    while true
        active = findall(!=(0.0), coefficients)
        isempty(active) && break
        below = active[abs.(coefficients[active]) .< lambda]
        isempty(below) && break
        coefficients[below] .= 0.0
        active = findall(!=(0.0), coefficients)
        isempty(active) && break
        updated = similar(coefficients, p)
        fill!(updated, zero(eltype(updated)))
        updated[active] = theta[:, active] \ target
        all(isfinite, updated) || throw(ArgumentError(
            "final active-set least-squares solve was non-finite",
        ))
        coefficients = updated
    end
    physical = similar(coefficients)
    for column in eachindex(coefficients)
        normalized_coefficient = coefficients[column]
        physical[column] =
            (normalized_coefficient / column_norms[column]) /
            column_norm_shapes[column]
        iszero(physical[column]) && !iszero(normalized_coefficient) &&
            throw(ArgumentError(
                "physical coupled-SINDy coefficient underflowed the supported range",
            ))
    end
    all(isfinite, physical) || throw(ArgumentError(
        "physical coupled-SINDy coefficients exceed the supported range",
    ))
    return physical
end

"""Fit both coupled targets after one normalization and one full-design factorization."""
function _paired_coupled_stlsq(theta::AbstractMatrix,
                               dst_target::AbstractVector,
                               ae_target::AbstractVector;
                               lambda::Real,
                               max_iter::Int=25)
    n, p = size(theta)
    length(dst_target) == n && length(ae_target) == n || throw(DimensionMismatch(
        "coupled design and both targets must have equal row counts",
    ))
    n > 0 && p > 0 ||
        throw(ArgumentError("coupled design must be nonempty, got size $(size(theta))"))
    isfinite(lambda) && lambda >= 0 ||
        throw(ArgumentError("lambda must be finite and nonnegative"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be at least 1"))
    all(isfinite, theta) && all(isfinite, dst_target) && all(isfinite, ae_target) ||
        throw(ArgumentError("coupled design and targets must be finite"))

    normalized, column_norms, column_norm_shapes =
        SolarSINDy._normalize_sindy_columns(theta)
    full_factorization = factorize(normalized)
    dst_initial = full_factorization \ dst_target
    ae_initial = full_factorization \ ae_target
    all(isfinite, dst_initial) && all(isfinite, ae_initial) ||
        throw(ArgumentError("initial coupled least-squares solve was non-finite"))
    return (
        dst=_finish_coupled_stlsq(
            normalized, dst_target, dst_initial, column_norms,
            column_norm_shapes, lambda, max_iter,
        ),
        ae=_finish_coupled_stlsq(
            normalized, ae_target, ae_initial, column_norms,
            column_norm_shapes, lambda, max_iter,
        ),
    )
end

function _paired_coupled_ensemble(design, lambda;
                                  n_models::Int=COUPLED_ENSEMBLE_DRAWS,
                                  subsample_fraction::Real=COUPLED_SUBSAMPLE_FRACTION,
                                  seed::Int=COUPLED_ENSEMBLE_SEED)
    n, p = size(design.theta)
    n_models >= 1 || throw(ArgumentError("n_models must be positive"))
    isfinite(subsample_fraction) && 0 < subsample_fraction <= 1 ||
        throw(ArgumentError("subsample_fraction must lie in (0, 1]"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    n_sub = round(Int, n * subsample_fraction)
    n_sub >= 1 || throw(ArgumentError("subsampling selected no rows"))
    rng = MersenneTwister(seed)
    dst = zeros(p, n_models)
    ae = zeros(p, n_models)
    for draw in 1:n_models
        indices = sort(randperm(rng, n)[1:n_sub])
        theta = design.theta[indices, :]
        fit = _paired_coupled_stlsq(
            theta, design.dst[indices], design.ae[indices]; lambda,
        )
        dst[:, draw] = fit.dst
        ae[:, draw] = fit.ae
    end
    return (dst=dst, ae=ae)
end

function _coupled_design_diagnostic_record(theta::AbstractMatrix,
                                           coefficients::AbstractVector,
                                           term_names::AbstractVector,
                                           indices::AbstractVector{Int};
                                           equation::AbstractString,
                                           block::AbstractString,
                                           selected_lambda::Real)
    size(theta, 2) == length(coefficients) == length(term_names) ||
        throw(DimensionMismatch("coupled diagnostic inputs must align"))
    all(index -> 1 <= index <= size(theta, 2), indices) ||
        throw(BoundsError(term_names, indices))
    length(unique(indices)) == length(indices) ||
        throw(ArgumentError("coupled diagnostic indices must be unique"))
    all(isfinite, theta) && all(isfinite, coefficients) ||
        throw(ArgumentError("coupled diagnostic inputs must be finite"))
    isfinite(selected_lambda) && selected_lambda >= 0 ||
        throw(ArgumentError("selected lambda must be finite and nonnegative"))

    normalized_rank = 0
    rank_tolerance = NaN
    condition_number = NaN
    largest_singular_value = NaN
    smallest_singular_value = NaN
    zero_norm_columns = 0
    if !isempty(indices)
        design = Matrix{Float64}(@view theta[:, indices])
        norms = [norm(@view design[:, column]) for column in axes(design, 2)]
        zero_norm_columns = count(iszero, norms)
        scaling = copy(norms)
        scaling[iszero.(scaling)] .= 1.0
        normalized = design ./ scaling'
        singular_values = svdvals(normalized)
        largest_singular_value = isempty(singular_values) ? 0.0 : first(singular_values)
        smallest_singular_value = isempty(singular_values) ? 0.0 : last(singular_values)
        rank_tolerance = max(size(normalized)...) * eps(Float64) * largest_singular_value
        normalized_rank = count(value -> value > rank_tolerance, singular_values)
        condition_number = largest_singular_value == 0.0 ||
                           normalized_rank < size(normalized, 2) ? Inf :
                           largest_singular_value / smallest_singular_value
    end

    net = zeros(size(theta, 1))
    gross = zeros(size(theta, 1))
    for column in indices
        coefficient = coefficients[column]
        @inbounds for row in axes(theta, 1)
            contribution = theta[row, column] * coefficient
            net[row] += contribution
            gross[row] += abs(contribution)
        end
    end
    net_absmax = isempty(net) ? 0.0 : maximum(abs, net)
    gross_absmax = isempty(gross) ? 0.0 : maximum(gross)
    cancellation = gross_absmax == 0.0 ? 0.0 :
        gross_absmax / max(net_absmax, eps(Float64))
    status = if isempty(indices)
        "not_present"
    elseif length(indices) == 1 && coefficients[only(indices)] == 0.0
        "inactive_basis_term"
    elseif normalized_rank < length(indices)
        "rank_deficient_grouped_net_only"
    elseif condition_number > COUPLED_CONDITION_WARNING
        "ill_conditioned_grouped_net_preferred"
    elseif length(indices) == 1
        "single_basis_term"
    else
        "full_rank_basis_dependent_group"
    end
    return (
        equation=String(equation),
        block=String(block),
        terms=isempty(indices) ? "none" : join(term_names[indices], ";"),
        row_scope="coupled_training_regression_rows",
        contribution_units="nT_per_hour",
        normalization="training_column_l2_norm",
        rank_rule="singular_value > max(n_rows,n_columns)*eps(Float64)*largest_singular_value",
        n_rows=size(theta, 1),
        n_columns=length(indices),
        zero_norm_columns=zero_norm_columns,
        normalized_design_rank=normalized_rank,
        normalized_design_rank_tolerance=rank_tolerance,
        normalized_design_condition_number=condition_number,
        largest_normalized_singular_value=largest_singular_value,
        smallest_normalized_singular_value=smallest_singular_value,
        n_active_terms=count(index -> coefficients[index] != 0.0, indices),
        net_contribution_min_nt_per_hour=isempty(net) ? 0.0 : minimum(net),
        net_contribution_max_nt_per_hour=isempty(net) ? 0.0 : maximum(net),
        net_contribution_absmax_nt_per_hour=net_absmax,
        gross_contribution_absmax_nt_per_hour=gross_absmax,
        cancellation_ratio=cancellation,
        condition_warning_threshold=COUPLED_CONDITION_WARNING,
        interpretation_status=status,
        selected_lambda=Float64(selected_lambda),
    )
end

function _coupled_design_diagnostic_records(design, model,
                                            term_names::AbstractVector,
                                            selected_lambda::Real)
    size(design.theta, 2) == length(term_names) ||
        throw(DimensionMismatch("coupled design and term names must align"))
    rows = NamedTuple[]
    for (equation, coefficients, cross_term) in (
        ("dDst_star/dt", model.dst, "AE"),
        ("dAE/dt", model.ae, "Dst_star"),
    )
        active = findall(!iszero, coefficients)
        cross = findall(term -> occursin(cross_term, term), term_names)
        for (block, indices) in (
            "full_design" => collect(eachindex(term_names)),
            "selected_active_block" => active,
            "cross_index_candidate_block" => cross,
            "selected_cross_index_block" => intersect(active, cross),
        )
            push!(rows, _coupled_design_diagnostic_record(
                design.theta, coefficients, term_names, collect(indices);
                equation, block, selected_lambda,
            ))
        end
    end
    return rows
end

function _conditional_quantiles(values)
    nonzero = values[values .!= 0.0]
    isempty(nonzero) && return (NaN, NaN, NaN)
    return (quantile(nonzero, 0.025), median(nonzero), quantile(nonzero, 0.975))
end

function _coupled_coefficient_records(model, draws, terms, lambda)
    rows = NamedTuple[]
    for (equation, coefficients, matrix) in
        (("dDst_star/dt", model.dst, draws.dst), ("dAE/dt", model.ae, draws.ae))
        for i in eachindex(terms)
            q025, q50, q975 = _conditional_quantiles(@view matrix[i, :])
            cross_index = (equation == "dDst_star/dt" && occursin("AE", terms[i])) ||
                          (equation == "dAE/dt" && occursin("Dst_star", terms[i]))
            push!(rows, (
                equation = equation,
                term = terms[i],
                coefficient = coefficients[i],
                inclusion_probability = count(!=(0.0), @view matrix[i, :]) /
                    size(matrix, 2),
                conditional_nonzero_q025 = q025,
                conditional_nonzero_median = q50,
                conditional_nonzero_q975 = q975,
                interval_kind = "conditional_nonzero_empirical_subsample_quantile",
                association_scope = cross_index ?
                    "cross_index_conditional_association" : "candidate_term",
                selected_lambda = lambda,
            ))
        end
    end
    return rows
end

function _coupled_draw_records(draws, terms, lambda)
    rows = NamedTuple[]
    for (equation, matrix) in (("dDst_star/dt", draws.dst), ("dAE/dt", draws.ae))
        for draw in axes(matrix, 2), term in axes(matrix, 1)
            push!(rows, (
                draw_id = draw,
                equation = equation,
                term = terms[term],
                coefficient = matrix[term, draw],
                selected_lambda = lambda,
                resampling = "m_out_of_n_without_replacement",
                subsample_fraction = COUPLED_SUBSAMPLE_FRACTION,
                structural_zeros_retained = true,
            ))
        end
    end
    return rows
end

function _score_coupled_storm(storm, coupled_model, single_coefficients,
                              base_lib::CandidateLibrary)
    anchor = findfirst(isfinite, storm.scoring_dst)
    anchor === nothing && error("storm $(storm.storm_id) has no common anchor")
    dst0 = storm.scoring_dst[anchor]
    ae0 = storm.scoring_ae[anchor]
    isfinite(ae0) || error("storm $(storm.storm_id) anchor AE is not original")
    swd = _slice_solar_wind(storm.swd, anchor)
    dst_observed = storm.scoring_dst[anchor:end]
    ae_observed = storm.scoring_ae[anchor:end]
    dst_indices = findall(isfinite, @view dst_observed[2:end]) .+ 1
    ae_indices = findall(isfinite, @view ae_observed[2:end]) .+ 1
    isempty(dst_indices) && error("storm $(storm.storm_id) has no Dst* score rows")
    isempty(ae_indices) && error("storm $(storm.storm_id) has no AE score rows")
    bs = max.(-swd.Bz, 0.0)
    coupled_dst, coupled_ae = simulate_coupled(
        coupled_model.dst, coupled_model.ae, base_lib, swd, ae0, 1.0; dst0,
    )
    predictions = (
        CoupledSINDy = coupled_dst,
        SingleSINDy = simulate_sindy(single_coefficients, base_lib, swd, 1.0; Dst0=dst0),
        Burton = simulate_burton(swd.V, bs, 1.0; Dst0=dst0),
        BurtonFull = simulate_burton_full(swd.V, bs, 1.0; Dst0=dst0),
        OBrienMcPherron = simulate_obrien(swd.V, bs, 1.0; Dst0=dst0),
    )
    model_names = (
        CoupledSINDy = "Coupled-SINDy",
        SingleSINDy = "Single-SINDy",
        Burton = "Burton",
        BurtonFull = "BurtonFull",
        OBrienMcPherron = "OBrien-McPherron",
    )
    anchor_global = storm.score_start_idx + anchor - 1
    score_global = [anchor_global + index - 1 for index in dst_indices]
    ae_score_global = [anchor_global + index - 1 for index in ae_indices]
    signature_source = join((
        string(storm.storm_id), string(anchor_global), string(anchor_global),
        string(storm.score_end_idx), join(score_global, ";"),
    ), "|")
    signature = bytes2hex(SHA.sha256(signature_source))
    metrics = NamedTuple[]
    cohorts = NamedTuple[]
    for key in propertynames(predictions)
        prediction = getproperty(predictions, key)
        first(prediction) == dst0 || error("$key did not use the shared Dst* anchor")
        summary = metrics_summary(prediction[dst_indices], dst_observed[dst_indices];
                                  name=getproperty(model_names, key))
        push!(metrics, (
            split = storm.entry.split,
            storm_id = storm.storm_id,
            target = "Dst_star",
            model = getproperty(model_names, key),
            rmse_nt = summary.rmse,
            mae_nt = summary.mae,
            correlation = summary.corr,
            pe = summary.pe,
            n_points = length(dst_indices),
            min_dst_star_observed_nt = minimum(dst_observed[isfinite.(dst_observed)]),
            anchor_catalog_idx = anchor_global,
            scored_catalog_indices = join(score_global, ";"),
            original_target_only = true,
        ))
        push!(cohorts, (
            split = storm.entry.split,
            storm_id = storm.storm_id,
            model = getproperty(model_names, key),
            anchor_catalog_idx = anchor_global,
            driver_start_catalog_idx = anchor_global,
            driver_end_catalog_idx = storm.score_end_idx,
            scored_dst_star_catalog_indices = join(score_global, ";"),
            n_scored_dst_star = length(dst_indices),
            cohort_signature_sha256 = signature,
            exact_cohort_equal = true,
        ))
    end
    ae_summary = metrics_summary(coupled_ae[ae_indices], ae_observed[ae_indices];
                                 name="Coupled-SINDy")
    push!(metrics, (
        split = storm.entry.split,
        storm_id = storm.storm_id,
        target = "AE",
        model = "Coupled-SINDy",
        rmse_nt = ae_summary.rmse,
        mae_nt = ae_summary.mae,
        correlation = ae_summary.corr,
        pe = ae_summary.pe,
        n_points = length(ae_indices),
        min_dst_star_observed_nt = minimum(dst_observed[isfinite.(dst_observed)]),
        anchor_catalog_idx = anchor_global,
        scored_catalog_indices = join(ae_score_global, ";"),
        original_target_only = true,
    ))
    return (metrics=metrics, cohorts=cohorts, predictions=predictions,
            coupled_ae=coupled_ae, dst_indices=dst_indices, ae_indices=ae_indices)
end

function _assert_exact_outer_cohorts(rows)
    frame = DataFrame(rows)
    expected = Set((
        "Coupled-SINDy", "Single-SINDy", "Burton", "BurtonFull",
        "OBrien-McPherron",
    ))
    for group in groupby(frame, [:split, :storm_id])
        Set(String.(group.model)) == expected || error(
            "outer storm $(first(group.storm_id)) does not contain all matched comparators",
        )
        length(unique(group.cohort_signature_sha256)) == 1 || error(
            "outer storm $(first(group.storm_id)) used unequal comparator cohorts",
        )
        all(group.exact_cohort_equal) || error("cohort equality audit failed")
    end
    return true
end

function _write_manifested_csv(path, data, context; selection_record,
                               seed::Union{Nothing,Int}=nothing,
                               deterministic::Bool=seed === nothing,
                               metadata=(;))
    frame = data isa DataFrame ? data : DataFrame(data)
    manifest_path = path * ".manifest.json"
    input_paths = Dict(
        "omni_extracted" => context.omni,
        "storm_catalog" => context.catalog,
    )
    for field in (:observation_inputs, :selection_inputs)
        hasproperty(context, field) || continue
        for (name, input) in getproperty(context, field)
            haskey(input_paths, String(name)) && throw(ArgumentError(
                "duplicate coupled manifest input name: $name",
            ))
            input_paths[String(name)] = String(input)
        end
    end
    snapshot = SolarSINDy._snapshot_regular_file_set([path, manifest_path])
    try
        _write_discovery_csv(path, frame)
        write_output_manifest(path;
            producer_script=@__FILE__,
            input_paths,
            selection_record,
            seed,
            deterministic,
            metadata=merge((rows=nrow(frame), columns=names(frame)), metadata),
            mode=context.mode,
        )
        verify_output_manifest(path;
            require_canonical=context.mode == :canonical,
            verify_source=true,
        )
    catch
        SolarSINDy._restore_regular_file_set!(snapshot)
        rethrow()
    end
    SolarSINDy._discard_regular_file_snapshot!(snapshot)
    push!(context.outputs, path)
    return path
end

function _manifest_selection_outputs!(paths, context, selection_record;
                                      producer_script::AbstractString=@__FILE__,
                                      _after_manifest_hook::Function=(field, path) -> nothing)
    return SolarSINDy._with_selection_csv_set_lock(paths) do
        output_paths = [getproperty(paths, field) for field in propertynames(paths)]
        transaction_paths = vcat(output_paths,
            [path * ".manifest.json" for path in output_paths])
        snapshot = SolarSINDy._snapshot_regular_file_set(transaction_paths)
        try
            for field in propertynames(paths)
                path = getproperty(paths, field)
                write_output_manifest(path;
                    producer_script,
                    input_paths=merge(
                        Dict("omni_extracted" => context.omni,
                             "storm_catalog" => context.catalog),
                        hasproperty(context, :observation_inputs) ?
                            context.observation_inputs : Dict{String,String}(),
                    ),
                    selection_record=merge(selection_record, (artifact=String(field),)),
                    deterministic=true,
                    mode=context.mode,
                )
                _after_manifest_hook(field, path)
            end
            for path in output_paths
                verify_output_manifest(path;
                    require_canonical=context.mode == :canonical,
                    verify_source=true,
                )
            end
        catch
            SolarSINDy._restore_regular_file_set!(snapshot)
            rethrow()
        end
        SolarSINDy._discard_regular_file_snapshot!(snapshot)
        append!(context.outputs, output_paths)
        paths
    end
end

function _write_manifested_selection_outputs!(selection, output_root::AbstractString,
                                               prefix::AbstractString, context,
                                               selection_record)
    paths = (
        split=joinpath(output_root, "$(prefix)_inner_split.csv"),
        candidates=joinpath(output_root, "$(prefix)_candidates.csv"),
        errors=joinpath(output_root, "$(prefix)_validation_errors.csv"),
        support=joinpath(output_root, "$(prefix)_support.csv"),
        decision=joinpath(output_root, "$(prefix)_decision.csv"),
    )
    output_paths = [getproperty(paths, field) for field in propertynames(paths)]
    snapshot = SolarSINDy._snapshot_regular_file_set(vcat(output_paths,
        [path * ".manifest.json" for path in output_paths]))
    try
        written = write_storm_lambda_selection(selection, output_root; prefix)
        propertynames(written) == propertynames(paths) || error(
            "selection writer returned an unexpected output inventory",
        )
        _manifest_selection_outputs!(written, context, selection_record)
    catch
        SolarSINDy._restore_regular_file_set!(snapshot)
        rethrow()
    end
    SolarSINDy._discard_regular_file_snapshot!(snapshot)
    return paths
end

function run_coupled_discovery()
    requested_root = strip(get(ENV, "SOLARSINDY_OUTPUT_ROOT", ""))
    isempty(requested_root) && error(
        "SOLARSINDY_OUTPUT_ROOT must be set for coupled discovery",
    )
    paths = validation_output_paths()
    paths.explicit || error("coupled discovery requires an explicit output root")
    omni = paths.omni
    catalog_path = joinpath(paths.data, "storm_catalog.csv")
    isfile(omni) || error("frozen OMNI extraction not found: $omni")
    isfile(catalog_path) || error("verified storm catalog not found: $catalog_path")
    verify_omni_input(omni; mode=paths.mode)
    catalog = load_verified_storm_catalog(catalog_path;
        omni_path=omni,
        parameters=storm_catalog_parameters(),
        mode=paths.mode,
    )

    println("Loading frozen OMNI data for coupled Dst*-AE discovery...")
    df = parse_omni2(omni; year_start=1963, year_end=2025)
    add_original_observation_flags!(df)
    clean_omni_data!(df)
    policy = DiscoveryObservationPolicy(require_ae=true)
    audit = _audit_coupled_observations(df, catalog, policy)
    isempty(audit.eligible_entries) && error("no storms satisfy the coupled observation policy")
    context = (omni=omni, catalog=catalog_path, mode=paths.mode, outputs=String[])
    policy_record = (
        kind="coupled_observation_policy",
        smooth_window=policy.smooth_window,
        min_regression_rows=policy.min_regression_rows,
        min_scoring_rows=policy.min_scoring_rows,
        min_scoring_fraction=policy.min_scoring_fraction,
        require_ae=policy.require_ae,
    )
    storm_audit = DataFrame(audit.storm_records)
    storm_audit_path = _write_manifested_csv(
        joinpath(paths.data, "coupled_observation_storm_audit.csv"),
        storm_audit, context; selection_record=policy_record)
    cycle_audit_path = _write_manifested_csv(
        joinpath(paths.data, "coupled_observation_cycle_audit.csv"),
        audit.cycle_records, context; selection_record=policy_record)
    exclusions = storm_audit[.!storm_audit.coupled_eligible,
        [:storm_id, :solar_cycle, :catalog_split, :eligible, :exclusion_reason,
         :coupled_exclusion_reason]]
    exclusion_audit_path = _write_manifested_csv(
        joinpath(paths.data, "coupled_exclusion_audit.csv"),
        exclusions, context; selection_record=policy_record)
    context = merge(context, (observation_inputs=Dict(
        "coupled_storm_observation_audit" => storm_audit_path,
        "coupled_cycle_observation_audit" => cycle_audit_path,
        "coupled_exclusion_audit" => exclusion_audit_path,
    ),))

    base_lib = build_solar_wind_library()
    coupled_lib = build_coupled_library(base_lib)
    storms = _prepare_coupled_storms(df, audit.eligible_entries, base_lib,
                                      coupled_lib, policy)
    training = filter(storm -> storm.entry.split == "train", storms)
    length(training) >= 2 || error("coupled selection needs at least two training storms")
    coupled_cache = Dict{Any,Any}()
    coupled_selection = _select_coupled_lambda(training, base_lib, coupled_lib,
                                                coupled_cache)
    single_cache = Dict{Any,Any}()
    single_selection = _select_discovery_lambda(training, base_lib, single_cache)
    split_audit = _assert_matched_inner_split(coupled_selection, single_selection)
    training_ids = Set(storm.storm_id for storm in training)
    all(parse(Int, row.storm_id) in training_ids for row in coupled_selection.split_records) ||
        error("coupled selector contacted a non-training storm")

    coupled_paths = _write_manifested_selection_outputs!(
        coupled_selection, paths.data, "coupled_lambda", context, (
        kind="coupled_shared_lambda_selection",
        selected_lambda=coupled_selection.selected_lambda,
        selection_rule=coupled_selection.decision_record.selection_rule,
    ))
    single_paths = _write_manifested_selection_outputs!(
        single_selection, paths.data, "coupled_single_lambda", context, (
        kind="matched_single_equation_lambda_selection",
        selected_lambda=single_selection.selected_lambda,
        selection_rule=single_selection.decision_record.selection_rule,
    ))
    selection_inputs = Dict{String,String}()
    for (prefix, selection_paths) in
        (("coupled", coupled_paths), ("coupled_single", single_paths))
        for field in propertynames(selection_paths)
            selection_inputs["$(prefix)_selection_$(field)"] =
                getproperty(selection_paths, field)
        end
    end
    context = merge(context, (selection_inputs=selection_inputs,))
    selection_record = (
        kind="coupled_and_matched_single_selection",
        coupled_lambda=coupled_selection.selected_lambda,
        single_lambda=single_selection.selected_lambda,
        training_storms=length(training),
    )
    _write_manifested_csv(joinpath(paths.data, "coupled_ae_secondary_candidates.csv"),
        coupled_selection.ae_candidate_records, context; selection_record)
    _write_manifested_csv(joinpath(paths.data, "coupled_ae_secondary_errors.csv"),
        coupled_selection.ae_error_records, context; selection_record)
    _write_manifested_csv(joinpath(paths.data, "coupled_equation_fit_audit.csv"),
        coupled_selection.equation_fit_records, context; selection_record)
    _write_manifested_csv(joinpath(paths.data, "coupled_single_inner_split_audit.csv"),
        split_audit, context; selection_record)
    _write_manifested_csv(joinpath(paths.data, "coupled_single_coefficients.csv"),
        [(
            term = get_term_names(base_lib)[i],
            coefficient = single_selection.model[i],
            selected_lambda = single_selection.selected_lambda,
            matched_ae_admitted_training_cohort = true,
        ) for i in eachindex(single_selection.model)],
        context; selection_record)

    design = _cached_coupled_design!(coupled_cache, training)
    diagnostics = _coupled_design_diagnostic_records(
        design, coupled_selection.model, get_term_names(coupled_lib),
        coupled_selection.selected_lambda,
    )
    _write_manifested_csv(joinpath(paths.data, "coupled_design_diagnostics.csv"),
        diagnostics, context; selection_record=merge(selection_record, (
            kind="actual_normalized_coupled_training_design",
            condition_warning_threshold=COUPLED_CONDITION_WARNING,
        )))
    draws = _paired_coupled_ensemble(design, coupled_selection.selected_lambda)
    terms = get_term_names(coupled_lib)
    coefficients = _coupled_coefficient_records(coupled_selection.model, draws,
                                                 terms, coupled_selection.selected_lambda)
    ensemble_record = merge(selection_record, (
        ensemble_seed=COUPLED_ENSEMBLE_SEED,
        ensemble_draws=COUPLED_ENSEMBLE_DRAWS,
        subsample_fraction=COUPLED_SUBSAMPLE_FRACTION,
    ))
    _write_manifested_csv(joinpath(paths.data, "coupled_coefficients.csv"),
        coefficients, context; selection_record=ensemble_record,
        seed=COUPLED_ENSEMBLE_SEED, deterministic=false)
    _write_manifested_csv(joinpath(paths.data, "coupled_ensemble_draws.csv"),
        _coupled_draw_records(draws, terms, coupled_selection.selected_lambda), context;
        selection_record=ensemble_record, seed=COUPLED_ENSEMBLE_SEED,
        deterministic=false)

    outer = filter(storm -> storm.entry.split in ("val", "test"), storms)
    for split in ("val", "test")
        any(storm -> storm.entry.split == split, outer) ||
            error("no coupled-policy-admitted storms in outer $split split")
    end
    isempty(intersect(training_ids, Set(storm.storm_id for storm in outer))) ||
        error("training and outer coupled storm cohorts overlap")
    metric_rows = NamedTuple[]
    cohort_rows = NamedTuple[]
    for storm in sort(outer; by=storm -> (storm.onset_time, storm.storm_id))
        scored = _score_coupled_storm(storm, coupled_selection.model,
                                      single_selection.model, base_lib)
        append!(metric_rows, scored.metrics)
        append!(cohort_rows, scored.cohorts)
    end
    _assert_exact_outer_cohorts(cohort_rows)
    outer_record = merge(selection_record, (
        kind="matched_outer_coupled_comparison",
        outer_storms=length(outer),
    ))
    _write_manifested_csv(joinpath(paths.data, "coupled_metrics.csv"), metric_rows,
        context; selection_record=outer_record)
    _write_manifested_csv(joinpath(paths.data, "coupled_cohort_audit.csv"), cohort_rows,
        context; selection_record=outer_record)

    for output in context.outputs
        verify_output_manifest(output;
            require_canonical=paths.mode == :canonical,
            verify_source=true,
        )
    end
    println("Coupled outputs verified under $(paths.data)")
    println("Coupled lambda: $(coupled_selection.selected_lambda); " *
            "matched single lambda: $(single_selection.selected_lambda)")
    return (
        coupled_selection=coupled_selection,
        single_selection=single_selection,
        n_training=length(training),
        n_outer=length(outer),
        outputs=copy(context.outputs),
    )
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && run_coupled_discovery()
