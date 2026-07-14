#!/usr/bin/env julia
# Canonical phase-dependent SINDy discovery and paired outer evaluation.

using SolarSINDy
using CSV, DataFrames, Dates, Statistics, Random, LinearAlgebra

isdefined(@__MODULE__, :validation_output_paths) ||
    include(joinpath(@__DIR__, "output_paths.jl"))
isdefined(@__MODULE__, :write_output_manifest) ||
    include(joinpath(@__DIR__, "canonical_provenance.jl"))
isdefined(@__MODULE__, :DiscoveryObservationPolicy) ||
    include(joinpath(@__DIR__, "real_discovery_helpers.jl"))

const PHASE_NAMES = ("quiet", "main", "recovery")
const PHASE_ENSEMBLE_DRAWS = 500
const PHASE_ENSEMBLE_SEED = 42
const PHASE_ENSEMBLE_SEEDS = NamedTuple{Symbol.(PHASE_NAMES)}(
    ntuple(phase -> PHASE_ENSEMBLE_SEED + phase - 1, length(PHASE_NAMES)),
)
const PHASE_MIN_REGRESSION_ROWS = 100
const PHASE_CONDITION_WARNING = 100.0
const PHASE_CLOCK_TERMS = (
    "sin(θ_c/2)", "sin²(θ_c/2)", "sin⁴(θ_c/2)",
    "sin^(8/3)(θ_c/2)", "V*sin²(θ_c/2)", "Newell_d_Φ",
)

function _phase_gate(dst_star::Real, previous_derivative::Real;
                     quiet_thresh::Real=-20.0,
                     deriv_thresh::Real=-2.0)
    all(isfinite, (dst_star, previous_derivative, quiet_thresh, deriv_thresh)) ||
        throw(ArgumentError("phase state, derivative, and thresholds must be finite"))
    return dst_star >= quiet_thresh ? 1 :
           previous_derivative < deriv_thresh ? 2 : 3
end

function _causal_phase_labels(dst_star::AbstractVector{<:Real}, dt::Real;
                              quiet_thresh::Real=-20.0,
                              deriv_thresh::Real=-2.0)
    isempty(dst_star) && throw(ArgumentError("phase labeling requires at least one row"))
    all(isfinite, dst_star) ||
        throw(ArgumentError("phase labeling requires finite Dst* values"))
    isfinite(dt) && dt > 0 ||
        throw(ArgumentError("phase-label dt must be finite and positive"))
    labels = Vector{Int}(undef, length(dst_star))
    @inbounds for index in eachindex(dst_star)
        previous_derivative = index == firstindex(dst_star) ? 0.0 :
            (dst_star[index] - dst_star[index - 1]) / dt
        labels[index] = _phase_gate(
            dst_star[index], previous_derivative; quiet_thresh, deriv_thresh,
        )
    end
    return labels
end

"""Forward-integrate the three-equation state-switching model."""
function _simulate_phase_switching(coefficients, lib::CandidateLibrary,
                                   swd::SolarWindData, dt::Real;
                                   quiet_thresh::Real=-20.0,
                                   deriv_thresh::Real=-2.0,
                                   Dst0::Real=swd.Dst_star[1])
    SolarSINDy._validate_candidate_library(lib)
    length(coefficients) == 3 ||
        throw(DimensionMismatch("phase model must contain three equations"))
    all(length(coefficient) == length(lib) for coefficient in coefficients) ||
        throw(DimensionMismatch("phase coefficient lengths must match the library"))
    n = length(swd.t)
    n >= 1 || throw(ArgumentError("phase simulation requires at least one row"))
    all(length(field) == n for field in
        (swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn, swd.Dst, swd.Dst_star)) ||
        throw(DimensionMismatch("solar-wind fields must have equal lengths"))
    all(all(isfinite, field) for field in
        (swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn)) ||
        throw(ArgumentError("phase simulation drivers must be finite"))
    all(>=(0), swd.V) && all(>=(0), swd.n) && all(>=(0), swd.Pdyn) ||
        throw(ArgumentError("speed, density, and dynamic pressure must be nonnegative"))
    all(coefficient -> all(isfinite, coefficient), coefficients) ||
        throw(ArgumentError("phase coefficients must be finite"))
    isfinite(dt) && dt > 0 ||
        throw(ArgumentError("dt must be finite and positive"))
    isfinite(quiet_thresh) && isfinite(deriv_thresh) && isfinite(Dst0) ||
        throw(ArgumentError("phase thresholds and initial Dst* must be finite"))

    prediction = zeros(Float64, n)
    theta = Vector{Float64}(undef, length(lib))
    prediction[1] = Float64(Dst0)
    for index in 1:(n - 1)
        previous_derivative = index == 1 ? 0.0 :
            (prediction[index] - prediction[index - 1]) / dt
        phase = _phase_gate(
            prediction[index], previous_derivative; quiet_thresh, deriv_thresh,
        )
        SolarSINDy._evaluate_point_vector_unchecked!(
            theta, lib, prediction[index], Float64(swd.V[index]),
            Float64(swd.Bz[index]), Float64(swd.By[index]),
            Float64(swd.n[index]), Float64(swd.Pdyn[index]),
        )
        derivative = clamp(dot(theta, coefficients[phase]), -200.0, 200.0)
        prediction[index + 1] = clamp(prediction[index] + dt * derivative,
                                      -2000.0, 50.0)
    end
    return prediction
end

function _with_phase_labels(storms::AbstractVector;
                            quiet_thresh::Real=-20.0,
                            deriv_thresh::Real=-2.0,
                            smooth_window::Int=5)
    return [begin
        full_data, full_target = prepare_sindy_data(
            storm.window_swd, 1.0; smooth_window,
        )
        length(storm.regression_mask) == length(full_target) ||
            throw(DimensionMismatch(
                "storm $(storm.storm_id) regression mask has the wrong length",
            ))
        # The regression target remains the centered derivative of smoothed Dst*,
        # but regime selection uses the causal previous-step derivative available
        # during a free-running switching simulation.
        phases = _causal_phase_labels(
            full_data["Dst_star"], 1.0; quiet_thresh, deriv_thresh,
        )
        labels = phases[storm.regression_mask]
        length(labels) == length(storm.target) ||
            throw(DimensionMismatch(
                "storm $(storm.storm_id) phase labels do not match regression rows",
            ))
        merge(storm, (phase_labels=labels,
                      phase_label_rule="smoothed_dst_star_with_backward_difference_gate",
                      quiet_thresh=Float64(quiet_thresh),
                      deriv_thresh=Float64(deriv_thresh)))
    end for storm in storms]
end

function _phase_design(storms::AbstractVector, n_terms::Int;
                       min_phase_rows::Int=max(n_terms, PHASE_MIN_REGRESSION_ROWS),
                       min_phase_storms::Int=2)
    isempty(storms) && throw(ArgumentError("phase fitting requires storms"))
    min_phase_rows >= 1 ||
        throw(ArgumentError("min_phase_rows must be positive"))
    min_phase_storms >= 1 ||
        throw(ArgumentError("min_phase_storms must be positive"))
    theta = Matrix{Float64}[]
    target = Vector{Float64}[]
    row_counts = Int[]
    storm_counts = Int[]
    for phase in 1:3
        matrices = Matrix{Float64}[]
        targets = Vector{Float64}[]
        storms_with_phase = 0
        for storm in storms
            length(storm.phase_labels) == length(storm.target) == size(storm.theta, 1) ||
                throw(DimensionMismatch(
                    "storm $(storm.storm_id) phase, target, and design rows differ",
                ))
            size(storm.theta, 2) == n_terms ||
                throw(DimensionMismatch("storm library width differs"))
            indices = findall(==(phase), storm.phase_labels)
            isempty(indices) && continue
            push!(matrices, Matrix(storm.theta[indices, :]))
            push!(targets, Float64.(storm.target[indices]))
            storms_with_phase += 1
        end
        rows = sum(length, targets; init=0)
        rows >= min_phase_rows || error(
            "$(PHASE_NAMES[phase]) phase has $rows regression rows; " *
            "need at least $min_phase_rows",
        )
        storms_with_phase >= min_phase_storms || error(
            "$(PHASE_NAMES[phase]) phase occurs in $storms_with_phase storms; " *
            "need at least $min_phase_storms",
        )
        phase_theta = reduce(vcat, matrices)
        phase_target = reduce(vcat, targets)
        all(isfinite, phase_theta) && all(isfinite, phase_target) ||
            error("$(PHASE_NAMES[phase]) phase design is non-finite")
        push!(theta, phase_theta)
        push!(target, phase_target)
        push!(row_counts, rows)
        push!(storm_counts, storms_with_phase)
    end
    return (; theta=Tuple(theta), target=Tuple(target),
            row_counts=Tuple(row_counts), storm_counts=Tuple(storm_counts))
end

function _select_phase_lambda(storms::AbstractVector, lib::CandidateLibrary;
                              quiet_thresh::Real=-20.0,
                              deriv_thresh::Real=-2.0,
                              min_phase_rows::Int=max(length(lib), PHASE_MIN_REGRESSION_ROWS),
                              min_phase_storms::Int=2)
    all(storm -> hasproperty(storm, :entry) && storm.entry.split == "train", storms) ||
        throw(ArgumentError(
            "phase lambda selection accepts only catalog training storms",
        ))
    terms = get_term_names(lib)
    cache = Dict{Tuple{Vararg{String}},Any}()
    fit = function (fit_storms, lambda)
        ids = Tuple(string(storm.storm_id) for storm in sort(
            collect(fit_storms); by=storm -> (storm.onset_time, storm.storm_id),
        ))
        design = get!(cache, ids) do
            _phase_design(fit_storms, length(lib);
                          min_phase_rows, min_phase_storms)
        end
        coefficients = ntuple(phase -> stlsq(
            design.theta[phase], design.target[phase];
            λ=lambda, normalize=true,
        ), 3)
        support = String[]
        for phase in 1:3, index in findall(!=(0.0), coefficients[phase])
            push!(support, "$(PHASE_NAMES[phase])::$(terms[index])")
        end
        model = (; coefficients, row_counts=design.row_counts,
                 storm_counts=design.storm_counts)
        return (; model, support)
    end
    integrate = function (model, storm, anchor_index, anchor_value)
        sliced = _slice_solar_wind(storm.swd, anchor_index)
        prediction = _simulate_phase_switching(
            model.coefficients, lib, sliced, 1.0;
            quiet_thresh, deriv_thresh, Dst0=anchor_value,
        )
        return prediction
    end
    selection = select_storm_lambda(
        storms; fit, integrate,
        observations=_discovery_scoring_observations,
    )
    decision = merge(selection.decision_record, (
        quiet_thresh=Float64(quiet_thresh),
        deriv_thresh=Float64(deriv_thresh),
    ))
    return merge(selection, (decision_record=decision,))
end

function _require_identical_inner_cohort(phase_selection, single_selection)
    phase_selection.split_records == single_selection.split_records || error(
        "phase and single-equation selectors used different inner storm cohorts",
    )
    return nothing
end

function _phase_scoring_context(storm)
    observations = _discovery_scoring_observations(storm)
    length(observations) == length(storm.swd.t) ||
        throw(DimensionMismatch("scoring observations and drivers differ"))
    all(value -> value isa Real && (isfinite(value) || isnan(value)), observations) ||
        throw(ArgumentError("scoring observations must be finite or NaN"))
    anchor = findfirst(isfinite, observations)
    anchor === nothing && error("storm $(storm.storm_id) has no observed anchor")
    swd = _slice_solar_wind(storm.swd, anchor)
    observed = observations[anchor:end]
    scored_indices = findall(isfinite, @view(observed[2:end])) .+ 1
    isempty(scored_indices) &&
        error("storm $(storm.storm_id) has no original post-anchor target")
    all(all(isfinite, field) for field in (swd.V, swd.Bz, swd.By, swd.n, swd.Pdyn)) ||
        error("storm $(storm.storm_id) has non-finite phase-scoring drivers")
    cohort = _discovery_cohort_identity(storm, anchor, scored_indices)
    return (; anchor, swd, observed, scored_indices, cohort)
end

function _score_phase_storm(storm, phase_model, single_coefficients,
                            lib::CandidateLibrary;
                            quiet_thresh::Real=-20.0,
                            deriv_thresh::Real=-2.0)
    context = _phase_scoring_context(storm)
    (; anchor, swd, observed, scored_indices, cohort) = context
    bs = max.(-swd.Bz, 0.0)
    switching = _simulate_phase_switching(
        phase_model.coefficients, lib, swd, 1.0;
        quiet_thresh, deriv_thresh, Dst0=observed[1],
    )
    predictions = (
        SwitchingSINDy = switching,
        SingleSINDy = simulate_sindy(
            single_coefficients, lib, swd, 1.0; Dst0=observed[1],
        ),
        Burton = simulate_burton(swd.V, bs, 1.0; Dst0=observed[1]),
        BurtonFull = simulate_burton_full(swd.V, bs, 1.0; Dst0=observed[1]),
        OBrienMcP = simulate_obrien(swd.V, bs, 1.0; Dst0=observed[1]),
    )
    model_labels = (
        SwitchingSINDy="Switching-SINDy",
        SingleSINDy="Single-SINDy",
        Burton="Burton",
        BurtonFull="BurtonFull",
        OBrienMcP="OBrien-McPherron",
    )
    score_offsets = join(scored_indices, ";")
    absolute_anchor = storm.observation_record.scoring_start_idx + anchor - 1
    absolute_rows = join(absolute_anchor .+ scored_indices .- 1, ";")
    rows = NamedTuple[]
    for model in propertynames(predictions)
        prediction = getproperty(predictions, model)
        length(prediction) == length(observed) ||
            throw(DimensionMismatch("$model prediction length differs"))
        all(isfinite, prediction) || error("$model prediction is non-finite")
        first(prediction) == first(observed) ||
            error("$model did not use the common observed anchor")
        model_label = getproperty(model_labels, model)
        metrics = metrics_summary(
            prediction[scored_indices], observed[scored_indices];
            name=model_label,
        )
        push!(rows, (
            split = storm.entry.split,
            storm_id = storm.storm_id,
            onset_time = string(storm.onset_time),
            model = model_label,
            rmse_nt = metrics.rmse,
            mae_nt = metrics.mae,
            correlation = metrics.corr,
            pe = metrics.pe,
            min_dst_star_observed_nt = minimum(observed[isfinite.(observed)]),
            anchor_index = absolute_anchor,
            n_points = length(scored_indices),
            scored_row_offsets = score_offsets,
            scored_absolute_rows = absolute_rows,
            cohort_signature_sha256 = cohort.cohort_signature_sha256,
            quiet_thresh = Float64(quiet_thresh),
            deriv_thresh = Float64(deriv_thresh),
        ))
    end
    return (; rows, predictions, observations=observed, swd,
            scored_indices, anchor_index=absolute_anchor)
end

function _paired_phase_metrics(rows::AbstractVector)
    frame = DataFrame(rows)
    output = NamedTuple[]
    for group in groupby(frame, [:split, :storm_id])
        nrow(group) == 5 || error(
            "storm $(group.storm_id[1]) does not have all five paired models",
        )
        length(unique(group.anchor_index)) == 1 &&
        length(unique(group.n_points)) == 1 &&
        length(unique(group.scored_absolute_rows)) == 1 &&
        length(unique(group.cohort_signature_sha256)) == 1 || error(
            "storm $(group.storm_id[1]) models were not scored on identical rows",
        )
        by_model = Dict(String(row.model) => row for row in eachrow(group))
        Set(keys(by_model)) == Set(("Switching-SINDy", "Single-SINDy",
                                    "Burton", "BurtonFull",
                                    "OBrien-McPherron")) ||
            error("paired model set is incomplete")
        switch = by_model["Switching-SINDy"]
        push!(output, (
            split = String(switch.split),
            storm_id = Int(switch.storm_id),
            onset_time = String(switch.onset_time),
            anchor_index = Int(switch.anchor_index),
            n_points = Int(switch.n_points),
            scored_absolute_rows = String(switch.scored_absolute_rows),
            cohort_signature_sha256 = String(switch.cohort_signature_sha256),
            quiet_thresh = Float64(switch.quiet_thresh),
            deriv_thresh = Float64(switch.deriv_thresh),
            switching_rmse_nt = Float64(switch.rmse_nt),
            single_rmse_nt = Float64(by_model["Single-SINDy"].rmse_nt),
            burton_rmse_nt = Float64(by_model["Burton"].rmse_nt),
            burton_published_rmse_nt = Float64(by_model["BurtonFull"].rmse_nt),
            obrien_rmse_nt = Float64(by_model["OBrien-McPherron"].rmse_nt),
            switching_minus_single_rmse_nt = Float64(
                switch.rmse_nt - by_model["Single-SINDy"].rmse_nt),
            switching_minus_burton_rmse_nt = Float64(
                switch.rmse_nt - by_model["Burton"].rmse_nt),
            switching_minus_burton_published_rmse_nt = Float64(
                switch.rmse_nt - by_model["BurtonFull"].rmse_nt),
            switching_minus_obrien_rmse_nt = Float64(
                switch.rmse_nt - by_model["OBrien-McPherron"].rmse_nt),
            switching_pe = Float64(switch.pe),
            single_pe = Float64(by_model["Single-SINDy"].pe),
            burton_pe = Float64(by_model["Burton"].pe),
            burton_published_pe = Float64(by_model["BurtonFull"].pe),
            obrien_pe = Float64(by_model["OBrien-McPherron"].pe),
        ))
    end
    return sort!(output; by=row -> (row.split, row.storm_id))
end

function _phase_conditional_quantiles(nonzero)
    isempty(nonzero) && return (NaN, NaN, NaN)
    return (quantile(nonzero, 0.025), median(nonzero), quantile(nonzero, 0.975))
end

function _phase_subsample_coefficients(storms::AbstractVector,
                                       model, lib::CandidateLibrary;
                                       draws::Int=PHASE_ENSEMBLE_DRAWS,
                                       seed::Int=PHASE_ENSEMBLE_SEED,
                                       subsample_fraction::Real=0.8,
                                       quiet_thresh::Real=-20.0,
                                       deriv_thresh::Real=-2.0)
    draws >= 1 || throw(ArgumentError("draws must be positive"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    isfinite(subsample_fraction) && 0 < subsample_fraction <= 1 ||
        throw(ArgumentError("subsample_fraction must be in (0, 1]"))
    design = _phase_design(storms, length(lib))
    names = get_term_names(lib)
    rows = NamedTuple[]
    for phase in 1:3
        theta = design.theta[phase]
        target = design.target[phase]
        n_subsample = max(1, round(Int, size(theta, 1) * subsample_fraction))
        rng = MersenneTwister(seed + phase - 1)
        coefficients = zeros(length(lib), draws)
        for draw in 1:draws
            indices = sort(randperm(rng, size(theta, 1))[1:n_subsample])
            coefficients[:, draw] = stlsq(
                theta[indices, :], target[indices];
                λ=model.selected_lambda, normalize=true,
            )
        end
        for index in eachindex(names)
            nonzero = coefficients[index, coefficients[index, :] .!= 0.0]
            q025, q50, q975 = _phase_conditional_quantiles(nonzero)
            push!(rows, (
                phase = PHASE_NAMES[phase],
                term = names[index],
                point_coefficient = model.model.coefficients[phase][index],
                inclusion_fraction = length(nonzero) / draws,
                conditional_nonzero_median = q50,
                conditional_nonzero_quantile_025 = q025,
                conditional_nonzero_quantile_975 = q975,
                interval_kind = "empirical_conditional_nonzero_row_subsample_quantile_not_confidence_interval",
                selected_lambda = model.selected_lambda,
                draws = draws,
                subsample_fraction = Float64(subsample_fraction),
                seed = seed + phase - 1,
                n_phase_rows = size(theta, 1),
                quiet_thresh = Float64(quiet_thresh),
                deriv_thresh = Float64(deriv_thresh),
            ))
        end
    end
    return rows
end

function _phase_design_diagnostic_record(theta::AbstractMatrix,
                                         coefficients::AbstractVector,
                                         term_names::AbstractVector,
                                         indices::AbstractVector{Int};
                                         phase::AbstractString,
                                         block::AbstractString,
                                         selected_lambda::Real)
    size(theta, 2) == length(coefficients) == length(term_names) ||
        throw(DimensionMismatch("phase diagnostic inputs must align"))
    all(index -> 1 <= index <= size(theta, 2), indices) ||
        throw(BoundsError(term_names, indices))
    all(isfinite, theta) && all(isfinite, coefficients) ||
        throw(ArgumentError("phase diagnostic inputs must be finite"))

    normalized_rank = 0
    rank_tolerance = NaN
    condition_number = NaN
    largest_singular_value = NaN
    smallest_singular_value = NaN
    if !isempty(indices)
        design = Matrix{Float64}(@view theta[:, indices])
        norms = [norm(@view design[:, column]) for column in axes(design, 2)]
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
    elseif condition_number > PHASE_CONDITION_WARNING
        "ill_conditioned_grouped_net_preferred"
    elseif length(indices) == 1
        "single_basis_term"
    else
        "full_rank_basis_dependent_group"
    end
    return (
        phase=String(phase),
        block=String(block),
        terms=isempty(indices) ? "none" : join(term_names[indices], ";"),
        row_scope="phase_training_regression_rows",
        contribution_units="nT_per_hour",
        n_rows=size(theta, 1),
        n_columns=length(indices),
        normalized_design_rank=normalized_rank,
        normalized_design_rank_tolerance=rank_tolerance,
        normalized_design_condition_number=condition_number,
        largest_normalized_singular_value=largest_singular_value,
        smallest_normalized_singular_value=smallest_singular_value,
        n_active_terms=count(index -> coefficients[index] != 0.0, indices),
        net_contribution_min=isempty(net) ? 0.0 : minimum(net),
        net_contribution_max=isempty(net) ? 0.0 : maximum(net),
        net_contribution_absmax=net_absmax,
        gross_contribution_absmax=gross_absmax,
        cancellation_ratio=cancellation,
        condition_warning_threshold=PHASE_CONDITION_WARNING,
        interpretation_status=status,
        selected_lambda=Float64(selected_lambda),
    )
end

function _phase_design_diagnostic_records(storms::AbstractVector,
                                          selection,
                                          lib::CandidateLibrary;
                                          min_phase_rows::Int=max(
                                              length(lib), PHASE_MIN_REGRESSION_ROWS,
                                          ),
                                          min_phase_storms::Int=2)
    design = _phase_design(
        storms, length(lib); min_phase_rows, min_phase_storms,
    )
    term_names = get_term_names(lib)
    clock = findall(in(PHASE_CLOCK_TERMS), term_names)
    rows = NamedTuple[]
    for phase in 1:3
        coefficients = selection.model.coefficients[phase]
        active = findall(!iszero, coefficients)
        active_clock = intersect(active, clock)
        for (block, indices) in (
            "full_design" => collect(eachindex(term_names)),
            "selected_active_block" => active,
            "clock_candidate_block" => clock,
            "selected_clock_block" => active_clock,
        )
            push!(rows, _phase_design_diagnostic_record(
                design.theta[phase], coefficients, term_names, collect(indices);
                phase=PHASE_NAMES[phase], block,
                selected_lambda=selection.selected_lambda,
            ))
        end
    end
    return rows
end

function _phase_cohort_records(storms::AbstractVector, inner_split_by_id)
    return [begin
        scores = findall(isfinite, storm.scoring_observations)
        phase_counts = ntuple(phase -> count(==(phase), storm.phase_labels), 3)
        catalog_split = String(storm.entry.split)
        analysis_role = haskey(inner_split_by_id, string(storm.storm_id)) ?
            String(inner_split_by_id[string(storm.storm_id)]) :
            catalog_split in ("val", "test") ? "outer" : "excluded"
        used = catalog_split in ("train", "val", "test")
        (
            storm_id = storm.storm_id,
            onset_time = string(storm.onset_time),
            catalog_split = catalog_split,
            inner_split = analysis_role,
            solar_cycle = storm.entry.solar_cycle,
            regression_rows = length(storm.target),
            quiet_regression_rows = phase_counts[1],
            main_regression_rows = phase_counts[2],
            recovery_regression_rows = phase_counts[3],
            scoring_rows = max(length(scores) - 1, 0),
            scoring_start_idx = storm.observation_record.scoring_start_idx,
            scoring_end_idx = storm.observation_record.scoring_end_idx,
            original_scoring_row_offsets = join(scores, ";"),
            quiet_thresh = storm.quiet_thresh,
            deriv_thresh = storm.deriv_thresh,
            phase_label_rule = storm.phase_label_rule,
            used_by_switching = used,
            used_by_single = used,
            used_by_burton = catalog_split in ("val", "test"),
            used_by_burton_published = catalog_split in ("val", "test"),
            used_by_obrien = catalog_split in ("val", "test"),
        )
    end for storm in sort(collect(storms); by=storm -> storm.storm_id)]
end

function _phase_write(path::AbstractString, rows;
                      output_paths, producer_script::AbstractString,
                      inputs, selection_record,
                      seed::Union{Nothing,Int}=nothing,
                      deterministic::Bool=false,
                      metadata=(;))
    transaction_paths = [String(path), String(path) * ".manifest.json"]
    snapshot = SolarSINDy._snapshot_regular_file_set(transaction_paths)
    try
        _write_discovery_csv(path, rows)
        write_output_manifest(
            path; producer_script, input_paths=inputs, selection_record,
            seed, deterministic, metadata, mode=output_paths.mode,
        )
        verify_output_manifest(
            path; require_canonical=output_paths.mode == :canonical,
        )
    catch
        SolarSINDy._restore_regular_file_set!(snapshot)
        rethrow()
    end
    SolarSINDy._discard_regular_file_snapshot!(snapshot)
    return path
end

function _phase_manifest_selection(selection, prefix::AbstractString;
                                   output_paths, producer_script, inputs,
                                   kind::AbstractString,
                                   _after_manifest_hook::Function=(field, path) -> nothing)
    isempty(prefix) && throw(ArgumentError("prefix must not be empty"))
    basename(prefix) == prefix ||
        throw(ArgumentError("prefix must be a file-name prefix, not a path"))
    paths = (
        split=joinpath(output_paths.data, "$(prefix)_inner_split.csv"),
        candidates=joinpath(output_paths.data, "$(prefix)_candidates.csv"),
        errors=joinpath(output_paths.data, "$(prefix)_validation_errors.csv"),
        support=joinpath(output_paths.data, "$(prefix)_support.csv"),
        decision=joinpath(output_paths.data, "$(prefix)_decision.csv"),
    )
    transaction_paths = vcat(
        String[getproperty(paths, field) for field in propertynames(paths)],
        String[getproperty(paths, field) * ".manifest.json"
               for field in propertynames(paths)],
    )
    snapshot = SolarSINDy._snapshot_regular_file_set(transaction_paths)
    record = (kind=kind, decision=selection.decision_record)
    try
        write_storm_lambda_selection(selection, output_paths.data; prefix)
        SolarSINDy._with_selection_csv_set_lock(paths) do
            for field in propertynames(paths)
                path = getproperty(paths, field)
                write_output_manifest(
                    path; producer_script, input_paths=inputs,
                    selection_record=record, deterministic=true,
                    mode=output_paths.mode,
                )
                verify_output_manifest(
                    path; require_canonical=output_paths.mode == :canonical,
                )
                _after_manifest_hook(field, path)
            end
        end
    catch
        SolarSINDy._restore_regular_file_set!(snapshot)
        rethrow()
    end
    SolarSINDy._discard_regular_file_snapshot!(snapshot)
    return paths
end

function run_phase_dependent_discovery()
    isempty(strip(get(ENV, "SOLARSINDY_OUTPUT_ROOT", ""))) && error(
        "SOLARSINDY_OUTPUT_ROOT must be set for canonical phase discovery",
    )
    output_paths = validation_output_paths()
    omni = output_paths.omni
    data_dir = output_paths.data
    producer = @__FILE__
    isfile(omni) || error("frozen OMNI extraction not found: $omni")
    verify_omni_input(omni; mode=output_paths.mode)

    df = parse_omni2(omni; year_start=1963, year_end=2025)
    add_original_observation_flags!(df)
    clean_omni_data!(df)
    catalog_path = joinpath(data_dir, "storm_catalog.csv")
    catalog = load_verified_storm_catalog(
        catalog_path; omni_path=omni,
        parameters=storm_catalog_parameters(), mode=output_paths.mode,
    )
    inputs = (omni_extracted=omni, storm_catalog=catalog_path)
    policy = DiscoveryObservationPolicy()
    audit = _audit_discovery_observations(df, catalog; policy)
    selection_record = (
        kind="discovery_observation_policy",
        smooth_window=policy.smooth_window,
        min_regression_rows=policy.min_regression_rows,
        min_scoring_rows=policy.min_scoring_rows,
        min_scoring_fraction=policy.min_scoring_fraction,
    )
    storm_audit_path = _phase_write(
        joinpath(data_dir, "phase_observation_audit.csv"),
        audit.storm_records; output_paths, producer_script=producer,
        inputs, selection_record, deterministic=true)
    cycle_audit_path = _phase_write(
        joinpath(data_dir, "phase_observation_cycle_audit.csv"),
        audit.cycle_records; output_paths, producer_script=producer,
        inputs, selection_record, deterministic=true)
    selection_inputs = Dict(
        "omni_extracted" => omni,
        "storm_catalog" => catalog_path,
        "phase_storm_observation_audit" => storm_audit_path,
        "phase_cycle_observation_audit" => cycle_audit_path,
    )

    lib = build_solar_wind_library()
    length(lib) == 20 || error("phase discovery requires the 20-term library")
    "n*V^2" in get_term_names(lib) &&
        error("phase discovery library contains redundant n*V^2")
    storms = _prepare_discovery_storms(
        df, audit.eligible_entries, lib; policy,
    )
    storms = _with_phase_labels(storms)
    training = filter(storm -> storm.entry.split == "train", storms)
    length(training) >= 2 || error("phase selection needs at least two training storms")

    phase_selection = _select_phase_lambda(training, lib)
    single_selection = _select_discovery_lambda(training, lib, Dict{Any,Any}())
    _require_identical_inner_cohort(phase_selection, single_selection)
    phase_paths = _phase_manifest_selection(
        phase_selection, "phase_switching_lambda";
        output_paths, producer_script=producer, inputs=selection_inputs,
        kind="state_switching_shared_lambda",
    )
    single_paths = _phase_manifest_selection(
        single_selection, "phase_single_lambda";
        output_paths, producer_script=producer, inputs=selection_inputs,
        kind="same_cohort_single_equation_control",
    )
    analysis_inputs = copy(selection_inputs)
    for (prefix, paths) in (("phase", phase_paths), ("phase_single", single_paths))
        for field in propertynames(paths)
            analysis_inputs["$(prefix)_selection_$(field)"] = getproperty(paths, field)
        end
    end

    inner_split_by_id = Dict(
        row.storm_id => row.inner_split for row in phase_selection.split_records
    )
    _phase_write(joinpath(data_dir, "phase_cohort_audit.csv"),
                 _phase_cohort_records(storms, inner_split_by_id);
                 output_paths, producer_script=producer, inputs=analysis_inputs,
                 selection_record=(
                    kind="identical_phase_single_baseline_cohort",
                    phase_decision=phase_selection.decision_record,
                    single_decision=single_selection.decision_record,
                 ), deterministic=true)

    coefficient_rows = _phase_subsample_coefficients(
        training, phase_selection, lib,
    )
    _phase_write(joinpath(data_dir, "phase_dependent_real_coefficients.csv"),
                 coefficient_rows; output_paths, producer_script=producer,
                 inputs=analysis_inputs, selection_record=(
                    kind="phase_coefficients_after_storm_selection",
                    decision=phase_selection.decision_record,
                    phase_seed_map=PHASE_ENSEMBLE_SEEDS,
                 ), seed=PHASE_ENSEMBLE_SEED,
                 metadata=(draws=PHASE_ENSEMBLE_DRAWS,
                           phase_seed_map=PHASE_ENSEMBLE_SEEDS))
    diagnostic_rows = _phase_design_diagnostic_records(
        training, phase_selection, lib,
    )
    _phase_write(joinpath(data_dir, "phase_design_diagnostics.csv"),
                 diagnostic_rows; output_paths, producer_script=producer,
                 inputs=analysis_inputs, selection_record=(
                    kind="actual_normalized_phase_training_design",
                    decision=phase_selection.decision_record,
                    coefficient_interpretation=
                        "grouped_net_when_rank_deficient_or_ill_conditioned",
                 ), deterministic=true)
    names = get_term_names(lib)
    _phase_write(joinpath(data_dir, "phase_single_control_coefficients.csv"),
                 [(term=names[index], coefficient=single_selection.model[index],
                   selected_lambda=single_selection.selected_lambda)
                  for index in eachindex(names)];
                 output_paths, producer_script=producer, inputs=analysis_inputs,
                 selection_record=(
                    kind="same_cohort_single_equation_control",
                    decision=single_selection.decision_record,
                 ), deterministic=true)

    metric_rows = NamedTuple[]
    for split in ("val", "test")
        outer = filter(storm -> storm.entry.split == split, storms)
        isempty(outer) && error("$split has no eligible phase-evaluation storms")
        for storm in outer
            append!(metric_rows, _score_phase_storm(
                storm, phase_selection.model, single_selection.model, lib,
            ).rows)
        end
    end
    paired_rows = _paired_phase_metrics(metric_rows)
    metric_selection = (
        kind="paired_outer_phase_evaluation",
        phase_decision=phase_selection.decision_record,
        single_decision=single_selection.decision_record,
        models=("Switching-SINDy", "Single-SINDy", "Burton", "BurtonFull",
                "OBrien-McPherron"),
    )
    _phase_write(joinpath(data_dir, "switching_model_metrics.csv"), metric_rows;
                 output_paths, producer_script=producer, inputs=analysis_inputs,
                 selection_record=metric_selection, deterministic=true)
    _phase_write(joinpath(data_dir, "switching_model_paired_metrics.csv"), paired_rows;
                 output_paths, producer_script=producer, inputs=analysis_inputs,
                 selection_record=metric_selection, deterministic=true)

    println("Phase-dependent discovery outputs written under: $data_dir")
    println("Shared switching lambda: $(phase_selection.selected_lambda)")
    println("Same-cohort single lambda: $(single_selection.selected_lambda)")
    return (; phase_selection, single_selection, diagnostic_rows,
            metric_rows, paired_rows)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_phase_dependent_discovery()
end
