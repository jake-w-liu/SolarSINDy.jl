#!/usr/bin/env julia
# Canonical real-data SINDy discovery and untouched outer evaluation.

using SolarSINDy
using CSV, DataFrames, Dates, Statistics, Random, LinearAlgebra

isdefined(@__MODULE__, :validation_output_paths) ||
    include(joinpath(@__DIR__, "output_paths.jl"))
isdefined(@__MODULE__, :write_output_manifest) ||
    include(joinpath(@__DIR__, "canonical_provenance.jl"))
isdefined(@__MODULE__, :DiscoveryObservationPolicy) ||
    include(joinpath(@__DIR__, "real_discovery_helpers.jl"))

const _REAL_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const _REAL_ENSEMBLE_SEED = 42
const _REAL_ENSEMBLE_DRAWS = 500
const _REAL_CLOCK_PROXY_TERMS = (
    "sin(θ_c/2)", "sin²(θ_c/2)", "sin⁴(θ_c/2)",
    "sin^(8/3)(θ_c/2)", "V*sin²(θ_c/2)",
)
const _REAL_CLOCK_RESPONSE_TERMS = (_REAL_CLOCK_PROXY_TERMS..., "Newell_d_Φ")
const _REAL_CONDITION_WARNING = 100.0

function _real_output_context()
    paths = validation_output_paths()
    paths.explicit || error(
        "SOLARSINDY_OUTPUT_ROOT must be set; canonical revision outputs may not use package data/",
    )
    paths.mode in (:canonical, :test) || error(
        "real_data_discovery.jl requires canonical mode (or explicit test mode)",
    )
    catalog = joinpath(paths.data, "storm_catalog.csv")
    isfile(paths.omni) || error("frozen OMNI extraction not found: $(paths.omni)")
    isfile(catalog) || error("verified storm catalog not found: $catalog")
    return merge(paths, (; catalog))
end

_real_base_inputs(context) = Dict(
    "omni_extracted" => context.omni,
    "storm_catalog" => context.catalog,
)

function _real_merge_inputs(context, extra_inputs)
    inputs = _real_base_inputs(context)
    for pair in pairs(extra_inputs)
        name = String(pair.first)
        haskey(inputs, name) && throw(ArgumentError("duplicate manifest input name: $name"))
        inputs[name] = String(pair.second)
    end
    return inputs
end

function _real_manifested_csv(context, filename::AbstractString, data;
                              selection_record,
                              seed::Union{Nothing,Int}=nothing,
                              extra_inputs=(;), metadata=(;),
                              producer_script::AbstractString=@__FILE__)
    basename(filename) == filename ||
        throw(ArgumentError("filename must not contain a path"))
    path = joinpath(context.data, filename)
    frame = data isa DataFrame ? data : DataFrame(data)
    return write_manifested_csv(path, frame;
        producer_script,
        input_paths=_real_merge_inputs(context, extra_inputs),
        selection_record,
        seed,
        deterministic=seed === nothing,
        metadata,
        package_root=_REAL_PACKAGE_ROOT,
        mode=context.mode,
        verify_source=true,
    )
end

function _real_policy_record(policy::DiscoveryObservationPolicy)
    return (
        smooth_window=policy.smooth_window,
        min_regression_rows=policy.min_regression_rows,
        min_scoring_rows=policy.min_scoring_rows,
        min_scoring_fraction=policy.min_scoring_fraction,
        require_ae=policy.require_ae,
        maximum_admissible_cleaned_gap_hours=3,
        regression_target_policy="original_Dst_V_n_full_smoothing_derivative_stencil",
        scoring_target_policy="original_Dst_V_n_derived_Dst_star_post_anchor_only",
    )
end

function _real_explicit_exclusion_records(records)
    return [merge(record, (
        exclusion_reason=isempty(strip(String(record.exclusion_reason))) ?
            "none" : String(record.exclusion_reason),
    )) for record in records]
end

function _real_selection_record(selection; basis, scope)
    return (
        kind="fixed_grid_whole_storm_forward_selection",
        basis=String(basis),
        scope=String(scope),
        grid="10.^range(-2,4,length=60)",
        normalize=true,
        decision=selection.decision_record,
    )
end

function _real_persist_selection(context, selection, prefix;
                                 basis, scope, extra_inputs=(;))
    paths = (
        split=joinpath(context.data, "$(prefix)_inner_split.csv"),
        candidates=joinpath(context.data, "$(prefix)_candidates.csv"),
        errors=joinpath(context.data, "$(prefix)_validation_errors.csv"),
        support=joinpath(context.data, "$(prefix)_support.csv"),
        decision=joinpath(context.data, "$(prefix)_decision.csv"),
    )
    transaction_paths = vcat(
        String[getproperty(paths, field) for field in propertynames(paths)],
        String[getproperty(paths, field) * ".manifest.json"
               for field in propertynames(paths)],
    )
    snapshot = SolarSINDy._snapshot_regular_file_set(transaction_paths)
    record = _real_selection_record(selection; basis, scope)
    try
        write_storm_lambda_selection(selection, context.data; prefix)
        SolarSINDy._with_selection_csv_set_lock(paths) do
            for field in propertynames(paths)
                path = getproperty(paths, field)
                frame = CSV.read(path, DataFrame)
                write_output_manifest(path;
                    producer_script=@__FILE__,
                    input_paths=_real_merge_inputs(context, extra_inputs),
                    selection_record=merge(record, (artifact=String(field),)),
                    deterministic=true,
                    metadata=(rows=nrow(frame), columns=names(frame)),
                    package_root=_REAL_PACKAGE_ROOT,
                    mode=context.mode,
                )
                verify_output_manifest(path;
                    package_root=_REAL_PACKAGE_ROOT,
                    require_canonical=context.mode == :canonical,
                )
            end
        end
    catch
        SolarSINDy._restore_regular_file_set!(snapshot)
        rethrow()
    end
    SolarSINDy._discard_regular_file_snapshot!(snapshot)
    return paths
end

function _real_selection_inputs(context, paths, prefix::AbstractString)
    return SolarSINDy._with_selection_csv_set_lock(paths) do
        inputs = Dict{String,String}()
        for field in propertynames(paths)
            path = getproperty(paths, field)
            verify_output_manifest(path;
                package_root=_REAL_PACKAGE_ROOT,
                require_canonical=context.mode == :canonical,
            )
            inputs["$(prefix)_$(field)"] = path
        end
        inputs
    end
end

function _real_require_subset(storms, predicate, label; minimum=1)
    subset = filter(predicate, storms)
    length(subset) >= minimum ||
        error("$label has $(length(subset)) eligible storms; need at least $minimum")
    return subset
end

function _real_assert_matching_cohort(full_storms, collapsed_storms, label)
    full_ordered = sort(collect(full_storms); by=s -> (s.onset_time, s.storm_id))
    collapsed_ordered = sort(collect(collapsed_storms); by=s -> (s.onset_time, s.storm_id))
    getproperty.(full_ordered, :storm_id) == getproperty.(collapsed_ordered, :storm_id) ||
        error("$label full and collapsed storm cohorts differ")
    for (full, collapsed) in zip(full_ordered, collapsed_ordered)
        full.entry.split == collapsed.entry.split ||
            error("$label catalog splits differ for storm $(full.storm_id)")
        full.regression_mask == collapsed.regression_mask ||
            error("$label regression masks differ for storm $(full.storm_id)")
        isequal(full.target, collapsed.target) ||
            error("$label regression targets differ for storm $(full.storm_id)")
        Set(keys(full.data)) == Set(keys(collapsed.data)) &&
            all(key -> isequal(full.data[key], collapsed.data[key]), keys(full.data)) ||
            error("$label regression inputs differ for storm $(full.storm_id)")
        isequal(full.scoring_observations, collapsed.scoring_observations) ||
            error("$label scoring targets differ for storm $(full.storm_id)")
        (full.observation_record.scoring_start_idx,
         full.observation_record.scoring_end_idx) ==
            (collapsed.observation_record.scoring_start_idx,
             collapsed.observation_record.scoring_end_idx) ||
            error("$label scoring bounds differ for storm $(full.storm_id)")
        all(field -> isequal(getproperty(full.window_swd, field),
                             getproperty(collapsed.window_swd, field)),
            (:t, :V, :Bz, :By, :n, :Pdyn, :Dst, :Dst_star)) ||
            error("$label source windows differ for storm $(full.storm_id)")
        all(field -> isequal(getproperty(full.swd, field),
                             getproperty(collapsed.swd, field)),
            (:t, :V, :Bz, :By, :n, :Pdyn, :Dst, :Dst_star)) ||
            error("$label scoring inputs differ for storm $(full.storm_id)")
        size(full.theta, 1) == size(collapsed.theta, 1) == length(full.target) ||
            error("$label regression design sizes differ for storm $(full.storm_id)")
    end
    return nothing
end

function _real_assert_matching_inner_split(full_selection, collapsed_selection, label)
    full_rows = sort(full_selection.split_records; by=row -> row.chronological_index)
    collapsed_rows = sort(collapsed_selection.split_records;
                          by=row -> row.chronological_index)
    [(row.storm_id, row.inner_split) for row in full_rows] ==
        [(row.storm_id, row.inner_split) for row in collapsed_rows] ||
        error("$label full and collapsed inner storm splits differ")
    length(full_selection.candidate_records) == 60 ||
        error("$label full selector did not evaluate the fixed 60-point grid")
    length(collapsed_selection.candidate_records) == 60 ||
        error("$label collapsed selector did not evaluate the fixed 60-point grid")
    return nothing
end

function _real_select_basis_pair(full_storms, collapsed_storms,
                                 full_library, collapsed_library,
                                 full_cache, collapsed_cache; label)
    _real_assert_matching_cohort(full_storms, collapsed_storms, label)
    full_selection = _select_discovery_lambda(full_storms, full_library, full_cache)
    collapsed_selection = _select_discovery_lambda(
        collapsed_storms, collapsed_library, collapsed_cache,
    )
    _real_assert_matching_inner_split(full_selection, collapsed_selection, label)
    return (; full=full_selection, collapsed=collapsed_selection)
end

function _real_normalized_block(theta::AbstractMatrix, indices::AbstractVector{Int})
    isempty(indices) && return (
        n_rows=size(theta, 1), n_columns=0, rank=0, rank_tolerance=NaN,
        condition_number=NaN, largest_singular_value=NaN,
        smallest_singular_value=NaN, zero_norm_columns=0,
    )
    block = Matrix{Float64}(@view theta[:, indices])
    norms = [norm(@view block[:, column]) for column in axes(block, 2)]
    zero_norm_columns = count(iszero, norms)
    scaling = copy(norms)
    scaling[iszero.(scaling)] .= 1.0
    normalized = block ./ scaling'
    singular_values = svdvals(normalized)
    largest = isempty(singular_values) ? 0.0 : first(singular_values)
    tolerance = max(size(normalized)...) * eps(Float64) * largest
    numerical_rank = count(value -> value > tolerance, singular_values)
    column_rank_deficient = numerical_rank < size(normalized, 2)
    condition_number = if largest == 0.0 || column_rank_deficient
        Inf
    else
        largest / last(singular_values)
    end
    return (
        n_rows=size(normalized, 1),
        n_columns=size(normalized, 2),
        rank=numerical_rank,
        rank_tolerance=tolerance,
        condition_number=condition_number,
        largest_singular_value=largest,
        smallest_singular_value=last(singular_values),
        zero_norm_columns=zero_norm_columns,
    )
end

function _real_block_record(theta, coefficients, term_names, indices, block, basis)
    diagnostics = _real_normalized_block(theta, indices)
    selected = count(index -> coefficients[index] != 0.0, indices)
    return merge((
        basis=String(basis),
        block=String(block),
        terms=isempty(indices) ? "none" : join(term_names[indices], ";"),
        active_terms=selected,
        normalization="training_column_l2_norm",
        rank_rule="singular_value > max(n_rows,n_columns)*eps(Float64)*largest_singular_value",
        condition_warning_threshold=_REAL_CONDITION_WARNING,
        ill_conditioned=!isnan(diagnostics.condition_number) &&
            diagnostics.condition_number > _REAL_CONDITION_WARNING,
    ), diagnostics)
end

function _real_conditioning_records(theta, coefficients, term_names; basis)
    size(theta, 2) == length(coefficients) == length(term_names) ||
        throw(DimensionMismatch("design, coefficients, and term names must align"))
    active = findall(!iszero, coefficients)
    proxy = findall(in(_REAL_CLOCK_PROXY_TERMS), term_names)
    response = findall(in(_REAL_CLOCK_RESPONSE_TERMS), term_names)
    active_proxy = intersect(active, proxy)
    active_response = intersect(active, response)
    return [
        _real_block_record(theta, coefficients, term_names,
                           collect(axes(theta, 2)), "full_design", basis),
        _real_block_record(theta, coefficients, term_names,
                           active, "selected_active_block", basis),
        _real_block_record(theta, coefficients, term_names,
                           proxy, "clock_proxy_candidate_block", basis),
        _real_block_record(theta, coefficients, term_names,
                           response, "clock_response_candidate_block", basis),
        _real_block_record(theta, coefficients, term_names,
                           active_proxy, "selected_clock_proxy_block", basis),
        _real_block_record(theta, coefficients, term_names,
                           active_response, "selected_clock_response_block", basis),
    ]
end

function _real_column_norm_records(theta, coefficients, term_names; basis)
    return [(
        basis=String(basis),
        term=term_names[index],
        training_column_l2_norm=norm(@view theta[:, index]),
        selected=coefficients[index] != 0.0,
        coefficient=coefficients[index],
        clock_proxy=term_names[index] in _REAL_CLOCK_PROXY_TERMS,
        clock_response=term_names[index] in _REAL_CLOCK_RESPONSE_TERMS,
    ) for index in axes(theta, 2)]
end

function _real_contribution_record(theta, coefficients, term_names, indices;
                                   basis, record_kind, name)
    n = size(theta, 1)
    net = zeros(n)
    gross = zeros(n)
    for column in indices
        coefficient = coefficients[column]
        coefficient == 0.0 && continue
        @inbounds for row in axes(theta, 1)
            contribution = theta[row, column] * coefficient
            net[row] += contribution
            gross[row] += abs(contribution)
        end
    end
    block = _real_normalized_block(theta, collect(indices))
    net_absmax = isempty(net) ? 0.0 : maximum(abs, net)
    gross_absmax = isempty(gross) ? 0.0 : maximum(gross)
    cancellation = gross_absmax == 0.0 ? 0.0 :
        gross_absmax / max(net_absmax, eps(Float64))
    status = if isempty(indices)
        "not_present"
    elseif length(indices) == 1
        coefficients[only(indices)] == 0.0 ? "inactive_basis_term" : "single_basis_term"
    elseif block.rank < block.n_columns
        "rank_deficient_grouped_net_only"
    elseif block.condition_number > _REAL_CONDITION_WARNING
        "ill_conditioned_grouped_net_preferred"
    else
        "full_rank_basis_dependent_group"
    end
    return (
        basis=String(basis),
        record_kind=String(record_kind),
        name=String(name),
        terms=isempty(indices) ? "none" : join(term_names[indices], ";"),
        row_scope="primary_training_regression_rows",
        contribution_units="nT_per_hour",
        n_terms=length(indices),
        n_active_terms=count(index -> coefficients[index] != 0.0, indices),
        normalized_design_rank=block.rank,
        normalized_design_rank_tolerance=block.rank_tolerance,
        normalized_design_condition_number=block.condition_number,
        net_contribution_min=isempty(net) ? 0.0 : minimum(net),
        net_contribution_max=isempty(net) ? 0.0 : maximum(net),
        net_contribution_absmax=net_absmax,
        gross_contribution_absmax=gross_absmax,
        cancellation_ratio=cancellation,
        interpretation_status=status,
    )
end

function _real_contribution_records(theta, coefficients, term_names; basis)
    records = NamedTuple[]
    for index in eachindex(term_names)
        push!(records, _real_contribution_record(
            theta, coefficients, term_names, [index];
            basis, record_kind="term", name=term_names[index],
        ))
    end
    active = findall(!iszero, coefficients)
    proxy = findall(in(_REAL_CLOCK_PROXY_TERMS), term_names)
    response = findall(in(_REAL_CLOCK_RESPONSE_TERMS), term_names)
    active_proxy = intersect(active, proxy)
    active_response = intersect(active, response)
    nonclock = filter(index -> !(term_names[index] in _REAL_CLOCK_RESPONSE_TERMS), active)
    for (name, indices) in (
        "selected_active_net" => active,
        "clock_proxy_net" => proxy,
        "clock_response_net" => response,
        "selected_clock_proxy_net" => active_proxy,
        "selected_clock_response_net" => active_response,
        "selected_nonclock_net" => nonclock,
    )
        push!(records, _real_contribution_record(
            theta, coefficients, term_names, indices;
            basis, record_kind="group", name,
        ))
    end
    return records
end

function _real_empirical_subsample_records(term_names, draws, inclusion_probability;
                                           lambda, seed, subsample_fraction)
    size(draws, 1) == length(term_names) == length(inclusion_probability) ||
        throw(DimensionMismatch("ensemble draws, terms, and inclusion must align"))
    records = NamedTuple[]
    for index in eachindex(term_names)
        values = @view draws[index, :]
        nonzero = values[values .!= 0.0]
        conditional_median = isempty(nonzero) ? NaN : median(nonzero)
        conditional_q025 = isempty(nonzero) ? NaN : quantile(nonzero, 0.025)
        conditional_q975 = isempty(nonzero) ? NaN : quantile(nonzero, 0.975)
        push!(records, (
            term=term_names[index],
            inclusion_probability=inclusion_probability[index],
            nonzero_draws=length(nonzero),
            structural_zero_draws=count(iszero, values),
            conditional_nonzero_median=conditional_median,
            conditional_nonzero_empirical_q025=conditional_q025,
            conditional_nonzero_empirical_q975=conditional_q975,
            interval_kind="conditional_nonzero_empirical_row_subsample_interval",
            confidence_interval=false,
            subsample_without_replacement=true,
            subsample_fraction=Float64(subsample_fraction),
            lambda=Float64(lambda),
            draws=size(draws, 2),
            seed=Int(seed),
        ))
    end
    return records
end

function _real_score_storms(storms, coefficients, library;
                            basis, selected_lambda, experiment=nothing)
    rows = NamedTuple[]
    for storm in storms
        scored = _score_discovery_storm(storm, coefficients, library)
        for row in scored.metrics
            prefix = experiment === nothing ? (;) : (experiment=String(experiment),)
            push!(rows, (
                prefix...,
                basis=String(basis),
                selected_lambda=Float64(selected_lambda),
                row...,
            ))
        end
    end
    return rows
end

function _real_assert_exact_metric_cohorts(rows)
    frame = DataFrame(rows)
    isempty(frame) && throw(ArgumentError("real metric rows must not be empty"))
    required = (:storm_id, :model, :anchor_catalog_index,
                :driver_start_catalog_index, :driver_end_catalog_index,
                :scored_catalog_indices, :cohort_signature_sha256)
    all(in(propertynames(frame)), required) || error(
        "real metric rows are missing exact cohort-identity columns",
    )
    keys = :experiment in propertynames(frame) ? [:experiment, :storm_id] : [:storm_id]
    expected_models = Set(("SINDy", "Burton", "BurtonFull", "OBrienMcP"))
    for group in groupby(frame, keys)
        Set(String.(group.model)) == expected_models && nrow(group) == 4 || error(
            "real outer storm $(first(group.storm_id)) lacks the exact comparator set",
        )
        for field in required[3:end]
            length(unique(group[!, field])) == 1 || error(
                "real outer storm $(first(group.storm_id)) used unequal $field values",
            )
        end
    end
    return true
end

function _real_print_metric_summary(rows)
    frame = DataFrame(rows)
    for model in ("SINDy", "Burton", "BurtonFull", "OBrienMcP")
        subset = frame[frame.model .== model, :]
        isempty(subset) && continue
        println("  $(rpad(model, 12)) RMSE=$(round(mean(subset.rmse_nt), digits=2)), " *
                "PE=$(round(mean(subset.pe), digits=3)), " *
                "r=$(round(mean(subset.correlation), digits=3))")
    end
end

function _real_event_frame(df, storm, score)
    start = storm.observation_record.scoring_start_idx + score.anchor_index - 1
    stop = start + length(score.swd.t) - 1
    rows = start:stop
    length(rows) == length(score.observations) ||
        error("event source-row alignment failed for storm $(storm.storm_id)")
    dst_observed = [Bool(df.Dst_observed[row]) ? Float64(df.Dst[row]) : NaN
                    for row in rows]
    dst_star_observed = Float64.(score.observations)
    dst_sindy = dst_star_to_dst.(score.predictions.SINDy, score.swd.Pdyn)
    dst_burton = dst_star_to_dst.(score.predictions.Burton, score.swd.Pdyn)
    dst_burton_full = dst_star_to_dst.(score.predictions.BurtonFull, score.swd.Pdyn)
    dst_obrien = dst_star_to_dst.(score.predictions.OBrienMcP, score.swd.Pdyn)
    return DataFrame(
        storm_id=fill(storm.storm_id, length(rows)),
        catalog_row=collect(rows),
        datetime=df.datetime[rows],
        time_hr=score.swd.t,
        dst_observed_nt=dst_observed,
        dst_star_observed_nt=dst_star_observed,
        dst_cleaned_nt=score.swd.Dst,
        dst_star_cleaned_nt=score.swd.Dst_star,
        dst_original_flag=Bool.(df.Dst_observed[rows]),
        dst_star_original_target_flag=isfinite.(dst_star_observed),
        dst_star_sindy_nt=score.predictions.SINDy,
        dst_star_burton_simplified_nt=score.predictions.Burton,
        dst_star_burton_published_nt=score.predictions.BurtonFull,
        dst_star_obrien_nt=score.predictions.OBrienMcP,
        dst_sindy_nt=dst_sindy,
        dst_burton_simplified_nt=dst_burton,
        dst_burton_published_nt=dst_burton_full,
        dst_obrien_nt=dst_obrien,
        v_kms=score.swd.V,
        bz_nt=score.swd.Bz,
        pdyn_npa=score.swd.Pdyn,
    )
end

_real_may2024_frame(df, storm, score) = _real_event_frame(df, storm, score)

function _real_event_metric_records(event, storm, frame, score)
    rows = NamedTuple[]
    model_columns = (
        ("SINDy", :dst_star_sindy_nt, :dst_sindy_nt),
        ("Burton", :dst_star_burton_simplified_nt, :dst_burton_simplified_nt),
        ("BurtonFull", :dst_star_burton_published_nt, :dst_burton_published_nt),
        ("OBrienMcP", :dst_star_obrien_nt, :dst_obrien_nt),
    )
    for (model, dst_star_column, dst_column) in model_columns
        for (target_space, observed_column, predicted_column) in (
            ("Dst_star", :dst_star_observed_nt, dst_star_column),
            ("Dst", :dst_observed_nt, dst_column),
        )
            scored = filter(score.scored_indices) do index
                isfinite(frame[index, observed_column])
            end
            isempty(scored) && error(
                "$event $target_space has no original post-anchor targets",
            )
            summary = metrics_summary(
                Float64.(frame[scored, predicted_column]),
                Float64.(frame[scored, observed_column]); name=model,
            )
            push!(rows, (
                event=String(event),
                storm_id=storm.storm_id,
                onset_time=string(storm.onset_time),
                minimum_catalog_dst_star_nt=storm.entry.min_dst_star,
                model,
                target_space,
                rmse_nt=summary.rmse,
                mae_nt=summary.mae,
                correlation=summary.corr,
                pe=summary.pe,
                n_points=length(scored),
                shared_anchor_excluded=true,
                scoring_policy="original_post_anchor_targets_only",
            ))
        end
    end
    return rows
end

function _real_affine_exact_trajectory(a::AbstractVector, q::AbstractVector,
                                       dst0::Real; dt::Real=1.0)
    length(a) == length(q) || throw(DimensionMismatch(
        "affine decay and forcing vectors must have equal length",
    ))
    isfinite(dst0) || throw(ArgumentError("affine initial state must be finite"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("affine dt must be finite and positive"))
    all(isfinite, a) && all(isfinite, q) ||
        throw(ArgumentError("affine coefficients must be finite"))
    trajectory = zeros(length(a) + 1)
    trajectory[1] = Float64(dst0)
    derivative_clamp_count = 0
    state_clamp_count = 0
    dt64 = Float64(dt)
    for k in eachindex(a)
        derivative = a[k] * trajectory[k] + q[k]
        abs(derivative) > 200.0 && (derivative_clamp_count += 1)
        z = a[k] * dt64
        forcing_factor = abs(z) <= sqrt(eps(Float64)) ? dt64 : expm1(z) / a[k]
        next_state = exp(z) * trajectory[k] + forcing_factor * q[k]
        bounded = clamp(next_state, -2000.0, 50.0)
        bounded != next_state && (state_clamp_count += 1)
        trajectory[k + 1] = bounded
    end
    return (; trajectory, derivative_clamp_count, state_clamp_count)
end

function _real_affine_euler_trajectory(a::AbstractVector, q::AbstractVector,
                                       dst0::Real; dt::Real=1.0)
    length(a) == length(q) || throw(DimensionMismatch(
        "affine decay and forcing vectors must have equal length",
    ))
    isfinite(dst0) || throw(ArgumentError("affine initial state must be finite"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("affine dt must be finite and positive"))
    all(isfinite, a) && all(isfinite, q) ||
        throw(ArgumentError("affine coefficients must be finite"))
    trajectory = zeros(length(a) + 1)
    trajectory[1] = Float64(dst0)
    derivative_clamp_count = 0
    state_clamp_count = 0
    dt64 = Float64(dt)
    for k in eachindex(a)
        raw_derivative = a[k] * trajectory[k] + q[k]
        derivative = clamp(raw_derivative, -200.0, 200.0)
        derivative != raw_derivative && (derivative_clamp_count += 1)
        next_state = trajectory[k] + dt64 * derivative
        bounded = clamp(next_state, -2000.0, 50.0)
        bounded != next_state && (state_clamp_count += 1)
        trajectory[k + 1] = bounded
    end
    return (; trajectory, derivative_clamp_count, state_clamp_count)
end

function _real_exact_sindy(coefficients, library, swd; dst0)
    n_steps = length(swd.t) - 1
    n_steps >= 1 || throw(ArgumentError("exact SINDy comparison needs two samples"))
    a = zeros(n_steps)
    q = zeros(n_steps)
    theta0 = Vector{Float64}(undef, length(library))
    theta1 = similar(theta0)
    theta2 = similar(theta0)
    for k in 1:n_steps
        drivers = (Float64(swd.V[k]), Float64(swd.Bz[k]), Float64(swd.By[k]),
                   Float64(swd.n[k]), Float64(swd.Pdyn[k]))
        SolarSINDy._evaluate_point_vector_unchecked!(theta0, library, 0.0, drivers...)
        SolarSINDy._evaluate_point_vector_unchecked!(theta1, library, 1.0, drivers...)
        SolarSINDy._evaluate_point_vector_unchecked!(theta2, library, 2.0, drivers...)
        f0 = dot(theta0, coefficients)
        f1 = dot(theta1, coefficients)
        f2 = dot(theta2, coefficients)
        slope = f1 - f0
        isapprox(f2, f0 + 2slope; rtol=1e-11, atol=1e-11 * max(1.0, abs(f2))) ||
            error("canonical SINDy equation is not affine in Dst_star")
        a[k] = slope
        q[k] = f0
    end
    result = _real_affine_exact_trajectory(a, q, dst0; dt=1.0)
    return (; result..., a, q)
end

function _real_exact_baseline(model::AbstractString, swd; dst0)
    bs = max.(-swd.Bz, 0.0)
    n_steps = length(swd.t) - 1
    a = zeros(n_steps)
    q = zeros(n_steps)
    for k in 1:n_steps
        speed = Float64(swd.V[k])
        southward = Float64(bs[k])
        if model == "Burton"
            a[k] = -1 / 7.7
            q[k] = -5.4e-3 * speed * southward
        elseif model == "BurtonFull"
            a[k] = -1 / 7.7
            vbs = speed * southward
            q[k] = vbs > 500.0 ? -5.4e-3 * (vbs - 500.0) : 0.0
        elseif model == "OBrienMcP"
            ec = speed * southward / 1000.0
            tau = 2.40 * exp(9.74 / (4.69 + ec))
            a[k] = -1 / tau
            q[k] = ec > 0.49 ? -4.4 * (ec - 0.49) : 0.0
        else
            throw(ArgumentError("unknown affine baseline: $model"))
        end
    end
    result = _real_affine_exact_trajectory(a, q, dst0; dt=1.0)
    return (; result..., a, q)
end

function _real_integration_sensitivity_records(event, storm, score,
                                                coefficients, library)
    exact = Dict(
        "SINDy" => _real_exact_sindy(
            coefficients, library, score.swd; dst0=first(score.observations),
        ),
        "Burton" => _real_exact_baseline(
            "Burton", score.swd; dst0=first(score.observations),
        ),
        "BurtonFull" => _real_exact_baseline(
            "BurtonFull", score.swd; dst0=first(score.observations),
        ),
        "OBrienMcP" => _real_exact_baseline(
            "OBrienMcP", score.swd; dst0=first(score.observations),
        ),
    )
    rows = NamedTuple[]
    observed = score.observations[score.scored_indices]
    for model in ("SINDy", "Burton", "BurtonFull", "OBrienMcP")
        euler = getproperty(score.predictions, Symbol(model))
        exact_result = exact[model]
        euler_result = _real_affine_euler_trajectory(
            exact_result.a, exact_result.q, first(score.observations); dt=1.0,
        )
        isapprox(euler_result.trajectory, euler; rtol=1e-12, atol=1e-12) ||
            error("affine Euler diagnostic does not reproduce $model simulation")
        for (integrator, result) in (
            "forward_Euler" => euler_result,
            "exact_piecewise_constant_affine" => exact_result,
        )
            prediction = result.trajectory
            summary = metrics_summary(
                prediction[score.scored_indices], observed; name=model,
            )
            push!(rows, (
                event=String(event),
                storm_id=storm.storm_id,
                onset_time=string(storm.onset_time),
                model,
                integrator,
                driver_interpolation="zero_order_hold_at_hourly_sample",
                rmse_nt=summary.rmse,
                mae_nt=summary.mae,
                correlation=summary.corr,
                pe=summary.pe,
                n_points=length(score.scored_indices),
                shared_anchor_excluded=true,
                max_abs_difference_from_euler_nt=maximum(abs.(prediction .- euler)),
                derivative_clamp_would_activate_count=result.derivative_clamp_count,
                state_clamp_activation_count=result.state_clamp_count,
            ))
        end
    end
    return rows
end

function _real_sindy_contribution_frames(event, storm, frame, score,
                                         coefficients, library,
                                         ensemble_records)
    term_names = get_term_names(library)
    active = findall(!iszero, coefficients)
    n_steps = length(score.swd.t) - 1
    theta = Matrix{Float64}(undef, n_steps, length(library))
    work = Vector{Float64}(undef, length(library))
    for k in 1:n_steps
        SolarSINDy._evaluate_point_vector_unchecked!(
            work, library, score.predictions.SINDy[k], Float64(score.swd.V[k]),
            Float64(score.swd.Bz[k]), Float64(score.swd.By[k]),
            Float64(score.swd.n[k]), Float64(score.swd.Pdyn[k]),
        )
        theta[k, :] = work
    end
    contributions = theta .* reshape(Float64.(coefficients), 1, :)
    total = vec(sum(contributions, dims=2))
    total_gross = vec(sum(abs.(contributions[:, active]), dims=2))
    time_rows = NamedTuple[]
    for k in 1:n_steps, index in active
        push!(time_rows, (
            event=String(event),
            storm_id=storm.storm_id,
            datetime=string(frame.datetime[k]),
            time_hr=Float64(frame.time_hr[k]),
            term=term_names[index],
            coefficient=Float64(coefficients[index]),
            contribution_nt_per_hour=contributions[k, index],
            total_derivative_nt_per_hour=total[k],
            clamped_derivative_nt_per_hour=clamp(total[k], -200.0, 200.0),
        ))
    end

    ensemble = Dict(String(row.term) => row for row in ensemble_records)
    summary_rows = NamedTuple[]
    total_gross_integral = sum(total_gross)
    function add_summary(record_kind, name, indices)
        net = vec(sum(contributions[:, indices], dims=2))
        gross = vec(sum(abs.(contributions[:, indices]), dims=2))
        term_row = length(indices) == 1 ? ensemble[term_names[only(indices)]] : nothing
        push!(summary_rows, (
            event=String(event),
            storm_id=storm.storm_id,
            record_kind=String(record_kind),
            name=String(name),
            terms=join(term_names[indices], ";"),
            n_terms=length(indices),
            coefficient=length(indices) == 1 ? Float64(coefficients[only(indices)]) : NaN,
            inclusion_probability=term_row === nothing ? NaN :
                Float64(term_row.inclusion_probability),
            conditional_nonzero_empirical_q025=term_row === nothing ? NaN :
                Float64(term_row.conditional_nonzero_empirical_q025),
            conditional_nonzero_empirical_q975=term_row === nothing ? NaN :
                Float64(term_row.conditional_nonzero_empirical_q975),
            interval_kind=term_row === nothing ? "not_applicable_to_group" :
                String(term_row.interval_kind),
            confidence_interval=false,
            contribution_min_nt_per_hour=minimum(net),
            contribution_max_nt_per_hour=maximum(net),
            contribution_mean_nt_per_hour=mean(net),
            contribution_mean_abs_nt_per_hour=mean(abs, net),
            contribution_absmax_nt_per_hour=maximum(abs, net),
            signed_integral_nt=sum(net),
            gross_integral_nt=sum(gross),
            gross_fraction_of_active_contributions=
                total_gross_integral == 0.0 ? 0.0 : sum(gross) / total_gross_integral,
            cancellation_ratio=_real_integrated_cancellation_ratio(net, gross),
            state_path="free_running_SINDy_prediction",
            driver_interpolation="zero_order_hold_at_hourly_sample",
        ))
    end
    for index in active
        add_summary("term", term_names[index], [index])
    end
    clock = intersect(active, findall(in(_REAL_CLOCK_RESPONSE_TERMS), term_names))
    nonclock = filter(index -> !(term_names[index] in _REAL_CLOCK_RESPONSE_TERMS), active)
    add_summary("group_net", "selected_active_net", active)
    isempty(clock) || add_summary("group_net", "selected_clock_response_net", clock)
    isempty(nonclock) || add_summary("group_net", "selected_nonclock_net", nonclock)
    return (; timeseries=time_rows, summary=summary_rows)
end

function _real_integrated_cancellation_ratio(net::AbstractVector,
                                              gross::AbstractVector)
    length(net) == length(gross) || throw(DimensionMismatch(
        "net and gross contribution vectors must have equal length",
    ))
    all(>=(0.0), gross) || throw(ArgumentError(
        "gross contributions must be nonnegative",
    ))
    return sum(gross) / max(sum(abs, net), eps(Float64))
end

function _real_intensity_stratified_records(rows;
                                            thresholds=(-50.0, -100.0, -150.0, -200.0))
    frame = DataFrame(rows)
    :experiment in propertynames(frame) ||
        throw(ArgumentError("intensity analysis requires experiment labels"))
    frame = frame[frame.experiment .== "C20-23->C25", :]
    isempty(frame) && error("C20-23->C25 outer metrics are absent")
    records = NamedTuple[]
    for threshold in thresholds
        subset = frame[frame.min_dst_star_observed_nt .<= threshold, :]
        storms = unique(subset.storm_id)
        isempty(storms) && error("no Cycle-25 storms satisfy Dst* <= $threshold nT")
        for model in ("SINDy", "Burton", "BurtonFull", "OBrienMcP")
            model_rows = subset[subset.model .== model, :]
            nrow(model_rows) == length(storms) || error(
                "$model intensity subset does not contain one row per storm",
            )
            push!(records, (
                experiment="C20-23->C25",
                intensity_threshold_dst_star_nt=threshold,
                threshold_rule="selected_segment_minimum_Dst_star_at_or_below_threshold",
                model,
                n_storms=length(storms),
                mean_rmse_nt=mean(model_rows.rmse_nt),
                sd_rmse_nt=length(storms) > 1 ? std(model_rows.rmse_nt) : NaN,
                mean_mae_nt=mean(model_rows.mae_nt),
                mean_correlation=mean(model_rows.correlation),
                mean_pe=mean(model_rows.pe),
            ))
        end
    end
    return records
end

function run_real_data_discovery(context)
    println("=" ^ 68)
    println("Canonical real-data SINDy discovery")
    println("=" ^ 68)
    verify_omni_input(context.omni; mode=context.mode)

    df = parse_omni2(context.omni; year_start=1963, year_end=2025)
    add_original_observation_flags!(df)
    clean_omni_data!(df)
    catalog = load_verified_storm_catalog(context.catalog;
        omni_path=context.omni,
        parameters=storm_catalog_parameters(),
        mode=context.mode,
    )

    policy = DiscoveryObservationPolicy()
    audit = _audit_discovery_observations(df, catalog; policy)
    isempty(audit.eligible_entries) && error("no storms satisfy the observation policy")
    policy_record = _real_policy_record(policy)
    storm_audit_records = _real_explicit_exclusion_records(audit.storm_records)
    cycle_audit_records = _real_explicit_exclusion_records(audit.cycle_records)
    storm_audit_path = _real_manifested_csv(
        context, "real_storm_eligibility.csv", storm_audit_records;
        selection_record=(kind="predeclared_observation_policy", policy=policy_record),
        metadata=(level="storm", policy=policy_record),
    )
    cycle_audit_path = _real_manifested_csv(
        context, "real_cycle_observation_audit.csv", cycle_audit_records;
        selection_record=(kind="predeclared_observation_policy", policy=policy_record),
        metadata=(level="solar_cycle", policy=policy_record),
    )
    audit_inputs = Dict(
        "storm_observation_audit" => storm_audit_path,
        "cycle_observation_audit" => cycle_audit_path,
    )
    println("Eligible storm windows: $(length(audit.eligible_entries)) / $(length(catalog))")

    full_library = build_solar_wind_library(clock_basis=:full)
    collapsed_library = build_solar_wind_library(clock_basis=:collapsed)
    length(full_library) == 20 || error("full canonical library must contain 20 terms")
    length(collapsed_library) == 15 ||
        error("collapsed canonical library must contain 15 terms")
    "n*V^2" in get_term_names(full_library) &&
        error("canonical library contains redundant n*V^2")

    full_storms = _prepare_discovery_storms(
        df, audit.eligible_entries, full_library; policy,
    )
    collapsed_storms = _prepare_discovery_storms(
        df, audit.eligible_entries, collapsed_library; policy,
    )
    _real_assert_matching_cohort(full_storms, collapsed_storms, "all eligible storms")
    full_cache = Dict{Any,Any}()
    collapsed_cache = Dict{Any,Any}()

    full_primary = _real_require_subset(
        full_storms, storm -> storm.entry.split == "train", "primary full training";
        minimum=2,
    )
    collapsed_primary = _real_require_subset(
        collapsed_storms, storm -> storm.entry.split == "train",
        "primary collapsed training"; minimum=2,
    )
    primary = _real_select_basis_pair(
        full_primary, collapsed_primary, full_library, collapsed_library,
        full_cache, collapsed_cache; label="primary",
    )
    full_primary_paths = _real_persist_selection(
        context, primary.full, "primary_lambda";
        basis="full", scope="primary", extra_inputs=audit_inputs,
    )
    collapsed_primary_paths = _real_persist_selection(
        context, primary.collapsed, "primary_collapsed_lambda";
        basis="collapsed", scope="primary", extra_inputs=audit_inputs,
    )
    full_primary_inputs = _real_selection_inputs(
        context, full_primary_paths, "full_primary",
    )
    collapsed_primary_inputs = _real_selection_inputs(
        context, collapsed_primary_paths, "collapsed_primary",
    )
    primary_inputs = merge(full_primary_inputs, collapsed_primary_inputs)
    full_coefficients = primary.full.model
    collapsed_coefficients = primary.collapsed.model
    full_names = get_term_names(full_library)
    collapsed_names = get_term_names(collapsed_library)
    full_design = _cached_subset_design!(full_cache, full_primary)
    collapsed_design = _cached_subset_design!(collapsed_cache, collapsed_primary)
    size(full_design.theta, 1) == size(collapsed_design.theta, 1) ||
        error("full and collapsed primary designs do not use the same regression rows")

    full_record = _real_selection_record(primary.full; basis="full", scope="primary")
    collapsed_record = _real_selection_record(
        primary.collapsed; basis="collapsed", scope="primary",
    )
    sweep_records = [(
        lambda=row.lambda,
        n_terms=row.n_active_terms,
        mean_validation_rmse_nt=row.mean_storm_rmse,
        standard_error_nt=row.standard_error,
        eligible=row.eligible,
        selected=row.selected,
    ) for row in primary.full.candidate_records]
    _real_manifested_csv(context, "real_lambda_sweep.csv", sweep_records;
        selection_record=full_record,
        extra_inputs=full_primary_inputs,
    )

    conditioning = vcat(
        _real_conditioning_records(
            full_design.theta, full_coefficients, full_names; basis="full",
        ),
        _real_conditioning_records(
            collapsed_design.theta, collapsed_coefficients, collapsed_names;
            basis="collapsed",
        ),
    )
    _real_manifested_csv(context, "real_design_conditioning.csv", conditioning;
        selection_record=(kind="actual_normalized_training_design",
                          full=full_record, collapsed=collapsed_record),
        extra_inputs=primary_inputs,
    )
    column_norms = vcat(
        _real_column_norm_records(
            full_design.theta, full_coefficients, full_names; basis="full",
        ),
        _real_column_norm_records(
            collapsed_design.theta, collapsed_coefficients, collapsed_names;
            basis="collapsed",
        ),
    )
    _real_manifested_csv(context, "real_design_column_norms.csv", column_norms;
        selection_record=(kind="actual_training_design_column_norms",
                          full=full_record, collapsed=collapsed_record),
        extra_inputs=primary_inputs,
    )
    contribution_records = vcat(
        _real_contribution_records(
            full_design.theta, full_coefficients, full_names; basis="full",
        ),
        _real_contribution_records(
            collapsed_design.theta, collapsed_coefficients, collapsed_names;
            basis="collapsed",
        ),
    )
    _real_manifested_csv(
        context, "real_contribution_diagnostics.csv", contribution_records;
        selection_record=(kind="physical_unit_term_and_group_contributions",
                          full=full_record, collapsed=collapsed_record),
        extra_inputs=primary_inputs,
        metadata=(condition_warning_threshold=_REAL_CONDITION_WARNING,
                  individual_terms_are_basis_dependent=true),
    )

    full_coefficient_path = _real_manifested_csv(
        context, "real_sindy_discovery_coefficients.csv",
        [(term=full_names[index], coefficient=full_coefficients[index])
         for index in eachindex(full_names)];
        selection_record=full_record,
        extra_inputs=full_primary_inputs,
        metadata=(coefficient_kind="selected_full_refit_point_coefficient",),
    )
    collapsed_coefficient_path = _real_manifested_csv(
        context, "real_sindy_collapsed_coefficients.csv",
        [(term=collapsed_names[index], coefficient=collapsed_coefficients[index])
         for index in eachindex(collapsed_names)];
        selection_record=collapsed_record,
        extra_inputs=collapsed_primary_inputs,
        metadata=(coefficient_kind="selected_full_refit_point_coefficient",),
    )

    println("Primary full lambda: $(primary.full.selected_lambda); " *
            "active terms: $(count(!iszero, full_coefficients))")
    println("Primary collapsed lambda: $(primary.collapsed.selected_lambda); " *
            "active terms: $(count(!iszero, collapsed_coefficients))")

    primary_data = _concat_discovery_data(full_primary)
    _, inclusion_probability, joint_draws = ensemble_sindy(
        primary_data, full_library, full_design.target;
        λ=primary.full.selected_lambda,
        n_models=_REAL_ENSEMBLE_DRAWS,
        subsample_frac=0.8,
        seed=_REAL_ENSEMBLE_SEED,
        bootstrap=false,
    )
    size(joint_draws) == (length(full_names), _REAL_ENSEMBLE_DRAWS) ||
        error("ensemble draw matrix has an unexpected shape")
    draw_inputs = merge(full_primary_inputs, Dict(
        "point_coefficients" => full_coefficient_path,
        "storm_observation_audit" => storm_audit_path,
        "cycle_observation_audit" => cycle_audit_path,
    ))
    draw_path = _real_manifested_csv(
        context, "real_sindy_ensemble_draws.csv",
        DataFrame(permutedims(joint_draws), Symbol.(full_names));
        selection_record=merge(full_record, (
            ensemble="500_raw_complete_80pct_row_subsamples_without_replacement",
            structural_zeros="retained",
        )),
        seed=_REAL_ENSEMBLE_SEED,
        extra_inputs=draw_inputs,
    )
    ensemble_records = _real_empirical_subsample_records(
        full_names, joint_draws, inclusion_probability;
        lambda=primary.full.selected_lambda,
        seed=_REAL_ENSEMBLE_SEED,
        subsample_fraction=0.8,
    )
    ensemble_inputs = merge(draw_inputs, Dict("raw_joint_draws" => draw_path))
    ensemble_summary_path = _real_manifested_csv(
        context, "real_ensemble_inclusion.csv", ensemble_records;
        selection_record=merge(full_record, (
            interval_kind="conditional_nonzero_empirical_row_subsample_interval",
            confidence_interval=false,
        )),
        seed=_REAL_ENSEMBLE_SEED,
        extra_inputs=ensemble_inputs,
    )
    _real_manifested_csv(
        context, "real_sindy_coefficients.csv",
        [(
            term=full_names[index],
            coefficient=full_coefficients[index],
            coefficient_kind="selected_full_refit_point_coefficient",
            inclusion=inclusion_probability[index],
        ) for index in eachindex(full_names)];
        selection_record=full_record,
        seed=_REAL_ENSEMBLE_SEED,
        extra_inputs=ensemble_inputs,
    )
    _real_manifested_csv(
        context, "real_sindy_discovery_provenance.csv", [
            (field="selected_lambda", value=string(primary.full.selected_lambda)),
            (field="selection_rule", value=primary.full.decision_record.selection_rule),
            (field="n_active_terms", value=string(count(!iszero, full_coefficients))),
            (field="n_training_storms", value=string(length(full_primary))),
            (field="n_training_points", value=string(size(full_design.theta, 1))),
            (field="library_terms", value=string(length(full_library))),
            (field="clock_basis", value="full"),
            (field="ensemble_draws", value=string(_REAL_ENSEMBLE_DRAWS)),
            (field="ensemble_seed", value=string(_REAL_ENSEMBLE_SEED)),
            (field="ensemble_convention", value="raw_complete_rows_structural_zeros_retained"),
        ];
        selection_record=full_record,
        seed=_REAL_ENSEMBLE_SEED,
        extra_inputs=merge(full_primary_inputs, Dict(
            "raw_joint_draws" => draw_path,
        )),
    )

    full_holdout = _real_require_subset(
        full_storms, storm -> storm.entry.split == "val", "cycle-24 full holdout",
    )
    collapsed_holdout = _real_require_subset(
        collapsed_storms, storm -> storm.entry.split == "val",
        "cycle-24 collapsed holdout",
    )
    _real_assert_matching_cohort(full_holdout, collapsed_holdout, "cycle-24 holdout")
    full_holdout_rows = _real_score_storms(
        full_holdout, full_coefficients, full_library;
        basis="full", selected_lambda=primary.full.selected_lambda,
    )
    collapsed_holdout_rows = _real_score_storms(
        collapsed_holdout, collapsed_coefficients, collapsed_library;
        basis="collapsed", selected_lambda=primary.collapsed.selected_lambda,
    )
    _real_assert_exact_metric_cohorts(full_holdout_rows)
    _real_assert_exact_metric_cohorts(collapsed_holdout_rows)
    _real_print_metric_summary(full_holdout_rows)
    _real_manifested_csv(context, "real_holdout_metrics.csv", full_holdout_rows;
        selection_record=merge(full_record, (outer_split="cycle_24",)),
        extra_inputs=merge(full_primary_inputs, Dict(
            "point_coefficients" => full_coefficient_path,
            "storm_observation_audit" => storm_audit_path,
        )),
    )
    _real_manifested_csv(
        context, "real_holdout_collapsed_metrics.csv", collapsed_holdout_rows;
        selection_record=merge(collapsed_record, (outer_split="cycle_24",)),
        extra_inputs=merge(collapsed_primary_inputs, Dict(
            "point_coefficients" => collapsed_coefficient_path,
            "storm_observation_audit" => storm_audit_path,
        )),
    )

    may_storms = filter(
        storm -> year(storm.onset_time) == 2024 && month(storm.onset_time) == 5,
        full_storms,
    )
    isempty(may_storms) && error(
        "May 2024 has no observation-policy-eligible catalog window",
    )
    may_storm = may_storms[argmin(
        [minimum(storm.scoring_observations[isfinite.(storm.scoring_observations)])
         for storm in may_storms],
    )]
    may_score = _score_discovery_storm(may_storm, full_coefficients, full_library)
    may_frame = _real_may2024_frame(df, may_storm, may_score)
    println("May 2024 original Dst minimum: $(minimum(may_frame.dst_observed_nt[isfinite.(may_frame.dst_observed_nt)])) nT")
    println("May 2024 original-target Dst* minimum: $(minimum(may_frame.dst_star_observed_nt[isfinite.(may_frame.dst_star_observed_nt)])) nT")
    may_path = _real_manifested_csv(context, "may2024_reconstruction.csv", may_frame;
        selection_record=merge(full_record, (
            event="May_2024",
            selected_storm_id=may_storm.storm_id,
            selected_storm_onset=string(may_storm.onset_time),
            catalog_window_start=may_storm.entry.onset_idx,
            catalog_window_end=may_storm.entry.end_idx,
            scoring_start_catalog_row=first(may_frame.catalog_row),
            scoring_end_catalog_row=last(may_frame.catalog_row),
            event_selection="eligible_May_2024_window_with_lowest_original_target_Dst_star_minimum",
        )),
        extra_inputs=merge(full_primary_inputs, Dict(
            "point_coefficients" => full_coefficient_path,
            "storm_observation_audit" => storm_audit_path,
        )),
    )

    moderate_storms = filter(full_storms) do storm
        Date(storm.onset_time) == Date(2022, 2, 3)
    end
    length(moderate_storms) == 1 || error(
        "expected exactly one eligible storm beginning on 2022-02-03, found " *
        string(length(moderate_storms)),
    )
    moderate_storm = only(moderate_storms)
    moderate_score = _score_discovery_storm(
        moderate_storm, full_coefficients, full_library,
    )
    moderate_frame = _real_event_frame(df, moderate_storm, moderate_score)
    moderate_path = _real_manifested_csv(
        context, "moderate2022_reconstruction.csv", moderate_frame;
        selection_record=merge(full_record, (
            event="February_2022_moderate",
            selected_storm_id=moderate_storm.storm_id,
            selected_storm_onset=string(moderate_storm.onset_time),
            catalog_window_start=moderate_storm.entry.onset_idx,
            catalog_window_end=moderate_storm.entry.end_idx,
            scoring_start_catalog_row=first(moderate_frame.catalog_row),
            scoring_end_catalog_row=last(moderate_frame.catalog_row),
            event_selection="reviewer_named_2022_02_03_moderate_event_selected_by_date_not_performance",
        )),
        extra_inputs=merge(full_primary_inputs, Dict(
            "point_coefficients" => full_coefficient_path,
            "storm_observation_audit" => storm_audit_path,
        )),
    )

    event_metric_rows = vcat(
        _real_event_metric_records("May_2024", may_storm, may_frame, may_score),
        _real_event_metric_records(
            "February_2022_moderate", moderate_storm, moderate_frame, moderate_score,
        ),
    )
    _real_manifested_csv(
        context, "event_reconstruction_metrics.csv", event_metric_rows;
        selection_record=merge(full_record, (
            kind="predeclared_event_reconstruction_metrics",
            targets=("Dst_star", "Dst"),
            shared_anchor_excluded=true,
        )),
        extra_inputs=merge(full_primary_inputs, Dict(
            "point_coefficients" => full_coefficient_path,
            "may2024_trajectory" => may_path,
            "moderate2022_trajectory" => moderate_path,
        )),
    )

    may_contributions = _real_sindy_contribution_frames(
        "May_2024", may_storm, may_frame, may_score, full_coefficients,
        full_library, ensemble_records,
    )
    _real_manifested_csv(
        context, "may2024_term_contribution_timeseries.csv",
        may_contributions.timeseries;
        selection_record=merge(full_record, (
            kind="free_running_event_term_contribution_timeseries",
            event="May_2024",
        )),
        extra_inputs=merge(full_primary_inputs, Dict(
            "point_coefficients" => full_coefficient_path,
            "ensemble_stability" => ensemble_summary_path,
            "may2024_trajectory" => may_path,
        )),
    )
    _real_manifested_csv(
        context, "may2024_term_contribution_summary.csv",
        may_contributions.summary;
        selection_record=merge(full_record, (
            kind="physical_unit_event_term_and_group_contributions",
            event="May_2024",
            individual_clock_terms_basis_dependent=true,
        )),
        extra_inputs=merge(full_primary_inputs, Dict(
            "point_coefficients" => full_coefficient_path,
            "ensemble_stability" => ensemble_summary_path,
            "may2024_trajectory" => may_path,
        )),
        metadata=(
            interval_interpretation="empirical_subsampling_stability_not_confidence_intervals",
            clock_block_interpretation="grouped_net_preferred_under_collinearity",
        ),
    )

    integration_rows = vcat(
        _real_integration_sensitivity_records(
            "May_2024", may_storm, may_score, full_coefficients, full_library,
        ),
        _real_integration_sensitivity_records(
            "February_2022_moderate", moderate_storm, moderate_score,
            full_coefficients, full_library,
        ),
    )
    _real_manifested_csv(
        context, "integration_sensitivity.csv", integration_rows;
        selection_record=merge(full_record, (
            kind="forward_Euler_vs_exact_piecewise_constant_affine_integration",
            events=("May_2024", "February_2022_moderate"),
            shared_anchor_excluded=true,
        )),
        extra_inputs=merge(full_primary_inputs, Dict(
            "point_coefficients" => full_coefficient_path,
            "may2024_trajectory" => may_path,
            "moderate2022_trajectory" => moderate_path,
        )),
    )

    experiments = [
        (
            label="C20-22->C23",
            prefix="cross_c20_22_to_c23_lambda",
            train=storm -> storm.entry.solar_cycle in 20:22,
            test=storm -> storm.entry.solar_cycle == 23,
        ),
        (
            label="even->odd",
            prefix="cross_even_to_odd_lambda",
            train=storm -> storm.entry.solar_cycle in (20, 22, 24),
            test=storm -> storm.entry.solar_cycle in (21, 23),
        ),
        (
            label="C20-23->C25",
            prefix="cross_c20_23_to_c25_lambda",
            train=storm -> storm.entry.solar_cycle in 20:23,
            test=storm -> storm.entry.solar_cycle == 25,
        ),
    ]
    full_cross_rows = NamedTuple[]
    collapsed_cross_rows = NamedTuple[]
    full_cross_inputs = Dict{String,String}()
    collapsed_cross_inputs = Dict{String,String}()
    full_cross_records = NamedTuple[]
    collapsed_cross_records = NamedTuple[]
    for (experiment_index, experiment) in enumerate(experiments)
        full_training = _real_require_subset(
            full_storms, experiment.train, "$(experiment.label) full training";
            minimum=2,
        )
        collapsed_training = _real_require_subset(
            collapsed_storms, experiment.train,
            "$(experiment.label) collapsed training"; minimum=2,
        )
        full_outer = _real_require_subset(
            full_storms, experiment.test, "$(experiment.label) full outer",
        )
        collapsed_outer = _real_require_subset(
            collapsed_storms, experiment.test, "$(experiment.label) collapsed outer",
        )
        isempty(intersect(
            Set(getproperty.(full_training, :storm_id)),
            Set(getproperty.(full_outer, :storm_id)),
        )) || error("$(experiment.label) training and outer storms overlap")
        _real_assert_matching_cohort(
            full_training, collapsed_training, "$(experiment.label) training",
        )
        _real_assert_matching_cohort(
            full_outer, collapsed_outer, "$(experiment.label) outer",
        )
        selections = _real_select_basis_pair(
            full_training, collapsed_training, full_library, collapsed_library,
            full_cache, collapsed_cache; label=experiment.label,
        )
        full_paths = _real_persist_selection(
            context, selections.full, experiment.prefix;
            basis="full", scope=experiment.label, extra_inputs=audit_inputs,
        )
        collapsed_paths = _real_persist_selection(
            context, selections.collapsed, "$(experiment.prefix)_collapsed";
            basis="collapsed", scope=experiment.label, extra_inputs=audit_inputs,
        )
        merge!(full_cross_inputs, _real_selection_inputs(
            context, full_paths, "cross_$(experiment_index)_full",
        ))
        merge!(collapsed_cross_inputs, _real_selection_inputs(
            context, collapsed_paths, "cross_$(experiment_index)_collapsed",
        ))
        push!(full_cross_records, _real_selection_record(
            selections.full; basis="full", scope=experiment.label,
        ))
        push!(collapsed_cross_records, _real_selection_record(
            selections.collapsed; basis="collapsed", scope=experiment.label,
        ))
        append!(full_cross_rows, _real_score_storms(
            full_outer, selections.full.model, full_library;
            basis="full", selected_lambda=selections.full.selected_lambda,
            experiment=experiment.label,
        ))
        append!(collapsed_cross_rows, _real_score_storms(
            collapsed_outer, selections.collapsed.model, collapsed_library;
            basis="collapsed", selected_lambda=selections.collapsed.selected_lambda,
            experiment=experiment.label,
        ))
    end
    _real_assert_exact_metric_cohorts(full_cross_rows)
    _real_assert_exact_metric_cohorts(collapsed_cross_rows)
    cross_metrics_path = _real_manifested_csv(
        context, "cross_cycle_metrics.csv", full_cross_rows;
        selection_record=(kind="independently_selected_cross_cycle_outer_metrics",
                          basis="full", experiments=full_cross_records),
        extra_inputs=merge(full_cross_inputs, Dict(
            "storm_observation_audit" => storm_audit_path,
        )),
    )
    _real_manifested_csv(
        context, "cross_cycle_collapsed_metrics.csv", collapsed_cross_rows;
        selection_record=(kind="independently_selected_cross_cycle_outer_metrics",
                          basis="collapsed", experiments=collapsed_cross_records),
        extra_inputs=merge(collapsed_cross_inputs, Dict(
            "storm_observation_audit" => storm_audit_path,
        )),
    )
    intensity_rows = _real_intensity_stratified_records(full_cross_rows)
    _real_manifested_csv(
        context, "intensity_stratified_metrics.csv", intensity_rows;
        selection_record=(
            kind="heldout_cycle25_event_minimum_intensity_stratification",
            source_experiment="C20-23->C25",
            thresholds_dst_star_nt=(-50.0, -100.0, -150.0, -200.0),
            refit_scope="C20_to_C23_training_only",
        ),
        extra_inputs=merge(full_cross_inputs, Dict(
            "cross_cycle_metrics" => cross_metrics_path,
            "storm_observation_audit" => storm_audit_path,
        )),
    )

    println("Canonical discovery outputs written under: $(context.data)")
    return nothing
end

run_real_data_discovery() = run_real_data_discovery(_real_output_context())

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    context = _real_output_context()
    run_real_data_discovery(context)
end
