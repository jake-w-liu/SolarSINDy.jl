#!/usr/bin/env julia
# Strict deterministic regeneration of the canonical raw coefficient draws.

isdefined(@__MODULE__, :run_real_data_discovery) ||
    include(joinpath(@__DIR__, "real_data_discovery.jl"))

function _ensemble_primary_selection_paths(data_dir)
    prefix = joinpath(data_dir, "primary_lambda")
    return (
        split="$(prefix)_inner_split.csv",
        candidates="$(prefix)_candidates.csv",
        errors="$(prefix)_validation_errors.csv",
        support="$(prefix)_support.csv",
        decision="$(prefix)_decision.csv",
    )
end

function _ensemble_selected_lambda(paths)
    selection = SolarSINDy._read_selection_csv_set(paths)
    candidates = selection.candidates
    decision = selection.decision
    nrow(candidates) == 60 || error("primary selection must contain 60 candidates")
    nrow(decision) == 1 || error("primary selection must contain one decision row")
    Float64.(candidates.lambda) == storm_lambda_grid() ||
        error("primary selection does not use the fixed 60-point lambda grid")
    selected = candidates[Bool.(candidates.selected), :]
    nrow(selected) == 1 || error("primary selection must mark exactly one candidate")
    lambda = Float64(only(decision.selected_lambda))
    Float64(only(selected.lambda)) == lambda ||
        error("candidate and decision selected lambdas disagree")
    return lambda, NamedTuple(decision[1, :])
end

function _ensemble_verify_observation_audit(context, audit)
    path = joinpath(context.data, "real_storm_eligibility.csv")
    verify_output_manifest(path;
        package_root=_REAL_PACKAGE_ROOT,
        require_canonical=context.mode == :canonical,
    )
    persisted = CSV.read(path, DataFrame)
    expected = DataFrame(audit.storm_records)
    names(persisted) == names(expected) ||
        error("persisted and regenerated storm-audit schemas differ")
    nrow(persisted) == nrow(expected) ||
        error("persisted and regenerated storm-audit row counts differ")
    for name in names(expected)
        persisted_values = persisted[!, name]
        expected_values = expected[!, name]
        equal = if name == "onset_time"
            string.(persisted_values) == string.(expected_values)
        elseif name == "exclusion_reason"
            normalize = values -> [
                ismissing(value) || isempty(strip(String(value))) ? "none" : String(value)
                for value in values
            ]
            normalize(persisted_values) == normalize(expected_values)
        else
            isequal(persisted_values, expected_values)
        end
        equal ||
            error("persisted storm audit differs in column $name")
    end
    return path
end

function _ensemble_read_point_coefficients(context, term_names, design, lambda)
    path = joinpath(context.data, "real_sindy_discovery_coefficients.csv")
    verify_output_manifest(path;
        package_root=_REAL_PACKAGE_ROOT,
        require_canonical=context.mode == :canonical,
    )
    frame = CSV.read(path, DataFrame)
    names(frame) == ["term", "coefficient"] ||
        error("point-coefficient schema is not canonical")
    string.(frame.term) == term_names ||
        error("point-coefficient term order differs from the full library")
    persisted = Float64.(frame.coefficient)
    refit = stlsq(design.theta, design.target; λ=lambda, normalize=true)
    persisted == refit || error(
        "point coefficients are not the selected-lambda full refit on the masked design",
    )
    return path, persisted, refit
end

function _persist_regenerated_ensemble_outputs!(
        context, refit_audit_rows, term_names, draws, inclusion_rows,
        coefficient_rows; record, refit_record, refit_inputs,
        seed::Int, _after_artifact_hook::Function=(name, path) -> nothing)
    names = (
        refit_audit="real_primary_refit_audit.csv",
        draws="real_sindy_ensemble_draws.csv",
        inclusion="real_ensemble_inclusion.csv",
        coefficients="real_sindy_coefficients.csv",
    )
    outputs = [joinpath(context.data, name) for name in values(names)]
    snapshot = SolarSINDy._snapshot_regular_file_set(vcat(
        outputs, [path * ".manifest.json" for path in outputs],
    ))
    local refit_audit_path, draw_path, inclusion_path, coefficient_path
    try
        refit_audit_path = _real_manifested_csv(
            context, names.refit_audit, refit_audit_rows;
            selection_record=refit_record, extra_inputs=refit_inputs,
            producer_script=@__FILE__,
        )
        _after_artifact_hook(:refit_audit, refit_audit_path)
        base_inputs = merge(refit_inputs, Dict(
            "primary_refit_audit" => refit_audit_path,
        ))
        draw_path = _real_manifested_csv(
            context, names.draws,
            DataFrame(permutedims(draws), Symbol.(term_names));
            selection_record=record, seed,
            extra_inputs=base_inputs, producer_script=@__FILE__,
        )
        Matrix{Float64}(CSV.read(draw_path, DataFrame)) == permutedims(draws) ||
            error("raw coefficient draws changed during persistence")
        _after_artifact_hook(:draws, draw_path)
        draw_inputs = merge(base_inputs, Dict("raw_joint_draws" => draw_path))
        inclusion_path = _real_manifested_csv(
            context, names.inclusion, inclusion_rows;
            selection_record=merge(record, (
                interval_kind="conditional_nonzero_empirical_row_subsample_interval",
                confidence_interval=false,
            )),
            seed, extra_inputs=draw_inputs, producer_script=@__FILE__,
        )
        _after_artifact_hook(:inclusion, inclusion_path)
        coefficient_path = _real_manifested_csv(
            context, names.coefficients, coefficient_rows;
            selection_record=record, seed,
            extra_inputs=draw_inputs, producer_script=@__FILE__,
        )
        _after_artifact_hook(:coefficients, coefficient_path)
    catch
        SolarSINDy._restore_regular_file_set!(snapshot)
        rethrow()
    end
    SolarSINDy._discard_regular_file_snapshot!(snapshot)
    return (
        refit_audit=refit_audit_path, draws=draw_path,
        inclusion=inclusion_path, coefficients=coefficient_path,
    )
end

function regenerate_ensemble_draws(
        context; _after_artifact_hook::Function=(name, path) -> nothing)
    verify_omni_input(context.omni; mode=context.mode)
    catalog = load_verified_storm_catalog(context.catalog;
        omni_path=context.omni,
        parameters=storm_catalog_parameters(),
        mode=context.mode,
    )
    df = parse_omni2(context.omni; year_start=1963, year_end=2025)
    add_original_observation_flags!(df)
    clean_omni_data!(df)
    policy = DiscoveryObservationPolicy()
    audit = _audit_discovery_observations(df, catalog; policy)
    audit_path = _ensemble_verify_observation_audit(context, audit)

    library = build_solar_wind_library(clock_basis=:full)
    length(library) == 20 || error("canonical full library must contain 20 terms")
    term_names = get_term_names(library)
    storms = _prepare_discovery_storms(df, audit.eligible_entries, library; policy)
    training = _real_require_subset(
        storms, storm -> storm.entry.split == "train", "primary ensemble training";
        minimum=2,
    )
    design = _cached_subset_design!(Dict{Any,Any}(), training)
    data = _concat_discovery_data(training)

    selection_paths = _ensemble_primary_selection_paths(context.data)
    selection_inputs = _real_selection_inputs(
        context, selection_paths, "full_primary",
    )
    lambda, decision_record = _ensemble_selected_lambda(selection_paths)
    coefficient_path, point_coefficients, fresh_refit = _ensemble_read_point_coefficients(
        context, term_names, design, lambda,
    )

    _, inclusion, draws = ensemble_sindy(
        data, library, design.target;
        λ=lambda,
        n_models=_REAL_ENSEMBLE_DRAWS,
        subsample_frac=0.8,
        seed=_REAL_ENSEMBLE_SEED,
        bootstrap=false,
    )
    size(draws) == (20, _REAL_ENSEMBLE_DRAWS) ||
        error("regenerated ensemble draw matrix has an unexpected shape")
    record = (
        kind="fixed_grid_whole_storm_selected_full_refit_ensemble",
        basis="full",
        selected_lambda=lambda,
        decision=decision_record,
        ensemble="500_raw_complete_80pct_row_subsamples_without_replacement",
        structural_zeros="retained_without_recentering_or_imputation",
    )
    refit_inputs = merge(selection_inputs, Dict(
        "point_coefficients" => coefficient_path,
        "storm_observation_audit" => audit_path,
    ))
    refit_audit_rows = [(
            term=term_names[index],
            persisted_coefficient=point_coefficients[index],
            fresh_selected_lambda_refit_coefficient=fresh_refit[index],
            exact_match=point_coefficients[index] == fresh_refit[index],
            absolute_difference=abs(point_coefficients[index] - fresh_refit[index]),
            selected_lambda=lambda,
            training_storms=length(training),
            training_rows=size(design.theta, 1),
        ) for index in eachindex(term_names)]
    point_coefficients == fresh_refit || error(
        "persisted primary coefficients differ from the fresh selected-lambda refit",
    )
    summary = _real_empirical_subsample_records(
        term_names, draws, inclusion;
        lambda,
        seed=_REAL_ENSEMBLE_SEED,
        subsample_fraction=0.8,
    )
    coefficient_rows = [(
            term=term_names[index],
            coefficient=point_coefficients[index],
            coefficient_kind="selected_full_refit_point_coefficient",
            inclusion=inclusion[index],
        ) for index in eachindex(term_names)]
    _persist_regenerated_ensemble_outputs!(
        context, refit_audit_rows, term_names, draws, summary, coefficient_rows;
        record,
        refit_record=merge(record, (
            audit="persisted_point_coefficients_equal_fresh_selected_lambda_full_refit",
            exact_match=true,
        )),
        refit_inputs, seed=_REAL_ENSEMBLE_SEED, _after_artifact_hook,
    )
    println("Regenerated and verified $(_REAL_ENSEMBLE_DRAWS) raw ensemble draws")
    return nothing
end

regenerate_ensemble_draws(; kwargs...) =
    regenerate_ensemble_draws(_real_output_context(); kwargs...)

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    context = _real_output_context()
    regenerate_ensemble_draws(context)
end
