#!/usr/bin/env julia
# Canonical phase-threshold sensitivity with independent storm-level selection.

using SolarSINDy
using CSV, DataFrames, Dates, Statistics, Random

isdefined(@__MODULE__, :_select_phase_lambda) ||
    include(joinpath(@__DIR__, "phase_dependent_discovery.jl"))

const PHASE_QUIET_THRESHOLDS = (-30.0, -25.0, -20.0, -15.0, -10.0)
const PHASE_DERIV_THRESHOLDS = (-4.0, -3.0, -2.0, -1.0, 0.0)

function _phase_threshold_selections(storms::AbstractVector,
                                     lib::CandidateLibrary;
                                     quiet_thresholds=PHASE_QUIET_THRESHOLDS,
                                     deriv_thresholds=PHASE_DERIV_THRESHOLDS,
                                     min_phase_rows::Int=max(length(lib), PHASE_MIN_REGRESSION_ROWS),
                                     min_phase_storms::Int=2)
    isempty(quiet_thresholds) &&
        throw(ArgumentError("quiet_thresholds must not be empty"))
    isempty(deriv_thresholds) &&
        throw(ArgumentError("deriv_thresholds must not be empty"))
    results = NamedTuple[]
    for quiet_thresh in quiet_thresholds, deriv_thresh in deriv_thresholds
        isfinite(quiet_thresh) && isfinite(deriv_thresh) ||
            throw(ArgumentError("phase thresholds must be finite"))
        labeled = _with_phase_labels(
            storms; quiet_thresh, deriv_thresh,
        )
        # A fresh selector call is mandatory for every threshold pair. No
        # coefficient vector or lambda is carried between configurations.
        selection = _select_phase_lambda(
            labeled, lib; quiet_thresh, deriv_thresh,
            min_phase_rows, min_phase_storms,
        )
        push!(results, (
            quiet_thresh=Float64(quiet_thresh),
            deriv_thresh=Float64(deriv_thresh),
            storms=labeled,
            selection,
        ))
    end
    return results
end

function _threshold_result_rows(results, lib::CandidateLibrary)
    terms = get_term_names(lib)
    coefficients = NamedTuple[]
    decisions = NamedTuple[]
    candidates = NamedTuple[]
    errors = NamedTuple[]
    support = NamedTuple[]
    inner_split = NamedTuple[]
    counts = NamedTuple[]
    for result in results
        quiet_thresh = result.quiet_thresh
        deriv_thresh = result.deriv_thresh
        selection = result.selection
        model = selection.model
        thresholds = (; quiet_thresh, deriv_thresh)
        push!(decisions, merge(thresholds, selection.decision_record))
        append!(candidates, [merge(thresholds, row)
                             for row in selection.candidate_records])
        append!(errors, [merge(thresholds, row)
                         for row in selection.error_records])
        append!(support, [merge(thresholds, row)
                          for row in selection.support_records])
        append!(inner_split, [merge(thresholds, row)
                              for row in selection.split_records])
        for phase in 1:3
            push!(counts, (
                quiet_thresh, deriv_thresh,
                phase=PHASE_NAMES[phase],
                selected_lambda=selection.selected_lambda,
                n_regression_rows=model.row_counts[phase],
                n_storms_with_phase=model.storm_counts[phase],
                n_training_storms=length(result.storms),
            ))
            for index in eachindex(terms)
                push!(coefficients, (
                    quiet_thresh, deriv_thresh,
                    phase=PHASE_NAMES[phase],
                    term=terms[index],
                    coefficient=model.coefficients[phase][index],
                    selected_lambda=selection.selected_lambda,
                    n_regression_rows=model.row_counts[phase],
                    n_storms_with_phase=model.storm_counts[phase],
                    n_active_terms=count(!=(0.0), model.coefficients[phase]),
                    selection_rule=selection.decision_record.selection_rule,
                ))
            end
        end
    end
    return (; coefficients, decisions, candidates, errors, support,
            inner_split, counts)
end

function _phase_threshold_outer_rows(results, outer_storms,
                                     single_coefficients,
                                     lib::CandidateLibrary)
    isempty(results) && throw(ArgumentError("threshold results must not be empty"))
    isempty(outer_storms) && throw(ArgumentError("outer storms must not be empty"))
    metrics = NamedTuple[]
    trajectories = NamedTuple[]
    for result in results
        quiet_thresh = result.quiet_thresh
        deriv_thresh = result.deriv_thresh
        selected_lambda = result.selection.selected_lambda
        for storm in sort(collect(outer_storms); by=s -> (s.onset_time, s.storm_id))
            score = _score_phase_storm(
                storm, result.selection.model, single_coefficients, lib;
                quiet_thresh, deriv_thresh,
            )
            _paired_phase_metrics(score.rows)
            prediction = score.predictions.SwitchingSINDy
            all(isfinite, prediction) || error(
                "threshold trajectory is non-finite for storm $(storm.storm_id)",
            )
            append!(metrics, [merge((selected_lambda=selected_lambda,), row)
                              for row in score.rows])
            score_mask = falses(length(score.observations))
            score_mask[score.scored_indices] .= true
            for index in eachindex(prediction)
                push!(trajectories, (
                    quiet_thresh,
                    deriv_thresh,
                    selected_lambda,
                    split=String(storm.entry.split),
                    storm_id=storm.storm_id,
                    onset_time=string(storm.onset_time),
                    catalog_index=score.anchor_index + index - 1,
                    time_hr=score.swd.t[index],
                    dst_star_observed_nt=score.observations[index],
                    dst_star_switching_nt=prediction[index],
                    original_target=isfinite(score.observations[index]),
                    scored_target=score_mask[index],
                    shared_observed_anchor=index == 1,
                ))
            end
        end
    end
    return (; metrics, trajectories)
end


function _phase_threshold_design_rows(results, lib::CandidateLibrary)
    rows = NamedTuple[]
    for result in results
        thresholds = (
            quiet_thresh=result.quiet_thresh,
            deriv_thresh=result.deriv_thresh,
        )
        append!(rows, [merge(thresholds, row) for row in
                       _phase_design_diagnostic_records(
                           result.storms, result.selection, lib,
                       )])
    end
    return rows
end

function run_phase_sensitivity()
    isempty(strip(get(ENV, "SOLARSINDY_OUTPUT_ROOT", ""))) && error(
        "SOLARSINDY_OUTPUT_ROOT must be set for canonical phase sensitivity",
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
    base_inputs = Dict(
        "omni_extracted" => omni,
        "storm_catalog" => catalog_path,
    )
    observation_path = _phase_write(
        joinpath(data_dir, "phase_threshold_observation_audit.csv"),
        audit.storm_records;
        output_paths, producer_script=producer, inputs=base_inputs,
        selection_record=(
            kind="phase_threshold_observation_policy",
            smooth_window=policy.smooth_window,
            min_regression_rows=policy.min_regression_rows,
            min_scoring_rows=policy.min_scoring_rows,
            min_scoring_fraction=policy.min_scoring_fraction,
        ), deterministic=true,
    )
    selection_inputs = merge(base_inputs, Dict(
        "phase_threshold_observation_audit" => observation_path,
    ))
    lib = build_solar_wind_library()
    length(lib) == 20 || error("phase sensitivity requires the 20-term library")
    "n*V^2" in get_term_names(lib) &&
        error("phase sensitivity library contains redundant n*V^2")
    eligible_training = filter(
        entry -> entry.split == "train", audit.eligible_entries,
    )
    length(eligible_training) >= 2 ||
        error("phase sensitivity needs at least two eligible training storms")
    storms = _prepare_discovery_storms(df, eligible_training, lib; policy)
    eligible_outer = filter(
        entry -> entry.split in ("val", "test"), audit.eligible_entries,
    )
    Set(entry.split for entry in eligible_outer) == Set(("val", "test")) ||
        error("phase sensitivity needs eligible validation and test storms")
    outer_storms = _prepare_discovery_storms(df, eligible_outer, lib; policy)

    results = _phase_threshold_selections(storms, lib)
    expected = length(PHASE_QUIET_THRESHOLDS) * length(PHASE_DERIV_THRESHOLDS)
    length(results) == expected ||
        error("phase sensitivity did not select every threshold configuration")
    rows = _threshold_result_rows(results, lib)
    single_selection = _select_discovery_lambda(storms, lib, Dict{Any,Any}())
    single_paths = _phase_manifest_selection(
        single_selection, "phase_threshold_single_lambda";
        output_paths, producer_script=producer, inputs=selection_inputs,
        kind="threshold_independent_same_cohort_single_equation_control",
    )
    outer_rows = _phase_threshold_outer_rows(
        results, outer_storms, single_selection.model, lib,
    )
    design_rows = _phase_threshold_design_rows(results, lib)
    selection_record = (
        kind="per_threshold_independent_storm_lambda_selection",
        quiet_thresholds=PHASE_QUIET_THRESHOLDS,
        deriv_thresholds=PHASE_DERIV_THRESHOLDS,
        configurations=expected,
        lambda_grid=storm_lambda_grid(),
        selected_decisions=[(
            quiet_thresh=result.quiet_thresh,
            deriv_thresh=result.deriv_thresh,
            selected_lambda=result.selection.selected_lambda,
            selection_rule=result.selection.decision_record.selection_rule,
        ) for result in results],
        comparator_models=("Switching-SINDy", "Single-SINDy", "Burton",
                           "BurtonFull", "OBrien-McPherron"),
        single_control_decision=single_selection.decision_record,
    )

    selection_outputs = (
        "phase_threshold_selection_decisions.csv" => rows.decisions,
        "phase_threshold_selection_candidates.csv" => rows.candidates,
        "phase_threshold_selection_errors.csv" => rows.errors,
        "phase_threshold_selection_support.csv" => rows.support,
        "phase_threshold_selection_inner_split.csv" => rows.inner_split,
    )
    analysis_inputs = copy(selection_inputs)
    for (filename, data) in selection_outputs
        path = _phase_write(
            joinpath(data_dir, filename), data;
            output_paths, producer_script=producer, inputs=selection_inputs,
            selection_record, deterministic=true,
        )
        analysis_inputs["threshold_selection_" *
            replace(splitext(filename)[1], "phase_threshold_selection_" => "")] = path
    end
    for field in propertynames(single_paths)
        analysis_inputs["threshold_single_selection_$(field)"] =
            getproperty(single_paths, field)
    end

    derived_outputs = (
        "phase_threshold_sensitivity.csv" => rows.coefficients,
        "phase_threshold_cohort_counts.csv" => rows.counts,
        "phase_threshold_outer_metrics.csv" => outer_rows.metrics,
        "phase_threshold_outer_trajectories.csv" => outer_rows.trajectories,
        "phase_threshold_design_diagnostics.csv" => design_rows,
    )
    for (filename, data) in derived_outputs
        _phase_write(
            joinpath(data_dir, filename), data;
            output_paths, producer_script=producer, inputs=analysis_inputs,
            selection_record, deterministic=true,
        )
    end
    println("Phase-threshold sensitivity outputs written under: $data_dir")
    println("Independently selected configurations: $(length(results))")
    names = get_term_names(lib)
    _phase_write(
        joinpath(data_dir, "phase_threshold_single_control_coefficients.csv"),
        [(term=names[index], coefficient=single_selection.model[index],
          selected_lambda=single_selection.selected_lambda)
         for index in eachindex(names)];
        output_paths, producer_script=producer,
        inputs=merge(selection_inputs, Dict(
            "threshold_single_selection_$(field)" => getproperty(single_paths, field)
            for field in propertynames(single_paths)
        )),
        selection_record=(
            kind="threshold_independent_same_cohort_single_equation_control",
            decision=single_selection.decision_record,
        ), deterministic=true,
    )
    return (; results, rows, outer_rows, design_rows, single_selection)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    run_phase_sensitivity()
end
