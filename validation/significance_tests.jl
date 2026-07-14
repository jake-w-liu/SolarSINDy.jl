#!/usr/bin/env julia
# Predeclared paired whole-storm inference for the four headline
# SINDy-versus-O'Brien--McPherron comparisons.
#
# Usage:
#   SOLARSINDY_OUTPUT_ROOT=/explicit/run/root \
#     julia --project=SolarSINDy.jl SolarSINDy.jl/validation/significance_tests.jl

using SolarSINDy
using CSV
using DataFrames
using Statistics

isdefined(@__MODULE__, :validation_output_paths) ||
    include(joinpath(@__DIR__, "output_paths.jl"))
isdefined(@__MODULE__, :write_output_manifest) ||
    include(joinpath(@__DIR__, "canonical_provenance.jl"))

const _SIGNIFICANCE_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const _SIGNIFICANCE_DRAWS = 10_000
const _SIGNIFICANCE_SEED = 42
const _SIGNIFICANCE_COVERAGE = 0.95
const _SIGNIFICANCE_SELECTION_RULE =
    "largest_lambda_within_one_standard_error_then_fewer_terms_then_larger_lambda"
const _SIGNIFICANCE_REAL_PRODUCER = abspath(joinpath(@__DIR__, "real_data_discovery.jl"))
const _SIGNIFICANCE_CATALOG_PRODUCER = abspath(joinpath(@__DIR__, "download_omni.jl"))

const HEADLINE_SIGNIFICANCE_SPECS = (
    (
        label = "Validation_C24",
        source = :holdout,
        source_file = "real_holdout_metrics.csv",
        experiment = nothing,
        prefix = "headline_validation_c24",
    ),
    (
        label = "C20-22->C23",
        source = :cross_cycle,
        source_file = "cross_cycle_metrics.csv",
        experiment = "C20-22->C23",
        prefix = "headline_c20_22_to_c23",
    ),
    (
        label = "even->odd",
        source = :cross_cycle,
        source_file = "cross_cycle_metrics.csv",
        experiment = "even->odd",
        prefix = "headline_even_to_odd",
    ),
    (
        label = "C20-23->C25",
        source = :cross_cycle,
        source_file = "cross_cycle_metrics.csv",
        experiment = "C20-23->C25",
        prefix = "headline_c20_23_to_c25",
    ),
)

const HEADLINE_HOLM_PREFIX = "headline_sindy_vs_obrienmcp_holm"
const HEADLINE_CLAIM_SOURCE_FILE = "headline_sindy_vs_obrienmcp_claim_sources.csv"
const ALL_BASELINE_CLAIM_SOURCE_FILE = "paired_sindy_vs_all_baselines_claim_sources.csv"
const PRIMARY_MODEL_SUMMARY_FILE = "primary_model_metric_summary.csv"
const PRIMARY_BASELINES = (
    (model="Burton", slug="burton_simplified"),
    (model="BurtonFull", slug="burton_published"),
    (model="OBrienMcP", slug="obrienmcp"),
)
const PRIMARY_MODEL_NAMES = ("SINDy", "Burton", "BurtonFull", "OBrienMcP")

function _significance_output_paths(output_root::AbstractString)
    paths = String[]
    for spec in HEADLINE_SIGNIFICANCE_SPECS
        append!(paths, [joinpath(output_root, "$(spec.prefix)_$suffix.csv")
                        for suffix in ("pairs", "bootstrap", "summary", "wilcoxon")])
        for baseline in PRIMARY_BASELINES
            baseline.model == "OBrienMcP" && continue
            prefix = replace(spec.prefix, "headline_" => "paired_") *
                     "_vs_$(baseline.slug)"
            append!(paths, [joinpath(output_root, "$(prefix)_$suffix.csv")
                            for suffix in ("pairs", "bootstrap", "summary")])
        end
    end
    append!(paths, [
        joinpath(output_root, "$(HEADLINE_HOLM_PREFIX)_adjusted.csv"),
        joinpath(output_root, HEADLINE_CLAIM_SOURCE_FILE),
        joinpath(output_root, ALL_BASELINE_CLAIM_SOURCE_FILE),
        joinpath(output_root, PRIMARY_MODEL_SUMMARY_FILE),
    ])
    length(paths) == 44 && length(unique(paths)) == 44 ||
        error("significance output inventory must contain exactly 44 unique files")
    return paths
end

function _require_significance_columns(frame::DataFrame, required, source::AbstractString)
    available = Set(Symbol.(names(frame)))
    absent = sort!(collect(setdiff(Set(required), available)); by=string)
    isempty(absent) || throw(ArgumentError(
        "$source is missing required columns $(join(string.(absent), ", "))"
    ))
    return frame
end

function _string_column(frame::DataFrame, column::Symbol, source::AbstractString)
    values = String[]
    for value in frame[!, column]
        value isa AbstractString || throw(ArgumentError(
            "$source column $column must contain only strings"
        ))
        text = String(value)
        isempty(text) && throw(ArgumentError(
            "$source column $column must not contain empty strings"
        ))
        push!(values, text)
    end
    return values
end

function _validate_cross_experiments(cross::DataFrame)
    actual = Set(_string_column(cross, :experiment, "cross-cycle metrics"))
    expected = Set(String(spec.experiment) for spec in HEADLINE_SIGNIFICANCE_SPECS
                   if spec.source === :cross_cycle)
    actual == expected || throw(ArgumentError(
        "cross-cycle experiment labels differ from the predeclared family: " *
        "missing=$(sort!(collect(setdiff(expected, actual)))), " *
        "extra=$(sort!(collect(setdiff(actual, expected))))"
    ))
    return cross
end

function _validate_metric_value(value, source::AbstractString, column::Symbol)
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "$source column $column must contain only real, non-Bool values",
    ))
    converted = try
        Float64(value)
    catch error_value
        error_value isa InterruptException && rethrow()
        throw(ArgumentError(
            "$source column $column contains a value outside the supported Float64 range",
        ))
    end
    isfinite(converted) && converted >= 0 || throw(ArgumentError(
        "$source column $column must contain finite nonnegative Float64 values",
    ))
    return converted
end

function _validate_metric_frame(frame::DataFrame, source::AbstractString;
                                cross_cycle::Bool)
    isempty(frame) && throw(ArgumentError("$source must contain at least two storms"))
    _string_column(frame, :model, source)
    for storm_id in frame.storm_id
        (storm_id === missing || storm_id === nothing || storm_id isa Bool ||
         (storm_id isa Real && !isfinite(storm_id)) ||
         isempty(strip(string(storm_id)))) && throw(ArgumentError(
            "$source contains an invalid storm id",
        ))
    end
    for column in (:rmse_nt, :mae_nt), value in frame[!, column]
        _validate_metric_value(value, source, column)
    end

    expected_models = Set(PRIMARY_MODEL_NAMES)
    grouping_keys = cross_cycle ? [:experiment, :storm_id] : [:storm_id]
    counts = Dict{String,Int}()
    for group in groupby(frame, grouping_keys)
        models = Set(String.(group.model))
        models == expected_models && nrow(group) == length(PRIMARY_MODEL_NAMES) ||
            throw(ArgumentError(
                "$source must contain exactly one row for each predeclared model " *
                "$(collect(PRIMARY_MODEL_NAMES)) per storm",
            ))
        experiment = cross_cycle ? String(first(group.experiment)) : "Validation_C24"
        counts[experiment] = get(counts, experiment, 0) + 1
    end
    expected_experiments = cross_cycle ?
        Set(String(spec.experiment) for spec in HEADLINE_SIGNIFICANCE_SPECS
            if spec.source === :cross_cycle) : Set(("Validation_C24",))
    Set(Base.keys(counts)) == expected_experiments || throw(ArgumentError(
        "$source does not contain the exact predeclared experiment family",
    ))
    for experiment in expected_experiments
        counts[experiment] >= 2 || throw(ArgumentError(
            "$source experiment $experiment requires at least two storms for " *
            "finite sample-standard-deviation summaries",
        ))
    end
    return frame
end

function _validate_significance_metric_frames(holdout::DataFrame, cross::DataFrame)
    _validate_metric_frame(holdout, "held-out metrics"; cross_cycle=false)
    _validate_metric_frame(cross, "cross-cycle metrics"; cross_cycle=true)
    return nothing
end

function _model_rows(frame::DataFrame, model::AbstractString, source::AbstractString)
    labels = _string_column(frame, :model, source)
    return frame[labels .== model, :]
end

function _comparison_frame(holdout::DataFrame, cross::DataFrame, spec)
    if spec.source === :holdout
        return holdout
    end
    experiments = _string_column(cross, :experiment, "cross-cycle metrics")
    return cross[experiments .== spec.experiment, :]
end

function _write_significance_csv(path::AbstractString, rows)
    mkpath(dirname(path))
    SolarSINDy._require_regular_output_target(path)
    temporary, io = mktemp(dirname(path); cleanup=false)
    close(io)
    try
        CSV.write(temporary, rows isa DataFrame ? rows : DataFrame(rows))
        SolarSINDy._atomic_replace_regular(temporary, path)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return path
end

function _significance_manifest_record(spec, artifact::Symbol;
                                       reference::AbstractString="OBrienMcP")
    in_holm_family = reference == "OBrienMcP"
    return (
        kind="predeclared_paired_whole_storm_inference",
        artifact=String(artifact),
        experiment=spec.label,
        source=String(spec.source),
        source_experiment=spec.experiment === nothing ?
            "Validation_C24" : String(spec.experiment),
        model="SINDy",
        reference=String(reference),
        effect="unweighted_mean_of_per_storm_model_minus_reference_RMSE",
        relative_effect="per_storm_RMSE_difference_divided_by_reference_RMSE",
        bootstrap_unit="whole_storm_pair",
        bootstrap_draws=_SIGNIFICANCE_DRAWS,
        interval="percentile",
        interval_coverage=_SIGNIFICANCE_COVERAGE,
        secondary_test=in_holm_family ?
            "two_sided_paired_Wilcoxon_normal_approximation" : "none",
        multiplicity_family=in_holm_family ?
            String[item.label for item in HEADLINE_SIGNIFICANCE_SPECS] : String[],
        multiplicity_method=in_holm_family ?
            "Holm_step_down_familywise_adjustment" : "none",
    )
end

function _write_interval_statistics(result, output_root::AbstractString,
                                    prefix::AbstractString)
    paths = (
        pairs=joinpath(output_root, "$(prefix)_pairs.csv"),
        bootstrap=joinpath(output_root, "$(prefix)_bootstrap.csv"),
        summary=joinpath(output_root, "$(prefix)_summary.csv"),
    )
    frames = (
        pairs=DataFrame(result.pair_records),
        bootstrap=DataFrame(result.bootstrap_records),
        summary=DataFrame(result.summary_records),
    )
    return SolarSINDy._write_selection_csv_set(paths, frames)
end

function _scaled_nonnegative_mean(values)
    converted = Float64.(values)
    scale = maximum(converted)
    estimate = iszero(scale) ? 0.0 : scale * mean(converted ./ scale)
    isfinite(estimate) || throw(ArgumentError("descriptive metric mean is not finite"))
    return estimate
end

function _scaled_sample_standard_deviation(values)
    converted = Float64.(values)
    length(converted) >= 2 || throw(ArgumentError(
        "sample standard deviation requires at least two storms",
    ))
    scale = maximum(converted)
    estimate = iszero(scale) ? 0.0 : scale * std(converted ./ scale; corrected=true)
    isfinite(estimate) || throw(ArgumentError(
        "descriptive metric sample standard deviation is not finite",
    ))
    return estimate
end

function _primary_metric_summary(holdout::DataFrame, cross::DataFrame)
    holdout_with_label = copy(holdout)
    holdout_with_label[!, :experiment] = fill("Validation_C24", nrow(holdout))
    combined = vcat(
        select(holdout_with_label, :experiment, :storm_id, :model, :rmse_nt, :mae_nt),
        select(cross, :experiment, :storm_id, :model, :rmse_nt, :mae_nt),
    )
    rows = NamedTuple[]
    for group in groupby(combined, [:experiment, :model])
        n = nrow(group)
        n >= 2 || throw(ArgumentError(
            "primary metric summary requires at least two storms per group",
        ))
        rmse_sd = _scaled_sample_standard_deviation(group.rmse_nt)
        mae_sd = _scaled_sample_standard_deviation(group.mae_nt)
        push!(rows, (
            experiment=String(first(group.experiment)),
            model=String(first(group.model)),
            n_storms=n,
            mean_rmse_nt=_scaled_nonnegative_mean(group.rmse_nt),
            standard_deviation_rmse_nt=rmse_sd,
            standard_error_rmse_nt=rmse_sd / sqrt(n),
            mean_mae_nt=_scaled_nonnegative_mean(group.mae_nt),
            standard_deviation_mae_nt=mae_sd,
            standard_error_mae_nt=mae_sd / sqrt(n),
            aggregation="unweighted_whole_storm_mean",
            standard_deviation_definition="sample_standard_deviation_n_minus_1",
            standard_error_definition="sample_standard_deviation_divided_by_sqrt_n",
        ))
    end
    sort!(rows; by=row -> (row.experiment, row.model))
    return rows
end

function _manifest_significance_outputs!(result, provenance)
    inputs = _significance_direct_input_paths(provenance)
    mode = provenance.mode
    package_root = hasproperty(provenance, :package_root) ?
        provenance.package_root : _SIGNIFICANCE_PACKAGE_ROOT

    outputs = copy(result.all_output_paths)
    manifest_paths = [path * ".manifest.json" for path in outputs]
    snapshot = SolarSINDy._snapshot_regular_file_set(manifest_paths)

    try
        length(unique(outputs)) == length(outputs) || error(
            "significance workflow produced duplicate output paths",
        )
        all(isfile, outputs) || error(
            "significance workflow did not produce every declared regular-file artifact",
        )
        specs = Dict(item.spec.label => item.spec for item in result.comparisons)
        for item in result.artifacts
            spec = specs[item.label]
            for field in propertynames(item.paths)
                path = getproperty(item.paths, field)
                seeded = field in (:bootstrap, :summary)
                write_output_manifest(path;
                    producer_script=@__FILE__,
                    input_paths=inputs,
                    selection_record=_significance_manifest_record(spec, field),
                    seed=seeded ? _SIGNIFICANCE_SEED : nothing,
                    deterministic=!seeded,
                    package_root,
                    mode,
                )
            end
        end
        for item in result.baseline_artifacts
            spec = specs[item.label]
            for field in propertynames(item.paths)
                path = getproperty(item.paths, field)
                seeded = field in (:bootstrap, :summary)
                write_output_manifest(path;
                    producer_script=@__FILE__,
                    input_paths=inputs,
                    selection_record=_significance_manifest_record(
                        spec, field; reference=item.reference,
                    ),
                    seed=seeded ? _SIGNIFICANCE_SEED : nothing,
                    deterministic=!seeded,
                    package_root,
                    mode,
                )
            end
        end

        family_record = (
            kind="predeclared_headline_multiplicity_family",
            artifact="holm_adjustment",
            labels=String[item.spec.label for item in result.comparisons],
            raw_test="two_sided_paired_Wilcoxon_normal_approximation",
            method="Holm_step_down_familywise_adjustment",
            family_size=length(result.comparisons),
        )
        write_output_manifest(result.holm_path;
            producer_script=@__FILE__, input_paths=inputs,
            selection_record=family_record, deterministic=true,
            package_root, mode,
        )
        write_output_manifest(result.claim_source_path;
            producer_script=@__FILE__, input_paths=inputs,
            selection_record=merge(family_record, (
                artifact="claim_source_summary",
                bootstrap_draws=_SIGNIFICANCE_DRAWS,
                bootstrap_seed=_SIGNIFICANCE_SEED,
                interval_coverage=_SIGNIFICANCE_COVERAGE,
            )),
            seed=_SIGNIFICANCE_SEED, deterministic=false,
            package_root, mode,
        )
        write_output_manifest(result.all_baseline_claim_source_path;
            producer_script=@__FILE__, input_paths=inputs,
            selection_record=(
                kind="paired_whole_storm_intervals_against_all_matched_baselines",
                artifact="all_baseline_claim_source_summary",
                references=String[baseline.model for baseline in PRIMARY_BASELINES],
                experiments=String[item.spec.label for item in result.comparisons],
                bootstrap_draws=_SIGNIFICANCE_DRAWS,
                interval_coverage=_SIGNIFICANCE_COVERAGE,
                holm_family="OBrienMcP_only",
            ),
            seed=_SIGNIFICANCE_SEED, deterministic=false,
            package_root, mode,
        )
        write_output_manifest(result.aggregate_path;
            producer_script=@__FILE__, input_paths=inputs,
            selection_record=(
                kind="unweighted_whole_storm_descriptive_metric_summary",
                metrics=("RMSE_nT", "MAE_nT"),
                experiments=String[item.spec.label for item in result.comparisons],
                models=PRIMARY_MODEL_NAMES,
                aggregation="unweighted_whole_storm_mean",
                standard_deviation="sample_standard_deviation_n_minus_1",
                standard_error="sample_standard_deviation_divided_by_sqrt_n",
            ),
            deterministic=true,
            package_root, mode,
        )

        for path in outputs
            verify_output_manifest(path;
                package_root,
                require_canonical=mode == :canonical,
                verify_source=true,
            )
        end
    catch
        SolarSINDy._restore_regular_file_set!(snapshot)
        rethrow()
    end
    SolarSINDy._discard_regular_file_snapshot!(snapshot)
    return outputs
end

function _significance_manifest_inputs(record)
    return Dict(String(_json_get(input, "name")) => input
                for input in _json_get(record, "inputs"))
end

function _significance_direct_input_paths(provenance)
    return (
        holdout_metrics=abspath(provenance.input_paths.holdout_metrics),
        cross_cycle_metrics=abspath(provenance.input_paths.cross_cycle_metrics),
        storm_eligibility=abspath(provenance.input_paths.storm_eligibility),
        omni_extracted=abspath(provenance.omni_path),
        storm_catalog=abspath(provenance.catalog_path),
        primary_selection_split=abspath(
            provenance.selection_split_paths.holdout["Validation C24"],
        ),
        cross_c20_22_to_c23_selection_split=abspath(
            provenance.selection_split_paths.cross["C20-22->C23"],
        ),
        cross_even_to_odd_selection_split=abspath(
            provenance.selection_split_paths.cross["even->odd"],
        ),
        cross_c20_23_to_c25_selection_split=abspath(
            provenance.selection_split_paths.cross["C20-23->C25"],
        ),
    )
end

function _significance_direct_input_hashes(provenance)
    paths = _significance_direct_input_paths(provenance)
    return NamedTuple{propertynames(paths)}(
        Tuple(provenance_sha256(getproperty(paths, name)) for name in propertynames(paths)),
    )
end

function _assert_significance_inputs_unchanged(provenance, expected_hashes)
    current = _significance_direct_input_hashes(provenance)
    for name in propertynames(expected_hashes)
        getproperty(current, name) == getproperty(expected_hashes, name) || error(
            "significance direct input $name changed during the workflow",
        )
    end
    return nothing
end

function _verify_significance_output_bindings!(result, provenance, expected_hashes)
    expected_names = Set(String.(propertynames(expected_hashes)))
    for path in result.all_output_paths
        record = verify_output_manifest(path;
            package_root=provenance.package_root,
            require_canonical=provenance.mode == :canonical,
            verify_source=true,
        )
        inputs = _significance_manifest_inputs(record)
        Set(Base.keys(inputs)) == expected_names || error(
            "significance output manifest has an unexpected direct-input set",
        )
        for name in propertynames(expected_hashes)
            String(_json_get(inputs[String(name)], "sha256")) ==
                getproperty(expected_hashes, name) || error(
                    "significance output manifest does not bind the initial $name input",
                )
        end
    end
    return nothing
end

function _require_significance_manifest_input(record, name::AbstractString,
                                              expected_path::AbstractString)
    inputs = _significance_manifest_inputs(record)
    haskey(inputs, name) || error("metrics manifest is missing required input $name")
    actual = String(_json_get(inputs[name], "path"))
    actual == abspath(expected_path) || error(
        "metrics manifest input $name points to $actual, expected $(abspath(expected_path))",
    )
    return inputs[name]
end

function _verify_selection_decision(selection, expected_scope::AbstractString)
    String(_json_get(selection, "kind")) ==
        "fixed_grid_whole_storm_forward_selection" ||
        error("metrics manifest does not record fixed-grid whole-storm selection")
    String(_json_get(selection, "basis")) == "full" ||
        error("headline metrics must use the full identifiable basis")
    String(_json_get(selection, "scope")) == expected_scope ||
        error("metrics manifest selection scope differs from $expected_scope")
    decision = _json_get(selection, "decision")
    String(_json_get(decision, "selection_rule")) ==
        _SIGNIFICANCE_SELECTION_RULE ||
        error("metrics manifest uses an unexpected lambda-selection rule")
    return nothing
end

function _verified_metric_split(record, input_name::AbstractString,
                                expected_basename::AbstractString, provenance)
    input = _significance_manifest_inputs(record)
    haskey(input, input_name) || error(
        "metrics manifest is missing selection split input $input_name",
    )
    path = String(_json_get(input[input_name], "path"))
    basename(path) == expected_basename || error(
        "selection split $input_name has an unexpected filename",
    )
    split_record = verify_output_manifest(path;
        package_root=provenance.package_root,
        require_canonical=provenance.mode == :canonical,
        verify_source=true,
    )
    String(_json_get(_json_get(split_record, "producer"), "path")) ==
        _SIGNIFICANCE_REAL_PRODUCER || error(
            "selection split was not produced by real_data_discovery.jl",
        )
    frame = CSV.read(path, DataFrame)
    all(in(propertynames(frame)), (:storm_id, :inner_split)) ||
        error("selection split is missing storm_id or inner_split")
    all(value -> String(value) in ("train", "validation"), frame.inner_split) ||
        error("selection split contains an invalid inner role")
    return path
end

function _verify_metric_manifest(path::AbstractString, provenance, role::Symbol)
    record = verify_output_manifest(path;
        package_root=provenance.package_root,
        require_canonical=provenance.mode == :canonical,
        verify_source=true,
    )
    producer = String(_json_get(_json_get(record, "producer"), "path"))
    producer == _SIGNIFICANCE_REAL_PRODUCER || error(
        "headline metric input was not produced by real_data_discovery.jl",
    )
    _require_significance_manifest_input(record, "omni_extracted", provenance.omni_path)
    _require_significance_manifest_input(record, "storm_catalog", provenance.catalog_path)
    _require_significance_manifest_input(
        record, "storm_observation_audit", provenance.input_paths.storm_eligibility,
    )
    selection = _json_get(record, "selection_record")
    if role == :holdout
        basename(path) == "real_holdout_metrics.csv" ||
            error("held-out metric input has an unexpected filename")
        _verify_selection_decision(selection, "primary")
        String(_json_get(selection, "outer_split")) == "cycle_24" ||
            error("held-out metrics are not the predeclared cycle-24 outer set")
        split_paths = Dict(
            "Validation C24" => _verified_metric_split(
                record, "full_primary_split", "primary_lambda_inner_split.csv",
                provenance,
            ),
        )
    elseif role == :cross_cycle
        basename(path) == "cross_cycle_metrics.csv" ||
            error("cross-cycle metric input has an unexpected filename")
        String(_json_get(selection, "kind")) ==
            "independently_selected_cross_cycle_outer_metrics" ||
            error("cross-cycle metrics lack independent-selection provenance")
        String(_json_get(selection, "basis")) == "full" ||
            error("cross-cycle headline metrics must use the full basis")
        experiments = collect(_json_get(selection, "experiments"))
        length(experiments) == 3 ||
            error("cross-cycle manifest must record exactly three selections")
        for experiment in experiments
            scope = String(_json_get(experiment, "scope"))
            _verify_selection_decision(experiment, scope)
        end
        Set(String(_json_get(experiment, "scope")) for experiment in experiments) ==
            Set(("C20-22->C23", "even->odd", "C20-23->C25")) ||
            error("cross-cycle manifest scopes differ from the predeclared family")
        split_paths = Dict(
            scope => _verified_metric_split(
                record, "cross_$(index)_full_split",
                "$(prefix)_inner_split.csv", provenance,
            ) for (index, (scope, prefix)) in enumerate((
                ("C20-22->C23", "cross_c20_22_to_c23_lambda"),
                ("even->odd", "cross_even_to_odd_lambda"),
                ("C20-23->C25", "cross_c20_23_to_c25_lambda"),
            ))
        )
    else
        error("unknown significance input role $role")
    end
    return (; record, split_paths)
end

function _assert_manifested_metric_cohorts(holdout::DataFrame, cross::DataFrame,
                                           catalog_path::AbstractString,
                                           eligibility_path::AbstractString,
                                           selection_split_paths)
    for (frame, label) in ((holdout, "held-out"), (cross, "cross-cycle"))
        for column in (:rmse_nt, :mae_nt)
            all(value -> value isa Real && !(value isa Bool) &&
                         isfinite(value) && value >= 0, frame[!, column]) ||
                error("$label $column values must be finite nonnegative numbers")
        end
    end
    catalog = CSV.read(catalog_path, DataFrame)
    cycle_by_id = Dict(string(row.storm_id) => Int(row.solar_cycle)
                       for row in eachrow(catalog))
    split_by_id = Dict(string(row.storm_id) => String(row.split)
                       for row in eachrow(catalog))
    eligibility = CSV.read(eligibility_path, DataFrame)
    required_eligibility = (:storm_id, :solar_cycle, :catalog_split, :eligible)
    all(in(propertynames(eligibility)), required_eligibility) || error(
        "storm eligibility audit is missing required columns",
    )
    length(unique(string.(eligibility.storm_id))) == nrow(eligibility) ||
        error("storm eligibility audit contains duplicate storm ids")
    for row in eachrow(eligibility)
        id = string(row.storm_id)
        haskey(cycle_by_id, id) || error("eligibility storm $id is absent from the catalog")
        Int(row.solar_cycle) == cycle_by_id[id] ||
            error("eligibility storm $id has a stale solar cycle")
        String(row.catalog_split) == split_by_id[id] ||
            error("eligibility storm $id has a stale catalog split")
    end
    eligible_ids(cycles) = Set(string(row.storm_id) for row in eachrow(eligibility)
                               if Bool(row.eligible) && Int(row.solar_cycle) in cycles)
    expected_models = Set(("SINDy", "Burton", "BurtonFull", "OBrienMcP"))
    function assert_groups(frame, keys, expected_cycles)
        identity_columns = (:anchor_catalog_index, :driver_start_catalog_index,
                            :driver_end_catalog_index, :scored_catalog_indices,
                            :cohort_signature_sha256)
        all(in(propertynames(frame)), identity_columns) || error(
            "manifested metrics are missing exact cohort-identity columns",
        )
        for group in groupby(frame, keys)
            id = string(first(group.storm_id))
            haskey(cycle_by_id, id) || error("metric storm $id is absent from the catalog")
            Set(String.(group.model)) == expected_models ||
                error("metric storm $id does not contain the exact four-model comparator set")
            nrow(group) == length(expected_models) ||
                error("metric storm $id contains duplicate model rows")
            all(field -> length(unique(group[!, field])) == 1, identity_columns) ||
                error("metric storm $id used unequal comparator cohorts")
            cycle_by_id[id] in expected_cycles ||
                error("metric storm $id is outside its predeclared outer solar cycle")
        end
    end
    assert_groups(holdout, [:storm_id], Set((24,)))
    all(id -> split_by_id[string(id)] == "val", unique(holdout.storm_id)) ||
        error("held-out metrics contain a storm outside the catalog validation split")
    Set(string.(unique(holdout.storm_id))) == eligible_ids(Set((24,))) ||
        error("held-out metrics do not contain every eligible cycle-24 storm exactly")
    holdout_selected = Set(string.(CSV.read(
        selection_split_paths.holdout["Validation C24"], DataFrame,
    ).storm_id))
    isempty(intersect(Set(string.(unique(holdout.storm_id))), holdout_selected)) ||
        error("held-out metrics contain a storm contacted during model selection")
    expected_cross_cycles = Dict(
        "C20-22->C23" => Set((23,)),
        "even->odd" => Set((21, 23)),
        "C20-23->C25" => Set((25,)),
    )
    for (experiment, cycles) in expected_cross_cycles
        subset = cross[String.(cross.experiment) .== experiment, :]
        assert_groups(subset, [:experiment, :storm_id], cycles)
        Set(cycle_by_id[string(id)] for id in unique(subset.storm_id)) == cycles ||
            error("$experiment does not contain every predeclared outer solar cycle")
        Set(string.(unique(subset.storm_id))) == eligible_ids(cycles) ||
            error("$experiment does not contain every eligible outer storm exactly")
        selected = Set(string.(CSV.read(
            selection_split_paths.cross[experiment], DataFrame,
        ).storm_id))
        isempty(intersect(Set(string.(unique(subset.storm_id))), selected)) ||
            error("$experiment metrics contain a storm contacted during model selection")
    end
    return nothing
end

function _verify_significance_provenance_inputs(provenance)
    for field in (:input_paths, :mode, :package_root, :omni_path, :catalog_path)
        hasproperty(provenance, field) ||
            throw(ArgumentError("significance provenance is missing $field"))
    end
    validated = merge(provenance, (
        mode=_provenance_mode(provenance.mode),
        package_root=abspath(provenance.package_root),
        omni_path=abspath(provenance.omni_path),
        catalog_path=abspath(provenance.catalog_path),
    ))
    verify_omni_input(validated.omni_path; mode=validated.mode)
    catalog_record = verify_storm_catalog(validated.catalog_path;
        omni_path=validated.omni_path,
        parameters=storm_catalog_parameters(),
        package_root=validated.package_root,
        mode=validated.mode,
        verify_source=true,
    )
    catalog_producer = String(_json_get(
        _json_get(catalog_record, "producer"), "path",
    ))
    catalog_producer == _SIGNIFICANCE_CATALOG_PRODUCER || error(
        "headline catalog was not produced by download_omni.jl",
    )
    inputs = validated.input_paths
    all(name -> hasproperty(inputs, name),
        (:holdout_metrics, :cross_cycle_metrics, :storm_eligibility)) ||
        throw(ArgumentError(
            "significance provenance requires metric and eligibility paths",
        ))
    eligibility_record = verify_output_manifest(inputs.storm_eligibility;
        package_root=validated.package_root,
        require_canonical=validated.mode == :canonical,
        verify_source=true,
    )
    String(_json_get(_json_get(eligibility_record, "producer"), "path")) ==
        _SIGNIFICANCE_REAL_PRODUCER || error(
            "storm eligibility input was not produced by real_data_discovery.jl",
        )
    basename(inputs.storm_eligibility) == "real_storm_eligibility.csv" ||
        error("storm eligibility input has an unexpected filename")
    holdout_verified = _verify_metric_manifest(
        inputs.holdout_metrics, validated, :holdout,
    )
    cross_verified = _verify_metric_manifest(
        inputs.cross_cycle_metrics, validated, :cross_cycle,
    )
    return merge(validated, (selection_split_paths=(
        holdout=holdout_verified.split_paths,
        cross=cross_verified.split_paths,
    ),))
end

"""
    run_headline_significance(holdout, cross, output_root)

Compute and persist the predeclared four-comparison SINDy-versus-O'Brien--
McPherron family. Each effect uses an unweighted paired whole-storm bootstrap
with 10,000 draws and seed 42. The four Wilcoxon p-values are adjusted once as
one Holm family. This DataFrame overload is intentionally computation-only;
canonical artifacts must use `run_manifested_headline_significance` so the
computed frames cannot diverge from their recorded input paths.
"""
function run_headline_significance(holdout::DataFrame, cross::DataFrame,
                                   output_root::AbstractString)
    isempty(strip(output_root)) && throw(ArgumentError("output_root must not be empty"))
    _require_significance_columns(
        holdout, (:storm_id, :model, :rmse_nt, :mae_nt), "held-out metrics"
    )
    _require_significance_columns(
        cross, (:experiment, :storm_id, :model, :rmse_nt, :mae_nt),
        "cross-cycle metrics",
    )
    _validate_cross_experiments(cross)
    _validate_significance_metric_frames(holdout, cross)

    # Validate the complete family before writing any artifact. Pairing by storm
    # id inside `paired_storm_statistics` catches missing pairs and duplicates.
    computed = NamedTuple[]
    for spec in HEADLINE_SIGNIFICANCE_SPECS
        frame = _comparison_frame(holdout, cross, spec)
        model = _model_rows(frame, "SINDy", spec.label)
        reference = _model_rows(frame, "OBrienMcP", spec.label)
        result = paired_storm_statistics(model, reference;
            rmse_value=row -> getproperty(row, :rmse_nt),
            draws=_SIGNIFICANCE_DRAWS,
            coverage=_SIGNIFICANCE_COVERAGE,
            seed=_SIGNIFICANCE_SEED,
        )
        isfinite(result.wilcoxon.p) || throw(ArgumentError(
            "$(spec.label) has an undefined paired Wilcoxon p-value"
        ))
        push!(computed, (; spec, result))
    end

    labels = [item.spec.label for item in computed]
    raw_p_values = [item.result.wilcoxon.p for item in computed]
    holm = holm_adjust(raw_p_values; labels=labels)

    headline_by_label = Dict(item.spec.label => item.result for item in computed)
    baseline_comparisons = NamedTuple[]
    for spec in HEADLINE_SIGNIFICANCE_SPECS
        frame = _comparison_frame(holdout, cross, spec)
        model = _model_rows(frame, "SINDy", spec.label)
        for baseline in PRIMARY_BASELINES
            result = baseline.model == "OBrienMcP" ? headline_by_label[spec.label] :
                paired_storm_statistics(
                    model, _model_rows(frame, baseline.model, spec.label);
                    rmse_value=row -> getproperty(row, :rmse_nt),
                    draws=_SIGNIFICANCE_DRAWS,
                    coverage=_SIGNIFICANCE_COVERAGE,
                    seed=_SIGNIFICANCE_SEED,
                )
            prefix = baseline.model == "OBrienMcP" ? spec.prefix :
                replace(spec.prefix, "headline_" => "paired_") * "_vs_$(baseline.slug)"
            push!(baseline_comparisons, (; spec, baseline, prefix, result))
        end
    end
    aggregate_rows = _primary_metric_summary(holdout, cross)

    output_root = abspath(output_root)
    mkpath(dirname(output_root))
    working_root = mktempdir(dirname(output_root))
    written_paths = String[]
    artifact_records = NamedTuple[]
    baseline_artifacts = NamedTuple[]
    holm_path = joinpath(working_root, "$(HEADLINE_HOLM_PREFIX)_adjusted.csv")
    claim_source_path = joinpath(working_root, HEADLINE_CLAIM_SOURCE_FILE)
    all_baseline_claim_source_path = joinpath(working_root, ALL_BASELINE_CLAIM_SOURCE_FILE)
    aggregate_path = joinpath(working_root, PRIMARY_MODEL_SUMMARY_FILE)
    try
        for item in computed
            paths = write_paired_storm_statistics(
                item.result, working_root; prefix=item.spec.prefix,
            )
            append!(written_paths, String[getproperty(paths, field)
                                          for field in propertynames(paths)])
            push!(artifact_records,
                  (; label=item.spec.label, prefix=item.spec.prefix, paths))
        end
        holm_path = write_holm_adjustment(
            holm, working_root; prefix=HEADLINE_HOLM_PREFIX,
        )
        push!(written_paths, holm_path)

        claim_rows = NamedTuple[]
        for (item, adjusted) in zip(computed, holm)
            result = item.result
            spec = item.spec
            push!(claim_rows, (
                experiment = spec.label,
                comparison = "SINDy_vs_OBrienMcP",
                source_file = spec.source_file,
                source_experiment = spec.experiment === nothing ?
                    "Validation_C24" : spec.experiment,
                artifact_prefix = spec.prefix,
                n_storms = length(result.pair_records),
                mean_rmse_difference_nt = result.mean_rmse_difference,
                rmse_ci_lower_nt = result.rmse_difference_interval[1],
                rmse_ci_upper_nt = result.rmse_difference_interval[2],
                mean_relative_difference_fraction = result.mean_relative_difference,
                relative_ci_lower_fraction = result.relative_difference_interval[1],
                relative_ci_upper_fraction = result.relative_difference_interval[2],
                interval_coverage = result.coverage,
                bootstrap_draws = result.draws,
                seed = result.seed,
                wilcoxon_n_nonzero = result.wilcoxon.n,
                wilcoxon_w = result.wilcoxon.w,
                wilcoxon_z = result.wilcoxon.z,
                wilcoxon_p_value = result.wilcoxon.p,
                holm_p_value = adjusted.holm_p_value,
                holm_rank = adjusted.holm_rank,
                holm_family_size = adjusted.family_size,
            ))
        end
        claim_source_path = _write_significance_csv(claim_source_path, claim_rows)
        push!(written_paths, claim_source_path)

        all_baseline_claim_rows = NamedTuple[]
        for item in baseline_comparisons
            result = item.result
            spec = item.spec
            baseline = item.baseline
            if baseline.model != "OBrienMcP"
                paths = _write_interval_statistics(result, working_root, item.prefix)
                append!(written_paths, String[getproperty(paths, field)
                                              for field in propertynames(paths)])
                push!(baseline_artifacts, (
                    label=spec.label,
                    reference=baseline.model,
                    prefix=item.prefix,
                    paths,
                ))
            end
            push!(all_baseline_claim_rows, (
                experiment=spec.label,
                comparison="SINDy_vs_$(baseline.model)",
                reference_model=baseline.model,
                source_file=spec.source_file,
                source_experiment=spec.experiment === nothing ?
                    "Validation_C24" : spec.experiment,
                artifact_prefix=item.prefix,
                n_storms=length(result.pair_records),
                mean_rmse_difference_nt=result.mean_rmse_difference,
                rmse_ci_lower_nt=result.rmse_difference_interval[1],
                rmse_ci_upper_nt=result.rmse_difference_interval[2],
                mean_relative_difference_fraction=result.mean_relative_difference,
                relative_ci_lower_fraction=result.relative_difference_interval[1],
                relative_ci_upper_fraction=result.relative_difference_interval[2],
                interval_coverage=result.coverage,
                bootstrap_draws=result.draws,
                seed=result.seed,
                in_predeclared_holm_family=baseline.model == "OBrienMcP",
            ))
        end
        all_baseline_claim_source_path = _write_significance_csv(
            all_baseline_claim_source_path, all_baseline_claim_rows,
        )
        push!(written_paths, all_baseline_claim_source_path)
        aggregate_path = _write_significance_csv(aggregate_path, aggregate_rows)
        push!(written_paths, aggregate_path)
        length(unique(written_paths)) == length(written_paths) || error(
            "significance workflow produced duplicate output paths",
        )
        all(isfile, written_paths) || error(
            "significance workflow did not produce every declared regular-file artifact",
        )
    catch
        rm(working_root; recursive=true, force=true)
        rethrow()
    end

    final_paths = [joinpath(output_root, basename(path)) for path in written_paths]
    try
        mkpath(output_root)
        SolarSINDy._atomic_install_regular_file_set(written_paths, final_paths)
    catch
        rm(working_root; recursive=true, force=true)
        rethrow()
    end
    rm(working_root; recursive=true, force=true)
    path_map = Dict(staged => final for (staged, final) in zip(written_paths, final_paths))
    remap_paths(paths) = NamedTuple{propertynames(paths)}(
        Tuple(path_map[getproperty(paths, field)] for field in propertynames(paths)),
    )
    artifact_records = [merge(item, (paths=remap_paths(item.paths),))
                        for item in artifact_records]
    baseline_artifacts = [merge(item, (paths=remap_paths(item.paths),))
                          for item in baseline_artifacts]
    holm_path = path_map[holm_path]
    claim_source_path = path_map[claim_source_path]
    all_baseline_claim_source_path = path_map[all_baseline_claim_source_path]
    aggregate_path = path_map[aggregate_path]
    all_output_paths = final_paths
    result = (;
        comparisons=computed,
        holm,
        artifacts=artifact_records,
        holm_path,
        claim_source_path,
        baseline_comparisons,
        baseline_artifacts,
        all_baseline_claim_source_path,
        aggregate_path,
        all_output_paths,
    )
    return result
end

"""Verify, read, compute, manifest, and reverify the canonical headline family."""
function run_manifested_headline_significance(
        holdout_path::AbstractString, cross_path::AbstractString,
        output_root::AbstractString;
        mode=validation_run_mode(),
        omni_path::AbstractString,
        catalog_path::AbstractString,
        eligibility_path::AbstractString=joinpath(
            dirname(abspath(holdout_path)), "real_storm_eligibility.csv",
        ),
        package_root::AbstractString=_SIGNIFICANCE_PACKAGE_ROOT,
        _after_compute_hook::Function=() -> nothing,
        _after_manifest_hook::Function=() -> nothing)
    provenance = _verify_significance_provenance_inputs((
        input_paths=(holdout_metrics=abspath(holdout_path),
                     cross_cycle_metrics=abspath(cross_path),
                     storm_eligibility=abspath(eligibility_path)),
        mode,
        package_root,
        omni_path,
        catalog_path,
    ))
    input_hashes = _significance_direct_input_hashes(provenance)
    holdout = CSV.read(provenance.input_paths.holdout_metrics, DataFrame)
    cross = CSV.read(provenance.input_paths.cross_cycle_metrics, DataFrame)
    _assert_significance_inputs_unchanged(provenance, input_hashes)
    _require_significance_columns(
        holdout, (:storm_id, :model, :rmse_nt, :mae_nt), "held-out metrics",
    )
    _require_significance_columns(
        cross, (:experiment, :storm_id, :model, :rmse_nt, :mae_nt),
        "cross-cycle metrics",
    )
    _validate_cross_experiments(cross)
    _validate_significance_metric_frames(holdout, cross)
    _assert_manifested_metric_cohorts(
        holdout, cross, provenance.catalog_path,
        provenance.input_paths.storm_eligibility,
        provenance.selection_split_paths,
    )
    expected_outputs = _significance_output_paths(abspath(output_root))
    transaction_paths = vcat(expected_outputs,
                             [path * ".manifest.json" for path in expected_outputs])
    snapshot = SolarSINDy._snapshot_regular_file_set(transaction_paths)
    local result
    try
        result = run_headline_significance(holdout, cross, output_root)
        _after_compute_hook()
        _assert_significance_inputs_unchanged(provenance, input_hashes)
        _manifest_significance_outputs!(result, provenance)
        _after_manifest_hook()
        _assert_significance_inputs_unchanged(provenance, input_hashes)
        _verify_significance_provenance_inputs(provenance)
        _assert_significance_inputs_unchanged(provenance, input_hashes)
        _verify_significance_output_bindings!(result, provenance, input_hashes)
    catch
        SolarSINDy._restore_regular_file_set!(snapshot)
        rethrow()
    end
    SolarSINDy._discard_regular_file_snapshot!(snapshot)
    return result
end

function _significance_main()
    explicit_root = get(ENV, "SOLARSINDY_OUTPUT_ROOT", "")
    isempty(strip(explicit_root)) && throw(ArgumentError(
        "SOLARSINDY_OUTPUT_ROOT must name an explicit validation output root"
    ))
    paths = validation_output_paths()
    paths.explicit || error("validation significance outputs require an explicit root")

    holdout_path = joinpath(paths.data, "real_holdout_metrics.csv")
    cross_path = joinpath(paths.data, "cross_cycle_metrics.csv")
    isfile(holdout_path) || error("missing canonical held-out metrics: $holdout_path")
    isfile(cross_path) || error("missing canonical cross-cycle metrics: $cross_path")
    result = run_manifested_headline_significance(
        holdout_path,
        cross_path,
        paths.data,
        mode=paths.mode,
        omni_path=paths.omni,
        catalog_path=joinpath(paths.data, "storm_catalog.csv"),
        package_root=_SIGNIFICANCE_PACKAGE_ROOT,
    )
    println("Wrote four-comparison significance artifacts under $(paths.data)")
    println("Claim-source summary: $(result.claim_source_path)")
    return result
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && _significance_main()
