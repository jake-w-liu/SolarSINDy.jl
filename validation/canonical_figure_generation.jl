#!/usr/bin/env julia
# Canonical AISR figures: verified CSV inputs only, PlotlySupply PDF outputs only.

module CanonicalFigureGeneration

using SolarSINDy
using CSV
using DataFrames
using Statistics
using PlotlySupply

include(joinpath(@__DIR__, "output_paths.jl"))
include(joinpath(@__DIR__, "canonical_provenance.jl"))
include(joinpath(@__DIR__, "canonical_artifact_inventory.jl"))

export prepare_canonical_figure_inputs, build_canonical_figures,
       run_canonical_figure_generation

const _FIGURE_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const _BLUE = "#0072B2"
const _ORANGE = "#D55E00"
const _GREEN = "#009E73"
const _PINK = "#CC79A7"
const _BLACK = "#000000"
const _GREY = "#666666"
const _TEXT = "#444444"
const _FONT_FAMILY = "Verdana"
const _BASE_FONT_SIZE = 24
const _AXIS_TITLE_FONT_SIZE = 24
const _FIGURE_WIDTH = 1008
const _DISCOVERY_FIGURE_HEIGHT = 560
const _LAMBDA_FIGURE_HEIGHT = 720
const _STABILITY_FIGURE_HEIGHT = 800
const _SYNTHETIC_FIGURE_HEIGHT = 800
const _PAIRED_FIGURE_HEIGHT = 560

const _LAMBDA_TICK_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0]
const _LAMBDA_TICK_LABELS = ["0.01", "0.1", "1", "10", "100", "1000", "10000"]

const _TERM_DISPLAY = Dict(
    "Dst_star" => "Dst*",
    "V^2" => "V²",
    "Bs^2" => "Bs²",
    "n^2" => "n²",
    "V*Bs" => "V Bs",
    "n*V" => "n V",
    "n*Bs" => "n Bs",
    "Pdyn*Bs" => "Pdyn Bs",
    "n*V*Bs" => "n V Bs",
    "sin(θ_c/2)" => "sin(θc/2)",
    "sin²(θ_c/2)" => "sin²(θc/2)",
    "sin⁴(θ_c/2)" => "sin⁴(θc/2)",
    "sin^(8/3)(θ_c/2)" => "sin^(8/3)(θc/2)",
    "V*sin²(θ_c/2)" => "V sin²(θc/2)",
    "Newell_d_Φ" => "dΦN/dt",
)

_display_terms(terms) = [get(_TERM_DISPLAY, term, term) for term in terms]

const _SCHEMAS = Dict(
    "real_sindy_discovery_coefficients.csv" => ["term", "coefficient"],
    "real_design_column_norms.csv" => [
        "basis", "term", "training_column_l2_norm", "selected", "coefficient",
        "clock_proxy", "clock_response",
    ],
    "may2024_reconstruction.csv" => [
        "storm_id", "catalog_row", "datetime", "time_hr", "dst_observed_nt",
        "dst_star_observed_nt", "dst_cleaned_nt", "dst_star_cleaned_nt",
        "dst_original_flag", "dst_star_original_target_flag", "dst_star_sindy_nt",
        "dst_star_burton_simplified_nt", "dst_star_burton_published_nt",
        "dst_star_obrien_nt", "v_kms", "bz_nt", "pdyn_npa",
    ],
    "primary_lambda_candidates.csv" => [
        "candidate_index", "lambda", "mean_storm_rmse_nt", "standard_error_nt",
        "n_active_terms", "eligible", "selected",
    ],
    "primary_lambda_decision.csv" => [
        "n_training_storms", "n_inner_training_storms",
        "n_inner_validation_storms", "minimum_candidate_index", "minimum_lambda",
        "minimum_mean_storm_rmse_nt", "minimum_standard_error_nt",
        "one_standard_error_cutoff_nt", "selected_candidate_index",
        "selected_lambda", "selection_rule",
    ],
    "real_ensemble_inclusion.csv" => [
        "term", "inclusion_probability", "nonzero_draws", "structural_zero_draws",
        "conditional_nonzero_median", "conditional_nonzero_empirical_q025",
        "conditional_nonzero_empirical_q975", "interval_kind", "confidence_interval",
        "subsample_without_replacement", "subsample_fraction", "lambda", "draws",
        "seed",
    ],
    "real_sindy_coefficients.csv" => [
        "term", "coefficient", "coefficient_kind", "inclusion",
    ],
    "synthetic_equation_recovery_trajectories.csv" => [
        "experiment", "objective", "seed", "global_row", "holdout_row",
        "time_hours", "dst_star_oracle_nt", "dst_star_simulated_nt", "error_nt",
        "is_anchor",
    ],
    "paired_sindy_vs_all_baselines_claim_sources.csv" => [
        "experiment", "comparison", "reference_model", "source_file",
        "source_experiment", "artifact_prefix", "n_storms",
        "mean_rmse_difference_nt", "rmse_ci_lower_nt", "rmse_ci_upper_nt",
        "mean_relative_difference_fraction", "relative_ci_lower_fraction",
        "relative_ci_upper_fraction", "interval_coverage", "bootstrap_draws", "seed",
        "in_predeclared_holm_family",
    ],
)

const _SYNTHETIC_SUMMARY_SCHEMA = [
    "experiment", "objective", "seed", "n_points", "dt_hours",
    "derivative_noise_std_nt_per_hour", "alpha_true", "tau_true",
    "true_decay_coefficient", "true_injection_coefficient", "library_terms",
    "active_terms", "active_term_names", "false_discoveries",
    "clock_false_discoveries", "support_precision", "support_recall",
    "decay_coefficient", "injection_coefficient", "decay_sign_ok",
    "injection_sign_ok", "decay_relative_error", "injection_relative_error",
    "heldout_forward_rmse_nt", "selected_lambda", "lambda_normalize",
    "lambda_protocol", "lambda_fit_rows", "lambda_validation_rows", "refit_rows",
    "heldout_rows", "validation_rmse_tolerance_nt_per_hour",
    "active_condition_number", "active_cancellation_ratio",
    "clock_block_condition_number", "clock_block_cancellation_ratio",
    "maximum_pair_correlation", "support_precision_min", "support_recall_min",
    "coefficient_relative_error_max", "heldout_forward_rmse_max_nt",
    "stress_condition_min", "stress_pair_correlation_min",
    "stress_clock_false_discoveries_min", "support_pass", "coefficient_pass",
    "forecast_pass", "recovery_pass", "stress_detected", "experiment_pass",
    "canonical_gate_applied", "outcome_label", "overall_validation_pass",
    "minimal_recovery_validation_pass", "full_stress_outcome_neutral",
]

function _require_paths(paths)
    all(hasproperty(paths, field) for field in (:root, :data, :figs, :mode, :explicit)) ||
        throw(ArgumentError("figure output paths are incomplete"))
    paths.explicit || error("canonical figure generation requires SOLARSINDY_OUTPUT_ROOT")
    mode = _provenance_mode(paths.mode)
    mode in (:canonical, :test) || error(
        "canonical figure generation requires canonical mode or explicit test mode",
    )
    return mode
end

function _json_has(value, key::AbstractString)
    haskey(value, key) || haskey(value, Symbol(key))
end

function _read_verified_csv(paths, filename::String, schema;
                            package_root::AbstractString)
    haskey(CANONICAL_DATA_ARTIFACT_INVENTORY, filename) || error(
        "figure input is not in the canonical data inventory: $filename",
    )
    path = joinpath(paths.data, filename)
    isfile(path) && !islink(path) || error(
        "canonical figure input must be a regular non-symlink file: $path",
    )
    record = verify_output_manifest(path;
        package_root,
        require_canonical=paths.mode == :canonical,
        verify_source=true,
    )
    String(_json_get(record, "run_mode")) == String(paths.mode) || error(
        "$filename manifest run mode differs from the figure run mode",
    )
    producer = String(_json_get(_json_get(record, "producer"), "path"))
    expected = CANONICAL_DATA_ARTIFACT_INVENTORY[filename]
    producer == expected || error(
        "$filename has producer $producer; expected $expected",
    )
    frame = CSV.read(path, DataFrame)
    schema !== nothing && names(frame) != schema && error(
        "$filename schema mismatch: expected $schema, got $(names(frame))",
    )
    metadata = _json_get(record, "metadata")
    _json_has(metadata, "rows") && Int(_json_get(metadata, "rows")) != nrow(frame) &&
        error("$filename manifest row count differs from the CSV")
    if _json_has(metadata, "columns")
        manifest_columns = String.(collect(_json_get(metadata, "columns")))
        manifest_columns == names(frame) || error(
            "$filename manifest schema differs from the CSV",
        )
    end
    return (; path=abspath(path), frame, record, sha256=provenance_sha256(path))
end

function _strings(frame, column::Symbol, label)
    values = String[]
    for value in frame[!, column]
        value isa AbstractString || error("$label $column must contain strings")
        text = String(value)
        isempty(strip(text)) && error("$label $column contains a blank value")
        push!(values, text)
    end
    return values
end

function _floats(frame, column::Symbol, label; allow_nan::Bool=false,
                 nonnegative::Bool=false)
    values = Float64[]
    for value in frame[!, column]
        value isa Real && !(value isa Bool) || error(
            "$label $column must contain real non-Bool values",
        )
        converted = Float64(value)
        (isfinite(converted) || (allow_nan && isnan(converted))) || error(
            "$label $column contains an invalid nonfinite value",
        )
        nonnegative && isfinite(converted) && converted < 0 && error(
            "$label $column must be nonnegative",
        )
        push!(values, converted)
    end
    return values
end

function _integers(frame, column::Symbol, label; nonnegative::Bool=false)
    values = Int[]
    for value in frame[!, column]
        value isa Integer && !(value isa Bool) || error(
            "$label $column must contain integers",
        )
        converted = Int(value)
        nonnegative && converted < 0 && error("$label $column must be nonnegative")
        push!(values, converted)
    end
    return values
end

function _bools(frame, column::Symbol, label)
    all(value -> value isa Bool, frame[!, column]) || error(
        "$label $column must contain Booleans",
    )
    return Bool.(frame[!, column])
end

function _validate_discovery(coefficients, norms, trajectory)
    terms = get_term_names(build_solar_wind_library(clock_basis=:full))
    nrow(coefficients) == 20 || error("discovery coefficients must contain 20 terms")
    _strings(coefficients, :term, "discovery coefficients") == terms || error(
        "discovery coefficient terms differ from the canonical full library",
    )
    point_coefficients = _floats(coefficients, :coefficient, "discovery coefficients")

    nrow(norms) == 35 || error("design column norms must contain 20 full and 15 collapsed rows")
    basis = _strings(norms, :basis, "design column norms")
    full = norms[basis .== "full", :]
    collapsed = norms[basis .== "collapsed", :]
    nrow(full) == 20 && nrow(collapsed) == 15 || error(
        "design column norms have the wrong basis row counts",
    )
    _strings(full, :term, "full design column norms") == terms || error(
        "full design-norm term order differs from the canonical library",
    )
    _floats(full, :training_column_l2_norm, "full design column norms";
            nonnegative=true)
    selected = _bools(full, :selected, "full design column norms")
    norm_coefficients = _floats(full, :coefficient, "full design column norms")
    norm_coefficients == point_coefficients || error(
        "design-norm and point-coefficient files disagree",
    )
    selected == .!iszero.(point_coefficients) || error(
        "persisted selected flags disagree with point coefficients",
    )
    count(selected) == 11 || error("canonical discovery support must contain 11 terms")

    nrow(trajectory) >= 8 || error("discovery trajectory has fewer than eight rows")
    time = _floats(trajectory, :time_hr, "May-2024 trajectory")
    all(diff(time) .> 0) || error("May-2024 trajectory time must be strictly increasing")
    observed = _floats(trajectory, :dst_star_observed_nt, "May-2024 trajectory";
                       allow_nan=true)
    count(isfinite, observed) >= 8 || error(
        "May-2024 trajectory has fewer than eight original Dst* targets",
    )
    original = _bools(trajectory, :dst_star_original_target_flag, "May-2024 trajectory")
    original == isfinite.(observed) || error(
        "May-2024 original-target flags disagree with persisted observations",
    )
    models = Dict{String,Vector{Float64}}()
    for (name, column) in (
        "SINDy" => :dst_star_sindy_nt,
        "Simplified Burton" => :dst_star_burton_simplified_nt,
        "Published Burton" => :dst_star_burton_published_nt,
        "O'Brien--McPherron" => :dst_star_obrien_nt,
    )
        models[name] = _floats(trajectory, column, "May-2024 trajectory")
    end
    isfinite(first(observed)) || error("May-2024 trajectory must begin at its observed anchor")
    all(first(prediction) == first(observed) for prediction in Base.values(models)) || error(
        "May-2024 comparator trajectories do not share the observed anchor",
    )
    return (; terms, selected, trajectory, time, observed, models)
end

function _validate_lambda(candidates, decision)
    nrow(candidates) == 60 || error("primary lambda selection must contain 60 candidates")
    nrow(decision) == 1 || error("primary lambda selection must contain one decision row")
    indices = _integers(candidates, :candidate_index, "lambda candidates";
                        nonnegative=true)
    indices == collect(1:60) || error("lambda candidate indices must be 1:60")
    lambdas = _floats(candidates, :lambda, "lambda candidates"; nonnegative=true)
    lambdas == storm_lambda_grid() || error("lambda candidates differ from the fixed grid")
    means = _floats(candidates, :mean_storm_rmse_nt, "lambda candidates";
                    nonnegative=true)
    errors = _floats(candidates, :standard_error_nt, "lambda candidates";
                     nonnegative=true)
    terms = _integers(candidates, :n_active_terms, "lambda candidates";
                      nonnegative=true)
    eligible = _bools(candidates, :eligible, "lambda candidates")
    selected = _bools(candidates, :selected, "lambda candidates")
    count(selected) == 1 || error("lambda candidates must mark exactly one selection")
    cutoff = only(_floats(decision, :one_standard_error_cutoff_nt, "lambda decision";
                          nonnegative=true))
    eligible == (means .<= cutoff) || error(
        "lambda eligibility flags disagree with the one-standard-error cutoff",
    )
    selected_index = only(_integers(decision, :selected_candidate_index,
                                    "lambda decision"; nonnegative=true))
    selected_index == only(findall(selected)) || error(
        "lambda decision and candidate selection disagree",
    )
    selected_index == findlast(eligible) || error(
        "selected lambda is not the largest eligible fixed-grid value",
    )
    only(_floats(decision, :selected_lambda, "lambda decision")) ==
        lambdas[selected_index] || error("selected lambda value is inconsistent")
    only(_strings(decision, :selection_rule, "lambda decision")) ==
        "largest_lambda_within_one_standard_error_then_fewer_terms_then_larger_lambda" ||
        error("lambda selection rule is not canonical")
    return (; candidates, lambdas, means, errors, terms, eligible, selected)
end

function _validate_stability(draws, summary, coefficients)
    terms = get_term_names(build_solar_wind_library(clock_basis=:full))
    names(draws) == terms || error("ensemble draw columns differ from the canonical library")
    nrow(draws) == 500 || error("coefficient stability requires exactly 500 raw draws")
    matrix = try
        Matrix{Float64}(draws)
    catch error_value
        error_value isa InterruptException && rethrow()
        error("ensemble draws must be representable as a Float64 matrix")
    end
    all(isfinite, matrix) || error("ensemble draws must be finite")
    nrow(summary) == 20 || error("coefficient summary must contain 20 terms")
    nrow(coefficients) == 20 || error("point-coefficient summary must contain 20 terms")
    _strings(summary, :term, "coefficient summary") == terms || error(
        "coefficient summary terms differ from raw draws",
    )
    _strings(coefficients, :term, "point coefficients") == terms || error(
        "point coefficient terms differ from raw draws",
    )
    inclusion = _floats(summary, :inclusion_probability, "coefficient summary")
    all((0 .<= inclusion) .& (inclusion .<= 1)) || error(
        "coefficient inclusion must lie in [0,1]",
    )
    nonzero = _integers(summary, :nonzero_draws, "coefficient summary";
                        nonnegative=true)
    zeros = _integers(summary, :structural_zero_draws, "coefficient summary";
                      nonnegative=true)
    all(nonzero .+ zeros .== 500) || error("coefficient draw counts do not total 500")
    _integers(summary, :draws, "coefficient summary") == fill(500, 20) || error(
        "coefficient summary does not record 500 draws",
    )
    _integers(summary, :seed, "coefficient summary") == fill(42, 20) || error(
        "coefficient summary does not use seed 42",
    )
    medians = _floats(summary, :conditional_nonzero_median, "coefficient summary";
                      allow_nan=true)
    lower = _floats(summary, :conditional_nonzero_empirical_q025,
                    "coefficient summary"; allow_nan=true)
    upper = _floats(summary, :conditional_nonzero_empirical_q975,
                    "coefficient summary"; allow_nan=true)
    for column in 1:20
        values = matrix[:, column]
        present = values[values .!= 0.0]
        nonzero[column] == length(present) || error(
            "coefficient summary nonzero count disagrees for $(terms[column])",
        )
        inclusion[column] == length(present) / 500 || error(
            "coefficient inclusion disagrees for $(terms[column])",
        )
        if isempty(present)
            all(isnan, (medians[column], lower[column], upper[column])) || error(
                "absent term $(terms[column]) must retain NaN conditional summaries",
            )
        else
            expected = (median(present), quantile(present, 0.025), quantile(present, 0.975))
            (medians[column], lower[column], upper[column]) == expected || error(
                "conditional coefficient summaries disagree for $(terms[column])",
            )
        end
    end
    coefficient_inclusion = _floats(coefficients, :inclusion, "point coefficients")
    coefficient_inclusion == inclusion || error(
        "point-coefficient and empirical inclusion summaries disagree",
    )
    point = _floats(coefficients, :coefficient, "point coefficients")
    return (; terms, inclusion, medians, lower, upper, point, summary)
end

function _validate_synthetic(summary, trajectories)
    nrow(summary) == 2 || error("synthetic recovery summary must contain two experiments")
    experiments = _strings(summary, :experiment, "synthetic recovery summary")
    experiments == ["minimal_identifiable", "full_canonical"] || error(
        "synthetic recovery experiments are not canonical",
    )
    _strings(summary, :objective, "synthetic recovery summary") == [
        "identifiable_recovery", "false_discovery_collinearity_stress",
    ] || error("synthetic recovery objectives are not canonical")
    _integers(summary, :seed, "synthetic recovery summary") == fill(31_415, 2) ||
        error("synthetic recovery does not use seed 31415")
    _integers(summary, :n_points, "synthetic recovery summary") == fill(801, 2) ||
        error("synthetic recovery does not use 801 generated points")
    _integers(summary, :refit_rows, "synthetic recovery summary") == fill(560, 2) ||
        error("synthetic recovery does not use the canonical refit split")
    heldout_counts = _integers(summary, :heldout_rows, "synthetic recovery summary")
    heldout_counts == fill(241, 2) || error(
        "synthetic recovery does not persist the canonical 241 held-out rows",
    )
    all(_bools(summary, :overall_validation_pass, "synthetic recovery summary")) ||
        error("synthetic minimal-library recovery gate did not pass")
    first(_bools(summary, :recovery_pass, "synthetic recovery summary")) ||
        error("identifiable minimal-library recovery did not pass")

    nrow(trajectories) == 482 || error(
        "synthetic recovery trajectories must contain 241 rows per experiment",
    )
    trajectory_experiments = _strings(trajectories, :experiment,
                                      "synthetic recovery trajectories")
    groups = Dict{String,DataFrame}()
    rmse_by_experiment = Dict{String,Float64}()
    for experiment in experiments
        frame = trajectories[trajectory_experiments .== experiment, :]
        nrow(frame) == 241 || error("$experiment must contain 241 held-out rows")
        _integers(frame, :holdout_row, "$experiment trajectory") == collect(1:241) ||
            error("$experiment holdout rows are not contiguous")
        _integers(frame, :global_row, "$experiment trajectory") == collect(561:801) ||
            error("$experiment global rows are not the canonical holdout")
        time = _floats(frame, :time_hours, "$experiment trajectory")
        all(diff(time) .> 0) || error("$experiment time is not strictly increasing")
        oracle = _floats(frame, :dst_star_oracle_nt, "$experiment trajectory")
        simulated = _floats(frame, :dst_star_simulated_nt, "$experiment trajectory")
        errors = _floats(frame, :error_nt, "$experiment trajectory")
        errors == simulated .- oracle || error("$experiment persisted errors are stale")
        anchors = _bools(frame, :is_anchor, "$experiment trajectory")
        findall(anchors) == [1] || error("$experiment must mark only its first row as anchor")
        first(errors) == 0.0 || error("$experiment anchor error must be zero")
        groups[experiment] = frame
        rmse_by_experiment[experiment] = sqrt(mean(abs2, errors))
    end
    first_group = groups[first(experiments)]
    second_group = groups[last(experiments)]
    first_group.time_hours == second_group.time_hours &&
        first_group.dst_star_oracle_nt == second_group.dst_star_oracle_nt || error(
            "synthetic experiments do not share the same held-out oracle",
        )
    reported_rmse = _floats(summary, :heldout_forward_rmse_nt,
                            "synthetic recovery summary"; nonnegative=true)
    reported_rmse == [rmse_by_experiment[experiment] for experiment in experiments] ||
        error("synthetic recovery summary RMSE disagrees with trajectory rows")
    return (; summary, experiments, groups)
end

function _validate_paired(frame)
    nrow(frame) == 12 || error("paired performance must contain exactly 12 comparisons")
    experiments = ("Validation_C24", "C20-22->C23", "even->odd", "C20-23->C25")
    references = ("Burton", "BurtonFull", "OBrienMcP")
    actual = Set(zip(_strings(frame, :experiment, "paired performance"),
                     _strings(frame, :reference_model, "paired performance")))
    actual == Set(Iterators.product(experiments, references)) || error(
        "paired performance does not contain the four-by-three predeclared family",
    )
    estimates = _floats(frame, :mean_rmse_difference_nt, "paired performance")
    lower = _floats(frame, :rmse_ci_lower_nt, "paired performance")
    upper = _floats(frame, :rmse_ci_upper_nt, "paired performance")
    all(lower .<= estimates .<= upper) || error(
        "paired performance estimate lies outside its percentile interval",
    )
    all(_integers(frame, :n_storms, "paired performance"; nonnegative=true) .>= 2) ||
        error("paired performance requires at least two storms per comparison")
    _integers(frame, :bootstrap_draws, "paired performance") == fill(10_000, 12) ||
        error("paired performance does not use 10000 bootstrap draws")
    _integers(frame, :seed, "paired performance") == fill(42, 12) ||
        error("paired performance does not use seed 42")
    _floats(frame, :interval_coverage, "paired performance") == fill(0.95, 12) ||
        error("paired performance does not use 95% intervals")
    return (; frame, experiments, references, estimates, lower, upper)
end

function prepare_canonical_figure_inputs(paths=validation_output_paths();
                                         package_root::AbstractString=_FIGURE_PACKAGE_ROOT)
    mode = _require_paths(paths)
    normalized_paths = merge(paths, (; mode))
    read_input(name, schema=get(_SCHEMAS, name, nothing)) =
        _read_verified_csv(normalized_paths, name, schema; package_root)

    discovery_coefficients = read_input("real_sindy_discovery_coefficients.csv")
    discovery_norms = read_input("real_design_column_norms.csv")
    discovery_trajectory = read_input("may2024_reconstruction.csv")
    discovery = _validate_discovery(
        discovery_coefficients.frame, discovery_norms.frame,
        discovery_trajectory.frame,
    )

    lambda_paths = SolarSINDy._storm_selection_paths(
        normalized_paths.data, "primary_lambda",
    )
    lambda_candidates, lambda_decision =
        SolarSINDy._with_selection_csv_set_lock(lambda_paths) do
            (
                read_input("primary_lambda_candidates.csv"),
                read_input("primary_lambda_decision.csv"),
            )
        end
    lambda = _validate_lambda(lambda_candidates.frame, lambda_decision.frame)

    raw_draws = read_input("real_sindy_ensemble_draws.csv", nothing)
    stability_summary = read_input("real_ensemble_inclusion.csv")
    stability_coefficients = read_input("real_sindy_coefficients.csv")
    stability = _validate_stability(
        raw_draws.frame, stability_summary.frame, stability_coefficients.frame,
    )

    synthetic_summary = read_input(
        "synthetic_equation_recovery_summary.csv", _SYNTHETIC_SUMMARY_SCHEMA,
    )
    synthetic_trajectories = read_input("synthetic_equation_recovery_trajectories.csv")
    synthetic = _validate_synthetic(
        synthetic_summary.frame, synthetic_trajectories.frame,
    )

    paired_source = read_input("paired_sindy_vs_all_baselines_claim_sources.csv")
    paired = _validate_paired(paired_source.frame)

    return (
        paths=normalized_paths,
        package_root=abspath(package_root),
        discovery=merge(discovery, (inputs=Dict(
            "point_coefficients" => discovery_coefficients.path,
            "design_column_norms" => discovery_norms.path,
            "outer_trajectory" => discovery_trajectory.path,
        ), input_hashes=Dict(
            "point_coefficients" => discovery_coefficients.sha256,
            "design_column_norms" => discovery_norms.sha256,
            "outer_trajectory" => discovery_trajectory.sha256,
        ),)),
        lambda=merge(lambda, (inputs=Dict(
            "candidate_grid" => lambda_candidates.path,
            "selection_decision" => lambda_decision.path,
        ), input_hashes=Dict(
            "candidate_grid" => lambda_candidates.sha256,
            "selection_decision" => lambda_decision.sha256,
        ),)),
        stability=merge(stability, (inputs=Dict(
            "raw_joint_draws" => raw_draws.path,
            "conditional_summary" => stability_summary.path,
            "point_coefficients" => stability_coefficients.path,
        ), input_hashes=Dict(
            "raw_joint_draws" => raw_draws.sha256,
            "conditional_summary" => stability_summary.sha256,
            "point_coefficients" => stability_coefficients.sha256,
        ),)),
        synthetic=merge(synthetic, (inputs=Dict(
            "recovery_summary" => synthetic_summary.path,
            "heldout_trajectories" => synthetic_trajectories.path,
        ), input_hashes=Dict(
            "recovery_summary" => synthetic_summary.sha256,
            "heldout_trajectories" => synthetic_trajectories.sha256,
        ),)),
        paired=merge(paired, (
            inputs=Dict("paired_effect_summary" => paired_source.path),
            input_hashes=Dict("paired_effect_summary" => paired_source.sha256),
        )),
    )
end

function _figure_canvas(rows, columns; height, horizontal_spacing=0.12,
                        vertical_spacing=0.14, per_subplot_legends=true)
    figure = subplots(rows, columns;
        sync=false, show=false, width=_FIGURE_WIDTH, height=height, title="",
        per_subplot_legends, horizontal_spacing, vertical_spacing,
    )
    return figure
end

function _submitted_style!(figure; height, top_margin=25)
    plot = _plot_object(figure)
    base_font = attr(
        family=_FONT_FAMILY, size=_BASE_FONT_SIZE, color=_TEXT,
    )
    title_font = attr(
        family=_FONT_FAMILY, size=_AXIS_TITLE_FONT_SIZE, color=_TEXT,
    )
    relayout!(plot;
        width=_FIGURE_WIDTH, height, font=base_font,
        margin=attr(l=53, r=20, t=top_margin, b=25, autoexpand=true),
    )
    update_xaxes!(plot;
        tickfont=base_font, title_font, automargin=true,
    )
    update_yaxes!(plot;
        tickfont=base_font, title_font, automargin=true,
    )
    for (key, value) in plot.layout.fields
        startswith(String(key), "legend") || continue
        if value isa AbstractDict
            value[:font] = base_font
        else
            value.fields[:font] = base_font
        end
    end
    return figure
end

_attribute_fields(value) = value isa AbstractDict ? value : value.fields

function _horizontal_outside_top_legend!(figure)
    set_legend!(figure; position=:outside_top)
    legend = _plot_object(figure).layout.fields[:legend]
    legend_fields = _attribute_fields(legend)
    legend_fields[:orientation] = "h"
    return figure
end

function _build_discovery_figure(data)
    selected_indices = findall(data.selected)
    isempty(selected_indices) && error("discovery support figure requires selected terms")
    selected_terms = reverse(_display_terms(data.terms[selected_indices]))
    term_positions = collect(1:length(selected_terms))
    figure = _figure_canvas(1, 2; height=_DISCOVERY_FIGURE_HEIGHT,
                            per_subplot_legends=false)
    plot_bar!(figure, ones(Float64, length(selected_terms)), term_positions;
        row=1, col=1, orientation="h", color=_BLUE, showlegend=false,
        xlabel="Selected in full refit", ylabel="Candidate term")
    xrange!(figure, [-0.05, 1.05]; row=1, col=1)
    support_axis = figure.fig.layout.fields[:yaxis]
    support_fields = _attribute_fields(support_axis)
    support_fields[:tickmode] = "array"
    support_fields[:tickvals] = term_positions
    support_fields[:ticktext] = selected_terms
    support_fields[:range] = [0.5, length(selected_terms) + 0.5]

    plot_scatter!(figure, data.time, data.observed;
        row=1, col=2, mode="lines", color=_BLACK, dash="solid", linewidth=2.4,
        legend="Observed Dst*")
    styles = (
        ("SINDy", "SINDy", _BLUE, "solid"),
        ("Simplified Burton", "Simplified Burton", _ORANGE, "dash"),
        ("Published Burton", "Published Burton", _PINK, "dot"),
        ("O'Brien--McPherron", "O'Brien-McPherron", _GREEN, "dashdot"),
    )
    for (data_name, display_name, color, dash) in styles
        plot_scatter!(figure, data.time, data.models[data_name];
            row=1, col=2, mode="lines", color, dash, linewidth=1.8,
            legend=display_name)
    end
    xlabel!(figure, "Time from shared anchor [h]"; row=1, col=2)
    ylabel!(figure, "Dst* [nT]"; row=1, col=2)
    _horizontal_outside_top_legend!(figure)
    return _submitted_style!(figure; height=_DISCOVERY_FIGURE_HEIGHT, top_margin=90)
end

function _build_lambda_figure(data)
    figure = _figure_canvas(2, 1; height=_LAMBDA_FIGURE_HEIGHT)
    plot_scatter!(figure, data.lambdas, data.means;
        row=1, col=1, mode="lines+markers", color=_BLUE, dash="solid", linewidth=1.8,
        marker_size=5, marker_symbol="circle", legend="Mean validation RMSE",
        error_y=data.errors, xscale="log")
    plot_scatter!(figure, data.lambdas[data.eligible], data.means[data.eligible];
        row=1, col=1, mode="markers", color=_GREEN, marker_size=7,
        marker_symbol="square", legend="Within one SE", xscale="log")
    plot_scatter!(figure, data.lambdas[data.selected], data.means[data.selected];
        row=1, col=1, mode="markers", color=_ORANGE, marker_size=10,
        marker_symbol="diamond", legend="Selected", xscale="log")
    xlabel!(figure, "Regularisation λ"; row=1, col=1)
    ylabel!(figure, "Mean storm RMSE [nT]"; row=1, col=1)

    plot_scatter!(figure, data.lambdas, Float64.(data.terms);
        row=2, col=1, mode="lines+markers", color=_BLUE, dash="solid", linewidth=1.8,
        marker_size=5, marker_symbol="circle", legend="Active terms", xscale="log")
    plot_scatter!(figure, data.lambdas[data.selected], Float64.(data.terms[data.selected]);
        row=2, col=1, mode="markers", color=_ORANGE, marker_size=10,
        marker_symbol="diamond", legend="Selected", xscale="log")
    xlabel!(figure, "Regularisation λ"; row=2, col=1)
    ylabel!(figure, "Active terms [count]"; row=2, col=1)
    for key in (:xaxis, :xaxis2)
        axis = _plot_object(figure).layout.fields[key]
        axis.fields[:tickmode] = "array"
        axis.fields[:tickvals] = _LAMBDA_TICK_VALUES
        axis.fields[:ticktext] = _LAMBDA_TICK_LABELS
    end
    # Left placement uses the flat low-λ region and avoids the selection transition.
    subplot_legends!(figure; position=:left)
    return _submitted_style!(figure; height=_LAMBDA_FIGURE_HEIGHT)
end

function _interval_segments(labels, lower, upper)
    x = Float64[]
    y = eltype(labels)[]
    for index in eachindex(labels)
        isfinite(lower[index]) && isfinite(upper[index]) || continue
        append!(x, (lower[index], upper[index], NaN))
        append!(y, (labels[index], labels[index], labels[index]))
    end
    return x, y
end

function _build_stability_figure(data)
    all_terms = reverse(_display_terms(data.terms))
    all_positions = collect(1:length(all_terms))
    figure = _figure_canvas(1, 2; height=_STABILITY_FIGURE_HEIGHT,
                            per_subplot_legends=false)
    plot_bar!(figure, reverse(data.inclusion), all_positions;
        row=1, col=1, orientation="h", color=_BLUE, showlegend=false,
        xlabel="Inclusion frequency", ylabel="Candidate term")
    xrange!(figure, [0.0, 1.0]; row=1, col=1)
    left_axis = figure.fig.layout.fields[:yaxis]
    left_fields = _attribute_fields(left_axis)
    left_fields[:tickmode] = "array"
    left_fields[:tickvals] = all_positions
    left_fields[:ticktext] = all_terms
    left_fields[:range] = [0.5, length(all_terms) + 0.5]

    has_scale = isfinite.(data.medians) .& .!iszero.(data.medians)
    valid_indices = findall(has_scale)
    scales = abs.(data.medians[has_scale])
    term_positions = Float64.(length(data.terms) .- reverse(valid_indices) .+ 1)
    lower = reverse(data.lower[has_scale] ./ scales)
    upper = reverse(data.upper[has_scale] ./ scales)
    medians = reverse(data.medians[has_scale] ./ scales)
    point = reverse(data.point[has_scale] ./ scales)
    interval_x, interval_y = _interval_segments(term_positions, lower, upper)
    plot_scatter!(figure, interval_x, interval_y;
        row=1, col=2, mode="lines", color=_GREY, dash="solid", linewidth=1.6,
        legend="2.5% to 97.5% range")
    plot_scatter!(figure, medians, term_positions;
        row=1, col=2, mode="markers", color=_BLUE, marker_size=7,
        marker_symbol="circle", legend="Conditional median")
    plot_scatter!(figure, point, term_positions;
        row=1, col=2, mode="markers", color=_ORANGE, marker_size=7,
        marker_symbol="diamond", legend="Full refit")
    xlabel!(figure, "Normalized coefficient"; row=1, col=2)
    right_axis = figure.fig.layout.fields[:yaxis2]
    right_fields = _attribute_fields(right_axis)
    right_fields[:tickmode] = "array"
    right_fields[:tickvals] = all_positions
    right_fields[:ticktext] = all_terms
    right_fields[:range] = [0.5, length(all_terms) + 0.5]
    right_fields[:showticklabels] = false
    _horizontal_outside_top_legend!(figure)
    return _submitted_style!(figure; height=_STABILITY_FIGURE_HEIGHT, top_margin=70)
end

function _build_synthetic_figure(data)
    figure = _figure_canvas(2, 1; height=_SYNTHETIC_FIGURE_HEIGHT,
                            per_subplot_legends=false)
    minimal = data.groups["minimal_identifiable"]
    full = data.groups["full_canonical"]
    time = Float64.(minimal.time_hours)
    oracle = Float64.(minimal.dst_star_oracle_nt)
    plot_scatter!(figure, time, oracle;
        row=1, col=1, mode="lines", color=_BLACK, dash="solid", linewidth=2.4,
        legend="Oracle")
    plot_scatter!(figure, time, Float64.(minimal.dst_star_simulated_nt);
        row=1, col=1, mode="lines", color=_BLUE, dash="dash", linewidth=1.8,
        legend="Minimal library")
    plot_scatter!(figure, time, Float64.(full.dst_star_simulated_nt);
        row=1, col=1, mode="lines", color=_ORANGE, dash="dot", linewidth=1.8,
        legend="Full library stress")
    ylabel!(figure, "Held-out Dst* [nT]"; row=1, col=1)

    plot_scatter!(figure, time, Float64.(minimal.error_nt);
        row=2, col=1, mode="lines", color=_BLUE, dash="dash", linewidth=1.8,
        legend="Minimal library", showlegend=false)
    plot_scatter!(figure, time, Float64.(full.error_nt);
        row=2, col=1, mode="lines", color=_ORANGE, dash="dot", linewidth=1.8,
        legend="Full library stress", showlegend=false)
    xlabel!(figure, "Time [h]"; row=2, col=1)
    ylabel!(figure, "Simulation error [nT]"; row=2, col=1)
    _horizontal_outside_top_legend!(figure)
    return _submitted_style!(figure; height=_SYNTHETIC_FIGURE_HEIGHT, top_margin=70)
end

function _paired_label(experiment, reference)
    experiment_label = Dict(
        "Validation_C24" => "C24",
        "C20-22->C23" => "C20-22 to C23",
        "even->odd" => "Even to odd",
        "C20-23->C25" => "C20-23 to C25",
    )[experiment]
    reference_label = Dict(
        "Burton" => "simplified Burton",
        "BurtonFull" => "published Burton",
        "OBrienMcP" => "O'Brien-McPherron",
    )[reference]
    return "$experiment_label | $reference_label"
end

function _build_paired_figure(data)
    frame = data.frame
    experiment = _strings(frame, :experiment, "paired performance")
    reference = _strings(frame, :reference_model, "paired performance")
    labels = [_paired_label(experiment[index], reference[index]) for index in 1:nrow(frame)]
    labels = reverse(labels)
    lower = reverse(data.lower)
    upper = reverse(data.upper)
    estimates = reverse(data.estimates)
    references = reverse(reference)
    interval_x, interval_y = _interval_segments(labels, lower, upper)
    figure = plot_scatter(interval_x, interval_y;
        mode="lines", color=_GREY, dash="solid", linewidth=1.6,
        legend="95% paired interval", xlabel="Mean paired RMSE difference [nT]",
        ylabel="Experiment | reference", width=_FIGURE_WIDTH,
        height=_PAIRED_FIGURE_HEIGHT,
        show=false)
    styles = Dict(
        "Burton" => (_ORANGE, "circle", "Simplified Burton"),
        "BurtonFull" => (_PINK, "square", "Published Burton"),
        "OBrienMcP" => (_GREEN, "diamond", "O'Brien-McPherron"),
    )
    for key in ("Burton", "BurtonFull", "OBrienMcP")
        mask = references .== key
        color, symbol, legend = styles[key]
        plot_scatter!(figure, estimates[mask], labels[mask];
            mode="markers", color, marker_size=8, marker_symbol=symbol, legend)
    end
    update_xaxes!(figure; zeroline=true, zerolinecolor=_BLACK, zerolinewidth=1.2)
    relayout!(figure; showlegend=false)
    return _submitted_style!(figure; height=_PAIRED_FIGURE_HEIGHT)
end

function build_canonical_figures(prepared)
    return (
        discovery_validation=_build_discovery_figure(prepared.discovery),
        lambda_selection=_build_lambda_figure(prepared.lambda),
        coefficient_stability=_build_stability_figure(prepared.stability),
        synthetic_recovery=_build_synthetic_figure(prepared.synthetic),
        paired_performance=_build_paired_figure(prepared.paired),
    )
end

_plot_object(figure::SubplotFigure) = figure.fig
_plot_object(figure) = figure

function _assert_inputs_unchanged(inputs, hashes)
    for (name, path) in inputs
        provenance_sha256(path) == hashes[name] || error(
            "figure input changed after verified loading: $path",
        )
    end
    return nothing
end

function _write_figure(prepared, filename, figure, inputs, input_hashes; height, role)
    haskey(CANONICAL_FIGURE_ARTIFACT_INVENTORY, filename) || error(
        "figure is not in the locked canonical inventory: $filename",
    )
    Set(keys(inputs)) == Set(keys(input_hashes)) || error(
        "figure input paths and verified hashes do not have identical names",
    )
    output = joinpath(prepared.paths.figs, filename)
    _assert_inputs_unchanged(inputs, input_hashes)
    render! = function (staged_path)
        _assert_inputs_unchanged(inputs, input_hashes)
        PlotlySupply.savefig(_plot_object(figure), staged_path;
                             width=_FIGURE_WIDTH, height=height)
        _assert_inputs_unchanged(inputs, input_hashes)
    end
    write_manifested_figure(output, render!;
        producer_script=@__FILE__,
        input_paths=inputs,
        selection_record=(
            kind="canonical_csv_only_figure",
            figure=filename,
            role,
            rendering="PlotlySupply_pdf",
        ),
        deterministic=true,
        backend="PlotlySupply.jl",
        metadata=(width_px=_FIGURE_WIDTH, height_px=height, role),
        package_root=prepared.package_root,
        mode=prepared.paths.mode,
        verify_source=true,
    )
    return output
end

function run_canonical_figure_generation(paths=validation_output_paths();
                                         package_root::AbstractString=_FIGURE_PACKAGE_ROOT,
                                         _figure_writer::Function=_write_figure,
                                         _after_figure_hook::Function=(name, path) -> nothing)
    prepared = prepare_canonical_figure_inputs(paths; package_root)
    figures = build_canonical_figures(prepared)
    specs = (
        (name="fig_discovery_validation.pdf", figure=figures.discovery_validation,
         inputs=prepared.discovery.inputs, input_hashes=prepared.discovery.input_hashes,
         height=_DISCOVERY_FIGURE_HEIGHT,
         role="required_domain_discovery_and_untouched_validation"),
        (name="fig_lambda_selection.pdf", figure=figures.lambda_selection,
         inputs=prepared.lambda.inputs, input_hashes=prepared.lambda.input_hashes,
         height=_LAMBDA_FIGURE_HEIGHT,
         role="fixed_60_point_one_standard_error_selection"),
        (name="fig_coefficient_stability.pdf", figure=figures.coefficient_stability,
         inputs=prepared.stability.inputs, input_hashes=prepared.stability.input_hashes,
         height=_STABILITY_FIGURE_HEIGHT,
         role="500_draw_full_library_coefficient_stability"),
        (name="fig_synthetic_recovery.pdf", figure=figures.synthetic_recovery,
         inputs=prepared.synthetic.inputs, input_hashes=prepared.synthetic.input_hashes,
         height=_SYNTHETIC_FIGURE_HEIGHT,
         role="identifiable_recovery_and_full_library_stress"),
        (name="fig_paired_performance.pdf", figure=figures.paired_performance,
         inputs=prepared.paired.inputs, input_hashes=prepared.paired.input_hashes,
         height=_PAIRED_FIGURE_HEIGHT,
         role="twelve_paired_whole_storm_effect_intervals"),
    )
    expected_outputs = [joinpath(prepared.paths.figs, spec.name) for spec in specs]
    snapshot = SolarSINDy._snapshot_regular_file_set(vcat(
        expected_outputs, [path * ".manifest.json" for path in expected_outputs],
    ))
    outputs = String[]
    try
        for spec in specs
            output = _figure_writer(
                prepared, spec.name, spec.figure, spec.inputs, spec.input_hashes;
                height=spec.height, role=spec.role,
            )
            push!(outputs, output)
            _after_figure_hook(spec.name, output)
        end
        Set(basename.(outputs)) == Set(keys(CANONICAL_FIGURE_ARTIFACT_INVENTORY)) ||
            error("canonical figure generation did not produce the locked five-file inventory")
        for path in outputs
            verify_output_manifest(path;
                package_root=prepared.package_root,
                require_canonical=prepared.paths.mode == :canonical,
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

function main()
    outputs = run_canonical_figure_generation()
    println("Generated and verified $(length(outputs)) canonical PlotlySupply figures")
    return outputs
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    CanonicalFigureGeneration.main()
end
