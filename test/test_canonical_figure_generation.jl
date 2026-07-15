using Test
using SolarSINDy
using CSV
using DataFrames
using Dates
using JSON3
using Statistics

if !isdefined(Main, :CanonicalFigureGeneration)
    include(joinpath(@__DIR__, "..", "validation", "canonical_figure_generation.jl"))
end
const CFG = Main.CanonicalFigureGeneration
const _FIGURE_TEST_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))

function _figure_fixture_write(paths, filename, frame;
                               producer=CFG.CANONICAL_DATA_ARTIFACT_INVENTORY[filename])
    path = joinpath(paths.data, filename)
    CFG.write_manifested_csv(path, frame;
        producer_script=producer,
        input_paths=(;),
        selection_record=(kind="canonical_figure_schema_fixture", artifact=filename),
        deterministic=true,
        package_root=_FIGURE_TEST_PACKAGE_ROOT,
        mode=:test,
    )
    return path
end

function _synthetic_summary_fixture(experiment, objective, rmse; minimal)
    return (
        experiment, objective, seed=31_415, n_points=801, dt_hours=0.25,
        derivative_noise_std_nt_per_hour=0.25, alpha_true=5.4e-3, tau_true=7.7,
        true_decay_coefficient=-1 / 7.7, true_injection_coefficient=-5.4e-3,
        library_terms=minimal ? 3 : 20, active_terms=minimal ? 2 : 3,
        active_term_names=minimal ? "Dst_star;V*Bs" : "Dst_star;V*Bs;Pdyn",
        false_discoveries=minimal ? 0 : 1, clock_false_discoveries=minimal ? 0 : 1,
        support_precision=minimal ? 1.0 : 2 / 3, support_recall=1.0,
        decay_coefficient=-1 / 7.7, injection_coefficient=-5.4e-3,
        decay_sign_ok=true, injection_sign_ok=true,
        decay_relative_error=0.0, injection_relative_error=0.0,
        heldout_forward_rmse_nt=rmse, selected_lambda=0.1, lambda_normalize=true,
        lambda_protocol="chronological fixture matching canonical schema",
        lambda_fit_rows=420, lambda_validation_rows=140, refit_rows=560,
        heldout_rows=241, validation_rmse_tolerance_nt_per_hour=0.01,
        active_condition_number=minimal ? 2.0 : 200.0,
        active_cancellation_ratio=minimal ? 1.0 : 20.0,
        clock_block_condition_number=minimal ? 1.0 : 300.0,
        clock_block_cancellation_ratio=minimal ? 1.0 : 30.0,
        maximum_pair_correlation=minimal ? 0.2 : 0.99,
        support_precision_min=1.0, support_recall_min=1.0,
        coefficient_relative_error_max=0.05, heldout_forward_rmse_max_nt=0.5,
        stress_condition_min=100.0, stress_pair_correlation_min=0.95,
        stress_clock_false_discoveries_min=1, support_pass=minimal,
        coefficient_pass=true, forecast_pass=rmse <= 0.5,
        recovery_pass=minimal, stress_detected=!minimal, experiment_pass=true,
        canonical_gate_applied=minimal,
        outcome_label=minimal ? "recovery_pass" : "stress_detected",
        overall_validation_pass=true, minimal_recovery_validation_pass=true,
        full_stress_outcome_neutral=true,
    )
end

function _canonical_figure_fixture(root)
    data = joinpath(root, "data")
    figs = joinpath(root, "figs")
    mkpath(data)
    mkpath(figs)
    paths = (; root, data, figs, mode=:test, explicit=true)

    terms = get_term_names(build_solar_wind_library(clock_basis=:full))
    collapsed_terms = get_term_names(build_solar_wind_library(clock_basis=:collapsed))
    coefficients = zeros(20)
    coefficients[[2, 4, 6, 8, 10, 11, 12, 14, 16, 18, 20]] .= 0.01
    coefficients[2] = -0.13
    coefficients[11] = -5.4e-3
    _figure_fixture_write(paths, "real_sindy_discovery_coefficients.csv",
        DataFrame(term=terms, coefficient=coefficients))

    norms = DataFrame(
        basis=String[], term=String[], training_column_l2_norm=Float64[],
        selected=Bool[], coefficient=Float64[], clock_proxy=Bool[],
        clock_response=Bool[],
    )
    for (basis, names, values) in (
        ("full", terms, coefficients), ("collapsed", collapsed_terms, zeros(15)),
    ), index in eachindex(names)
        push!(norms, (
            basis, names[index], Float64(index), values[index] != 0.0, values[index],
            occursin("sin", names[index]),
            occursin("sin", names[index]) || names[index] == "Newell_d_Φ",
        ))
    end
    _figure_fixture_write(paths, "real_design_column_norms.csv", norms)

    n = 24
    time = collect(0.0:(n - 1))
    observed = -10.0 .- 1.2 .* time
    sindy = observed .+ 0.3 .* sin.(time)
    burton = observed .+ 1.0 .* sin.(time ./ 3)
    published = observed .+ 0.8 .* sin.(time ./ 4)
    obrien = observed .+ 0.5 .* sin.(time ./ 5)
    for prediction in (sindy, burton, published, obrien)
        prediction[1] = observed[1]
    end
    may = DataFrame(
        storm_id=fill(1, n), catalog_row=collect(1001:(1000 + n)),
        datetime=[DateTime(2024, 5, 10) + Hour(i - 1) for i in 1:n],
        time_hr=time, dst_observed_nt=observed, dst_star_observed_nt=observed,
        dst_cleaned_nt=observed, dst_star_cleaned_nt=observed,
        dst_original_flag=fill(true, n), dst_star_original_target_flag=fill(true, n),
        dst_star_sindy_nt=sindy, dst_star_burton_simplified_nt=burton,
        dst_star_burton_published_nt=published, dst_star_obrien_nt=obrien,
        v_kms=fill(500.0, n), bz_nt=fill(-5.0, n), pdyn_npa=fill(2.0, n),
    )
    _figure_fixture_write(paths, "may2024_reconstruction.csv", may)

    lambdas = storm_lambda_grid()
    means = 10.0 .+ abs.(collect(1:60) .- 20)
    standard_errors = fill(2.0, 60)
    cutoff = 12.0
    eligible = means .<= cutoff
    selected = falses(60)
    selected[findlast(eligible)] = true
    candidate_frame = DataFrame(
        candidate_index=collect(1:60), lambda=lambdas,
        mean_storm_rmse_nt=means, standard_error_nt=standard_errors,
        n_active_terms=max.(0, 20 .- fld.((collect(1:60) .- 1), 3)),
        eligible=eligible, selected=selected,
    )
    _figure_fixture_write(paths, "primary_lambda_candidates.csv", candidate_frame)
    selected_index = only(findall(selected))
    decision = DataFrame([(
        n_training_storms=100, n_inner_training_storms=80,
        n_inner_validation_storms=20, minimum_candidate_index=20,
        minimum_lambda=lambdas[20], minimum_mean_storm_rmse_nt=means[20],
        minimum_standard_error_nt=standard_errors[20],
        one_standard_error_cutoff_nt=cutoff,
        selected_candidate_index=selected_index, selected_lambda=lambdas[selected_index],
        selection_rule="largest_lambda_within_one_standard_error_then_fewer_terms_then_larger_lambda",
    )])
    _figure_fixture_write(paths, "primary_lambda_decision.csv", decision)

    draw_matrix = zeros(500, 20)
    draw_matrix[:, 2] .= -0.13
    draw_matrix[1:475, 11] .= -5.4e-3 .+ 1e-4 .* sin.(1:475)
    draws = DataFrame(draw_matrix, Symbol.(terms))
    _figure_fixture_write(paths, "real_sindy_ensemble_draws.csv", draws)
    summary_rows = NamedTuple[]
    inclusion = Float64[]
    for index in 1:20
        values = draw_matrix[:, index]
        present = values[values .!= 0.0]
        probability = length(present) / 500
        push!(inclusion, probability)
        push!(summary_rows, (
            term=terms[index], inclusion_probability=probability,
            nonzero_draws=length(present), structural_zero_draws=count(iszero, values),
            conditional_nonzero_median=isempty(present) ? NaN : median(present),
            conditional_nonzero_empirical_q025=isempty(present) ? NaN : quantile(present, 0.025),
            conditional_nonzero_empirical_q975=isempty(present) ? NaN : quantile(present, 0.975),
            interval_kind="conditional_nonzero_empirical_row_subsample_interval",
            confidence_interval=false, subsample_without_replacement=true,
            subsample_fraction=0.8, lambda=lambdas[selected_index], draws=500, seed=42,
        ))
    end
    _figure_fixture_write(paths, "real_ensemble_inclusion.csv", DataFrame(summary_rows))
    stability_coefficients = zeros(20)
    stability_coefficients[2] = coefficients[2]
    stability_coefficients[11] = coefficients[11]
    _figure_fixture_write(paths, "real_sindy_coefficients.csv", DataFrame(
        term=terms, coefficient=stability_coefficients,
        coefficient_kind=fill("selected_full_refit_point_coefficient", 20),
        inclusion=inclusion,
    ))

    synthetic_time = collect(0.0:0.25:60.0)
    oracle = -20.0 .- 0.2 .* synthetic_time
    minimal_error = 0.1 .* sin.(synthetic_time)
    full_error = 0.4 .* sin.(synthetic_time)
    trajectory_rows = NamedTuple[]
    synthetic_rmse = Dict{String,Float64}()
    for (experiment, objective, error_values) in (
        ("minimal_identifiable", "identifiable_recovery", minimal_error),
        ("full_canonical", "false_discovery_collinearity_stress", full_error),
    )
        simulated = oracle .+ error_values
        persisted_errors = simulated .- oracle
        synthetic_rmse[experiment] = sqrt(mean(abs2, persisted_errors))
        for holdout_row in 1:241
            push!(trajectory_rows, (
                experiment, objective, seed=31_415,
                global_row=560 + holdout_row, holdout_row,
                time_hours=synthetic_time[holdout_row],
                dst_star_oracle_nt=oracle[holdout_row],
                dst_star_simulated_nt=simulated[holdout_row],
                error_nt=persisted_errors[holdout_row],
                is_anchor=holdout_row == 1,
            ))
        end
    end
    synthetic_summary = DataFrame([
        _synthetic_summary_fixture(
            "minimal_identifiable", "identifiable_recovery",
            synthetic_rmse["minimal_identifiable"]; minimal=true,
        ),
        _synthetic_summary_fixture(
            "full_canonical", "false_discovery_collinearity_stress",
            synthetic_rmse["full_canonical"]; minimal=false,
        ),
    ])
    _figure_fixture_write(paths, "synthetic_equation_recovery_summary.csv",
                          synthetic_summary)
    _figure_fixture_write(paths, "synthetic_equation_recovery_trajectories.csv",
                          DataFrame(trajectory_rows))

    experiments = ("Validation_C24", "C20-22->C23", "even->odd", "C20-23->C25")
    references = ("Burton", "BurtonFull", "OBrienMcP")
    paired_rows = NamedTuple[]
    index = 0
    for experiment in experiments, reference in references
        index += 1
        estimate = -1.0 + 0.1 * index
        push!(paired_rows, (
            experiment, comparison="SINDy_vs_$reference", reference_model=reference,
            source_file=experiment == "Validation_C24" ? "real_holdout_metrics.csv" :
                "cross_cycle_metrics.csv",
            source_experiment=experiment, artifact_prefix="fixture_$index",
            n_storms=20, mean_rmse_difference_nt=estimate,
            rmse_ci_lower_nt=estimate - 0.5, rmse_ci_upper_nt=estimate + 0.7,
            mean_relative_difference_fraction=estimate / 10,
            relative_ci_lower_fraction=(estimate - 0.5) / 10,
            relative_ci_upper_fraction=(estimate + 0.7) / 10,
            interval_coverage=0.95, bootstrap_draws=10_000, seed=42,
            in_predeclared_holm_family=reference == "OBrienMcP",
        ))
    end
    _figure_fixture_write(paths, "paired_sindy_vs_all_baselines_claim_sources.csv",
                          DataFrame(paired_rows))
    return paths
end

@testset "Canonical figure producer verifies provenance, schemas, and density" begin
    mktempdir() do root
        paths = _canonical_figure_fixture(root)
        prepared = CFG.prepare_canonical_figure_inputs(paths;
            package_root=_FIGURE_TEST_PACKAGE_ROOT)
        @test nrow(prepared.discovery.trajectory) == 24
        @test length(prepared.lambda.lambdas) == 60
        @test length(prepared.stability.terms) == 20
        @test nrow(prepared.synthetic.groups["minimal_identifiable"]) == 241
        @test nrow(prepared.paired.frame) == 12
        @test Set(keys(prepared.discovery.trajectory_inputs)) ==
              Set(["outer_trajectory"])

        figures = CFG.build_canonical_figures(prepared)
        @test length(figures.inclusion_frequency.data) == 3
        @test length(figures.may2024_reconstruction.fig.data) == 5
        @test length(figures.lambda_selection.fig.data) == 5
        @test length(figures.coefficient_stability.data) == 3
        @test length(figures.synthetic_recovery.fig.data) == 5
        @test length(figures.paired_performance.data) == 4
        @test CFG._display_terms(["Dst_star", "n*V", "Newell_d_Φ"]) ==
              ["Dst*", "n V", "dΦN/dt"]

        stability_medians = Float64.(
            figures.coefficient_stability.data[2].fields[:y],
        )
        @test length(stability_medians) == 2
        @test all(abs.(stability_medians) .== 1.0)
        inclusion_layout = figures.inclusion_frequency.layout.fields
        @test !haskey(inclusion_layout, :xaxis2)
        @test !haskey(inclusion_layout, :yaxis2)
        inclusion_order = sortperm(prepared.stability.inclusion;
                                   rev=true, alg=MergeSort)
        ordered_inclusion = prepared.stability.inclusion[inclusion_order]
        core = ordered_inclusion .>= 0.9
        inclusion_positions = collect(1:20)
        @test figures.inclusion_frequency.data[1].fields[:x] ==
              inclusion_positions[core]
        @test figures.inclusion_frequency.data[1].fields[:y] ==
              ordered_inclusion[core]
        @test figures.inclusion_frequency.data[2].fields[:x] ==
              inclusion_positions[.!core]
        @test figures.inclusion_frequency.data[2].fields[:y] ==
              ordered_inclusion[.!core]
        @test figures.inclusion_frequency.data[3].fields[:x] == [0.5, 20.5]
        @test figures.inclusion_frequency.data[3].fields[:y] == [0.9, 0.9]
        @test [trace.fields[:name] for trace in
               figures.inclusion_frequency.data] == [
            "Core (pi >= 0.9)", "Peripheral (pi < 0.9)",
            "pi = 0.9 threshold",
        ]
        @test inclusion_layout[:xaxis][:tickmode] == "array"
        @test inclusion_layout[:xaxis][:tickvals] == inclusion_positions
        @test inclusion_layout[:xaxis][:ticktext] ==
              prepared.stability.terms[inclusion_order]
        @test inclusion_layout[:xaxis][:tickangle] == -45
        @test inclusion_layout[:legend][:xanchor] == "right"
        @test inclusion_layout[:legend][:yanchor] == "top"
        may_layout = figures.may2024_reconstruction.fig.layout.fields
        @test figures.may2024_reconstruction.fig.data[1].fields[:y] ==
              prepared.discovery.velocity
        @test figures.may2024_reconstruction.fig.data[1].fields[:xaxis] == "x"
        @test figures.may2024_reconstruction.fig.data[1].fields[:yaxis] == "y"
        @test figures.may2024_reconstruction.fig.data[2].fields[:xaxis] == "x2"
        @test figures.may2024_reconstruction.fig.data[2].fields[:yaxis] == "y2"
        @test may_layout[:xaxis2][:title][:text] ==
              "Time [hours from shared anchor]"
        @test [trace.fields[:name] for trace in
               figures.may2024_reconstruction.fig.data] == [
            "V [km/s]", "Observed", "SINDy (11-term)",
            "Burton (1975)", "O'Brien-McP (2000)",
        ]
        @test figures.may2024_reconstruction.fig.data[4].fields[:line][:dash] ==
              "dash"
        @test figures.may2024_reconstruction.fig.data[5].fields[:line][:dash] ==
              "dashdot"
        @test figures.lambda_selection.fig.layout.fields[:legend][:yanchor] == "middle"
        @test figures.lambda_selection.fig.layout.fields[:legend2][:yanchor] == "middle"
        stability_layout = figures.coefficient_stability.layout.fields
        stability_legend = stability_layout[:legend]
        @test stability_legend[:xanchor] == "right"
        @test stability_legend[:yanchor] == "top"
        @test !haskey(stability_layout, :xaxis2)
        @test !haskey(stability_layout, :yaxis2)
        stability_valid = findall(
            isfinite.(prepared.stability.medians) .& .!iszero.(prepared.stability.medians),
        )
        stability_positions = Float64.(stability_valid)
        stability_scales = abs.(prepared.stability.medians[stability_valid])
        expected_stability_medians =
            prepared.stability.medians[stability_valid] ./ stability_scales
        expected_stability_point =
            prepared.stability.point[stability_valid] ./ stability_scales
        @test figures.coefficient_stability.data[2].fields[:x] == stability_positions
        @test figures.coefficient_stability.data[3].fields[:x] == stability_positions
        @test figures.coefficient_stability.data[2].fields[:y] ==
              expected_stability_medians
        @test figures.coefficient_stability.data[3].fields[:y] ==
              expected_stability_point
        expected_interval_y, expected_interval_x = CFG._interval_segments(
            stability_positions,
            prepared.stability.lower[stability_valid] ./ stability_scales,
            prepared.stability.upper[stability_valid] ./ stability_scales,
        )
        @test isequal(
            figures.coefficient_stability.data[1].fields[:x],
            expected_interval_x,
        )
        @test isequal(
            figures.coefficient_stability.data[1].fields[:y],
            expected_interval_y,
        )
        @test may_layout[:xaxis][:domain] == may_layout[:xaxis2][:domain]
        @test collect(may_layout[:xaxis][:domain]) == [0.0, 1.0]
        @test maximum(may_layout[:yaxis2][:domain]) <
              minimum(may_layout[:yaxis][:domain])
        for (legend_key, xaxis_key, yaxis_key) in (
            (:legend, :xaxis, :yaxis), (:legend2, :xaxis2, :yaxis2),
        )
            legend = may_layout[legend_key]
            @test first(may_layout[xaxis_key][:domain]) <= legend[:x] <=
                  last(may_layout[xaxis_key][:domain])
            @test first(may_layout[yaxis_key][:domain]) <= legend[:y] <=
                  last(may_layout[yaxis_key][:domain])
        end
        for figure in (figures.lambda_selection, figures.synthetic_recovery)
            layout = figure.fig.layout.fields
            @test layout[:xaxis][:domain] == layout[:xaxis2][:domain]
            @test layout[:yaxis][:domain][1] - layout[:yaxis2][:domain][2] ≈ 0.14
        end
        synthetic_layout = figures.synthetic_recovery.fig.layout.fields
        synthetic_legend = synthetic_layout[:legend]
        @test synthetic_legend[:orientation] == "h"
        @test synthetic_legend[:xanchor] == "center"
        @test synthetic_legend[:yanchor] == "bottom"
        @test synthetic_legend[:y] > maximum(synthetic_layout[:yaxis][:domain])
        @test !haskey(synthetic_layout, :legend2)
        @test figures.synthetic_recovery.fig.data[4].fields[:showlegend] == false
        @test figures.synthetic_recovery.fig.data[5].fields[:showlegend] == false
        for (upper, lower) in ((2, 4), (3, 5))
            @test figures.synthetic_recovery.fig.data[upper].fields[:line][:color] ==
                  figures.synthetic_recovery.fig.data[lower].fields[:line][:color]
            @test figures.synthetic_recovery.fig.data[upper].fields[:line][:dash] ==
                  figures.synthetic_recovery.fig.data[lower].fields[:line][:dash]
        end
        @test figures.paired_performance.layout.fields[:showlegend] == false
        @test CFG._INCLUSION_FREQUENCY_FIGURE_HEIGHT == 360
        @test CFG._MAY2024_RECONSTRUCTION_FIGURE_HEIGHT == 540
        @test CFG._COEFFICIENT_STABILITY_FIGURE_HEIGHT == 360
        @test CFG._SYNTHETIC_FIGURE_HEIGHT == 800
        @test synthetic_layout[:margin][:t] == 70
        submitted_layout = figures.lambda_selection.fig.layout.fields
        @test CFG._BASE_FONT_SIZE == 24
        @test CFG._AXIS_TITLE_FONT_SIZE == 24
        @test CFG._BASE_FONT_SIZE * 345 / CFG._FIGURE_WIDTH >= 8
        @test CFG._AXIS_TITLE_FONT_SIZE * 345 / CFG._FIGURE_WIDTH >= 8
        @test submitted_layout[:font][:family] == CFG._FONT_FAMILY
        @test submitted_layout[:font][:size] == CFG._BASE_FONT_SIZE
        @test submitted_layout[:font][:color] == CFG._TEXT
        @test submitted_layout[:margin][:l] == 53
        @test submitted_layout[:margin][:r] == 20
        @test submitted_layout[:margin][:t] == 25
        @test submitted_layout[:margin][:b] == 25
        @test submitted_layout[:xaxis][:title][:font][:size] ==
              CFG._AXIS_TITLE_FONT_SIZE
        @test !haskey(may_layout, :legend3)
        lambda_layout = figures.lambda_selection.fig.layout.fields
        for key in (:xaxis, :xaxis2)
            @test lambda_layout[key][:tickmode] == "array"
            @test lambda_layout[key][:tickvals] == CFG._LAMBDA_TICK_VALUES
            @test lambda_layout[key][:ticktext] == CFG._LAMBDA_TICK_LABELS
            @test first(lambda_layout[key][:tickvals]) ≈ minimum(prepared.lambda.lambdas)
            @test last(lambda_layout[key][:tickvals]) ≈ maximum(prepared.lambda.lambdas)
        end

        discovery_points = CSV.read(
            joinpath(paths.data, "real_sindy_discovery_coefficients.csv"), DataFrame,
        )
        discovery_norms = CSV.read(
            joinpath(paths.data, "real_design_column_norms.csv"), DataFrame,
        )
        selected_index = findfirst(x -> !iszero(x), discovery_points.coefficient)
        ten_points = copy(discovery_points)
        ten_norms = copy(discovery_norms)
        ten_points.coefficient[selected_index] = 0.0
        ten_norms.coefficient[selected_index] = 0.0
        ten_norms.selected[selected_index] = false
        @test_throws ErrorException CFG._validate_discovery(
            ten_points, ten_norms, prepared.discovery.trajectory,
        )
        unselected_index = findfirst(iszero, discovery_points.coefficient)
        twelve_points = copy(discovery_points)
        twelve_norms = copy(discovery_norms)
        twelve_points.coefficient[unselected_index] = 0.02
        twelve_norms.coefficient[unselected_index] = 0.02
        twelve_norms.selected[unselected_index] = true
        @test_throws ErrorException CFG._validate_discovery(
            twelve_points, twelve_norms, prepared.discovery.trajectory,
        )

        # Rendering must stay bound to the bytes verified during preparation,
        # even if a changed CSV subsequently receives a valid fresh manifest.
        point_path = joinpath(paths.data, "real_sindy_discovery_coefficients.csv")
        point_frame = CSV.read(point_path, DataFrame)
        stability_point_path = joinpath(paths.data, "real_sindy_coefficients.csv")
        stability_point_frame = CSV.read(stability_point_path, DataFrame)
        changed_stability_point_frame = copy(stability_point_frame)
        changed_stability_point_frame.coefficient[2] = -0.12
        _figure_fixture_write(
            paths, "real_sindy_coefficients.csv", changed_stability_point_frame,
        )
        @test_throws ErrorException CFG._write_figure(
            prepared, "fig_discovery_validation.pdf", figures.inclusion_frequency,
            prepared.stability.inputs, prepared.stability.input_hashes;
            height=CFG._INCLUSION_FREQUENCY_FIGURE_HEIGHT,
            role="stale_input_regression",
        )
        _figure_fixture_write(
            paths, "real_sindy_coefficients.csv", stability_point_frame,
        )

        output_inputs = Dict(
            "fig_discovery_validation.pdf" => prepared.stability.inputs,
            "fig_may2024_reconstruction.pdf" =>
                prepared.discovery.trajectory_inputs,
            "fig_lambda_selection.pdf" => prepared.lambda.inputs,
            "fig_coefficient_stability.pdf" => prepared.stability.inputs,
            "fig_synthetic_recovery.pdf" => prepared.synthetic.inputs,
            "fig_paired_performance.pdf" => prepared.paired.inputs,
        )
        producer = joinpath(
            _FIGURE_TEST_PACKAGE_ROOT, "validation", "canonical_figure_generation.jl",
        )
        outputs = [joinpath(paths.figs, name) for name in keys(output_inputs)]
        for output in outputs
            name = basename(output)
            CFG.write_manifested_figure(
                output, path -> write(path, "%PDF-1.4\nold-$name\n%%EOF\n");
                producer_script=producer,
                input_paths=output_inputs[name],
                selection_record=(kind="valid_prior_figure", figure=name),
                backend="PlotlySupply.jl", deterministic=true,
                package_root=_FIGURE_TEST_PACKAGE_ROOT, mode=:test,
            )
        end
        transaction_paths = vcat(
            outputs, [path * ".manifest.json" for path in outputs],
        )
        prior_bytes = Dict(path => read(path) for path in transaction_paths)
        replacement_writer = function (prepared, name, figure, inputs, input_hashes;
                                       height, role)
            CFG._assert_inputs_unchanged(inputs, input_hashes)
            output = joinpath(prepared.paths.figs, name)
            return CFG.write_manifested_figure(
                output, path -> write(path, "%PDF-1.4\nnew-$name\n%%EOF\n");
                producer_script=producer, input_paths=inputs,
                selection_record=(kind="replacement_figure", figure=name, role),
                backend="PlotlySupply.jl", deterministic=true,
                metadata=(width_px=CFG._FIGURE_WIDTH, height_px=height, role),
                package_root=prepared.package_root, mode=prepared.paths.mode,
                verify_source=true,
            )
        end
        hook_calls = Ref(0)
        changed_before_failure = Ref(false)
        @test_throws ErrorException CFG.run_canonical_figure_generation(
            paths; package_root=_FIGURE_TEST_PACKAGE_ROOT,
            _figure_writer=replacement_writer,
            _after_figure_hook=(name, output) -> begin
                hook_calls[] += 1
                if name == "fig_coefficient_stability.pdf"
                    changed_before_failure[] = read(output) != prior_bytes[output]
                    error("injected late figure failure")
                end
            end,
        )
        @test hook_calls[] == 4
        @test changed_before_failure[]
        @test all(read(path) == prior_bytes[path] for path in transaction_paths)
        @test all(output -> CFG.verify_output_manifest(
            output; package_root=_FIGURE_TEST_PACKAGE_ROOT,
            require_canonical=false, verify_source=true,
        ) !== nothing, outputs)

        if get(ENV, "SOLARSINDY_FIGURE_EXPORT_SMOKE", "0") == "1"
            outputs = CFG.run_canonical_figure_generation(paths;
                package_root=_FIGURE_TEST_PACKAGE_ROOT)
            @test Set(basename.(outputs)) ==
                Set(keys(CFG.CANONICAL_FIGURE_ARTIFACT_INVENTORY))
            @test all(path -> isfile(path) && filesize(path) > 0 &&
                              isfile(path * ".manifest.json"), outputs)
        end

        # A stale source identity must be rejected; production never disables
        # verify_source and test mode exercises the same check.
        manifest_path = point_path * ".manifest.json"
        original_manifest = read(manifest_path, String)
        record = JSON3.read(original_manifest)
        source_hash = String(record["source_identity"]["source_tree_sha256"])
        stale_manifest = replace(original_manifest, source_hash => repeat("0", 64))
        @test stale_manifest != original_manifest
        write(manifest_path, stale_manifest)
        @test_throws ErrorException CFG.prepare_canonical_figure_inputs(
            paths; package_root=_FIGURE_TEST_PACKAGE_ROOT,
        )
        write(manifest_path, original_manifest)

        # Correct schema with a non-inventory producer is still rejected.
        _figure_fixture_write(paths, "real_sindy_discovery_coefficients.csv", point_frame;
            producer=joinpath(@__DIR__, "test_canonical_figure_generation.jl"))
        @test_throws ErrorException CFG.prepare_canonical_figure_inputs(
            paths; package_root=_FIGURE_TEST_PACKAGE_ROOT,
        )
        _figure_fixture_write(paths, "real_sindy_discovery_coefficients.csv", point_frame)

        # A correctly manifested but sparse 59-row grid cannot be plotted.
        candidates_path = joinpath(paths.data, "primary_lambda_candidates.csv")
        candidates = CSV.read(candidates_path, DataFrame)[1:59, :]
        _figure_fixture_write(paths, "primary_lambda_candidates.csv", candidates)
        @test_throws ErrorException CFG.prepare_canonical_figure_inputs(
            paths; package_root=_FIGURE_TEST_PACKAGE_ROOT,
        )
    end
end
