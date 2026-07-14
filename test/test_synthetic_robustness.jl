using Test
using SolarSINDy
using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "validation", "synthetic_robustness.jl"))
using .SyntheticRobustnessValidation

@testset "Synthetic robustness protocols" begin
    @test DEFAULT_ROBUSTNESS_SEED == 20_000
    @test DEFAULT_NOISE_LEVELS == [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    @test DEFAULT_SCALABILITY_TRAINING_ROWS == [80, 160, 240, 320, 400, 480, 560]

    canonical_variants = SyntheticRobustnessValidation._canonical_ablation_variants()
    signatures = SyntheticRobustnessValidation._ablation_signatures(
        canonical_variants,
    )
    @test first.(canonical_variants) == [
        "minimal_identifiable", "no_clock_angle_terms",
        "no_polynomial_cross_terms", "full_canonical",
    ]
    no_clock = only(filter(row -> row.name == "no_clock_angle_terms", signatures))
    @test !("Newell_d_Φ" in no_clock.terms)
    @test all(term -> !occursin("sin", term), no_clock.terms)
    no_cross = only(filter(
        row -> row.name == "no_polynomial_cross_terms", signatures,
    ))
    @test no_cross.decay_term_available
    @test !no_cross.injection_term_available

    paired_low = SyntheticRobustnessValidation.synthetic_recovery_dataset(
        seed=20_001, derivative_noise_std=0.1,
    )
    paired_high = SyntheticRobustnessValidation.synthetic_recovery_dataset(
        seed=20_001, derivative_noise_std=0.25,
    )
    @test paired_low.data == paired_high.data
    low_standard_noise =
        (paired_low.observed_derivative .- paired_low.true_derivative) ./ 0.1
    high_standard_noise =
        (paired_high.observed_derivative .- paired_high.true_derivative) ./ 0.25
    @test low_standard_noise ≈ high_standard_noise rtol=2e-13 atol=2e-13

    bundle = SyntheticRobustnessValidation.synthetic_recovery_dataset(
        seed=20_001, derivative_noise_std=0.25,
    )
    library = build_minimal_library()
    theta = evaluate_library(library, bundle.data)
    lambdas = [0.1, 10.0, 250.0]
    selection = SyntheticRobustnessValidation._select_lambda_rows(
        theta, bundle.observed_derivative;
        lambdas, fit_rows=1:100, validation_rows=101:140,
    )
    contaminated = copy(bundle.observed_derivative)
    contaminated[141:end] .= 1.0e12
    repeated = SyntheticRobustnessValidation._select_lambda_rows(
        theta, contaminated;
        lambdas, fit_rows=1:100, validation_rows=101:140,
    )
    @test repeated.lambda == selection.lambda
    @test repeated.sweep == selection.sweep
    @test count(row -> row.selected, selection.sweep) == 1
    @test_throws ArgumentError SyntheticRobustnessValidation._select_lambda_rows(
        theta, bundle.observed_derivative;
        lambdas, fit_rows=1:100, validation_rows=100:140,
    )

    mktempdir() do root
        result = withenv(
            "SOLARSINDY_OUTPUT_ROOT" => root,
            "SOLARSINDY_RUN_MODE" => "test",
            "SOLARSINDY_OMNI_EXTRACTED" => nothing,
        ) do
            run_synthetic_robustness(
                noise_levels=[0.0, 0.25],
                noise_replicates=2,
                scalability_training_rows=[80, 160],
                lambdas=lambdas,
            )
        end
        frames = result.frames
        @test nrow(frames.noise_metrics) == 8
        @test nrow(frames.noise_lambda) == 24
        @test nrow(frames.ablation_metrics) == 4
        @test nrow(frames.ablation_lambda) == 12
        @test nrow(frames.scalability_metrics) == 2
        @test nrow(frames.scalability_lambda) == 6
        @test Set(frames.noise_metrics.seed) == Set([20_001, 20_002])
        @test Set(frames.noise_metrics.variant) ==
              Set(["minimal_identifiable", "full_canonical"])
        @test all(count(group.selected) == 1 for group in groupby(
            frames.noise_lambda,
            [:variant, :derivative_noise_std_nt_per_hour, :replicate],
        ))
        @test all(count(group.selected) == 1 for group in groupby(
            frames.ablation_lambda, :variant,
        ))
        @test all(count(group.selected) == 1 for group in groupby(
            frames.scalability_lambda, :training_rows,
        ))
        @test all(frames.scalability_metrics.fixed_holdout)
        @test all(frames.scalability_metrics.holdout_first_row .== 641)
        @test all(frames.scalability_metrics.holdout_last_row .== 801)
        @test frames.scalability_metrics.requested_training_rows == [80, 160]
        @test all(isfinite, frames.noise_metrics.heldout_forward_rmse_nt)
        @test all(isfinite, frames.ablation_metrics.holdout_derivative_rmse_nt_per_hour)
        @test all(isfinite, frames.scalability_metrics.heldout_forward_rmse_nt)
        omitted = only(eachrow(frames.ablation_metrics[
            frames.ablation_metrics.variant .== "no_polynomial_cross_terms", :,
        ]))
        @test omitted.decay_term_available
        @test !omitted.injection_term_available
        @test ismissing(omitted.injection_coefficient)
        @test ismissing(omitted.injection_relative_error)
        @test !omitted.injection_sign_ok
        @test omitted.support_recall <= 0.5

        for (name, path) in pairs(result.paths)
            @test isfile(path)
            @test isfile(path * ".manifest.json")
            record = SyntheticRobustnessValidation.
                SyntheticEquationRecoveryValidation.verify_output_manifest(
                    path; require_canonical=false, verify_source=true,
            )
            @test Int(record["randomness"]["seed"]) == DEFAULT_ROBUSTNESS_SEED
            @test String(record["selection_record"]["artifact"]) == String(name)
            @test Int(record["selection_record"]["root_seed"]) ==
                  DEFAULT_ROBUSTNESS_SEED
            @test Int.(record["selection_record"]["noise_sweep"]["seeds"]) ==
                  collect(20_001:20_002)
            @test Set(String.(record["selection_record"]["noise_sweep"][
                "libraries"
            ])) == Set(["minimal_identifiable", "full_canonical"])
            scalability = record["selection_record"]["scalability"]
            @test Bool(scalability[
                "evaluation_holdout_fixed_across_training_sizes"
            ])
        end
    end

    mktempdir() do root
        withenv(
            "SOLARSINDY_OUTPUT_ROOT" => root,
            "SOLARSINDY_RUN_MODE" => "canonical",
        ) do
            @test_throws ArgumentError run_synthetic_robustness(
                root_seed=DEFAULT_ROBUSTNESS_SEED + 1,
                persist=false,
            )
            @test_throws ArgumentError run_synthetic_robustness(
                noise_levels=[0.0],
                persist=false,
            )
            canonical_names = first.(
                SyntheticRobustnessValidation._canonical_ablation_variants(),
            )
            @test_throws ArgumentError run_synthetic_robustness(
                ablation_variants=[
                    name => build_minimal_library() for name in canonical_names
                ],
                persist=false,
            )
        end
    end

    @test_throws ArgumentError run_synthetic_robustness(
        noise_levels=[0.0],
        noise_replicates=1,
        scalability_training_rows=Int[],
        lambdas=[0.1],
        ablation_variants=["minimal_identifiable" => build_minimal_library()],
        persist=false,
    )

    original_first_noise = DEFAULT_NOISE_LEVELS[1]
    try
        DEFAULT_NOISE_LEVELS[1] = 9.0
        @test Tuple(DEFAULT_NOISE_LEVELS) !=
              SyntheticRobustnessValidation._CANONICAL_NOISE_LEVELS
        mktempdir() do root
            withenv(
                "SOLARSINDY_OUTPUT_ROOT" => root,
                "SOLARSINDY_RUN_MODE" => "canonical",
            ) do
                @test_throws ArgumentError run_synthetic_robustness(
                    persist=false,
                )
            end
        end
    finally
        DEFAULT_NOISE_LEVELS[1] = original_first_noise
    end
end
