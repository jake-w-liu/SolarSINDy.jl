using Test
using SolarSINDy
using CSV
using DataFrames
using Statistics

include(joinpath(@__DIR__, "..", "validation", "synthetic_equation_recovery.jl"))
using .SyntheticEquationRecoveryValidation

@testset "Synthetic equation recovery" begin
    @testset "input contracts" begin
        @test_throws ArgumentError synthetic_recovery_dataset(seed=-1)
        bundle = synthetic_recovery_dataset(derivative_noise_std=0.0)
        library = build_minimal_library()
        @test_throws ArgumentError run_recovery_experiment(
            bundle, library; experiment="x", objective="unknown",
        )
        @test_throws ArgumentError run_recovery_experiment(
            bundle, library; experiment="x", objective="identifiable_recovery",
            lambdas=Float64[],
        )
        @test_throws ArgumentError run_recovery_experiment(
            bundle, library; experiment="x", objective="identifiable_recovery",
            lambdas=[0.1, 0.1],
        )
        @test_throws ArgumentError run_recovery_experiment(
            bundle, library; experiment="x", objective="identifiable_recovery",
            forward_rmse_max=Inf,
        )
    end

    @testset "independent noiseless minimal-library oracle" begin
        bundle = synthetic_recovery_dataset(
            seed=DEFAULT_RECOVERY_SEED,
            derivative_noise_std=0.0,
        )
        oracle = -CANONICAL_ALPHA .* bundle.swd.V .* bundle.data["Bs"] .-
                 bundle.swd.Dst_star ./ CANONICAL_TAU
        @test bundle.true_derivative == oracle
        @test bundle.observed_derivative == oracle

        result = run_recovery_experiment(
            bundle,
            build_minimal_library();
            experiment="noiseless_oracle",
            objective="identifiable_recovery",
        )
        @test result.names[result.active] == ["Dst_star", "V*Bs"]
        @test result.summary.support_precision == 1.0
        @test result.summary.support_recall == 1.0
        @test result.summary.decay_sign_ok
        @test result.summary.injection_sign_ok
        @test result.summary.decay_relative_error < 1e-12
        @test result.summary.injection_relative_error < 1e-12
        @test result.summary.heldout_forward_rmse_nt < 1e-9
        @test result.summary.recovery_pass
    end

    @testset "deterministic noisy recovery and canonical stress artifacts" begin
        mktempdir() do root
            result = withenv(
                "SOLARSINDY_OUTPUT_ROOT" => root,
                "SOLARSINDY_RUN_MODE" => "test",
                "SOLARSINDY_OMNI_EXTRACTED" => nothing,
            ) do
                run_synthetic_equation_recovery(
                    seed=DEFAULT_RECOVERY_SEED,
                    derivative_noise_std=0.25,
                )
            end
            repeated_bundle = synthetic_recovery_dataset(
                seed=DEFAULT_RECOVERY_SEED,
                derivative_noise_std=0.25,
            )

            minimal = result.minimal.summary
            full = result.full.summary
            @test result.validation_pass
            @test minimal.selected_lambda == 250.0
            @test minimal.support_precision == 1.0
            @test minimal.support_recall == 1.0
            @test minimal.recovery_pass
            @test isapprox(minimal.decay_coefficient, -0.1298364667; rtol=1e-6)
            @test isapprox(minimal.injection_coefficient, -0.0054019547; rtol=1e-6)
            @test minimal.heldout_forward_rmse_nt < 0.1

            @test full.selected_lambda == 250.0
            @test full.support_recall == 1.0
            @test full.support_precision < 1.0
            @test full.false_discoveries >= 1
            @test full.clock_false_discoveries >= 1
            @test full.decay_sign_ok && full.injection_sign_ok
            @test !full.recovery_pass
            @test full.stress_detected
            @test full.clock_block_condition_number > 100.0
            @test full.maximum_pair_correlation > 0.95
            @test full.experiment_pass
            @test !full.canonical_gate_applied
            @test full.outcome_label == "stress_detected"
            @test minimal.canonical_gate_applied
            @test minimal.outcome_label == "recovery_pass"
            @test result.validation_pass == minimal.experiment_pass

            @test repeated_bundle.observed_derivative == result.bundle.observed_derivative
            @test repeated_bundle.swd.Dst_star == result.bundle.swd.Dst_star

            @test result.paths.summary == joinpath(
                abspath(root), "data", "synthetic_equation_recovery_summary.csv",
            )
            @test all(isfile, values(result.paths))
            summary = CSV.read(result.paths.summary, DataFrame)
            coefficients = CSV.read(result.paths.coefficients, DataFrame)
            lambda_selection = CSV.read(result.paths.lambda_selection, DataFrame)
            trajectories = CSV.read(result.paths.trajectories, DataFrame)
            @test nrow(summary) == 2
            @test all(summary.seed .== DEFAULT_RECOVERY_SEED)
            @test all(summary.alpha_true .== CANONICAL_ALPHA)
            @test all(summary.tau_true .== CANONICAL_TAU)
            @test all(summary.overall_validation_pass)
            @test all(summary.minimal_recovery_validation_pass)
            @test all(summary.full_stress_outcome_neutral)
            @test all(occursin.("chronological internal validation",
                                summary.lambda_protocol))
            @test count(coefficients.true_support) == 4
            @test all(coefficients.lambda_fit_column_norm .> 0.0)
            @test all(coefficients.refit_column_norm .> 0.0)
            @test count(lambda_selection.selected) == 2
            @test Set(lambda_selection[lambda_selection.selected, :lambda]) == Set([250.0])
            @test nrow(trajectories) == 482
            @test count(trajectories.is_anchor) == 2
            @test trajectories.error_nt ==
                  trajectories.dst_star_simulated_nt .- trajectories.dst_star_oracle_nt
            for group in groupby(trajectories, :experiment)
                experiment = String(first(group.experiment))
                expected = only(summary[summary.experiment .== experiment,
                                        :heldout_forward_rmse_nt])
                @test isapprox(sqrt(mean(abs2, group.error_nt)), expected; rtol=1e-12)
                @test group.global_row == collect(561:801)
                @test group.holdout_row == collect(1:241)
            end
            for (name, path) in pairs(result.paths)
                @test isfile(path * ".manifest.json")
                record = SyntheticEquationRecoveryValidation.verify_output_manifest(
                    path; require_canonical=false, verify_source=true,
                )
                @test String(record["run_mode"]) == "test"
                @test isempty(record["inputs"])
                @test String(record["randomness"]["kind"]) == "seeded"
                @test Int(record["randomness"]["seed"]) == DEFAULT_RECOVERY_SEED
                selection = record["selection_record"]
                @test String(selection["artifact"]) == String(name)
                @test String(selection["split"]["holdout_rows"]) == "561:801"
                @test length(selection["libraries"]["full"]["terms"]) == 20
                @test !("n*V^2" in String.(selection["libraries"]["full"]["terms"]))
                @test Bool(selection["overall_validation_pass"])
                @test Bool(selection["full_stress_outcome_neutral"])
                @test String(selection["canonical_gate_scope"]) ==
                      "minimal_identifiable_recovery_only"
            end
        end
    end

    @testset "full-library outcome is not a recovery gate" begin
        bundle = synthetic_recovery_dataset(
            seed=DEFAULT_RECOVERY_SEED,
            derivative_noise_std=0.0,
        )
        full = run_recovery_experiment(
            bundle,
            build_solar_wind_library();
            experiment="outcome_neutral_full_stress",
            objective="false_discovery_collinearity_stress",
            lambdas=[1.0e12],
        )
        @test full.summary.support_recall < 1.0
        @test !full.summary.recovery_pass
        @test !full.summary.experiment_pass
        @test !full.summary.canonical_gate_applied
        @test full.summary.outcome_label == "stress_not_detected"
        @test full.summary.clock_block_condition_number > 0.0
        @test full.summary.maximum_pair_correlation > 0.0
    end

    @testset "canonical protocol and persistence rollback" begin
        mktempdir() do root
            withenv("SOLARSINDY_OUTPUT_ROOT" => root,
                    "SOLARSINDY_RUN_MODE" => "canonical") do
                @test_throws ArgumentError run_synthetic_equation_recovery(
                    seed=DEFAULT_RECOVERY_SEED + 1, persist=false,
                )
                @test_throws ArgumentError run_synthetic_equation_recovery(
                    derivative_noise_std=0.3, persist=false,
                )
            end
        end

        mktempdir() do root
            paths = (
                summary=joinpath(root, "summary.csv"),
                coefficients=joinpath(root, "coefficients.csv"),
                lambda_selection=joinpath(root, "lambda.csv"),
                trajectories=joinpath(root, "trajectories.csv"),
            )
            frames = NamedTuple{propertynames(paths)}(
                Tuple(DataFrame(value=[index]) for index in 1:length(paths)),
            )
            @test_throws ArgumentError SyntheticEquationRecoveryValidation._persist_synthetic_outputs!(
                    paths, frames, (kind="rollback_test",);
                    seed=DEFAULT_RECOVERY_SEED,
                    mode=:test,
                    producer_script=joinpath(root, "missing_producer.jl"),
                )
            @test isempty(readdir(root))
        end

        mktempdir() do root
            paths = (
                summary=joinpath(root, "summary.csv"),
                coefficients=joinpath(root, "coefficients.csv"),
                lambda_selection=joinpath(root, "lambda.csv"),
                trajectories=joinpath(root, "trajectories.csv"),
            )
            frames = NamedTuple{propertynames(paths)}(
                Tuple(DataFrame(value=[index]) for index in 1:length(paths)),
            )
            mkpath(paths.coefficients)
            producer = joinpath(
                @__DIR__, "..", "validation", "synthetic_equation_recovery.jl",
            )
            @test_throws ArgumentError SyntheticEquationRecoveryValidation._persist_synthetic_outputs!(
                paths, frames, (kind="second_artifact_rollback_test",);
                seed=DEFAULT_RECOVERY_SEED,
                mode=:test,
                producer_script=producer,
            )
            @test isdir(paths.coefficients)
            @test isempty(readdir(paths.coefficients))
            @test readdir(root) == [basename(paths.coefficients)]
        end

        mktempdir() do root
            paths = (
                summary=joinpath(root, "summary.csv"),
                coefficients=joinpath(root, "coefficients.csv"),
                lambda_selection=joinpath(root, "lambda.csv"),
                trajectories=joinpath(root, "trajectories.csv"),
            )
            frames = NamedTuple{propertynames(paths)}(
                Tuple(DataFrame(value=[index]) for index in 1:length(paths)),
            )
            original = Dict{String,Vector{UInt8}}()
            for (index, path) in enumerate(values(paths))
                write(path, "old-csv-$index\n")
                write(path * ".manifest.json", "old-manifest-$index\n")
                original[path] = read(path)
                original[path * ".manifest.json"] = read(path * ".manifest.json")
            end
            producer = joinpath(
                @__DIR__, "..", "validation", "synthetic_equation_recovery.jl",
            )
            @test_throws ErrorException SyntheticEquationRecoveryValidation._persist_synthetic_outputs!(
                paths, frames, (kind="late_rollback_test",);
                seed=DEFAULT_RECOVERY_SEED,
                mode=:test,
                producer_script=producer,
                _after_artifact_hook=(name, path) ->
                    name == :lambda_selection && error("injected late failure"),
            )
            for path in keys(original)
                @test read(path) == original[path]
            end
            @test sort(readdir(root)) == sort(vcat(
                collect(basename.(values(paths))),
                [basename(path) * ".manifest.json" for path in values(paths)],
            ))
        end
    end
end
