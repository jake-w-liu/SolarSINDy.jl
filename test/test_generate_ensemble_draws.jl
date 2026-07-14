using Test
using SolarSINDy
using CSV
using DataFrames
using Dates

if !isdefined(@__MODULE__, :_ensemble_selected_lambda)
    include(joinpath(@__DIR__, "..", "validation", "generate_ensemble_draws.jl"))
end

const _ENSEMBLE_TEST_PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
const _ENSEMBLE_TEST_PRODUCER = joinpath(
    _ENSEMBLE_TEST_PACKAGE_ROOT, "validation", "real_data_discovery.jl",
)

@testset "Ensemble output batch restores every valid prior pair" begin
    mktempdir() do root
        omni = joinpath(root, "omni.txt")
        catalog = joinpath(root, "catalog.csv")
        write(omni, "omni fixture\n")
        write(catalog, "catalog fixture\n")
        context = (data=root, mode=:test, omni, catalog)
        term_names = ["term_a", "term_b"]
        record = (kind="ensemble_transaction_fixture", selected_lambda=0.1)
        refit_record = merge(record, (audit="exact_refit_fixture", exact_match=true))

        function persist_fixture(offset; hook=(name, path) -> nothing)
            draws = [1.0 2.0 3.0; 4.0 5.0 6.0] .+ offset
            return _persist_regenerated_ensemble_outputs!(
                context,
                [(term=name, persisted_coefficient=offset + index)
                 for (index, name) in enumerate(term_names)],
                term_names,
                draws,
                [(term=name, inclusion_probability=offset + index / 10)
                 for (index, name) in enumerate(term_names)],
                [(term=name, coefficient=offset + index)
                 for (index, name) in enumerate(term_names)];
                record, refit_record, refit_inputs=Dict{String,String}(),
                seed=42, _after_artifact_hook=hook,
            )
        end

        prior_outputs = persist_fixture(0.0)
        outputs = String[getproperty(prior_outputs, field)
                         for field in propertynames(prior_outputs)]
        transaction_paths = vcat(
            outputs, [path * ".manifest.json" for path in outputs],
        )
        @test all(output -> verify_output_manifest(
            output; package_root=_ENSEMBLE_TEST_PACKAGE_ROOT,
            require_canonical=false, verify_source=true,
        ) !== nothing, outputs)
        prior_bytes = Dict(path => read(path) for path in transaction_paths)
        hook_calls = Symbol[]
        changed_before_failure = Ref(false)
        @test_throws ErrorException persist_fixture(10.0;
            hook=(name, path) -> begin
                push!(hook_calls, name)
                if name == :inclusion
                    changed_before_failure[] = read(path) != prior_bytes[path]
                    error("injected late ensemble failure")
                end
            end,
        )
        @test hook_calls == [:refit_audit, :draws, :inclusion]
        @test changed_before_failure[]
        @test all(read(path) == prior_bytes[path] for path in transaction_paths)
        @test all(output -> verify_output_manifest(
            output; package_root=_ENSEMBLE_TEST_PACKAGE_ROOT,
            require_canonical=false, verify_source=true,
        ) !== nothing, outputs)
    end
end

function _write_ensemble_test_fixture(path, frame)
    return write_manifested_csv(path, frame;
        producer_script=_ENSEMBLE_TEST_PRODUCER,
        input_paths=(;),
        selection_record=(kind="ensemble_producer_test_fixture",),
        deterministic=true,
        package_root=_ENSEMBLE_TEST_PACKAGE_ROOT,
        mode=:test,
    )
end

@testset verbose=true "Ensemble-draw producer rejects stale selection and refit state" begin
    mktempdir() do root
        # Independent expansion of the locked grid catches a swapped, omitted,
        # or perturbed candidate before a persisted decision can be reused.
        expected_grid = 10.0 .^ collect(range(-2.0, 4.0; length=60))
        selected_index = 17
        selected = falses(60)
        selected[selected_index] = true
        candidates = DataFrame(
            candidate_index=collect(1:60),
            lambda=expected_grid,
            mean_storm_rmse_nt=abs.(collect(1:60) .- selected_index),
            standard_error_nt=zeros(60),
            n_active_terms=fill(2, 60),
            eligible=selected,
            selected=selected,
        )
        decision = DataFrame([(
            n_training_storms=4,
            n_inner_training_storms=3,
            n_inner_validation_storms=1,
            minimum_candidate_index=selected_index,
            minimum_lambda=expected_grid[selected_index],
            minimum_mean_storm_rmse_nt=0.0,
            minimum_standard_error_nt=0.0,
            one_standard_error_cutoff_nt=0.0,
            selected_candidate_index=selected_index,
            selected_lambda=expected_grid[selected_index],
            selection_rule="largest_lambda_within_one_standard_error_then_fewer_terms_then_larger_lambda",
        )])
        selection_paths = (
            candidates=joinpath(root, "primary_lambda_candidates.csv"),
            decision=joinpath(root, "primary_lambda_decision.csv"),
        )
        CSV.write(selection_paths.candidates, candidates)
        CSV.write(selection_paths.decision, decision)

        lambda, decision_record = _ensemble_selected_lambda(selection_paths)
        @test lambda == expected_grid[selected_index]
        @test decision_record.selected_candidate_index == selected_index

        wrong_grid = copy(candidates)
        wrong_grid.lambda[23] = nextfloat(wrong_grid.lambda[23])
        CSV.write(selection_paths.candidates, wrong_grid)
        @test_throws ErrorException _ensemble_selected_lambda(selection_paths)
        CSV.write(selection_paths.candidates, candidates)

        wrong_decision = copy(decision)
        wrong_decision.selected_lambda[1] = expected_grid[selected_index + 1]
        CSV.write(selection_paths.decision, wrong_decision)
        @test_throws ErrorException _ensemble_selected_lambda(selection_paths)
        CSV.write(selection_paths.decision, decision)

        context = (data=root, mode=:test)
        onset = DateTime(2001, 2, 3, 4)
        regenerated_audit = (storm_records=[(
            storm_id=7, onset_time=onset, eligible=true, exclusion_reason="",
        )],)
        audit_path = joinpath(root, "real_storm_eligibility.csv")
        persisted_audit = DataFrame(
            storm_id=[7], onset_time=[onset], eligible=[true],
            exclusion_reason=["none"],
        )
        _write_ensemble_test_fixture(audit_path, persisted_audit)
        @test _ensemble_verify_observation_audit(context, regenerated_audit) == audit_path

        wrong_audit = copy(persisted_audit)
        wrong_audit.eligible[1] = false
        _write_ensemble_test_fixture(audit_path, wrong_audit)
        @test_throws ErrorException _ensemble_verify_observation_audit(
            context, regenerated_audit,
        )

        # Identity design gives the independently exact selected-lambda refit
        # [2, -3]; changing either persisted coefficient must be rejected.
        coefficient_path = joinpath(root, "real_sindy_discovery_coefficients.csv")
        terms = ["term_a", "term_b"]
        design = (theta=[1.0 0.0; 0.0 1.0], target=[2.0, -3.0])
        point_frame = DataFrame(term=terms, coefficient=[2.0, -3.0])
        _write_ensemble_test_fixture(coefficient_path, point_frame)
        path, persisted, fresh = _ensemble_read_point_coefficients(
            context, terms, design, lambda,
        )
        @test path == coefficient_path
        @test persisted == [2.0, -3.0]
        @test fresh == [2.0, -3.0]

        wrong_point_frame = copy(point_frame)
        wrong_point_frame.coefficient[1] = 2.25
        _write_ensemble_test_fixture(coefficient_path, wrong_point_frame)
        @test_throws ErrorException _ensemble_read_point_coefficients(
            context, terms, design, lambda,
        )
    end
end
