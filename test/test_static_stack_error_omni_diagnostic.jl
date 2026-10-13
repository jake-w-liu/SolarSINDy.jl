using CSV
using DataFrames
using Dates
using Random
using SolarSINDy
using Statistics
using Test

include(joinpath(
    @__DIR__, "..", "validation", "operational",
    "static_stack_error_omni_diagnostic.jl",
))

function _err_test_source_row(issue::DateTime, lead::Int, split::String;
        spread::Float64=17.0)
    columns = _err_omni_source_columns()
    values = map(columns) do column
        column == :issue_time_utc && return issue
        column == :target_time_utc && return issue + Hour(lead)
        column == :model_step_hours && return lead
        column == :split_label && return split
        column == :baseline_spread_nt && return spread
        column == :observation_dst_nt && return -10.0 - lead
        return Float64(findfirst(==(column), columns)) / 10.0
    end
    return NamedTuple{columns}(Tuple(values))
end

function _err_test_record(issue::DateTime, lead::Int, center::String;
        split::String="fit", observation::Float64=-5.0,
        base::Float64=-4.0)
    issue_features = ntuple(index -> Float64(index),
                            length(STATIC_STACK_RESIDUAL_FEATURES))
    trajectory = reshape(Float64.(1:70), 14, 5)
    return StaticStackM3OmniBaseRecord(
        issue, lead, 1, split, center, issue_features, trajectory,
        base, observation,
    )
end

function _err_test_complete_records(issue::DateTime, center::String;
        split::String="fit")
    records = Dict{Tuple{DateTime,Int},StaticStackM3OmniBaseRecord}()
    for lag in 1:STATIC_STACK_ERROR_OMNI_HISTORY_HOURS
        time = issue - Hour(lag)
        records[(time, 1)] = _err_test_record(
            time, 1, center; split=split,
            observation=-Float64(lag), base=-Float64(lag) - 0.5,
        )
    end
    for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
        records[(issue, lead)] = _err_test_record(
            issue, lead, center; split=split,
            observation=-10.0 - lead, base=-9.0 - lead,
        )
    end
    return records
end

function _err_test_fit_example(fold::String, issue::DateTime,
        center::String; lead::Int=1, observation::Float64=0.0)
    values = ntuple(index -> index ==
        length(STATIC_STACK_ERROR_EXOGENOUS_FEATURES) ? 0.0 :
        Float64(index) + 0.01 * hour(issue),
        length(STATIC_STACK_ERROR_EXOGENOUS_FEATURES))
    features = StaticStackErrorExogenousFeatures(
        issue, lead, center, 0.0,
        STATIC_STACK_ERROR_EXOGENOUS_FEATURES, values, true, :none,
    )
    fit_row = StaticStackErrorExogenousFitRow(
        features, issue + Hour(lead), issue + Hour(lead), observation,
    )
    return StaticStackM3OmniExample(fold, fit_row, -5.0, 0.0)
end

@testset "static stack-M3 OMNI diagnostic" begin
    @testset "point-only source and frozen feature semantics" begin
        table = DataFrame([
            _err_test_source_row(DateTime(2017, 10, 13, 15), 1, "fit"),
            _err_test_source_row(DateTime(2017, 10, 20, 23), 1,
                                   "validation"; spread=23.0),
            _err_test_source_row(DateTime(2020, 5, 23, 3), 1,
                                   "calibration"),
        ])
        mktemp() do path, io
            close(io)
            CSV.write(path, table)
            checksum = _drv_omni_sha256(path)
            source = static_stack_error_omni_read_source(
                path; expected_sha256=checksum,
            )
            @test nrow(source) == 2
            @test source.split_label == ["fit", "validation"]
            @test _err_omni_point_row_count(path) == 2
            features = _err_omni_issue_features(source, 2, 10.0)
            @test features[16] == source.served_inertial_sindy_dst_nt[2] -
                                  source.frozen_inertial_sindy_dst_nt[2]
            @test features[17] == 10.0 - source.served_inertial_sindy_dst_nt[2]
            @test features[22] == 23.0
        end
        bad = vcat(table[[1, 3], :], table[[2], :])
        mktemp() do path, io
            close(io)
            CSV.write(path, bad)
            @test _err_omni_point_row_count(path) == 1
            checksum = _drv_omni_sha256(path)
            point_only = static_stack_error_omni_read_source(
                path; expected_sha256=checksum,
            )
            @test nrow(point_only) == 1
            @test only(point_only.split_label) == "fit"
        end

        mktemp() do path, io
            close(io)
            CSV.write(path, table[1:2, :])
            open(path, "a") do append_io
                write(append_io,
                      "2020-05-23T03:00:00,2020-05-23T04:00:00,1," *
                      "calibration,\"unmaterialized calibration payload\n")
            end
            point_only = static_stack_error_omni_read_source(
                path; expected_sha256=_drv_omni_sha256(path),
            )
            @test nrow(point_only) == 2
            @test _err_omni_point_row_count(path) == 2
        end

        post_2022_target = DataFrame([
            _err_test_source_row(DateTime(2022, 12, 31, 23), 1, "fit"),
        ])
        mktemp() do path, io
            close(io)
            CSV.write(path, post_2022_target)
            @test_throws ArgumentError static_stack_error_omni_read_source(
                path; expected_sha256=_drv_omni_sha256(path),
            )
        end
    end

    @testset "regime-safe selection and exact null artifacts" begin
        audit = DataFrame(
            model_step_hours=collect(STATIC_STACK_DRIVER_OMNI_LEADS_H),
            ridge=fill(100.0, length(STATIC_STACK_DRIVER_OMNI_LEADS_H)),
            threshold=fill(0.0, length(STATIC_STACK_DRIVER_OMNI_LEADS_H)),
            valid=trues(length(STATIC_STACK_DRIVER_OMNI_LEADS_H)),
            mean_validation_rmse_nt=fill(1.0,
                                          length(STATIC_STACK_DRIVER_OMNI_LEADS_H)),
            fold_standard_error_nt=zeros(length(STATIC_STACK_DRIVER_OMNI_LEADS_H)),
            max_active_groups=fill(1, length(STATIC_STACK_DRIVER_OMNI_LEADS_H)),
            overall_gain_nt=[0.2, 0.2, 0.0, -0.1, 0.2, 0.2],
            regime_safe=Bool[true, false, true, true, false, false],
            one_standard_error_eligible=
                falses(length(STATIC_STACK_DRIVER_OMNI_LEADS_H)),
            selected=falses(length(STATIC_STACK_DRIVER_OMNI_LEADS_H)),
        )
        selected = _err_omni_choose_hyperparameters!(audit)
        @test selected[1].use_correction
        @test audit.selected == Bool[true, false, false, false, false, false]
        @test all(!selected[lead].use_correction
                  for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H if lead != 1)

        null_selected = Dict(
            lead => (
                use_correction=false,
                ridge=100.0,
                threshold=0.0,
                reason="no_regime_safe_positive_oof_gain",
            ) for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H
        )
        examples = StaticStackM3OmniExample[]
        for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H, offset in 1:2
            issue = STATIC_STACK_DRIVER_OMNI_FINAL_TRAIN_STOP - Hour(48 + offset)
            push!(examples, _err_test_fit_example(
                "fold_a", issue, repeat("a", 64); lead=lead,
            ))
        end
        artifacts = _err_omni_fit_final_artifacts(
            examples, (selected=null_selected,), repeat("c", 64),
            Dict{String,String}(), repeat("d", 64),
        )
        @test length(artifacts) == length(STATIC_STACK_DRIVER_OMNI_LEADS_H)
        @test all(iszero(artifact.intercept_nt) for artifact in artifacts)
        @test all(all(iszero, artifact.coefficients) for artifact in artifacts)
        @test all(!any(artifact.support_mask) for artifact in artifacts)
    end

    @testset "24-hour readiness, lineage, and target invariance" begin
        center = repeat("a", 64)
        issue = DateTime(2016, 6, 1, 12)
        records = _err_test_complete_records(issue, center)
        examples = _err_omni_examples(
            records, "fold_a", issue, issue, "fit",
        )
        @test length(examples) == length(STATIC_STACK_DRIVER_OMNI_LEADS_H)
        @test all(example.fit_row.features.ready for example in examples)
        @test all(example.fit_row.features.feature_names ==
                  STATIC_STACK_ERROR_EXOGENOUS_FEATURES
                  for example in examples)

        mutated = copy(records)
        original = records[(issue, 1)]
        mutated[(issue, 1)] = StaticStackM3OmniBaseRecord(
            original.issue_time, original.model_step_hours,
            original.source_row, original.split_label,
            original.prediction_base_sha256, original.issue_features,
            original.m2_trajectory, original.base_prediction_dst_nt,
            original.observation_dst_nt - 1000.0,
        )
        mutated_examples = _err_omni_examples(
            mutated, "fold_a", issue, issue, "fit",
        )
        original_h1 = only(filter(example ->
            example.fit_row.features.model_step_hours == 1, examples))
        mutated_h1 = only(filter(example ->
            example.fit_row.features.model_step_hours == 1,
            mutated_examples))
        @test mutated_h1.fit_row.features.values ==
              original_h1.fit_row.features.values
        @test mutated_h1.fit_row.observation_dst_nt !=
              original_h1.fit_row.observation_dst_nt

        missing_intermediate = copy(records)
        delete!(missing_intermediate, (issue - Hour(5), 1))
        @test isempty(_err_omni_examples(
            missing_intermediate, "fold_a", issue, issue, "fit",
        ))

        crossed = copy(records)
        previous = crossed[(issue - Hour(1), 1)]
        crossed[(issue - Hour(1), 1)] = StaticStackM3OmniBaseRecord(
            previous.issue_time, previous.model_step_hours,
            previous.source_row, previous.split_label, repeat("b", 64),
            previous.issue_features, previous.m2_trajectory,
            previous.base_prediction_dst_nt, previous.observation_dst_nt,
        )
        @test_throws ArgumentError _err_omni_examples(
            crossed, "fold_a", issue, issue, "fit",
        )
    end

    @testset "heterogeneous OOF centers and whole-anchor embargo" begin
        fold_a = repeat("a", 64)
        fold_b = repeat("b", 64)
        serving = repeat("c", 64)
        manifest = repeat("d", 64)
        cutoff = DateTime(2014, 3, 1)
        examples = StaticStackM3OmniExample[]
        for index in 1:80
            issue = DateTime(2014, 1, 1) + Hour(index)
            fold = index <= 40 ? "fold_a" : "fold_b"
            center = index <= 40 ? fold_a : fold_b
            push!(examples, _err_test_fit_example(
                fold, issue, center; observation=0.0,
            ))
        end
        for index in 1:6
            push!(examples, _err_test_fit_example(
                "fold_b", cutoff - Hour(index), fold_b; observation=0.0,
            ))
        end
        expected = Dict("fold_a" => fold_a, "fold_b" => fold_b)
        artifact = _err_omni_fit_crossfit_error_exogenous(
            examples;
            serving_base_center_sha256=serving,
            expected_fold_hashes=expected,
            training_manifest_sha256=manifest,
            model_step_hours=1,
            fit_as_of=cutoff,
            ridge=100.0,
            threshold=0.3,
            minimum_rows=32,
        )
        @test artifact.base_center_sha256 == serving
        @test artifact.fit_rows == 80
        @test occursin(manifest[1:16], artifact.label)
        wrong = copy(expected)
        wrong["fold_b"] = repeat("e", 64)
        @test_throws ArgumentError _err_omni_fit_crossfit_error_exogenous(
            examples;
            serving_base_center_sha256=serving,
            expected_fold_hashes=wrong,
            training_manifest_sha256=manifest,
            model_step_hours=1,
            fit_as_of=cutoff,
            ridge=100.0,
            threshold=0.3,
            minimum_rows=32,
        )
    end

    @testset "common-row month bootstrap and fail-closed decision" begin
        methods = ("m3_sparse_error_state", STATIC_STACK_ERROR_OMNI_COMPARATORS...)
        issues = (
            DateTime(2018, 1, 5), DateTime(2018, 1, 20),
            DateTime(2018, 2, 5), DateTime(2018, 2, 20),
        )
        rows = NamedTuple[]
        for method in methods, lead in STATIC_STACK_DRIVER_OMNI_LEADS_H, issue in issues
            error = method == "m3_sparse_error_state" ? 1.0 : 3.0
            push!(rows, (
                issue_time_utc=issue,
                target_time_utc=issue + Hour(lead),
                model_step_hours=lead,
                method=method,
                comparison_role=method == "m3_sparse_error_state" ?
                    "candidate" : "available_comparator",
                candidate_ready=true,
                fallback_reason="none",
                split_label="validation",
                predicted_dst_nt=error,
                observed_dst_nt=0.0,
                squared_error_nt2=error^2,
                absolute_error_nt=abs(error),
                raw_candidate_correction_nt=0.0,
                applied_candidate_correction_nt=0.0,
                latest_dst_nt=0.0,
                dst_delta_1h_nt=0.0,
                target_storm=false,
                active_or_deepening=false,
                recovery=false,
            ))
        end
        errors = DataFrame(rows)
        bootstrap = static_stack_error_omni_bootstrap(
            errors; draws=100, seed=951,
        )
        @test nrow(bootstrap) ==
              length(STATIC_STACK_DRIVER_OMNI_LEADS_H) * length(STATIC_STACK_ERROR_OMNI_COMPARATORS)
        @test all(bootstrap.point_improvement_nt .== 2.0)
        @test all(bootstrap.one_sided_95_lower_nt .== 2.0)
        @test only(unique(
            bootstrap.simultaneous_one_sided_95_lower_nt,
        )) == 2.0
        @test !any(bootstrap.event_block_complete)

        variable_rows = NamedTuple[]
        variable_comparators = ("served_inertial_sindy", "frozen_inertial_sindy")
        variable_issues = DateTime[
            DateTime(2018, month_value, day)
            for month_value in 1:4 for day in (5, 20)
        ]
        for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H,
                (index, issue) in pairs(variable_issues)
            candidate_error = 4.0 + 0.1 * isodd(index) + 0.01 * lead
            push!(variable_rows, merge(NamedTuple(errors[1, :]), (
                issue_time_utc=issue,
                target_time_utc=issue + Hour(lead),
                model_step_hours=lead,
                method="m3_sparse_error_state",
                predicted_dst_nt=candidate_error,
                squared_error_nt2=candidate_error^2,
                absolute_error_nt=abs(candidate_error),
            )))
            for comparator in variable_comparators
                block_adjustments = comparator == "served_inertial_sindy" ?
                    (-1.2, -0.4, 0.4, 1.2) : (1.2, 0.4, -0.4, -1.2)
                adjustment = block_adjustments[month(issue)]
                reference_error = candidate_error + adjustment
                push!(variable_rows, merge(NamedTuple(errors[1, :]), (
                    issue_time_utc=issue,
                    target_time_utc=issue + Hour(lead),
                    model_step_hours=lead,
                    method=comparator,
                    predicted_dst_nt=reference_error,
                    squared_error_nt2=reference_error^2,
                    absolute_error_nt=abs(reference_error),
                )))
            end
        end
        variable = DataFrame(variable_rows)
        variable_bootstrap = static_stack_error_omni_bootstrap(
            variable; draws=500, seed=951,
            comparators=variable_comparators,
        )
        specifications = NamedTuple[]
        for lead in STATIC_STACK_DRIVER_OMNI_LEADS_H,
                comparator in variable_comparators
            candidate = variable[
                (variable.method .== "m3_sparse_error_state") .&
                (variable.model_step_hours .== lead), :]
            reference = variable[
                (variable.method .== comparator) .&
                (variable.model_step_hours .== lead), :]
            blocks = month.(candidate.issue_time_utc)
            push!(specifications, (
                candidate_sums=[sum(candidate.squared_error_nt2[blocks .== m])
                                for m in 1:4],
                reference_sums=[sum(reference.squared_error_nt2[blocks .== m])
                                for m in 1:4],
                counts=[count(==(m), blocks) for m in 1:4],
            ))
        end
        expected_draws = Matrix{Float64}(
            undef, 500, length(specifications),
        )
        reference_rng = MersenneTwister(951)
        for draw in 1:500
            sampled = rand(reference_rng, 1:4, 4)
            for (comparison, specification) in pairs(specifications)
                rows = sum(specification.counts[sampled])
                expected_draws[draw, comparison] = sqrt(
                    sum(specification.reference_sums[sampled]) / rows,
                ) - sqrt(
                    sum(specification.candidate_sums[sampled]) / rows,
                )
            end
        end
        expected_marginal = [
            quantile(@view(expected_draws[:, column]), 0.05)
            for column in axes(expected_draws, 2)
        ]
        expected_simultaneous = quantile(
            vec(minimum(expected_draws; dims=2)), 0.05,
        )
        @test variable_bootstrap.one_sided_95_lower_nt ≈ expected_marginal
        @test all(variable_bootstrap.simultaneous_one_sided_95_lower_nt .≈
                  expected_simultaneous)
        @test length(unique(expected_marginal)) > 1
        @test expected_simultaneous < minimum(expected_marginal)

        metrics = static_stack_error_omni_metrics(errors)
        by_comparison, decision = static_stack_error_omni_decision(
            metrics, bootstrap,
            DataFrame(diagnostic_outcome=[
                "PREDICTABILITY_NOT_SUPPORTED",
            ]),
            scheduled_issues=100,
            m2_eligible_issues=length(issues),
        )
        @test all(by_comparison.available_comparator_diagnostic_pass)
        @test decision.available_comparator_diagnostic_supported[1]
        @test !decision.full_retrospective_contract_pass[1]
        @test !decision.promotion_eligible[1]
        @test !decision.upstream_sparse_driver_supported[1]
        @test !decision.event_block_inference_available[1]
        @test decision.minimum_conditional_readiness_given_driver[1] == 1.0
        @test decision.minimum_effective_scheduled_readiness[1] == 0.04
        @test decision.m2_eligibility[1] == 0.04

        mismatched = copy(errors)
        row = findfirst((mismatched.method .== "served_inertial_sindy") .&
                        (mismatched.model_step_hours .== 1))
        mismatched.issue_time_utc[row] += Hour(1)
        @test_throws ArgumentError static_stack_error_omni_bootstrap(
            mismatched; draws=100, seed=951,
        )
    end
end
