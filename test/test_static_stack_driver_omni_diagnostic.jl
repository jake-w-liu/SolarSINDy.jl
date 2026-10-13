using CSV
using DataFrames
using Dates
using LinearAlgebra
using Random
using SolarSINDy
using Test

include(joinpath(
    @__DIR__, "..", "validation", "operational",
    "static_stack_driver_omni_diagnostic.jl",
))

function _drv_omni_test_raw(rows::Int=60)
    start = DateTime(2010, 1, 1)
    index = collect(1:rows)
    speed = 410.0 .+ 8.0 .* sin.(index ./ 11)
    return DataFrame(
        time_utc=[string(start + Minute(i - 1)) * ".000Z" for i in index],
        percent_interpolated=zeros(Int, rows),
        timeshift_seconds=fill(3_600, rows),
        time_between_observations_seconds=fill(60, rows),
        bx_gse_nt=sin.(index ./ 7),
        by_gsm_nt=cos.(index ./ 9),
        bz_gsm_nt=-2.0 .+ sin.(index ./ 5),
        speed_km_s=speed,
        vx_gse_km_s=-speed,
        density_cm3=5.0 .+ 0.2 .* cos.(index ./ 13),
    )
end

function _drv_omni_test_bins(rows::Int=180; omitted=Int[])
    rng = MersenneTwister(22_022)
    start = DateTime(2012, 1, 1)
    keep = setdiff(collect(1:rows), omitted)
    return DataFrame(
        bin_end_utc=[start + Minute(30 * i) for i in keep],
        Bx_nt=randn(rng, rows)[keep],
        By_nt=randn(rng, rows)[keep],
        Bz_nt=(-2.0 .+ randn(rng, rows))[keep],
        logV=(6.0 .+ 0.08 .* randn(rng, rows))[keep],
        logn=(1.6 .+ 0.15 .* randn(rng, rows))[keep],
        joint_minutes=fill(30, length(keep)),
    )
end

@testset "static stack-M2 OMNI diagnostic" begin
    @testset "strict HAPI parsing and physical aggregation" begin
        raw = _drv_omni_test_raw()
        bins = static_stack_driver_omni_aggregate_month(raw)
        @test nrow(bins) == 2
        @test bins.bin_end_utc == [DateTime(2010, 1, 1, 0, 30),
                                  DateTime(2010, 1, 1, 1, 0)]
        @test bins.joint_minutes == [30, 30]
        @test bins.Bx_nt[1] == median(raw.bx_gse_nt[1:30])
        @test bins.logV[2] == log(median(raw.speed_km_s[31:60]))

        sparse = copy(raw)
        sparse.bz_gsm_nt[31:45] .= 9_999.99
        retained = static_stack_driver_omni_aggregate_month(sparse)
        @test nrow(retained) == 2
        @test retained.joint_minutes == [30, 15]
        sparse.bz_gsm_nt[46] = 9_999.99
        @test nrow(static_stack_driver_omni_aggregate_month(sparse)) == 1

        reordered = raw[:, reverse(names(raw))]
        @test_throws ArgumentError static_stack_driver_omni_aggregate_month(reordered)
        broken_time = copy(raw)
        broken_time.time_utc[10] = broken_time.time_utc[9]
        @test_throws ArgumentError static_stack_driver_omni_aggregate_month(broken_time)
        @test_throws ArgumentError static_stack_driver_omni_aggregate_month(
            raw; minimum_joint_minutes=true,
        )
        @test occursin("time.max=2010-02-01T00:00:00Z",
                       _drv_omni_hapi_url(DateTime(2010, 1, 1)))
        mktemp() do path, io
            write(io, "{\"status\":{\"code\":1411}}\n")
            close(io)
            @test_throws ArgumentError _drv_omni_validate_source_object(
                path, "201001",
            )
        end
    end

    @testset "gaps, complete anchors, and persistence oracle" begin
        gapped = static_stack_driver_omni_series(
            _drv_omni_test_bins(180; omitted=[71]),
        )
        training = static_stack_driver_omni_training_segments(
            gapped, maximum(gapped.times),
        )
        @test training.retained_segments == 2
        @test size.(training.segments, 1) == [70, 109]
        @test training.retained_rows == 179

        complete = static_stack_driver_omni_series(_drv_omni_test_bins())
        issue = complete.times[80]
        case_result = static_stack_driver_omni_cases(complete, issue, issue)
        @test case_result.scheduled == 1
        @test case_result.eligible == 1
        case = only(case_result.cases)
        history = _drv_omni_history(complete, case)
        target = _drv_omni_target(complete, case)
        no_prefix = _drv_omni_prediction(
            :persistence, nothing, complete, case, 0,
        )
        @test no_prefix == repeat(history[end:end, :], 14, 1)
        one_prefix = _drv_omni_prediction(
            :persistence, nothing, complete, case, 1,
        )
        @test one_prefix[1, :] == target[1, :]
        @test one_prefix[2:end, :] == repeat(target[1:1, :], 13, 1)

        strided = static_stack_driver_omni_cases(
            complete, issue, issue + Hour(6); stride_hours=3,
        )
        @test strided.scheduled == 3
        @test_throws ArgumentError static_stack_driver_omni_cases(
            complete, issue, issue; stride_hours=false,
        )
    end

    @testset "post-cutoff mutation invariance" begin
        original = static_stack_driver_omni_series(_drv_omni_test_bins(220))
        cutoff = original.times[140]
        changed_states = copy(original.states)
        changed_states[141:end, :] .+= 1.0e6
        mutated = StaticStackM2OmniSeries(
            copy(original.times), changed_states, copy(original.joint_minutes),
        )
        original_fit = static_stack_driver_omni_training_segments(original, cutoff)
        mutated_fit = static_stack_driver_omni_training_segments(mutated, cutoff)
        @test original_fit.segments == mutated_fit.segments
        first_artifact = _drv_omni_trial_artifact(
            original_fit.segments, 1.0e2, 0.0, "mutation-oracle",
        )
        second_artifact = _drv_omni_trial_artifact(
            mutated_fit.segments, 1.0e2, 0.0, "mutation-oracle",
        )
        @test static_stack_driver_sha256(first_artifact) ==
              static_stack_driver_sha256(second_artifact)
    end

    @testset "one-standard-error selection is deterministic" begin
        audit = DataFrame(
            mean_fourteen_step_standardized_rmse=[1.00, 1.02, 1.04, 1.20],
            fold_standard_error=[0.05, 0.01, 0.01, 0.01],
            max_active_groups=[30, 5, 5, 1],
            ridge=[1.0e-6, 1.0e-4, 1.0e2, 1.0e2],
            threshold=[0.0, 0.1, 0.3, 0.3],
        )
        selected, eligible = _drv_omni_one_se_index(audit, 1:4)
        @test selected == 3
        @test eligible == Set(1:3)
        @test_throws ArgumentError _drv_omni_one_se_index(audit, Int[])
    end

    @testset "paired clustered evidence and non-promotion decision" begin
        @test _drv_omni_block_id(STATIC_STACK_DRIVER_OMNI_GRID_START + Hour(167)) == 0
        @test _drv_omni_block_id(STATIC_STACK_DRIVER_OMNI_GRID_START + Hour(168)) == 1
        @test _drv_omni_elapsed_hours(
            STATIC_STACK_DRIVER_OMNI_GRID_START,
            STATIC_STACK_DRIVER_OMNI_GRID_START + Hour(168),
        ) == 168.0
        rows = NamedTuple[]
        issue_times = [STATIC_STACK_DRIVER_OMNI_GRID_START + Day(7 * i) for i in 0:3]
        for prefix in (0, 1), lead in STATIC_STACK_DRIVER_OMNI_LEADS_H,
                issue in issue_times, method in ("sparse", "dense", "persistence")
            error = method == "sparse" ? 1.0 : method == "dense" ? 4.0 : 9.0
            push!(rows, (
                issue_time_utc=issue,
                known_prefix_steps=prefix,
                method=method,
                lead_hours=lead,
                bz_squared_error_nt2=error,
                ey_squared_error_mvm2=error,
            ))
        end
        errors = DataFrame(rows)
        evidence = static_stack_driver_omni_bootstrap(errors; draws=100, seed=4404)
        @test nrow(evidence) == 48
        @test all(evidence.point_improvement .> 0.0)
        @test all(evidence.one_sided_95_lower .> 0.0)

        mismatched = copy(errors)
        row = findfirst(mismatched.method .== "dense")
        mismatched.issue_time_utc[row] += Hour(1)
        @test_throws ArgumentError static_stack_driver_omni_bootstrap(
            mismatched; draws=100, seed=4404,
        )

        metric_rows = NamedTuple[]
        for prefix in (0, 1), lead in STATIC_STACK_DRIVER_OMNI_LEADS_H,
                method in ("sparse", "dense", "persistence")
            value = method == "sparse" ? 1.0 : method == "dense" ? 2.0 : 3.0
            push!(metric_rows, (
                known_prefix_steps=prefix,
                method=method,
                lead_hours=lead,
                bz_rmse_nt=value,
                ey_rmse_mvm=value,
            ))
        end
        by_lead, decision = static_stack_driver_omni_decision(
            DataFrame(metric_rows), evidence,
        )
        @test all(by_lead.driver_gate_pass)
        @test decision.diagnostic_outcome[1] ==
              "PREDICTABILITY_SUPPORTED_WITHOUT_PREFIX"
        @test !decision.promotion_eligible[1]
        @test !decision.receipt_causal[1]

        weakened = copy(evidence)
        weakened.one_sided_95_lower[
            (weakened.known_prefix_steps .== 0) .&
            (weakened.lead_hours .== 7)
        ] .= -0.1
        _, oracle_only = static_stack_driver_omni_decision(
            DataFrame(metric_rows), weakened,
        )
        @test oracle_only.diagnostic_outcome[1] ==
              "PREDICTABILITY_SUPPORTED_ONLY_WITH_ORACLE_PREFIX"
    end
end
