@testset "V2.2 recoverability probe helpers" begin
    include(joinpath(
        @__DIR__, "..", "validation", "operational",
        "v2_2_recoverability_probe.jl",
    ))

    issues = [DateTime(2020, 1, 1) + Hour(hour) for hour in 0:6]
    rows = NamedTuple[]
    for (index, issue) in enumerate(issues), lead in (1, 2)
        observation = 10.0 + index + lead
        oracle = observation - index
        push!(rows, (
            issue_time_utc=issue,
            target_time_utc=issue + Hour(lead),
            model_step_hours=lead,
            observation_dst_nt=observation,
            noncausal_input_oracle_dst_nt=oracle,
        ))
    end
    table = DataFrame(rows)
    enriched = _v22_recoverability_add_oracle_lags(table; lags_h=(1, 2))
    @test minimum(enriched.issue_time_utc) == issues[3]
    @test nrow(enriched) == 2 * (length(issues) - 2)
    probe_row = only(eachrow(enriched[
        (enriched.issue_time_utc .== issues[4]) .&
        (enriched.model_step_hours .== 2),
        :,
    ]))
    @test probe_row.oracle_h1_innovation_nt_lag_1h == 3.0
    @test probe_row.oracle_h1_innovation_nt_lag_2h == 2.0

    mutated = copy(table)
    future = (mutated.issue_time_utc .== issues[5]) .&
             (mutated.model_step_hours .== 1)
    mutated.observation_dst_nt[future] .+= 10_000.0
    enriched_mutated = _v22_recoverability_add_oracle_lags(
        mutated; lags_h=(1, 2),
    )
    mutation_row = only(eachrow(enriched_mutated[
        (enriched_mutated.issue_time_utc .== issues[4]) .&
        (enriched_mutated.model_step_hours .== 2),
        :,
    ]))
    @test mutation_row.oracle_h1_innovation_nt_lag_1h ==
          probe_row.oracle_h1_innovation_nt_lag_1h
    @test mutation_row.oracle_h1_innovation_nt_lag_2h ==
          probe_row.oracle_h1_innovation_nt_lag_2h

    duplicate = vcat(table, table[1:1, :])
    @test_throws ErrorException _v22_recoverability_add_oracle_lags(
        duplicate; lags_h=(1, 2),
    )
    @test_throws ArgumentError _v22_recoverability_add_oracle_lags(
        table; lags_h=(0, 1),
    )
    @test_throws ArgumentError _v22_recoverability_add_oracle_lags(
        table; lags_h=(1, 1),
    )

    post_2022 = copy(table)
    post_2022.target_time_utc .= DateTime(2023, 1, 1)
    @test_throws ErrorException _v22_recoverability_add_oracle_lags(
        post_2022; lags_h=(1, 2),
    )

    nonfinite = copy(table)
    nonfinite.noncausal_input_oracle_dst_nt[1] = Inf
    @test_throws ErrorException _v22_recoverability_add_oracle_lags(
        nonfinite; lags_h=(1, 2),
    )

    bootstrap_issues = [DateTime(2020, 2, 1) + Hour(hour) for hour in 0:7]
    bootstrap_observed = zeros(length(bootstrap_issues))
    bootstrap_results = [
        (
            bootstrap_payload=(
                issues=bootstrap_issues,
                observed=bootstrap_observed,
                candidate=fill(1.0, length(bootstrap_issues)),
                comparator=fill(2.0, length(bootstrap_issues)),
            ),
        ),
        (
            bootstrap_payload=(
                issues=bootstrap_issues,
                observed=bootstrap_observed,
                candidate=fill(2.0, length(bootstrap_issues)),
                comparator=fill(3.0, length(bootstrap_issues)),
            ),
        ),
    ]
    inference = _v22_recoverability_bootstrap(
        bootstrap_results; replicates=100, seed=7, block_hours=2,
    )
    @test inference.per_lead_lower_95_nt == [1.0, 1.0]
    @test inference.simultaneous_lower_95_nt == 1.0
    @test inference.blocks == 4
    @test inference.replicates == 100
    @test _v22_recoverability_bootstrap(
        bootstrap_results; replicates=100, seed=7, block_hours=2,
    ) == inference
    mismatched = copy(bootstrap_results)
    mismatched[2] = (
        bootstrap_payload=merge(
            bootstrap_results[2].bootstrap_payload,
            (issues=reverse(bootstrap_issues),),
        ),
    )
    @test_throws ErrorException _v22_recoverability_bootstrap(
        mismatched; replicates=10, seed=7, block_hours=2,
    )
end
