using Test
using SolarSINDy
using DataFrames
using Dates

include(joinpath(
    @__DIR__, "..", "validation", "operational",
    "v2_2_history_crossfit.jl",
))

@testset "V2.2-M1 purged history crossfit helpers" begin
    @testset "post-issue OMNI mutation cannot change the causal driver path" begin
        issue = DateTime(2017, 1, 8)
        latest = OperationalV22HistoryDriver(400.0, -8.0, 2.0, 5.0, 2.0)
        future_a = OperationalV22HistoryDriver(900.0, -40.0, 9.0, 20.0, 20.0)
        future_b = OperationalV22HistoryDriver(250.0, 30.0, -9.0, 1.0, 0.2)
        lookup_a = Dict(issue - Hour(1) => latest)
        lookup_b = Dict(issue - Hour(1) => latest)
        for step in 1:7
            lookup_a[issue + Hour(step - 1)] = future_a
            lookup_b[issue + Hour(step - 1)] = future_b
        end
        causal_a = _v22_history_causal_drivers(lookup_a, issue, -5.0)
        causal_b = _v22_history_causal_drivers(lookup_b, issue, -5.0)
        @test causal_a == causal_b
        tau = min(48.0, 3.0 * (1.0 + 5.0 / 7.5))
        @test causal_a[1].bz_nt ≈ latest.bz_nt * exp(-1.0 / tau)
        @test causal_a[7].by_nt ≈ latest.by_nt * exp(-7.0 / tau)
        @test all(driver -> driver.speed_km_s == latest.speed_km_s, causal_a)
        @test all(driver -> driver.pdyn_npa == latest.pdyn_npa, causal_a)
        @test _v22_history_realized_drivers(lookup_a, issue) !=
              _v22_history_realized_drivers(lookup_b, issue)
    end

    @testset "whole-anchor filter admits exactly the frozen six leads" begin
        issue = DateTime(2017, 1, 8)
        complete = DataFrame(
            issue_time_utc=fill(issue, 6),
            target_time_utc=[issue + Hour(lead) for lead in V22_HISTORY_LEADS_H],
            model_step_hours=collect(V22_HISTORY_LEADS_H),
        )
        incomplete = complete[1:5, :]
        @test length(_v22_history_anchor_rows(complete)) == 1
        @test_throws ErrorException _v22_history_anchor_rows(incomplete)
    end

    @testset "simultaneous block bootstrap uses every lead and comparator" begin
        rows = NamedTuple[]
        epoch = DateTime(2010, 1, 1)
        for block in 0:11, lead in V22_HISTORY_LEADS_H
            issue = epoch + Hour(168 * block)
            values = (
                served_v2_1_dst_nt=1.0,
                frozen_v2_1_dst_nt=1.2,
                raw_sindy_dst_nt=1.4,
                persistence_dst_nt=1.6,
                burton_dst_nt=1.8,
                burton_full_dst_nt=2.0,
                obrien_dst_nt=2.2,
            )
            push!(rows, merge((
                issue_time_utc=issue,
                model_step_hours=lead,
                observation_dst_nt=0.0,
                v2_2_m1_dst_nt=0.0,
            ), values))
        end
        inference = _v22_history_simultaneous_bootstrap(
            DataFrame(rows); replicates=250, seed=22,
        )
        @test inference.blocks == 12
        @test inference.replicates == 250
        @test inference.simultaneous_lower_95_nt ≈ 1.0
        @test all(≈(1.0), inference.per_lead_lower_95_nt)
    end
end

