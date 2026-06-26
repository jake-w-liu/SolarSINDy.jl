using Test
using DataFrames
using Dates

include(joinpath(@__DIR__, "..", "..", "live_forecasts", "v2_broad_replay.jl"))

@testset "V2 broad historical replay helpers" begin
    @testset "catalog threshold selection has fixed independent counts" begin
        cat = load_storm_catalog()
        @test nrow(cat) == 741
        @test nrow(select_broad_storms(cat; threshold_nt = -100.0)) == 204
        @test nrow(select_broad_storms(cat; threshold_nt = -200.0)) == 32
        @test nrow(select_broad_storms(cat; threshold_nt = -300.0)) == 8
        @test nrow(select_broad_storms(cat; threshold_nt = -100.0, split = "test")) == 23

        may2024 = cat[Int.(cat.storm_id) .== 710, :][1, :]
        storm = _storm_from_row(may2024)
        @test storm.t0 == DateTime(2024, 5, 10, 18)
        @test storm.min_dst <= -400.0
    end

    @testset "row CRC catches causality and computes independent RMSE" begin
        toy = DataFrame(storm_id = [1, 1, 1, 1],
                        storm = fill("toy", 4),
                        storm_split = fill("test", 4),
                        storm_solar_cycle = fill(25, 4),
                        storm_min_dst_nt = fill(-120.0, 4),
                        storm_onset_utc = fill(DateTime(2024, 1, 1), 4),
                        storm_min_dst_utc = fill(DateTime(2024, 1, 1, 3), 4),
                        storm_recovery_end_utc = fill(DateTime(2024, 1, 2), 4),
                        issue_utc = fill(DateTime(2024, 1, 1), 4),
                        target_utc = DateTime(2024, 1, 1) .+ Hour.([1, 2, 3, 6]),
                        lead = [1, 2, 3, 6],
                        obs = [-10.0, -20.0, -30.0, -40.0],
                        audit_baseline = [-12.0, -22.0, -32.0, -42.0],
                        v2 = [-11.0, -21.0, -29.0, -39.0],
                        v2_frozen = [-12.0, -22.0, -32.0, -42.0],
                        persistence = [-8.0, -18.0, -33.0, -45.0],
                        rate = [NaN, -1.0, -2.0, -3.0])
        @test _validate_broad_rows(toy)
        sm = broad_summary(toy; threshold_nt = -100.0)
        one = sm[(sm.cohort .== "all") .& (sm.lead_h .== 1), :][1, :]
        @test one.n_rows == 1
        @test one.n_storms == 1
        @test isapprox(one.rmse_v2_nt, 1.0; atol = 1e-12)
        @test isapprox(one.rmse_preupgrade_nt, 2.0; atol = 1e-12)
        @test isapprox(one.rmse_persistence_nt, 2.0; atol = 1e-12)
        @test isapprox(one.improvement_vs_best_nt, 1.0; atol = 1e-12)
        @test isapprox(one.fair_max_abs_nt, 0.0; atol = 1e-12)

        broken = copy(toy)
        broken.target_utc[1] = broken.issue_utc[1]
        @test_throws ErrorException _validate_broad_rows(broken)
    end
end
