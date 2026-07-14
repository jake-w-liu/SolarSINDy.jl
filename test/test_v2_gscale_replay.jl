module V2GScaleReplayTests

using Test
using DataFrames
using Dates

const GSCALE_REPLAY_SCRIPT = normpath(joinpath(@__DIR__, "..", "..", "live_forecasts",
                                               "v2_gscale_replay.jl"))

@testset "V2 exact Kp/G-scale replay helpers" begin
    if !isfile(GSCALE_REPLAY_SCRIPT)
        @test_skip "research-workspace G-scale replay script is not present"
    else
        include(GSCALE_REPLAY_SCRIPT)

        @testset "GFZ Kp parsing and NOAA G-scale event selection" begin
            @test noaa_g_level(4.999) == 0
            @test noaa_g_level(5.0) == 1
            @test noaa_g_level(6.0) == 2
            @test noaa_g_level(7.0) == 3
            @test noaa_g_level(8.0) == 4
            @test noaa_g_level(9.0) == 5

            kp = load_gfz_kp()
            @test nrow(kp) > 250_000
            @test minimum(kp.utc) == DateTime(1932, 1, 1)

            may2024 = kp[(kp.utc .>= DateTime(2024, 5, 10)) .&
                         (kp.utc .<= DateTime(2024, 5, 12, 21)), :]
            @test maximum(Float64.(may2024.kp)) == 9.0

            events_g3 = build_gscale_events(kp)
            events_g4 = build_gscale_events(kp; min_kp = 8.0)
            events_g5 = build_gscale_events(kp; min_kp = 9.0)
            @test nrow(events_g3) == 311
            @test nrow(events_g4) == 100
            @test nrow(events_g5) == 10
            @test all(events_g3.replay_start_utc[2:end] .> events_g3.replay_end_utc[1:end-1])
            @test minimum(Float64.(events_g3.peak_kp)) >= 7.0
            @test maximum(Int.(events_g3.peak_g_level)) == 5
        end

        @testset "row CRC catches causality and computes independent RMSE" begin
            toy = DataFrame(g_event_id = [1, 1, 1, 1],
                            storm = fill("toy", 4),
                            g_level = fill(3, 4),
                            peak_kp = fill(7.0, 4),
                            peak_kp_utc = fill(DateTime(2024, 1, 1, 3), 4),
                            n_kp_bins = fill(1, 4),
                            event_start_utc = fill(DateTime(2024, 1, 1, 3), 4),
                            event_end_utc = fill(DateTime(2024, 1, 1, 6), 4),
                            replay_start_utc = fill(DateTime(2024, 1, 1), 4),
                            replay_end_utc = fill(DateTime(2024, 1, 2), 4),
                            issue_utc = fill(DateTime(2024, 1, 1), 4),
                            target_utc = DateTime(2024, 1, 1) .+ Hour.([1, 2, 3, 6]),
                            lead = [1, 2, 3, 6],
                            obs = [-10.0, -20.0, -30.0, -40.0],
                            audit_baseline = [-12.0, -22.0, -32.0, -42.0],
                            v2 = [-11.0, -21.0, -29.0, -39.0],
                            v2_frozen = [-12.0, -22.0, -32.0, -42.0],
                            persistence = [-8.0, -18.0, -33.0, -45.0],
                            rate = [NaN, -1.0, -2.0, -3.0])
            @test _validate_gscale_rows(toy)
            sm = gscale_summary(toy)
            one = sm[(sm.cohort .== "all_G3plus") .& (sm.lead_h .== 1), :][1, :]
            @test one.n_rows == 1
            @test one.n_events == 1
            @test isapprox(one.rmse_v2_nt, 1.0; atol = 1e-12)
            @test isapprox(one.rmse_preupgrade_nt, 2.0; atol = 1e-12)
            @test isapprox(one.rmse_persistence_nt, 2.0; atol = 1e-12)
            @test isapprox(one.improvement_vs_best_nt, 1.0; atol = 1e-12)
            @test isapprox(one.fair_max_abs_nt, 0.0; atol = 1e-12)

            broken = copy(toy)
            broken.target_utc[1] = broken.issue_utc[1]
            @test_throws ErrorException _validate_gscale_rows(broken)
        end
    end
end

end # module V2GScaleReplayTests
