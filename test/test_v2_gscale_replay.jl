module V2GScaleReplayTests

using Test
using DataFrames
using Dates

const GSCALE_REPLAY_SCRIPT = normpath(joinpath(@__DIR__, "..", "validation", "operational",
                                               "v2_gscale_replay.jl"))
include(GSCALE_REPLAY_SCRIPT)

@testset "V2 exact Kp/G-scale replay helpers" begin
        @testset "GFZ Kp parsing and NOAA G-scale event selection" begin
            # SWPC mapping (owner decision V11-02): a Kp third belongs to its integer class
            # (SWPC prints "4.67 (G1)"); G4 is "Kp = 8, including a 9-"; G5 is 9o only.
            # The former integer pins (noaa_g_level(4.999) == 0 etc.) encoded Kp >= 5o/6o/...,
            # which contradicts SWPC, and were replaced by these thirds-scale pins.
            @test noaa_g_level(5.0) == 1
            @test noaa_g_level(6.0) == 2
            @test noaa_g_level(7.0) == 3
            @test noaa_g_level(8.0) == 4
            @test noaa_g_level(9.0) == 5
            # GFZ thirds as stored in the GFZ file (3 decimals) and SWPC's 2-decimal print.
            @test noaa_g_level(4.333) == 0
            @test noaa_g_level(4.667) == 1
            @test noaa_g_level(4.67) == 1
            @test noaa_g_level(5.667) == 2
            @test noaa_g_level(6.333) == 2
            @test noaa_g_level(6.667) == 3
            @test noaa_g_level(7.667) == 4
            @test noaa_g_level(8.667) == 4
            @test noaa_g_level(8.67) == 4
            @test noaa_g_level(0.0) == 0

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
            # Selection stays GFZ Kp >= 7o (311 events, supervisor Option B); the G level of
            # each peak follows SWPC, so the 49 events peaking at 7.667 (8-) are G4, not G3.
            @test all(Int.(events_g3.peak_g_level) .== noaa_g_level.(Float64.(events_g3.peak_kp)))
            @test count(==(3), Int.(events_g3.peak_g_level)) == 163
            @test count(==(4), Int.(events_g3.peak_g_level)) == 139
            @test count(==(5), Int.(events_g3.peak_g_level)) == 9
            # Human-readable selection label no longer claims the cohort is NOAA G3+.
            @test occursin("7o", _gscale_selection_text(GSCALE_MIN_KP))
            @test occursin("7- (6.67)", _gscale_selection_text(GSCALE_MIN_KP))
            @test occursin("G4 = 8- through 9-", NOAA_G_SCALE_NOTE)
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
                            v2_1 = [-11.0, -21.0, -29.0, -39.0],
                            v2_1_pre_rate_guard = [-11.0, -21.0, -29.0, -39.0],
                            v2_1_pre_one_hour_inertia = [-11.0, -21.0, -29.0, -39.0],
                            v2_1_pre_state_inertia = [-11.0, -21.0, -29.0, -39.0],
                            v2_0 = [-12.0, -22.0, -32.0, -42.0],
                            v2_1_frozen = [-11.5, -21.5, -31.5, -41.5],
                            persistence = [-8.0, -18.0, -33.0, -45.0],
                            rate = [NaN, -1.0, -2.0, -3.0])
            @test _validate_gscale_rows(toy)
            sm = gscale_summary(toy)
            one = sm[(sm.cohort .== "all_G3plus") .& (sm.lead_h .== 1), :][1, :]
            @test one.n_rows == 1
            @test one.n_events == 1
            @test isapprox(one.rmse_v2_1_nt, 1.0; atol = 1e-12)
            @test isapprox(one.rmse_v2_0_nt, 2.0; atol = 1e-12)
            @test isapprox(one.rmse_persistence_nt, 2.0; atol = 1e-12)
            @test isapprox(one.improvement_vs_best_nt, 1.0; atol = 1e-12)
            @test isapprox(one.max_tail_effect_nt, 0.5; atol = 1e-12)
            @test isapprox(one.max_core_change_nt, 0.5; atol = 1e-12)

            broken = copy(toy)
            broken.target_utc[1] = broken.issue_utc[1]
            @test_throws ErrorException _validate_gscale_rows(broken)
        end
end

end # module V2GScaleReplayTests
