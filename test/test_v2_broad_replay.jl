module V2BroadReplayTests

using Test
using DataFrames
using Dates

const BROAD_REPLAY_SCRIPT = normpath(joinpath(@__DIR__, "..", "..", "live_forecasts",
                                              "v2_broad_replay.jl"))

@testset "V2 broad historical replay helpers" begin
    if !isfile(BROAD_REPLAY_SCRIPT)
        @test_skip "research-workspace broad replay script is not present"
    else
        include(BROAD_REPLAY_SCRIPT)

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
end

@testset "V2 forecast-layer oracles (_v2_forecast)" begin
    # The forecast-layer oracles live in the research-workspace replay scripts
    # (v2_broad_replay.jl -> v2_replay.jl), alongside the workspace calibration.
    if !isfile(BROAD_REPLAY_SCRIPT)
        @test_skip "research-workspace broad replay script is not present"
    else
    # Wire the shipped behavioral oracles (continuity to the pre-upgrade baseline,
    # regime awareness, recovery relaxation, near-term extreme inertia) into Pkg.test.
    @test _selftest_v2()

    lib, ξ0, _ = _shadow_library()
    cal = _load_calibration_for_model(LiveVerifyConfig(model = :v2))
    anchor = -150.0; latest = -148.0
    slow = (V = 300.0, Bz = -10.0, By = 1.0, n = 6.0, Pdyn = 2.0)      # Δ≈1.39 h ⇒ kΔ=1 (look-ahead fires at k=1)
    fast = (V = 800.0, Bz = -10.0, By = 1.0, n = 6.0, Pdyn = 2.0)      # Δ≈0.52 h ⇒ kΔ=0 (no look-ahead)
    fut_slow = (V = 320.0, Bz = -22.0, By = 0.0, n = 8.0, Pdyn = 3.0)  # slow, stronger southward ⇒ admissible at k=1
    fut_fast = (V = 800.0, Bz = -30.0, By = 0.0, n = 9.0, Pdyn = 4.0)  # fast (accelerated) ⇒ transit<1 h ⇒ rejected

    # CAUSALITY — fast wind (kΔ=0): no L1-known window, so the forecast is invariant to the future
    # closure. Shifting/replacing future drivers must not change the forecast.
    for h in (1, 2, 3, 6)
        f_a = _v2_forecast(lib, ξ0, anchor, fast, _ -> fut_slow, latest, cal, h, -5.0)
        f_b = _v2_forecast(lib, ξ0, anchor, fast, _ -> fut_fast, latest, cal, h, -5.0)
        f_0 = _v2_forecast(lib, ξ0, anchor, fast, _ -> nothing,  latest, cal, h, -5.0)
        @test f_a == f_b == f_0
    end

    # ACCELERATION GUARD — slow issue wind (kΔ=1) but the arrival-hour record is FAST (transit<1 h):
    # it left L1 after issue, so it is rejected and the forecast equals the no-look-ahead (frozen) case.
    for h in (1, 3, 6)
        leaked = _v2_forecast(lib, ξ0, anchor, slow, _ -> fut_fast, latest, cal, h, -5.0)
        frozen = _v2_forecast(lib, ξ0, anchor, slow, _ -> nothing,  latest, cal, h, -5.0)
        @test leaked == frozen
    end

    # ADMISSION — slow issue wind with a SLOW arrival record (transit≥1 h): the record is L1-known at
    # k=1, so a stronger incoming southward driver deepens the forecast below the no-look-ahead case.
    la1 = _v2_forecast(lib, ξ0, anchor, slow, _ -> fut_slow, latest, cal, 1, -5.0)[1]
    fr1 = _v2_forecast(lib, ξ0, anchor, slow, _ -> nothing,  latest, cal, 1, -5.0)[1]
    @test la1 < fr1

    # PERSISTENCE-LIMIT — once observed Dst is already in the extreme core, 1–2 h forecasts serve
    # persistence exactly; longer leads retain the model tail.
    @test _v2_forecast(lib, ξ0, -250.0, slow, _ -> nothing, -250.0, cal, 1, +10.0)[2] == -250.0
    @test _v2_forecast(lib, ξ0, -250.0, slow, _ -> nothing, -250.0, cal, 2, +10.0)[2] == -250.0
    @test _v2_forecast(lib, ξ0, -250.0, slow, _ -> nothing, -250.0, cal, 3, +10.0)[2] != -250.0

    # RATE GAP-GUARD neutrality — a NaN recent rate (the replay sentinel for a >1 h Dst gap) is neutral,
    # identical to a zero rate, so a gap-misread multi-hour delta can never lengthen the relaxation tail.
    for h in (1, 3, 6)
        @test _v2_forecast(lib, ξ0, anchor, slow, _ -> nothing, latest, cal, h, NaN) ==
              _v2_forecast(lib, ξ0, anchor, slow, _ -> nothing, latest, cal, h, 0.0)
    end
    end
end

end # module V2BroadReplayTests
