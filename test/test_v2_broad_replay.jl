module V2BroadReplayTests

using Test
using DataFrames
using Dates

const BROAD_REPLAY_SCRIPT = normpath(joinpath(@__DIR__, "..", "validation", "operational",
                                              "v2_broad_replay.jl"))
include(BROAD_REPLAY_SCRIPT)

@testset "V2 broad historical replay helpers" begin
    @testset "catalog threshold selection has fixed independent counts" begin
        cat = load_storm_catalog()
        @test nrow(cat) == 714
        @test names(cat) == ["storm_id", "onset_time", "min_dst_star",
                             "min_dst_star_time", "recovery_end_time", "duration_hr",
                             "solar_cycle", "split", "onset_idx", "end_idx"]
        @test nrow(select_broad_storms(cat; threshold_dst_star_nt = -100.0)) == 193
        @test nrow(select_broad_storms(cat; threshold_dst_star_nt = -200.0)) == 30
        @test nrow(select_broad_storms(cat; threshold_dst_star_nt = -300.0)) == 8
        @test nrow(select_broad_storms(cat; threshold_dst_star_nt = -100.0, split = "test")) == 23

        may2024 = cat[Int.(cat.storm_id) .== 684, :][1, :]
        storm = _storm_from_row(may2024)
        @test storm.storm_id == 684
        @test storm.t0 == DateTime(2024, 5, 10, 18)
        @test storm.min_dst_star <= -400.0
        @test occursin("min Dst*", storm.name)
        @test !hasproperty(storm, :min_dst)

        legacy = select(cat, Not([:min_dst_star, :min_dst_star_time, :onset_idx, :end_idx]))
        legacy[!, :min_dst] = cat.min_dst_star
        legacy[!, :min_dst_time] = cat.min_dst_star_time
        mktempdir() do directory
            path = joinpath(directory, "legacy_catalog.csv")
            CSV.write(path, legacy)
            @test_throws ErrorException load_storm_catalog(path)
        end
    end

    @testset "row CRC catches causality and computes independent RMSE" begin
        toy = DataFrame(storm_id = [1, 1, 1, 1],
                        storm = fill("toy", 4),
                        storm_split = fill("test", 4),
                        storm_solar_cycle = fill(25, 4),
                        storm_min_dst_star_nt = fill(-120.0, 4),
                        storm_onset_utc = fill(DateTime(2024, 1, 1), 4),
                        storm_min_dst_star_utc = fill(DateTime(2024, 1, 1, 3), 4),
                        storm_recovery_end_utc = fill(DateTime(2024, 1, 2), 4),
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
        @test _validate_broad_rows(toy)
        sm = broad_summary(toy; threshold_dst_star_nt = -100.0)
        one = sm[(sm.cohort .== "all") .& (sm.lead_h .== 1), :][1, :]
        @test one.n_rows == 1
        @test one.n_storms == 1
        @test isapprox(one.rmse_v2_1_nt, 1.0; atol = 1e-12)
        @test isapprox(one.rmse_v2_0_nt, 2.0; atol = 1e-12)
        @test isapprox(one.rmse_persistence_nt, 2.0; atol = 1e-12)
        @test isapprox(one.improvement_vs_best_nt, 1.0; atol = 1e-12)
        @test isapprox(one.max_tail_effect_nt, 0.5; atol = 1e-12)
        @test isapprox(one.max_core_change_nt, 0.5; atol = 1e-12)
        @test one.threshold_dst_star_nt == -100.0
        @test :threshold_nt ∉ propertynames(sm)
        @test any(startswith.(String.(sm.cohort), "storm_min_dst_star<="))
        @test !any(startswith.(String.(sm.cohort), "storm_min_dst<="))

        legacy_labeled = rename(copy(toy), :storm_min_dst_star_nt => :storm_min_dst_nt)
        @test_throws ErrorException _validate_broad_rows(legacy_labeled)

        broken = copy(toy)
        broken.target_utc[1] = broken.issue_utc[1]
        @test_throws ErrorException _validate_broad_rows(broken)
    end

    source = read(BROAD_REPLAY_SCRIPT, String)
    @test !occursin("storm_min_dst_nt", source)
    @test !occursin("storm_min_dst_utc", source)
    @test !occursin("min Dst <=", source)
end

@testset "V2 forecast-layer oracles (_v2_forecast)" begin
    # The forecast-layer oracles ship with the package replay scripts
    # (v2_broad_replay.jl -> v2_replay.jl) and deployed calibration.
    # Wire the shipped behavioral oracles (continuity to the primary V2.1 path,
    # regime awareness, recovery relaxation, near-term extreme inertia) into Pkg.test.
    @test _selftest_v2()

    lib, ξ0, _ = _shadow_library()
    cal = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
    )
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

    # RATE GAP-GUARD neutrality — a non-finite recent rate is normalized to the
    # documented zero fallback before both the tail and every safeguard. Probe
    # several states because a quiet-state inertia rule can otherwise expose a
    # difference even when the tail itself is neutral.
    for state in (-5.0, -25.0, -60.0, -148.0), h in (1, 2, 3, 6)
        @test _v2_forecast(lib, ξ0, anchor, slow, _ -> nothing, state, cal, h, NaN) ==
              _v2_forecast(lib, ξ0, anchor, slow, _ -> nothing, state, cal, h, 0.0)
    end

    # Every causally filled replay tuple must preserve the exact proton-only
    # dynamic-pressure identity. V, density, and the source Pdyn column can have
    # different missingness, so pressure may not be forward-filled independently.
    for lookup in (_driver_lookup_range(2019, 2022), _driver_lookup(2022))
        @test !isempty(lookup)
        @test all(values(lookup)) do driver
            driver.Pdyn == dynamic_pressure(driver.n, driver.V)
        end
    end

    blended = _blend(0.4, fut_slow, slow)
    @test blended.Pdyn == dynamic_pressure(blended.n, blended.V)
end

end # module V2BroadReplayTests
