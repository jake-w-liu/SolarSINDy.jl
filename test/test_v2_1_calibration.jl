module V21CalibrationTests

using Test
using SolarSINDy
using DataFrames
using Dates

include(joinpath(@__DIR__, "..", "validation", "operational", "v2_1_calibration.jl"))

@testset "Operational V2.1 calibration table" begin
    t0 = DateTime(2020, 1, 1)
    times = t0 .+ Hour.(0:12)
    plasma = DataFrame(
        time_tag=times,
        speed=fill(450.0, length(times)),
        density=fill(6.0, length(times)),
    )
    mag = DataFrame(
        time_tag=times,
        bz_gsm=fill(-5.0, length(times)),
        by_gsm=fill(2.0, length(times)),
    )
    dst = collect(-20.0:-1.0:-32.0)
    horizons = [1, 2, 3, 6]
    table = build_v2_1_calibration_table(
        plasma, mag, times, dst; horizons=horizons,
    )

    @test nrow(table) == sum(12 - h for h in horizons)
    @test all(table.operational_core_version .== "v2.1")
    @test all(table.operational_candidate_count .== 20)
    @test all(table.operational_active_count .== 11)
    @test all(table.source_driver_end_utc .== table.issue_time_utc)
    @test all(table.target_time_utc .> table.issue_time_utc)
    # Pin the two construction equations directly. Subtracting the rounded
    # endpoints can differ from 2.0 by a few ulps even when both endpoints are
    # exactly the values produced by the declared ±1 nT seed band.
    @test table.pred_dst_ci05_nt == table.pred_dst_nt .- V21_SEED_HALF_WIDTH_NT
    @test table.pred_dst_ci95_nt == table.pred_dst_nt .+ V21_SEED_HALF_WIDTH_NT

    # Independent engine oracle for the first row.
    row = table[1, :]
    driver = (
        V=row.V_kms, Bz=row.Bz_nt, By=row.By_nt,
        n=row.n_cm3, Pdyn=row.Pdyn_npa,
    )
    direct = _forecast_one_replay(
        row.issue_time_utc,
        row.target_time_utc,
        row.latest_dst_nt,
        driver;
        n_steps=row.model_step_hours,
        core_version=OPERATIONAL_V2_1_MODEL_VERSION,
    )
    @test row.pred_dst_nt == direct.pred_dst

    # A V2.0 core cannot be mislabeled as a V2.1 calibration source.
    old = load_operational_core(:v2_0)
    @test_throws ArgumentError build_v2_1_calibration_table(
        plasma, mag, times, dst; horizons=[1], core=old,
    )

    # The split audit is forecast-origin causal: every later issue must occur
    # strictly after the preceding split's latest target observation.
    causal = DataFrame(
        v2_split=["fit", "validation", "holdout"],
        issue_time_utc=[t0, t0 + Hour(3), t0 + Hour(6)],
        target_time_utc=[t0 + Hour(2), t0 + Hour(5), t0 + Hour(7)],
        model_step_hours=[2, 2, 1],
    )
    audited = _v21_split_audit(causal)
    @test audited.maximum_target_utc[1] < audited.minimum_issue_utc[2]
    @test audited.maximum_target_utc[2] < audited.minimum_issue_utc[3]

    leaky = copy(causal)
    leaky.issue_time_utc[2] = t0 + Hour(1)
    # Target ranges remain ordered, so this mutation specifically detects the
    # old target-to-target check that missed forecast-origin leakage.
    @test leaky.target_time_utc[1] < leaky.target_time_utc[2]
    @test_throws ErrorException _v21_split_audit(leaky)
end

end
