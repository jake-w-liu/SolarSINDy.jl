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
    horizons = copy(OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS)
    table = build_v2_1_calibration_table(
        plasma, mag, times, dst; horizons=horizons,
    )

    @test nrow(table) == sum(12 - h for h in horizons)
    @test all(table.operational_core_version .== "v2.1")
    @test all(table.operational_candidate_count .== 20)
    @test all(table.operational_active_count .== 11)
    @test all(table.source_driver_end_utc .== table.issue_time_utc)
    @test all(table.target_time_utc .> table.issue_time_utc)
    @test sort(unique(table.model_step_hours)) == OPERATIONAL_V2_1_SUPPORTED_MODEL_STEPS
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

    # The validation -> holdout boundary is a separate comparison from the
    # fit -> validation one, and only a holdout that leaks past validation while
    # staying clear of fit can tell them apart: an index copied from the first
    # comparison into the second would accept this frame. The holdout is issued
    # one hour before the last validation target is observable, so it is trained
    # on an outcome it could not have known, while validation itself stays
    # causal and the three issue-time sets remain disjoint.
    leaky_holdout = copy(causal)
    leaky_holdout.issue_time_utc[3] = t0 + Hour(4)
    leaky_holdout.target_time_utc[3] = t0 + Hour(5)
    leaky_holdout.model_step_hours[3] = 1
    @test leaky_holdout.issue_time_utc[2] > leaky_holdout.target_time_utc[1]   # validation causal
    @test leaky_holdout.issue_time_utc[3] > leaky_holdout.target_time_utc[1]   # clear of fit
    @test leaky_holdout.issue_time_utc[3] < leaky_holdout.target_time_utc[2]   # leaks validation
    @test length(unique(leaky_holdout.issue_time_utc)) == 3
    @test_throws ErrorException _v21_split_audit(leaky_holdout)
    # The refusal must name the boundary that was crossed, so that a check
    # comparing the wrong split cannot pass this test by throwing for another
    # reason.
    holdout_message = try
        _v21_split_audit(leaky_holdout)
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("holdout issuance begins before all validation targets", holdout_message)
    fit_message = try
        _v21_split_audit(leaky)
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("validation issuance begins before all fit targets", fit_message)

    # The boundary is strict: issuing exactly at the preceding split's last
    # observable target is still leakage, because that target is observed at the
    # end of the hour the forecast is issued in.
    boundary = copy(causal)
    boundary.issue_time_utc[3] = boundary.target_time_utc[2]
    @test_throws ErrorException _v21_split_audit(boundary)
end

end
