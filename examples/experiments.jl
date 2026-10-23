#!/usr/bin/env julia

using Dates
using DataFrames
using SolarSINDy

include(joinpath(@__DIR__, "live_forecast_verify.jl"))
include(joinpath(@__DIR__, "..", "validation", "operational", "superlearner_interval_upgrade.jl"))
include(joinpath(@__DIR__, "..", "validation", "operational", "superlearner_point_upgrade.jl"))
include(joinpath(@__DIR__, "..", "validation", "operational", "superlearner_point_evaluation.jl"))
include(joinpath(@__DIR__, "..", "validation", "operational", "ground_delay_check.jl"))

function main()
    ground_model = (;beta=[log(6.),0,0,0,0],mu=zeros(4),sigma=ones(4),
        calibration_seed=ones(1440),climatology_native=3.,climatology_log=2.)
    ground_times = collect(DateTime(2020):Minute(1):DateTime(2020)+Minute(99))
    ground = GroundDelayCheck.evaluate_delay(ground_times,Float64.(1:100),ground_model,10)
    @assert first(ground.target) == 71
    @assert first(ground.target_end) == DateTime(2020)+Minute(70)
    @assert isapprox(first(ground.prediction),5;atol=1e-14)
    @assert GroundDelayCheck.metrics(ground,ground_model).n == 30

    cal = default_inertial_sindy_calibration()
    latest_dst = -80.0
    drivers = (V=420.0, Bz=-10.0, By=3.0, n=6.0, Pdyn=2.0)
    sindy_core_pred = -95.0
    v1_ci05 = -115.0
    v1_ci95 = -75.0
    baselines = (persistence=latest_dst, burton=-90.0, burton_full=-92.0, obrien=-98.0)

    selected = _select_model_prediction(
        :inertial_sindy,
        cal,
        latest_dst,
        drivers,
        sindy_core_pred,
        v1_ci05,
        v1_ci95;
        baselines=baselines,
    )
    @assert selected.model_version == OPERATIONAL_INERTIAL_SINDY_MODEL_VERSION
    @assert selected.inertial_sindy_correction == 0.0
    @assert selected.inertial_sindy_pred_dst == sindy_core_pred
    @assert selected.operational_ci05_dst == v1_ci05
    @assert selected.operational_ci95_dst == v1_ci95

    @assert _operational_tail_tau(-30.0) > _operational_tail_tau(5.0)
    @assert _rapid_deepening_projection_guard(-100.0, -80.0, 6, -20.0) == -125.0
    @assert _one_hour_inertia_blend(-120.0, -100.0, 1) == -115.0
    @assert _one_hour_inertia_blend(-120.0, -100.0, 2) == -120.0
    @assert _near_term_extreme_inertia_guard(-250.0, 2)
    @assert !_near_term_extreme_inertia_guard(-250.0, 3)
    recovery = _relaxed_tail_driver(drivers, 1, 5.0)
    deepening = _relaxed_tail_driver(drivers, 1, -30.0)
    @assert abs(recovery.Bz) < abs(drivers.Bz)
    @assert abs(deepening.Bz) > abs(recovery.Bz)

    lo, hi = _shift_interval_to_center(-100.0, -95.0, -115.0, -75.0)
    @assert (lo, hi) == (-120.0, -80.0)

    t0 = DateTime(2026, 1, 1)
    plasma = DataFrame(time_tag=[t0 + Minute(5)], speed=[410.0], density=[5.0])
    mag = DataFrame(time_tag=[t0 + Minute(5)], bz_gsm=[-3.0], by_gsm=[1.0])
    s = _subhourly_driver_with_status(plasma, mag, t0 + Hour(2), drivers, t0)
    @assert !s.l1_measured
    @assert s.driver == drivers

    core = load_operational_core()
    @assert length(core.library) == 20
    @assert count(!=(0.0), core.coefficients) == 11
    @assert OPERATIONAL_SERVED_TAIL_VERSION ==
            "inertial-sindy+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"
    # static stack predecessor: the inertial-SINDy operator followed by the fitted static regime stack.
    @assert STATIC_STACK_SERVED_TAIL_VERSION ==
            "staticstack+sindy60_fit407598+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"
    # Served pipeline: the super-learner over the ten experts, the static stack among them.
    @assert SUPERLEARNER_SERVED_TAIL_VERSION ==
            "dst-superlearner+sindyfloor+conformal"
    @assert static_stack_serving_coupling_active(2.5, -4.0) == 2.5
    @assert static_stack_serving_coupling_active(2.5, 1.0) == 0.0
    @assert static_stack_serving_coupling_active(0.0, -4.0) == 0.0
    @assert static_stack_serving_depth_safe_center(-90.0, -120.0) == -120.0
    @assert static_stack_serving_depth_safe_center(-150.0, -120.0) == -150.0
    # Published severity is the deepest of the served center and both continuity partners.
    @assert superlearner_serving_depth_safe_center(-90.0, -120.0, -60.0) == -120.0
    @assert superlearner_serving_depth_safe_center(-150.0, -120.0, -60.0) == -150.0
    @assert superlearner_serving_depth_safe_center(-90.0, NaN, -95.0) == -95.0
    # The published watch edge is the same rule on the edges themselves, so a narrower served band
    # cannot lower a watch tier a predecessor's own edge would have raised.
    @assert superlearner_serving_depth_safe_center(-92.0, -100.0, -105.0) == -105.0
    @assert superlearner_serving_depth_safe_center(-160.0, -100.0, -105.0) == -160.0
    @assert superlearner_serving_depth_safe_center(-92.0, NaN, -105.0) == -105.0
    # Depth bins and the deepening cell of the super-learner guard, on their closed-from-below edges.
    @assert superlearner_serving_depth_bin(-29.9) === :shallow
    @assert superlearner_serving_depth_bin(-30.0) === :moderate
    @assert superlearner_serving_depth_bin(-70.0) === :deep
    @assert superlearner_serving_deepening(-16.0, 0.0, 0.0)
    @assert !superlearner_serving_deepening(-14.0, 0.0, 0.0)
    @assert superlearner_serving_deepening(-1.0, 2.0, -50.0)
    @assert !superlearner_serving_deepening(-1.0, 0.0, -50.0)
    @assert superlearner_serving_guard(-80.0, -60.0, -20.0, 0.0, -10.0) == -80.0
    @assert superlearner_serving_guard(-80.0, -60.0, -1.0, 0.0, -10.0) == -60.0

    serve_deployed_superlearner_row()

    # The development-only interval layer keeps the point at zero. Hand values:
    # median(2,4)=3; static half-widths 2/4; the standardized upper score is 2.
    translated = SuperlearnerIntervalUpgrade.candidate_interval(
        "L24", 0.0, -2.0, 4.0, [2.0, 4.0], Float64[]; warmup=2)
    standardized = SuperlearnerIntervalUpgrade.candidate_interval(
        "S48", 0.0, -2.0, 4.0, [2.0, 4.0], [0.5, 2.0]; warmup=2)
    @assert (translated.lo, translated.hi) == (0.5, 8.0)
    @assert (standardized.lo, standardized.hi) == (-1.0, 11.0)

    correction_history = [fill(1.0,29);31.0]
    mean_point = SuperlearnerPointUpgrade.corrected_point("Mean24",-10,-12,-6,1,correction_history)
    median_point = SuperlearnerPointUpgrade.corrected_point("Median24",-10,-12,-6,1,correction_history)
    @assert (mean_point.point,mean_point.lower,mean_point.upper) == (-7.75,-10.25,-2.75)
    @assert (median_point.point,median_point.lower,median_point.upper) == (-9.0,-11.5,-4.0)
    names=collect(SuperlearnerPointUpgrade.POINT_CANDIDATES)
    gates=DataFrame(candidate=names,condition=fill("worked_example",6),dataset=fill("toy",6),
        delay_hours=zeros(Int,6),cell=fill("all",6),model_step_hours=zeros(Int,6),pass=falses(6))
    ranks=DataFrame(candidate=names,worst_step_gain_nt=ones(6),live_rmse_nt=ones(6))
    @assert SuperlearnerPointEvaluation.choose_candidate(gates,ranks)===nothing
    gates.pass.=true
    gates.pass[1]=false
    @assert SuperlearnerPointEvaluation.choose_candidate(gates,ranks)=="Mean48"

    println("SolarSINDy experiments: super-learner serving and predecessor smoke PASS")
    println("Interval-development arithmetic smoke PASS; served policy unchanged")
    println("Point-development arithmetic smoke PASS; served policy unchanged")
    println("Point-development no-winner and all-gate selection smoke PASS")
    return true
end

"""
    serve_deployed_superlearner_row()

Load the deployed super-learner bundle and serve one row through the operator that publishes the forecast.

The rest of this smoke exercises the inertial-SINDy tail and the serving helpers in isolation; without this
step nothing here touches the artifact that actually produces the published center, and the file
could not be cited as evidence for it. The row's center is recomputed from the resolved cell's own
weights and its band from the bundle's own conformal stratum, so the check is an independent
combination rather than a repetition of the operator's output.
"""
function serve_deployed_superlearner_row()
    dir = _superlearner_deploy_dir()
    isdir(dir) || error("deployed super-learner bundle absent at $dir")
    artifacts = load_superlearner_serving_artifacts(dir)
    @assert artifacts.identity == SUPERLEARNER_SERVED_TAIL_VERSION
    @assert artifacts.identity == SUPERLEARNER_SERVED_IDENTITY
    @assert artifacts.served_variant == SUPERLEARNER_SERVED_VARIANT
    @assert !artifacts.guard_applied

    # A deep, actively deepening state, so the row resolves a storm cell rather than the pooled one.
    step = 3
    latest_dst = -120.0
    rate = -25.0
    vbsouth = 3.0
    experts = NamedTuple{SUPERLEARNER_SERVING_EXPERTS}((
        -128.0, -124.0, -131.0, latest_dst, -119.0, -121.5, -117.0, -134.0, -95.0, -126.0,
    ))
    row = superlearner_serving_center(artifacts; model_steps=step, latest_dst=latest_dst,
                             dst_delta_1h_nt=rate, vbsouth_mvm=vbsouth, experts=experts)

    resolved = superlearner_serving_cell(artifacts, step, row.cell_regime, row.cell_depth)
    expected_center = sum(resolved.cell.weights[j] * values(experts)[j]
                          for j in 1:SUPERLEARNER_SERVING_EXPERT_COUNT)
    @assert isapprox(row.l1_center, expected_center; atol=1e-12)
    @assert !row.guard_applied
    @assert !row.projection_applied
    @assert row.center == row.l1_center
    @assert row.depth_bin === superlearner_serving_depth_bin(latest_dst)
    @assert row.deepening_cell
    # A deep but flat, uncoupled state is not a deepening cell. This distinguishes
    # the observed Dst from its hourly rate in the label's argument mapping.
    flat = superlearner_serving_center(artifacts; model_steps=step, latest_dst=latest_dst,
                              dst_delta_1h_nt=0.0, vbsouth_mvm=0.0, experts=experts)
    @assert !flat.deepening_cell
    half = artifacts.conformal[(step, row.cell_depth)].half_width_nt
    @assert row.half_width_nt == half
    @assert isapprox(row.ci05_nt, row.center - half; atol=1e-12)
    @assert isapprox(row.ci95_nt, row.center + half; atol=1e-12)
    @assert row.sindy_family_mass >= SUPERLEARNER_SERVING_SINDY_FLOOR - 1e-9

    println("Deployed super-learner operator ($(artifacts.identity)) served one row from $(dir): " *
            "cell $(row.regime_cell), center $(round(row.center; digits=3)) nT, " *
            "band ±$(round(half; digits=3)) nT")
    return true
end

include(joinpath(@__DIR__, "receipt_maturity_synthetic.jl"))
main()
run_receipt_maturity_synthetic()
include("interval_mean_forecaster_synthetic.jl")
run_interval_mean_forecaster_synthetic()
include(joinpath(@__DIR__, "lund_dst_2002_reference.jl"))
run_lund_dst_2002_reference()
include(joinpath(@__DIR__, "completed_minute_replay_synthetic.jl"))
run_completed_minute_replay_synthetic()
include(joinpath(@__DIR__, "duration_forcing_synthetic.jl"))
include(joinpath(@__DIR__, "duration_mean_synthetic.jl"))
include(joinpath(@__DIR__, "b_stream_bridge_synthetic.jl"))
include(joinpath(@__DIR__, "versioned_shadow_trial_synthetic.jl"))
run_versioned_shadow_trial_synthetic()

include(joinpath(@__DIR__, "completed_source_dbdt_synthetic.jl"))
run_synthetic_completed_source_replay()
include(joinpath(@__DIR__, "completed_label_events_synthetic.jl"))
run_synthetic_completed_label_events()
run_synthetic_completed_label_events_bounded()
