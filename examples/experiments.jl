#!/usr/bin/env julia

using Dates
using DataFrames
using SolarSINDy

include(joinpath(@__DIR__, "live_forecast_verify.jl"))

function main()
    cal = default_operational_v2_calibration()
    latest_dst = -80.0
    drivers = (V=420.0, Bz=-10.0, By=3.0, n=6.0, Pdyn=2.0)
    v1_pred = -95.0
    v1_ci05 = -115.0
    v1_ci95 = -75.0
    baselines = (persistence=latest_dst, burton=-90.0, burton_full=-92.0, obrien=-98.0)

    selected = _select_model_prediction(
        :v2,
        cal,
        latest_dst,
        drivers,
        v1_pred,
        v1_ci05,
        v1_ci95;
        baselines=baselines,
    )
    @assert selected.model_version == OPERATIONAL_V2_1_MODEL_VERSION
    @assert selected.v2_correction == 0.0
    @assert selected.v2_pred_dst == v1_pred
    @assert selected.v2_ci05_dst == v1_ci05
    @assert selected.v2_ci95_dst == v1_ci95

    @assert _v2_tail_tau(-30.0) > _v2_tail_tau(5.0)
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
    @assert V2_SERVED_TAIL_VERSION ==
            "v2.1+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"
    # Served pipeline: the V2.1 operator followed by the fitted static regime stack.
    @assert V2_2_SERVED_TAIL_VERSION ==
            "v2.2+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia+staticstack(sindy60_fit407598)"
    # The V2.3 analog candidate is a shadow forecast; its confirmatory decision was NO_GO.
    @assert V2_3_SHADOW_TAIL_VERSION ==
            "v2.3-shadow+sindy20x11+L1A+ADC(magnetic,K25)+T1rcal+LAT+E"
    # Served pipeline: the V2.4e super-learner over the ten experts, the static stack among them.
    @assert V2_4_SERVED_TAIL_VERSION ==
            "v2.4+sindy20x11+superlearner10floor+conformal"
    @assert v22_serving_coupling_active(2.5, -4.0) == 2.5
    @assert v22_serving_coupling_active(2.5, 1.0) == 0.0
    @assert v22_serving_coupling_active(0.0, -4.0) == 0.0
    @assert v22_serving_depth_safe_center(-90.0, -120.0) == -120.0
    @assert v22_serving_depth_safe_center(-150.0, -120.0) == -150.0
    # Published severity is the deepest of the served center and both continuity partners.
    @assert v24_serving_depth_safe_center(-90.0, -120.0, -60.0) == -120.0
    @assert v24_serving_depth_safe_center(-150.0, -120.0, -60.0) == -150.0
    @assert v24_serving_depth_safe_center(-90.0, NaN, -95.0) == -95.0
    # The published watch edge is the same rule on the edges themselves, so a narrower served band
    # cannot lower a watch tier a predecessor's own edge would have raised.
    @assert v24_serving_depth_safe_center(-92.0, -100.0, -105.0) == -105.0
    @assert v24_serving_depth_safe_center(-160.0, -100.0, -105.0) == -160.0
    @assert v24_serving_depth_safe_center(-92.0, NaN, -105.0) == -105.0
    # Depth bins and the deepening cell of the V2.4 guard, on their closed-from-below edges.
    @assert v24_serving_depth_bin(-29.9) === :shallow
    @assert v24_serving_depth_bin(-30.0) === :moderate
    @assert v24_serving_depth_bin(-70.0) === :deep
    @assert v24_serving_deepening(-16.0, 0.0, 0.0)
    @assert !v24_serving_deepening(-14.0, 0.0, 0.0)
    @assert v24_serving_deepening(-1.0, 2.0, -50.0)
    @assert !v24_serving_deepening(-1.0, 0.0, -50.0)
    @assert v24_serving_guard(-80.0, -60.0, -20.0, 0.0, -10.0) == -80.0
    @assert v24_serving_guard(-80.0, -60.0, -1.0, 0.0, -10.0) == -60.0

    serve_deployed_v24_row()

    println("SolarSINDy experiments: V2.1 deterministic smoke PASS")
    return true
end

"""
    serve_deployed_v24_row()

Load the deployed V2.4e bundle and serve one row through the operator that publishes the forecast.

The rest of this smoke exercises the V2.1 tail and the serving helpers in isolation; without this
step nothing here touches the artifact that actually produces the published center, and the file
could not be cited as evidence for it. The row's center is recomputed from the resolved cell's own
weights and its band from the bundle's own conformal stratum, so the check is an independent
combination rather than a repetition of the operator's output.
"""
function serve_deployed_v24_row()
    dir = _v2_4_deploy_dir()
    isdir(dir) || error("deployed V2.4e bundle absent at $dir")
    artifacts = load_v24_serving_artifacts(dir)
    @assert artifacts.identity == V2_4_SERVED_TAIL_VERSION
    @assert artifacts.identity == V24_SERVED_IDENTITY
    @assert artifacts.served_variant == V24_SERVED_VARIANT
    @assert !artifacts.guard_applied

    # A deep, actively deepening state, so the row resolves a storm cell rather than the pooled one.
    step = 3
    latest_dst = -120.0
    rate = -25.0
    vbsouth = 3.0
    experts = NamedTuple{V24_SERVING_EXPERTS}((
        -128.0, -124.0, -131.0, latest_dst, -119.0, -121.5, -117.0, -134.0, -95.0, -126.0,
    ))
    row = v24_serving_center(artifacts; model_steps=step, latest_dst=latest_dst,
                             dst_delta_1h_nt=rate, vbsouth_mvm=vbsouth, experts=experts)

    resolved = v24_serving_cell(artifacts, step, row.cell_regime, row.cell_depth)
    expected_center = sum(resolved.cell.weights[j] * values(experts)[j]
                          for j in 1:V24_SERVING_EXPERT_COUNT)
    @assert isapprox(row.l1_center, expected_center; atol=1e-12)
    @assert !row.guard_applied
    @assert !row.projection_applied
    @assert row.center == row.l1_center
    @assert row.depth_bin === v24_serving_depth_bin(latest_dst)
    @assert row.deepening_cell == v24_serving_deepening(latest_dst, vbsouth, rate)
    half = artifacts.conformal[(step, row.cell_depth)].half_width_nt
    @assert row.half_width_nt == half
    @assert isapprox(row.ci05_nt, row.center - half; atol=1e-12)
    @assert isapprox(row.ci95_nt, row.center + half; atol=1e-12)
    @assert row.sindy_family_mass >= V24_SERVING_SINDY_FLOOR - 1e-9

    println("Deployed V2.4e operator ($(artifacts.identity)) served one row from $(dir): " *
            "cell $(row.regime_cell), center $(round(row.center; digits=3)) nT, " *
            "band ±$(round(half; digits=3)) nT")
    return true
end

main()
