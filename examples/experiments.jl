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
    @assert v22_serving_coupling_active(2.5, -4.0) == 2.5
    @assert v22_serving_coupling_active(2.5, 1.0) == 0.0
    @assert v22_serving_coupling_active(0.0, -4.0) == 0.0
    @assert v22_serving_depth_safe_center(-90.0, -120.0) == -120.0
    @assert v22_serving_depth_safe_center(-150.0, -120.0) == -150.0

    println("SolarSINDy experiments: V2.1 deterministic smoke PASS")
    return true
end

main()
