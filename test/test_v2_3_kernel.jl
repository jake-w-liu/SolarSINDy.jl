module V23KernelTests

using Test
using Dates
using SolarSINDy

const V23_KERNEL_SCRIPT = normpath(joinpath(
    @__DIR__, "..", "validation", "operational", "v2_3_kernel.jl",
))
include(V23_KERNEL_SCRIPT)   # also loads v2_replay.jl: _v2_forecast and the shared L1 admission

const CORE = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
const CAL = read_operational_v2_calibration(
    operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
)
const LIB = CORE.library
const XI = CORE.coefficients

const SLOW = (V = 300.0, Bz = -12.0, By = 2.0, n = 6.0, Pdyn = 2.0)
const FAST = (V = 800.0, Bz = -9.0, By = 1.0, n = 4.0, Pdyn = 3.0)
const FUT_SLOW = (V = 320.0, Bz = -24.0, By = 0.0, n = 8.0, Pdyn = 3.0)
const FUT_FAST = (V = 780.0, Bz = -30.0, By = 3.0, n = 9.0, Pdyn = 5.0)

"""
Independent reference rollout for the realized-driver oracle. The L1 standoff, the transit gate,
the anchor rollout, the pressure reconstruction, and the calibration lookup are re-derived here
from the documented physical definitions rather than taken from the kernel, so agreement is
evidence about the kernel and not a restatement of it.
"""
function _reference_realized(anchor, issue_drv, future, latest, h, rate)
    transit(V) = (isfinite(V) && V > 0) ? 1.5e6 / V / 3600.0 : 0.0
    steps_known = floor(Int, transit(issue_drv.V))
    filter = init_assimilation(LIB, XI, Int[], anchor)
    last_known = issue_drv
    final_drv = issue_drv
    for k in 1:h
        record = future(k)
        driver = if k <= steps_known
            admitted = record !== nothing && k <= transit(record.V)
            last_known = admitted ? record : last_known
            last_known
        else
            record === nothing ? last_known : record
        end
        final_drv = driver
        assimilation_predict!(filter, driver)
        filter.mean[1] = clamp(filter.mean[1], -2000.0, 50.0)
    end
    raw = current_dst(filter) + 7.26 * sqrt(max(final_drv.Pdyn, 0.0)) - 11.0
    features = _v2_calibration_features(
        CAL, latest, issue_drv; v1_pred_dst=raw, model_steps=h, context="V2.3 oracle reference",
    )
    corrected = raw + SolarSINDy.operational_v2_correction(CAL, features)
    return (
        clamp(raw, -2000.0, 50.0),
        _apply_v2_1_safeguards(corrected, latest, h, isfinite(rate) ? Float64(rate) : 0.0),
    )
end

@testset "V2.1 served forecast is unchanged by the shared-admission refactor" begin
    # Regression literals captured from `_v2_forecast` before the L1-admission helper was factored
    # out for reuse by the V2.3 kernel. Any drift in the served numbers changes the deployed
    # product and must fail here.
    baseline = [
        (-150.0, SLOW, k -> FUT_SLOW, -148.0, 1, 0.0, false,
         -155.3050264198818, -145.2411342770414),
        (-150.0, SLOW, k -> FUT_SLOW, -148.0, 3, -20.0, false,
         -164.42491528247515, -170.5),
        (-200.0, SLOW, k -> FUT_SLOW, -198.0, 6, -40.0, false,
         -211.19980040858346, -210.27705786668258),
        (-200.0, SLOW, k -> nothing, -198.0, 7, 10.0, false,
         -156.43216361373678, -98.428106426765),
        (-80.0, FAST, k -> FUT_FAST, -78.0, 4, -5.0, false,
         -83.12007882312795, -71.20214165668055),
        (-200.0, SLOW, k -> FUT_SLOW, -198.0, 6, -40.0, true,
         -184.50043351801574, -133.57769097611487),
        (-30.0, FAST, k -> nothing, -28.0, 2, 0.0, false,
         -35.99369099514358, -37.63280779435224),
        (-260.0, SLOW, k -> (k <= 1 ? FUT_SLOW : nothing), -255.0, 6, -40.0, false,
         -254.60045642991412, -236.05755175321676),
        # Slow issue wind opens the transit window (kΔ = 1) but the arrival-hour record is fast, so
        # it left L1 after issue and the acceleration gate must reject it. These rows fail if the
        # gate is dropped from the shared admission helper.
        (-150.0, SLOW, k -> FUT_FAST, -148.0, 1, -20.0, false,
         -150.27703693987064, -153.625),
        (-150.0, SLOW, k -> FUT_FAST, -148.0, 3, -20.0, false,
         -147.56916115017665, -170.5),
        (-150.0, SLOW, k -> FUT_FAST, -148.0, 6, -20.0, false,
         -140.22363295283748, -154.75717298531944),
    ]
    for (anchor, drv, future, latest, h, rate, frozen, raw, center) in baseline
        @test _v2_forecast(LIB, XI, anchor, drv, future, latest, CAL, h, rate;
                           force_frozen = frozen) === (raw, center)
    end

    # The factored helpers are the served definitions, checked against hand arithmetic.
    @test _v2_admitted_driver(k -> FUT_SLOW, 1, 1, SLOW) === FUT_SLOW      # slow arrival hour, admitted
    @test _v2_admitted_driver(k -> FUT_FAST, 1, 1, SLOW) === SLOW          # accelerated, rejected
    @test _v2_admitted_driver(k -> nothing, 1, 1, SLOW) === SLOW           # missing record, frozen
    relaxed = _v2_relaxed_tail_driver(SLOW, 3, 1, 2.0)
    @test relaxed.V == SLOW.V && relaxed.n == SLOW.n && relaxed.Pdyn == SLOW.Pdyn
    @test relaxed.Bz == SLOW.Bz * exp(-1.0)
    @test relaxed.By == SLOW.By * exp(-1.0)
    @test _v2_relaxed_tail_driver(SLOW, 5, 0, Inf) === SLOW                # frozen-tail limit
    @test _v23_tail_tau(0.0) == TAU0_V2
    @test _v23_tail_tau(-15.0) == TAU0_V2 * (1.0 + 15.0 / R0_V2)
    @test _v23_tail_tau(+15.0) == TAU0_V2
    @test _v23_tail_tau(-1e9) == TAU_MAX
    @test _v23_tail_tau(NaN) == TAU0_V2
end

@testset "V2.3 kernel shipped oracles" begin
    @test _selftest_v23()
end

@testset "V2.3 kernel reproduces the served V2.1 path exactly" begin
    for (anchor, latest, rate) in ((-150.0, -148.0, -20.0), (-45.0, -44.0, 0.0),
                                   (-260.0, -255.0, -40.0))
        for issue_drv in (SLOW, FAST), future in (k -> FUT_SLOW, k -> FUT_FAST, k -> nothing)
            for h in (1, 2, 3, 4, 6, 7)
                # A calibration row built through the live-forecast feature path, so the
                # correction stage is identical by construction and only the rollout is compared.
                row = _v2_features(latest, issue_drv; v1_pred_dst = -160.0, model_steps = h)
                served = _v2_forecast(LIB, XI, anchor, issue_drv, future, latest, CAL, h, rate;
                                      calibration_features = row)
                candidate = _v23_forecast(LIB, XI, anchor, issue_drv, future, latest, CAL, h, rate;
                                          tail_members = [_v23_relaxed_tail_member(rate)],
                                          calibration_features = row)
                @test candidate === served
            end
        end
    end
end

@testset "V2.3 kernel realized-driver oracle" begin
    for (anchor, latest, rate, issue_drv, future) in (
        (-150.0, -148.0, -20.0, SLOW, k -> FUT_SLOW),
        (-150.0, -148.0, -20.0, FAST, k -> FUT_SLOW),
        (-260.0, -255.0, -40.0, SLOW, k -> (k <= 2 ? FUT_SLOW : nothing)),
        (-40.0, -38.0, 5.0, FAST, k -> FUT_FAST),
    )
        for h in (1, 2, 3, 4, 6, 7)
            candidate = _v23_forecast(LIB, XI, anchor, issue_drv, future, latest, CAL, h, rate;
                                      tail_members = [v23_realized_driver_member(future)])
            @test candidate === _reference_realized(anchor, issue_drv, future, latest, h, rate)
        end
    end

    # The oracle is a different forecast from the served tail whenever the tail actually acts.
    realized = _v23_forecast(LIB, XI, -200.0, FAST, k -> FUT_SLOW, -198.0, CAL, 6, -30.0;
                             tail_members = [v23_realized_driver_member(k -> FUT_SLOW)])
    served = _v2_forecast(LIB, XI, -200.0, FAST, k -> FUT_SLOW, -198.0, CAL, 6, -30.0)
    @test realized != served
end

@testset "V2.3 kernel averages members and confines them to the tail" begin
    anchor, latest, rate, h = -200.0, -198.0, -30.0, 6
    single = _v23_forecast(LIB, XI, anchor, SLOW, k -> nothing, latest, CAL, h, rate;
                           tail_members = [_v23_relaxed_tail_member(rate)])

    # Repeating one member cannot change the ensemble mean beyond the rounding of a K-term sum,
    # bounded by K * eps * |center| < 1e-9 nT for K = 200 and centers below 2000 nT.
    for K in (2, 5, 200)
        repeated = _v23_forecast(LIB, XI, anchor, SLOW, k -> nothing, latest, CAL, h, rate;
                                 tail_members = [_v23_relaxed_tail_member(rate) for _ in 1:K])
        @test all(abs.(repeated .- single) .< 1e-9)
    end
    @test _v23_forecast(LIB, XI, anchor, SLOW, k -> nothing, latest, CAL, h, rate;
                        tail_members = [_v23_relaxed_tail_member(rate) for _ in 1:2]) === single

    # Two distinct members average in the raw Dst domain, not in the driver domain: the ensemble
    # center is the mean of the two single-member raw centers.
    strong = (k, last_known, kΔ) -> (V = 700.0, Bz = -25.0, By = 0.0, n = 12.0,
                                     Pdyn = dynamic_pressure(12.0, 700.0))
    quiet = (k, last_known, kΔ) -> (V = 350.0, Bz = 2.0, By = 0.0, n = 3.0,
                                    Pdyn = dynamic_pressure(3.0, 350.0))
    raw_strong = _v23_forecast(LIB, XI, anchor, SLOW, k -> nothing, latest, CAL, h, rate;
                               tail_members = [strong])[1]
    raw_quiet = _v23_forecast(LIB, XI, anchor, SLOW, k -> nothing, latest, CAL, h, rate;
                              tail_members = [quiet])[1]
    raw_pair = _v23_forecast(LIB, XI, anchor, SLOW, k -> nothing, latest, CAL, h, rate;
                             tail_members = [strong, quiet])[1]
    @test raw_pair == (raw_strong + raw_quiet) / 2
    @test raw_strong < raw_pair < raw_quiet
    # Member order is a labelling of the ensemble, not information.
    @test _v23_forecast(LIB, XI, anchor, SLOW, k -> nothing, latest, CAL, h, rate;
                        tail_members = [quiet, strong])[1] == raw_pair

    # Members are consulted only beyond the L1-known window, and exactly once per step there.
    for (issue_drv, expected_kd) in ((SLOW, 1), (FAST, 0))
        seen = Int[]
        recorder = (k, last_known, kΔ) -> (push!(seen, k); last_known)
        _v23_forecast(LIB, XI, anchor, issue_drv, k -> FUT_SLOW, latest, CAL, 7, rate;
                      tail_members = [recorder])
        @test seen == collect((expected_kd + 1):7)
    end
end

@testset "V2.3 kernel correction and safeguard stages" begin
    # The extreme-core inertia guard serves persistence at short leads; switching the safeguard
    # stage off must expose the corrected model center instead.
    guarded = _v23_forecast(LIB, XI, -250.0, SLOW, k -> nothing, -250.0, CAL, 2, 10.0;
                            tail_members = [_v23_relaxed_tail_member(10.0)])
    bare = _v23_forecast(LIB, XI, -250.0, SLOW, k -> nothing, -250.0, CAL, 2, 10.0;
                         tail_members = [_v23_relaxed_tail_member(10.0)], safeguards = false)
    @test guarded[2] == -250.0
    @test bare[2] != -250.0
    @test bare[1] == guarded[1]

    # The correction is evaluated on the supplied calibration row, so a row from a different lead
    # produces a different center while the raw rollout is untouched.
    row6 = _v2_features(-198.0, SLOW; v1_pred_dst = -160.0, model_steps = 6)
    row1 = _v2_features(-198.0, SLOW; v1_pred_dst = -160.0, model_steps = 1)
    with6 = _v23_forecast(LIB, XI, -200.0, SLOW, k -> nothing, -198.0, CAL, 6, -30.0;
                          tail_members = [_v23_relaxed_tail_member(-30.0)],
                          calibration_features = row6)
    with1 = _v23_forecast(LIB, XI, -200.0, SLOW, k -> nothing, -198.0, CAL, 6, -30.0;
                          tail_members = [_v23_relaxed_tail_member(-30.0)],
                          calibration_features = row1)
    @test with6[1] == with1[1]
    @test with6[2] != with1[2]

    @test_throws ArgumentError _v23_forecast(
        LIB, XI, -200.0, SLOW, k -> nothing, -198.0, CAL, 6, -30.0; tail_members = Function[],
    )
    @test_throws ArgumentError _v23_forecast(
        LIB, XI, -200.0, SLOW, k -> nothing, -198.0, CAL, 0, -30.0;
        tail_members = [_v23_relaxed_tail_member(-30.0)],
    )
    @test_throws ErrorException _v23_forecast(
        LIB, XI, -200.0, SLOW, k -> nothing, -198.0, CAL, 6, -30.0;
        tail_members = [_v23_relaxed_tail_member(-30.0)],
        calibration_features = (latest_dst_nt = -198.0,),
    )
end

end # module
