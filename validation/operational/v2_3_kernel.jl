# v2_3_kernel.jl — Operational V2.3 forecast kernel: one SINDy rollout per driver-continuation member.
#
# The served V2.1 point forecast integrates the unchanged 20-candidate / 11-active-term SINDy core
# along a single driver sequence: the L1-admitted OMNI hours first, then a regime-aware exponential
# relaxation of the last admitted driver. V2.3 keeps every other stage of that pipeline byte-exact
# and replaces only the tail. Because the core is nonlinear in the driver, integrating it once
# against a mean continuation is not the same as averaging the integrations of an ensemble of
# continuations; the kernel therefore rolls the core out once per member and averages the resulting
# Dst centers, which is what lets an analog ensemble express driver uncertainty rather than collapse
# it.
#
# Shared-path guarantee: steps inside the L1-known window use `_v2_admitted_driver`, the same
# admission helper the served product uses in `_v2_forecast`. A member function is consulted only
# for steps k > kΔ, so any difference between a V2.3 candidate and the served product is confined to
# hours that were never measured at L1 by issue time. With one member reproducing the served
# relaxation, `_v23_forecast` returns the served numbers exactly; that equality is an executable
# oracle in `_selftest_v23` and in test/test_v2_3_kernel.jl, not a design intention.
#
# Run from the package root: julia --project=. validation/operational/v2_3_kernel.jl

include(joinpath(@__DIR__, "v2_replay.jl"))   # _v2_forecast, _v2_admitted_driver, _v2_relaxed_tail_driver,
                                              # _transit_hours, TAU0_V2, R0_V2, TAU_MAX, _apply_v2_1_safeguards

"""
    _v23_tail_tau(rate; tau0 = TAU0_V2, r0 = R0_V2)

Regime-aware relaxation timescale of the served V2.1 tail for an issue-time Dst rate `rate`,
computed exactly as `_v2_forecast` computes it.
"""
function _v23_tail_tau(rate; tau0 = TAU0_V2, r0 = R0_V2)
    rate_effective = isfinite(rate) ? Float64(rate) : 0.0
    return min(TAU_MAX, tau0 * (1.0 + max(0.0, -rate_effective) / r0))
end

"""
    _v23_relaxed_tail_member(rate; tau0 = TAU0_V2, r0 = R0_V2) -> Function

Ensemble member reproducing the served V2.1 tail. Supplied as the only member, it makes
[`_v23_forecast`](@ref) reproduce `_v2_forecast` exactly, which is the continuity oracle for the
kernel.
"""
function _v23_relaxed_tail_member(rate; tau0 = TAU0_V2, r0 = R0_V2)
    tau = _v23_tail_tau(rate; tau0 = tau0, r0 = r0)
    return (k, last_known, kΔ) -> _v2_relaxed_tail_driver(last_known, k, kΔ, tau)
end

"""
    v23_realized_driver_member(future) -> Function

Ensemble member that continues the tail with the driver that was actually observed, `future(k)`,
falling back to the last L1-admitted driver when the archive has no record for that hour. This
member is not causal and is never a candidate: it defines the realized-driver oracle, the upper
bound on what a perfect driver continuation could deliver through the unchanged core, and therefore
the headroom against which a candidate tail is judged.
"""
function v23_realized_driver_member(future)
    return function (k, last_known, kΔ)
        realized = future(k)
        return realized === nothing ? last_known : realized
    end
end

"""
    _v23_forecast(lib, ξ0, anchor_dst_star, issue_drv, future, latest_dst, cal, h, rate;
                  tail_members, calibration_features = nothing, safeguards = true, ...)

h-step V2.3 point forecast. Rollout steps inside the L1-known window (`k ≤ kΔ`, with
`kΔ = ⌊transit(V_issue)⌋`) take the admitted OMNI record through the shared `_v2_admitted_driver`
path; later steps take the driver returned by each member `m(k, last_known, kΔ)`, where `last_known`
is the final L1-admitted driver. Every member drives its own rollout of the unchanged core from the
same anchor, contributes `dst* + 7.26√Pdyn_final − 11` at the target step, and the raw center is the
member mean. The V2.1 ridge correction is then evaluated on the row's calibration features and, when
`safeguards` is set, the published V2.1 point operator is applied in its declared order.

Returns `(raw, center)`, matching `_v2_forecast`.
"""
function _v23_forecast(lib, ξ0, anchor_dst_star, issue_drv, future, latest_dst, cal, h::Int, rate;
                       tail_members,
                       calibration_features = nothing,
                       safeguards::Bool = true,
                       apply_rate_guard::Bool = true,
                       apply_one_hour_inertia::Bool = true,
                       apply_state_inertia::Bool = true)
    h >= 1 || throw(ArgumentError("model step h must be positive, got $h"))
    isempty(tail_members) && throw(ArgumentError(
        "V2.3 forecast needs at least one tail-continuation member",
    ))
    rate_effective = isfinite(rate) ? Float64(rate) : 0.0
    kΔ = floor(Int, _transit_hours(issue_drv.V))
    member_pred = Vector{Float64}(undef, length(tail_members))
    for (m, member) in enumerate(tail_members)
        last_known = issue_drv
        final_drv = issue_drv                 # driver of the final rollout step (target-step Pdyn, Eq. 4)
        fc = init_assimilation(lib, ξ0, Int[], anchor_dst_star)
        for k in 1:h
            if k <= kΔ
                drv_k = _v2_admitted_driver(future, k, kΔ, last_known)
                last_known = drv_k
            else
                drv_k = member(k, last_known, kΔ)
            end
            final_drv = drv_k
            assimilation_predict!(fc, drv_k)
            fc.mean[1] = clamp(fc.mean[1], -2000.0, 50.0)
        end
        member_pred[m] = current_dst(fc) + 7.26 * sqrt(max(final_drv.Pdyn, 0.0)) - 11.0
    end
    pred_dst = mean(member_pred)
    fallback_features = _v2_features(
        latest_dst, issue_drv; v1_pred_dst=pred_dst, model_steps=h,
    )
    feature_source = calibration_features === nothing ? fallback_features : calibration_features
    available = propertynames(feature_source)
    missing_features = [c for c in cal.feature_names if !(c in available)]
    isempty(missing_features) || error(
        "operational calibration feature source omits: $(join(String.(missing_features), ", "))",
    )
    feats = NamedTuple{Tuple(cal.feature_names)}(
        Tuple(Float64(getproperty(feature_source, c)) for c in cal.feature_names),
    )
    corr = SolarSINDy.operational_v2_correction(cal, feats)
    corrected = clamp(pred_dst + corr, -2000.0, 50.0)
    if safeguards
        corrected = _apply_v2_1_safeguards(
            pred_dst + corr,
            latest_dst,
            h,
            rate_effective;
            apply_rate_guard=apply_rate_guard,
            apply_one_hour_inertia=apply_one_hour_inertia,
            apply_state_inertia=apply_state_inertia,
            apply_extreme_inertia=true,
        )
    end
    return clamp(pred_dst, -2000.0, 50.0), corrected
end

# ---- CRC self-test ----
_cal_v23 = Ref{Any}(nothing)
_calV23() = (_cal_v23[] === nothing && (_cal_v23[] = read_operational_v2_calibration(
    operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
)); _cal_v23[])

"""
    _selftest_v23() -> Bool

Executable oracles for the V2.3 kernel: exact continuity with the served V2.1 path under a single
relaxed-tail member, ensemble invariance under member duplication, and confinement of the member
influence to steps beyond the L1-known window.
"""
function _selftest_v23()
    core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
    lib, ξ0 = core.library, core.coefficients
    cal = _calV23()
    slow = (V = 300.0, Bz = -12.0, By = 2.0, n = 6.0, Pdyn = 2.0)
    fut = (V = 320.0, Bz = -24.0, By = 0.0, n = 8.0, Pdyn = 3.0)

    # Oracle 1 — CONTINUITY: one member reproducing the served relaxation returns the served numbers
    # bit-for-bit, at guarded and unguarded settings, for both admitted and frozen look-ahead cases.
    for (anchor, latest, rate) in ((-150.0, -148.0, -20.0), (-200.0, -198.0, +10.0),
                                   (-60.0, -58.0, 0.0))
        for h in (1, 2, 3, 4, 6, 7), future in (k -> fut, k -> nothing)
            served = _v2_forecast(lib, ξ0, anchor, slow, future, latest, cal, h, rate)
            candidate = _v23_forecast(
                lib, ξ0, anchor, slow, future, latest, cal, h, rate;
                tail_members=[_v23_relaxed_tail_member(rate)],
            )
            @assert candidate == served "V2.3 kernel breaks V2.1 continuity at h=$h: $candidate vs $served"
        end
    end

    # Oracle 2 — ENSEMBLE INVARIANCE: duplicating one member cannot move the center by more than the
    # summation rounding of the mean (K terms, bounded by K*eps relative).
    single = _v23_forecast(lib, ξ0, -200.0, slow, k -> nothing, -198.0, cal, 6, -30.0;
                           tail_members=[_v23_relaxed_tail_member(-30.0)])
    duplicated = _v23_forecast(lib, ξ0, -200.0, slow, k -> nothing, -198.0, cal, 6, -30.0;
                               tail_members=[_v23_relaxed_tail_member(-30.0) for _ in 1:8])
    @assert all(isapprox.(duplicated, single; atol=8 * eps(Float64) * 2000.0, rtol=0.0)) (
        "duplicated members moved the V2.3 center: $duplicated vs $single"
    )

    # Oracle 3 — MEMBER CONFINEMENT: with fast issue wind there is no L1-known window (kΔ=0), so the
    # realized-driver member changes the forecast; with h=1 and slow wind (kΔ=1) the admitted record
    # already fixes step 1 and the member cannot act.
    fast = (V = 800.0, Bz = -9.0, By = 1.0, n = 4.0, Pdyn = 3.0)
    realized = _v23_forecast(lib, ξ0, -150.0, fast, k -> fut, -148.0, cal, 6, -20.0;
                             tail_members=[v23_realized_driver_member(k -> fut)])
    relaxed = _v23_forecast(lib, ξ0, -150.0, fast, k -> fut, -148.0, cal, 6, -20.0;
                            tail_members=[_v23_relaxed_tail_member(-20.0)])
    @assert realized != relaxed "realized-driver member had no effect beyond the L1 window"
    admitted_realized = _v23_forecast(lib, ξ0, -150.0, slow, k -> fut, -148.0, cal, 1, -20.0;
                                      tail_members=[v23_realized_driver_member(k -> fut)])
    admitted_relaxed = _v23_forecast(lib, ξ0, -150.0, slow, k -> fut, -148.0, cal, 1, -20.0;
                                     tail_members=[_v23_relaxed_tail_member(-20.0)])
    @assert admitted_realized == admitted_relaxed "member leaked into the L1-admitted window"

    println("  ✓ V2.3 kernel self-test: served-path continuity, ensemble invariance, member confinement")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest_v23()
end
