# Frozen pre-admission-refactor oracle from commit
# 1f337757cd242d4d1a917635b1fb8c83690a647a, validation/operational/v2_replay.jl.
# Only the function name differs; do not regenerate from the current implementation.
function _v21_before_admission_refactor(lib, ξ0, anchor_dst_star, issue_drv, future, latest_dst, cal, h::Int, rate;
                      tau0 = TAU0_V2, r0 = R0_V2, force_frozen::Bool = false,
                      calibration_features = nothing, apply_rate_guard::Bool = true,
                      apply_one_hour_inertia::Bool = true,
                      apply_state_inertia::Bool = true)
    rate_effective = isfinite(rate) ? Float64(rate) : 0.0
    Δ  = force_frozen ? 0.0 : _transit_hours(issue_drv.V)
    kΔ = floor(Int, Δ)
    tau = force_frozen ? Inf : min(
        TAU_MAX,
        tau0 * (1.0 + max(0.0, -rate_effective) / r0),
    )
    last_known = issue_drv
    final_drv = issue_drv                     # driver of the final rollout step (target-step Pdyn, Eq. 4)
    fc = init_assimilation(lib, ξ0, Int[], anchor_dst_star)
    for k in 1:h
        if k <= kΔ
            # `future(k)` returns the arrival-hour record tagged it+k-1 (the row covering at-Earth
            # interval [it+k-1, it+k)). Admit it only when that whole hour is L1-measured by issue time:
            # the issue-time transit gate (k<=kΔ) AND the admitted record's own speed (k<=transit(fut.V))
            # guard against intra-hour acceleration (a shock arriving into slow wind). Otherwise persist
            # the last known driver (freeze), exactly as for a missing in-window record.
            fut = future(k)
            drv_k = (fut !== nothing && k <= _transit_hours(fut.V)) ? fut : last_known
            last_known = drv_k
        else
            relax = exp(-(k - kΔ) / tau)
            drv_k = (V = last_known.V, Bz = last_known.Bz * relax, By = last_known.By * relax,
                     n = last_known.n, Pdyn = last_known.Pdyn)
        end
        final_drv = drv_k
        assimilation_predict!(fc, drv_k)
        fc.mean[1] = clamp(fc.mean[1], -2000.0, 50.0)
    end
    pred_dst_star = current_dst(fc)
    pred_dst = pred_dst_star + 7.26 * sqrt(max(final_drv.Pdyn, 0.0)) - 11.0
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
    if !force_frozen
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
