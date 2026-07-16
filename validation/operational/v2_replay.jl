# v2_replay.jl — V2 replay: A (L1 look-ahead) + REGIME-AWARE B relaxation.
#
# Defect found by adversarial reflection: plain B relaxes Bz→0 with a fixed τ regardless of regime, so on
# ACTIVELY-DEEPENING deep storms it stops injecting too early and under-predicts depth (obs−pred ≈ −48 nT) — the
# dangerous direction for a severe-storm warning system. Fix at the source: make the relaxation timescale
# REGIME-AWARE. During active deepening (recent Dst rate < 0 / strong sustained southward driving), lengthen τ
# so the tail keeps injecting and the forecast does not go shallow; in recovery, use the normal τ0 (B's win).
#   τ_eff = τ0 · (1 + max(0, −rate)/R0)    (rate = recent dDst/dt; capped),  force_frozen ⇒ pre-upgrade baseline.
#
# Validates: (1) the deep-deepening shallow bias is removed (signed error ≈ 0, not −48), (2) the multi-hour
# RMSE gain over persistence survives, and (3) already-extreme 1-2 h targets do not lose to persistence after
# applying a causal inertia guard when latest Dst has already reached the extreme core.
#
# Run from the package root: julia --project=. validation/operational/v2_replay.jl

include(joinpath(@__DIR__, "v2_lookahead_replay.jl"))   # _driver_lookup, _transit_hours, _shadow_library, etc.

const OUT_CSV_V2 = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_replay_scored.csv")
const OUT_MD_V2  = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_replay_report.md")
const OUT_CSV_R0 = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_replay_r0_sweep.csv")
const TAU0_V2    = 3.0      # recovery relaxation timescale (h)
const R0_V2      = 7.5      # deepening-rate scale (nT/h): τ lengthens by 1+|rate|/R0 during active deepening
const TAU_MAX     = 48.0     # cap on τ_eff (h)
const EXTREME_INERTIA_DST_NT = -240.0
const EXTREME_INERTIA_MAX_H = 2

function _extreme_inertia_guard(latest_dst::Real, h::Int;
                                threshold::Real = EXTREME_INERTIA_DST_NT,
                                max_h::Int = EXTREME_INERTIA_MAX_H)
    latest = Float64(latest_dst)
    return isfinite(latest) && 0 < h <= max_h && latest <= Float64(threshold)
end

"h-step V2 forecast: A for k≤⌊Δ⌋, then REGIME-AWARE B relaxation of the last L1-known driver (τ longer
when actively deepening). `rate` = recent dDst/dt at issue. force_frozen ⇒ pre-upgrade baseline."
function _v2_forecast(lib, ξ0, anchor_dst_star, issue_drv, future, latest_dst, cal, h::Int, rate;
                      tau0 = TAU0_V2, r0 = R0_V2, force_frozen::Bool = false)
    Δ  = force_frozen ? 0.0 : _transit_hours(issue_drv.V)
    kΔ = floor(Int, Δ)
    tau = force_frozen ? Inf : min(TAU_MAX, tau0 * (1.0 + max(0.0, isfinite(rate) ? -rate : 0.0) / r0))
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
    feat_df = DataFrame(latest_dst_nt = [latest_dst], V_kms = [issue_drv.V], Bz_nt = [issue_drv.Bz],
                        By_nt = [issue_drv.By], n_cm3 = [issue_drv.n], Pdyn_npa = [issue_drv.Pdyn])
    prep = SolarSINDy.add_operational_v2_features!(feat_df)
    feats = NamedTuple{Tuple(cal.feature_names)}(Tuple(Float64(prep[1, c]) for c in cal.feature_names))
    corr = SolarSINDy.operational_v2_correction(cal, feats)
    corrected = clamp(pred_dst + corr, -2000.0, 50.0)
    if !force_frozen && _extreme_inertia_guard(latest_dst, h)
        guarded = clamp(Float64(latest_dst), -2000.0, 50.0)
        return guarded, guarded
    end
    return clamp(pred_dst, -2000.0, 50.0), corrected
end

function replay_v2_storm(storm, lib, ξ0, cal, lookup; r0 = R0_V2)
    yr = year(storm.t1)
    plasma, mag, dst_times, dst_vals = _omni_replay_inputs(OMNI, yr - 1, yr)
    win_lo, win_hi = storm.t0 - Hour(6), storm.t1 + Hour(7)
    # _omni_replay_inputs returns driver-gated plasma/mag and a separately built, generally LARGER
    # finite-Dst series; the two frames are no longer row-aligned, so window each by its own time vector.
    mp = (plasma.time_tag .>= win_lo) .& (plasma.time_tag .<= win_hi)
    md = (dst_times .>= win_lo) .& (dst_times .<= win_hi)
    (any(mp) && any(md)) || error("No finite OMNI drivers/Dst inside replay window")
    rh = Int(ceil(Dates.value(win_hi - win_lo) / 3_600_000))
    df = replay_recent_table(plasma[mp, :], mag[mp, :], dst_times[md], dst_vals[md];
                             replay_hours = rh, horizons = LEADS, model = :v2, calibration = cal)
    df[!, :issue_dt] = DateTime.(df.issue_time_utc)
    df = df[(df.issue_dt .>= storm.t0) .& (df.issue_dt .<= storm.t1), :]
    sort!(df, [:issue_dt, :model_step_hours])
    dst_map = Dict{DateTime,Float64}(zip(dst_times[md], Float64.(dst_vals[md])))  # true hourly Dst series
    out = DataFrame(storm = String[], issue_utc = DateTime[], lead = Int[], obs = Float64[],
                    audit_baseline = Float64[], v2 = Float64[], v2_frozen = Float64[],
                    persistence = Float64[], rate = Float64[])
    for it in sort!(unique(df.issue_dt))
        g = df[df.issue_dt .== it, :]; r1 = g[1, :]
        issue_drv = (V = Float64(r1.V_kms), Bz = Float64(r1.Bz_nt), By = Float64(r1.By_nt),
                     n = Float64(r1.n_cm3), Pdyn = Float64(r1.Pdyn_npa))
        latest = Float64(r1.latest_dst_nt)
        all(isfinite, (issue_drv.V, issue_drv.Bz, issue_drv.By, issue_drv.n, issue_drv.Pdyn, latest)) || continue
        anchor_star = pressure_correct_dst([latest], [issue_drv.Pdyn])[1]
        # Rollout step k is driven by the arrival-hour record tagged it+k-1 (fully L1-measured iff k<=Δ).
        future = k -> get(lookup, it + Hour(k - 1), nothing)
        # Recent Dst rate = true 1 h delta from the Dst series (NaN when the prior hour is absent),
        # matching the live dst_delta_1h guard; a multi-hour gap is never misread as an nT/h rate.
        prev1 = get(dst_map, it - Hour(1), NaN)
        rate = isfinite(prev1) ? latest - prev1 : NaN
        for r in eachrow(g)
            (ismissing(r.observation_dst_nt) || ismissing(r.v2_pred_dst_nt)) && continue
            isfinite(Float64(r.observation_dst_nt)) && isfinite(Float64(r.v2_pred_dst_nt)) || continue
            h = Int(r.model_step_hours)
            _, v2 = _v2_forecast(lib, ξ0, anchor_star, issue_drv, future, latest, cal, h, rate; r0 = r0)
            _, frozen = _v2_forecast(lib, ξ0, anchor_star, issue_drv, future, latest, cal, h, rate; r0 = r0, force_frozen = true)
            isfinite(v2) && isfinite(frozen) || continue
            push!(out, (storm.name, it, h, Float64(r.observation_dst_nt), Float64(r.v2_pred_dst_nt), v2, frozen, latest, rate))
        end
    end
    return out
end

_run_v2(lib, ξ0, cal, luc, r0) = reduce(vcat, [replay_v2_storm(s, lib, ξ0, cal, luc[year(s.t1)]; r0 = r0) for s in STORMS])

function _cell_v2(rows)
    nrow(rows) == 0 && return nothing
    eb = rows.obs .- rows.audit_baseline; ev2 = rows.obs .- rows.v2; ep = rows.obs .- rows.persistence
    rb, rv2, rp = _rmse(eb), _rmse(ev2), _rmse(ep)
    strong_pers = rp <= rb
    Δ, lo, hi = paired_improvement(strong_pers ? ep : eb, ev2; storms = rows.storm)
    return (n = nrow(rows), rmse_baseline = rb, rmse_v2 = rv2, rmse_pers = rp,
            stronger = strong_pers ? "pers" : "baseline",
            improve = Δ, ci_lo = lo, ci_hi = hi, fair = maximum(abs.(rows.v2_frozen .- rows.audit_baseline)))
end

"Signed error mean(obs − pred) on the deep ACTIVELY-DEEPENING subset (the shallow-bias failure mode)."
function _deep_bias(rows)
    d = rows[isfinite.(rows.rate) .& (rows.rate .< MAIN_RATE) .& (rows.obs .< -100.0), :]
    nrow(d) == 0 && return (n = 0, v2 = NaN, baseline = NaN)
    return (n = nrow(d), v2 = mean(d.obs .- d.v2), baseline = mean(d.obs .- d.audit_baseline))
end

function main_v2()
    lib, ξ0, i_decay = _shadow_library()
    cal = _load_calibration_for_model(LiveVerifyConfig(model = :v2))
    luc = Dict(year(s.t1) => _driver_lookup(year(s.t1)) for s in STORMS)
    println("V2 replay (A + regime-aware B + near-term extreme inertia), R0=", R0_V2, ", τ0=", TAU0_V2, ", leads ", LEADS, "\n", "="^80)
    all_rows = _run_v2(lib, ξ0, cal, luc, R0_V2)
    CSV.write(OUT_CSV_V2, all_rows)
    db = _deep_bias(all_rows[all_rows.lead .== 6, :])
    plain_rows = _run_v2(lib, ξ0, cal, luc, 1e9)
    plain_db = _deep_bias(plain_rows[plain_rows.lead .== 6, :])
    sweep_rows = DataFrame(r0 = Float64[], plain_b = Bool[], deep_n = Int[],
                           deep_signed_err_v2_nt = Float64[], deep_signed_err_baseline_nt = Float64[],
                           rmse_baseline_6h_nt = Float64[], rmse_v2_6h_nt = Float64[],
                           rmse_pers_6h_nt = Float64[], improve_6h_nt = Float64[],
                           improve_ci_lo_nt = Float64[], improve_ci_hi_nt = Float64[])
    open(OUT_MD_V2, "w") do io
        println(io, "# V2 replay — A + REGIME-AWARE B relaxation + near-term extreme inertia (multi-lead, causal)\n")
        println(io, "Fixes B's shallow bias on actively-deepening deep storms by lengthening the relaxation ",
                    "timescale τ=τ0·(1+max(0,−rate)/R0) during deepening (R0=", R0_V2, " nT/h, τ0=", TAU0_V2,
                    " h, cap ", TAU_MAX, " h). For 1–2 h forecasts with latest Dst≤", EXTREME_INERTIA_DST_NT,
                    " nT, the V2 point forecast uses persistence to respect near-term ring-current inertia. ",
                    "`improve` = paired RMSE(stronger of {pre-upgrade baseline, persistence}) − RMSE(V2), storm-cluster 95% CI; `fair`=max|v2_frozen−audit_baseline|.\n")
        @printf(io, "**Deep deepening subset (rate<%.0f & obs<−100, 6 h, n=%d): signed error mean(obs−pred) — V2 %+.1f nT vs plain-B %+.1f nT vs pre-upgrade baseline %+.1f nT.** Target ≈0 (no shallow bias).\n\n", MAIN_RATE, db.n, db.v2, plain_db.v2, db.baseline)
        println(io, "| lead [h] | regime | n | RMSE pre-upgrade baseline | RMSE V2 | RMSE persistence | stronger | improve [nT] (95% CI) | fair |")
        println(io, "|---|---|---|---|---|---|---|---|---|")
        println("\n  lead regime    n  base     v2   pers  strong    improve[CI]          fair")
        for h in LEADS
            sub_h = all_rows[all_rows.lead .== h, :]
            for (label, sub) in (("pooled", sub_h),
                                 ("main",   sub_h[isfinite.(sub_h.rate) .& (sub_h.rate .< MAIN_RATE), :]))
                c = _cell_v2(sub); c === nothing && continue
                @printf("  %3d  %-6s %4d %5.1f %5.1f %5.1f  %-8s %+5.2f [%+5.2f,%+5.2f]  %.2f\n",
                        h, label, c.n, c.rmse_baseline, c.rmse_v2, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.fair)
                @printf(io, "| %d | %s | %d | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] | %.2f |\n",
                        h, label, c.n, c.rmse_baseline, c.rmse_v2, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.fair)
            end
        end
        # R0 sweep: deep-bias and 6h-pooled improvement — pick the V2 R0 (bias≈0 AND gain retained).
        # Persist the sweep to CSV as well as the markdown so the −47.4 nT plain-B signed error and
        # the retained 6 h gain are traceable to a data artifact, not only the report table.
        println(io, "\n## R0 sweep (deep-bias at 6 h ; 6 h-pooled improvement vs persistence)\n")
        println(io, "| R0 [nT/h] | deep signed err [nT] | 6h-pooled improve (95% CI) |")
        println(io, "|---|---|---|")
        println("\n  R0 sweep (deep signed-err / 6h-pooled improve):")
        for r0 in (3.75, 7.5, 15.0, 1e9)   # 1e9 ≈ plain B (no regime awareness)
            rows = _run_v2(lib, ξ0, cal, luc, r0)
            b = _deep_bias(rows[rows.lead .== 6, :]); c6 = _cell_v2(rows[rows.lead .== 6, :])
            @printf("    R0=%-6s deep_err=%+6.1f   6h-pool %+5.2f [%+5.2f,%+5.2f]\n", string(r0), b.v2, c6.improve, c6.ci_lo, c6.ci_hi)
            @printf(io, "| %s | %+.1f | %+.2f [%+.2f, %+.2f] |\n", r0 >= 1e9 ? "∞ (plain B)" : string(r0), b.v2, c6.improve, c6.ci_lo, c6.ci_hi)
            push!(sweep_rows, (r0 = r0, plain_b = r0 >= 1e9,
                               deep_n = b.n, deep_signed_err_v2_nt = b.v2, deep_signed_err_baseline_nt = b.baseline,
                               rmse_baseline_6h_nt = c6.rmse_baseline, rmse_v2_6h_nt = c6.rmse_v2,
                               rmse_pers_6h_nt = c6.rmse_pers, improve_6h_nt = c6.improve,
                               improve_ci_lo_nt = c6.ci_lo, improve_ci_hi_nt = c6.ci_hi))
        end
    end
    CSV.write(OUT_CSV_R0, sweep_rows)
    println("  wrote ", OUT_CSV_V2, ", ", OUT_MD_V2, ", and ", OUT_CSV_R0)
    return all_rows
end

# ---- CRC self-test ----
_cal_v2 = Ref{Any}(nothing)
_calV2() = (_cal_v2[] === nothing && (_cal_v2[] = _load_calibration_for_model(LiveVerifyConfig(model = :v2))); _cal_v2[])
function _selftest_v2()
    lib, ξ0, i_decay = _shadow_library()
    slow = (V = 300.0, Bz = -12.0, By = 2.0, n = 6.0, Pdyn = 2.0)
    fut  = (V = 320.0, Bz = -24.0, By = 0.0, n = 8.0, Pdyn = 3.0)
    # Oracle 1 — CONTINUITY: force_frozen ⇒ v2-equivalent _shadow_forecast.
    for h in (1, 3, 6)
        a = _v2_forecast(lib, ξ0, -150.0, slow, _ -> fut, -148.0, _calV2(), h, -20.0; force_frozen = true)
        b = _shadow_forecast(lib, ξ0, i_decay, ξ0[i_decay], -150.0, slow, -148.0, _calV2(); nsteps = h)
        @assert a == b "continuity broken at h=$h: v2_frozen=$a baseline=$b"
    end
    # Oracle 2 — REGIME AWARENESS: at 6 h, a strongly-deepening issue (rate=−40) forecasts a DEEPER (more
    # negative) Dst than a recovering issue (rate=+10) from the same anchor — the τ-lengthening keeps injecting.
    deepening = _v2_forecast(lib, ξ0, -200.0, slow, _ -> nothing, -198.0, _calV2(), 6, -40.0)[1]
    recovering = _v2_forecast(lib, ξ0, -200.0, slow, _ -> nothing, -198.0, _calV2(), 6, +10.0)[1]
    @assert deepening < recovering "regime awareness inverted: deepening=$deepening !< recovering=$recovering"
    # Oracle 3 — recovery branch still relaxes vs the frozen baseline (B's win preserved): recovering 6 h is less
    # negative than the frozen tail.
    frozen6 = _v2_forecast(lib, ξ0, -200.0, slow, _ -> nothing, -198.0, _calV2(), 6, +10.0; force_frozen = true)[1]
    @assert recovering > frozen6 "recovery branch did not relax vs frozen ($recovering !> $frozen6)"
    # Oracle 4 — NEAR-TERM EXTREME INERTIA: once observed Dst is already in the
    # extreme-core range, 1-2 h forecasts use persistence; longer leads retain
    # the V2 tail.
    guarded2 = _v2_forecast(lib, ξ0, -250.0, slow, _ -> nothing, -250.0, _calV2(), 2, +10.0)[2]
    unguarded3 = _v2_forecast(lib, ξ0, -250.0, slow, _ -> nothing, -250.0, _calV2(), 3, +10.0)[2]
    @assert guarded2 == -250.0 "near-term extreme inertia guard did not serve persistence"
    @assert unguarded3 != -250.0 "near-term extreme inertia guard leaked into 3 h"
    println("  ✓ V2 self-test: continuity to pre-upgrade baseline, regime awareness, recovery relaxation, near-term extreme inertia")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest_v2()
    main_v2()
end
