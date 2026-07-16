# v2_envelope_replay.jl — Direction B: driver-envelope forecast vs V2 + persistence.
#
# v2 FREEZES the issue-time driver across the whole rollout, so a transient southward Bz keeps injecting ring
# current for the full 6 h — over-driving the storm at long lead. The IMF Bz/By are mean-zero and turbulent
# (autocorrelation a few hours), so a better forecast of the DRIVER than "freeze" is "relax Bz/By toward 0
# with timescale τ" while persisting the more-autocorrelated speed/density. This integrates the discovered
# ODE with the forecast driver. It is fully causal (uses only the issue-time driver + a fixed relaxation
# rule — no future information).
#
# CRC continuity: τ→∞ ⇒ exp(-k/τ)=1 ⇒ driver frozen ⇒ reduces EXACTLY to v2.
#
# Run from the package root: julia --project=. validation/operational/v2_envelope_replay.jl

include(joinpath(@__DIR__, "ekf_storm_replay.jl"))   # STORMS, LEADS, MAIN_RATE, _rmse, OMNI, paired_improvement,
                                                     # _shadow_library, _shadow_forecast

const OUT_CSV_B = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_envelope_replay_scored.csv")
const OUT_MD_B  = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_envelope_replay_report.md")
# IMF relaxation timescales to sweep (hours). Inf = no relaxation = v2 (continuity check). Principled range:
# solar-wind Bz autocorrelation is a few hours, so τ∈{1,3,6} brackets it; not tuned on the test storms.
const TAU_SWEEP = [1.0, 3.0, 6.0, Inf]

"""h-step forecast that FORECASTS the driver: relax Bz/By toward 0 with timescale τ, persist V/n/Pdyn, then
integrate the discovered ODE (engine-faithful per-step + final Dst* clamp) and add the issue-time v2
correction. τ=Inf freezes the driver ⇒ identical to v2."""
function _envelope_forecast(lib, ξ0, anchor_dst_star, issue_drv, latest_dst, cal, h::Int; tau::Float64)
    fc = init_assimilation(lib, ξ0, Int[], anchor_dst_star)
    for k in 1:h
        relax = exp(-k / tau)                       # 1 at τ=Inf (freeze); →0 fast for small τ
        drv_k = (V = issue_drv.V, Bz = issue_drv.Bz * relax, By = issue_drv.By * relax,
                 n = issue_drv.n, Pdyn = issue_drv.Pdyn)
        assimilation_predict!(fc, drv_k)
        fc.mean[1] = clamp(fc.mean[1], -2000.0, 50.0)
    end
    pred_dst_star = current_dst(fc)
    pred_dst = pred_dst_star + 7.26 * sqrt(max(issue_drv.Pdyn, 0.0)) - 11.0
    feat_df = DataFrame(latest_dst_nt = [latest_dst], V_kms = [issue_drv.V], Bz_nt = [issue_drv.Bz],
                        By_nt = [issue_drv.By], n_cm3 = [issue_drv.n], Pdyn_npa = [issue_drv.Pdyn])
    prep = SolarSINDy.add_operational_v2_features!(feat_df)
    feats = NamedTuple{Tuple(cal.feature_names)}(Tuple(Float64(prep[1, c]) for c in cal.feature_names))
    corr = SolarSINDy.operational_v2_correction(cal, feats)
    return clamp(pred_dst, -2000.0, 50.0), clamp(pred_dst + corr, -2000.0, 50.0)
end

function replay_env_storm(storm, lib, ξ0, cal, tau)
    yr = year(storm.t1)
    plasma, mag, dst_times, dst_vals = _omni_replay_inputs(OMNI, yr - 1, yr)
    m = (dst_times .>= storm.t0 - Hour(6)) .& (dst_times .<= storm.t1 + Hour(7))
    rh = Int(ceil(Dates.value((storm.t1 + Hour(7)) - (storm.t0 - Hour(6))) / 3_600_000))
    df = replay_recent_table(plasma[m, :], mag[m, :], dst_times[m], dst_vals[m];
                             replay_hours = rh, horizons = LEADS, model = :v2, calibration = cal)
    df[!, :issue_dt] = DateTime.(df.issue_time_utc)
    df = df[(df.issue_dt .>= storm.t0) .& (df.issue_dt .<= storm.t1), :]
    sort!(df, [:issue_dt, :model_step_hours])
    out = DataFrame(storm = String[], issue_utc = DateTime[], lead = Int[], obs = Float64[], v2 = Float64[],
                    b = Float64[], b_frozen = Float64[], persistence = Float64[], rate = Float64[])
    prev_anchor = NaN
    for it in sort!(unique(df.issue_dt))
        g = df[df.issue_dt .== it, :]; r1 = g[1, :]
        issue_drv = (V = Float64(r1.V_kms), Bz = Float64(r1.Bz_nt), By = Float64(r1.By_nt),
                     n = Float64(r1.n_cm3), Pdyn = Float64(r1.Pdyn_npa))
        latest = Float64(r1.latest_dst_nt)
        all(isfinite, (issue_drv.V, issue_drv.Bz, issue_drv.By, issue_drv.n, issue_drv.Pdyn, latest)) ||
            (prev_anchor = NaN; continue)
        anchor_star = pressure_correct_dst([latest], [issue_drv.Pdyn])[1]
        rate = isfinite(prev_anchor) ? latest - prev_anchor : NaN
        prev_anchor = latest
        for r in eachrow(g)
            (ismissing(r.observation_dst_nt) || ismissing(r.v2_pred_dst_nt)) && continue
            isfinite(Float64(r.observation_dst_nt)) && isfinite(Float64(r.v2_pred_dst_nt)) || continue
            h = Int(r.model_step_hours)
            _, b  = _envelope_forecast(lib, ξ0, anchor_star, issue_drv, latest, cal, h; tau = tau)
            _, bf = _envelope_forecast(lib, ξ0, anchor_star, issue_drv, latest, cal, h; tau = Inf)
            isfinite(b) && isfinite(bf) || continue
            push!(out, (storm.name, it, h, Float64(r.observation_dst_nt), Float64(r.v2_pred_dst_nt), b, bf, latest, rate))
        end
    end
    return out
end

function _cell_b(rows)
    nrow(rows) == 0 && return nothing
    ev2 = rows.obs .- rows.v2; eb = rows.obs .- rows.b; ep = rows.obs .- rows.persistence
    rv2, rb, rp = _rmse(ev2), _rmse(eb), _rmse(ep)
    strong_pers = rp <= rv2
    Δ, lo, hi = paired_improvement(strong_pers ? ep : ev2, eb)
    return (n = nrow(rows), rmse_v2 = rv2, rmse_b = rb, rmse_pers = rp, stronger = strong_pers ? "pers" : "v2",
            improve = Δ, ci_lo = lo, ci_hi = hi, fair = maximum(abs.(rows.b_frozen .- rows.v2)))
end

function _run_tau(lib, ξ0, cal, tau)
    all_rows = DataFrame()
    for s in STORMS
        append!(all_rows, replay_env_storm(s, lib, ξ0, cal, tau))
    end
    return all_rows
end

function main_env()
    lib, ξ0, i_decay = _shadow_library()
    cal = _load_calibration_for_model(LiveVerifyConfig(model = :v2))
    println("Direction B — driver-envelope forecast (relax Bz/By, τ) vs v2 + persistence, leads ", LEADS, "\n", "="^80)
    best_rows = _run_tau(lib, ξ0, cal, 3.0)     # τ=3 h primary (mid of the principled range); artifact written for it
    CSV.write(OUT_CSV_B, best_rows)
    open(OUT_MD_B, "w") do io
        println(io, "# Direction B — driver-envelope forecast vs V2 + persistence (multi-lead, causal)\n")
        println(io, "v2 freezes the driver; this relaxes Bz/By toward 0 with timescale τ (persisting V/n/Pdyn) ",
                    "and integrates the ODE. Fully causal (issue-time driver + fixed rule). τ=3 h shown; ",
                    "`fair` = max|b_frozen − v2| (τ=∞ continuity).\n")
        println(io, "| lead [h] | regime | n | RMSE v2 | RMSE B | RMSE pers | stronger | improve [nT] (95% CI) | fair |")
        println(io, "|---|---|---|---|---|---|---|---|---|")
        println("\n  τ=3h:  lead regime    n   v2    B     pers  strong  improve[CI]            fair")
        for h in LEADS
            sub_h = best_rows[best_rows.lead .== h, :]
            for (label, sub) in (("pooled", sub_h),
                                 ("main",   sub_h[isfinite.(sub_h.rate) .& (sub_h.rate .< MAIN_RATE), :]))
                c = _cell_b(sub); c === nothing && continue
                @printf("         %3d  %-6s %4d %5.1f %5.1f %5.1f  %-4s  %+5.2f [%+5.2f,%+5.2f]  %.2f\n",
                        h, label, c.n, c.rmse_v2, c.rmse_b, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.fair)
                @printf(io, "| %d | %s | %d | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] | %.2f |\n",
                        h, label, c.n, c.rmse_v2, c.rmse_b, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.fair)
            end
        end
        # τ sweep: 3h-main + 6h-main improvement vs stronger baseline (the tail where B is meant to help)
        println(io, "\n## τ sensitivity sweep (3h-main and 6h-main improvement vs stronger baseline)\n")
        println(io, "| τ [h] | 3h-main improve (95% CI) | 6h-main improve (95% CI) |")
        println(io, "|---|---|---|")
        println("\n  τ sweep (3h-main / 6h-main improve vs stronger):")
        for tau in TAU_SWEEP
            rows = _run_tau(lib, ξ0, cal, tau)
            c3 = _cell_b(rows[(rows.lead .== 3) .& isfinite.(rows.rate) .& (rows.rate .< MAIN_RATE), :])
            c6 = _cell_b(rows[(rows.lead .== 6) .& isfinite.(rows.rate) .& (rows.rate .< MAIN_RATE), :])
            @printf("    τ=%-4s  3h-main %+6.2f [%+6.2f,%+6.2f]   6h-main %+6.2f [%+6.2f,%+6.2f]\n",
                    string(tau), c3.improve, c3.ci_lo, c3.ci_hi, c6.improve, c6.ci_lo, c6.ci_hi)
            @printf(io, "| %s | %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] |\n",
                    string(tau), c3.improve, c3.ci_lo, c3.ci_hi, c6.improve, c6.ci_lo, c6.ci_hi)
        end
        println(io, "\nτ=∞ is the v2 continuity check (improvement ≈ 0). A τ with CI>0 at 3–6 h main would mean ",
                    "forecasting the driver beats freezing it where v2 currently loses to persistence.")
    end
    println("  wrote ", OUT_CSV_B, " and ", OUT_MD_B)
    return best_rows
end

# ---- CRC self-test ----
_cal_b = Ref{Any}(nothing)
_calB() = (_cal_b[] === nothing && (_cal_b[] = _load_calibration_for_model(LiveVerifyConfig(model = :v2))); _cal_b[])
function _selftest_env()
    lib, ξ0, i_decay = _shadow_library()
    drv = (V = 500.0, Bz = -12.0, By = 3.0, n = 6.0, Pdyn = 2.0)
    # Oracle 1 — CONTINUITY: τ=Inf freezes the driver ⇒ reproduces the v2-equivalent _shadow_forecast.
    for h in (1, 3, 6)
        a = _envelope_forecast(lib, ξ0, -150.0, drv, -148.0, _calB(), h; tau = Inf)
        b = _shadow_forecast(lib, ξ0, i_decay, ξ0[i_decay], -150.0, drv, -148.0, _calB(); nsteps = h)
        @assert a == b "continuity broken at h=$h: B(τ=∞)=$a v2-equiv=$b"
    end
    # Oracle 2 — RELAXATION reduces injection: with a finite τ, the southward Bz decays over the rollout, so a
    # 6 h forecast is LESS negative (less over-injection) than the frozen v2 when Bz is southward.
    relaxed = _envelope_forecast(lib, ξ0, -150.0, drv, -148.0, _calB(), 6; tau = 3.0)[1]
    frozen  = _envelope_forecast(lib, ξ0, -150.0, drv, -148.0, _calB(), 6; tau = Inf)[1]
    @assert relaxed > frozen "relaxing a southward Bz should make the 6h forecast less negative ($relaxed !> $frozen)"
    # Oracle 3 — τ→0 limit: near-instant relaxation kills the injection, so the forecast is no more negative
    # than a zero-Bz rollout from the same anchor (driver becomes V/n/Pdyn only within a step).
    fast = _envelope_forecast(lib, ξ0, -150.0, drv, -148.0, _calB(), 6; tau = 0.01)[1]
    @assert fast >= relaxed - 1e-6 "faster relaxation should not deepen vs slower ($fast vs $relaxed)"
    println("  ✓ Direction B self-test: continuity to v2 (τ=∞), relaxation reduces over-injection, τ→0 limit sane")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest_env()
    main_env()
end
