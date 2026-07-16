# v2_improved_replay.jl — improved = Direction A (L1 look-ahead) + Direction B (driver-envelope relaxation),
# composed by PRECEDENCE (not additively, since both rewrite the early-step Bz):
#   * steps k <= floor(Δ): use the actual L1-knowable at-Earth driver (A, strict-causal, leakage-safe);
#   * steps k >  floor(Δ): relax the LAST L1-known driver toward 0 with timescale τ (B), over the frozen tail.
# So the measured L1 leading edge drives the first transit window, and envelope relaxation forecasts the
# unknown tail — A supplies short-lead skill, B supplies the 3-6 h tail. Fully causal & leakage-safe:
# k<=floor(Δ) uses only parcels measured at L1 by issue time; the tail uses only a fixed relaxation rule.
#
# CRC: force_frozen (Δ=0) + τ=∞ ⇒ frozen issue driver ⇒ v2. τ=∞ alone ⇒ A with the tail frozen at the L1
# edge. Δ=0 alone ⇒ pure B (relax from issue).
#
# Run from the package root: julia --project=. validation/operational/v2_improved_replay.jl

include(joinpath(@__DIR__, "v2_lookahead_replay.jl"))   # _driver_lookup, _transit_hours, L1_DIST_KM,
                                                        # STRICT_CAUSAL, _la_forecast, + all shared machinery

const OUT_CSV_IMPROVED = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_improved_replay_scored.csv")
const OUT_MD_IMPROVED  = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_improved_replay_report.md")
const TAU_IMPROVED     = 3.0      # B relaxation timescale (Bz autocorrelation ~3 h; from Direction B's principled sweep)

"""improved h-step forecast: A (actual L1 driver) for k<=floor(Δ), then B (relax the last L1-known driver toward 0,
timescale τ) for the frozen tail. Engine-faithful per-step + final Dst* clamp, then the issue-time v2 correction."""
function _improved_forecast(lib, ξ0, anchor_dst_star, issue_drv, future, latest_dst, cal, h::Int; tau = TAU_IMPROVED, force_frozen::Bool = false)
    Δ  = force_frozen ? 0.0 : _transit_hours(issue_drv.V)
    kΔ = floor(Int, Δ)                          # last fully-L1-resident hourly step
    last_known = issue_drv                       # driver at the L1-known edge (issue if kΔ=0)
    final_drv = issue_drv                        # driver of the final rollout step (target-step Pdyn)
    fc = init_assimilation(lib, ξ0, Int[], anchor_dst_star)
    for k in 1:h
        if k <= kΔ
            # A: actual L1-knowable driver — `future(k)` returns the arrival-hour record it+k-1; admit it
            # only if that whole hour is L1-measured by issue (k<=kΔ AND k<=transit(fut.V), the intra-hour
            # acceleration guard), else persist the last known driver (freeze).
            fut = future(k)
            drv_k = (fut !== nothing && k <= _transit_hours(fut.V)) ? fut : last_known
            last_known = drv_k
        else
            relax = exp(-(k - kΔ) / tau)                  # B: relax the last L1-known driver over the unknown tail
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
    return clamp(pred_dst, -2000.0, 50.0), clamp(pred_dst + corr, -2000.0, 50.0)
end

function replay_improved_storm(storm, lib, ξ0, cal, lookup)
    yr = year(storm.t1)
    plasma, mag, dst_times, dst_vals = _omni_replay_inputs(OMNI, yr - 1, yr)
    m = (dst_times .>= storm.t0 - Hour(6)) .& (dst_times .<= storm.t1 + Hour(7))
    rh = Int(ceil(Dates.value((storm.t1 + Hour(7)) - (storm.t0 - Hour(6))) / 3_600_000))
    df = replay_recent_table(plasma[m, :], mag[m, :], dst_times[m], dst_vals[m];
                             replay_hours = rh, horizons = LEADS, model = :v2, calibration = cal)
    df[!, :issue_dt] = DateTime.(df.issue_time_utc)
    df = df[(df.issue_dt .>= storm.t0) .& (df.issue_dt .<= storm.t1), :]
    sort!(df, [:issue_dt, :model_step_hours])
    dst_map = Dict{DateTime,Float64}(zip(dst_times[m], Float64.(dst_vals[m])))  # true hourly Dst series
    out = DataFrame(storm = String[], issue_utc = DateTime[], lead = Int[], obs = Float64[], v2 = Float64[],
                    improved = Float64[], improved_frozen = Float64[], persistence = Float64[], transit_h = Float64[], rate = Float64[])
    for it in sort!(unique(df.issue_dt))
        g = df[df.issue_dt .== it, :]; r1 = g[1, :]
        issue_drv = (V = Float64(r1.V_kms), Bz = Float64(r1.Bz_nt), By = Float64(r1.By_nt),
                     n = Float64(r1.n_cm3), Pdyn = Float64(r1.Pdyn_npa))
        latest = Float64(r1.latest_dst_nt)
        all(isfinite, (issue_drv.V, issue_drv.Bz, issue_drv.By, issue_drv.n, issue_drv.Pdyn, latest)) || continue
        anchor_star = pressure_correct_dst([latest], [issue_drv.Pdyn])[1]
        # Rollout step k is driven by the arrival-hour record tagged it+k-1 (fully L1-measured iff k<=Δ).
        future = k -> get(lookup, it + Hour(k - 1), nothing)
        # Recent Dst rate = true 1 h delta from the Dst series (NaN when the prior hour is absent).
        prev1 = get(dst_map, it - Hour(1), NaN)
        rate = isfinite(prev1) ? latest - prev1 : NaN
        for r in eachrow(g)
            (ismissing(r.observation_dst_nt) || ismissing(r.v2_pred_dst_nt)) && continue
            isfinite(Float64(r.observation_dst_nt)) && isfinite(Float64(r.v2_pred_dst_nt)) || continue
            h = Int(r.model_step_hours)
            _, improved  = _improved_forecast(lib, ξ0, anchor_star, issue_drv, future, latest, cal, h)
            _, improvedf = _improved_forecast(lib, ξ0, anchor_star, issue_drv, future, latest, cal, h; tau = Inf, force_frozen = true)
            isfinite(improved) && isfinite(improvedf) || continue
            push!(out, (storm.name, it, h, Float64(r.observation_dst_nt), Float64(r.v2_pred_dst_nt),
                        improved, improvedf, latest, _transit_hours(issue_drv.V), rate))
        end
    end
    return out
end

function _cell_improved(rows)
    nrow(rows) == 0 && return nothing
    ev2 = rows.obs .- rows.v2; e3 = rows.obs .- rows.improved; ep = rows.obs .- rows.persistence
    rv2, r3, rp = _rmse(ev2), _rmse(e3), _rmse(ep)
    strong_pers = rp <= rv2
    Δ, lo, hi = paired_improvement(strong_pers ? ep : ev2, e3; storms = rows.storm)
    return (n = nrow(rows), rmse_v2 = rv2, rmse_improved = r3, rmse_pers = rp, stronger = strong_pers ? "pers" : "v2",
            improve = Δ, ci_lo = lo, ci_hi = hi, fair = maximum(abs.(rows.improved_frozen .- rows.v2)))
end

function main_improved()
    lib, ξ0, i_decay = _shadow_library()
    cal = _load_calibration_for_model(LiveVerifyConfig(model = :v2))
    println("improved = A (L1 look-ahead, k≤floor(Δ)) + B (relax tail, τ=", TAU_IMPROVED, ") vs v2 + persistence, leads ", LEADS, "\n", "="^80)
    luc = Dict{Int,Any}(); all_rows = DataFrame()
    for s in STORMS
        yr = year(s.t1)
        haskey(luc, yr) || (luc[yr] = _driver_lookup(yr))
        rows = replay_improved_storm(s, lib, ξ0, cal, luc[yr])
        append!(all_rows, rows)
        @printf("  %-24s rows=%d\n", s.name, nrow(rows))
    end
    CSV.write(OUT_CSV_IMPROVED, all_rows)
    open(OUT_MD_IMPROVED, "w") do io
        println(io, "# improved — A (L1 look-ahead) + B (driver-envelope relaxation), precedence composition (multi-lead, causal)\n")
        println(io, "Steps k≤floor(Δ) use the actual L1-knowable driver (A, strict-causal); the tail relaxes the last ",
                    "L1-known driver toward 0 with τ=", TAU_IMPROVED, " h (B). `improve` = paired RMSE(stronger baseline) − ",
                    "RMSE(improved), storm-cluster 95% CI; `fair` = max|improved_frozen − v2| (Δ=0,τ=∞ continuity).\n")
        println(io, "| lead [h] | regime | n | RMSE v2 | RMSE improved | RMSE pers | stronger | improve [nT] (95% CI) | fair |")
        println(io, "|---|---|---|---|---|---|---|---|---|")
        println("\n  lead regime    n   v2    improved    pers  strong  improve[CI]            fair")
        for h in LEADS
            sub_h = all_rows[all_rows.lead .== h, :]
            for (label, sub) in (("pooled", sub_h),
                                 ("main",   sub_h[isfinite.(sub_h.rate) .& (sub_h.rate .< MAIN_RATE), :]))
                c = _cell_improved(sub); c === nothing && continue
                @printf("  %3d  %-6s %4d %5.1f %5.1f %5.1f  %-4s  %+5.2f [%+5.2f,%+5.2f]  %.2f\n",
                        h, label, c.n, c.rmse_v2, c.rmse_improved, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.fair)
                @printf(io, "| %d | %s | %d | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] | %.2f |\n",
                        h, label, c.n, c.rmse_v2, c.rmse_improved, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.fair)
            end
        end
        max_fair = maximum(abs.(all_rows.improved_frozen .- all_rows.v2))
        beats = any(LEADS) do h
            any((("pooled", all_rows[all_rows.lead .== h, :]),
                 ("main",   all_rows[(all_rows.lead .== h) .& isfinite.(all_rows.rate) .& (all_rows.rate .< MAIN_RATE), :]))) do (_, sub)
                c = _cell_improved(sub); c !== nothing && c.ci_lo > 0
            end
        end
        verdict = max_fair > 2.0 ? "INCONCLUSIVE: fairness gap $(round(max_fair;digits=2)) nT" :
                  beats ? "PROMISING: improved (A+B) beats the stronger of {v2, persistence} with CI>0 at some lead/regime" :
                  "NO GAIN over the stronger baseline at any lead/regime"
        println(io, "\nMax fairness gap max|improved_frozen − v2| = ", round(max_fair; digits=2), " nT.\n")
        println(io, "**improved verdict:** ", verdict)
        println("\n  improved verdict: ", verdict)
    end
    println("  wrote ", OUT_CSV_IMPROVED, " and ", OUT_MD_IMPROVED)
    return all_rows
end

# ---- CRC self-test ----
_cal_improved = Ref{Any}(nothing)
_calIMPROVED() = (_cal_improved[] === nothing && (_cal_improved[] = _load_calibration_for_model(LiveVerifyConfig(model = :v2))); _cal_improved[])
function _selftest_improved()
    lib, ξ0, i_decay = _shadow_library()
    slow = (V = 300.0, Bz = -10.0, By = 2.0, n = 6.0, Pdyn = 2.0)   # Δ≈1.39 ⇒ A window fires
    fut  = (V = 320.0, Bz = -22.0, By = 0.0, n = 8.0, Pdyn = 3.0)   # stronger incoming southward driver
    # Oracle 1 — CONTINUITY: force_frozen (Δ=0) + τ=∞ ⇒ frozen issue driver ⇒ v2-equivalent _shadow_forecast.
    for h in (1, 3, 6)
        a = _improved_forecast(lib, ξ0, -150.0, slow, _ -> fut, -148.0, _calIMPROVED(), h; tau = Inf, force_frozen = true)
        b = _shadow_forecast(lib, ξ0, i_decay, ξ0[i_decay], -150.0, slow, -148.0, _calIMPROVED(); nsteps = h)
        @assert a == b "continuity broken at h=$h: improved_frozen=$a v2=$b"
    end
    # Oracle 2 — A WINDOW acts: with Δ≥1 and a stronger incoming southward driver, the 1 h forecast is more
    # negative than the all-frozen v2 (the L1 leading edge deepens it).
    la1 = _improved_forecast(lib, ξ0, -150.0, slow, _ -> fut,     -148.0, _calIMPROVED(), 1)[1]
    fr1 = _improved_forecast(lib, ξ0, -150.0, slow, _ -> fut,     -148.0, _calIMPROVED(), 1; tau = Inf, force_frozen = true)[1]
    @assert la1 < fr1 "A window did not deepen the 1 h forecast ($la1 !< $fr1)"
    # Oracle 3 — B TAIL relaxes: with Δ=0 (no A window) and finite τ, the 6 h forecast is LESS negative than the
    # frozen v2 (relaxing the southward issue Bz avoids v2's over-injection).
    relaxed = _improved_forecast(lib, ξ0, -150.0, slow, _ -> fut, -148.0, _calIMPROVED(), 6; tau = 3.0, force_frozen = true)[1]
    frozen6 = _improved_forecast(lib, ξ0, -150.0, slow, _ -> fut, -148.0, _calIMPROVED(), 6; tau = Inf, force_frozen = true)[1]
    @assert relaxed > frozen6 "B tail did not relax the 6 h forecast ($relaxed !> $frozen6)"
    println("  ✓ improved self-test: continuity to v2 (Δ=0,τ=∞), A window deepens short lead, B tail relaxes long lead")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest_improved()
    main_improved()
end
