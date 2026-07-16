# v2_lookahead_replay.jl — Direction A: L1 transit look-ahead driver vs V2 + persistence.
#
# v2 freezes the solar-wind driver for the entire multi-hour rollout (forecast.jl forecast_ahead). But the
# live feed is L1 data (~1.5e6 km sunward of Earth), so at issue time t the at-Earth wind for the next
# Δ = L1_dist/V hours is ALREADY measured at L1 — knowable, not a forecast. This variant drives the first
# transit-window of the rollout with the actual (L1-knowable) OMNI driver instead of freezing, then freezes
# beyond. Δ is sub-hourly-to-~1h, so on the HOURLY OMNI replay this is a leakage-safe LOWER BOUND on the live
# advantage (its real value lives in sub-hourly L1 data); a flat result here means resolution, not a broken
# mechanism.
#
# Leakage safety: at rollout step k (covering at-Earth hour [t+k-1, t+k]), only the fraction within [t, t+Δ]
# is L1-knowable, so the step driver is a fraction-Δ blend of the actual future OMNI[t+k] and the frozen
# issue-time driver: frac_k = clamp(Δ - (k-1), 0, 1). frac_k=0 (Δ=0) reduces EXACTLY to v2 (continuity oracle).
#
# Run from the package root: julia --project=. validation/operational/v2_lookahead_replay.jl

include(joinpath(@__DIR__, "ekf_storm_replay.jl"))   # STORMS, LEADS, MAIN_RATE, _rmse, _ffill!, OMNI, _cell,
                                                     # paired_improvement, _shadow_library, _shadow_forecast

const OUT_CSV_LA = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_lookahead_replay_scored.csv")
const OUT_MD_LA  = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_lookahead_replay_report.md")
# Driver-lookup shift direction: +1 = real future look-ahead; -1 = PAST (leakage negative control — a
# genuine look-ahead gain MUST collapse to ~0 under a past shift, since the past is already in the frozen driver).
const SHIFT_DIR  = parse(Int, get(ENV, "LA_SHIFT", "1"))
# STRICT-CAUSAL blend (default, leakage-safe): use the actual OMNI[t+k] ONLY for hourly steps fully inside the
# L1-known window (k <= Δ), i.e. parcels already measured at L1 by issue time t. The earlier fractional blend
# (LA_STRICT=false) weighted OMNI[t+k] by frac=clamp(Δ-(k-1),0,1); on hourly OMNI that injects the hour-END
# value OMNI[t+1] (which left L1 at t+1-Δ > t) for the 77% of issues with Δ<1 — a subtle look-ahead beyond the
# t+Δ horizon that the past-shift control cannot detect. Strict-causal removes it (reduces to v2 for Δ<1).
const STRICT_CAUSAL = parse(Bool, get(ENV, "LA_STRICT", "true"))
const L1_DIST_KM = 1.5e6                  # Sun-Earth L1 standoff from Earth (km)
_transit_hours(V) = (isfinite(V) && V > 0) ? L1_DIST_KM / V / 3600.0 : 0.0   # Δ = L1->Earth transit time [h]

"Per-storm-year time->driver lookup (ffilled), for the actual at-Earth driver at issue+k hours."
function _driver_lookup(yr::Int)
    # Causal cleaning: forward-fill short gaps (no post-gap/future value) so the
    # at-Earth look-ahead driver is built only from L1-measurable samples; the
    # subsequent _ffill! covers longer gaps, also causally.
    df = parse_omni2(OMNI; year_start = yr - 1, year_end = yr); clean_omni_data!(df; causal = true)
    V  = _ffill!(Float64.(coalesce.(df.V, NaN)));  Bz = _ffill!(Float64.(coalesce.(df.Bz, NaN)))
    By = _ffill!(Float64.(coalesce.(df.By, NaN))); nn = _ffill!(Float64.(coalesce.(df.n, NaN)))
    Pd = _ffill!(Float64.(coalesce.(df.Pdyn, NaN)))
    d = Dict{DateTime,NamedTuple{(:V, :Bz, :By, :n, :Pdyn),NTuple{5,Float64}}}()
    for (k, t) in enumerate(df.datetime)
        all(isfinite, (V[k], Bz[k], By[k], nn[k], Pd[k])) || continue
        d[t] = (V = V[k], Bz = Bz[k], By = By[k], n = nn[k], Pdyn = Pd[k])
    end
    return d
end

_blend(frac, fut, base) = (V = frac * fut.V + (1 - frac) * base.V, Bz = frac * fut.Bz + (1 - frac) * base.Bz,
                           By = frac * fut.By + (1 - frac) * base.By, n = frac * fut.n + (1 - frac) * base.n,
                           Pdyn = frac * fut.Pdyn + (1 - frac) * base.Pdyn)

"""h-step forecast with the L1 transit look-ahead driver (engine-faithful per-step + final Dst* clamp, then
the issue-time v2 correction — identical to v2 except the v1 rollout uses look-ahead drivers within the
transit window). `future` maps step k -> actual at-Earth OMNI driver at issue+k (or nothing). force_frozen
disables the look-ahead (Δ=0) for the continuity/fairness check, reproducing v2."""
function _la_forecast(lib, ξ0, anchor_dst_star, issue_drv, future, latest_dst, cal, h::Int; force_frozen::Bool = false)
    Δ = force_frozen ? 0.0 : _transit_hours(issue_drv.V)
    fc = init_assimilation(lib, ξ0, Int[], anchor_dst_star)
    final_drv = issue_drv                     # driver of the final rollout step (target-step Pdyn, Eq. 4)
    for k in 1:h
        # `future(k)` now returns the ARRIVAL-hour record tagged it+k-1 (the row covering at-Earth
        # interval [it+k-1, it+k)); strict-causal admits it only if that whole hour is L1-measured by
        # issue time, gated on BOTH the issue-time transit (k<=Δ) AND the admitted record's own speed
        # (intra-hour acceleration guard). Fractional (legacy, NOT leakage-safe on hourly data): clamp.
        fut = future(k)
        if STRICT_CAUSAL
            admit = fut !== nothing && k <= Δ && k <= _transit_hours(fut.V)
            drv_k = admit ? fut : issue_drv    # frac∈{0,1}; frac=1 blend == fut
        else
            frac = fut !== nothing ? clamp(Δ - (k - 1), 0.0, 1.0) : 0.0
            drv_k = frac > 0 ? _blend(frac, fut, issue_drv) : issue_drv
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

function replay_la_storm(storm, lib, ξ0, cal, lookup)
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
    out = DataFrame(storm = String[], issue_utc = DateTime[], lead = Int[], obs = Float64[], v2 = Float64[],
                    la = Float64[], la_frozen = Float64[], persistence = Float64[], transit_h = Float64[], rate = Float64[])
    for it in sort!(unique(df.issue_dt))
        g = df[df.issue_dt .== it, :]; r1 = g[1, :]
        issue_drv = (V = Float64(r1.V_kms), Bz = Float64(r1.Bz_nt), By = Float64(r1.By_nt),
                     n = Float64(r1.n_cm3), Pdyn = Float64(r1.Pdyn_npa))
        latest = Float64(r1.latest_dst_nt)
        all(isfinite, (issue_drv.V, issue_drv.Bz, issue_drv.By, issue_drv.n, issue_drv.Pdyn, latest)) || continue
        anchor_star = pressure_correct_dst([latest], [issue_drv.Pdyn])[1]
        # Step k is driven by the arrival-hour record it+k-1 (forward look-ahead); the past shift
        # (SHIFT_DIR=-1) feeds record it-k as a leakage negative control (genuine gain must collapse).
        future = k -> get(lookup, it + Hour(SHIFT_DIR == 1 ? (k - 1) : SHIFT_DIR * k), nothing)
        # Recent Dst rate = true 1 h delta from the Dst series (NaN when the prior hour is absent),
        # matching the live dst_delta_1h guard; never a multi-hour delta misread as nT/h across a gap.
        prev1 = get(dst_map, it - Hour(1), NaN)
        rate = isfinite(prev1) ? latest - prev1 : NaN
        for r in eachrow(g)
            (ismissing(r.observation_dst_nt) || ismissing(r.v2_pred_dst_nt)) && continue
            isfinite(Float64(r.observation_dst_nt)) && isfinite(Float64(r.v2_pred_dst_nt)) || continue
            h = Int(r.model_step_hours)
            _, la   = _la_forecast(lib, ξ0, anchor_star, issue_drv, future, latest, cal, h)
            _, lafr = _la_forecast(lib, ξ0, anchor_star, issue_drv, future, latest, cal, h; force_frozen = true)
            isfinite(la) && isfinite(lafr) || continue
            push!(out, (storm.name, it, h, Float64(r.observation_dst_nt), Float64(r.v2_pred_dst_nt),
                        la, lafr, latest, _transit_hours(issue_drv.V), rate))
        end
    end
    return out
end

"Cell stats for the look-ahead arm: RMSE per method, A vs the STRONGER baseline with paired CI, fairness gap."
function _cell_la(rows)
    nrow(rows) == 0 && return nothing
    ev2 = rows.obs .- rows.v2; ela = rows.obs .- rows.la; ep = rows.obs .- rows.persistence
    rv2, ra, rp = _rmse(ev2), _rmse(ela), _rmse(ep)
    strong_pers = rp <= rv2
    Δ, lo, hi = paired_improvement(strong_pers ? ep : ev2, ela; storms = rows.storm)
    return (n = nrow(rows), rmse_v2 = rv2, rmse_ekf = ra, rmse_pers = rp, stronger = strong_pers ? "pers" : "v2",
            improve = Δ, ci_lo = lo, ci_hi = hi, fair = maximum(abs.(rows.la_frozen .- rows.v2)))
end

function main_la()
    lib, ξ0, i_decay = _shadow_library()
    cal = _load_calibration_for_model(LiveVerifyConfig(model = :v2))
    println("Direction A — L1 transit look-ahead driver vs v2 + persistence, leads ", LEADS,
            "  (L1=", L1_DIST_KM, " km; Δ=L1/V)\n", "="^80)
    luc = Dict{Int,Any}(); all_rows = DataFrame()
    for s in STORMS
        yr = year(s.t1)
        haskey(luc, yr) || (luc[yr] = _driver_lookup(yr))
        rows = replay_la_storm(s, lib, ξ0, cal, luc[yr])
        append!(all_rows, rows)
        @printf("  %-24s rows=%d  median transit=%.2f h\n", s.name, nrow(rows),
                nrow(rows) > 0 ? median(rows.transit_h) : NaN)
    end
    CSV.write(OUT_CSV_LA, all_rows)
    open(OUT_MD_LA, "w") do io
        println(io, "# Direction A — L1 transit look-ahead driver vs V2 + persistence (multi-lead, causal)\n")
        println(io, "v2 freezes the driver across the rollout; this drives the L1-transit window (Δ=L1/V) with the ",
                    "actual at-Earth OMNI driver for hourly steps FULLY inside the L1-known window (strict-causal: ",
                    "k≤Δ; reduces to v2 for Δ<1). On hourly OMNI most storm wind is too fast for a ≥1 h look-ahead, ",
                    "so the full-pooled gain is marginal; the genuinely-knowable gain is isolated in the Δ≥1 subset ",
                    "below. `improve` = paired RMSE(stronger baseline) − RMSE(A), storm-cluster 95% CI; `fair` = max|la_frozen − v2| ",
                    "confirms Δ=0 reproduces v2. (The earlier fractional blend, LA_STRICT=false, over-stated the gain ",
                    "by sampling the hour-end OMNI[t+1] for Δ<1 — wind beyond the t+Δ L1 horizon.)\n")
        println(io, "| lead [h] | regime | n | RMSE v2 | RMSE A | RMSE pers | stronger | improve [nT] (95% CI) | fair |")
        println(io, "|---|---|---|---|---|---|---|---|---|")
        println("\n  lead regime    n   v2    A     pers  strong  improve[CI]            fair")
        for h in LEADS
            sub_h = all_rows[all_rows.lead .== h, :]
            for (label, sub) in (("pooled", sub_h),
                                 ("main",   sub_h[isfinite.(sub_h.rate) .& (sub_h.rate .< MAIN_RATE), :]))
                c = _cell_la(sub); c === nothing && continue
                @printf("  %3d  %-6s %4d %5.1f %5.1f %5.1f  %-4s  %+5.2f [%+5.2f,%+5.2f]  %.2f\n",
                        h, label, c.n, c.rmse_v2, c.rmse_ekf, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.fair)
                @printf(io, "| %d | %s | %d | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] | %.2f |\n",
                        h, label, c.n, c.rmse_v2, c.rmse_ekf, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.fair)
            end
        end
        # Leakage-safe defensible subset: issues with Δ≥1 h, where the at-Earth target hour OMNI[t+1] is
        # genuinely L1-resident at issue time. The full-pooled gain is marginal under strict-causal because most
        # storm wind is too fast for ≥1 h hourly look-ahead; this subset isolates the genuinely-knowable gain.
        sub1 = all_rows[all_rows.transit_h .>= 1.0, :]
        println(io, "\n## Δ≥1 h subset — genuinely L1-resident look-ahead (leakage-safe)\n")
        println(io, "| lead [h] | n | RMSE v2 | RMSE A | RMSE pers | stronger | improve [nT] (95% CI) |")
        println(io, "|---|---|---|---|---|---|---|")
        println("\n  Δ≥1 leakage-safe subset (n=", nrow(sub1[sub1.lead .== 1, :]), " issues/lead):")
        for h in (1, 2, 3)
            c = _cell_la(sub1[sub1.lead .== h, :]); c === nothing && continue
            @printf(io, "| %d | %d | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] |\n",
                    h, c.n, c.rmse_v2, c.rmse_ekf, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi)
            @printf("    lead %d  n=%d  improve=%+.2f [%+.2f,%+.2f] vs %s\n",
                    h, c.n, c.improve, c.ci_lo, c.ci_hi, c.stronger)
        end
        max_fair = maximum(abs.(all_rows.la_frozen .- all_rows.v2))
        beats = any(LEADS) do h
            any((("pooled", all_rows[all_rows.lead .== h, :]),
                 ("main",   all_rows[(all_rows.lead .== h) .& isfinite.(all_rows.rate) .& (all_rows.rate .< MAIN_RATE), :]))) do (_, sub)
                c = _cell_la(sub); c !== nothing && c.ci_lo > 0
            end
        end
        verdict = max_fair > 2.0 ? "INCONCLUSIVE: fairness gap $(round(max_fair;digits=2)) nT" :
                  beats ? "PROMISING: beats the stronger baseline with CI>0 at some lead/regime" :
                  "NO GAIN: at hourly resolution the L1 look-ahead does not beat the stronger baseline with CI>0 (transit Δ<1 h at storm speeds rounds out; needs sub-hourly L1 data)"
        println(io, "\nMax fairness gap max|la_frozen − v2| = ", round(max_fair; digits=2), " nT. Median transit Δ = ",
                    round(median(all_rows.transit_h); digits=2), " h; fraction of issues with Δ ≥ 1 h: ",
                    round(mean(all_rows.transit_h .>= 1.0); digits=2), ".\n")
        println(io, "**Direction A verdict:** ", verdict)
        println("\n  A verdict: ", verdict)
    end
    println("  wrote ", OUT_CSV_LA, " and ", OUT_MD_LA)
    return all_rows
end

# ---- CRC self-test ----
_cal_la = Ref{Any}(nothing)
_calA() = (_cal_la[] === nothing && (_cal_la[] = _load_calibration_for_model(LiveVerifyConfig(model = :v2))); _cal_la[])
function _selftest_la()
    lib, ξ0, i_decay = _shadow_library()
    drv = (V = 300.0, Bz = -8.0, By = 0.0, n = 6.0, Pdyn = 2.0)    # slow wind ⇒ Δ≈1.39 h ⇒ strict-causal look-ahead fires at k=1
    fut = (V = 320.0, Bz = -18.0, By = 0.0, n = 8.0, Pdyn = 3.0)   # a more-southward incoming driver
    # Oracle 1 — transit physics: Δ = L1/V is ~1.04 h at 400 km/s, ~0.52 h at 800 km/s.
    @assert isapprox(_transit_hours(400.0), 1.5e6/400/3600; atol = 1e-6) && _transit_hours(800.0) < 0.6 "transit-time wrong"
    # Oracle 2 — CONTINUITY: force_frozen reproduces the frozen v1+correction (the v2-equivalent _shadow_forecast).
    for h in (1, 3, 6)
        a = _la_forecast(lib, ξ0, -120.0, drv, _ -> fut, -118.0, _calA(), h; force_frozen = true)
        b = _shadow_forecast(lib, ξ0, i_decay, ξ0[i_decay], -120.0, drv, -118.0, _calA(); nsteps = h)
        @assert a == b "continuity broken at h=$h: la_frozen=$a v2-equiv=$b"
    end
    # Oracle 3 — LOOK-AHEAD acts: with a more-southward incoming driver and Δ>0, the h=1 forecast is MORE
    # negative than frozen (the anticipation lowers Dst), and zero future driver leaves it == frozen.
    la   = _la_forecast(lib, ξ0, -120.0, drv, _ -> fut,     -118.0, _calA(), 1)[1]
    fr   = _la_forecast(lib, ξ0, -120.0, drv, _ -> fut,     -118.0, _calA(), 1; force_frozen = true)[1]
    none = _la_forecast(lib, ξ0, -120.0, drv, _ -> nothing, -118.0, _calA(), 1)[1]
    @assert la < fr "look-ahead with a stronger incoming driver did not deepen the forecast ($la !< $fr)"
    @assert none == fr "missing future driver must fall back to frozen ($none != $fr)"
    println("  ✓ Direction A self-test: transit physics, continuity to v2, look-ahead deepens, gap-safe fallback")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest_la()
    main_la()
end
