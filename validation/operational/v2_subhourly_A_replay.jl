# v2_subhourly_A_replay.jl — sub-hourly Direction A: measure A's TRUE magnitude using 1-min L1 (OMNI HRO).
#
# The hourly replay under-states A: with transit Δ=L1/V ≈ 0.5–1.1 h, most storm wind is too fast for a ≥1 h
# hourly look-ahead, so strict-causal floors most issues to v2. With 1-min HRO data, at issue t the at-Earth
# wind for the next Δ HOURS is resolved to the minute, so step 1 of the rollout is driven by the AVERAGE of
# the actual incoming wind over the genuinely-known window [t, t+Δ] — capturing the sub-hour anticipation the
# hourly grid threw away. Leakage-safe: only minutes ≤ t+Δ (parcels measured at L1 by issue time t) are used;
# the unknown remainder of the step keeps the frozen issue driver. Step-k driver = f·(HRO mean over the known
# sub-window) + (1-f)·issue_drv, f=clamp(Δ-(k-1),0,1). f=0 (Δ=0) ⇒ frozen issue driver ⇒ v2 (continuity).
#
# Run from the package root: julia --project=. validation/operational/v2_subhourly_A_replay.jl

include(joinpath(@__DIR__, "v2_lookahead_replay.jl"))   # _transit_hours, _blend, L1_DIST_KM, _shadow_library, etc.

const HRO_DIR   = OPERATIONAL_HRO_CACHE
const OUT_CSV_SA = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_subhourly_A_replay_scored.csv")
const OUT_MD_SA  = joinpath(OPERATIONAL_OUTPUT_DIR, "v2_subhourly_A_replay_report.md")
const HRO_BASE_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/omni/high_res_omni/monthly_1min"

# OMNI HRO 1-min, whitespace-delimited. 1-based split columns (verified against hroformat.txt):
#   f[1]year f[2]doy f[3]hour f[4]minute  f[10]timeshift_sec  f[18]By_GSM f[19]Bz_GSM  f[22]V  f[26]n  f[28]Pdyn
# `ltime` = the L1-spacecraft measurement time = bow-shock time − timeshift; a sample is knowable at issue t
# iff ltime ≤ t. This is the EXACT per-parcel L1 horizon (handles wind acceleration, unlike a constant Δ).
"Parse one OMNI HRO 1-min monthly file → DataFrame(time, ltime, V, Bz, By, n, Pdyn) with fills→NaN, time-sorted."
function parse_hro(path::AbstractString)
    t = DateTime[]; lt = DateTime[]; V = Float64[]; Bz = Float64[]; By = Float64[]; n = Float64[]; Pd = Float64[]
    for line in eachline(path)
        f = split(line)
        length(f) >= 28 || continue
        yr = parse(Int, f[1]); doy = parse(Int, f[2]); hr = parse(Int, f[3]); mn = parse(Int, f[4])
        ts = parse(Float64, f[10])                                     # timeshift, seconds (fill 999999)
        bygsm = parse(Float64, f[18]); bzgsm = parse(Float64, f[19])
        v = parse(Float64, f[22]); nn = parse(Float64, f[26]); pd = parse(Float64, f[28])
        bt = DateTime(yr, 1, 1) + Day(doy - 1) + Hour(hr) + Minute(mn)
        push!(t, bt)
        push!(lt, ts >= 999999 ? bt : bt - Second(round(Int, ts)))     # L1 measurement time
        push!(Bz, abs(bzgsm) >= 9999 ? NaN : bzgsm); push!(By, abs(bygsm) >= 9999 ? NaN : bygsm)
        push!(V, v >= 99999 ? NaN : v); push!(n, nn >= 999 ? NaN : nn); push!(Pd, pd >= 99 ? NaN : pd)
    end
    df = DataFrame(time = t, ltime = lt, V = V, Bz = Bz, By = By, n = n, Pdyn = Pd)
    sort!(df, :time)
    return df
end

"Calendar months intersecting the padded HRO window for a storm."
function _hro_months_for_storm(storm)
    first_month = Date(year(storm.t0 - Hour(6)), month(storm.t0 - Hour(6)), 1)
    last_month = Date(year(storm.t1 + Hour(7)), month(storm.t1 + Hour(7)), 1)
    return collect(first_month:Month(1):last_month)
end

_hro_month_path(d::Date) = joinpath(HRO_DIR, "omni_min$(year(d))$(lpad(month(d), 2, '0')).asc")

"Load 1-min HRO covering a storm window, concatenated and time-sorted."
function _hro_for_storm(storm)
    dfs = DataFrame[]
    for d in _hro_months_for_storm(storm)
        p = _hro_month_path(d)
        isfile(p) || error(
            "missing NASA OMNI HRO month file: $p\n" *
            "Run `julia --project=. validation/operational/fetch_omni_hro.jl` from the package root.",
        )
        push!(dfs, parse_hro(p))
    end
    df = vcat(dfs...); sort!(df, :time); return df
end

"""Mean HRO driver over the KNOWN minutes of [t0, t1) — those whose L1 measurement time ltime ≤ issue_t — and
the known fraction f = (#known minutes)/60. Returns (nothing, 0) if no finite known rows. This is the exact
per-parcel L1 horizon: only minutes already measured at L1 by issue time enter (leakage-safe by construction)."""
function _hro_known_avg(hro::DataFrame, t0::DateTime, t1::DateTime, issue_t::DateTime)
    m = (hro.time .>= t0) .& (hro.time .< t1) .& (hro.ltime .<= issue_t)
    cnt = count(m)
    cnt == 0 && return (nothing, 0.0)
    sub = hro[m, :]
    mv(col) = (v = filter(isfinite, col); isempty(v) ? NaN : mean(v))
    out = (V = mv(sub.V), Bz = mv(sub.Bz), By = mv(sub.By), n = mv(sub.n), Pdyn = mv(sub.Pdyn))
    all(isfinite, (out.V, out.Bz, out.By, out.n, out.Pdyn)) ? (out, clamp(cnt / 60.0, 0.0, 1.0)) : (nothing, 0.0)
end

"""h-step sub-hourly A forecast: step-k driver = f·(HRO mean over the L1-KNOWN minutes of the step) +
(1-f)·issue_drv, where "known" = the minute's L1 measurement time ≤ issue t (exact per-parcel horizon, no
constant-Δ approximation). The revised 20/11 core and V2.1 calibration are used. `force_frozen` reproduces
the V2.1 frozen-tail ablation; this component study does not reproduce the fully served product."""
function _subA_forecast(lib, ξ0, anchor_dst_star, issue_drv, hro, it::DateTime,
                        latest_dst, cal, h::Int; force_frozen::Bool=false,
                        calibration_features=nothing)
    fc = init_assimilation(lib, ξ0, Int[], anchor_dst_star)
    for k in 1:h
        drv_k = issue_drv
        if !force_frozen
            kavg, f = _hro_known_avg(hro, it + Hour(k - 1), it + Hour(k), it)
            kavg !== nothing && f > 0 && (drv_k = _blend(f, kavg, issue_drv))   # only L1-known minutes enter
        end
        assimilation_predict!(fc, drv_k)
        fc.mean[1] = clamp(fc.mean[1], -2000.0, 50.0)
    end
    pred_dst_star = current_dst(fc)
    pred_dst = pred_dst_star + 7.26 * sqrt(max(issue_drv.Pdyn, 0.0)) - 11.0
    feats = _v2_calibration_features(
        cal, latest_dst, issue_drv; v1_pred_dst=pred_dst, model_steps=h,
        feature_source=calibration_features, context="subhourly-A",
    )
    corr = SolarSINDy.operational_v2_correction(cal, feats)
    return clamp(pred_dst, -2000.0, 50.0), clamp(pred_dst + corr, -2000.0, 50.0)
end

function replay_subA_storm(storm, lib, ξ0, cal)
    hro = _hro_for_storm(storm)
    yr = year(storm.t1)
    plasma, mag, dst_times, dst_vals = _omni_replay_inputs(OMNI, yr - 1, yr)
    win_lo, win_hi = storm.t0 - Hour(6), storm.t1 + Hour(7)
    plasma, mag, dst_times, dst_vals = _slice_replay_window(
        plasma, mag, dst_times, dst_vals, win_lo, win_hi,
    )
    rh = Int(ceil(Dates.value(win_hi - win_lo) / 3_600_000))
    df = replay_recent_table(plasma, mag, dst_times, dst_vals;
                             replay_hours = rh, horizons = LEADS, model = :v2, calibration = cal)
    df[!, :issue_dt] = DateTime.(df.issue_time_utc)
    df = df[(df.issue_dt .>= storm.t0) .& (df.issue_dt .<= storm.t1), :]
    sort!(df, [:issue_dt, :model_step_hours])
    out = DataFrame(storm = String[], issue_utc = DateTime[], lead = Int[], obs = Float64[], v2 = Float64[],
                    subA = Float64[], subA_frozen = Float64[], persistence = Float64[], transit_h = Float64[], rate = Float64[])
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
            _, sa = _subA_forecast(
                lib, ξ0, anchor_star, issue_drv, hro, it, latest, cal, h;
                calibration_features=r,
            )
            _, saf = _subA_forecast(
                lib, ξ0, anchor_star, issue_drv, hro, it, latest, cal, h;
                force_frozen=true, calibration_features=r,
            )
            isfinite(sa) && isfinite(saf) || continue
            push!(out, (storm.name, it, h, Float64(r.observation_dst_nt), Float64(r.v2_pred_dst_nt),
                        sa, saf, latest, _transit_hours(issue_drv.V), rate))
        end
    end
    return out
end

function _cell_subA(rows)
    nrow(rows) == 0 && return nothing
    ev2 = rows.obs .- rows.v2; es = rows.obs .- rows.subA; ep = rows.obs .- rows.persistence
    rv2, rs, rp = _rmse(ev2), _rmse(es), _rmse(ep)
    strong_pers = rp <= rv2
    Δ, lo, hi = paired_improvement(strong_pers ? ep : ev2, es; storms=rows.storm)
    return (n = nrow(rows), rmse_v2 = rv2, rmse_s = rs, rmse_pers = rp, stronger = strong_pers ? "pers" : "frozen-tail",
            improve = Δ, ci_lo = lo, ci_hi = hi, fair = maximum(abs.(rows.subA_frozen .- rows.v2)))
end

function main_subA()
    lib, ξ0, i_decay = _shadow_library()
    cal = _load_calibration_for_model(LiveVerifyConfig(model = :v2))
    println("Sub-hourly Direction A (1-min HRO L1) vs v2 + persistence, leads ", LEADS, "\n", "="^80)
    all_rows = DataFrame()
    for s in STORMS
        rows = replay_subA_storm(s, lib, ξ0, cal)
        append!(all_rows, rows)
        @printf("  %-24s rows=%d  median transit=%.2f h\n", s.name, nrow(rows),
                nrow(rows) > 0 ? median(rows.transit_h) : NaN)
    end
    CSV.write(OUT_CSV_SA, all_rows)
    open(OUT_MD_SA, "w") do io
        println(io, "# Sub-hourly Direction A (1-min OMNI HRO) vs V2.1 frozen-tail ablation + persistence\n")
        println(io, "At issue t, each rollout step is driven by the AVERAGE 1-min HRO wind over its L1-KNOWN ",
                    "minutes — those whose L1 measurement time (bow-shock time − HRO timeshift) is ≤ t — blended ",
                    "with the frozen issue driver by the known fraction. This is the EXACT per-parcel L1 horizon ",
                    "(no constant-Δ approximation), so it is leakage-safe even when the wind accelerates. The arm ",
                    "uses the revised 20/11 core and V2.1 calibration but substitutes this driver path for the frozen ",
                    "rollout. It is component-development evidence, not a replay of the fully served V2.1 product. ",
                    "`improve` = paired RMSE(stronger {frozen-tail, persistence}) − RMSE(A), with a storm-cluster ",
                    "95% CI; `fair` verifies exact continuity to the frozen-tail ablation.\n")
        println(io, "| lead [h] | regime | n | RMSE frozen-tail | RMSE subA | RMSE pers | stronger | improve [nT] (95% CI) | fair |")
        println(io, "|---|---|---|---|---|---|---|---|---|")
        println("\n  lead regime    n   v2    subA  pers  strong  improve[CI]            fair")
        for h in LEADS
            sub_h = all_rows[all_rows.lead .== h, :]
            for (label, sub) in (("pooled", sub_h),
                                 ("main",   sub_h[isfinite.(sub_h.rate) .& (sub_h.rate .< MAIN_RATE), :]))
                c = _cell_subA(sub); c === nothing && continue
                @printf("  %3d  %-6s %4d %5.1f %5.1f %5.1f  %-4s  %+5.2f [%+5.2f,%+5.2f]  %.2f\n",
                        h, label, c.n, c.rmse_v2, c.rmse_s, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.fair)
                @printf(io, "| %d | %s | %d | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] | %.2f |\n",
                        h, label, c.n, c.rmse_v2, c.rmse_s, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.fair)
            end
        end
        max_fair = maximum(abs.(all_rows.subA_frozen .- all_rows.v2))
        println(io, "\nMax continuity gap max|subA_frozen − V2.1 frozen-tail| = ", round(max_fair; digits=2),
                    " nT. Compare to the hourly Direction A (Δ≥1 subset +1.40/+2.14/+2.37 at 1/2/3 h): the sub-hourly ",
                    "version uses every issue's true sub-hour window, not just Δ≥1.\n")
        println(io, "This measured-timeshift arm is a high-information diagnostic; a live system cannot know the ",
                    "per-parcel OMNI propagation shift. The ballistic component is evaluated separately.\n")
        println("\n  sub-hourly A: max frozen-tail continuity gap = ", round(max_fair; digits=2), " nT")
    end
    println("  wrote ", OUT_CSV_SA, " and ", OUT_MD_SA)
    return all_rows
end

# ---- CRC self-test ----
_cal_sa = Ref{Any}(nothing)
_calSA() = (_cal_sa[] === nothing && (_cal_sa[] = _load_calibration_for_model(LiveVerifyConfig(model = :v2))); _cal_sa[])
function _selftest_subA()
    lib, ξ0, i_decay = _shadow_library()
    # Oracle 0 — PARSER + LEAKAGE HORIZON: a storm-hour HRO averages to physical values; and at issue time
    # h0 the L1-known fraction of the NEXT hour [h0, h0+1) is < 1 (the late minutes have ltime > h0), while
    # with a far-future issue time every minute is known (fraction = 1). This pins the leakage horizon.
    p = joinpath(HRO_DIR, "omni_min202405.asc")
    if isfile(p)
        hro = parse_hro(p)
        h0 = DateTime(2024, 5, 10, 18)
        wv, _    = _hro_known_avg(hro, h0, h0 + Hour(1), DateTime(2024, 5, 11))   # far-future issue ⇒ all known
        _, fnow  = _hro_known_avg(hro, h0, h0 + Hour(1), h0)                      # issue = h0 ⇒ only L1-resident
        @assert wv !== nothing && -2000 < wv.Bz < 2000 && 100 < wv.V < 3000 "HRO parse implausible: $wv"
        @assert 0.0 <= fnow < 1.0 "known fraction at issue time must be < 1 (leakage horizon): $fnow"
    end
    drv = (V = 300.0, Bz = -10.0, By = 2.0, n = 6.0, Pdyn = 2.0)
    # Oracle 1 — ABLATION CONTINUITY: force_frozen reproduces the revised-core frozen-tail reference.
    dummy = DataFrame(time = DateTime[], ltime = DateTime[], V = Float64[], Bz = Float64[], By = Float64[], n = Float64[], Pdyn = Float64[])
    for h in (1, 3, 6)
        a = _subA_forecast(lib, ξ0, -150.0, drv, dummy, DateTime(2024, 5, 10, 18), -148.0, _calSA(), h; force_frozen = true)
        b = _shadow_forecast(lib, ξ0, i_decay, ξ0[i_decay], -150.0, drv, -148.0, _calSA(); nsteps = h)
        @assert a == b "frozen-tail continuity broken at h=$h: subA_frozen=$a reference=$b"
    end
    # Oracle 2 — empty HRO window ⇒ falls back to the frozen issue driver (no spurious look-ahead).
    e = _subA_forecast(lib, ξ0, -150.0, drv, dummy, DateTime(2024, 5, 10, 18), -148.0, _calSA(), 1)
    f = _subA_forecast(lib, ξ0, -150.0, drv, dummy, DateTime(2024, 5, 10, 18), -148.0, _calSA(), 1; force_frozen = true)
    @assert e == f "empty-HRO look-ahead must fall back to frozen ($e != $f)"
    println("  ✓ sub-hourly A self-test: HRO parse plausible, frozen-tail continuity, empty-window fallback")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest_subA()
    main_subA()
end
