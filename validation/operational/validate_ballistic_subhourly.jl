# Validate the LIVE ballistic sub-hourly-A method (the exact algorithm deployed in live_forecast_verify.jl's
# _subhourly_driver) against v2 and persistence, on the historical storm set. The deployed live system cannot
# use OMNI HRO's MEASURED propagation timeshift; it only has a ballistic transit estimate (L1_DIST / V). So the
# HRO-timeshift validation (v2_subhourly_A_replay) does NOT transfer directly — this backtests the ballistic
# variant that the live engine actually runs, so promotion (serving sub-hourly-A at h=1) is justified by the
# exact deployed algorithm, not a more-informed proxy.
#
# Ballistic variant: for Earth-arrival step [it+(k-1)h, it+k h), include only HRO minutes whose L1 measurement
# time (ltime) is <= issue (leakage guard) AND whose ballistic arrival ltime + L1/V falls in the step. This
# mirrors _subhourly_driver exactly (single lag from the issue speed; cap at the last measured sample).
#
# Run from the package root: julia --project=. validation/operational/validate_ballistic_subhourly.jl

include(joinpath(@__DIR__, "v2_subhourly_A_replay.jl"))   # infra: HRO, STORMS, library, _blend, L1_DIST_KM, replay_recent_table, _cell_subA, LEADS, MAIN_RATE

"""Ballistic sub-hourly-A: identical to _subA_forecast but the L1 horizon is the ballistic L1/V transit (a
single lag from the issue speed), not the HRO per-parcel measured timeshift — i.e. exactly what the live engine
can compute. Leakage-safe: only minutes with ltime <= issue enter. force_frozen ⇒ v2."""
function _subA_ballistic_forecast(lib, ξ0, anchor_dst_star, issue_drv, hro, it::DateTime, latest_dst, cal, h::Int; force_frozen::Bool=false)
    fc = init_assimilation(lib, ξ0, Int[], anchor_dst_star)
    lag = Millisecond(round(Int, (L1_DIST_KM / max(issue_drv.V, 1.0) / 3600.0) * 3_600_000))
    for k in 1:h
        drv_k = issue_drv
        if !force_frozen
            t0 = it + Hour(k - 1); t1 = it + Hour(k)
            m = (hro.ltime .>= (t0 - lag)) .& (hro.ltime .< (t1 - lag)) .& (hro.ltime .<= it)   # ballistic arrival in step, measured by issue
            cnt = count(m)
            if cnt > 0
                sub = hro[m, :]
                mv(col) = (v = filter(isfinite, col); isempty(v) ? NaN : mean(v))
                kavg = (V=mv(sub.V), Bz=mv(sub.Bz), By=mv(sub.By), n=mv(sub.n), Pdyn=mv(sub.Pdyn))
                f = clamp(cnt / 60.0, 0.0, 1.0)
                all(isfinite, (kavg.V, kavg.Bz, kavg.By, kavg.n, kavg.Pdyn)) && f > 0 && (drv_k = _blend(f, kavg, issue_drv))
            end
        end
        assimilation_predict!(fc, drv_k)
        fc.mean[1] = clamp(fc.mean[1], -2000.0, 50.0)
    end
    pred_dst_star = current_dst(fc)
    pred_dst = pred_dst_star + 7.26 * sqrt(max(issue_drv.Pdyn, 0.0)) - 11.0
    feat_df = DataFrame(latest_dst_nt=[latest_dst], V_kms=[issue_drv.V], Bz_nt=[issue_drv.Bz],
                        By_nt=[issue_drv.By], n_cm3=[issue_drv.n], Pdyn_npa=[issue_drv.Pdyn])
    prep = SolarSINDy.add_operational_v2_features!(feat_df)
    feats = NamedTuple{Tuple(cal.feature_names)}(Tuple(Float64(prep[1, c]) for c in cal.feature_names))
    corr = SolarSINDy.operational_v2_correction(cal, feats)
    return clamp(pred_dst + corr, -2000.0, 50.0)
end

function replay_ballistic_storm(storm, lib, ξ0, cal)
    hro = _hro_for_storm(storm)
    yr = year(storm.t1)
    plasma, mag, dst_times, dst_vals = _omni_replay_inputs(OMNI, yr - 1, yr)
    m = (dst_times .>= storm.t0 - Hour(6)) .& (dst_times .<= storm.t1 + Hour(7))
    rh = Int(ceil(Dates.value((storm.t1 + Hour(7)) - (storm.t0 - Hour(6))) / 3_600_000))
    df = replay_recent_table(plasma[m, :], mag[m, :], dst_times[m], dst_vals[m];
                             replay_hours=rh, horizons=LEADS, model=:v2, calibration=cal)
    df[!, :issue_dt] = DateTime.(df.issue_time_utc)
    df = df[(df.issue_dt .>= storm.t0) .& (df.issue_dt .<= storm.t1), :]
    sort!(df, [:issue_dt, :model_step_hours])
    out = DataFrame(storm=String[], issue_utc=DateTime[], lead=Int[], obs=Float64[], v2=Float64[],
                    subA=Float64[], subA_frozen=Float64[], persistence=Float64[], rate=Float64[])
    prev_anchor = NaN
    for it in sort!(unique(df.issue_dt))
        g = df[df.issue_dt .== it, :]; r1 = g[1, :]
        issue_drv = (V=Float64(r1.V_kms), Bz=Float64(r1.Bz_nt), By=Float64(r1.By_nt), n=Float64(r1.n_cm3), Pdyn=Float64(r1.Pdyn_npa))
        latest = Float64(r1.latest_dst_nt)
        all(isfinite, (issue_drv.V, issue_drv.Bz, issue_drv.By, issue_drv.n, issue_drv.Pdyn, latest)) || (prev_anchor = NaN; continue)
        anchor_star = pressure_correct_dst([latest], [issue_drv.Pdyn])[1]
        rate = isfinite(prev_anchor) ? latest - prev_anchor : NaN
        prev_anchor = latest
        for r in eachrow(g)
            (ismissing(r.observation_dst_nt) || ismissing(r.v2_pred_dst_nt)) && continue
            isfinite(Float64(r.observation_dst_nt)) && isfinite(Float64(r.v2_pred_dst_nt)) || continue
            h = Int(r.model_step_hours)
            sa  = _subA_ballistic_forecast(lib, ξ0, anchor_star, issue_drv, hro, it, latest, cal, h)
            saf = _subA_ballistic_forecast(lib, ξ0, anchor_star, issue_drv, hro, it, latest, cal, h; force_frozen=true)
            isfinite(sa) && isfinite(saf) || continue
            push!(out, (storm.name, it, h, Float64(r.observation_dst_nt), Float64(r.v2_pred_dst_nt), sa, saf, latest, rate))
        end
    end
    return out
end

function main_ballistic()
    lib, ξ0, _ = _shadow_library()
    cal = _load_calibration_for_model(LiveVerifyConfig(model=:v2))
    all_rows = DataFrame()
    for s in STORMS
        append!(all_rows, replay_ballistic_storm(s, lib, ξ0, cal))
    end
    fairmax = isempty(all_rows) ? NaN : maximum(abs.(all_rows.subA_frozen .- all_rows.v2))
    println("\n=== LIVE ballistic sub-hourly-A vs v2 + persistence (HRO storm set) ===")
    println("continuity max|ballistic_frozen - v2| = ", round(fairmax; digits=4), " nT  (0 ⇒ reduces to v2 exactly)")
    println("lead regime    n   RMSE_v2 RMSE_subA RMSE_pers stronger  improve[95% CI]")
    for h in LEADS
        sub_h = all_rows[all_rows.lead .== h, :]
        for (label, sub) in (("pooled", sub_h), ("main", sub_h[isfinite.(sub_h.rate) .& (sub_h.rate .< MAIN_RATE), :]))
            c = _cell_subA(sub); c === nothing && continue
            @printf("  %3d  %-6s %4d  %6.2f  %7.2f  %7.2f  %-4s  %+5.2f [%+5.2f,%+5.2f]\n",
                    h, label, c.n, c.rmse_v2, c.rmse_s, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_ballistic()
end
