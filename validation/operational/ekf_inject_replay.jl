# ekf_inject_replay.jl — injection-adaptive EKF vs the V2.1 frozen-tail ablation.
#
# Diagnosis of the decay-only shadow EKF: its sole adaptive lever is the ring-current DECAY coefficient
# ("Dst_star"), a restoring/recovery term, so adaptation can only push the forecast toward zero — correct in
# recovery, harmful at storm onset where Dst is still dropping (verified +3.37 nT main-phase bias). The
# current V2.1 SINDy ODE contains a southward-field INJECTION term ("Bs") that
# drives Dst* DOWN; it is held fixed. This variant adds that injection coefficient to the EKF's adaptive
# state, giving the filter a downward lever to track a fast drop — staying entirely within the discovered
# sparse SINDy model (more of its coefficients adapted online, no new model, no swap to O'Brien).
#
# Everything else mirrors ekf_storm_replay.jl exactly: same 7 causal storms, leads [1,2,3,6], pooled +
# main-phase regimes, paired-bootstrap CI vs the STRONGER of {V2.1 frozen-tail, persistence}, and the
# ekf_fixed fairness oracle (fixed discovered coefficients must reproduce the frozen-tail engine).
#
# Run from the package root: julia --project=. validation/operational/ekf_inject_replay.jl

isdefined(@__MODULE__, :STORMS) || include(joinpath(@__DIR__, "ekf_storm_replay.jl"))   # STORMS, LEADS, MAIN_RATE, _rmse, _ffill!, OMNI,
                                                     # paired_improvement, _cell, _per_storm_signs,
                                                     # _shadow_library, _shadow_decay_cap, BOOT_YEAR0, Q_* consts

const OUT_CSV_INJ = joinpath(OPERATIONAL_OUTPUT_DIR, "ekf_inject_replay_scored.csv")
const OUT_MD_INJ  = joinpath(OPERATIONAL_OUTPUT_DIR, "ekf_inject_replay_report.md")

const BS_TERM = "Bs"
# Injection-coefficient stability box: the southward-field term must stay a DRIVER (≤ 0, never a source that
# would unphysically reduce the ring current), bounded away from runaway at ≈3× the discovered magnitude.
const BS_BOX  = (-2.0, 0.0)
# Process-noise for the injection coefficient, set by RELATIVE-rate matching to the validated decay walk
# (q/ξ² equal): q_Bs = Q_COEFF · (ξ_Bs/ξ_decay)². This is a principled, leakage-free choice — NOT tuned on
# the test storms. The ratio is recomputed from the current V2.1 coefficients.
_q_bs(ξ0, i_decay, i_Bs) = Q_COEFF * (ξ0[i_Bs] / ξ0[i_decay])^2

"Index of the discovered southward-injection term in the library."
function _bs_index(lib)
    i = findfirst(==(BS_TERM), get_term_names(lib))
    i === nothing && error("injection term \"$BS_TERM\" not found in library")
    return i
end

"Constrained EKF adapting BOTH [decay, Bs]: decay box (-Inf, cap], Bs box BS_BOX, decay walk Q_COEFF, Bs walk q_bs."
function _new_inject_filter(lib, ξ0, i_decay, i_Bs, dst0; q_bs)
    decay_cap = _shadow_decay_cap(ξ0, i_decay)
    f = init_assimilation(lib, ξ0, [i_decay, i_Bs], dst0; q_coeff = Q_COEFF, q_dst = Q_DST, R = R_OBS,
                          dst_var0 = DST_VAR0, coeff_var0 = COEFF_VAR0,
                          coeff_bounds = [(-Inf, decay_cap), BS_BOX])
    f.Q[3, 3] = q_bs        # decay keeps Q_COEFF (index 2); injection gets its own walk variance (index 3)
    return f
end

"h-step forecast with adapted [decay, Bs], engine-faithful per-step + final Dst* clamp, then the v2 correction.
Identical to _shadow_forecast except two discovered coefficients are overridden instead of one."
function _inject_forecast(lib, ξ0, i_decay, i_Bs, decay, bscoef,
                          anchor_dst_star, drv, latest_dst, cal;
                          nsteps::Int=1, calibration_features=nothing)
    ξ = copy(ξ0); ξ[i_decay] = decay; ξ[i_Bs] = bscoef
    fc = init_assimilation(lib, ξ, Int[], anchor_dst_star)
    for _ in 1:nsteps
        assimilation_predict!(fc, drv)
        fc.mean[1] = clamp(fc.mean[1], -2000.0, 50.0)
    end
    pred_dst_star = current_dst(fc)
    pred_dst = pred_dst_star + 7.26 * sqrt(max(drv.Pdyn, 0.0)) - 11.0
    fallback = _v2_features(
        latest_dst, drv; v1_pred_dst=pred_dst, model_steps=nsteps,
    )
    source = calibration_features === nothing ? fallback : calibration_features
    missing_features = [c for c in cal.feature_names if !(c in propertynames(source))]
    isempty(missing_features) || error(
        "injection-EKF calibration feature source omits: $(join(String.(missing_features), ", "))",
    )
    feats = NamedTuple{Tuple(cal.feature_names)}(
        Tuple(Float64(getproperty(source, c)) for c in cal.feature_names),
    )
    corr = SolarSINDy.operational_v2_correction(cal, feats)
    return clamp(pred_dst, -2000.0, 50.0), clamp(pred_dst + corr, -2000.0, 50.0)
end

"Causal injection-EKF bootstrap over OMNI [BOOT_YEAR0, year_end] — no storm-year leakage."
function causal_bootstrap_inject(lib, ξ0, i_decay, i_Bs, year_end::Int; q_bs)
    # Persisted EKF replay CSVs predate causal cleaning and are retired negative-result artifacts (not rerun);
    # causal=true keeps this bootstrap strictly causal if the retired script is ever re-executed.
    df = parse_omni2(OMNI; year_start = BOOT_YEAR0, year_end = year_end); clean_omni_data!(df; causal = true)
    V  = _ffill!(Float64.(coalesce.(df.V, NaN)));  Bz = _ffill!(Float64.(coalesce.(df.Bz, NaN)))
    By = _ffill!(Float64.(coalesce.(df.By, NaN))); nn = _ffill!(Float64.(coalesce.(df.n, NaN)))
    Pd = [dynamic_pressure(nn[k], V[k]) for k in eachindex(V)]
    dst = Float64.(coalesce.(df.Dst_star, NaN))
    f = _new_inject_filter(lib, ξ0, i_decay, i_Bs, isfinite(dst[1]) ? dst[1] : 0.0; q_bs = q_bs)
    for k in 1:length(dst)-1
        (isfinite(dst[k+1]) && isfinite(V[k])) || continue
        assimilation_predict!(f, (V = V[k], Bz = Bz[k], By = By[k], n = nn[k], Pdyn = Pd[k]))
        assimilation_update!(f, dst[k+1])
    end
    return f
end

"Walk one storm through the injection EKF; emit one row per (issue, lead) with both adapted coefficients."
function replay_inject_storm(storm, lib, ξ0, i_decay, i_Bs, cal, f)
    yr = year(storm.t1)
    plasma, mag, dst_times, dst_vals = _omni_replay_inputs(OMNI, yr - 1, yr)
    win_lo, win_hi = storm.t0 - Hour(6), storm.t1 + Hour(7)
    plasma, mag, dst_times, dst_vals = _slice_replay_window(
        plasma, mag, dst_times, dst_vals, win_lo, win_hi,
    )
    (nrow(plasma) > 0 && !isempty(dst_times)) || error("No finite OMNI drivers/Dst inside replay window")
    rh = Int(ceil(Dates.value(win_hi - win_lo) / 3_600_000))
    df = replay_recent_table(plasma, mag, dst_times, dst_vals;
                             replay_hours = rh, horizons = LEADS, model = :v2, calibration = cal)
    df[!, :issue_dt] = DateTime.(df.issue_time_utc)
    df = df[(df.issue_dt .>= storm.t0) .& (df.issue_dt .<= storm.t1), :]
    sort!(df, [:issue_dt, :model_step_hours])
    dst_map = Dict{DateTime,Float64}(zip(dst_times, Float64.(dst_vals)))
    out = DataFrame(storm = String[], issue_utc = DateTime[], lead = Int[], obs = Float64[],
                    frozen_tail = Float64[], ekf = Float64[], ekf_fixed = Float64[], persistence = Float64[],
                    adapted_decay = Float64[], adapted_bs = Float64[], rate = Float64[])
    for it in sort!(unique(df.issue_dt))
        g = df[df.issue_dt .== it, :]; r1 = g[1, :]
        drv = (V = Float64(r1.V_kms), Bz = Float64(r1.Bz_nt), By = Float64(r1.By_nt),
               n = Float64(r1.n_cm3), Pdyn = Float64(r1.Pdyn_npa))
        latest = Float64(r1.latest_dst_nt)
        all(isfinite, (drv.V, drv.Bz, drv.By, drv.n, drv.Pdyn, latest)) || continue
        anchor_star = pressure_correct_dst([latest], [drv.Pdyn])[1]
        assimilation_predict!(f, drv); assimilation_update!(f, anchor_star)
        decay, bs = current_coeffs(f)            # [decay, Bs] in adapt_idx order
        prev1 = get(dst_map, it - Hour(1), NaN)
        rate = isfinite(prev1) ? latest - prev1 : NaN
        for r in eachrow(g)
            (ismissing(r.observation_dst_nt) || ismissing(r.v2_pred_dst_nt)) && continue
            isfinite(Float64(r.observation_dst_nt)) && isfinite(Float64(r.v2_pred_dst_nt)) || continue
            h = Int(r.model_step_hours)
            _, ekf_a = _inject_forecast(lib, ξ0, i_decay, i_Bs, decay, bs,
                anchor_star, drv, latest, cal; nsteps=h, calibration_features=r)
            _, ekf_f = _inject_forecast(lib, ξ0, i_decay, i_Bs,
                ξ0[i_decay], ξ0[i_Bs], anchor_star, drv, latest, cal;
                nsteps=h, calibration_features=r)
            isfinite(ekf_a) && isfinite(ekf_f) || continue
            push!(out, (storm.name, it, h, Float64(r.observation_dst_nt), Float64(r.v2_pred_dst_nt),
                        ekf_a, ekf_f, latest, decay, bs, rate))
        end
    end
    return out
end

"Full causal replay over all storms for one injection walk-variance q_bs. Returns the pooled rows."
function replay_all(lib, ξ0, i_decay, i_Bs, cal, q_bs)
    boot = Dict{Int,Any}(); all_rows = DataFrame()
    for s in STORMS
        ye = year(s.t1) - 1
        haskey(boot, ye) || (boot[ye] = causal_bootstrap_inject(lib, ξ0, i_decay, i_Bs, ye; q_bs = q_bs))
        append!(all_rows, replay_inject_storm(s, lib, ξ0, i_decay, i_Bs, cal, deepcopy(boot[ye])))
    end
    return all_rows
end

"Sensitivity sweep over the injection walk-variance: does ANY q_Bs beat persistence? q_Bs→0 reduces to the
decay-only EKF (continuity oracle), so this brackets the whole injection-adaptation family."
function _sweep(lib, ξ0, i_decay, i_Bs, cal)
    q_rate = _q_bs(ξ0, i_decay, i_Bs)
    println("\nq_Bs sensitivity sweep (1h pooled vs persistence; 6h-main vs persistence):")
    println("  q_Bs        Bs range          1h-pool improve[CI]      6h-main improve[CI]")
    open(OUT_MD_INJ, "a") do io
        println(io, "\n## q_Bs sensitivity sweep\n")
        println(io, "Does ANY injection walk-variance beat persistence? q_Bs→0 reduces to the decay-only EKF ",
                    "(continuity oracle), so this brackets the whole injection family.\n")
        println(io, "| q_Bs | Bs range | 1h-pooled improve [nT] (95% CI) | 6h-main improve [nT] (95% CI) |")
        println(io, "|---|---|---|---|")
        for q in (1.0e-3, q_rate, 1.0e-1)
            rows = replay_all(lib, ξ0, i_decay, i_Bs, cal, q)
            c1 = _cell(rows[rows.lead .== 1, :])
            s6 = rows[(rows.lead .== 6) .& isfinite.(rows.rate) .& (rows.rate .< MAIN_RATE), :]
            c6 = _cell(s6)
            blo, bhi = minimum(rows.adapted_bs), maximum(rows.adapted_bs)
            @printf("  %.2e  [%5.2f,%5.2f]   %+5.2f [%+5.2f,%+5.2f]    %+6.2f [%+6.2f,%+6.2f]\n",
                    q, blo, bhi, c1.improve, c1.ci_lo, c1.ci_hi, c6.improve, c6.ci_lo, c6.ci_hi)
            @printf(io, "| %.2e | [%.2f, %.2f] | %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] |\n",
                    q, blo, bhi, c1.improve, c1.ci_lo, c1.ci_hi, c6.improve, c6.ci_lo, c6.ci_hi)
        end
        println(io, "\nThe table is the current sensitivity result. A setting earns promotion only if its ",
                    "paired confidence interval and cross-storm checks satisfy the declared gates; no ",
                    "historical V2.0 verdict is carried forward.")
    end
end

function main_inject()
    lib, ξ0, i_decay = _shadow_library()
    i_Bs = _bs_index(lib)
    decay_cap = _shadow_decay_cap(ξ0, i_decay)
    q_bs = _q_bs(ξ0, i_decay, i_Bs)
    println("Injection-adaptive EKF: adapt [decay, Bs] vs V2.1 frozen-tail ablation + persistence, leads ", LEADS)
    @printf("  discovered ξ_Bs=%.4f  box=%s  q_Bs=%.3e  (decay cap %.4f, Q_decay=%.0e)\n",
            ξ0[i_Bs], string(BS_BOX), q_bs, decay_cap, Q_COEFF)
    println("="^80)
    cal = _load_calibration_for_model(LiveVerifyConfig(model = :v2))
    all_rows = replay_all(lib, ξ0, i_decay, i_Bs, cal, q_bs)
    for s in STORMS
        @printf("  %-24s rows=%d\n", s.name, nrow(all_rows[all_rows.storm .== s.name, :]))
    end
    CSV.write(OUT_CSV_INJ, all_rows)

    candidates = NamedTuple[]
    open(OUT_MD_INJ, "w") do io
        println(io, "# Injection-adaptive EKF — V2.1 frozen-tail ablation and persistence\n")
        println(io, "Same causal protocol as the decay-only retirement test, but the EKF also adapts the discovered ",
                    "southward-injection coefficient `Bs` (box $(BS_BOX), q_Bs ≈ $(round(_q_bs(ξ0,i_decay,i_Bs);sigdigits=2))), ",
                    "giving the filter a downward lever. `improve` = paired |err stronger-baseline| − |err EKF| ",
                    "(positive = EKF beats the stronger of {V2.1 frozen-tail, persistence}); v3 needs its 95% CI > 0. `bias` = ",
                    "mean(obs − EKF). `fair` = ",
                    "max|ekf_fixed − frozen_tail|, must be ≈0.\n")
        println(io, "| lead [h] | regime | n | RMSE frozen-tail | RMSE EKF | RMSE pers | stronger | improve [nT] (95% CI) | bias | fair |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|")
        println("\n  lead regime    n frozen  EKF   pers  strong       improve[CI]            bias  fair")
        for h in LEADS
            sub_h = all_rows[all_rows.lead .== h, :]
            for (label, sub) in (("pooled", sub_h),
                                 ("main",   sub_h[isfinite.(sub_h.rate) .& (sub_h.rate .< MAIN_RATE), :]))
                c = _cell(sub); c === nothing && continue
                bias = mean(sub.obs .- sub.ekf)
                @printf("  %3d  %-6s %4d %5.1f %5.1f %5.1f  %-4s  %+5.2f [%+5.2f,%+5.2f]  %+5.2f %.2f\n",
                        h, label, c.n, c.rmse_frozen, c.rmse_ekf, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, bias, c.fair)
                @printf(io, "| %d | %s | %d | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] | %+.2f | %.2f |\n",
                        h, label, c.n, c.rmse_frozen, c.rmse_ekf, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, bias, c.fair)
                c.ci_lo > 0 && push!(candidates, (lead = h, regime = label, rows = sub, cell = c))
            end
        end
        dlo, dhi = minimum(all_rows.adapted_decay), maximum(all_rows.adapted_decay)
        blo, bhi = minimum(all_rows.adapted_bs), maximum(all_rows.adapted_bs)
        max_fair = maximum(abs.(all_rows.ekf_fixed .- all_rows.frozen_tail))
        promotable = false; detail = ""
        for cand in candidates
            signs = _per_storm_signs(cand.rows); npos = count(>(0), values(signs)); ntot = length(signs)
            if ntot >= 3 && npos >= ntot - 1
                promotable = true
                detail = "lead $(cand.lead) $(cand.regime): improve $(round(cand.cell.improve;digits=2)) nT " *
                         "[$(round(cand.cell.ci_lo;digits=2)),$(round(cand.cell.ci_hi;digits=2))] vs $(cand.cell.stronger), " *
                         "$(npos)/$(ntot) storms positive"
                break
            end
        end
        verdict = max_fair > 2.0 ? "INCONCLUSIVE: fairness gap $(round(max_fair;digits=2)) nT — rollout mismatch" :
                  promotable ? "PROMOTABLE: $detail" :
                  "NOT PROMOTABLE: injection adaptation does not beat the stronger of {V2.1 frozen-tail, persistence} with CI>0 at any lead/regime"
        println(io, "\nAdapted decay ∈ [", round(dlo;digits=3), ", ", round(dhi;digits=3), "] (cap ", decay_cap,
                    "); adapted Bs ∈ [", round(blo;digits=3), ", ", round(bhi;digits=3), "] (box ", BS_BOX,
                    "). Max fairness gap: ", round(max_fair;digits=2), " nT.\n")
        println(io, "**v3 promotion verdict:** ", verdict)
        println("\n  adapted Bs ∈ [", round(blo;digits=3), ", ", round(bhi;digits=3), "]   v3 verdict: ", verdict)
    end
    println("  wrote ", OUT_CSV_INJ, " and ", OUT_MD_INJ)
    return all_rows
end

# ---- CRC self-test: independent oracles for the injection variant ----
function _selftest_inject()
    lib, ξ0, i_decay = _shadow_library(); i_Bs = _bs_index(lib)
    decay_cap = _shadow_decay_cap(ξ0, i_decay)
    # Oracle A — discovered injection coefficient sanity (the lever we adapt is the real Bs term).
    @assert isfinite(ξ0[i_Bs]) && ξ0[i_Bs] < 0.0 "V2.1 Bs coefficient is not a finite driver: $(ξ0[i_Bs])"
    drv = (V = 500.0, Bz = -10.0, By = 0.0, n = 6.0, Pdyn = 2.0)   # Bs>0 (southward) → injection active
    # Oracle B — CONTINUITY: with the injection held at its discovered value, the 2-coeff forecast is
    # byte-identical to the decay-only _shadow_forecast. This is the strongest correctness check: injection
    # adaptation OFF must reduce exactly to the retired EKF.
    for h in (1, 3, 6), dc in (decay_cap, -0.2)
        a = _inject_forecast(lib, ξ0, i_decay, i_Bs, dc, ξ0[i_Bs], -120.0, drv, -118.0, cal_selftest(); nsteps = h)
        b = _shadow_forecast(lib, ξ0, i_decay, dc, -120.0, drv, -118.0, cal_selftest(); nsteps = h)
        @assert a == b "continuity broken at h=$h decay=$dc: inject=$a shadow=$b"
    end
    # Oracle C — DOWNWARD LEVER: strengthening the injection (more negative Bs) makes the rollout MORE
    # negative under southward driving — the mechanism that lets the filter track a fast drop.
    v1_disc = _inject_forecast(lib, ξ0, i_decay, i_Bs, decay_cap, ξ0[i_Bs],
        -120.0, drv, -118.0, cal_selftest(); nsteps=3)[1]
    v1_strong = _inject_forecast(lib, ξ0, i_decay, i_Bs, decay_cap, ξ0[i_Bs]*1.6,
        -120.0, drv, -118.0, cal_selftest(); nsteps=3)[1]
    @assert v1_strong < v1_disc "stronger injection did not deepen the forecast ($v1_strong !< $v1_disc)"
    # Oracle D — SIGN BOX binds: a causal bootstrap keeps decay ≤ cap and Bs within its (≤0) box.
    f = causal_bootstrap_inject(lib, ξ0, i_decay, i_Bs, 2022; q_bs = _q_bs(ξ0, i_decay, i_Bs))
    dc, bs = current_coeffs(f)
    @assert dc <= decay_cap + 1e-9 "decay above cap: $dc"
    @assert BS_BOX[1] - 1e-9 <= bs <= BS_BOX[2] + 1e-9 "Bs left its box: $bs"
    println("  ✓ injection self-test: ξ_Bs sane, continuity to decay-only EKF, downward lever, sign box binds")
    return true
end
_cal_cache = Ref{Any}(nothing)
cal_selftest() = (_cal_cache[] === nothing && (_cal_cache[] = _load_calibration_for_model(LiveVerifyConfig(model = :v2))); _cal_cache[])

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest_inject()
    main_inject()
    let
        lib, ξ0, i_decay = _shadow_library()
        _sweep(lib, ξ0, i_decay, _bs_index(lib), cal_selftest())
    end
end
