# ekf_storm_replay.jl — storm-time replay: constrained EKF vs the V2.1 frozen-tail ablation.
#
# Purpose: build the storm-time promotion evidence for the constrained-EKF shadow (the v3 candidate). The
# locked-live record is quiet-only, and the EKF only differs during storms, so we score the EKF's
# 1 h forecast against the V2.1 frozen-tail ablation (and persistence) on the SAME named storms as
# storm_replay.jl. The
# evaluation is fully CAUSAL: the EKF is bootstrapped on PRE-storm OMNI only (years < the storm year), then
# walked over the storm's hourly anchors (assimilate anchor -> adapt decay -> issue 1 h forecast). It reuses
# the retired `_shadow_forecast` / constrained `_new_filter` with a decay cap
# derived from the current V2.1 core and the current replay feature rows.
#
# Promotion criterion (1 h): EKF is v3-promotable iff the pooled paired improvement |e_v2|-|e_ekf| has a
# bootstrap 95% CI strictly above 0 (consistent 1 h gain) AND the adapted decay stays in the stable band
# [decay_cap-2, decay_cap]. This is an exploratory extension audit, not part of
# the served V2.1 method or its paper evidence.
#
# Run from the package root: julia --project=. validation/operational/ekf_storm_replay.jl

using Dates, DataFrames, Statistics, Printf, CSV, Random
include(joinpath(@__DIR__, "paths.jl"))
include(joinpath(OPERATIONAL_PACKAGE_ROOT, "examples", "live_forecast_verify.jl"))  # replay_recent_table, _omni_replay_inputs, LiveVerifyConfig, _load_calibration_for_model
include(joinpath(@__DIR__, "ekf_shadow.jl"))   # _shadow_library, _new_filter, _shadow_forecast, BOOT_YEAR0

const OMNI    = OPERATIONAL_OMNI
const OUT_CSV = joinpath(OPERATIONAL_OUTPUT_DIR, "ekf_storm_replay_scored.csv")
const OUT_MD  = joinpath(OPERATIONAL_OUTPUT_DIR, "ekf_storm_replay_report.md")

# Seven real G4/G5 storms spanning 2023–2025 (peak Dst from paper_v2_monitor/data/omni_extracted.csv): widening the
# set lifts the main-phase sample and tests cross-storm sign-consistency, both required by the v3 bar.
const STORMS = [
    (name = "May 2024 (Gannon, G5)", t0 = DateTime(2024, 5, 9, 12),  t1 = DateTime(2024, 5, 13, 0)),  # -339 nT
    (name = "Oct 2024 (G4)",         t0 = DateTime(2024, 10, 9, 0),  t1 = DateTime(2024, 10, 12, 0)), # -148 nT
    (name = "Mar 2023 (G4)",         t0 = DateTime(2023, 3, 23, 0),  t1 = DateTime(2023, 3, 25, 12)), # -125 nT
    (name = "Apr 2023 (G4)",         t0 = DateTime(2023, 4, 23, 0),  t1 = DateTime(2023, 4, 25, 12)), # -165 nT
    (name = "Nov 2023 (G4)",         t0 = DateTime(2023, 11, 4, 0),  t1 = DateTime(2023, 11, 6, 12)), # -163 nT
    (name = "Aug 2024 (G4)",         t0 = DateTime(2024, 8, 11, 0),  t1 = DateTime(2024, 8, 13, 12)), # -188 nT
    (name = "Jan 2025 (G4)",         t0 = DateTime(2024, 12, 31, 0), t1 = DateTime(2025, 1, 2, 12)),  # -212 nT
]
const LEADS    = [1, 2, 3, 6]   # forecast horizons (h)
const MAIN_RATE = -15.0          # dDst/dt below this (nT/h) = rapid main phase
_rmse(r) = sqrt(mean(abs2, r))

"Carry-forward fill of non-finite entries (last-observation-carried-forward)."
function _ffill!(v)
    l = NaN
    for i in eachindex(v)
        isfinite(v[i]) ? (l = v[i]) : (isfinite(l) && (v[i] = l))
    end
    return v
end

"Causal constrained-EKF bootstrap over OMNI [BOOT_YEAR0, year_end] — no storm-year leakage. Returns the filter."
function causal_bootstrap(lib, ξ0, i_decay, year_end::Int)
    # Persisted EKF replay CSVs predate causal cleaning and are retired negative-result artifacts (not rerun);
    # causal=true keeps this bootstrap strictly causal if the retired script is ever re-executed.
    df = parse_omni2(OMNI; year_start = BOOT_YEAR0, year_end = year_end); clean_omni_data!(df; causal = true)
    V  = _ffill!(Float64.(coalesce.(df.V, NaN)));  Bz = _ffill!(Float64.(coalesce.(df.Bz, NaN)))
    By = _ffill!(Float64.(coalesce.(df.By, NaN))); nn = _ffill!(Float64.(coalesce.(df.n, NaN)))
    Pd = [dynamic_pressure(nn[k], V[k]) for k in eachindex(V)]
    dst = Float64.(coalesce.(df.Dst_star, NaN))
    f = _new_filter(lib, ξ0, i_decay, isfinite(dst[1]) ? dst[1] : 0.0)
    for k in 1:length(dst)-1
        (isfinite(dst[k+1]) && isfinite(V[k])) || continue
        assimilation_predict!(f, (V = V[k], Bz = Bz[k], By = By[k], n = nn[k], Pdyn = Pd[k]))
        assimilation_update!(f, dst[k+1])
    end
    return f
end

"""Walk one storm's hourly anchors through the constrained EKF `f` (mutated), emitting one row per
(issue, lead). The filter assimilates each anchor ONCE per issue hour (advancing the adapted decay), then
issues a forecast at every lead in LEADS by free-running the discovered ODE forward h hours. `ekf_fixed`
repeats the rollout with the FIXED discovered decay; ekf_fixed≈frozen_tail at every lead is the fairness
oracle that the multi-step rollout matches the V2.1 frozen-tail engine, so any EKF gap is purely the decay
adaptation. `rate`
is the hourly dDst/dt of the anchor (for the main-phase/recovery split)."""
function replay_ekf_storm(storm, lib, ξ0, i_decay, cal, f)
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
    dst_map = Dict{DateTime,Float64}(zip(dst_times, Float64.(dst_vals)))  # true hourly Dst series
    out = DataFrame(storm = String[], issue_utc = DateTime[], lead = Int[], obs = Float64[],
                    frozen_tail = Float64[], ekf = Float64[], ekf_fixed = Float64[], persistence = Float64[],
                    adapted_decay = Float64[], rate = Float64[])
    issues = sort!(unique(df.issue_dt))
    for it in issues
        g = df[df.issue_dt .== it, :]                       # all leads for this issue hour
        r1 = g[1, :]                                        # issue-time drivers/anchor (lead-invariant)
        drv = (V = Float64(r1.V_kms), Bz = Float64(r1.Bz_nt), By = Float64(r1.By_nt),
               n = Float64(r1.n_cm3), Pdyn = Float64(r1.Pdyn_npa))
        latest = Float64(r1.latest_dst_nt)
        all(isfinite, (drv.V, drv.Bz, drv.By, drv.n, drv.Pdyn, latest)) || continue
        anchor_star = pressure_correct_dst([latest], [drv.Pdyn])[1]   # latest observed Dst -> Dst*
        assimilation_predict!(f, drv); assimilation_update!(f, anchor_star)   # assimilate anchor ONCE per hour
        decay = current_coeffs(f)[1]
        prev1 = get(dst_map, it - Hour(1), NaN)                              # true 1 h delta (NaN if prior hour absent)
        rate = isfinite(prev1) ? latest - prev1 : NaN                        # dDst/dt of the anchor [nT/h]
        for r in eachrow(g)
            (ismissing(r.observation_dst_nt) || ismissing(r.v2_pred_dst_nt)) && continue
            isfinite(Float64(r.observation_dst_nt)) && isfinite(Float64(r.v2_pred_dst_nt)) || continue
            h = Int(r.model_step_hours)
            _, ekf_a = _shadow_forecast(lib, ξ0, i_decay, decay,
                anchor_star, drv, latest, cal; nsteps=h, calibration_features=r)
            _, ekf_f = _shadow_forecast(lib, ξ0, i_decay, ξ0[i_decay],
                anchor_star, drv, latest, cal; nsteps=h, calibration_features=r)
            isfinite(ekf_a) && isfinite(ekf_f) || continue
            push!(out, (storm.name, it, h, Float64(r.observation_dst_nt), Float64(r.v2_pred_dst_nt),
                        ekf_a, ekf_f, latest, decay, rate))
        end
    end
    return out
end

"""Paired RMSE improvement rmse(e_base) − rmse(e_method) (positive = method better) with a 95% CI.
The tested statistic is the paired RMSE difference, matching the RMSE used to select the stronger
baseline. When storm labels are supplied the CI comes from a STORM-LEVEL cluster bootstrap (resample
whole storms with replacement, pooling all their rows) so it respects the strong serial correlation of
storm-time forecast errors — an iid row bootstrap is anti-conservative. Falls back to iid rows when no
storm labels are given."""
function paired_improvement(ebase, emethod; storms = nothing, nboot = 4000, seed = 1)
    m = length(ebase); m == 0 && return (NaN, NaN, NaN)
    length(emethod) == m || throw(DimensionMismatch("paired error vectors differ in length"))
    point = _rmse(ebase) - _rmse(emethod)
    rng = MersenneTwister(seed)
    boot = Vector{Float64}(undef, nboot)
    if storms === nothing
        for b in 1:nboot
            idx = rand(rng, 1:m, m)
            boot[b] = _rmse(@view ebase[idx]) - _rmse(@view emethod[idx])
        end
    else
        length(storms) == m || throw(DimensionMismatch("storms length != errors length"))
        groups = [findall(==(s), storms) for s in unique(storms)]
        G = length(groups)
        for b in 1:nboot
            idx = reduce(vcat, groups[rand(rng, 1:G, G)])
            boot[b] = _rmse(@view ebase[idx]) - _rmse(@view emethod[idx])
        end
    end
    return point, quantile(boot, 0.025), quantile(boot, 0.975)
end

"Stats for a row subset: RMSE of each method, EKF vs the STRONGER (lower-RMSE) baseline with paired CI,
and the fairness gap max|ekf_fixed − frozen_tail| (should be ≈0 for a fair comparison)."
function _cell(rows)
    nrow(rows) == 0 && return nothing
    efrozen = rows.obs .- rows.frozen_tail; eekf = rows.obs .- rows.ekf; ep = rows.obs .- rows.persistence
    rfrozen, re, rp = _rmse(efrozen), _rmse(eekf), _rmse(ep)
    strong_pers = rp <= rfrozen
    Δ, lo, hi = paired_improvement(strong_pers ? ep : efrozen, eekf; storms = rows.storm)
    return (n = nrow(rows), rmse_frozen = rfrozen, rmse_ekf = re, rmse_pers = rp,
            stronger = strong_pers ? "pers" : "frozen-tail", improve = Δ, ci_lo = lo, ci_hi = hi,
            fair = maximum(abs.(rows.ekf_fixed .- rows.frozen_tail)),
            decay_min = minimum(rows.adapted_decay), decay_max = maximum(rows.adapted_decay))
end

"Per-storm sign of EKF-vs-stronger improvement, for the cross-storm-consistency clause of the v3 bar."
function _per_storm_signs(rows)
    signs = Dict{String,Float64}()
    for st in unique(rows.storm)
        c = _cell(rows[rows.storm .== st, :]); c === nothing && continue
        signs[st] = c.improve
    end
    return signs
end

function main_ekf_storm_replay()
    println("EKF storm replay: constrained EKF vs V2.1 frozen-tail ablation + persistence, leads ", LEADS,
            " (causal, pre-storm bootstrap)\n", "="^80)
    lib, ξ0, i_decay = _shadow_library()
    decay_cap = _shadow_decay_cap(ξ0, i_decay)
    cal = _load_calibration_for_model(LiveVerifyConfig(model = :v2))
    boot = Dict{Int,Any}()
    all_rows = DataFrame()
    for s in STORMS
        ye = year(s.t1) - 1
        haskey(boot, ye) || (boot[ye] = causal_bootstrap(lib, ξ0, i_decay, ye))
        rows = replay_ekf_storm(s, lib, ξ0, i_decay, cal, deepcopy(boot[ye]))
        append!(all_rows, rows)
        @printf("  %-24s rows=%d  leads=%s\n", s.name, nrow(rows), string(sort(unique(rows.lead))))
    end
    CSV.write(OUT_CSV, all_rows)

    # lead × regime grid: a cell is a v3 candidate iff EKF beats the STRONGER baseline with CI>0.
    candidates = NamedTuple[]
    open(OUT_MD, "w") do io
        println(io, "# EKF storm replay — constrained EKF vs V2.1 frozen-tail ablation and persistence\n")
        println(io, "Causal head-to-head on ", length(STORMS), " G4/G5 storms (2023–2025). The constrained EKF ",
                    "is bootstrapped on OMNI before each storm year (no leakage), walked over the storm's hourly ",
                    "anchors (assimilate once per hour), then issues a forecast at each lead by free-running the ",
                    "current 20/11 SINDy ODE forward. `improve` is paired RMSE(stronger baseline) − RMSE(EKF) ",
                    "(positive = EKF beats the stronger of {V2.1 frozen-tail, persistence}); the v3 bar needs ",
                    "its storm-cluster 95% CI strictly above zero. `fair` = max|ekf_fixed − frozen_tail|: with ",
                    "the discovered decay held fixed the EKF rollout must reproduce the V2.1 frozen-tail engine.\n")
        println(io, "| lead [h] | regime | n | RMSE frozen-tail | RMSE EKF | RMSE pers | stronger | improve vs stronger [nT] (95% CI) | fair |")
        println(io, "|---|---|---|---|---|---|---|---|---|")
        println("\n  lead regime    n frozen  EKF   pers  strong       improve[CI]            fair")
        for h in LEADS
            sub_h = all_rows[all_rows.lead .== h, :]
            for (label, sub) in (("pooled", sub_h),
                                 ("main",   sub_h[isfinite.(sub_h.rate) .& (sub_h.rate .< MAIN_RATE), :]))
                c = _cell(sub); c === nothing && continue
                @printf("  %3d  %-6s %4d %5.1f %5.1f %5.1f  %-4s  %+5.2f [%+5.2f,%+5.2f]  %.2f\n",
                        h, label, c.n, c.rmse_frozen, c.rmse_ekf, c.rmse_pers, c.stronger,
                        c.improve, c.ci_lo, c.ci_hi, c.fair)
                @printf(io, "| %d | %s | %d | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] | %.2f |\n",
                        h, label, c.n, c.rmse_frozen, c.rmse_ekf, c.rmse_pers, c.stronger,
                        c.improve, c.ci_lo, c.ci_hi, c.fair)
                c.ci_lo > 0 && push!(candidates, (lead = h, regime = label, rows = sub, cell = c))
            end
        end

        # Verdict: promotable iff a cell beats the stronger baseline with CI>0 AND the gain is cross-storm
        # sign-consistent (no single storm carrying it). Otherwise the overlay is retired with evidence.
        decay_lo = minimum(all_rows.adapted_decay); decay_hi = maximum(all_rows.adapted_decay)
        stable = decay_lo >= decay_cap - 2.0 && decay_hi <= decay_cap + 1e-9
        max_fair = maximum(abs.(all_rows.ekf_fixed .- all_rows.frozen_tail))
        promotable = false; detail = ""
        for cand in candidates
            signs = _per_storm_signs(cand.rows)
            npos = count(>(0), values(signs)); ntot = length(signs)
            if ntot >= 3 && npos >= ntot - 1   # consistent across storms (allow one exception)
                promotable = true
                detail = "lead $(cand.lead) $(cand.regime): improve $(round(cand.cell.improve;digits=2)) nT " *
                         "[$(round(cand.cell.ci_lo;digits=2)),$(round(cand.cell.ci_hi;digits=2))] vs $(cand.cell.stronger), " *
                         "$(npos)/$(ntot) storms positive"
                break
            end
        end
        verdict = !stable ? "NOT PROMOTABLE: adapted decay left the stable band [$(decay_cap-2), $(decay_cap)]" :
                  max_fair > 2.0 ? "INCONCLUSIVE at multi-step: fairness gap max|ekf_fixed−frozen_tail|=$(round(max_fair;digits=2)) nT; trust only fair≈0 leads" :
                  promotable ? "PROMOTABLE: $detail" :
                  "NOT PROMOTABLE: at no lead or regime does the EKF beat the stronger of {V2.1 frozen-tail, persistence} with a 95% CI above 0; the constrained-EKF overlay should remain in shadow"
        println(io, "\nDecay band over all storms: [", round(decay_lo; digits=3), ", ", round(decay_hi; digits=3),
                    "] (cap ", decay_cap, "). Max fairness gap (all leads): ", round(max_fair; digits=2), " nT.\n")
        println(io, "**v3 promotion verdict:** ", verdict)
        println("\n  v3 verdict: ", verdict)
    end
    println("  wrote ", OUT_CSV, " and ", OUT_MD)
    return all_rows
end

# ---- CRC self-test (independent oracles; runs before main) ----
function _selftest()
    lib, ξ0, i_decay = _shadow_library()
    decay_cap = _shadow_decay_cap(ξ0, i_decay)
    cal = _load_calibration_for_model(LiveVerifyConfig(model = :v2))
    f = causal_bootstrap(lib, ξ0, i_decay, 2022)
    # Oracle 1: the constraint binds — the adapted decay never exceeds the cap.
    @assert current_coeffs(f)[1] <= decay_cap + 1e-9 "decay above cap after bootstrap"
    # Oracle 2: a 1 h EKF forecast on a representative anchor is finite and in a physical Dst range.
    drv = (V = 450.0, Bz = -5.0, By = 0.0, n = 5.0, Pdyn = 1.5)
    _, ekf = _shadow_forecast(lib, ξ0, i_decay, current_coeffs(f)[1], -20.0,
        drv, -18.0, cal)
    @assert isfinite(ekf) && -2000 < ekf < 100 "EKF forecast non-finite or out of physical range: $ekf"
    # Oracle 3: under quiet conditions the adapted-decay and fixed-decay centers stay close.
    qdrv = (V=400.0, Bz=0.0, By=0.0, n=5.0, Pdyn=1.5)
    _, ekf_q = _shadow_forecast(lib, ξ0, i_decay, current_coeffs(f)[1], 2.0,
        qdrv, 2.0, cal)
    _, fixed_q = _shadow_forecast(lib, ξ0, i_decay, ξ0[i_decay], 2.0,
        qdrv, 2.0, cal)
    @assert(
        isfinite(ekf_q) && abs(ekf_q - fixed_q) <= 5.0,
        "quiet EKF differs from fixed V2.1 core by more than 5 nT: $(ekf_q-fixed_q)",
    )
    # Oracle 4: nsteps default is byte-identical to an explicit single step (live 1 h path must be unchanged).
    qdrv = (V=450.0, Bz=-5.0, By=0.0, n=5.0, Pdyn=1.5)
    @assert _shadow_forecast(lib, ξ0, i_decay, -0.1, -50.0, qdrv, -48.0,
                cal) ==
            _shadow_forecast(lib, ξ0, i_decay, -0.1, -50.0, qdrv, -48.0,
                cal; nsteps=1) "nsteps default != 1-step"
    # Oracle 5: multi-step monotone recovery — with no southward driving (Bz=0) and a negative restoring
    # decay, the v1 rollout must move a negative Dst* TOWARD zero with more steps. This is the very
    # mechanism that makes the EKF over-recover at onset; the oracle pins its sign.
    rdrv = (V=400.0, Bz=0.0, By=0.0, n=5.0, Pdyn=1.5)
    v1_1 = _shadow_forecast(lib, ξ0, i_decay, -0.1, -100.0, rdrv, -100.0,
        cal; nsteps=1)[1]
    v1_3 = _shadow_forecast(lib, ξ0, i_decay, -0.1, -100.0, rdrv, -100.0,
        cal; nsteps=3)[1]
    @assert v1_3 > v1_1 "multi-step recovery not monotone toward zero (v1_3=$v1_3 ≤ v1_1=$v1_1)"
    println("  ✓ EKF storm-replay self-test: decay≤cap, finite+physical, quiet well-posed, nsteps=1 identity, multi-step recovery monotone")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest()
    main_ekf_storm_replay()
end
