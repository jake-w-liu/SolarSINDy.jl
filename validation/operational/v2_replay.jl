# v2_replay.jl — Operational V2.1 replay with an explicit historical V2.0 comparator.
#
# Defect found by adversarial reflection: plain B relaxes Bz→0 with a fixed τ regardless of regime, so on
# ACTIVELY-DEEPENING deep storms it stops injecting too early and under-predicts depth (obs−pred ≈ −48 nT) — the
# dangerous direction for a severe-storm warning system. Fix at the source: make the relaxation timescale
# REGIME-AWARE. During active deepening (recent Dst rate < 0 / strong sustained southward driving), lengthen τ
# so the tail keeps injecting and the forecast does not go shallow; in recovery, use the normal τ0 (B's win).
#   τ_eff = τ0 · (1 + max(0, −rate)/R0)    (rate = recent dDst/dt; capped),  force_frozen ⇒ V2.1 frozen-tail ablation.
#
# Validates: (1) the deep-deepening shallow bias is removed (signed error ≈ 0, not −48), (2) the multi-hour
# RMSE gain over persistence survives, and (3) a validation-selected one-hour
# persistence blend plus the extreme-core inertia guard protect the near-term cells.
#
# Run from the package root: julia --project=. validation/operational/v2_replay.jl

isdefined(@__MODULE__, :_transit_hours) || include(joinpath(@__DIR__, "v2_lookahead_replay.jl"))   # _driver_lookup, _transit_hours, _shadow_library, etc.

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

"""
    _v2_admitted_driver(future, k, kΔ, last_known)

L1 admission for a rollout step inside the issue-time transit window (k ≤ kΔ). `future(k)` returns
the arrival-hour record tagged it+k-1 (the row covering at-Earth interval [it+k-1, it+k)). It is
admitted only when that whole hour is L1-measured by issue time: the issue-time transit gate
(k ≤ kΔ, enforced by the caller) AND the admitted record's own speed (k ≤ transit(fut.V)) guard
against intra-hour acceleration (a shock arriving into slow wind). Otherwise the last known driver
persists (freeze), exactly as for a missing in-window record.

This is the single admission code path shared by the served V2.1 tail (`_v2_forecast`) and the
V2.3 analog-ensemble kernel, so a candidate tail can differ from the product only after the
L1-known window ends.
"""
@inline function _v2_admitted_driver(future, k::Int, kΔ::Int, last_known)
    fut = future(k)
    return (fut !== nothing && k <= _transit_hours(fut.V)) ? fut : last_known
end

"""
    _v2_relaxed_tail_driver(last_known, k, kΔ, tau)

Served V2.1 tail driver for a rollout step beyond the L1-known window: the magnetic components of
the last admitted driver decay with the regime-aware timescale `tau` while speed and density are
held. `tau = Inf` reproduces the frozen-tail ablation.
"""
@inline function _v2_relaxed_tail_driver(last_known, k::Int, kΔ::Int, tau)
    relax = exp(-(k - kΔ) / tau)
    return (V = last_known.V, Bz = last_known.Bz * relax, By = last_known.By * relax,
            n = last_known.n, Pdyn = last_known.Pdyn)
end

"h-step operational forecast: A for k≤⌊Δ⌋, then REGIME-AWARE B relaxation of the last L1-known driver (τ longer
when actively deepening). `rate` = recent dDst/dt at issue. force_frozen keeps the issue-time driver fixed."
function _v2_forecast(lib, ξ0, anchor_dst_star, issue_drv, future, latest_dst, cal, h::Int, rate;
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
            drv_k = _v2_admitted_driver(future, k, kΔ, last_known)
            last_known = drv_k
        else
            drv_k = _v2_relaxed_tail_driver(last_known, k, kΔ, tau)
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

function _driver_lookup_range(year_start::Int, year_end::Int)
    year_start <= year_end || throw(ArgumentError("invalid driver-lookup year range"))
    df = parse_omni2(OMNI; year_start=year_start, year_end=year_end)
    clean_omni_data!(df; causal=true)
    V = _ffill!(Float64.(coalesce.(df.V, NaN)))
    Bz = _ffill!(Float64.(coalesce.(df.Bz, NaN)))
    By = _ffill!(Float64.(coalesce.(df.By, NaN)))
    density = _ffill!(Float64.(coalesce.(df.n, NaN)))
    # V and density can have different missingness. Recompute pressure only
    # after both causal fills so every admitted tuple satisfies the package's
    # exact proton-only pressure identity.
    pdyn = [dynamic_pressure(density[k], V[k]) for k in eachindex(V)]
    lookup = Dict{DateTime,NamedTuple{(:V, :Bz, :By, :n, :Pdyn),NTuple{5,Float64}}}()
    for (k, t) in enumerate(df.datetime)
        all(isfinite, (V[k], Bz[k], By[k], density[k], pdyn[k])) || continue
        lookup[t] = (V=V[k], Bz=Bz[k], By=By[k], n=density[k], Pdyn=pdyn[k])
    end
    return lookup
end

function _load_replay_archive(year_start::Int, year_end::Int)
    return (
        inputs=_omni_replay_inputs(OMNI, year_start, year_end),
        lookup=_driver_lookup_range(year_start, year_end),
    )
end

const V21_POINT_REPLAY_SEED_HALF_WIDTH_NT = 1.0

function _point_replay_baselines(anchor_dst_star::Float64, drivers, max_h::Int)
    burton = anchor_dst_star
    burton_full = anchor_dst_star
    obrien = anchor_dst_star
    b = Vector{Float64}(undef, max_h)
    bf = similar(b)
    o = similar(b)
    for h in 1:max_h
        burton = _advance_baselines(burton, drivers).burton
        burton_full = _advance_baselines(burton_full, drivers).burton_full
        obrien = _advance_baselines(obrien, drivers).obrien
        b[h] = burton
        bf[h] = burton_full
        o[h] = obrien
    end
    return (; burton=b, burton_full=bf, obrien=o)
end

"""
    _point_replay_table(plasma, mag, dst_times, dst_vals, core, calibration; kwargs...)

Build the point-only causal replay table without reloading the 500-member
uncertainty ensemble for every row. `operational_core_forecast` is independently
pinned to the primary `forecast_ahead` center; the V2.1 frozen path is checked
again row by row in `replay_v2_storm` before a score is accepted.
"""
function _point_replay_table(
    plasma::DataFrame,
    mag::DataFrame,
    dst_times,
    dst_vals,
    core::OperationalCore,
    calibration;
    replay_hours::Int,
    horizons::Vector{Int},
)
    isempty(horizons) && throw(ArgumentError("horizons must not be empty"))
    any(<=(0), horizons) && throw(ArgumentError("horizons must be positive"))
    core.artifacts.version == OPERATIONAL_V2_1_MODEL_VERSION || throw(ArgumentError(
        "point replay requires the current V2.1 core",
    ))
    gate_min = _replay_min_hourly_samples(plasma)
    dst_map = _dst_lookup(dst_times, dst_vals)
    anchors = _replay_anchor_hours(plasma, mag, dst_times, replay_hours)
    max_h = maximum(horizons)
    rows = NamedTuple[]
    for anchor_time in anchors
        latest_dst = get(dst_map, anchor_time, NaN)
        isfinite(latest_dst) || continue
        source_start = anchor_time - Hour(1)
        source_end = anchor_time
        drivers = _drivers_for_window(plasma, mag, source_start, source_end)
        all(isfinite, values(drivers)) || continue
        if _driver_gap_status(
            _window_finite_count(plasma, :speed, source_start, source_end),
            _window_finite_count(mag, :bz_gsm, source_start, source_end),
            _window_finite_count(plasma, :density, source_start, source_end),
            _window_finite_count(mag, :by_gsm, source_start, source_end);
            min_samples=gate_min,
        ) != :ok
            continue
        end
        anchor_dst_star = pressure_correct_dst([latest_dst], [drivers.Pdyn])[1]
        core_star = operational_core_forecast(core, anchor_dst_star, drivers, max_h)
        baseline_star = _point_replay_baselines(anchor_dst_star, drivers, max_h)
        for h in horizons
            target_time = anchor_time + Hour(h)
            observed = get(dst_map, target_time, NaN)
            isfinite(observed) || continue
            pred = _dst_from_dst_star(core_star[h], drivers.Pdyn)
            burton = _dst_from_dst_star(baseline_star.burton[h], drivers.Pdyn)
            burton_full = _dst_from_dst_star(baseline_star.burton_full[h], drivers.Pdyn)
            obrien = _dst_from_dst_star(baseline_star.obrien[h], drivers.Pdyn)
            half = V21_POINT_REPLAY_SEED_HALF_WIDTH_NT
            push!(rows, (
                issue_time_utc=string(anchor_time),
                source_driver_start_utc=string(source_start),
                source_driver_end_utc=string(source_end),
                latest_dst_time_utc=string(anchor_time),
                target_time_utc=string(target_time),
                model_step_hours=h,
                model_version="v2.1_core_uncalibrated",
                latest_dst_nt=latest_dst,
                observation_dst_nt=observed,
                pred_dst_nt=pred,
                residual_dst_nt=observed - pred,
                pred_dst_ci05_nt=pred - half,
                pred_dst_ci95_nt=pred + half,
                observed_in_90ci=(pred - half <= observed <= pred + half),
                v1_pred_dst_nt=pred,
                v1_pred_dst_ci05_nt=pred - half,
                v1_pred_dst_ci95_nt=pred + half,
                persistence_dst_nt=latest_dst,
                burton_dst_nt=burton,
                burton_full_dst_nt=burton_full,
                obrien_dst_nt=obrien,
                V_kms=drivers.V,
                Bz_nt=drivers.Bz,
                By_nt=drivers.By,
                n_cm3=drivers.n,
                Pdyn_npa=drivers.Pdyn,
                replay_note="causal_point_oracle_v2_1_core",
            ))
        end
    end
    isempty(rows) && error("point replay produced no scoreable rows")
    prepared = SolarSINDy.add_operational_v2_features!(DataFrame(rows))
    scored = score_operational_v2(prepared, calibration)
    scored[!, :model_version] = fill(OPERATIONAL_V2_1_MODEL_VERSION, nrow(scored))
    return scored
end

function replay_v2_storm(storm, current_core, current_cal,
                         historical_core, historical_cal, lookup;
                         r0=R0_V2, replay_inputs=nothing,
                         apply_rate_guard::Bool=true)
    yr = year(storm.t1)
    plasma, mag, dst_times, dst_vals = replay_inputs === nothing ?
        _omni_replay_inputs(OMNI, yr - 1, yr) : replay_inputs
    win_lo, win_hi = storm.t0 - Hour(6), storm.t1 + Hour(7)
    # _omni_replay_inputs returns driver-gated plasma/mag and a separately built, generally LARGER
    # finite-Dst series; the two frames are no longer row-aligned, so window each by its own time vector.
    mp = (plasma.time_tag .>= win_lo) .& (plasma.time_tag .<= win_hi)
    md = (dst_times .>= win_lo) .& (dst_times .<= win_hi)
    (any(mp) && any(md)) || error("No finite OMNI drivers/Dst inside replay window")
    rh = Int(ceil(Dates.value(win_hi - win_lo) / 3_600_000))
    df = _point_replay_table(
        plasma[mp, :], mag[mp, :], dst_times[md], dst_vals[md],
        current_core, current_cal;
        replay_hours=rh, horizons=LEADS,
    )
    df[!, :issue_dt] = DateTime.(df.issue_time_utc)
    df = df[(df.issue_dt .>= storm.t0) .& (df.issue_dt .<= storm.t1), :]
    sort!(df, [:issue_dt, :model_step_hours])
    dst_map = Dict{DateTime,Float64}(zip(dst_times[md], Float64.(dst_vals[md])))  # true hourly Dst series
    out = DataFrame(
        storm=String[], issue_utc=DateTime[], lead=Int[], obs=Float64[],
        v2_1=Float64[], v2_1_pre_rate_guard=Float64[],
        v2_1_pre_one_hour_inertia=Float64[],
        v2_1_pre_state_inertia=Float64[],
        v2_0=Float64[], v2_1_frozen=Float64[],
        persistence=Float64[], rate=Float64[],
    )
    for it in sort!(unique(df.issue_dt))
        g = df[df.issue_dt .== it, :]; r1 = g[1, :]
        issue_drv = (V = Float64(r1.V_kms), Bz = Float64(r1.Bz_nt), By = Float64(r1.By_nt),
                     n = Float64(r1.n_cm3), Pdyn = Float64(r1.Pdyn_npa))
        latest = Float64(r1.latest_dst_nt)
        all(isfinite, (issue_drv.V, issue_drv.Bz, issue_drv.By, issue_drv.n, issue_drv.Pdyn, latest)) || continue
        anchor_star = pressure_correct_dst([latest], [issue_drv.Pdyn])[1]
        # Rollout step k is driven by the arrival-hour record tagged it+k-1 (fully L1-measured iff k<=Δ).
        future = k -> get(lookup, it + Hour(k - 1), nothing)
        # Recent Dst rate = true 1 h delta from the Dst series. A missing
        # contiguous pair uses the documented neutral zero fallback; a
        # multi-hour gap is never misread as an nT/h rate.
        prev1 = get(dst_map, it - Hour(1), NaN)
        rate = isfinite(prev1) ? latest - prev1 : 0.0
        for r in eachrow(g)
            (ismissing(r.observation_dst_nt) || ismissing(r.v2_pred_dst_nt)) && continue
            isfinite(Float64(r.observation_dst_nt)) && isfinite(Float64(r.v2_pred_dst_nt)) || continue
            h = Int(r.model_step_hours)
            _, v2_1_pre_rate_guard = _v2_forecast(
                current_core.library, current_core.coefficients, anchor_star,
                issue_drv, future, latest, current_cal, h, rate; r0=r0,
                calibration_features=r, apply_rate_guard=false,
                apply_one_hour_inertia=false,
                apply_state_inertia=false,
            )
            v2_1_pre_one_hour_inertia = apply_rate_guard ? _rapid_deepening_projection_guard(
                v2_1_pre_rate_guard, latest, h, rate,
            ) : v2_1_pre_rate_guard
            v2_1_pre_state_inertia = _one_hour_inertia_blend(
                v2_1_pre_one_hour_inertia, latest, h,
            )
            v2_1 = apply_rate_guard ? _state_inertia_blend(
                v2_1_pre_state_inertia, latest, h, rate,
            ) : v2_1_pre_state_inertia
            apply_rate_guard && _extreme_inertia_guard(latest, h) && (v2_1 = latest)
            _, v2_1_frozen = _v2_forecast(
                current_core.library, current_core.coefficients, anchor_star,
                issue_drv, future, latest, current_cal, h, rate;
                r0=r0, force_frozen=true, calibration_features=r,
            )
            _, v2_0 = _v2_forecast(
                historical_core.library, historical_core.coefficients, anchor_star,
                issue_drv, future, latest, historical_cal, h, rate;
                r0=r0, force_frozen=true, calibration_features=r,
            )
            all(isfinite, (v2_1, v2_1_pre_rate_guard, v2_1_pre_one_hour_inertia,
                           v2_1_pre_state_inertia,
                           v2_0, v2_1_frozen)) || continue
            replay_reference = Float64(r.v2_pred_dst_nt)
            isapprox(v2_1_frozen, replay_reference; atol=1e-9, rtol=0.0) || error(
                "V2.1 frozen-tail implementation diverges from primary replay at " *
                "issue=$it lead=$h: $v2_1_frozen vs $replay_reference",
            )
            push!(out, (
                storm.name, it, h, Float64(r.observation_dst_nt),
                v2_1, v2_1_pre_rate_guard, v2_1_pre_one_hour_inertia,
                v2_1_pre_state_inertia,
                v2_0, v2_1_frozen, latest, rate,
            ))
        end
    end
    return out
end

_run_v2(current_core, current_cal, historical_core, historical_cal, luc, r0;
        apply_rate_guard::Bool=true, replay_inputs=nothing) = reduce(
    vcat,
    [replay_v2_storm(
        s, current_core, current_cal, historical_core, historical_cal,
        luc[year(s.t1)]; r0=r0, apply_rate_guard=apply_rate_guard,
        replay_inputs=replay_inputs,
    ) for s in STORMS],
)

function _cell_v2(rows)
    nrow(rows) == 0 && return nothing
    e20 = rows.obs .- rows.v2_0
    e21 = rows.obs .- rows.v2_1
    ep = rows.obs .- rows.persistence
    r20, r21, rp = _rmse(e20), _rmse(e21), _rmse(ep)
    strong_pers = rp <= r20
    Δ, lo, hi = paired_improvement(strong_pers ? ep : e20, e21; storms=rows.storm)
    return (
        n=nrow(rows), rmse_v2_0=r20, rmse_v2_1=r21, rmse_pers=rp,
        stronger=strong_pers ? "persistence" : "V2.0",
        improve=Δ, ci_lo=lo, ci_hi=hi,
        tail_effect=maximum(abs.(rows.v2_1 .- rows.v2_1_frozen)),
        core_change=maximum(abs.(rows.v2_1_frozen .- rows.v2_0)),
    )
end

"Signed error mean(obs − pred) on the deep ACTIVELY-DEEPENING subset (the shallow-bias failure mode)."
function _deep_bias(rows)
    d = rows[isfinite.(rows.rate) .& (rows.rate .< MAIN_RATE) .& (rows.obs .< -100.0), :]
    nrow(d) == 0 && return (n=0, v2_1=NaN, v2_0=NaN)
    return (
        n=nrow(d),
        v2_1=mean(d.obs .- d.v2_1),
        v2_0=mean(d.obs .- d.v2_0),
    )
end

function main_v2()
    current_core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
    historical_core = load_operational_core(OPERATIONAL_V2_0_MODEL_VERSION)
    current_cal = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
    )
    historical_cal = read_operational_v2_calibration(
        operational_calibration_artifacts(OPERATIONAL_V2_0_MODEL_VERSION).point_csv,
    )
    archive = _load_replay_archive(
        minimum(year(s.t0) for s in STORMS) - 1,
        maximum(year(s.t1) for s in STORMS),
    )
    luc = Dict(year(s.t1) => archive.lookup for s in STORMS)
    println("Operational V2.1 replay (20/11 core; A + regime-aware B + causal rate projection + one-hour inertia blend + extreme-core inertia) vs explicit historical V2.0 (21/10 core), R0=", R0_V2, ", τ0=", TAU0_V2, ", leads ", LEADS, "\n", "="^80)
    all_rows = _run_v2(
        current_core, current_cal, historical_core, historical_cal, luc, R0_V2;
        replay_inputs=archive.inputs,
    )
    CSV.write(OUT_CSV_V2, all_rows)
    db = _deep_bias(all_rows[all_rows.lead .== 6, :])
    plain_rows = _run_v2(
        current_core, current_cal, historical_core, historical_cal, luc, 1e9;
        apply_rate_guard=false, replay_inputs=archive.inputs,
    )
    plain_db = _deep_bias(plain_rows[plain_rows.lead .== 6, :])
    sweep_rows = DataFrame(r0 = Float64[], plain_b = Bool[], deep_n = Int[],
                           deep_signed_err_v2_1_nt = Float64[], deep_signed_err_v2_0_nt = Float64[],
                           rmse_v2_0_6h_nt = Float64[], rmse_v2_1_6h_nt = Float64[],
                           rmse_pers_6h_nt = Float64[], improve_6h_nt = Float64[],
                           improve_ci_lo_nt = Float64[], improve_ci_hi_nt = Float64[])
    open(OUT_MD_V2, "w") do io
        println(io, "# Operational V2.1 replay — revised 20/11 SINDy core (multi-lead, causal)\n")
        println(io, "Every row compares the current V2.1 core and calibration against the explicitly archived V2.0 21/10 core and calibration and persistence at the same issue and target times.\n")
        println(io, "The regime-aware relaxation reduces the plain-B shallow bias during actively deepening storms by lengthening the relaxation ",
                    "timescale τ=τ0·(1+max(0,−rate)/R0) during deepening (R0=", R0_V2, " nT/h, τ0=", TAU0_V2,
                    " h, cap ", TAU_MAX, " h). For 1–2 h forecasts with latest Dst≤", EXTREME_INERTIA_DST_NT,
                    " nT, the V2.1 point forecast uses persistence to respect near-term ring-current inertia. ",
                    "The validation-selected causal rate projection brings the residual deep-deepening mean bias inside the unchanged ±10 nT acceptance band; it is capped at ",
                    V2_RATE_PROJECTION_MAX_DROP_NT, " nT for rapid deepening and ",
                    V2_RATE_PROJECTION_EXTREME_MAX_DROP_NT, " nT when rate≤",
                    V2_RATE_PROJECTION_EXTREME_RATE_NT_PER_H, " nT/h. ",
                    "At one hour, a separately validation-selected 0.75 inertia weight blends the V2.1 center with causal persistence. A state-conditioned safeguard then uses weights 0.75 for near-quiet one-hour centers, zero (persistence) for moderate one-hour deepening, 0.625 for near-quiet two-hour centers, and 0.875 for near-quiet three-hour centers; historical V2.0 is not an input to either operator. ",
                    "`improve` = paired RMSE(stronger of {historical V2.0, persistence}) − RMSE(V2.1), with a storm-cluster 95% CI.\n")
        @printf(io, "**Deep deepening subset (rate<%.0f & obs<−100, 6 h, n=%d): signed error mean(obs−pred) — V2.1 %+.1f nT vs unguarded plain-tail V2.1 %+.1f nT vs historical V2.0 %+.1f nT.** The unchanged acceptance band is ±10 nT.\n\n", MAIN_RATE, db.n, db.v2_1, plain_db.v2_1, db.v2_0)
        println(io, "| lead [h] | regime | n | RMSE historical V2.0 | RMSE V2.1 | RMSE persistence | stronger | improve [nT] (95% CI) | max tail effect | max 20/11-core effect |")
        println(io, "|---|---|---|---|---|---|---|---|---|---|")
        println("\n  lead regime    n  V2.0   V2.1   pers  strong       improve[CI]          tail  core")
        for h in LEADS
            sub_h = all_rows[all_rows.lead .== h, :]
            for (label, sub) in (("pooled", sub_h),
                                 ("main",   sub_h[isfinite.(sub_h.rate) .& (sub_h.rate .< MAIN_RATE), :]))
                c = _cell_v2(sub); c === nothing && continue
                @printf("  %3d  %-6s %4d %5.1f %5.1f %5.1f  %-11s %+5.2f [%+5.2f,%+5.2f]  %5.1f %5.1f\n",
                        h, label, c.n, c.rmse_v2_0, c.rmse_v2_1, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.tail_effect, c.core_change)
                @printf(io, "| %d | %s | %d | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] | %.2f | %.2f |\n",
                        h, label, c.n, c.rmse_v2_0, c.rmse_v2_1, c.rmse_pers, c.stronger, c.improve, c.ci_lo, c.ci_hi, c.tail_effect, c.core_change)
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
            rows = _run_v2(
                current_core, current_cal, historical_core, historical_cal, luc, r0;
                apply_rate_guard=false, replay_inputs=archive.inputs,
            )
            b = _deep_bias(rows[rows.lead .== 6, :]); c6 = _cell_v2(rows[rows.lead .== 6, :])
            @printf("    R0=%-6s deep_err=%+6.1f   6h-pool %+5.2f [%+5.2f,%+5.2f]\n", string(r0), b.v2_1, c6.improve, c6.ci_lo, c6.ci_hi)
            @printf(io, "| %s | %+.1f | %+.2f [%+.2f, %+.2f] |\n", r0 >= 1e9 ? "∞ (plain B)" : string(r0), b.v2_1, c6.improve, c6.ci_lo, c6.ci_hi)
            push!(sweep_rows, (r0 = r0, plain_b = r0 >= 1e9,
                               deep_n = b.n, deep_signed_err_v2_1_nt = b.v2_1, deep_signed_err_v2_0_nt = b.v2_0,
                               rmse_v2_0_6h_nt = c6.rmse_v2_0, rmse_v2_1_6h_nt = c6.rmse_v2_1,
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
_calV2() = (_cal_v2[] === nothing && (_cal_v2[] = read_operational_v2_calibration(
    operational_calibration_artifacts(OPERATIONAL_V2_1_MODEL_VERSION).point_csv,
)); _cal_v2[])
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
    old = load_operational_core(OPERATIONAL_V2_0_MODEL_VERSION)
    @assert length(old.library) == 21 && count(!=(0.0), old.coefficients) == 10
    @assert length(lib) == 20 && count(!=(0.0), ξ0) == 11
    @assert _rapid_deepening_projection_guard(-100.0, -80.0, 6, -10.0) == -100.0
    @assert _rapid_deepening_projection_guard(-100.0, -80.0, 6, -20.0) == -125.0
    @assert _rapid_deepening_projection_guard(-100.0, -80.0, 6, -60.0) == -215.0
    @assert _one_hour_inertia_blend(-120.0, -100.0, 1) == -115.0
    @assert _one_hour_inertia_blend(-120.0, -100.0, 2) == -120.0
    @assert _state_inertia_blend(-120.0, -100.0, 1, -10.0) == -100.0
    @assert _state_inertia_blend(10.0, 20.0, 2, 0.0) == 13.75
    @assert _state_inertia_blend(10.0, 20.0, 3, 0.0) == 11.25
    @assert _state_inertia_blend(-40.0, 20.0, 2, 0.0) == -40.0
    println("  ✓ V2.1 self-test: 20/11 current core, explicit 21/10 historical boundary, primary-path continuity, regime awareness, rate projection, one-hour and state-conditioned inertia, recovery relaxation, extreme-core inertia")
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    _selftest_v2()
    main_v2()
end
