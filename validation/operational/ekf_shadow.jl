# ekf_shadow.jl — retired shadow-EKF implementation retained for reproducibility.
#
# The constrained EKF holds the adapted decay coefficient at or below the
# current V2.1 discovered decay. Earlier V2.0 numerical findings motivated this
# retired shadow module; they are not treated as V2.1 performance evidence. The
# retired module can:
#   * bootstrap a persistent constrained-EKF state from the historical OMNI record (so the adapted decay
#     reflects long history, not just the short live tail),
#   * each monitor cycle, advance one constrained update with the latest locked-log anchor Dst*,
#   * emit a 1 h constrained-EKF v1+v2 forecast to a separate validation log,
#   * fall back to the locked v2 forecast (flagged) if state/inputs/output are unsafe.
# The locked v1/v2 record is never modified. This output is not part of the live product.
#
# Reads the current anchor + drivers from the latest locked-log row the monitor just wrote (no feed
# coupling, reuses the locked forecast's gap-checked inputs). Assumes SolarSINDy is loaded.

using SolarSINDy, CSV, DataFrames, Statistics, LinearAlgebra, Dates

include(joinpath(@__DIR__, "paths.jl"))

const _SHADOW_DIR   = OPERATIONAL_OUTPUT_DIR
const SHADOW_STATE  = joinpath(_SHADOW_DIR, "ekf_shadow_state.csv")
const SHADOW_LOG    = joinpath(_SHADOW_DIR, "ekf_shadow_log.csv")
const BOOT_YEAR0    = 2009                          # bootstrap window start (cycles 24-25 for current relevance)
const FALLBACK_NT   = 50.0                          # |shadow - locked| above this => distrust, fall back
# EKF hyperparameters — defined ONCE so the bootstrap filter and every reloaded filter are identical.
# Q_COEFF is the decay-coefficient random-walk variance and MUST match the validated QEKF=1e-4; a mismatch
# here (e.g. the init default 1e-6) silently slows the live adaptation by 100×.
const Q_COEFF=1e-4; const Q_DST=1.0; const R_OBS=4.0; const DST_VAR0=25.0; const COEFF_VAR0=1.0e-2

function _shadow_decay_cap(ξ0, i_decay)
    cap = Float64(ξ0[i_decay])
    isfinite(cap) && cap < 0.0 || error("current V2.1 decay coefficient is not stable")
    return cap
end

_new_filter(lib, ξ0, i_decay, dst0) =
    init_assimilation(lib, ξ0, [i_decay], dst0; q_coeff=Q_COEFF, q_dst=Q_DST, R=R_OBS,
                      dst_var0=DST_VAR0, coeff_var0=COEFF_VAR0,
                      coeff_bounds=[(-Inf, _shadow_decay_cap(ξ0, i_decay))])

# --- current V2.1 library + deployed coefficients ---
function _shadow_library()
    core = load_operational_core(OPERATIONAL_V2_1_MODEL_VERSION)
    i_decay = findfirst(==("Dst_star"), get_term_names(core.library))
    i_decay === nothing && error("Operational V2.1 core omits Dst_star")
    return core.library, copy(core.coefficients), i_decay
end

# --- bootstrap a constrained-EKF state over the historical OMNI record ---
function _bootstrap_state(lib, ξ0, i_decay)
    extracted = OPERATIONAL_OMNI
    # Persisted EKF shadow CSVs predate causal cleaning and are retired negative-result artifacts (not rerun);
    # causal=true keeps this bootstrap strictly causal if the retired script is ever re-executed.
    df = parse_omni2(extracted; year_start=BOOT_YEAR0, year_end=2025); clean_omni_data!(df; causal=true)
    fill!(v) = (l=NaN; for i in eachindex(v); isfinite(v[i]) ? (l=v[i]) : (isfinite(l)&&(v[i]=l)); end; v)
    V=fill!(Float64.(coalesce.(df.V,NaN))); Bz=fill!(Float64.(coalesce.(df.Bz,NaN)))
    By=fill!(Float64.(coalesce.(df.By,NaN))); nn=fill!(Float64.(coalesce.(df.n,NaN)))
    Pd=[dynamic_pressure(nn[k], V[k]) for k in eachindex(V)]
    dst=Float64.(coalesce.(df.Dst_star,NaN))
    f = _new_filter(lib, ξ0, i_decay, isfinite(dst[1]) ? dst[1] : 0.0)
    for k in 1:length(dst)-1
        assimilation_predict!(f, (V=V[k], Bz=Bz[k], By=By[k], n=nn[k], Pdyn=Pd[k]))
        assimilation_update!(f, dst[k+1])
    end
    # `last` = drivers AT the final observed hour (datetime[end]), to bridge to the next live anchor.
    last = (V=V[end], Bz=Bz[end], By=By[end], n=nn[end], Pdyn=Pd[end])
    return f, df.datetime[end], last
end

_state_cols() = [:last_obs_utc, :m1, :m2, :c11, :c12, :c21, :c22, :V, :Bz, :By, :n, :Pdyn]

function _save_state(state_time::DateTime, f::AssimilationFilter, last)
    row = DataFrame(last_obs_utc=[string(state_time)], m1=[f.mean[1]], m2=[f.mean[2]],
                    c11=[f.cov[1,1]], c12=[f.cov[1,2]], c21=[f.cov[2,1]], c22=[f.cov[2,2]],
                    V=[last.V], Bz=[last.Bz], By=[last.By], n=[last.n], Pdyn=[last.Pdyn])
    tmp = SHADOW_STATE * ".tmp"            # atomic write: a crash mid-write can't corrupt the live state
    CSV.write(tmp, row); mv(tmp, SHADOW_STATE; force=true)
end

function _load_state(lib, ξ0, i_decay)
    isfile(SHADOW_STATE) || return nothing
    r = CSV.read(SHADOW_STATE, DataFrame)[1, :]
    m2 = Float64(r.m2)
    isfinite(m2) && m2 <= _shadow_decay_cap(ξ0, i_decay) + 1e-9 || return nothing
    f = _new_filter(lib, ξ0, i_decay, Float64(r.m1))   # SAME hyperparameters as the bootstrap (esp. Q_COEFF)
    f.mean .= [Float64(r.m1), m2]
    f.cov  .= [Float64(r.c11) Float64(r.c12); Float64(r.c21) Float64(r.c22)]
    last = (V=Float64(r.V), Bz=Float64(r.Bz), By=Float64(r.By), n=Float64(r.n), Pdyn=Float64(r.Pdyn))
    # CSV.jl may auto-parse the timestamp as a DateTime or keep it a String; accept either.
    lt = r.last_obs_utc isa DateTime ? r.last_obs_utc : DateTime(first(split(string(r.last_obs_utc), ".")))
    return f, lt, last
end

# --- h-step forecast with the adapted decay, then the V2 correction ---
# `nsteps` defaults to 1 (the live 1 h path — byte-identical to the original single predict). nsteps>1
# free-runs the discovered ODE forward h hours holding the issue-time drivers constant, matching the
# locked v2 engine's multi-step rollout, so the only EKF-vs-v2 difference at any lead is the adapted decay.
function _shadow_forecast(lib, ξ0, i_decay, adapted_decay, anchor_dst_star, drv, latest_dst, cal;
                          nsteps::Int=1, calibration_features=nothing)
    ξ = copy(ξ0); ξ[i_decay] = adapted_decay
    fc = init_assimilation(lib, ξ, Int[], anchor_dst_star)        # no adaptation: pure rollout
    for _ in 1:nsteps
        assimilation_predict!(fc, drv)
        fc.mean[1] = clamp(fc.mean[1], -2000.0, 50.0)   # per-step Dst* ceiling — matches the engine's rollout
    end
    pred_dst_star = current_dst(fc)
    pred_dst = pred_dst_star + 7.26 * sqrt(max(drv.Pdyn, 0.0)) - 11.0   # _dst_from_dst_star
    fallback_features = _v2_features(
        latest_dst, drv; v1_pred_dst=pred_dst, model_steps=nsteps,
    )
    feature_source = calibration_features === nothing ? fallback_features : calibration_features
    available = propertynames(feature_source)
    missing_features = [c for c in cal.feature_names if !(c in available)]
    isempty(missing_features) || error(
        "shadow calibration feature source omits: $(join(String.(missing_features), ", "))",
    )
    feats = NamedTuple{Tuple(cal.feature_names)}(
        Tuple(Float64(getproperty(feature_source, c)) for c in cal.feature_names),
    )
    corr = SolarSINDy.operational_v2_correction(cal, feats)
    # Apply the SAME physical Dst ceiling the locked v2 engine uses (clamp to [-2000, 50] nT; see
    # forecast.jl operational_v2 corrected-center). Without it the shadow over-predicts positive Dst on
    # quiet sudden-commencement hours where v2 is clamped to +50 — a spurious shadow-vs-v2 gap. Storm-time
    # (deeply negative Dst) is unaffected: clamp(-300, -2000, 50) = -300.
    return clamp(pred_dst, -2000.0, 50.0), clamp(pred_dst + corr, -2000.0, 50.0)
end

# --- one cycle: read latest locked-log row, advance the filter, emit a shadow row (guarded) ---
function issue_ekf_shadow!(locked_log::AbstractString, calibration_csv::AbstractString)
    lib, ξ0, i_decay = _shadow_library()
    fallback = false; reason = ""
    log = CSV.read(locked_log, DataFrame)
    # Most recent cycle = latest anchor (latest_dst_time). Among rows for it, take the FRESHEST
    # re-issuance (max issue_time) for the anchor/drivers the EKF assimilates — a later cycle may carry a
    # revised Dst for the same hour, and using the stale one would mis-feed the filter. For the 1 h
    # locked-v2 comparison + the target, use that anchor's h=1 row (its true 1 h forecast) when present.
    latest_anchor = maximum(string.(log.latest_dst_time_utc))
    cyc = log[string.(log.latest_dst_time_utc) .== latest_anchor, :]
    fresh = cyc[argmax(string.(cyc.issue_time_utc)), :]
    h1 = cyc[cyc.model_step_hours .== 1, :]
    cmp = nrow(h1) > 0 ? h1[argmax(string.(h1.issue_time_utc)), :] : fresh
    issue_latest = fresh.issue_time_utc
    anchor = Float64(fresh.anchor_dst_star_nt); latest_dst = Float64(fresh.latest_dst_nt)
    drv = (V=Float64(fresh.V_kms), Bz=Float64(fresh.Bz_nt), By=Float64(fresh.By_nt),
           n=Float64(fresh.n_cm3), Pdyn=Float64(fresh.Pdyn_npa))
    obs_time = DateTime(first(split(string(fresh.latest_dst_time_utc), ".")))
    locked_v2 = ismissing(cmp.v2_pred_dst_nt) ? NaN : Float64(cmp.v2_pred_dst_nt)

    f, stime, last = recover_shadow_state(
        () -> _load_state(lib, ξ0, i_decay),
        () -> _bootstrap_state(lib, ξ0, i_decay),
    )
    # Advance to the new anchor when a newer observation has arrived. Coast-predict ONE step
    # per missing hour so the covariance accrues g·Q and the mean propagates g hours before
    # assimilating the g-hour-ahead observation — matching the strictly-hourly bootstrap loop.
    # A single predict for a multi-hour gap would under-inflate the covariance (Kalman gain too
    # small) and compare a 1 h-ahead mean against a several-hour-ahead anchor. The anchor Dst is
    # hourly, so a gap above a day signals a stale/discontinuous state → re-bootstrap instead.
    if obs_time > stime && isfinite(anchor)
        g = round(Int, Dates.value(obs_time - stime) / 3.6e6)   # whole-hour gap (ms → h)
        if g >= 1 && g <= 24
            for _ in 1:g
                assimilation_predict!(f, last)
            end
            assimilation_update!(f, anchor)
            stime = obs_time; last = drv
        else
            # Gap too large (e.g. a lost/torn state file while the bootstrap OMNI extract lags
            # the live clock): adopt the fresh bootstrap posterior but mark it CURRENT, so the
            # next cycle has g=1 and resumes hourly assimilation instead of re-bootstrapping
            # forever (which would pin the adapted decay at the bootstrap value).
            f, _, _ = _bootstrap_state(lib, ξ0, i_decay)
            assimilation_update!(f, anchor)
            stime = obs_time; last = drv
        end
    end
    _save_state(stime, f, last)
    decay = current_coeffs(f)[1]

    cal = read_operational_v2_calibration(calibration_csv)
    ekf_v1 = NaN; ekf_v2 = NaN
    try
        ekf_v1, ekf_v2 = _shadow_forecast(lib, ξ0, i_decay, decay, anchor, drv, latest_dst, cal)
    catch e
        fallback = true; reason = "forecast_error:" * sprint(showerror, e)
    end
    # guards: decay must stay in a stable band; shadow must not wildly disagree with locked v2
    decay_cap = _shadow_decay_cap(ξ0, i_decay)
    if !fallback && !(decay_cap - 2.0 <= decay <= decay_cap + 1e-9 && isfinite(ekf_v2))
        fallback = true; reason = "decay_or_pred_out_of_range(decay=$(round(decay,digits=4)))"
    end
    if !fallback && isfinite(locked_v2) && abs(ekf_v2 - locked_v2) > FALLBACK_NT
        fallback = true; reason = "shadow_far_from_locked(Δ=$(round(ekf_v2-locked_v2,digits=1)))"
    end
    out_v1 = fallback ? locked_v2 : ekf_v1
    out_v2 = fallback ? locked_v2 : ekf_v2

    # 1 h target: the locked h=1 row's target when present (exact join), else anchor + 1 h.
    target = nrow(h1) > 0 ? DateTime(first(split(string(cmp.target_time_utc), "."))) : obs_time + Hour(1)
    srow = DataFrame(issue_time_utc=[string(issue_latest)], latest_dst_time_utc=[string(obs_time)],
                     target_time_utc=[string(target)], anchor_dst_star_nt=[anchor], latest_dst_nt=[latest_dst],
                     adapted_decay=[decay], fixed_decay=[ξ0[i_decay]],
                     ekf_v1_pred_dst_nt=[out_v1], ekf_v2_pred_dst_nt=[out_v2],
                     locked_v2_pred_dst_nt=[locked_v2], fallback=[fallback], fallback_reason=[reason])
    out = if isfile(SHADOW_LOG)
        prev = CSV.read(SHADOW_LOG, DataFrame)
        # idempotent: one shadow row per cycle, keyed on the anchor (latest_dst_time, stable per cycle)
        prev = prev[string.(prev.latest_dst_time_utc) .!= string(obs_time), :]
        vcat(prev, srow; cols=:union)
    else
        srow
    end
    tmp = SHADOW_LOG * ".tmp"; CSV.write(tmp, out); mv(tmp, SHADOW_LOG; force=true)  # atomic write: reader never sees a torn file
    return (; decay, ekf_v2=out_v2, locked_v2, fallback, reason)
end

const SHADOW_REPORT = joinpath(_SHADOW_DIR, "ekf_shadow_report.md")
const STORM_DST = -50.0   # observed Dst below this counts as a storm row (matches the monitor convention)

# Score the accumulated shadow 1 h forecasts against realized observations and compare to served V2.1.
# The realized observation comes from the locked log's verified h=1 row for the same target time (the
# shadow never re-fetches Dst). Non-fallback rows only. Writes a compact report and returns a summary.
function score_ekf_shadow!(locked_log::AbstractString;
                           shadow_log::AbstractString=SHADOW_LOG, report::AbstractString=SHADOW_REPORT)
    isfile(shadow_log) || return (; scored=0, msg="no shadow log yet")
    sh = CSV.read(shadow_log, DataFrame)
    lk = CSV.read(locked_log, DataFrame)
    _norm(t) = replace(first(split(string(t), ".")), "Z" => "")
    # Current h=1 verified rows: normalized target -> (observation, served V2.1 prediction).
    obsmap = Dict{String,Tuple{Float64,Float64}}()
    h1 = lk[(lk.model_step_hours .== 1) .& (.!ismissing.(lk.observation_dst_nt)), :]
    served_col = "served_pred_dst_nt" in names(lk) ? :served_pred_dst_nt : :v2_pred_dst_nt
    for r in eachrow(h1)
        o = r.observation_dst_nt; v = r[served_col]
        (ismissing(o) || !isfinite(Float64(o))) && continue
        obsmap[_norm(r.target_time_utc)] = (Float64(o), ismissing(v) ? NaN : Float64(v))
    end
    ek = Float64[]; lo = Float64[]; isstorm = Bool[]; nfb = 0
    for r in eachrow(sh)
        (("fallback" in names(sh)) && r.fallback === true) && (nfb += 1; continue)
        k = _norm(r.target_time_utc); haskey(obsmap, k) || continue
        obs, lockv2 = obsmap[k]; isfinite(lockv2) || continue
        e = r.ekf_v2_pred_dst_nt; (ismissing(e) || !isfinite(Float64(e))) && continue
        push!(ek, obs - Float64(e)); push!(lo, obs - lockv2); push!(isstorm, obs < STORM_DST)
    end
    n = length(ek)
    rmse(v) = isempty(v) ? NaN : sqrt(sum(abs2, v) / length(v))
    re, rl = rmse(ek), rmse(lo)
    si = findall(isstorm); rse, rsl = rmse(ek[si]), rmse(lo[si])
    decay_now = ("adapted_decay" in names(sh)) && nrow(sh) > 0 ? Float64(sh.adapted_decay[end]) : NaN
    _, ξ0, i_decay = _shadow_library()
    fixed_decay = _shadow_decay_cap(ξ0, i_decay)

    open(report, "w") do io
        println(io, "# Constrained-EKF Shadow vs Served V2.1 — live 1 h comparison\n")
        println(io, "Experimental shadow series (separate log; the locked v1/v2 record is untouched). ",
                    "Lower 1 h RMSE is better; the EKF's validated edge is at storm hours.\n")
        println(io, "- scored (verified, non-fallback) rows: **$n**   (storm rows obs<$(STORM_DST) nT: **$(length(si))**)   fallback rows skipped: $nfb")
        println(io, "- latest adapted decay: $(isnan(decay_now) ? "n/a" : round(decay_now, digits=4))  (current fixed $(round(fixed_decay, digits=4)))\n")
        if n == 0
            println(io, "No verified shadow rows yet — wait for shadow targets to be observed. ",
                        "During quiet conditions the EKF ≈ locked, so a meaningful gap needs storm hours.")
        else
            println(io, "| set | n | EKF 1 h RMSE [nT] | served V2.1 1 h RMSE [nT] | EKF − V2.1 |")
            println(io, "|---|---|---|---|---|")
            println(io, "| all | $n | $(round(re,digits=2)) | $(round(rl,digits=2)) | $(round(re-rl,digits=2)) |")
            length(si) > 0 && println(io, "| storm (obs<$(STORM_DST)) | $(length(si)) | $(round(rse,digits=2)) | $(round(rsl,digits=2)) | $(round(rse-rsl,digits=2)) |")
            println(io, "\nNegative \"EKF − V2.1\" means the shadow EKF beats served V2.1 at 1 h. ",
                        n < 30 ? "n is small — treat as indicative until more rows (especially storm rows) accrue." :
                                 "")
        end
    end
    return (; scored=n, storm=length(si), ekf_rmse=re, locked_rmse=rl,
              ekf_storm_rmse=rse, locked_storm_rmse=rsl, fallback=nfb)
end
