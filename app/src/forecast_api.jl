# forecast_api.jl — data + threat layer for the Space-Weather Threat Monitor.
#
# Reads the locked-live forecast log produced by the SolarSINDy operational daemon and
# turns it into honest JSON payloads. Design principles:
#   * The locked log is the single source of truth. This layer never re-computes a
#     forecast; it serves exactly what was issued and (later) verified.
#   * Threat is assessed from the *calibrated lower bound* of the predictive interval,
#     not only the point forecast, so the worst credible storm within the 90% band drives
#     the alert.
#   * Calibration (coverage, RMSE) is recomputed from the log itself, so the dashboard
#     cannot drift from a stale report file.
#
# No new heavy dependencies: CSV, DataFrames, Dates, Statistics are already in the stack.

using CSV, DataFrames, Dates, Statistics, JSON3

# ---- Dst storm-intensity threat scale --------------------------------------------------
# Verified against the geomagnetic-storm literature: the widely adopted Dst classification
# uses primary division points -50 / -100 / -200 nT, with an extended minor tier
# (-30..-50 nT). Sources recorded in app/README.md. Dst is in nT; more negative = stronger.
const THREAT_LABELS = ("Quiet", "Minor storm", "Moderate storm", "Intense storm", "Extreme storm")
const THREAT_BANDS_NT = (-30.0, -50.0, -100.0, -200.0)   # upper edge of levels 1..4

"""Return (level::Int 0..4, label::String) for a Dst value in nT."""
function dst_threat_level(dst::Real)
    dst > THREAT_BANDS_NT[1] && return (0, THREAT_LABELS[1])
    dst > THREAT_BANDS_NT[2] && return (1, THREAT_LABELS[2])
    dst > THREAT_BANDS_NT[3] && return (2, THREAT_LABELS[3])
    dst > THREAT_BANDS_NT[4] && return (3, THREAT_LABELS[4])
    return (4, THREAT_LABELS[5])
end

# ---- small helpers ---------------------------------------------------------------------
# JSON-friendly number: finite Float64 or nothing (JSON null). Guards missing/NaN/Inf.
jnum(x) = (x === missing || x === nothing) ? nothing : (isa(x, Real) && isfinite(x) ? Float64(x) : nothing)

# Tolerant ISO8601 -> DateTime (handles 0..n fractional-second digits); missing-safe.
function parse_dt(s)
    (s === missing || s === nothing) && return missing
    str = strip(String(s))
    isempty(str) && return missing
    try
        return DateTime(str)                      # fast path (<=3 fractional digits)
    catch
        m = match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", str)
        m === nothing && return missing
        dt = tryparse(DateTime, m.captures[1])    # whole-second fallback; tryparse → nothing on a
        return dt === nothing ? missing : dt      # format-valid but invalid date (e.g. month 13), never throws
    end
end

# DateTime -> ISO UTC string with explicit Z, or nothing.
jdt(dt) = (dt === missing || dt === nothing) ? nothing : string(dt) * "Z"

# ---- log loading (mtime-cached, append-safe) -------------------------------------------
const _TIME_COLS = ["issue_time_utc", "latest_solar_wind_utc", "latest_dst_time_utc", "target_time_utc"]
const _LOG_CACHE = Ref{Any}(nothing)              # (mtime::Float64, df::DataFrame)
const _LOG_LOCK = ReentrantLock()

function _load_log(path::AbstractString)
    # Read time columns as String and parse ourselves (robust to fractional-second variation).
    df = CSV.read(path, DataFrame;
                  types=Dict(c => String for c in _TIME_COLS),
                  missingstring=["", "NaN", "missing"], silencewarnings=true)
    for c in _TIME_COLS
        if hasproperty(df, Symbol(c))
            df[!, Symbol(c * "_dt")] = parse_dt.(df[!, Symbol(c)])
        end
    end
    return df
end

"""Return the current forecast log as a DataFrame, reloading only when the file changes.
On a read failure mid-append, serve the last good frame (raising only if none exists)."""
function get_log(path::AbstractString)
    lock(_LOG_LOCK) do
        isfile(path) || error("forecast log not found: $path")
        m = mtime(path)
        c = _LOG_CACHE[]
        if c === nothing || c[1] != m
            try
                _LOG_CACHE[] = (m, _load_log(path))
            catch e
                c === nothing && rethrow(e)        # no fallback available
                @warn "log reload failed; serving cached copy" exception=(e, catch_backtrace())
            end
        end
        return _LOG_CACHE[][2]
    end
end

# Pick forecast columns with graceful fallbacks. hasproperty is checked BEFORE
# access so older CSV schemas fall back instead of throwing "column not found".
function _col(row, a::Symbol, b::Symbol)
    hasproperty(row, a) && getproperty(row, a) !== missing && return getproperty(row, a)
    hasproperty(row, b) ? getproperty(row, b) : missing
end
_audit_pred(row) = _col(row, :v2_pred_dst_nt,      :pred_dst_nt)
_audit_ci05(row) = _col(row, :v2_pred_dst_ci05_nt, :pred_dst_ci05_nt)
_audit_ci95(row) = _col(row, :v2_pred_dst_ci95_nt, :pred_dst_ci95_nt)

# Upgraded V2 is the product forecast. The live CSV keeps older product-column
# names, with fallback to older v2_/pred_* columns for legacy rows.
_v2_pred(row) = (x = _col(row, :served_pred_dst_nt,      :v2_pred_dst_nt);      x === missing ? _audit_pred(row) : x)
_v2_ci05(row) = (x = _col(row, :served_pred_dst_ci05_nt, :v2_pred_dst_ci05_nt); x === missing ? _audit_ci05(row) : x)
_v2_ci95(row) = (x = _col(row, :served_pred_dst_ci95_nt, :v2_pred_dst_ci95_nt); x === missing ? _audit_ci95(row) : x)

# Rows of the most recent forecast cycle = those sharing the freshest input-data vintage
# (latest_solar_wind_utc). Returns a sub-DataFrame sorted by target time.
function latest_cycle(df::DataFrame)
    nrow(df) == 0 && return df
    sw = df.latest_solar_wind_utc_dt
    valid = .!ismissing.(sw)
    if !any(valid)
        # No usable solar-wind timestamp; fall back to the newest issue time — but if those are
        # all missing too, `maximum(skipmissing(...))` would throw on an empty collection, so
        # return an empty sub-frame (downstream build_* already handle the no-cycle case).
        isempty(collect(skipmissing(df.issue_time_utc_dt))) && return df[1:0, :]
        cyc0 = df[coalesce.(df.issue_time_utc_dt .== maximum(skipmissing(df.issue_time_utc_dt)), false), :]
        sort!(cyc0, :target_time_utc_dt)   # match the normal branch so horizons stay chronological
        return cyc0
    end
    newest = maximum(skipmissing(sw))
    cyc = df[coalesce.(sw .== newest, false), :]
    sort!(cyc, :target_time_utc_dt)
    return cyc
end

# Verified rows: a real forecast (target strictly AFTER both the issue time and the last observed Dst — a
# 0-lead row is not a forecast and is what the locked comparison report excludes), with a FINITE observation
# and a finite V2 point + band. jnum(obs) rejects NaN/Inf observations that would otherwise reach the
# calibration summary and make JSON serialization throw.
function verified_rows(df::DataFrame)
    nrow(df) == 0 && return df
    has_issue  = hasproperty(df, :issue_time_utc_dt)
    has_anchor = hasproperty(df, :latest_dst_time_utc_dt)
    keep = trues(nrow(df))
    for (i, r) in enumerate(eachrow(df))
        tt = r.target_time_utc_dt
        valid_lead = tt !== missing &&
                     (!has_issue  || r.issue_time_utc_dt      === missing || tt > r.issue_time_utc_dt) &&
                     (!has_anchor || r.latest_dst_time_utc_dt === missing || tt > r.latest_dst_time_utc_dt)
        keep[i] = valid_lead && jnum(r.observation_dst_nt) !== nothing && jnum(_v2_pred(r)) !== nothing &&
                  jnum(_v2_ci05(r)) !== nothing && jnum(_v2_ci95(r)) !== nothing
    end
    return df[keep, :]
end

# ---- calibration summary (computed from the log, not a static report) ------------------
# Honesty note: the *live* interval method (ACI) and the *verified* track record can differ.
# ACI was deployed after most verified rows were issued, so coverage is reported overall AND
# broken down by interval method, and the live method is reported separately. We never imply
# the historical coverage was achieved by a method that has no verified rows yet.
_src_label(s) = (s === missing || s === nothing || s == "") ? "legacy" : String(s)

function calibration_summary(df::DataFrame)
    # Live interval method = source of the most recent issued forecast cycle.
    cyc = latest_cycle(df)
    live_src = (nrow(cyc) > 0 && "interval_source" in names(cyc)) ? _src_label(first(cyc).interval_source) : "unknown"

    v = verified_rows(df)
    n = nrow(v)
    n == 0 && return (n_verified=0, coverage_90=nothing, rmse_nt=nothing,
                      v2_n_verified=0, v2_coverage_90=nothing, v2_rmse_nt=nothing,
                      audit_baseline_rmse_nt=nothing,
                      rmse_persistence_nt=nothing, rmse_obrien_nt=nothing,
                      current_interval_source=live_src, n_verified_current_source=0,
                      deepest_obs_dst_nt=nothing, n_storm_verified=0, by_source=[])
    obs   = Float64.(v.observation_dst_nt)
    pred  = Float64.(_v2_pred.(eachrow(v)))
    ci05  = Float64.(_v2_ci05.(eachrow(v)))
    ci95  = Float64.(_v2_ci95.(eachrow(v)))
    audit_pred = Float64.(_audit_pred.(eachrow(v)))
    inside = (obs .>= ci05) .& (obs .<= ci95)
    rmse(p) = sqrt(mean((obs .- p).^2))
    product_col_mask = falses(n)
    if all(c -> c in names(v), ("served_pred_dst_nt", "served_pred_dst_ci05_nt", "served_pred_dst_ci95_nt"))
        product_col_mask .= [jnum(v[i, :served_pred_dst_nt]) !== nothing &&
                             jnum(v[i, :served_pred_dst_ci05_nt]) !== nothing &&
                             jnum(v[i, :served_pred_dst_ci95_nt]) !== nothing for i in 1:n]
    end
    product_col_n = count(product_col_mask)
    product_col_cov = nothing
    product_col_rmse = nothing
    audit_rmse = jnum(round(rmse(audit_pred); digits=2))
    if product_col_n > 0
        spred = Float64.(v[product_col_mask, :served_pred_dst_nt])
        slo = Float64.(v[product_col_mask, :served_pred_dst_ci05_nt])
        shi = Float64.(v[product_col_mask, :served_pred_dst_ci95_nt])
        sobs = obs[product_col_mask]
        sinside = (sobs .>= slo) .& (sobs .<= shi)
        product_col_cov = jnum(round(mean(sinside); digits=3))
        product_col_rmse = jnum(round(sqrt(mean((sobs .- spred).^2)); digits=2))
        audit_rmse = jnum(round(sqrt(mean((sobs .- audit_pred[product_col_mask]).^2)); digits=2))
    end
    product_mask = product_col_n > 0 ? product_col_mask : trues(n)
    product_n = count(product_mask)
    product_cov = product_col_n > 0 ? product_col_cov : jnum(round(mean(inside); digits=3))
    product_rmse = product_col_n > 0 ? product_col_rmse : jnum(round(rmse(pred); digits=2))
    pers = "persistence_dst_nt" in names(v) ? collect(v.persistence_dst_nt) : fill(missing, n)
    obri = "obrien_dst_nt" in names(v) ? collect(v.obrien_dst_nt) : fill(missing, n)
    rmse_opt(col, mask=trues(n)) = begin
        m = mask .& (.!ismissing.(col)) .& isfinite.(coalesce.(col, NaN))
        any(m) ? sqrt(mean((obs[m] .- Float64.(col[m])).^2)) : nothing
    end

    # per-method coverage breakdown
    srcs = "interval_source" in names(v) ? _src_label.(v.interval_source) : fill("unknown", n)
    by_source = NamedTuple[]
    for s in sort(unique(srcs))
        m = srcs .== s
        push!(by_source, (source=s, n=count(m), coverage_90=round(mean(inside[m]); digits=3)))
    end
    n_live = count(srcs .== live_src)

    return (n_verified=product_n,
            coverage_90=product_cov,
            rmse_nt=product_rmse,
            v2_n_verified=product_n,
            v2_coverage_90=product_cov,
            v2_rmse_nt=product_rmse,
            audit_baseline_rmse_nt=audit_rmse,
            rmse_persistence_nt=(x = rmse_opt(pers, product_mask); x === nothing ? nothing : round(x; digits=2)),
            rmse_obrien_nt=(x = rmse_opt(obri, product_mask); x === nothing ? nothing : round(x; digits=2)),
            current_interval_source=live_src,
            n_verified_current_source=n_live,
            deepest_obs_dst_nt=jnum(round(minimum(obs); digits=1)),
            n_storm_verified=count(obs .< -50),
            by_source=by_source)
end

# ---- payload builders ------------------------------------------------------------------
# Recent observed Dst straight from the log's latest_dst columns (every row records the freshest Kyoto Dst at
# its issue). This runs to the forecast anchor with no extra network dependency, so the observed line is
# continuous up to where the forecast begins (the verified-row track lags by the verification delay). Later
# rows overwrite earlier ones at the same timestamp, so revised Dst values win.
function _recent_observed(df::DataFrame; hours::Real=48)
    nrow(df) == 0 && return NamedTuple[]
    (hasproperty(df, :latest_dst_time_utc_dt) && hasproperty(df, :latest_dst_nt)) || return NamedTuple[]
    cutoff = now(UTC) - Hour(round(Int, hours))
    seen = Dict{DateTime,Float64}()
    for r in eachrow(df)
        t = r.latest_dst_time_utc_dt; v = jnum(r.latest_dst_nt)
        (t === missing || v === nothing || t < cutoff) && continue
        seen[t] = v
    end
    out = [(target_utc=jdt(t), observed_dst_nt=v) for (t, v) in seen]
    sort!(out, by=x -> x.target_utc)
    return out
end

# Sub-hour MODEL trajectory (display only) written by the engine next to the log: V2 integrated
# at a sub-hour step. Not a validated sub-hour forecast (Dst is observed hourly; the ODE is hourly-fit) — it is the
# hourly model's own interpolation. Defensive: missing/unreadable/stale → [].
function _subhour_traj(log_path::AbstractString)
    isempty(log_path) && return NamedTuple[]
    f = joinpath(dirname(log_path), "subhour_trajectory.json")
    isfile(f) || return NamedTuple[]
    try
        d = JSON3.read(read(f, String))
        return [(target_utc = String(p.t) * "Z", dst_nt = jnum(p.dst)) for p in d.points]
    catch
        return NamedTuple[]
    end
end

"""Forecast trajectory of the most recent cycle: anchor + per-horizon point and 90% band."""
function build_forecast(df::DataFrame, log_path::AbstractString="")
    cyc = latest_cycle(df)
    nrow(cyc) == 0 && return (issue_time_utc=nothing, horizons=[])
    horizons = NamedTuple[]
    for r in eachrow(cyc)
        push!(horizons, (target_utc=jdt(r.target_time_utc_dt),
                         horizon_hours=jnum(r.horizon_hours),
                         pred_dst_nt=jnum(_v2_pred(r)),
                         ci05_dst_nt=jnum(_v2_ci05(r)),
                         ci95_dst_nt=jnum(_v2_ci95(r)),
                         audit_baseline_dst_nt=jnum(_audit_pred(r))))
    end
    r1 = first(cyc)
    return (issue_time_utc=jdt(r1.issue_time_utc_dt),
            latest_solar_wind_utc=jdt(r1.latest_solar_wind_utc_dt),
            anchor_dst_nt=jnum(r1.latest_dst_nt),
            anchor_dst_time_utc=jdt(r1.latest_dst_time_utc_dt),
            interval_source=("interval_source" in names(cyc) ? String(coalesce(r1.interval_source, "unknown")) : "unknown"),
            model_version=("model_version" in names(cyc) ? String(coalesce(r1.model_version, "v2")) : "v2"),
            recent_observed=_recent_observed(df),
            subhour_trajectory=_subhour_traj(log_path),
            horizons=horizons)
end

"""Storm-time replay summary: serves the offline replay report and its scored-row provenance.
Defensive (never throws): missing/unreadable artifacts return available=false."""
function build_storm_replay(log_path::AbstractString)
    dir = dirname(log_path)
    report = joinpath(dir, "storm_replay_report.md")
    scored = joinpath(dir, "storm_replay_scored.csv")
    isfile(report) || return (available=false,
                              reason="no storm-replay report; run live_forecasts/storm_replay.jl")
    md = try read(report, String) catch; return (available=false, reason="report unreadable") end
    age = round((time() - mtime(report)) / 60; digits=1)
    n = 0; storms = String[]
    if isfile(scored)
        try
            df = CSV.read(scored, DataFrame)
            n = nrow(df)
            "storm" in names(df) && (storms = unique(string.(skipmissing(df.storm))))
        catch
        end
    end
    return (available=true, report_age_min=age, n_scored=n, storms=storms, report_markdown=md)
end

"""Threat status: current observation + worst credible storm over the latest cycle."""
function build_status(df::DataFrame)
    cyc = latest_cycle(df)
    cal = calibration_summary(df)
    if nrow(cyc) == 0
        return (generated_utc=jdt(now(UTC)), available=false,
                message="No forecast rows in log yet.", calibration=cal)
    end
    r1 = first(cyc)
    preds = filter(!isnothing, [jnum(_v2_pred(r)) for r in eachrow(cyc)])
    lbs   = filter(!isnothing, [jnum(_v2_ci05(r)) for r in eachrow(cyc)])
    point_min = isempty(preds) ? nothing : minimum(preds)
    worst_cred = isempty(lbs)  ? nothing : minimum(lbs)
    lvl_pt, lbl_pt = point_min === nothing ? (0, THREAT_LABELS[1]) : dst_threat_level(point_min)
    lvl_wc, lbl_wc = worst_cred === nothing ? (0, THREAT_LABELS[1]) : dst_threat_level(worst_cred)
    # Reported threat level is the point-forecast level; a "watch" flag fires when the
    # calibrated lower bound reaches a stronger storm tier than the point forecast.
    watch = lvl_wc > lvl_pt
    horizon_max = (h = filter(!isnothing, jnum.(eachrow(cyc) .|> r -> r.horizon_hours)); isempty(h) ? nothing : maximum(h))
    return (generated_utc=jdt(now(UTC)),
            available=true,
            latest_observation=(dst_nt=jnum(r1.latest_dst_nt), time_utc=jdt(r1.latest_dst_time_utc_dt)),
            latest_solar_wind_utc=jdt(r1.latest_solar_wind_utc_dt),
            forecast_issue_utc=jdt(r1.issue_time_utc_dt),
            threat=(level=lvl_pt, label=lbl_pt, watch=watch,
                    watch_level=lvl_wc, watch_label=lbl_wc,
                    point_min_dst_nt=point_min, worst_credible_dst_nt=worst_cred,
                    basis="Dst storm-intensity scale (-30/-50/-100/-200 nT)"),
            lead_time=(forecast_horizon_hours=horizon_max,
                       driver_assumption="L1 measured look-ahead, then regime-aware relaxation beyond the L1-known window, with a near-term extreme-Dst inertia guard",
                       physical_upstream_lead_min=[30, 60],
                       note="Genuine upstream lead for new severity is the L1 advection time (~30-60 min). " *
                            "Multi-day lead requires CME eruption/propagation models, not yet in this system."),
            calibration=cal,
            model_version=("model_version" in names(cyc) ? String(coalesce(r1.model_version, "v2")) : "v2"))
end

"""Recent verified track record (observed vs predicted with band) for the last `hours`."""
function build_history(df::DataFrame, hours::Real=72)
    v = verified_rows(df)
    nrow(v) == 0 && return (rows=[], coverage_90=nothing, rmse_nt=nothing, hours=hours)
    cutoff = now(UTC) - Hour(round(Int, hours))
    rows = NamedTuple[]
    for r in eachrow(v)
        t = r.target_time_utc_dt
        (t === missing || t < cutoff) && continue
        obs = jnum(r.observation_dst_nt); p = jnum(_v2_pred(r))
        lo = jnum(_v2_ci05(r)); hi = jnum(_v2_ci95(r))
        inside = (obs !== nothing && lo !== nothing && hi !== nothing) ? (obs >= lo && obs <= hi) : nothing
        push!(rows, (target_utc=jdt(t), horizon_hours=jnum(r.horizon_hours),
                     observed_dst_nt=obs, pred_dst_nt=p, ci05_dst_nt=lo, ci95_dst_nt=hi,
                     audit_baseline_dst_nt=jnum(_audit_pred(r)),
                     inside_90ci=inside))
    end
    sort!(rows, by=x -> something(x.target_utc, ""))
    # Stats must match the WINDOW the rows show, not the whole log. Compute coverage/RMSE over
    # the windowed rows; also expose the (more robust) all-log figures under explicit names.
    flags = [r.inside_90ci for r in rows if r.inside_90ci !== nothing]
    sq = [(r.observed_dst_nt - r.pred_dst_nt)^2 for r in rows
          if r.observed_dst_nt !== nothing && r.pred_dst_nt !== nothing]
    win_cov = isempty(flags) ? nothing : round(count(flags) / length(flags); digits=3)
    win_rmse = isempty(sq) ? nothing : round(sqrt(sum(sq) / length(sq)); digits=2)
    cal = calibration_summary(df)
    return (rows=rows, hours=hours, n=length(rows),
            coverage_90=win_cov, rmse_nt=win_rmse,
            coverage_90_all=something(cal.v2_coverage_90, cal.coverage_90),
            rmse_nt_all=something(cal.v2_rmse_nt, cal.rmse_nt))
end

"""Active alert summary derived from the current threat status."""
function build_alerts(df::DataFrame)
    st = build_status(df)
    getproperty(st, :available) == false && return (active=false, alerts=[], generated_utc=st.generated_utc)
    th = st.threat
    alerts = NamedTuple[]
    if th.level >= 1
        push!(alerts, (severity=th.label, level=th.level, kind="forecast",
                       message="Forecast Dst reaches $(round(th.point_min_dst_nt; digits=0)) nT " *
                               "($(th.label)) within the next $(round(something(st.lead_time.forecast_horizon_hours, 0.0); digits=1)) h."))
    end
    if th.watch && th.watch_level > th.level
        push!(alerts, (severity=th.watch_label, level=th.watch_level, kind="watch",
                       message="Calibrated 90% lower bound reaches $(round(th.worst_credible_dst_nt; digits=0)) nT " *
                               "($(th.watch_label)); storm cannot be excluded at the 90% level."))
    end
    return (active=!isempty(alerts), generated_utc=st.generated_utc,
            threat_level=th.level, threat_label=th.label, alerts=alerts)
end
