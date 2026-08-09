# forecast_api.jl — data + threat layer for the Space-Weather Threat Monitor.
#
# Reads the locked-live forecast log produced by the SolarSINDy operational daemon and
# turns it into honest JSON payloads. Design principles:
#   * The locked log is the single source of truth. This layer never re-computes a
#     forecast; it serves exactly what was issued and (later) verified.
#   * A watch is assessed from the most negative lower edge among the displayed 90%-target
#     predictive intervals, not only from the point forecast. The watch is a conservative
#     screen, not a one-sided confidence statement or a storm probability.
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
const CURRENT_V2_MODEL_VERSION = "v2.1"
const CURRENT_V2_SERVED_MODEL_VERSION =
    "v2.1+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"
const LIVE_SKILL_MIN_VERIFIED = 48

"""Return `(level, label)` for a finite Dst value in nT; non-finite input is unknown."""
function dst_threat_level(dst::Real)
    isfinite(dst) || return (nothing, "Unknown")
    dst > THREAT_BANDS_NT[1] && return (0, THREAT_LABELS[1])
    dst > THREAT_BANDS_NT[2] && return (1, THREAT_LABELS[2])
    dst > THREAT_BANDS_NT[3] && return (2, THREAT_LABELS[3])
    dst > THREAT_BANDS_NT[4] && return (3, THREAT_LABELS[4])
    return (4, THREAT_LABELS[5])
end

# ---- small helpers ---------------------------------------------------------------------
# JSON-friendly number: finite Float64 or nothing (JSON null). Guards missing/NaN/Inf,
# boolean-as-number coercion, and finite wide values that overflow during Float64 conversion.
function jnum(x)
    (x === missing || x === nothing || !(x isa Real) || x isa Bool) && return nothing
    value = try
        Float64(x)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    return isfinite(value) ? value : nothing
end

# Tolerant ISO8601 -> DateTime (handles 0..n fractional-second digits); missing-safe.
function parse_dt(s)
    (s === missing || s === nothing) && return missing
    str = strip(String(s))
    isempty(str) && return missing
    try
        return DateTime(str)                      # fast path (<=3 fractional digits)
    catch e
        e isa InterruptException && rethrow()
        m = match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", str)
        m === nothing && return missing
        dt = tryparse(DateTime, m.captures[1])    # whole-second fallback; tryparse → nothing on a
        return dt === nothing ? missing : dt      # format-valid but invalid date (e.g. month 13), never throws
    end
end

# DateTime -> ISO UTC string with explicit Z, or nothing.
jdt(dt) = (dt === missing || dt === nothing) ? nothing : string(dt) * "Z"

# ---- log loading (identity-cached, append-safe) ----------------------------------------
const _TIME_COLS = ["issue_time_utc", "latest_solar_wind_utc", "latest_dst_time_utc", "target_time_utc"]
const _LOG_CACHE = Ref{Any}(nothing)
const _LOG_LOCK = ReentrantLock()
const _CALIBRATION_CACHE = Ref{Any}(nothing)
const _CALIBRATION_LOCK = ReentrantLock()
const _LATEST_CYCLE_CACHE = Ref{Any}(nothing)
const _LATEST_CYCLE_LOCK = ReentrantLock()
const _VERIFIED_ROWS_CACHE = Ref{Any}(nothing)
const _VERIFIED_ROWS_LOCK = ReentrantLock()
const _HISTORY_CACHE = Ref{Any}(nothing)
const _HISTORY_LOCK = ReentrantLock()

function _log_file_identity(path::AbstractString)
    resolved = realpath(path)
    info = stat(path)
    return (resolved, info.device, info.inode, info.size, info.mtime, info.ctime)
end

function _cached_log_key(df::DataFrame)
    return lock(_LOG_LOCK) do
        cached = _LOG_CACHE[]
        cached !== nothing && cached[2] === df ? (cached[1], objectid(df)) : nothing
    end
end

function _canonical_missing_log_path(path::AbstractString)
    absolute = abspath(normpath(path))
    parent = dirname(absolute)
    return isdir(parent) ? joinpath(realpath(parent), basename(absolute)) : absolute
end

function _load_log(path::AbstractString)
    # Read time columns as String and parse ourselves (robust to fractional-second variation).
    df = CSV.read(path, DataFrame;
                  types=Dict(c => String for c in _TIME_COLS),
                  validate=false, missingstring=["", "NaN", "missing"], silencewarnings=true)
    for c in _TIME_COLS
        if hasproperty(df, Symbol(c))
            df[!, Symbol(c * "_dt")] = parse_dt.(df[!, Symbol(c)])
        end
    end
    return df
end

"""Return the current forecast log as a DataFrame, reloading only when the file changes.
When the file is absent, serve the last good frame if cached, else an empty frame so the
log-backed endpoints degrade to `available=false` (never a 500). On a read failure mid-append,
serve the last good frame (raising only if none exists)."""
function get_log(path::AbstractString)
    lock(_LOG_LOCK) do
        c = _LOG_CACHE[]
        if !isfile(path)
            # Absent (fresh install, before the daemon's first write, or log rotation): fall back
            # to the cached frame, or an empty frame — do not throw, so /api/status et al. return
            # available=false rather than propagating a 500 that also leaks the log path.
            wanted = _canonical_missing_log_path(path)
            return c === nothing || c[1][1] != wanted ? DataFrame() : c[2]
        end
        # Resolve the file identity defensively: the file can be unlinked (log rotation/rewrite)
        # between the isfile check above and here, in which case realpath/stat throw. Degrade to
        # the absent-file branch instead of surfacing a 500 (and leaking the log path).
        key = try
            _log_file_identity(path)
        catch e
            e isa InterruptException && rethrow()
            wanted = _canonical_missing_log_path(path)
            return c === nothing || c[1][1] != wanted ? DataFrame() : c[2]
        end
        if c === nothing || c[1] != key
            try
                loaded = _load_log(path)
                key == _log_file_identity(path) || error(
                    "forecast log changed while it was being loaded",
                )
                _LOG_CACHE[] = (key, loaded)
            catch e
                e isa InterruptException && rethrow()
                # If the file vanished mid-load, degrade like the absent branch rather than 500.
                if !isfile(path)
                    wanted = _canonical_missing_log_path(path)
                    return c === nothing || c[1][1] != wanted ? DataFrame() : c[2]
                end
                (c === nothing || c[1][1] != key[1]) && rethrow(e)
                @warn "log reload failed; serving cached copy" exception=(e, catch_backtrace())
                return c[2]
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
_rowget(row, name::Symbol) = hasproperty(row, name) ? getproperty(row, name) : missing

function _row_is_current_v2(row)
    model = _rowget(row, :model_version)
    served_model = _rowget(row, :sub_hourly_model_version)
    return model isa AbstractString && served_model isa AbstractString &&
           model == CURRENT_V2_MODEL_VERSION &&
           served_model == CURRENT_V2_SERVED_MODEL_VERSION
end

# Rows of the most recent forecast cycle, defined by ISSUE epoch (not driver vintage). When the
# L1 feed stalls across issue boundaries, several hourly cycles share one latest_solar_wind_utc;
# keying on that vintage would merge superseded issues into one payload (conflicting horizons per
# target, a stale issue/anchor, and threat minima computed over superseded forecasts). Per-row
# issue timestamps within one cycle may differ by seconds, so the product identity is the UTC issue
# hour and `_valid_live_cycle` separately bounds the within-cycle spread. This prevents adjacent-hour
# cycles from mixing while accepting the sequential-per-horizon writer and a short retry. Returns a
# sub-DataFrame sorted by target time.
const _CYCLE_TOL = Minute(5)   # sidecar matching tolerance; live-cycle grouping uses issue hour

_sort_by_target!(cyc) = (hasproperty(cyc, :target_time_utc_dt) && sort!(cyc, :target_time_utc_dt); cyc)

function _latest_cycle_uncached(df::DataFrame)
    nrow(df) == 0 && return df
    if hasproperty(df, :issue_time_utc_dt)
        iss = df.issue_time_utc_dt
        if !isempty(collect(skipmissing(iss)))
            tmax = maximum(skipmissing(iss))
            issue_hour = floor(tmax, Hour)
            keep = map(iss) do issue
                issue !== missing && floor(issue, Hour) == issue_hour
            end
            return _sort_by_target!(df[keep, :])
        end
    end
    # No usable issue time (legacy schema): fall back to the freshest solar-wind vintage.
    if hasproperty(df, :latest_solar_wind_utc_dt)
        sw = df.latest_solar_wind_utc_dt
        if !isempty(collect(skipmissing(sw)))
            newest = maximum(skipmissing(sw))
            return _sort_by_target!(df[coalesce.(sw .== newest, false), :])
        end
    end
    return df[1:0, :]   # nothing usable; downstream build_* handle the no-cycle case
end


function latest_cycle(df::DataFrame)
    key = _cached_log_key(df)
    key === nothing && return _latest_cycle_uncached(df)
    return lock(_LATEST_CYCLE_LOCK) do
        cached = _LATEST_CYCLE_CACHE[]
        if cached === nothing || cached[1] != key
            cached = (key, _latest_cycle_uncached(df))
            _LATEST_CYCLE_CACHE[] = cached
        end
        return cached[2]
    end
end

# Verified rows: a real forecast (target strictly AFTER both the issue time and the last observed Dst — a
# 0-lead row is not a forecast and is what the locked comparison report excludes), with a FINITE observation
# and a finite V2 point + band. jnum(obs) rejects NaN/Inf observations that would otherwise reach the
# calibration summary and make JSON serialization throw.
function _verified_rows_uncached(df::DataFrame)
    nrow(df) == 0 && return view(df, Int[], :)
    (hasproperty(df, :target_time_utc_dt) &&
     hasproperty(df, :observation_dst_nt)) || return view(df, Int[], :)
    has_issue  = hasproperty(df, :issue_time_utc_dt)
    has_anchor = hasproperty(df, :latest_dst_time_utc_dt)
    (has_issue && has_anchor) || return view(df, Int[], :)
    keep = trues(nrow(df))
    for (i, r) in enumerate(eachrow(df))
        tt = _rowget(r, :target_time_utc_dt)
        issue = _rowget(r, :issue_time_utc_dt)
        anchor = _rowget(r, :latest_dst_time_utc_dt)
        valid_lead = tt isa DateTime && issue isa DateTime && anchor isa DateTime &&
                     tt > issue && tt > anchor
        lo = jnum(_v2_ci05(r))
        hi = jnum(_v2_ci95(r))
        keep[i] = _row_is_current_v2(r) && valid_lead &&
                  jnum(_rowget(r, :observation_dst_nt)) !== nothing &&
                  jnum(_v2_pred(r)) !== nothing && lo !== nothing && hi !== nothing && lo <= hi
    end
    return view(df, keep, :)
end


function verified_rows(df::DataFrame)
    key = _cached_log_key(df)
    key === nothing && return _verified_rows_uncached(df)
    return lock(_VERIFIED_ROWS_LOCK) do
        cached = _VERIFIED_ROWS_CACHE[]
        if cached === nothing || cached[1] != key
            cached = (key, _verified_rows_uncached(df))
            _VERIFIED_ROWS_CACHE[] = cached
        end
        return cached[2]
    end
end

# ---- calibration summary (computed from the log, not a static report) ------------------
# Honesty note: the *live* interval method (ACI) and the *verified* track record can differ.
# ACI was deployed after most verified rows were issued, so coverage is reported overall AND
# broken down by interval method, and the live method is reported separately. We never imply
# the historical coverage was achieved by a method that has no verified rows yet.
_src_label(s) = (s === missing || s === nothing || s == "") ? "legacy" : string(s)

function _wide_rmse(observed, predicted)
    n = length(observed)
    n == length(predicted) || throw(DimensionMismatch(
        "observed and predicted RMSE vectors must have equal lengths",
    ))
    n >= 1 || return nothing
    squared_sum = BigFloat(0)
    for index in eachindex(observed, predicted)
        difference = BigFloat(observed[index]) - BigFloat(predicted[index])
        squared_sum += difference^2
    end
    value = sqrt(squared_sum / n)
    value <= BigFloat(floatmax(Float64)) || return nothing
    converted = Float64(value)
    return isfinite(converted) ? converted : nothing
end

function _stable_rmse_or_nothing(observed, predicted)
    n = length(observed)
    n == length(predicted) || throw(DimensionMismatch(
        "observed and predicted RMSE vectors must have equal lengths",
    ))
    n >= 1 || return nothing
    scale = 0.0
    scaled_sum = 1.0
    for index in eachindex(observed, predicted)
        left = Float64(observed[index])
        right = Float64(predicted[index])
        isfinite(left) && isfinite(right) || return nothing
        difference = left - right
        isfinite(difference) || return _wide_rmse(observed, predicted)
        magnitude = abs(difference)
        iszero(magnitude) && continue
        if scale < magnitude
            scaled_sum = 1.0 + scaled_sum * (scale / magnitude)^2
            scale = magnitude
        else
            scaled_sum += (magnitude / scale)^2
        end
    end
    iszero(scale) && return 0.0
    value = scale * sqrt(scaled_sum / n)
    return isfinite(value) ? value : _wide_rmse(observed, predicted)
end

_prefer_metric(primary, fallback) = primary === nothing ? fallback : primary

function _compute_calibration_summary(df::DataFrame)
    # Live interval method = source of the most recent issued forecast cycle.
    cyc = latest_cycle(df)
    live_src = _valid_live_cycle(cyc) ? string(_common_cycle_field(cyc, :interval_source)) : "unknown"

    v = verified_rows(df)
    n = nrow(v)
    n == 0 && return (n_verified=0, coverage_90=nothing, rmse_nt=nothing,
                      v2_n_verified=0, v2_coverage_90=nothing, v2_rmse_nt=nothing,
                      frozen_tail_ablation_rmse_nt=nothing,
                      audit_baseline_rmse_nt=nothing,
                      rmse_sindy_v1_nt=nothing, rmse_persistence_nt=nothing,
                      rmse_burton_nt=nothing, rmse_burton_full_nt=nothing,
                      rmse_obrien_nt=nothing,
                      comparison_n_verified=0,
                      v2_matched_rmse_nt=nothing,
                      frozen_tail_ablation_matched_rmse_nt=nothing,
                      sindy_v1_matched_rmse_nt=nothing,
                      persistence_matched_rmse_nt=nothing,
                      burton_matched_rmse_nt=nothing,
                      burton_full_matched_rmse_nt=nothing,
                      obrien_matched_rmse_nt=nothing,
                      live_skill_min_verified=LIVE_SKILL_MIN_VERIFIED,
                      live_skill_mature=false,
                      live_skill_rows_remaining=LIVE_SKILL_MIN_VERIFIED,
                      interval_target_coverage=0.90,
                      served_interval_coverage_scope="empirical_only",
                      current_interval_source=live_src, n_verified_current_source=0,
                      deepest_obs_dst_nt=nothing, n_storm_verified=0, by_source=[])
    obs   = Float64.(v.observation_dst_nt)
    pred  = Float64.(_v2_pred.(eachrow(v)))
    ci05  = Float64.(_v2_ci05.(eachrow(v)))
    ci95  = Float64.(_v2_ci95.(eachrow(v)))
    audit_pred = [jnum(_audit_pred(r)) for r in eachrow(v)]
    inside = (obs .>= ci05) .& (obs .<= ci95)
    rmse(p) = _stable_rmse_or_nothing(obs, p)
    product_col_mask = falses(n)
    if all(c -> c in names(v), ("served_pred_dst_nt", "served_pred_dst_ci05_nt", "served_pred_dst_ci95_nt"))
        product_col_mask .= [jnum(v[i, :served_pred_dst_nt]) !== nothing &&
                             jnum(v[i, :served_pred_dst_ci05_nt]) !== nothing &&
                             jnum(v[i, :served_pred_dst_ci95_nt]) !== nothing for i in 1:n]
    end
    product_col_n = count(product_col_mask)
    product_col_cov = nothing
    product_col_rmse = nothing
    rmse_optional(values, mask=trues(n)) = begin
        valid = mask .& .!isnothing.(values)
        any(valid) || return nothing
        value = _stable_rmse_or_nothing(obs[valid], Float64.(values[valid]))
        return value === nothing ? nothing : jnum(round(value; digits=2))
    end
    audit_rmse = rmse_optional(audit_pred)
    if product_col_n > 0
        spred = Float64.(v[product_col_mask, :served_pred_dst_nt])
        slo = Float64.(v[product_col_mask, :served_pred_dst_ci05_nt])
        shi = Float64.(v[product_col_mask, :served_pred_dst_ci95_nt])
        sobs = obs[product_col_mask]
        sinside = (sobs .>= slo) .& (sobs .<= shi)
        product_col_cov = jnum(round(mean(sinside); digits=3))
        product_value = _stable_rmse_or_nothing(sobs, spred)
        product_col_rmse = product_value === nothing ? nothing :
            jnum(round(product_value; digits=2))
        audit_rmse = rmse_optional(audit_pred, product_col_mask)
    end
    product_mask = product_col_n > 0 ? product_col_mask : trues(n)
    product_n = count(product_mask)
    product_cov = product_col_n > 0 ? product_col_cov : jnum(round(mean(inside); digits=3))
    fallback_rmse = rmse(pred)
    product_rmse = product_col_n > 0 ? product_col_rmse :
        fallback_rmse === nothing ? nothing : jnum(round(fallback_rmse; digits=2))
    v1   = "v1_pred_dst_nt" in names(v) ? jnum.(v.v1_pred_dst_nt) : fill(nothing, n)
    pers = "persistence_dst_nt" in names(v) ? jnum.(v.persistence_dst_nt) : fill(nothing, n)
    burt = "burton_dst_nt" in names(v) ? jnum.(v.burton_dst_nt) : fill(nothing, n)
    burf = "burton_full_dst_nt" in names(v) ? jnum.(v.burton_full_dst_nt) : fill(nothing, n)
    obri = "obrien_dst_nt" in names(v) ? jnum.(v.obrien_dst_nt) : fill(nothing, n)

    # The displayed comparison is a genuinely matched cohort: a target enters only when
    # the served product, frozen-tail ablation, and every declared comparator are finite.
    # This prevents a method with missing hard cases from receiving an incomparable RMSE.
    comparison_mask = copy(product_mask)
    for values in (audit_pred, v1, pers, burt, burf, obri)
        comparison_mask .&= .!isnothing.(values)
    end
    comparison_n = count(comparison_mask)
    matched_rmse(values) = rmse_optional(values, comparison_mask)
    v2_values = jnum.(pred)
    live_skill_mature = comparison_n >= LIVE_SKILL_MIN_VERIFIED

    # per-method coverage breakdown
    srcs = "interval_source" in names(v) ? _src_label.(v.interval_source) : fill("unknown", n)
    product_srcs = srcs[product_mask]
    product_inside = inside[product_mask]
    by_source = NamedTuple[]
    for s in sort(unique(product_srcs))
        m = product_srcs .== s
        push!(by_source, (source=s, n=count(m),
                         coverage_90=round(mean(product_inside[m]); digits=3)))
    end
    n_live = count(product_srcs .== live_src)

    return (n_verified=product_n,
            coverage_90=product_cov,
            rmse_nt=product_rmse,
            v2_n_verified=product_n,
            v2_coverage_90=product_cov,
            v2_rmse_nt=product_rmse,
            frozen_tail_ablation_rmse_nt=audit_rmse,
            # Deprecated compatibility alias. New clients should use the
            # explicit V2.1 frozen-tail-ablation field above.
            audit_baseline_rmse_nt=audit_rmse,
            rmse_sindy_v1_nt=rmse_optional(v1, product_mask),
            rmse_persistence_nt=rmse_optional(pers, product_mask),
            rmse_burton_nt=rmse_optional(burt, product_mask),
            rmse_burton_full_nt=rmse_optional(burf, product_mask),
            rmse_obrien_nt=rmse_optional(obri, product_mask),
            comparison_n_verified=comparison_n,
            v2_matched_rmse_nt=matched_rmse(v2_values),
            frozen_tail_ablation_matched_rmse_nt=matched_rmse(audit_pred),
            sindy_v1_matched_rmse_nt=matched_rmse(v1),
            persistence_matched_rmse_nt=matched_rmse(pers),
            burton_matched_rmse_nt=matched_rmse(burt),
            burton_full_matched_rmse_nt=matched_rmse(burf),
            obrien_matched_rmse_nt=matched_rmse(obri),
            live_skill_min_verified=LIVE_SKILL_MIN_VERIFIED,
            live_skill_mature=live_skill_mature,
            live_skill_rows_remaining=max(0, LIVE_SKILL_MIN_VERIFIED - comparison_n),
            interval_target_coverage=0.90,
            served_interval_coverage_scope="empirical_only",
            current_interval_source=live_src,
            n_verified_current_source=n_live,
            deepest_obs_dst_nt=jnum(round(minimum(obs[product_mask]); digits=1)),
            n_storm_verified=count(obs[product_mask] .< -50),
            by_source=by_source)
end

function calibration_summary(df::DataFrame)
    key = _cached_log_key(df)
    key === nothing && return _compute_calibration_summary(df)
    return lock(_CALIBRATION_LOCK) do
        cached = _CALIBRATION_CACHE[]
        if cached === nothing || cached[1] != key
            cached = (key, _compute_calibration_summary(df))
            _CALIBRATION_CACHE[] = cached
        end
        return cached[2]
    end
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
#
# Staleness gate (documented contract): the engine writes the sidecar in a try/catch AFTER
# appending the log row, so a failed trajectory computation leaves an old sidecar while the log
# advances — which would draw a sub-hour curve from a different cycle than the served hourly
# forecast. The sidecar records its own issue/anchor time; we serve it only when its issue time
# matches the current cycle's, and drop any point that predates the cycle anchor.
function _subhour_traj(log_path::AbstractString; cycle_issue=nothing)
    isempty(log_path) && return NamedTuple[]
    f = joinpath(dirname(log_path), "subhour_trajectory.json")
    isfile(f) || return NamedTuple[]
    try
        d = JSON3.read(read(f, String))
        if cycle_issue !== nothing && cycle_issue !== missing
            sc_issue = haskey(d, :issue_time_utc) ? parse_dt(get(d, :issue_time_utc, nothing)) : missing
            (sc_issue === missing || abs(sc_issue - cycle_issue) > _CYCLE_TOL) && return NamedTuple[]
        end
        anchor = haskey(d, :anchor_time_utc) ? parse_dt(get(d, :anchor_time_utc, nothing)) : missing
        out = NamedTuple[]
        for p in d.points
            t = parse_dt(String(p.t))
            t === missing && continue
            (anchor !== missing && t < anchor) && continue
            push!(out, (target_utc = String(p.t) * "Z", dst_nt = jnum(p.dst)))
        end
        return out
    catch e
        e isa InterruptException && rethrow()
        return NamedTuple[]
    end
end

# Staleness of a served cycle: age since the newest issue time, and whether every target hour is
# already in the past. A daemon crash or feed retirement freezes the log, so a cycle issued
# hours-to-days ago must not be served as the current operational status without a machine-readable
# flag. Threshold is tied to the hourly issue cadence.
const STALE_CYCLE_HOURS = 3.0   # hourly issue cadence; flag stale after ~3 missed issue cycles
const FUTURE_CYCLE_TOL_HOURS = 5 / 60  # tolerate only the documented intra-cycle clock spread

function _cycle_staleness(cyc::DataFrame)
    issue_max = (hasproperty(cyc, :issue_time_utc_dt) && !isempty(collect(skipmissing(cyc.issue_time_utc_dt)))) ?
                maximum(skipmissing(cyc.issue_time_utc_dt)) : missing
    age_hours = issue_max === missing ? nothing : (now(UTC) - issue_max) / Millisecond(3_600_000)
    invalid_future = age_hours !== nothing && age_hours < -FUTURE_CYCLE_TOL_HOURS
    stale = age_hours !== nothing && (age_hours > STALE_CYCLE_HOURS || invalid_future)
    tmax = (hasproperty(cyc, :target_time_utc_dt) && !isempty(collect(skipmissing(cyc.target_time_utc_dt)))) ?
           maximum(skipmissing(cyc.target_time_utc_dt)) : missing
    expired = tmax !== missing && tmax < now(UTC)
    return (age_hours=age_hours, stale=stale, expired=expired,
            invalid_future=invalid_future, issue_max=issue_max)
end

# A batch issues these four requested horizons sequentially. The log stores wall-clock lead in
# `horizon_hours`, so completeness is identified from the hourly target schedule instead.
const LIVE_CYCLE_HORIZONS = (1, 2, 3, 6)

function _common_cycle_field(cyc::DataFrame, name::Symbol)
    hasproperty(cyc, name) || return nothing
    values = [_rowget(r, name) for r in eachrow(cyc)]
    isempty(values) && return nothing
    any(x -> x === missing || x === nothing, values) && return nothing
    value = first(values)
    return all(x -> isequal(x, value), values) ? value : nothing
end

"""Validate one current forecast cycle as a complete, ordered, positive-lead trajectory."""
function _valid_live_cycle(cyc::DataFrame)
    nrow(cyc) == length(LIVE_CYCLE_HORIZONS) || return false
    model = _common_cycle_field(cyc, :model_version)
    served_model = _common_cycle_field(cyc, :sub_hourly_model_version)
    interval = _common_cycle_field(cyc, :interval_source)
    anchor = _common_cycle_field(cyc, :latest_dst_time_utc_dt)
    anchor_dst = jnum(_common_cycle_field(cyc, :latest_dst_nt))
    vintage = _common_cycle_field(cyc, :latest_solar_wind_utc_dt)
    model == CURRENT_V2_MODEL_VERSION || return false
    served_model == CURRENT_V2_SERVED_MODEL_VERSION || return false
    interval isa AbstractString && !isempty(strip(interval)) || return false
    anchor isa DateTime && anchor_dst !== nothing && vintage isa DateTime || return false

    issues = DateTime[]
    targets = DateTime[]
    for r in eachrow(cyc)
        issue = _rowget(r, :issue_time_utc_dt)
        target = _rowget(r, :target_time_utc_dt)
        horizon = jnum(_rowget(r, :horizon_hours))
        pred = jnum(_v2_pred(r))
        lo = jnum(_v2_ci05(r))
        hi = jnum(_v2_ci95(r))
        issue isa DateTime && target isa DateTime || return false
        target > issue || return false
        target > anchor || return false
        horizon !== nothing && horizon > 0 || return false
        lead_hours = (target - issue) / Millisecond(3_600_000)
        abs(horizon - lead_hours) <= 0.05 || return false
        pred !== nothing && lo !== nothing && hi !== nothing || return false
        lo <= pred <= hi || return false
        push!(issues, issue)
        push!(targets, target)
    end
    issue_hour = floor(first(issues), Hour)
    all(issue -> floor(issue, Hour) == issue_hour, issues) || return false
    maximum(issues) - minimum(issues) <= _CYCLE_TOL || return false
    expected_targets = [issue_hour + Hour(h) for h in LIVE_CYCLE_HORIZONS]
    return targets == expected_targets
end

"""Forecast trajectory of the most recent cycle: anchor + per-horizon point and 90% band."""
function build_forecast(df::DataFrame, log_path::AbstractString="")
    cyc = latest_cycle(df)
    nrow(cyc) == 0 && return (available=false, issue_time_utc=nothing, horizons=[])
    stale = _cycle_staleness(cyc)
    if stale.invalid_future || stale.expired || stale.stale
        reason = stale.invalid_future ? "future issue time" :
                 stale.expired ? "expired targets" : "stale issue time"
        return (available=false, issue_time_utc=jdt(stale.issue_max), horizons=[],
                stale=true, expired=stale.expired,
                invalid_future=stale.invalid_future,
                age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
                message="Latest forecast cycle is unavailable: $reason.")
    end
    _valid_live_cycle(cyc) ||
        return (available=false, issue_time_utc=jdt(stale.issue_max), horizons=[],
                stale=false, expired=false, invalid_future=false,
                message="Latest forecast cycle has incomplete or inconsistent rows.")
    horizons = NamedTuple[]
    for r in eachrow(cyc)
        target = _rowget(r, :target_time_utc_dt)
        pred = jnum(_v2_pred(r))
        lo = jnum(_v2_ci05(r))
        hi = jnum(_v2_ci95(r))
        push!(horizons, (target_utc=jdt(target),
                         horizon_hours=jnum(_rowget(r, :horizon_hours)),
                         pred_dst_nt=pred,
                         ci05_dst_nt=lo,
                         ci95_dst_nt=hi,
                         frozen_tail_ablation_dst_nt=jnum(_audit_pred(r)),
                         audit_baseline_dst_nt=jnum(_audit_pred(r))))
    end
    issue = stale.issue_max
    cycle_issue = issue === missing ? nothing : issue
    return (available=true,
            issue_time_utc=jdt(issue),
            latest_solar_wind_utc=jdt(_common_cycle_field(cyc, :latest_solar_wind_utc_dt)),
            anchor_dst_nt=jnum(_common_cycle_field(cyc, :latest_dst_nt)),
            anchor_dst_time_utc=jdt(_common_cycle_field(cyc, :latest_dst_time_utc_dt)),
            interval_source=string(_common_cycle_field(cyc, :interval_source)),
            model_version=string(_common_cycle_field(cyc, :model_version)),
            served_model_version=string(_common_cycle_field(cyc, :sub_hourly_model_version)),
            stale=stale.stale, expired=stale.expired, invalid_future=stale.invalid_future,
            age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
            recent_observed=_recent_observed(df),
            subhour_trajectory=_subhour_traj(log_path; cycle_issue=cycle_issue),
            horizons=horizons)
end

"""Storm-time replay summary: serves the offline replay report and its scored-row provenance.
Defensive (never throws): missing/unreadable artifacts return available=false."""
function build_storm_replay(log_path::AbstractString; evidence_dir=nothing)
    dir = evidence_dir === nothing ? dirname(log_path) : String(evidence_dir)
    report = joinpath(dir, "storm_replay_report.md")
    scored = joinpath(dir, "storm_replay_scored.csv")
    isfile(report) || return (available=false,
                              reason="no storm-replay report; run validation/operational/storm_replay.jl")
    md = try
        read(report, String)
    catch e
        e isa InterruptException && rethrow()
        return (available=false, reason="report unreadable")
    end
    age = round((time() - mtime(report)) / 60; digits=1)
    n = 0; storms = String[]
    if isfile(scored)
        try
            df = CSV.read(scored, DataFrame)
            n = nrow(df)
            "storm" in names(df) && (storms = unique(string.(skipmissing(df.storm))))
        catch e
            e isa InterruptException && rethrow()
        end
    end
    return (available=true, report_age_min=age, n_scored=n, storms=storms, report_markdown=md)
end

"""Threat status: current observation and most negative edge among the latest 90% intervals."""
function build_status(df::DataFrame)
    cyc = latest_cycle(df)
    cal = calibration_summary(df)
    if nrow(cyc) == 0
        return (generated_utc=jdt(now(UTC)), available=false,
                message="No forecast rows in log yet.", calibration=cal)
    end
    stale = _cycle_staleness(cyc)
    if stale.invalid_future
        return (generated_utc=jdt(now(UTC)), available=false, stale=true, expired=false,
                invalid_future=true,
                age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
                forecast_issue_utc=jdt(stale.issue_max),
                message="Latest forecast cycle has an issue time too far in the future.",
                calibration=cal)
    end
    if stale.expired
        # Every target hour is in the past: the daemon has stopped issuing (crash or feed
        # retirement). Serve available=false with an explicit stale/expired flag so build_alerts
        # suppresses forecast/watch alerts and the dashboard does not present an expired forecast
        # as the current threat.
        return (generated_utc=jdt(now(UTC)), available=false, stale=true, expired=true,
                age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
                forecast_issue_utc=jdt(stale.issue_max),
                message="Latest forecast cycle expired; all target hours are in the past.",
                calibration=cal)
    end
    if stale.stale
        return (generated_utc=jdt(now(UTC)), available=false, stale=true,
                expired=false, invalid_future=false,
                age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
                forecast_issue_utc=jdt(stale.issue_max),
                message="Latest forecast cycle is stale.", calibration=cal)
    end
    if !_valid_live_cycle(cyc)
        return (generated_utc=jdt(now(UTC)), available=false, stale=stale.stale,
                expired=false, invalid_future=false,
                age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
                forecast_issue_utc=jdt(stale.issue_max),
                message="Latest forecast cycle has incomplete or invalid point/interval rows.",
                calibration=cal)
    end
    preds = [Float64(_v2_pred(r)) for r in eachrow(cyc)]
    lbs = [Float64(_v2_ci05(r)) for r in eachrow(cyc)]
    point_min = minimum(preds)
    interval_lower_edge_min = minimum(lbs)
    lvl_pt, lbl_pt = dst_threat_level(point_min)
    lvl_wc, lbl_wc = dst_threat_level(interval_lower_edge_min)
    # Reported threat level is the point-forecast level; a "watch" flag fires when the
    # lower edge of a displayed 90%-target interval reaches a stronger storm tier than the
    # point forecast. This does not turn the marginal interval into a one-sided bound.
    watch = lvl_wc > lvl_pt
    horizon_max = (h = filter(!isnothing,
        [jnum(_rowget(r, :horizon_hours)) for r in eachrow(cyc)]);
        isempty(h) ? nothing : maximum(h))
    return (generated_utc=jdt(now(UTC)),
            available=true,
            stale=stale.stale, expired=false,
            age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
            latest_observation=(dst_nt=jnum(_common_cycle_field(cyc, :latest_dst_nt)),
                                time_utc=jdt(_common_cycle_field(cyc, :latest_dst_time_utc_dt))),
            latest_solar_wind_utc=jdt(_common_cycle_field(cyc, :latest_solar_wind_utc_dt)),
            forecast_issue_utc=jdt(stale.issue_max),
            threat=(level=lvl_pt, label=lbl_pt, watch=watch,
                    watch_level=lvl_wc, watch_label=lbl_wc,
                    point_min_dst_nt=point_min,
                    interval_lower_edge_min_dst_nt=interval_lower_edge_min,
                    # Deprecated value-equivalent aliases for already-loaded dashboards and API
                    # clients. New code and all visible labels use interval_lower_edge_min_dst_nt.
                    lower_bound_min_dst_nt=interval_lower_edge_min,
                    worst_credible_dst_nt=interval_lower_edge_min,
                    basis="Dst storm-intensity scale (-30/-50/-100/-200 nT)"),
            lead_time=(forecast_horizon_hours=horizon_max,
                       driver_assumption="Ballistically propagated L1 forcing, then regime-aware relaxation beyond the measured L1 window, followed by a causal rate projection, validation-selected one-hour and state-conditioned inertia blends, and an extreme-Dst inertia guard",
                       physical_upstream_lead_min=[30, 60],
                       note="Genuine upstream lead for new severity is the L1 advection time (~30-60 min). " *
                            "Multi-day lead requires CME eruption/propagation models, not yet in this system."),
            calibration=cal,
            model_version=string(_common_cycle_field(cyc, :model_version)),
            served_model_version=string(_common_cycle_field(cyc, :sub_hourly_model_version)))
end

function _build_history_uncached(df::DataFrame, hours::Real, reference_time::DateTime)
    v = verified_rows(df)
    nrow(v) == 0 && return (rows=[], coverage_90=nothing, rmse_nt=nothing, hours=hours)
    cutoff = reference_time - Hour(round(Int, hours))
    rows = NamedTuple[]
    for r in eachrow(v)
        t = _rowget(r, :target_time_utc_dt)
        (t === missing || t < cutoff) && continue
        obs = jnum(_rowget(r, :observation_dst_nt)); p = jnum(_v2_pred(r))
        lo = jnum(_v2_ci05(r)); hi = jnum(_v2_ci95(r))
        inside = (obs !== nothing && lo !== nothing && hi !== nothing) ? (obs >= lo && obs <= hi) : nothing
        push!(rows, (target_utc=jdt(t), horizon_hours=jnum(_rowget(r, :horizon_hours)),
                     observed_dst_nt=obs, pred_dst_nt=p, ci05_dst_nt=lo, ci95_dst_nt=hi,
                     frozen_tail_ablation_dst_nt=jnum(_audit_pred(r)),
                     audit_baseline_dst_nt=jnum(_audit_pred(r)),
                     inside_90ci=inside))
    end
    sort!(rows, by=x -> something(x.target_utc, ""))
    # Stats must match the WINDOW the rows show, not the whole log. Compute coverage/RMSE over
    # the windowed rows; also expose the (more robust) all-log figures under explicit names.
    flags = [r.inside_90ci for r in rows if r.inside_90ci !== nothing]
    paired = [(r.observed_dst_nt, r.pred_dst_nt) for r in rows
              if r.observed_dst_nt !== nothing && r.pred_dst_nt !== nothing]
    win_cov = isempty(flags) ? nothing : round(count(flags) / length(flags); digits=3)
    window_value = isempty(paired) ? nothing : _stable_rmse_or_nothing(
        first.(paired), last.(paired),
    )
    win_rmse = window_value === nothing ? nothing : round(window_value; digits=2)
    cal = calibration_summary(df)
    return (rows=rows, hours=hours, n=length(rows),
            coverage_90=win_cov, rmse_nt=win_rmse,
            coverage_90_all=_prefer_metric(cal.v2_coverage_90, cal.coverage_90),
            rmse_nt_all=_prefer_metric(cal.v2_rmse_nt, cal.rmse_nt))
end


"""Recent verified track record (observed vs predicted with band) for the last `hours`."""
function build_history(df::DataFrame, hours::Real=72)
    reference_time = now(UTC)
    log_key = _cached_log_key(df)
    log_key === nothing && return _build_history_uncached(df, hours, reference_time)
    key = (log_key, Float64(hours), floor(reference_time, Hour))
    return lock(_HISTORY_LOCK) do
        cached = _HISTORY_CACHE[]
        if cached === nothing || cached[1] != key
            cached = (key, _build_history_uncached(df, hours, reference_time))
            _HISTORY_CACHE[] = cached
        end
        return cached[2]
    end
end

"""Active alert summary derived from the current threat status."""
function build_alerts(df::DataFrame, st=build_status(df))
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
                       message="A displayed 90% target interval extends to " *
                               "$(round(th.interval_lower_edge_min_dst_nt; digits=0)) nT " *
                               "($(th.watch_label) range) at one or more horizons."))
    end
    return (active=!isempty(alerts), generated_utc=st.generated_utc,
            threat_level=th.level, threat_label=th.label, alerts=alerts)
end
