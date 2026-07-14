# swpc.jl — NOAA SWPC public feed ingestion (open, no-auth, real-time space-weather feeds).
#
# Adds the *upstream* picture that complements our Dst nowcast: real-time L1 solar wind
# (DSCOVR Bz/speed/density), planetary Kp, the NOAA G/S/R scales, and official alerts.
#
# Robustness (CRC): every fetch degrades gracefully — a network failure or schema surprise
# returns `available=false`, never throws into the request path. A short-TTL cache keeps the
# dashboard's frequent polls from hammering SWPC (a good-citizen constraint on a public API).
#
# Depends on jnum/jdt helpers from forecast_api.jl (included before this file).

using HTTP, JSON3, Dates

const SWPC_BASE = "https://services.swpc.noaa.gov"
const SWPC_TTL = 50.0                       # seconds; SWPC products update ~1/min
const _SWPC_CACHE = Ref{Any}(nothing)       # (fetch_time::Float64, snapshot)
const _SWPC_LOCK = ReentrantLock()
const _SWPC_REFRESH_TASK = Ref{Union{Nothing,Task}}(nothing)
# Solar-wind inputs older than this no longer represent current upstream conditions for the
# 30-min GIC forecast (RTSW products update ~1/min, and the physical L1 lead is ~30-60 min, so
# a >15-min gap means the driver forecast would run on frozen values).
const SW_INPUT_MAX_AGE_MIN = 15.0
const SWPC_FUTURE_TOL_MIN = 2.0
const KP_MAX_AGE_MIN = 240.0                  # Kp is a 3-hour product
const SCALES_MAX_AGE_MIN = 120.0

# parse a possibly-string numeric to Float64 or nothing
_pf(x) = x === nothing ? nothing : (x isa Number ? Float64(x) : try
    parse(Float64, strip(String(x)))
catch e
    e isa InterruptException && rethrow()
    nothing
end)

function _swpc_row_field(idx, row, name)
    i = get(idx, name, nothing)
    (i === nothing || i > length(row)) && return nothing
    return jnum(_pf(row[i]))
end

# SWPC times: "2026-06-20 10:52:00.000" or "2026-06-20T06:00:00"
function _swpc_dt(s)
    (s === nothing || s === missing) && return missing
    str = replace(strip(String(s)), " " => "T")
    try
        return DateTime(first(str, 19))
    catch e
        e isa InterruptException && rethrow()
        return missing
    end
end

function _source_freshness(timestamp, max_age_min::Real;
                           reference::DateTime=now(UTC),
                           future_tolerance_min::Real=SWPC_FUTURE_TOL_MIN)
    dt = timestamp isa DateTime ? timestamp : parse_dt(timestamp)
    age = dt === missing ? nothing : (reference - dt) / Millisecond(60_000)
    future = age !== nothing && age < -future_tolerance_min
    stale = age === nothing || future || age > max_age_min
    return (age_min = age === nothing ? nothing : round(age; digits=1),
            stale = stale, invalid_future = future)
end

function _swpc_get(path; readtimeout=1, connect_timeout=1)
    try
        r = HTTP.get(SWPC_BASE * path; readtimeout=readtimeout,
                     connect_timeout=connect_timeout, retries=0, status_exception=true)
        return JSON3.read(r.body)
    catch e
        e isa InterruptException && rethrow()
        @warn "SWPC fetch failed" path exception=e
        return nothing
    end
end

# --- individual products ---------------------------------------------------------------
# RTSW real-time solar wind (the retired /products/solar-wind/{mag,plasma}-1-day.json now 404).
# Both feeds are ARRAYS OF OBJECTS with named keys (never positional columns):
#   mag  (/json/rtsw/rtsw_mag_1m.json):  time_tag, active, source, bt, bz_gsm, ...
#   wind (/json/rtsw/rtsw_wind_1m.json): time_tag, active, source, proton_speed, proton_density, ...
# Records from several spacecraft (e.g. SOLAR1, ACE) are interleaved and can repeat a time_tag;
# `active=true` marks the currently-designated primary L1 source. We keep only active,
# physically-valid rows and take the newest by time_tag, so a trailing null row does not blank
# the panel and a stale/secondary record cannot masquerade as the current reading.

# Named-key numeric field from a JSON object (JSON3.Object or Dict), or nothing on
# missing key / null / non-finite value.
function _rtsw_field(obj, key::Symbol)
    obj === nothing && return nothing
    try
        haskey(obj, key) || return nothing
        return jnum(_pf(obj[key]))
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
end

# `active` flag: true when the key is absent (single-source feeds) or explicitly
# Boolean true. Malformed encodings are not preferred as the designated source;
# the second selection pass can still use their otherwise valid measurements.
function _rtsw_active(obj)
    try
        haskey(obj, :active) || return true
        return obj[:active] === true
    catch e
        e isa InterruptException && rethrow()
        return false
    end
end

# Newest physically-valid RTSW record: schema-validated (all `reqkeys` present, finite, within
# `bounds`), preferring the currently-active source and falling back to any valid source. Records
# are documented newest-first but we select by parsed time_tag so ordering is not assumed.
function _rtsw_latest(arr, reqkeys::Vector{Symbol};
                      bounds::Dict{Symbol,Tuple{Float64,Float64}} = Dict{Symbol,Tuple{Float64,Float64}}(),
                      reference::DateTime = now(UTC))
    (arr === nothing || length(arr) < 1) && return nothing
    _valid(obj) = begin
        time_value = try
            haskey(obj, :time_tag) ? obj[:time_tag] : nothing
        catch e
            e isa InterruptException && rethrow()
            nothing
        end
        t = _swpc_dt(time_value)
        t === missing && return (false, missing)
        t > reference + Millisecond(round(Int, SWPC_FUTURE_TOL_MIN * 60_000)) &&
            return (false, missing)
        for k in reqkeys
            v = _rtsw_field(obj, k)
            v === nothing && return (false, missing)
            if haskey(bounds, k)
                lo, hi = bounds[k]
                (v < lo || v > hi) && return (false, missing)
            end
        end
        return (true, t)
    end
    for require_active in (true, false)          # prefer the active source, else any valid source
        best = nothing; best_t = missing
        for obj in arr
            require_active && !_rtsw_active(obj) && continue
            ok, t = _valid(obj)
            ok || continue
            if best_t === missing || t > best_t
                best = obj; best_t = t
            end
        end
        best !== nothing && return best
    end
    return nothing
end

function swpc_solar_wind()
    sw = Dict{Symbol,Any}(:available => false)
    # The RTSW products are ~1-2 MB (the retired 1-day products were KB-scale), so the 1 s default
    # read timeout is too tight for a cold-connection download + parse; 5 s tolerates it while
    # still bounding a stalled feed (the snapshot is cached and fetched off the request lock).
    mag = _swpc_get("/json/rtsw/rtsw_mag_1m.json"; readtimeout=5, connect_timeout=3)
    # bt is a magnitude (>= 0); |B| components rarely exceed a few hundred nT at L1 — generous
    # bounds only reject sentinel/garbage fill values, not real extreme-storm readings.
    magrow = _rtsw_latest(mag, [:bz_gsm, :bt];
                          bounds = Dict(:bt => (0.0, 1.0e3), :bz_gsm => (-1.0e3, 1.0e3)))
    if magrow !== nothing
        sw[:bz_gsm_nt]    = _rtsw_field(magrow, :bz_gsm)
        sw[:bt_nt]        = _rtsw_field(magrow, :bt)
        sw[:mag_time_utc] = jdt(_swpc_dt(magrow[:time_tag]))
        sw[:mag_source]   = haskey(magrow, :source) && magrow[:source] !== nothing ? String(magrow[:source]) : nothing
        sw[:available]    = true
    end
    pls = _swpc_get("/json/rtsw/rtsw_wind_1m.json"; readtimeout=5, connect_timeout=3)
    # Solar-wind speed spans ~200-1000 km/s in quiet-to-storm conditions, up to ~3000 in extreme
    # events; density is a few to a few hundred cm^-3. Bounds gate obvious non-physical fills.
    plsrow = _rtsw_latest(pls, [:proton_speed, :proton_density];
                          bounds = Dict(:proton_speed => (50.0, 5.0e3), :proton_density => (0.0, 1.0e3)))
    if plsrow !== nothing
        sw[:speed_kms]       = _rtsw_field(plsrow, :proton_speed)
        sw[:density_cm3]     = _rtsw_field(plsrow, :proton_density)
        sw[:plasma_time_utc] = jdt(_swpc_dt(plsrow[:time_tag]))
        sw[:plasma_source]   = haskey(plsrow, :source) && plsrow[:source] !== nothing ? String(plsrow[:source]) : nothing
    end
    return sw
end

# Age [min] of the OLDER of the mag/plasma feeds (both must be current for a valid driver
# forecast). Returns nothing when neither timestamp is parseable.
function solar_wind_input_age_min(sw)
    f = solar_wind_input_freshness(sw)
    return f.age_min
end

# array of objects {time_tag, Kp, a_running, station_count} — no header row
function _parse_swpc_kp(kp; reference::DateTime=now(UTC))
    (kp === nothing || length(kp) < 1) && return nothing
    valid = NamedTuple[]
    for row in kp
        t_raw = try get(row, :time_tag, nothing) catch e
            e isa InterruptException && rethrow()
            nothing
        end
        v_raw = try get(row, :Kp, nothing) catch e
            e isa InterruptException && rethrow()
            nothing
        end
        t = _swpc_dt(t_raw)
        value = jnum(_pf(v_raw))
        (t === missing || value === nothing || !(0.0 <= value <= 9.0) ||
         t > reference + Millisecond(round(Int, SWPC_FUTURE_TOL_MIN * 60_000))) && continue
        push!(valid, (time=t, value=value))
    end
    isempty(valid) && return nothing
    sort!(valid; by=row -> row.time)
    newest = last(valid)
    first_trend = max(1, length(valid) - 7)
    trend = [(time_utc=jdt(row.time), kp=row.value)
             for row in @view valid[first_trend:end]]
    return (value=newest.value, time_utc=jdt(newest.time), trend=trend)
end

function swpc_kp()
    kp = _swpc_get("/products/noaa-planetary-k-index.json")
    return _parse_swpc_kp(kp)
end

# object keyed "0".."N"; "0" is current conditions, R/S/G each with Scale/Text
function swpc_scales()
    sc = _swpc_get("/products/noaa-scales.json")
    sc === nothing && return nothing
    try
        cur = sc[Symbol("0")]
        gv(k) = haskey(cur, k) && cur[k] !== nothing ? String(cur[k].Scale) : nothing
        gt(k) = haskey(cur, k) && cur[k] !== nothing ? String(cur[k].Text)  : nothing
        ts = (haskey(cur, :DateStamp) && haskey(cur, :TimeStamp)) ?
             string(String(cur.DateStamp), "T", String(cur.TimeStamp), "Z") : nothing
        return (time_utc = ts, G = gv(:G), G_text = gt(:G), S = gv(:S), R = gv(:R))
    catch e
        e isa InterruptException && rethrow()
        @warn "SWPC scales parse failed" exception=e
        return nothing
    end
end

# array of {product_id, issue_datetime, message}; extract a readable summary line
function swpc_alerts(n::Int = 6)
    al = _swpc_get("/products/alerts.json")
    (al === nothing || isempty(al)) && return NamedTuple[]
    out = NamedTuple[]
    for i in 1:min(n, length(al))
        m = al[i]
        msg = String(get(m, :message, ""))
        summary = ""
        for kw in ("WARNING:", "ALERT:", "WATCH:", "SUMMARY:", "EXTENDED WARNING:")
            r = findfirst(kw, msg)
            if r !== nothing
                rest = msg[first(r):end]
                nl = findfirst(c -> c == '\n' || c == '\r', rest)
                summary = strip(rest[1:(nl === nothing ? length(rest) : nl - 1)])
                break
            end
        end
        isempty(summary) && (summary = strip(replace(first(msg, 80), r"[\r\n]+" => " ")))
        push!(out, (product_id = String(get(m, :product_id, "")),
                    issue_utc = jdt(_swpc_dt(get(m, :issue_datetime, nothing))),
                    summary = summary))
    end
    return out
end

# --- combined snapshot (cached) --------------------------------------------------------
function _build_swpc_snapshot()
    sw = swpc_solar_wind(); kp = swpc_kp(); sc = swpc_scales(); al = swpc_alerts(6)
    available = sw[:available] || kp !== nothing || sc !== nothing
    return (source = "NOAA SWPC", fetched_utc = jdt(now(UTC)), available = available,
            solar_wind = sw, kp = kp, scales = sc, alerts = al)
end

function _run_swpc_refresh(; build_fn=_build_swpc_snapshot)
    cache_result = false
    val = nothing
    try
        val = try
            out = build_fn()
            if get(out, :available, false)
                cache_result = true
                out
            else
                stale = lock(_SWPC_LOCK) do
                    c = _SWPC_CACHE[]; c === nothing ? nothing : c[2]
                end
                stale === nothing ? out : stale
            end
        catch e
            e isa InterruptException && rethrow()
            stale = lock(_SWPC_LOCK) do
                c = _SWPC_CACHE[]; c === nothing ? nothing : c[2]
            end
            if stale !== nothing
                @warn "SWPC snapshot build failed; serving cached" exception=e
                stale
            else
                (source="NOAA SWPC", available=false, error=string(e))
            end
        end
        return val
    finally
        lock(_SWPC_LOCK) do
            cache_result && (_SWPC_CACHE[] = (time(), val))
            _SWPC_REFRESH_TASK[] = nothing
        end
    end
end

# Caller holds _SWPC_LOCK. Both blocking and stale-while-refresh entry points use this task.
function _start_swpc_refresh_locked()
    task = _SWPC_REFRESH_TASK[]
    if task === nothing || istaskdone(task)
        task = @async _run_swpc_refresh()
        _SWPC_REFRESH_TASK[] = task
    end
    return task
end

function swpc_snapshot()
    fresh, task = lock(_SWPC_LOCK) do
        c = _SWPC_CACHE[]
        c !== nothing && (time() - c[1]) <= SWPC_TTL ? (c[2], nothing) :
            (nothing, _start_swpc_refresh_locked())
    end
    return fresh === nothing ? fetch(task) : fresh
end

function swpc_snapshot_cached_or_refresh()
    fresh, stale = lock(_SWPC_LOCK) do
        c = _SWPC_CACHE[]
        if c !== nothing && (time() - c[1]) <= SWPC_TTL
            (c[2], nothing)
        else
            _start_swpc_refresh_locked()
            (nothing, c === nothing ? nothing : c[2])
        end
    end
    fresh !== nothing && return fresh
    stale !== nothing && return stale
    return (source = "NOAA SWPC", fetched_utc = jdt(now(UTC)), available = false,
            solar_wind = Dict{Symbol,Any}(:available => false), kp = nothing,
            scales = nothing, alerts = NamedTuple[])
end

# Honest threshold-based upstream indicator (NOT a fused scalar): flags "elevated" when any
# standard active-condition marker trips. Thresholds are conventional space-weather markers.
function solar_wind_input_freshness(sw; reference::DateTime=now(UTC))
    mag = _source_freshness(get(sw, :mag_time_utc, nothing), SW_INPUT_MAX_AGE_MIN;
                            reference=reference)
    plasma = _source_freshness(get(sw, :plasma_time_utc, nothing), SW_INPUT_MAX_AGE_MIN;
                               reference=reference)
    ages = filter(!isnothing, (mag.age_min, plasma.age_min))
    return (age_min = length(ages) == 2 ? maximum(ages) : nothing,
            stale = mag.stale || plasma.stale,
            invalid_future = mag.invalid_future || plasma.invalid_future,
            mag = mag, plasma = plasma)
end

function upstream_assessment(snap; reference::DateTime=now(UTC))
    (snap === nothing || !get(snap, :available, false)) && return (available = false,)
    reasons = String[]; elevated = false
    sc = get(snap, :scales, nothing)
    kp = get(snap, :kp, nothing)
    sw = get(snap, :solar_wind, Dict{Symbol,Any}())
    g = sc === nothing ? nothing : get(sc, :G, nothing)
    kpv = kp === nothing ? nothing : get(kp, :value, nothing)
    bz = get(sw, :bz_gsm_nt, nothing); sp = get(sw, :speed_kms, nothing)
    scf = _source_freshness(sc === nothing ? nothing : get(sc, :time_utc, nothing),
                            SCALES_MAX_AGE_MIN; reference=reference)
    kpf = _source_freshness(kp === nothing ? nothing : get(kp, :time_utc, nothing),
                            KP_MAX_AGE_MIN; reference=reference)
    swf = solar_wind_input_freshness(sw; reference=reference)
    if g !== nothing && !scf.stale && g != "0"
        elevated = true; push!(reasons, "NOAA G$g geomagnetic storm")
    end
    if kpv !== nothing && !kpf.stale && kpv >= 5
        elevated = true; push!(reasons, "Kp $(kpv) (storm level)")
    end
    if bz !== nothing && !swf.mag.stale && bz < -10
        elevated = true; push!(reasons, "L1 Bz $(round(bz; digits=1)) nT southward")
    end
    if sp !== nothing && !swf.plasma.stale && sp > 600
        elevated = true; push!(reasons, "L1 wind $(round(Int, sp)) km/s")
    end
    current = (g !== nothing && !scf.stale) || (kpv !== nothing && !kpf.stale) ||
              (bz !== nothing && !swf.mag.stale) || (sp !== nothing && !swf.plasma.stale)
    return (available = current, elevated = elevated, reasons = reasons,
            g_scale = g, kp = kpv, bz_gsm_nt = bz, speed_kms = sp,
            scales_age_min = scf.age_min, scales_stale = scf.stale,
            kp_age_min = kpf.age_min, kp_stale = kpf.stale,
            mag_age_min = swf.mag.age_min, mag_stale = swf.mag.stale,
            plasma_age_min = swf.plasma.age_min, plasma_stale = swf.plasma.stale,
            solar_wind_age_min = swf.age_min, solar_wind_stale = swf.stale,
            invalid_future = scf.invalid_future || kpf.invalid_future || swf.invalid_future)
end
