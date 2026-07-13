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
const _SWPC_REFRESHING = Ref(false)
# Solar-wind inputs older than this no longer represent current upstream conditions for the
# 30-min GIC forecast (RTSW products update ~1/min, and the physical L1 lead is ~30-60 min, so
# a >15-min gap means the driver forecast would run on frozen values).
const SW_INPUT_MAX_AGE_MIN = 15.0

# parse a possibly-string numeric to Float64 or nothing
_pf(x) = x === nothing ? nothing : (x isa Number ? Float64(x) :
         (try parse(Float64, strip(String(x))) catch; nothing end))

function _swpc_row_field(idx, row, name)
    i = get(idx, name, nothing)
    (i === nothing || i > length(row)) && return nothing
    return jnum(_pf(row[i]))
end

# SWPC times: "2026-06-20 10:52:00.000" or "2026-06-20T06:00:00"
function _swpc_dt(s)
    (s === nothing || s === missing) && return missing
    str = replace(strip(String(s)), " " => "T")
    try; return DateTime(first(str, 19)); catch; return missing; end
end

function _swpc_get(path; readtimeout=1, connect_timeout=1)
    try
        r = HTTP.get(SWPC_BASE * path; readtimeout=readtimeout,
                     connect_timeout=connect_timeout, retries=0, status_exception=true)
        return JSON3.read(r.body)
    catch e
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
    (obj === nothing || !haskey(obj, key)) && return nothing
    return jnum(_pf(obj[key]))
end

# `active` flag: true when the key is absent (single-source feeds) or explicitly boolean-true;
# any other encoding is treated as active so a schema surprise never silently drops every row.
function _rtsw_active(obj)
    haskey(obj, :active) || return true
    a = obj[:active]
    (a === nothing) && return true
    return a isa Bool ? a : true
end

# Newest physically-valid RTSW record: schema-validated (all `reqkeys` present, finite, within
# `bounds`), preferring the currently-active source and falling back to any valid source. Records
# are documented newest-first but we select by parsed time_tag so ordering is not assumed.
function _rtsw_latest(arr, reqkeys::Vector{Symbol};
                      bounds::Dict{Symbol,Tuple{Float64,Float64}} = Dict{Symbol,Tuple{Float64,Float64}}())
    (arr === nothing || length(arr) < 1) && return nothing
    _valid(obj) = begin
        t = _swpc_dt(haskey(obj, :time_tag) ? obj[:time_tag] : nothing)
        t === missing && return (false, missing)
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
    ts = DateTime[]
    for k in (:mag_time_utc, :plasma_time_utc)
        v = get(sw, k, nothing)
        v === nothing && continue
        dt = parse_dt(v)
        dt === missing || push!(ts, dt)
    end
    isempty(ts) && return nothing
    return (now(UTC) - minimum(ts)) / Millisecond(60_000)
end

# array of objects {time_tag, Kp, a_running, station_count} — no header row
function swpc_kp()
    kp = _swpc_get("/products/noaa-planetary-k-index.json")
    (kp === nothing || length(kp) < 1) && return nothing
    last = kp[end]
    n = length(kp); s = max(1, n - 7)        # last ~24 h of 3-hourly Kp
    trend = [(time_utc = jdt(_swpc_dt(kp[i].time_tag)), kp = jnum(_pf(kp[i].Kp))) for i in s:n]
    return (value = jnum(_pf(last.Kp)), time_utc = jdt(_swpc_dt(last.time_tag)), trend = trend)
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

function swpc_snapshot()
    # Lock only to read the cache; build (blocking HTTP) outside it so one stalled SWPC
    # product cannot serialize every dashboard poller behind the mutex.
    fresh = lock(_SWPC_LOCK) do
        c = _SWPC_CACHE[]
        (c !== nothing && (time() - c[1]) <= SWPC_TTL) ? c[2] : nothing
    end
    fresh !== nothing && return fresh
    val = try
        _build_swpc_snapshot()
    catch e
        stale = lock(_SWPC_LOCK) do
            c = _SWPC_CACHE[]; c === nothing ? nothing : c[2]
        end
        if stale !== nothing
            @warn "SWPC snapshot build failed; serving cached" exception = e
            return stale
        end
        return (source = "NOAA SWPC", available = false, error = string(e))
    end
    lock(_SWPC_LOCK) do; _SWPC_CACHE[] = (time(), val); end
    return val
end

function swpc_snapshot_cached_or_refresh()
    c = lock(_SWPC_LOCK) do
        _SWPC_CACHE[]
    end
    if c !== nothing && (time() - c[1]) <= SWPC_TTL
        return c[2]
    end
    if !_SWPC_REFRESHING[]
        _SWPC_REFRESHING[] = true
        @async begin
            try
                swpc_snapshot()
            catch e
                @warn "async SWPC refresh failed" exception=e
            finally
                _SWPC_REFRESHING[] = false
            end
        end
    end
    if c !== nothing
        return c[2]
    end
    return (source = "NOAA SWPC", fetched_utc = jdt(now(UTC)), available = false,
            solar_wind = Dict{Symbol,Any}(:available => false), kp = nothing,
            scales = nothing, alerts = NamedTuple[])
end

# Honest threshold-based upstream indicator (NOT a fused scalar): flags "elevated" when any
# standard active-condition marker trips. Thresholds are conventional space-weather markers.
function upstream_assessment(snap)
    (snap === nothing || getproperty(snap, :available) == false) && return (available = false,)
    reasons = String[]; elevated = false
    g = snap.scales !== nothing ? snap.scales.G : nothing
    if g !== nothing && g != "0"; elevated = true; push!(reasons, "NOAA G$g geomagnetic storm"); end
    kpv = snap.kp !== nothing ? snap.kp.value : nothing
    if kpv !== nothing && kpv >= 5; elevated = true; push!(reasons, "Kp $(kpv) (storm level)"); end
    sw = snap.solar_wind
    bz = get(sw, :bz_gsm_nt, nothing); sp = get(sw, :speed_kms, nothing)
    if bz !== nothing && bz < -10; elevated = true; push!(reasons, "L1 Bz $(round(bz; digits=1)) nT southward"); end
    if sp !== nothing && sp > 600; elevated = true; push!(reasons, "L1 wind $(round(Int, sp)) km/s"); end
    return (available = true, elevated = elevated, reasons = reasons,
            g_scale = g, kp = kpv, bz_gsm_nt = bz, speed_kms = sp)
end
