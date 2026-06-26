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

function _swpc_get(path; readtimeout=15, connect_timeout=10)
    try
        r = HTTP.get(SWPC_BASE * path; readtimeout=readtimeout,
                     connect_timeout=connect_timeout, retries=1, status_exception=true)
        return JSON3.read(r.body)
    catch e
        @warn "SWPC fetch failed" path exception=e
        return nothing
    end
end

# --- individual products ---------------------------------------------------------------
# array-of-arrays with a header row: [time_tag, bx_gsm, by_gsm, bz_gsm, lon, lat, bt]
function swpc_solar_wind()
    sw = Dict{Symbol,Any}(:available => false)
    mag = _swpc_get("/products/solar-wind/mag-1-day.json")
    if mag !== nothing && length(mag) > 1
        h = String.(collect(mag[1])); idx = Dict(h[i] => i for i in eachindex(h))
        row = mag[end]
        sw[:bz_gsm_nt] = _swpc_row_field(idx, row, "bz_gsm")
        sw[:bt_nt]     = _swpc_row_field(idx, row, "bt")
        sw[:mag_time_utc] = jdt(_swpc_dt(row[1]))
        sw[:available] = true
    end
    pls = _swpc_get("/products/solar-wind/plasma-1-day.json")
    if pls !== nothing && length(pls) > 1
        h = String.(collect(pls[1])); idx = Dict(h[i] => i for i in eachindex(h))
        row = pls[end]
        sw[:speed_kms]   = _swpc_row_field(idx, row, "speed")
        sw[:density_cm3] = _swpc_row_field(idx, row, "density")
        sw[:plasma_time_utc] = jdt(_swpc_dt(row[1]))
    end
    return sw
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
