# dbdt.jl — live dB/dt (geomagnetically-induced-current driver) nowcast from USGS real-time
# ground-magnetometer data. Matches paper3's convention exactly:
#   dB/dt = sqrt(dX^2 + dY^2) per 1-min step  [nT/min];  Pulkkinen et al. (2013) thresholds
#   18 / 42 / 66 / 90 nT/min.
#
# This is an honest *nowcast* of the currently observed ground dB/dt (the actual GIC driver),
# NOT the paper3 conformal forecast. Robust: graceful degradation if USGS is unreachable;
# short-TTL cache so frequent dashboard polls do not hammer the public service.
#
# Depends on jnum/jdt from forecast_api.jl (included before this file).

using HTTP, JSON3, Dates

const USGS_BASE = "https://geomag.usgs.gov/ws/data/"
const PULK = (18.0, 42.0, 66.0, 90.0)        # Pulkkinen 2013 dB/dt alert thresholds [nT/min]
const DBDT_TIERS = ("Quiet", "Active", "Strong", "Severe", "Extreme")
const DBDT_TTL = 55.0
const _DBDT_CACHE = Dict{String,Any}()       # station => (fetch_time, nowcast)
const _DBDT_LOCK = ReentrantLock()

function dbdt_tier(v)
    (v === nothing || !(v isa Real) || !isfinite(v)) && return (level = nothing, label = "—")
    v < PULK[1] && return (level = 0, label = DBDT_TIERS[1])
    v < PULK[2] && return (level = 1, label = DBDT_TIERS[2])
    v < PULK[3] && return (level = 2, label = DBDT_TIERS[3])
    v < PULK[4] && return (level = 3, label = DBDT_TIERS[4])
    return (level = 4, label = DBDT_TIERS[5])
end

_num(x) = (x === nothing || x === missing) ? nothing : (x isa Real ? Float64(x) : tryparse(Float64, String(x)))

function _fetch_usgs(station::AbstractString, minutes::Int)
    t1 = now(UTC); t0 = t1 - Minute(minutes)
    f(t) = Dates.format(t, "yyyy-mm-ddTHH:MM:SS") * "Z"
    url = string(USGS_BASE, "?id=", station, "&starttime=", f(t0), "&endtime=", f(t1),
                 "&elements=X,Y&format=json&sampling_period=60")
    try
        # Fail fast when USGS throttles or stalls. The dashboard is single-process, so
        # a slow third-party dB/dt nowcast must not block status/forecast endpoints.
        r = HTTP.get(url; readtimeout=3, connect_timeout=2, retries=0, status_exception=true)
        return JSON3.read(r.body)
    catch e
        @warn "USGS dB/dt fetch failed" station exception=e
        return nothing
    end
end

using Statistics: mean

# Linear-interpolate `nothing` gaps in a numeric vector -> Float64 vector (edge gaps held flat).
function _interp_gaps(v::Vector)
    n = length(v); out = Vector{Float64}(undef, n)
    valid = [x !== nothing && (x isa Real) && isfinite(x) for x in v]
    any(valid) || return nothing
    for i in 1:n; valid[i] && (out[i] = Float64(v[i])); end
    f = findfirst(valid); l = findlast(valid)
    for i in 1:f-1; out[i] = out[f]; end
    for i in l+1:n; out[i] = out[l]; end
    i = f
    while i <= l
        if valid[i]; i += 1; continue; end
        j = i; while j <= l && !valid[j]; j += 1; end       # gap [i, j-1]; out[i-1], out[j] valid
        for k in i:j-1; out[k] = out[i-1] + (out[j]-out[i-1])*(k-(i-1))/(j-(i-1)); end
        i = j
    end
    return out
end

_detrend(y::Vector{Float64}) = begin                          # remove best-fit line (reduce leakage)
    n = length(y); t = collect(0.0:n-1); tm = mean(t); ym = mean(y)
    b = sum((t .- tm) .* (y .- ym)) / max(sum((t .- tm).^2), eps())
    y .- ((ym - b*tm) .+ b .* t)
end

# Geoelectric-field nowcast (plane-wave, 1-D half-space) on the recent window; nothing if too gappy.
function _geoe_nowcast(xv, yv, dt_s; rho=1000.0, window=120)
    n = length(xv); s = max(1, n - window + 1)
    xs = [_num(xv[i]) for i in s:n]; ys = [_num(yv[i]) for i in s:n]
    count(i -> xs[i] !== nothing && ys[i] !== nothing, eachindex(xs)) < 0.9 * length(xs) && return nothing
    xf = _interp_gaps(xs); yf = _interp_gaps(ys)
    (xf === nothing || yf === nothing || length(xf) < 16) && return nothing
    ex, ey = geoelectric_field(_detrend(xf), _detrend(yf), dt_s; rho_ohm_m=rho)
    emag = sqrt.(ex.^2 .+ ey.^2); m = length(emag)
    # The circular DFT/IDFT concentrates wraparound ringing at the window edges, so the raw
    # endpoint emag[end] is edge-contaminated (biased low on a rising ramp). Serve the last
    # edge-trimmed sample as "current", and take the reported max over the trailing ~30 interior
    # samples (the "30-min max" the dashboard shows), excluding the 3 edge-ringing samples.
    hi = max(1, m - 3)
    inner = emag[max(min(4, m), m - 32):hi]
    return (current = emag[hi], max = isempty(inner) ? maximum(emag) : maximum(inner), rho = rho)
end

function _compute_dbdt(station::AbstractString, minutes::Int)
    d = _fetch_usgs(station, minutes)
    d === nothing && return (station=station, available=false)
    times = get(d, :times, nothing)
    (times === nothing || length(times) < 2) && return (station=station, available=false)
    xv = nothing; yv = nothing
    for v in d.values
        el = String(v.metadata.element)
        el == "X" && (xv = v.values); el == "Y" && (yv = v.values)
    end
    (xv === nothing || yv === nothing) && return (station=station, available=false)

    n = length(times)
    dbdt = fill(NaN, n)
    for i in 2:n
        x1 = _num(xv[i]); x0 = _num(xv[i-1]); y1 = _num(yv[i]); y0 = _num(yv[i-1])
        (x1 === nothing || x0 === nothing || y1 === nothing || y0 === nothing) && continue
        dbdt[i] = sqrt((x1 - x0)^2 + (y1 - y0)^2)
    end
    finite_idx = [i for i in 1:n if isfinite(dbdt[i])]
    isempty(finite_idx) && return (station=station, available=false)

    cur_i = last(finite_idx); current = dbdt[cur_i]
    w30 = [dbdt[i] for i in finite_idx if i > n - 30]       # last 30 min (exactly 30 samples)
    max30 = isempty(w30) ? current : maximum(w30)
    # exceedance counts in the returned window
    valid = dbdt[finite_idx]
    exceed = [(threshold = Int(thr), count = count(>=(thr), valid)) for thr in PULK]
    # series for plotting (finite points only, last ~60 min)
    keep = [i for i in finite_idx if i >= n - 60]
    series = [(t = jdt_str(times[i]), dbdt = round(dbdt[i]; digits=2)) for i in keep]

    geoe = nothing
    try; geoe = _geoe_nowcast(xv, yv, 60.0); catch e; @warn "geoE nowcast failed" exception=e; end

    ct = dbdt_tier(current); mt = dbdt_tier(max30)
    return (station = station, available = true,
            current_dbdt = round(current; digits=2), current_tier = ct,
            current_time_utc = jdt_str(times[cur_i]),
            max30_dbdt = round(max30; digits=2), max30_tier = mt,
            thresholds = collect(PULK), exceedances = exceed,
            geoelectric = geoe === nothing ? nothing :
                (current_vkm = round(geoe.current; digits=3), max_vkm = round(geoe.max; digits=3),
                 tier = geo_tier(geoe.max), rho_ohm_m = geoe.rho,
                 note = "1-D uniform half-space estimate"),
            n_minutes = length(keep), series = series)
end

# USGS times look like "2026-06-20T11:04:00.000Z"; normalize to "...Z" without millis
jdt_str(s) = s === nothing ? nothing : (string(first(String(s), 19)) * "Z")

function usgs_dbdt(; station::AbstractString = "FRD", minutes::Int = 120)
    # Hold the lock only to read the cache; fetch OUTSIDE it so one slow/hanging upstream
    # cannot serialize every concurrent poller behind the mutex for the full timeout.
    fresh = lock(_DBDT_LOCK) do
        c = get(_DBDT_CACHE, station, nothing)
        (c !== nothing && (time() - c[1]) <= DBDT_TTL) ? c[2] : nothing
    end
    fresh !== nothing && return fresh
    val = try
        _compute_dbdt(station, minutes)
    catch e
        stale = lock(_DBDT_LOCK) do
            c = get(_DBDT_CACHE, station, nothing); c === nothing ? nothing : c[2]
        end
        if stale !== nothing
            @warn "dB/dt nowcast failed; serving cached" station exception=e
            return stale
        end
        return (station=station, available=false, error=string(e))
    end
    lock(_DBDT_LOCK) do; _DBDT_CACHE[station] = (time(), val); end
    return val
end
