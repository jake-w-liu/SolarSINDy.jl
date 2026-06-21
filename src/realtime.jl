# Real-time solar wind data fetching from NOAA SWPC

using HTTP
using JSON3

const SWPC_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
const SWPC_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
const KYOTO_DST_JSON_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"

const DEFAULT_SWPC_MAX_RETRIES = 3
const DEFAULT_SWPC_RETRY_DELAY_SEC = 1.0

function _fetch_swpc_json(url::String;
                          max_retries::Int=DEFAULT_SWPC_MAX_RETRIES,
                          retry_delay_sec::Real=DEFAULT_SWPC_RETRY_DELAY_SEC,
                          http_get::Function=HTTP.get)
    max_retries >= 1 || throw(ArgumentError("max_retries must be >= 1"))

    last_error = nothing
    for attempt in 1:max_retries
        try
            resp = http_get(url; connect_timeout=15, readtimeout=30)
            status = getproperty(resp, :status)
            status == 200 || error("HTTP status $status")
            raw = JSON3.read(String(getproperty(resp, :body)))
            length(raw) >= 2 || error("response has no data rows")
            return raw
        catch e
            last_error = e
            attempt == max_retries && error(
                "Failed to fetch SWPC JSON from $url after $max_retries attempts: " *
                sprint(showerror, last_error)
            )
            sleep(retry_delay_sec)
        end
    end
end

function _require_swpc_columns(row, ncols::Int, feed::String)
    length(row) >= ncols || error("$feed row has $(length(row)) columns; expected at least $ncols")
    return row
end

"""
    fetch_swpc_plasma(; url=SWPC_PLASMA_URL, max_retries=3)

Fetch real-time solar wind plasma data (V, n, T) from NOAA SWPC.
Returns DataFrame with columns: time_tag, density, speed, temperature.
"""
function fetch_swpc_plasma(; url::String=SWPC_PLASMA_URL,
                             max_retries::Int=DEFAULT_SWPC_MAX_RETRIES,
                             retry_delay_sec::Real=DEFAULT_SWPC_RETRY_DELAY_SEC,
                             http_get::Function=HTTP.get)
    raw = _fetch_swpc_json(url;
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        http_get=http_get,
    )
    rows = raw[2:end]  # skip header row

    df = DataFrame()
    plasma_rows = [_require_swpc_columns(r, 4, "plasma") for r in rows]
    df.time_tag = [DateTime(String(r[1]), dateformat"yyyy-mm-dd HH:MM:SS.sss") for r in plasma_rows]
    df.density = [_parse_swpc_float(r[2]) for r in plasma_rows]
    df.speed = [_parse_swpc_float(r[3]) for r in plasma_rows]
    df.temperature = [_parse_swpc_float(r[4]) for r in plasma_rows]

    return df
end

"""
    fetch_swpc_mag(; url=SWPC_MAG_URL, max_retries=3)

Fetch real-time IMF data (Bx, By, Bz) from NOAA SWPC.
Returns DataFrame with columns: time_tag, bx_gsm, by_gsm, bz_gsm, bt.
"""
function fetch_swpc_mag(; url::String=SWPC_MAG_URL,
                          max_retries::Int=DEFAULT_SWPC_MAX_RETRIES,
                          retry_delay_sec::Real=DEFAULT_SWPC_RETRY_DELAY_SEC,
                          http_get::Function=HTTP.get)
    raw = _fetch_swpc_json(url;
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        http_get=http_get,
    )
    rows = raw[2:end]  # skip header row

    df = DataFrame()
    mag_rows = [_require_swpc_columns(r, 7, "mag") for r in rows]
    df.time_tag = [DateTime(String(r[1]), dateformat"yyyy-mm-dd HH:MM:SS.sss") for r in mag_rows]
    df.bx_gsm = [_parse_swpc_float(r[2]) for r in mag_rows]
    df.by_gsm = [_parse_swpc_float(r[3]) for r in mag_rows]
    df.bz_gsm = [_parse_swpc_float(r[4]) for r in mag_rows]
    df.bt = [_parse_swpc_float(r[7]) for r in mag_rows]

    return df
end

function _parse_swpc_float(v)
    v === nothing && return NaN
    s = String(string(v))
    (isempty(s) || s == "null") && return NaN
    val = tryparse(Float64, s)
    return val === nothing ? NaN : val
end

"""
    fetch_swpc_dst(; url=KYOTO_DST_JSON_URL, max_retries=3)

Fetch the hourly Kyoto Dst index (provisional/quicklook, served by NOAA SWPC).
Returns `(times::Vector{DateTime}, dst::Vector{Float64})`. This is the observed
geomagnetic state used to anchor the real-time forecaster, so the monitor does
not free-run from an arbitrary Dst*=0 initial condition.
"""
function fetch_swpc_dst(; url::String=KYOTO_DST_JSON_URL,
                          max_retries::Int=DEFAULT_SWPC_MAX_RETRIES,
                          retry_delay_sec::Real=DEFAULT_SWPC_RETRY_DELAY_SEC,
                          http_get::Function=HTTP.get)
    raw = _fetch_swpc_json(url;
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        http_get=http_get,
    )
    # The Kyoto Dst product is served as an array of OBJECTS with ISO-8601 timestamps,
    #   [{"time_tag":"2026-06-14T05:00:00","dst":-15}, ...],
    # NOT the header + array-of-arrays format the plasma/mag feeds use. Parse the object
    # form, while tolerating a legacy array-of-arrays form (with or without a header row)
    # and both the ISO `T` and space timestamp separators.
    times = DateTime[]
    dst = Float64[]
    for r in raw
        tt = nothing; dv = nothing
        if r isa AbstractVector
            length(r) < 2 && continue
            String(string(r[1])) == "time_tag" && continue   # skip a header row if present
            tt = r[1]; dv = r[2]
        else                                                  # object form (JSON3.Object/Dict)
            tt = get(r, :time_tag, get(r, "time_tag", nothing))
            dv = get(r, :dst, get(r, "dst", nothing))
        end
        tt === nothing && continue
        t = _parse_swpc_time(String(string(tt)))
        t === nothing && continue
        push!(times, t)
        push!(dst, _parse_swpc_float(dv))
    end
    return times, dst
end

"""
    _parse_swpc_time(s) -> Union{DateTime,Nothing}

Parse an SWPC/Kyoto timestamp, tolerating the ISO-8601 `T` separator
(`2026-06-14T05:00:00`), the space separator (`2026-06-14 05:00:00`), and an
optional fractional-second suffix. Returns `nothing` if unparseable.
"""
function _parse_swpc_time(s::AbstractString)
    str = String(s)
    base = split(str, '.')[1]                # drop any fractional-second suffix
    for fmt in (dateformat"yyyy-mm-ddTHH:MM:SS", dateformat"yyyy-mm-dd HH:MM:SS")
        t = tryparse(DateTime, base, fmt)
        t !== nothing && return t
    end
    return tryparse(DateTime, str)            # last resort: default ISO constructor
end

"""
    _hourly_dst_lookup(dst_times, dst_vals)

Build a Dict mapping each observed Dst timestamp (floored to the hour) to its
value, dropping non-finite entries. Used to anchor hourly forecast bins.
"""
function _hourly_dst_lookup(dst_times, dst_vals)
    lookup = Dict{DateTime,Float64}()
    for (t, v) in zip(dst_times, dst_vals)
        dt = t isa DateTime ? t : DateTime(String(t))
        val = Float64(v)
        isfinite(val) && (lookup[DateTime(year(dt), month(dt), day(dt), hour(dt))] = val)
    end
    return lookup
end

"""
    fetch_realtime_solar_wind(; hours=168)

Fetch and merge real-time solar wind data from SWPC, averaged to hourly cadence.
Returns SolarWindData struct ready for forecasting.

Data covers the last `hours` hours (default 7 days).
"""
function fetch_realtime_solar_wind(; hours::Int=168,
                                    plasma::Union{Nothing,DataFrame}=nothing,
                                    mag::Union{Nothing,DataFrame}=nothing,
                                    dst::Union{Nothing,Tuple}=nothing,
                                    max_retries::Int=DEFAULT_SWPC_MAX_RETRIES,
                                    retry_delay_sec::Real=DEFAULT_SWPC_RETRY_DELAY_SEC,
                                    http_get::Function=HTTP.get)
    plasma_data = plasma === nothing ? fetch_swpc_plasma(;
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        http_get=http_get,
    ) : plasma
    mag_data = mag === nothing ? fetch_swpc_mag(;
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        http_get=http_get,
    ) : mag
    # Observed Dst lookup for anchoring (hour-keyed). When absent, the
    # forecaster runs unanchored (Dst* stays NaN) — but the monitor supplies it.
    dst_lookup = dst === nothing ? Dict{DateTime,Float64}() :
        _hourly_dst_lookup(dst[1], dst[2])

    # Determine common time range
    t_start = max(minimum(plasma_data.time_tag), minimum(mag_data.time_tag))
    t_end = min(maximum(plasma_data.time_tag), maximum(mag_data.time_tag))

    # Truncate to requested hours
    t_start = max(t_start, t_end - Hour(hours))

    # Floor the binning grid to the hour so bin starts (t_edges[i], used as the
    # Dst lookup key below) coincide with the hour-floored keys produced by
    # _hourly_dst_lookup. Live SWPC feeds are 1-min cadence, so the earliest
    # t_start is generically not on :00; without this floor the [t0,t1) bins
    # straddle hour boundaries and the observed-Dst anchor never matches.
    t_start = floor(t_start, Hour(1))

    # Create hourly bins
    t_edges = collect(t_start:Hour(1):t_end)
    n_bins = length(t_edges) - 1
    n_bins < 1 && error("Insufficient data: only $(t_end - t_start) available")

    # Hourly averages
    V_hr = fill(NaN, n_bins)
    n_hr = fill(NaN, n_bins)
    Bz_hr = fill(NaN, n_bins)
    By_hr = fill(NaN, n_bins)
    Dst_hr = fill(NaN, n_bins)
    t_hr = Vector{DateTime}(undef, n_bins)

    for i in 1:n_bins
        t0, t1 = t_edges[i], t_edges[i+1]
        t_hr[i] = t0

        # Plasma averages
        mask_p = (plasma_data.time_tag .>= t0) .& (plasma_data.time_tag .< t1)
        if any(mask_p)
            vals_v = filter(!isnan, plasma_data.speed[mask_p])
            vals_n = filter(!isnan, plasma_data.density[mask_p])
            !isempty(vals_v) && (V_hr[i] = mean(vals_v))
            !isempty(vals_n) && (n_hr[i] = mean(vals_n))
        end

        # Mag averages
        mask_m = (mag_data.time_tag .>= t0) .& (mag_data.time_tag .< t1)
        if any(mask_m)
            vals_bz = filter(!isnan, mag_data.bz_gsm[mask_m])
            vals_by = filter(!isnan, mag_data.by_gsm[mask_m])
            !isempty(vals_bz) && (Bz_hr[i] = mean(vals_bz))
            !isempty(vals_by) && (By_hr[i] = mean(vals_by))
        end

        # Observed Dst for this hour bin (anchoring)
        haskey(dst_lookup, t0) && (Dst_hr[i] = dst_lookup[t0])
    end

    # Fill short gaps (≤3 hours) in measured quantities first.
    for arr in [V_hr, n_hr, Bz_hr, By_hr]
        _interp_short_gaps!(arr, 3)
    end

    # Dynamic pressure recomputed from interpolated n, V so it keeps the
    # n*V^2 identity (do not interpolate Pdyn independently).
    Pdyn_hr = [isnan(n_hr[i]) || isnan(V_hr[i]) ? NaN :
               1.6726e-6 * n_hr[i] * V_hr[i]^2 for i in 1:n_bins]

    # Pressure-corrected Dst* where an observed Dst and Pdyn are both available;
    # NaN bins leave the forecaster unanchored for that step.
    Dst_star_hr = [isnan(Dst_hr[i]) ? NaN :
                   (isnan(Pdyn_hr[i]) ? Dst_hr[i] + 11.0 :
                    Dst_hr[i] - 7.26 * sqrt(max(Pdyn_hr[i], 0.0)) + 11.0)
                   for i in 1:n_bins]

    # Convert to hours from first timestamp
    t_hours = Float64[(t_hr[i] - t_hr[1]) / Hour(1) for i in 1:n_bins]

    return SolarWindData(t_hours, V_hr, Bz_hr, By_hr, n_hr, Pdyn_hr,
                         Dst_hr, Dst_star_hr), t_hr
end

"""
    _interp_short_gaps!(x, max_gap)

In-place linear interpolation of NaN gaps ≤ max_gap points.
"""
function _interp_short_gaps!(x::Vector{Float64}, max_gap::Int)
    n = length(x)
    i = 1
    while i <= n
        if isnan(x[i])
            j = i
            while j <= n && isnan(x[j])
                j += 1
            end
            gap_len = j - i
            if gap_len <= max_gap && i > 1 && j <= n
                for k in i:j-1
                    frac = (k - i + 1) / (gap_len + 1)
                    x[k] = x[i-1] + frac * (x[j] - x[i-1])
                end
            end
            i = j
        else
            i += 1
        end
    end
end
