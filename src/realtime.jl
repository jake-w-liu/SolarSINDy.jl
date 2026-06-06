# Real-time solar wind data fetching from NOAA SWPC

using HTTP
using JSON3

const SWPC_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
const SWPC_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
const KYOTO_DST_URL = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/"

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
    fetch_realtime_solar_wind(; hours=168)

Fetch and merge real-time solar wind data from SWPC, averaged to hourly cadence.
Returns SolarWindData struct ready for forecasting.

Data covers the last `hours` hours (default 7 days).
"""
function fetch_realtime_solar_wind(; hours::Int=168,
                                    plasma::Union{Nothing,DataFrame}=nothing,
                                    mag::Union{Nothing,DataFrame}=nothing,
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

    # Determine common time range
    t_start = max(minimum(plasma_data.time_tag), minimum(mag_data.time_tag))
    t_end = min(maximum(plasma_data.time_tag), maximum(mag_data.time_tag))

    # Truncate to requested hours
    t_start = max(t_start, t_end - Hour(hours))

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
    end

    # Compute derived quantities
    Pdyn_hr = [isnan(n_hr[i]) || isnan(V_hr[i]) ? NaN :
               1.6726e-6 * n_hr[i] * V_hr[i]^2 for i in 1:n_bins]

    # Pressure-corrected Dst* (Dst not available from SWPC in real-time;
    # use NaN — the forecaster will run in prediction mode without anchoring)
    Dst_star_hr = fill(NaN, n_bins)

    # Fill short gaps (≤3 hours) via linear interpolation
    for arr in [V_hr, n_hr, Bz_hr, By_hr, Pdyn_hr]
        _interp_short_gaps!(arr, 3)
    end

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
