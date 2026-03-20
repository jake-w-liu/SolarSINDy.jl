# Real-time solar wind data fetching from NOAA SWPC

using HTTP
using JSON3

const SWPC_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
const SWPC_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
const KYOTO_DST_URL = "https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/"

"""
    fetch_swpc_plasma(; url=SWPC_PLASMA_URL)

Fetch real-time solar wind plasma data (V, n, T) from NOAA SWPC.
Returns DataFrame with columns: time_tag, density, speed, temperature.
"""
function fetch_swpc_plasma(; url::String=SWPC_PLASMA_URL)
    resp = HTTP.get(url; connect_timeout=15, readtimeout=30)
    raw = JSON3.read(String(resp.body))
    rows = raw[2:end]  # skip header row

    df = DataFrame()
    df.time_tag = [DateTime(String(r[1]), dateformat"yyyy-mm-dd HH:MM:SS.sss") for r in rows]
    df.density = [_parse_swpc_float(r[2]) for r in rows]
    df.speed = [_parse_swpc_float(r[3]) for r in rows]
    df.temperature = [_parse_swpc_float(r[4]) for r in rows]

    return df
end

"""
    fetch_swpc_mag(; url=SWPC_MAG_URL)

Fetch real-time IMF data (Bx, By, Bz) from NOAA SWPC.
Returns DataFrame with columns: time_tag, bx_gsm, by_gsm, bz_gsm, bt.
"""
function fetch_swpc_mag(; url::String=SWPC_MAG_URL)
    resp = HTTP.get(url; connect_timeout=15, readtimeout=30)
    raw = JSON3.read(String(resp.body))
    rows = raw[2:end]  # skip header row

    df = DataFrame()
    df.time_tag = [DateTime(String(r[1]), dateformat"yyyy-mm-dd HH:MM:SS.sss") for r in rows]
    df.bx_gsm = [_parse_swpc_float(r[2]) for r in rows]
    df.by_gsm = [_parse_swpc_float(r[3]) for r in rows]
    df.bz_gsm = [_parse_swpc_float(r[4]) for r in rows]
    df.bt = [_parse_swpc_float(r[6]) for r in rows]

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
    fetch_realtime_solar_wind(; hours=168, cadence_minutes=60)

Fetch and merge real-time solar wind data from SWPC, averaged to hourly cadence.
Returns SolarWindData struct ready for forecasting.

Data covers the last `hours` hours (default 7 days).
"""
function fetch_realtime_solar_wind(; hours::Int=168)
    plasma = fetch_swpc_plasma()
    mag = fetch_swpc_mag()

    # Determine common time range
    t_start = max(minimum(plasma.time_tag), minimum(mag.time_tag))
    t_end = min(maximum(plasma.time_tag), maximum(mag.time_tag))

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
        mask_p = (plasma.time_tag .>= t0) .& (plasma.time_tag .< t1)
        if any(mask_p)
            vals_v = filter(!isnan, plasma.speed[mask_p])
            vals_n = filter(!isnan, plasma.density[mask_p])
            !isempty(vals_v) && (V_hr[i] = mean(vals_v))
            !isempty(vals_n) && (n_hr[i] = mean(vals_n))
        end

        # Mag averages
        mask_m = (mag.time_tag .>= t0) .& (mag.time_tag .< t1)
        if any(mask_m)
            vals_bz = filter(!isnan, mag.bz_gsm[mask_m])
            vals_by = filter(!isnan, mag.by_gsm[mask_m])
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
