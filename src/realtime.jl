# Real-time solar wind data fetching from NOAA SWPC

using HTTP
using JSON3

# Real-time solar wind (RTSW) products. The former /products/solar-wind/{plasma,mag}-7-day.json
# header + array-of-arrays feeds were retired by SWPC (they now return HTTP 404). The live
# replacements are the RTSW 1-minute products, served as ARRAYS OF OBJECTS with named keys
# (never positional columns). Names kept as *_PLASMA_/*_MAG_ for call-site compatibility.
const SWPC_PLASMA_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
const SWPC_MAG_URL = "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"
const KYOTO_DST_JSON_URL = "https://services.swpc.noaa.gov/products/kyoto-dst.json"

const DEFAULT_SWPC_MAX_RETRIES = 3
const DEFAULT_SWPC_RETRY_DELAY_SEC = 1.0

# Dst fill-sentinel magnitude. Real Dst |x| ≪ 9000 nT (even great storms are a
# few hundred nT), so any |value| beyond this is a feed fill/sentinel (e.g. 9999)
# and must not anchor the forecaster. Matches the hardened live Kyoto parser.
const DST_SENTINEL_ABS = 9000.0

# Minimum finite 1-min samples for an hour bin to count as a measured hourly
# average. The RTSW feed is 1-min cadence (≈60 samples/hour), so 10 is ≈17%
# coverage: it rejects feed-brownout hours where 1–9 noisy minutes would
# otherwise masquerade as the whole hour's average (the model was trained on
# ≈60-min OMNI hourly means) while still admitting legitimately partial boundary
# hours (a live sample showed ≥13 samples in every real hour).
const MIN_HOURLY_DRIVER_SAMPLES = 10

function _fetch_swpc_json(url::String;
                          max_retries::Int=DEFAULT_SWPC_MAX_RETRIES,
                          retry_delay_sec::Real=DEFAULT_SWPC_RETRY_DELAY_SEC,
                          http_get::Function=HTTP.get,
                          fallback_url::Union{Nothing,String}=nothing)
    max_retries >= 1 || throw(ArgumentError("max_retries must be >= 1"))

    # Opt-in secondary vendor: only when fallback_url is supplied do we try a second
    # source after the primary exhausts its retries. Default nothing => unchanged behavior.
    urls = fallback_url === nothing ? (url,) : (url, fallback_url)
    last_error = nothing
    for (ui, u) in enumerate(urls)
        for attempt in 1:max_retries
            try
                resp = http_get(u; connect_timeout=15, readtimeout=30)
                status = getproperty(resp, :status)
                status == 200 || error("HTTP status $status")
                raw = JSON3.read(String(getproperty(resp, :body)))
                length(raw) >= 2 || error("response has no data rows")
                return raw
            catch e
                last_error = e
                if ui == length(urls) && attempt == max_retries
                    error(
                        "Failed to fetch SWPC JSON from $(join(urls, " then ")) " *
                        "after $max_retries attempts each: " * sprint(showerror, last_error)
                    )
                end
                sleep(retry_delay_sec)
            end
        end
    end
end

# --- RTSW named-key parsing helpers ---------------------------------------------------
# The RTSW feeds interleave records from several spacecraft (e.g. SOLAR1, ACE) that repeat a
# time_tag; `active=true` marks the currently-designated primary L1 source. We keep only active,
# physically-valid rows so a secondary/sentinel record cannot masquerade as the primary reading.

# `active` flag: true when the key is absent (schema surprise -> keep, so a change never silently
# drops every row) or explicitly boolean-true; any other encoding is treated as active.
function _rtsw_active(obj)::Bool
    (obj isa AbstractDict || obj isa JSON3.Object) || return true
    haskey(obj, :active) || return true
    a = obj[:active]
    a === nothing && return true
    return a isa Bool ? a : true
end

# Named-key numeric field parsed to Float64, or NaN when the key is missing/null/unparseable or
# (when finite bounds are given) outside the physical range — a generous schema guard that rejects
# sentinel/garbage fill without discarding real extreme-storm values.
function _rtsw_num(obj, key::Symbol; lo::Float64=-Inf, hi::Float64=Inf)::Float64
    ((obj isa AbstractDict || obj isa JSON3.Object) && haskey(obj, key)) || return NaN
    v = _parse_swpc_float(obj[key])
    (isfinite(v) && lo <= v <= hi) || return NaN
    return v
end

_rtsw_time(obj) =
    ((obj isa AbstractDict || obj isa JSON3.Object) && haskey(obj, :time_tag)) ?
    _parse_swpc_time(String(string(obj[:time_tag]))) : nothing

"""
    fetch_swpc_plasma(; url=SWPC_PLASMA_URL, max_retries=3)

Fetch real-time solar-wind plasma (V, n, T) from the NOAA SWPC RTSW wind product
(`rtsw_wind_1m.json`, an array of named-key objects). Keeps only the currently-active L1
source and rows whose speed and density are physically valid (both drive dynamic pressure).
Returns a chronologically sorted DataFrame with columns: time_tag, density, speed, temperature.
"""
function fetch_swpc_plasma(; url::String=SWPC_PLASMA_URL,
                             max_retries::Int=DEFAULT_SWPC_MAX_RETRIES,
                             retry_delay_sec::Real=DEFAULT_SWPC_RETRY_DELAY_SEC,
                             http_get::Function=HTTP.get,
                             fallback_url::Union{Nothing,String}=nothing)
    raw = _fetch_swpc_json(url;
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        http_get=http_get,
        fallback_url=fallback_url,
    )
    time_tag = DateTime[]; density = Float64[]; speed = Float64[]; temperature = Float64[]
    for obj in raw
        _rtsw_active(obj) || continue
        t = _rtsw_time(obj); t === nothing && continue
        V = _rtsw_num(obj, :proton_speed; lo=50.0, hi=5.0e3)
        n = _rtsw_num(obj, :proton_density; lo=0.0, hi=1.0e3)
        (isfinite(V) && isfinite(n)) || continue        # need both for Pdyn = k n V^2
        push!(time_tag, t); push!(speed, V); push!(density, n)
        push!(temperature, _rtsw_num(obj, :proton_temperature; lo=0.0, hi=1.0e9))
    end
    isempty(time_tag) && error("RTSW wind feed returned no active physically-valid rows")
    df = DataFrame(; time_tag, density, speed, temperature)
    sort!(df, :time_tag)                                 # RTSW is newest-first; downstream expects ascending
    return df
end

"""
    fetch_swpc_mag(; url=SWPC_MAG_URL, max_retries=3)

Fetch real-time IMF (Bx, By, Bz, Bt) from the NOAA SWPC RTSW magnetometer product
(`rtsw_mag_1m.json`, an array of named-key objects). Keeps only the currently-active L1
source and rows whose GSM Bz is physically valid; a null/out-of-range component is stored as
NaN (downstream driver averaging tolerates it). Returns a chronologically sorted DataFrame with
columns: time_tag, bx_gsm, by_gsm, bz_gsm, bt.
"""
function fetch_swpc_mag(; url::String=SWPC_MAG_URL,
                          max_retries::Int=DEFAULT_SWPC_MAX_RETRIES,
                          retry_delay_sec::Real=DEFAULT_SWPC_RETRY_DELAY_SEC,
                          http_get::Function=HTTP.get,
                          fallback_url::Union{Nothing,String}=nothing)
    raw = _fetch_swpc_json(url;
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        http_get=http_get,
        fallback_url=fallback_url,
    )
    time_tag = DateTime[]; bx_gsm = Float64[]; by_gsm = Float64[]; bz_gsm = Float64[]; bt = Float64[]
    for obj in raw
        _rtsw_active(obj) || continue
        t = _rtsw_time(obj); t === nothing && continue
        bz = _rtsw_num(obj, :bz_gsm; lo=-1.0e3, hi=1.0e3)
        isfinite(bz) || continue                          # Bz is the primary southward driver
        push!(time_tag, t); push!(bz_gsm, bz)
        push!(bx_gsm, _rtsw_num(obj, :bx_gsm; lo=-1.0e3, hi=1.0e3))
        push!(by_gsm, _rtsw_num(obj, :by_gsm; lo=-1.0e3, hi=1.0e3))
        push!(bt, _rtsw_num(obj, :bt; lo=0.0, hi=1.0e3))
    end
    isempty(time_tag) && error("RTSW mag feed returned no active physically-valid rows")
    df = DataFrame(; time_tag, bx_gsm, by_gsm, bz_gsm, bt)
    sort!(df, :time_tag)                                  # RTSW is newest-first; downstream expects ascending
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
        v = _parse_swpc_float(dv)
        # Reject null / numeric fill sentinels (e.g. 9999) so a feed outage cannot
        # anchor the monitor at a nonphysical Dst; real Dst |x| ≪ DST_SENTINEL_ABS.
        (isfinite(v) && abs(v) <= DST_SENTINEL_ABS) || continue
        push!(times, t)
        push!(dst, v)
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
value. Non-finite entries are dropped, and out-of-range fill sentinels
(`|Dst| > DST_SENTINEL_ABS`, e.g. 9999) are rejected with a warning so a
data-quality event cannot inject a nonphysical anchor into the forecast bins.
"""
function _hourly_dst_lookup(dst_times, dst_vals)
    lookup = Dict{DateTime,Float64}()
    n_rejected = 0
    for (t, v) in zip(dst_times, dst_vals)
        dt = t isa DateTime ? t : DateTime(String(t))
        val = Float64(v)
        if !isfinite(val)
            continue                       # legitimately missing hour
        elseif abs(val) > DST_SENTINEL_ABS
            n_rejected += 1                # numeric fill sentinel (e.g. 9999)
            continue
        end
        lookup[DateTime(year(dt), month(dt), day(dt), hour(dt))] = val
    end
    n_rejected > 0 &&
        @warn "dropped $n_rejected out-of-range Dst sentinel value(s) (|Dst| > $DST_SENTINEL_ABS)" maxlog=1
    return lookup
end

"""
    fetch_realtime_solar_wind(; hours=168, min_hourly_samples=MIN_HOURLY_DRIVER_SAMPLES)

Fetch and merge real-time solar wind data from SWPC, averaged to hourly cadence.
Returns SolarWindData struct ready for forecasting.

Data covers the last `hours` hours (default 7 days). An hour bin is treated as a
measured hourly average only when it holds at least `min_hourly_samples` finite
1-min samples for that driver; sparser bins are left as gaps (NaN) so a
feed-brownout hour is interpolated or refused rather than served as a one-minute
"average". Pass `min_hourly_samples=1` to reproduce the pre-gate averaging (used
by fixtures that inject coarse synthetic series).
"""
function fetch_realtime_solar_wind(; hours::Int=168,
                                    plasma::Union{Nothing,DataFrame}=nothing,
                                    mag::Union{Nothing,DataFrame}=nothing,
                                    dst::Union{Nothing,Tuple}=nothing,
                                    min_hourly_samples::Int=MIN_HOURLY_DRIVER_SAMPLES,
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

        # Plasma averages. A bin is only measured when it holds at least
        # min_hourly_samples finite 1-min samples; below that it stays NaN (a
        # gap), so a feed brownout is not served as a one-minute "hourly average".
        mask_p = (plasma_data.time_tag .>= t0) .& (plasma_data.time_tag .< t1)
        if any(mask_p)
            vals_v = filter(!isnan, plasma_data.speed[mask_p])
            vals_n = filter(!isnan, plasma_data.density[mask_p])
            length(vals_v) >= min_hourly_samples && (V_hr[i] = mean(vals_v))
            length(vals_n) >= min_hourly_samples && (n_hr[i] = mean(vals_n))
        end

        # Mag averages (same minimum-coverage gate).
        mask_m = (mag_data.time_tag .>= t0) .& (mag_data.time_tag .< t1)
        if any(mask_m)
            vals_bz = filter(!isnan, mag_data.bz_gsm[mask_m])
            vals_by = filter(!isnan, mag_data.by_gsm[mask_m])
            length(vals_bz) >= min_hourly_samples && (Bz_hr[i] = mean(vals_bz))
            length(vals_by) >= min_hourly_samples && (By_hr[i] = mean(vals_by))
        end

        # Observed Dst for this hour bin (anchoring)
        haskey(dst_lookup, t0) && (Dst_hr[i] = dst_lookup[t0])
    end

    # Fill short gaps (≤3 hours) in measured quantities first.
    for arr in [V_hr, n_hr, Bz_hr, By_hr]
        _interp_short_gaps!(arr, 3)
    end

    # Dynamic pressure recomputed proton-only from interpolated n, V so it keeps
    # the n*V^2 identity (do not interpolate Pdyn independently).
    Pdyn_hr = [dynamic_pressure(n_hr[i], V_hr[i]) for i in 1:n_bins]

    # Pressure-corrected Dst*. When Pdyn is missing (plasma outage) use a
    # physically-defensible pressure — carry the last known Pdyn forward over a
    # short outage, else the climatological quiet-time default — instead of the
    # old +11-only fallback (which implied Pdyn=0 and served outage-hour anchors
    # ~10 nT too shallow). NaN Dst bins leave the forecaster unanchored.
    Dst_star_hr = Vector{Float64}(undef, n_bins)
    last_pdyn = NaN
    last_pdyn_age = PDYN_CARRY_MAX_AGE_H + 1
    for i in 1:n_bins
        if isfinite(Pdyn_hr[i])
            last_pdyn = Pdyn_hr[i]
            last_pdyn_age = 0
        else
            last_pdyn_age = min(last_pdyn_age + 1, PDYN_CARRY_MAX_AGE_H + 1)
        end
        Dst_star_hr[i] = isnan(Dst_hr[i]) ? NaN :
            dst_to_dst_star(Dst_hr[i], resolve_pdyn(Pdyn_hr[i], last_pdyn, last_pdyn_age))
    end

    # Convert to hours from first timestamp
    t_hours = Float64[(t_hr[i] - t_hr[1]) / Hour(1) for i in 1:n_bins]

    # Third return is the newest actual common sample time (NOT the last hour-floored bin
    # start): the bins drop the trailing partial hour and are floored, so t_hr[end] lags real
    # time by up to ~2 h. Freshness/staleness must be measured from t_end; the floored t_hr
    # stays the Dst-anchor key and step time. (2-value callers ignore the extra by destructuring.)
    return SolarWindData(t_hours, V_hr, Bz_hr, By_hr, n_hr, Pdyn_hr,
                         Dst_hr, Dst_star_hr), t_hr, t_end
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

# ---------------------------------------------------------------------------
# Operational resilience helpers (testable offline; never hot-applied to the
# running daemon — adoption is left to the user's maintenance window).
# ---------------------------------------------------------------------------

"""
    recover_shadow_state(load_fn, bootstrap_fn)

Resilient persistent-state load. Return `load_fn()`; if it returns `nothing`
(state file absent) or throws (torn/corrupt file), warn and fall back to
`bootstrap_fn()`. Pure: callers inject the two thunks, so the recovery decision
is unit-testable without any filesystem or filter state.
"""
function recover_shadow_state(load_fn, bootstrap_fn)
    st = try
        load_fn()
    catch e
        @warn "state load failed; re-bootstrapping from history (live drift discarded)" exception=e
        nothing
    end
    return st === nothing ? bootstrap_fn() : st
end

"Default sustained-feed-failure threshold: consecutive failed fetch cycles before tripping."
const DEFAULT_FEED_DEADMAN_THRESHOLD = 6

"""
    feed_deadman_tripped(consecutive_failures; threshold=DEFAULT_FEED_DEADMAN_THRESHOLD)

Sustained-feed-failure dead-man's switch. Returns `true` once `consecutive_failures`
upstream fetch cycles reach `threshold`, signalling the forecaster should stop
issuing fresh forecasts rather than extrapolate on stale inputs. Pure predicate.
"""
function feed_deadman_tripped(consecutive_failures::Integer;
                              threshold::Integer=DEFAULT_FEED_DEADMAN_THRESHOLD)
    threshold >= 1 || throw(ArgumentError("threshold must be >= 1"))
    consecutive_failures >= 0 || throw(ArgumentError("consecutive_failures must be >= 0"))
    return consecutive_failures >= threshold
end
