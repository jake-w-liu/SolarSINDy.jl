# network.jl — multi-station dB/dt network nowcast for the GIC alert map.
#
# Fetches several USGS observatories concurrently and reports each station's current dB/dt +
# Pulkkinen tier + geographic coordinates (taken from the API's own metadata, not hardcoded),
# for a spatial alert map. Reuses the dB/dt convention and Pulkkinen tiers from dbdt.jl;
# depends on _fetch_usgs, _num, dbdt_tier, jdt_str (defined there).
#
# Robust: each station fetch degrades independently (a slow/missing station is omitted, not
# fatal); cached with a short TTL so the dashboard's polls don't hammer USGS.

const NET_TTL = 55.0
const NET_CACHE_MAX = 16
const _NET_CACHE = Dict{Tuple,Any}()             # station tuple => (fetch_time, results)
const _NET_LOCK = ReentrantLock()
# Latitude-spread US/territory observatories (low-lat -> auroral); missing ones are skipped.
const NET_STATIONS = ["SJG", "FRD", "BSL", "TUC", "BOU", "NEW", "CMO", "BRW"]

# Parse a fetched USGS JSON into a station brief (separated from the fetch so the parse logic
# is unit-testable when USGS is unreachable). Returns nothing on missing/short/gappy data.
function _station_parse(station::AbstractString, d; reference::DateTime=now(UTC))
    d === nothing && return nothing
    try
        times = d.times
        (times === nothing || length(times) < 2) && return nothing
        xv = nothing; yv = nothing
        for v in d.values
            el = String(v.metadata.element)
            el == "X" && (xv = v.values); el == "Y" && (yv = v.values)
        end
        (xv === nothing || yv === nothing) && return nothing
        n = length(times)
        dts = parse_dt.(times)
        (any(ismissing, dts) || any(i -> dts[i] <= dts[i-1], 2:n)) && return nothing
        dbdt = _dbdt_series(times, xv, yv)
        fin = [i for i in 1:n if isfinite(dbdt[i])]
        isempty(fin) && return nothing
        cur_i = last(fin)
        freshness = _source_freshness(times[cur_i], DBDT_MAX_AGE_MIN; reference=reference)
        freshness.stale && return nothing
        cur = dbdt[cur_i]
        cutoff = dts[cur_i] - Minute(30)
        recent = [dbdt[i] for i in fin if dts[i] > cutoff && dts[i] <= dts[cur_i]]
        mx = isempty(recent) ? cur : maximum(recent)
        coords = d.metadata.intermagnet.imo.coordinates    # [lon, lat, elev]
        name = String(d.metadata.intermagnet.imo.name)
        return (station = station, name = name,
                lon = Float64(coords[1]), lat = Float64(coords[2]),
                current_dbdt = round(cur; digits=2), max_dbdt = round(mx; digits=2),
                tier = dbdt_tier(mx), time_utc = jdt_str(times[cur_i]),
                age_minutes = freshness.age_min, stale = false, invalid_future = false)
    catch e
        e isa InterruptException && rethrow()
        @warn "station parse failed" station exception=e
        return nothing
    end
end

_station_brief(station::AbstractString) = _station_parse(station, _fetch_usgs(station, 40))

function _current_stations(rows; reference::DateTime=now(UTC))
    return [merge(row, (age_minutes=f.age_min, stale=false, invalid_future=false))
            for row in rows
            for f in (_source_freshness(get(row, :time_utc, nothing), DBDT_MAX_AGE_MIN;
                                        reference=reference),)
            if !f.stale]
end

function _merge_station_rows(fetched, cached, codes)
    selected = Dict{String,Tuple{Any,Bool}}()
    for (rows, from_cache) in ((cached, true), (fetched, false))
        for row in rows
            code = try
                String(get(row, :station, ""))
            catch e
                e isa InterruptException && rethrow()
                ""
            end
            code in codes || continue
            previous = get(selected, code, nothing)
            if previous === nothing
                selected[code] = (row, from_cache)
                continue
            end
            previous_time = _observation_time(previous[1], :time_utc)
            row_time = _observation_time(row, :time_utc)
            if previous_time === nothing ||
               (row_time !== nothing && row_time >= previous_time)
                selected[code] = (row, from_cache)
            end
        end
    end
    rows = Any[]
    used_cache = false
    for code in codes
        haskey(selected, code) || continue
        row, from_cache = selected[code]
        push!(rows, row)
        used_cache |= from_cache
    end
    return rows, used_cache
end

function usgs_network(; stations::Vector{String} = NET_STATIONS,
                      brief_fn=_station_brief, reference::DateTime=now(UTC))
    isempty(stations) && throw(ArgumentError("stations must not be empty"))
    length(stations) <= 32 || throw(ArgumentError("at most 32 stations are allowed"))
    codes = unique(_checked_station.(stations))
    key = Tuple(codes)
    # Lock only to read the cache; fetch all stations OUTSIDE it so one slow USGS station
    # cannot hold the mutex (and stall every poller) for the whole asyncmap timeout.
    cached_entry = lock(_NET_LOCK) do
        c = get(_NET_CACHE, key, nothing)
        c
    end
    st = cached_entry === nothing ? nothing :
         _current_stations(cached_entry[2]; reference=reference)
    cached_fresh = cached_entry !== nothing && (time() - cached_entry[1]) <= NET_TTL
    cached_fresh && !isempty(st) &&
        return (generated_utc=jdt(now(UTC)), available=true, stale=false, cached=true,
                n_stations=length(st), thresholds=collect(PULK), stations=st)
    fetched = try
            # Limited concurrency (ntasks=3): fast enough, but polite to USGS — firing all
            # stations at once triggers rate-limiting that fails every request.
            [r for r in asyncmap(brief_fn, codes; ntasks=min(3, length(codes))) if r !== nothing]
    catch e
        e isa InterruptException && rethrow()
        e isa CapturedException && e.ex isa InterruptException && throw(e.ex)
        @warn "network nowcast failed; serving cached" exception=e
        Any[]
    end
    fetched = _current_stations(fetched; reference=reference)
    used_cache = false
    if !isempty(fetched)
        st, used_cache = lock(_NET_LOCK) do
            latest_entry = get(_NET_CACHE, key, nothing)
            latest = latest_entry === nothing ? Any[] :
                     _current_stations(latest_entry[2]; reference=reference)
            merged, retained = _merge_station_rows(fetched, latest, codes)
            _bounded_time_cache_put!(_NET_CACHE, key, (time(), merged), NET_CACHE_MAX)
            (merged, retained)
        end
    else
        st = lock(_NET_LOCK) do
            latest_entry = get(_NET_CACHE, key, nothing)
            latest_entry === nothing ? Any[] :
                _current_stations(latest_entry[2]; reference=reference)
        end
        used_cache = !isempty(st)
    end
    available = !isempty(st)
    return (generated_utc = jdt(now(UTC)), available=available, stale=!available,
            cached=available && used_cache, n_stations = length(st),
            thresholds = collect(PULK), stations = st)
end
