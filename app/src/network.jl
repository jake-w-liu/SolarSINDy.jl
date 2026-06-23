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
const _NET_CACHE = Ref{Any}(nothing)            # (fetch_time, results)
const _NET_LOCK = ReentrantLock()
# Latitude-spread US/territory observatories (low-lat -> auroral); missing ones are skipped.
const NET_STATIONS = ["SJG", "FRD", "BSL", "TUC", "BOU", "NEW", "CMO", "BRW"]

# Parse a fetched USGS JSON into a station brief (separated from the fetch so the parse logic
# is unit-testable when USGS is unreachable). Returns nothing on missing/short/gappy data.
function _station_parse(station::AbstractString, d)
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
        n = length(times); dbdt = fill(NaN, n)
        for i in 2:n
            x1 = _num(xv[i]); x0 = _num(xv[i-1]); y1 = _num(yv[i]); y0 = _num(yv[i-1])
            (x1 === nothing || x0 === nothing || y1 === nothing || y0 === nothing) && continue
            dbdt[i] = sqrt((x1 - x0)^2 + (y1 - y0)^2)
        end
        fin = [i for i in 1:n if isfinite(dbdt[i])]
        isempty(fin) && return nothing
        cur = dbdt[last(fin)]; mx = maximum(dbdt[i] for i in fin)
        coords = d.metadata.intermagnet.imo.coordinates    # [lon, lat, elev]
        name = String(d.metadata.intermagnet.imo.name)
        return (station = station, name = name,
                lon = Float64(coords[1]), lat = Float64(coords[2]),
                current_dbdt = round(cur; digits=2), max_dbdt = round(mx; digits=2),
                tier = dbdt_tier(mx), time_utc = jdt_str(times[last(fin)]))
    catch e
        @warn "station parse failed" station exception=e
        return nothing
    end
end

_station_brief(station::AbstractString) = _station_parse(station, _fetch_usgs(station, 40))

function usgs_network(; stations::Vector{String} = NET_STATIONS)
    # Lock only to read the cache; fetch all stations OUTSIDE it so one slow USGS station
    # cannot hold the mutex (and stall every poller) for the whole asyncmap timeout.
    st = lock(_NET_LOCK) do
        c = _NET_CACHE[]
        (c !== nothing && (time() - c[1]) <= NET_TTL) ? c[2] : nothing
    end
    if st === nothing
        st = try
            # Limited concurrency (ntasks=3): fast enough, but polite to USGS — firing all
            # stations at once triggers rate-limiting that fails every request.
            fetched = [r for r in asyncmap(_station_brief, stations; ntasks=3) if r !== nothing]
            lock(_NET_LOCK) do; _NET_CACHE[] = (time(), fetched); end
            fetched
        catch e
            stale = lock(_NET_LOCK) do
                c = _NET_CACHE[]; c === nothing ? nothing : c[2]
            end
            stale === nothing && rethrow(e)
            @warn "network nowcast failed; serving cached" exception=e
            stale
        end
    end
    return (generated_utc = jdt_str(string(now(UTC))), n_stations = length(st),
            thresholds = collect(PULK), stations = st)
end
