# server.jl — minimal, robust HTTP/JSON server for the Space-Weather Threat Monitor.
#
# Serves the static dashboard and a small JSON API over the locked-live forecast log.
# Built directly on HTTP.jl (already in the stack) — no web framework, no build step,
# single origin (dashboard + API share host:port, so no CORS gymnastics).
#
# Endpoints:
#   GET /                -> dashboard (public/index.html)
#   GET /<static>        -> public/<static>   (whitelisted extensions, traversal-guarded)
#   GET /api/health      -> log age + complete-cycle health
#   GET /api/status      -> current threat status, lead time, calibration
#   GET /api/forecast    -> latest forecast cycle (point + 90% band per horizon)
#   GET /api/history?hours=72 -> recent verified track record
#   GET /api/alerts      -> active alerts derived from status
#
# Config via environment:
#   SWM_HOST (default 127.0.0.1)   SWM_PORT (default 8723)
#   SOLARSINDY_LOG (the live forecast log; if unset, auto-discovered — see default_log)

using HTTP, JSON3, Dates

const APP_DIR    = normpath(joinpath(@__DIR__, ".."))      # SolarSINDy.jl/app
const PKG_ROOT   = normpath(joinpath(APP_DIR, ".."))       # the package root (SolarSINDy.jl)
const PUBLIC_DIR = joinpath(APP_DIR, "public")

include(joinpath(@__DIR__, "operational_paths.jl"))

function app_operational_evidence_dir(required_files::AbstractString...)
    output_override = strip(get(ENV, "SOLARSINDY_OPERATIONAL_OUTPUT_DIR", ""))
    output_dir = isempty(output_override) ?
        joinpath(PKG_ROOT, "validation", "output", "operational") : output_override
    return select_operational_evidence_dir(
        required_files...;
        output_dir=output_dir,
        paper_dir=joinpath(dirname(PKG_ROOT), "paper_v2_monitor", "data", "source", "operational"),
        package_dir=joinpath(PKG_ROOT, "data", "operational_validation"),
    )
end

include(joinpath(@__DIR__, "forecast_api.jl"))
include(joinpath(@__DIR__, "swpc.jl"))
include(joinpath(@__DIR__, "geoelectric.jl"))
include(joinpath(@__DIR__, "dbdt.jl"))
include(joinpath(@__DIR__, "forecaster.jl"))
include(joinpath(@__DIR__, "network.jl"))

function dashboard_dbdt_nowcast(; wait_timeout::Real=0.0)
    first_result = nothing
    for station in FORECASTER_STATIONS
        candidate = usgs_dbdt(; station=station, wait_timeout=wait_timeout)
        first_result === nothing && (first_result = candidate)
        get(candidate, :available, false) && return candidate
    end
    return first_result
end

include(joinpath(@__DIR__, "notify.jl"))

# Resolve the live forecast log: $SOLARSINDY_LOG if set, otherwise the package's
# runtime-state location. Request handlers degrade gracefully before the first row exists.
function default_log()
    env = get(ENV, "SOLARSINDY_LOG", "")
    isempty(env) || return env
    return joinpath(PKG_ROOT, "var", "monitor", "live_forecast_log.csv")
end

const _CT = Dict(".html"=>"text/html; charset=utf-8", ".js"=>"application/javascript; charset=utf-8",
                 ".css"=>"text/css; charset=utf-8", ".json"=>"application/json; charset=utf-8",
                 ".svg"=>"image/svg+xml", ".png"=>"image/png", ".ico"=>"image/x-icon",
                 ".woff2"=>"font/woff2", ".map"=>"application/json")
content_type(path) = get(_CT, lowercase(splitext(path)[2]), "application/octet-stream")

json_response(obj; status::Int=200) =
    HTTP.Response(status, ["Content-Type" => "application/json; charset=utf-8",
                           "Cache-Control" => "no-store"], JSON3.write(obj))

function _safe_queryparams(query::AbstractString)
    params = try
        HTTP.queryparams(query)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    all(isvalid(key) && isvalid(value) for (key, value) in params) || return nothing
    return params
end

function serve_static(path::AbstractString)
    rel = lstrip(path, '/')
    isempty(rel) && (rel = "index.html")
    (occursin("..", rel) || occursin('\0', rel)) && return HTTP.Response(403, "forbidden")
    file = normpath(joinpath(PUBLIC_DIR, rel))
    isfile(file) || return HTTP.Response(404, "not found")
    resolved = try
        realpath(file)
    catch e
        e isa InterruptException && rethrow()
        return HTTP.Response(404, "not found")
    end
    public_root = realpath(PUBLIC_DIR)
    relative = relpath(resolved, public_root)
    separator = string(Base.Filesystem.path_separator)
    confined = !isabspath(relative) && relative != ".." &&
               !startswith(relative, ".." * separator)
    (confined && isfile(resolved)) || return HTTP.Response(403, "forbidden")
    # Extension whitelist: only the known dashboard asset types are served. A stray file dropped
    # into public/ (editor backup, .DS_Store, a mis-copied config) is treated as "not found"
    # rather than leaked as application/octet-stream — this is the guard the endpoint documents.
    haskey(_CT, lowercase(splitext(resolved)[2])) || return HTTP.Response(404, "not found")
    return HTTP.Response(200, ["Content-Type" => content_type(resolved),
                              "Cache-Control" => "no-store"], read(resolved))
end

function api_handler(path::AbstractString, query::AbstractString, log_path::AbstractString)
    # The log is loaded lazily, only inside the branches that use it. /api/swpc, /api/dbdt,
    # /api/network, and /api/storm_replay have no log dependency and must stay up (and degrade
    # to their own available=false payloads) even when the forecast log is missing.
    if path == "/api/status"
        snap = swpc_snapshot_cached_or_refresh()
        return json_response(merge(build_status(get_log(log_path)),
                                   (upstream = snap, upstream_status = upstream_assessment(snap))))
    elseif path == "/api/swpc"
        # Dashboard requests must not inherit third-party latency. A cold or expired cache starts
        # exactly one background refresh and returns the last snapshot (or unavailable) immediately;
        # the front-end's next poll observes the completed refresh.
        return json_response(swpc_snapshot_cached_or_refresh())
    elseif path == "/api/dbdt"
        q = _safe_queryparams(query)
        q === nothing && return json_response(
            (available=false, error="malformed query"); status=400,
        )
        requested_station = get(q, "station", nothing)
        station = uppercase(requested_station === nothing ? first(FORECASTER_STATIONS) : requested_station)
        # Whitelist against the known observatory set: bounds the _DBDT_CACHE and the outbound
        # USGS request rate against arbitrary station codes. The bundled front-end never sends a
        # station param, and NET_STATIONS covers both forecaster models (FRD, CMO) and the map.
        if requested_station !== nothing && !(station in NET_STATIONS)
            return json_response((available=false, error="unsupported station"); status=400)
        end
        # The default dashboard selects the first available FRD/CMO ground nowcast so a
        # station-specific outage does not blank the panel. An explicit station query remains
        # exact and never falls back silently.
        nc = requested_station === nothing ?
             dashboard_dbdt_nowcast(; wait_timeout=0.0) :
             usgs_dbdt(; station=station, wait_timeout=0.0)
        station = String(get(nc, :station, station))
        # The retrospective artifact was fit with OMNI HRO drivers time-shifted to the
        # bow-shock nose. NOAA RTSW supplies unshifted L1 observations. Feeding the newest L1
        # value into that artifact changes the feature time reference, so the live forecast is
        # disabled until a tested L1-to-bow-shock alignment is available. The observed ground
        # nowcast remains valid and independent of this model-transfer boundary.
        forecast_status = (
            available=false,
            reason="disabled: live L1 drivers are not aligned to the bow-shock-shifted training features",
        )
        return json_response(merge(nc, (forecast=nothing,
                                        forecast_status=forecast_status)))
    elseif path == "/api/network"
        return json_response(usgs_network(; wait_timeout=0.0))
    elseif path == "/api/forecast"
        return json_response(build_forecast(get_log(log_path), log_path))
    elseif path == "/api/storm_replay"
        evidence_dir = app_operational_evidence_dir(
            "storm_replay_report.md", "storm_replay_scored.csv",
        )
        return json_response(build_storm_replay(log_path; evidence_dir=evidence_dir))
    elseif path == "/api/history"
        q = _safe_queryparams(query)
        q === nothing && return json_response(
            (available=false, error="malformed query"); status=400,
        )
        # Validate finiteness before clamping: parse(Float64, "NaN") succeeds and clamp(NaN,…)
        # returns NaN (never triggering the catch), which then throws in Hour(round(Int, NaN)).
        hours = try
            h = parse(Float64, get(q, "hours", "72"))
            isfinite(h) ? clamp(h, 1.0, 24.0 * 30) : 72.0
        catch e
            e isa InterruptException && rethrow()
            72.0
        end
        return json_response(build_history(get_log(log_path), hours))
    elseif path == "/api/alerts"
        df = get_log(log_path)
        snap = swpc_snapshot_cached_or_refresh()
        status = build_status(df)
        combined = compute_alert_state(
            status, upstream_assessment(snap),
            dashboard_dbdt_nowcast(; wait_timeout=0.0),
        )
        # An active outage sentinel forces the stale/unknown flag and adds an explicit reason, so a
        # dead or unloaded daemon (which the live layers cannot report) is not read as "quiet".
        outage = outage_state(log_path)
        reasons = String[combined.reasons...]
        getproperty(outage, :active) && push!(reasons,
            "forecast monitor outage: " *
            something(getproperty(outage, :summary), "issuance stopped (sentinel present)"))
        overall_stale = combined.stale || getproperty(outage, :active)
        return json_response(merge(build_alerts(df, status),
                                   (overall_level = combined.level, overall_reasons = reasons,
                                    overall_stale = overall_stale, overall_outage = outage)))
    else
        return json_response((error="unknown endpoint", path=path); status=404)
    end
end

# The issuance dead-man (in the daemon) and the external liveness watchdog both write an OUTAGE.md
# sentinel beside the forecast log when issuance stops or a process dies/unloads. Reading it here is
# what makes an outage visible on the dashboard even when the daemon is not running — the exact case
# the sentinel exists to cover, which nothing previously surfaced. `active` is the sentinel presence;
# the labeled Source/Detected UTC/Summary lines (emitted by both writers) are parsed best-effort.
function _outage_sentinel_path(log_path::AbstractString)
    joinpath(dirname(log_path), "OUTAGE.md")
end

function outage_state(log_path::AbstractString)
    sentinel = _outage_sentinel_path(log_path)
    isfile(sentinel) || return (active = false,)
    body = try
        read(sentinel, String)
    catch e
        e isa InterruptException && rethrow()
        return (active = true, source = nothing, detected_utc = nothing,
                summary = nothing, sentinel_age_min = nothing)
    end
    field(label) = begin
        m = match(Regex("(?m)^" * label * ":[ \t]*(.+?)[ \t]*\$"), body)
        m === nothing ? nothing : String(m.captures[1])
    end
    age_min = try
        round((time() - mtime(sentinel)) / 60; digits = 1)
    catch e
        e isa InterruptException && rethrow()
        nothing
    end
    return (active = true, source = field("Source"), detected_utc = field("Detected UTC"),
            summary = field("Summary"), sentinel_age_min = age_min)
end

function make_handler(log_path::AbstractString)
    return function (req::HTTP.Request)
        uri = HTTP.URI(req.target)
        path = uri.path
        try
            if path == "/api/health"
                ok = isfile(log_path)
                age = ok ? round((time() - mtime(log_path)) / 60; digits=1) : nothing
                cycle = ok ? try
                    latest_cycle(get_log(log_path))
                catch e
                    e isa InterruptException && rethrow()
                    DataFrame()
                end : DataFrame()
                cycle_complete = ok && _valid_live_cycle(cycle)
                cycle_state = ok && nrow(cycle) > 0 ? _cycle_staleness(cycle) : nothing
                cycle_stale = cycle_state !== nothing && (
                    cycle_state.stale || cycle_state.expired || cycle_state.invalid_future
                )
                outage = outage_state(log_path)
                # The daemon rewrites the log every cycle, so a file mtime older than the
                # staleness window means it has stopped issuing — report "stale" (dashboard dot
                # turns red). A recent file is still unhealthy when its latest issue hour lacks
                # the complete, internally consistent 1/2/3/6 h product cycle. An active outage
                # sentinel (dead-man trip, or the watchdog detecting a dead/unloaded process)
                # dominates: it reports "outage" even when the log file mtime still looks fresh.
                status = getproperty(outage, :active) ? "outage" :
                         !ok ? "no_log" :
                         ((age !== nothing && age > STALE_CYCLE_HOURS * 60) || cycle_stale) ?
                             "stale" :
                         !cycle_complete ? "incomplete" : "ok"
                return json_response((status = status,
                                      log_age_min=age,
                                      cycle_complete=cycle_complete,
                                      outage=outage,
                                      server_time_utc=string(now(UTC)) * "Z"))
            elseif startswith(path, "/api/")
                return api_handler(path, uri.query === nothing ? "" : uri.query, log_path)
            else
                return serve_static(path)
            end
        catch e
            e isa InterruptException && rethrow()
            # Log the full exception server-side, but never echo it (and its absolute log path)
            # to clients — return a generic detail.
            @error "request failed" path=path exception=(e, catch_backtrace())
            return json_response((error="internal error",); status=500)
        end
    end
end

"""
    start_server(; host, port, log_path, blocking=true)

Start the dashboard server. Returns the `HTTP.Server` when `blocking=false`.
"""
function start_server(; host::AbstractString = get(ENV, "SWM_HOST", "127.0.0.1"),
                        port::Integer = parse(Int, get(ENV, "SWM_PORT", "8723")),
                        log_path::AbstractString = default_log(),
                        blocking::Bool = true)
    handler = make_handler(log_path)
    url = "http://$(host):$(port)"
    println("""
    ┌────────────────────────────────────────────────────────────────────┐
    │  Space-Weather Threat Monitor — open-source forecasting dashboard    │
    ├────────────────────────────────────────────────────────────────────┤
    │  Dashboard : $(rpad(url, 53)) │
    │  Log       : $(rpad(isfile(log_path) ? "found" : "MISSING: $(log_path)", 53)) │
    │  API       : /api/status  /api/forecast  /api/history  /api/alerts   │
    └────────────────────────────────────────────────────────────────────┘
    """)
    isfile(log_path) || @warn "Forecast log not found; API will report unavailable until the daemon writes it." log_path
    start_notify_loop(log_path)          # no-op unless SWM_WEBHOOK_URL is set
    if blocking
        HTTP.serve(handler, host, port)
    else
        return HTTP.serve!(handler, host, port)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    start_server(; blocking=true)
end
