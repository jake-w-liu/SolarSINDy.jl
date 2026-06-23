# server.jl — minimal, robust HTTP/JSON server for the Space-Weather Threat Monitor.
#
# Serves the static dashboard and a small JSON API over the locked-live forecast log.
# Built directly on HTTP.jl (already in the stack) — no web framework, no build step,
# single origin (dashboard + API share host:port, so no CORS gymnastics).
#
# Endpoints:
#   GET /                -> dashboard (public/index.html)
#   GET /<static>        -> public/<static>   (whitelisted extensions, traversal-guarded)
#   GET /api/health      -> liveness + log path/age
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

include(joinpath(@__DIR__, "forecast_api.jl"))
include(joinpath(@__DIR__, "swpc.jl"))
include(joinpath(@__DIR__, "geoelectric.jl"))
include(joinpath(@__DIR__, "dbdt.jl"))
include(joinpath(@__DIR__, "forecaster.jl"))
include(joinpath(@__DIR__, "network.jl"))
include(joinpath(@__DIR__, "notify.jl"))

# Resolve the live forecast log: $SOLARSINDY_LOG if set, else the first existing candidate
# (package-local, then a parent project's live_forecasts/ — so the bundled dashboard finds the
# live log automatically when the package is checked out inside the research project), else the
# package-local default path (the request handlers degrade gracefully when the file is absent).
function default_log()
    env = get(ENV, "SOLARSINDY_LOG", "")
    isempty(env) || return env
    candidates = [joinpath(PKG_ROOT, "live_forecasts", "live_forecast_log.csv"),
                  joinpath(PKG_ROOT, "..", "live_forecasts", "live_forecast_log.csv")]
    for c in candidates
        isfile(c) && return normpath(c)
    end
    return normpath(candidates[1])
end

const _CT = Dict(".html"=>"text/html; charset=utf-8", ".js"=>"application/javascript; charset=utf-8",
                 ".css"=>"text/css; charset=utf-8", ".json"=>"application/json; charset=utf-8",
                 ".svg"=>"image/svg+xml", ".png"=>"image/png", ".ico"=>"image/x-icon",
                 ".woff2"=>"font/woff2", ".map"=>"application/json")
content_type(path) = get(_CT, lowercase(splitext(path)[2]), "application/octet-stream")

json_response(obj; status::Int=200) =
    HTTP.Response(status, ["Content-Type" => "application/json; charset=utf-8",
                           "Cache-Control" => "no-store"], JSON3.write(obj))

function serve_static(path::AbstractString)
    rel = lstrip(path, '/')
    isempty(rel) && (rel = "index.html")
    (occursin("..", rel) || occursin('\0', rel)) && return HTTP.Response(403, "forbidden")
    file = normpath(joinpath(PUBLIC_DIR, rel))
    # Confine to PUBLIC_DIR.
    (startswith(file, PUBLIC_DIR) && isfile(file)) || return HTTP.Response(404, "not found")
    return HTTP.Response(200, ["Content-Type" => content_type(file),
                              "Cache-Control" => "no-cache"], read(file))
end

function api_handler(path::AbstractString, query::AbstractString, log_path::AbstractString)
    df = get_log(log_path)
    if path == "/api/status"
        snap = swpc_snapshot()
        return json_response(merge(build_status(df),
                                   (upstream = snap, upstream_status = upstream_assessment(snap))))
    elseif path == "/api/swpc"
        return json_response(swpc_snapshot())
    elseif path == "/api/dbdt"
        q = HTTP.queryparams(query)
        station = uppercase(get(q, "station", "FRD"))
        all(c -> 'A' <= c <= 'Z', station) || (station = "FRD")   # whitelist station codes
        nc = usgs_dbdt(; station=station)
        fc = nothing                                             # calibrated next-30-min forecast (paper3 conformal)
        if getproperty(nc, :available) == true
            sw = swpc_snapshot().solar_wind
            if sw !== nothing && get(sw, :available, false)
                fc = forecast_dbdt([s.dbdt for s in nc.series], get(sw, :speed_kms, nothing),
                                   get(sw, :bz_gsm_nt, nothing); station=station)
            end
        end
        return json_response(fc === nothing ? nc : merge(nc, (forecast = fc,)))
    elseif path == "/api/network"
        return json_response(usgs_network())
    elseif path == "/api/forecast"
        return json_response(build_forecast(df))
    elseif path == "/api/ekf_shadow"
        return json_response(build_ekf_shadow(log_path))
    elseif path == "/api/storm_replay"
        return json_response(build_storm_replay(log_path))
    elseif path == "/api/history"
        q = HTTP.queryparams(query)
        hours = try clamp(parse(Float64, get(q, "hours", "72")), 1, 24*30) catch; 72.0 end
        return json_response(build_history(df, hours))
    elseif path == "/api/alerts"
        snap = swpc_snapshot()
        combined = compute_alert_state(build_status(df), upstream_assessment(snap), usgs_dbdt())
        return json_response(merge(build_alerts(df),
                                   (overall_level = combined.level, overall_reasons = combined.reasons)))
    else
        return json_response((error="unknown endpoint", path=path); status=404)
    end
end

function make_handler(log_path::AbstractString)
    return function (req::HTTP.Request)
        uri = HTTP.URI(req.target)
        path = uri.path
        try
            if path == "/api/health"
                ok = isfile(log_path)
                age = ok ? round((time() - mtime(log_path)) / 60; digits=1) : nothing
                return json_response((status = ok ? "ok" : "no_log",
                                      log_path=log_path, log_age_min=age,
                                      server_time_utc=string(now(UTC)) * "Z"))
            elseif startswith(path, "/api/")
                return api_handler(path, uri.query === nothing ? "" : uri.query, log_path)
            else
                return serve_static(path)
            end
        catch e
            @error "request failed" path=path exception=(e, catch_backtrace())
            return json_response((error="internal error", detail=string(e)); status=500)
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
