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

# Response hardening carried by every response this server emits.
#
# The dashboard renders third-party feed text (NOAA product identifiers and alert summaries, the
# NOAA scale token) and log-written text (served labels, interval sources, station codes). Those are
# escaped where they are rendered; the policy here is the second line, bounding what injected markup
# could reach even if a sink were missed. Every source is the page's own origin: the scripts, the
# stylesheet, and the API the page fetches.
#
# `style-src` admits inline styles deliberately. The panels carry per-element style attributes for
# the threat-tier colours, and the plotting library writes its own style rules into the document at
# runtime; refusing them would leave the page rendered without its colour coding and without plots.
# `script-src` admits the plotting CDN because the page falls back to it when no vendored copy is
# present, which is the documented offline-first behaviour of this deployment.
#
# `connect-src 'self'` is the whole fetch surface, and it governs one request the page does not
# issue itself: the plotting library fetches the network panel's base map (a topojson file for the
# requested scope and resolution) at draw time, from its own default source, the plotting CDN. That
# request is not something an offline deployment could serve either, so the map is vendored beside
# the bundle (`public/vendor/topojson/`) and `app.js` points the library at this origin before the
# first plot is drawn. The policy therefore stays closed and the map is drawn from the same origin
# as the rest of the page. The pairing is asserted in the app test suite against the shipped bundle,
# so a bundle upgrade that changes the default source or the map name cannot pass silently.
const CONTENT_SECURITY_POLICY = join([
    "default-src 'self'",
    "base-uri 'none'",
    "object-src 'none'",
    "frame-ancestors 'none'",
    "form-action 'none'",
    # `blob:` is required for the plot toolbar's PNG export: the plotting library renders the chart
    # into a blob URL and loads it as an image before handing the file to the browser. Without it the
    # download silently fails with the library's own "problem downloading your snapshot" notice.
    "img-src 'self' data: blob:",
    "style-src 'self' 'unsafe-inline'",
    "script-src 'self' https://cdn.plot.ly",
    "connect-src 'self'",
], "; ")

const SECURITY_HEADERS = [
    "Content-Security-Policy" => CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options" => "nosniff",
    "Referrer-Policy" => "no-referrer",
]

hardened_response(status::Int, headers, body) =
    HTTP.Response(status, vcat(SECURITY_HEADERS, collect(headers)), body)

json_response(obj; status::Int=200) =
    hardened_response(status, ["Content-Type" => "application/json; charset=utf-8",
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
    (occursin("..", rel) || occursin('\0', rel)) &&
        return hardened_response(403, [], "forbidden")
    file = normpath(joinpath(PUBLIC_DIR, rel))
    isfile(file) || return hardened_response(404, [], "not found")
    resolved = try
        realpath(file)
    catch e
        e isa InterruptException && rethrow()
        return hardened_response(404, [], "not found")
    end
    public_root = realpath(PUBLIC_DIR)
    relative = relpath(resolved, public_root)
    separator = string(Base.Filesystem.path_separator)
    confined = !isabspath(relative) && relative != ".." &&
               !startswith(relative, ".." * separator)
    (confined && isfile(resolved)) || return hardened_response(403, [], "forbidden")
    # Extension whitelist: only the known dashboard asset types are served. A stray file dropped
    # into public/ (editor backup, .DS_Store, a mis-copied config) is treated as "not found"
    # rather than leaked as application/octet-stream — this is the guard the endpoint documents.
    haskey(_CT, lowercase(splitext(resolved)[2])) || return hardened_response(404, [], "not found")
    return hardened_response(200, ["Content-Type" => content_type(resolved),
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
        # An active outage sentinel forces the stale/unknown flag and adds an explicit reason, so a
        # dead or unloaded daemon (which the live layers cannot report) is not read as "quiet".
        # `compose_alert_state` is the same rule the webhook loop delivers on, so the endpoint and
        # the notification cannot disagree about whether the monitor is blind. The persisted
        # first-data marker is restored once per process for the same reason: without it a restart
        # into an outage would report the cycle-less log as a deployment that never served data.
        _ALERT_STATE_LOADED[] || _load_alert_state!(_default_alert_state_path(log_path))
        outage = outage_state(log_path)
        combined = compose_alert_state(status, upstream_assessment(snap),
                                       dashboard_dbdt_nowcast(; wait_timeout=0.0), outage)
        return json_response(merge(build_alerts(df, status),
                                   (overall_level = combined.level,
                                    overall_reasons = String[combined.reasons...],
                                    overall_stale = combined.stale, overall_outage = outage)))
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
        # This server reads: it publishes a log the daemon writes and holds no state a request may
        # change. The method was previously ignored, so a POST, PUT or DELETE was answered with the
        # GET body and a 200 — a write that reads as accepted. Only retrieval is answered.
        method = uppercase(String(req.method))
        if method != "GET" && method != "HEAD"
            return hardened_response(405, ["Allow" => "GET, HEAD",
                                           "Content-Type" => "application/json; charset=utf-8",
                                           "Cache-Control" => "no-store"],
                                     JSON3.write((error="method not allowed", method=method)))
        end
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
                # Deliberately the NEWEST issue hour, never the cycle the live payloads fell back
                # to. The fallback exists to restore availability, and health exists to state that
                # the newest issuance is not complete: reading the served cycle here would report a
                # healthy deployment while the daemon was failing to issue, and the six-cycle
                # issuance dead-man reads exactly this field.
                readable = !ok || log_parse_state(log_path).readable
                cycle_complete = ok && readable && _valid_live_cycle(cycle)
                superseded = ok ? try
                    serving_cycle(get_log(log_path)).superseded
                catch e
                    e isa InterruptException && rethrow()
                    nothing
                end : nothing
                cycle_state = ok && nrow(cycle) > 0 ? _cycle_staleness(cycle) : nothing
                cycle_stale = cycle_state !== nothing && (
                    cycle_state.stale || cycle_state.expired || cycle_state.invalid_future
                )
                outage = outage_state(log_path)
                # Which product the log says is being served, and how often the served/shadow stages
                # fell back over the trailing day. A fresh log that has silently reverted to the
                # previous served pipeline is not a healthy deployment, so the identity and the
                # fallback rate belong in the same payload as the freshness.
                served = ok ? try
                    build_served_health(get_log(log_path))
                catch e
                    e isa InterruptException && rethrow()
                    nothing
                end : nothing
                # Whether the newest bytes on disk actually parse. Everything else here is computed
                # from the cached frame plus the file's mtime, so a log the loader cannot read would
                # otherwise be reported as a fresh, complete, healthy cycle: the age is the mtime of
                # the very file that was rejected, and the cycle is one that may no longer be in it.
                # The absent-log case was already honest; the unreadable one was not, and the
                # container probe and the issuance dead-man both read this verdict.
                parse_state = ok ? log_parse_state(log_path) :
                    (readable=true, error_age_min=nothing, error_type=nothing,
                     serving_cached_copy=false)
                # The daemon rewrites the log every cycle, so a file mtime older than the
                # staleness window means it has stopped issuing — report "stale" (dashboard dot
                # turns red). A recent file is still unhealthy when its latest issue hour lacks
                # the complete, internally consistent 1/2/3/6 h product cycle. An active outage
                # sentinel (dead-man trip, or the watchdog detecting a dead/unloaded process)
                # dominates: it reports "outage" even when the log file mtime still looks fresh.
                status = getproperty(outage, :active) ? "outage" :
                         !ok ? "no_log" :
                         !parse_state.readable ? "unreadable" :
                         ((age !== nothing && age > STALE_CYCLE_HOURS * 60) || cycle_stale) ?
                             "stale" :
                         !cycle_complete ? "incomplete" : "ok"
                return json_response((status = status,
                                      log_age_min=age,
                                      log_readable=parse_state.readable,
                                      log_parse_error_age_min=parse_state.error_age_min,
                                      log_parse_error_type=parse_state.error_type,
                                      serving_cached_log_copy=parse_state.serving_cached_copy,
                                      cycle_complete=cycle_complete,
                                      superseded_cycle_incomplete=superseded,
                                      outage=outage,
                                      served=served,
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
    warmup(log_path) -> seconds

Compile and cache the log-backed endpoint paths (the `get_log` CSV parse plus the
forecast/status/history builders and their JSON serialization) before the listener opens.

On a fresh process, the first `get_log` JIT-compiles the CSV-parse specialization while
holding `_LOG_LOCK`; without this warm-up, the first log-backed request after a (re)start
holds the lock for tens of seconds and stalls every request queued behind it — including
`/api/health`, which also takes the lock (observed live on the production dashboard:
first `/api/forecast` >20 s with `/api/health` timing out behind it, then 0.14 s once
compiled). Running the same paths once before `HTTP.serve` binds converts that random
mid-life stall into up-front startup cost. On a cold process this first-call JIT of the
CSV/DataFrame/JSON paths can run from seconds to a few minutes; it reads as ordinary restart
downtime to the supervisors, but a restart that long keeps the listener closed long enough to
trip the external watchdog's outage/recovery cycle for a probe interval. Steady-state
per-cycle log reloads use the compiled path and complete quickly.

Deliberately NOT exercised: network-touching endpoints (`/api/alerts` SWPC refresh, the
dB/dt nowcast) — warm-up must never block startup on external feeds. When the log does not
exist yet (fresh install, daemon still warming), only the cheap empty-frame paths compile;
the first real parse then pays the one-time JIT cost when the log appears.

Never throws: a warm-up failure only costs the optimization, never the server.
"""
function warmup(log_path::AbstractString)
    t0 = time()
    try
        df = get_log(log_path)
        json_response(build_forecast(df, log_path))
        json_response(build_status(df))
        json_response(build_history(df, 72.0))
        latest_cycle(df)
        outage_state(log_path)
    catch e
        e isa InterruptException && rethrow()
        @warn "endpoint warm-up failed; server starts anyway (first request may be slow)" exception=(e, catch_backtrace())
    end
    return round(time() - t0; digits=1)
end

"""
    bind_setting_from_env(name, default) -> String

Read a launchd/shell bind setting, rejecting an unsubstituted plist placeholder with a message that
names the file to repair. A hand-copied `com.example.solarsindy.dashboard.plist` whose
`__SWM_PORT__` was never replaced otherwise reaches `parse(Int, "__SWM_PORT__")`, and launchd turns
that `ArgumentError` into a sixty-second crash loop whose cause is one line of a stack trace.
"""
function bind_setting_from_env(name::AbstractString, default::AbstractString)
    value = strip(get(ENV, name, default))
    if startswith(value, "__") && endswith(value, "__")
        error("$name is still the launchd template placeholder $(value). Render the plist with " *
              "deploy/install_launchd.sh, or replace every placeholder listed in the template's " *
              "header comment before bootstrapping the job.")
    end
    isempty(value) && error("$name is empty; set it to a value or unset it to take the default " *
                            "$(default).")
    return String(value)
end

"Parse `SWM_PORT` with a message that distinguishes a bad value from an unrendered placeholder."
function port_from_env(; name::AbstractString = "SWM_PORT", default::AbstractString = "8723")
    raw = bind_setting_from_env(name, default)
    port = tryparse(Int, raw)
    (port === nothing || port < 1 || port > 65535) &&
        error("$name must be a TCP port between 1 and 65535, got $(repr(raw)).")
    return port
end

"""
    start_server(; host, port, log_path, blocking=true)

Start the dashboard server. Returns the `HTTP.Server` when `blocking=false`.
"""
function start_server(; host::AbstractString = bind_setting_from_env("SWM_HOST", "127.0.0.1"),
                        port::Integer = port_from_env(),
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
    wsec = warmup(log_path)
    println("warm-up: log/endpoint paths compiled in $(wsec) s; opening listener")
    # Explicit flush: when stdout/stderr are files (nohup/launchd), Julia block-buffers BOTH,
    # so the banner, warm-up line, and `@warn`/`@error` diagnostics would otherwise stay invisible
    # until the buffer fills — which blinds crash forensics and readiness checks that read the log.
    flush(stdout)
    flush(stderr)
    # Keep the block-buffered file-backed streams draining while the server runs, so request-time
    # `@error`/`@warn` forensics (and HTTP.jl's readiness line) reach the log within seconds
    # instead of only at process exit, where a SIGKILL/OOM-kill would otherwise lose them entirely.
    @async while true
        sleep(2)
        try
            flush(stdout)
            flush(stderr)
        catch
        end
    end
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
