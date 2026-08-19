# forecast_api.jl — data + threat layer for the Space-Weather Threat Monitor.
#
# Reads the locked-live forecast log produced by the SolarSINDy operational daemon and
# turns it into honest JSON payloads. Design principles:
#   * The locked log is the single source of truth. This layer never re-computes a
#     forecast; it serves exactly what was issued and (later) verified.
#   * A watch is assessed from the most negative lower edge among the displayed 90%-target
#     predictive intervals, not only from the point forecast. The watch is a conservative
#     screen, not a one-sided confidence statement or a storm probability.
#   * Calibration (coverage, RMSE) is recomputed from the log itself, so the dashboard
#     cannot drift from a stale report file.
#
# No new heavy dependencies: CSV, DataFrames, Dates, Statistics are already in the stack.

using CSV, DataFrames, Dates, Statistics, JSON3

# The depth-safe alerting center is defined once, in the package's dependency-free
# `src/serving_depth_safe.jl`. This application runs in its own environment and cannot load the
# package, so it includes that file: a second copy of the rule here is exactly how the published
# severity would drift from the served contract. The container image copies the file next to the
# application sources, hence the second candidate path.
const _DEPTH_SAFE_CANDIDATES = (
    normpath(joinpath(@__DIR__, "..", "..", "src", "serving_depth_safe.jl")),
    normpath(joinpath(@__DIR__, "..", "src_shared", "serving_depth_safe.jl")),
)
let source = findfirst(isfile, collect(_DEPTH_SAFE_CANDIDATES))
    source === nothing && error(
        "the shared depth-safe-center definition was not found at any of " *
        join(_DEPTH_SAFE_CANDIDATES, ", ") *
        "; the dashboard refuses to publish a severity it cannot derive from the served contract",
    )
    include(_DEPTH_SAFE_CANDIDATES[source])
end

# ---- Dst storm-intensity threat scale --------------------------------------------------
# Verified against the geomagnetic-storm literature: the widely adopted Dst classification
# uses primary division points -50 / -100 / -200 nT, with an extended minor tier
# (-30..-50 nT). Sources recorded in app/README.md. Dst is in nT; more negative = stronger.
const THREAT_LABELS = ("Quiet", "Minor storm", "Moderate storm", "Intense storm", "Extreme storm")
const THREAT_BANDS_NT = (-30.0, -50.0, -100.0, -200.0)   # upper edge of levels 1..4
const CURRENT_V2_MODEL_VERSION = "v2.1"
# Served pipeline: the V2.4e ten-expert super-learner with the SINDy-family mass floor, which counts
# the fixed static V2.2 stack among the family and takes it as an expert. The two earlier labels remain
# acceptable
# because the served stage falls back through them and says so per row: a cycle whose V2.4 stage could
# not act is served by the static stack, and a cycle that also lost the stack is served by the V2.1
# operator. Refusing those labels would blank the dashboard during a disclosed degradation instead of
# showing the forecast that was actually issued.
const CURRENT_V2_SERVED_MODEL_VERSION =
    "v2.4+sindy20x11+superlearner10floor+conformal"
const STACK_V2_SERVED_MODEL_VERSION =
    "v2.2+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia+staticstack(sindy60_fit407598)"
const PREVIOUS_V2_SERVED_MODEL_VERSION =
    "v2.1+sindy20x11+L1A+Bregime+Rprojection+H1inertia+Sinertia+Pinertia"
# Ordered strongest to weakest: a cycle whose horizons were served by different stages is published
# under the weakest label any of its rows carries, which is the stage that actually produced it.
const ACCEPTED_V2_SERVED_MODEL_VERSIONS =
    (CURRENT_V2_SERVED_MODEL_VERSION, STACK_V2_SERVED_MODEL_VERSION,
     PREVIOUS_V2_SERVED_MODEL_VERSION)
# The V2.3 analog candidate is logged as a shadow forecast: its confirmatory scoring returned NO_GO,
# so it never reaches the served center, the threat level or an alert.
const V2_3_SHADOW_MODEL_VERSION =
    "v2.3-shadow+sindy20x11+L1A+ADC(magnetic,K25)+T1rcal+LAT+E"
const LIVE_SKILL_MIN_VERIFIED = 48

# Reader-facing product name of a served-pipeline label: the label's leading version token, so the
# dashboard and the API name the product the log actually recorded instead of a hardcoded version.
function served_product_name(label)
    label isa AbstractString || return "unknown"
    token = first(split(String(label), "+"))
    isempty(token) && return "unknown"
    return startswith(token, "v") ? uppercase(token) : token
end

# Reader-facing driver-assumption sentence of a served row. The log records a machine token per row,
# and a cycle whose stack stage could not act records the V2.1 token; reading the row is what keeps a
# disclosed degradation from being described as the full stack pipeline.
const _V2_4_DRIVER_TOKEN =
    "ballistically_propagated_l1_then_ten_expert_nnls_superlearner_with_sindy_family_floor_over_" *
    "served_frozen_analog_and_the_static_regime_stack_then_depth_stratified_conformal_interval"
const _V2_2_DRIVER_TOKEN =
    "ballistically_propagated_l1_then_regime_aware_relaxation_then_rate_projection_then_one_hour_" *
    "inertia_blend_then_state_inertia_then_extreme_inertia_guard_then_static_regime_stack"
const _V2_1_DRIVER_TOKEN =
    "ballistically_propagated_l1_then_regime_aware_relaxation_then_rate_projection_then_one_hour_" *
    "inertia_blend_then_state_inertia_then_extreme_inertia_guard"
const _V2_1_DRIVER_SENTENCE =
    "Ballistically propagated L1 forcing, then regime-aware relaxation beyond the measured L1 " *
    "window, followed by a causal rate projection, validation-selected one-hour and " *
    "state-conditioned inertia blends, and an extreme-Dst inertia guard"
const _V2_2_DRIVER_SENTENCE =
    _V2_1_DRIVER_SENTENCE * ", and a fitted static regime stack over the six point components"
const _V2_4_DRIVER_SENTENCE =
    "Ballistically propagated L1 forcing through a fitted combination of ten causal forecasts, " *
    "weighted per lead, activity regime and ring-current depth with a majority weight held by the " *
    "SINDy family, which includes the static regime stack among the combined forecasts, with a " *
    "conformal interval calibrated per lead and depth"
const _DRIVER_ASSUMPTION_SENTENCES = Dict(
    _V2_4_DRIVER_TOKEN => _V2_4_DRIVER_SENTENCE,
    _V2_2_DRIVER_TOKEN => _V2_2_DRIVER_SENTENCE,
    _V2_1_DRIVER_TOKEN => _V2_1_DRIVER_SENTENCE,
    "ballistically_propagated_l1_then_regime_aware_relaxation" =>
        "Ballistically propagated L1 forcing, then regime-aware relaxation beyond the measured " *
        "L1 window",
)

function driver_assumption_text(token)
    token isa AbstractString || return "unrecorded"
    key = String(token)
    return get(_DRIVER_ASSUMPTION_SENTENCES, key, key)
end

"True for a served-pipeline label this build knows how to present."
_is_accepted_served_model(value) =
    value isa AbstractString && value in ACCEPTED_V2_SERVED_MODEL_VERSIONS

"""Return `(level, label)` for a finite Dst value in nT; non-finite input is unknown."""
function dst_threat_level(dst::Real)
    isfinite(dst) || return (nothing, "Unknown")
    dst > THREAT_BANDS_NT[1] && return (0, THREAT_LABELS[1])
    dst > THREAT_BANDS_NT[2] && return (1, THREAT_LABELS[2])
    dst > THREAT_BANDS_NT[3] && return (2, THREAT_LABELS[3])
    dst > THREAT_BANDS_NT[4] && return (3, THREAT_LABELS[4])
    return (4, THREAT_LABELS[5])
end

# ---- small helpers ---------------------------------------------------------------------
# JSON-friendly number: finite Float64 or nothing (JSON null). Guards missing/NaN/Inf,
# boolean-as-number coercion, and finite wide values that overflow during Float64 conversion.
function jnum(x)
    (x === missing || x === nothing || !(x isa Real) || x isa Bool) && return nothing
    value = try
        Float64(x)
    catch e
        e isa InterruptException && rethrow()
        return nothing
    end
    return isfinite(value) ? value : nothing
end

# Tolerant ISO8601 -> DateTime (handles 0..n fractional-second digits); missing-safe.
function parse_dt(s)
    (s === missing || s === nothing) && return missing
    str = strip(String(s))
    isempty(str) && return missing
    try
        return DateTime(str)                      # fast path (<=3 fractional digits)
    catch e
        e isa InterruptException && rethrow()
        m = match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", str)
        m === nothing && return missing
        dt = tryparse(DateTime, m.captures[1])    # whole-second fallback; tryparse → nothing on a
        return dt === nothing ? missing : dt      # format-valid but invalid date (e.g. month 13), never throws
    end
end

# DateTime -> ISO UTC string with explicit Z, or nothing.
jdt(dt) = (dt === missing || dt === nothing) ? nothing : string(dt) * "Z"

# ---- log loading (identity-cached, append-safe) ----------------------------------------
const _TIME_COLS = ["issue_time_utc", "latest_solar_wind_utc", "latest_dst_time_utc", "target_time_utc"]
const _LOG_CACHE = Ref{Any}(nothing)
const _LOG_LOCK = ReentrantLock()
# The file identity whose parse failed, the failure itself, and when it was recorded. Guarded by
# `_LOG_LOCK` together with `_LOG_CACHE`.
const _LOG_PARSE_FAILURE = Ref{Any}(nothing)
const _CALIBRATION_CACHE = Ref{Any}(nothing)
const _CALIBRATION_LOCK = ReentrantLock()
const _LATEST_CYCLE_CACHE = Ref{Any}(nothing)
const _LATEST_CYCLE_LOCK = ReentrantLock()
const _SERVING_CYCLE_CACHE = Ref{Any}(nothing)
const _SERVING_CYCLE_LOCK = ReentrantLock()
const _VERIFIED_ROWS_CACHE = Ref{Any}(nothing)
const _VERIFIED_ROWS_LOCK = ReentrantLock()
const _HISTORY_CACHE = Ref{Any}(nothing)
const _HISTORY_LOCK = ReentrantLock()

function _log_file_identity(path::AbstractString)
    resolved = realpath(path)
    info = stat(path)
    return (resolved, info.device, info.inode, info.size, info.mtime, info.ctime)
end

function _cached_log_key(df::DataFrame)
    return lock(_LOG_LOCK) do
        cached = _LOG_CACHE[]
        cached !== nothing && cached[2] === df ? (cached[1], objectid(df)) : nothing
    end
end

function _canonical_missing_log_path(path::AbstractString)
    absolute = abspath(normpath(path))
    parent = dirname(absolute)
    return isdir(parent) ? joinpath(realpath(parent), basename(absolute)) : absolute
end

function _load_log(path::AbstractString)
    # Read time columns as String and parse ourselves (robust to fractional-second variation).
    df = CSV.read(path, DataFrame;
                  types=Dict(c => String for c in _TIME_COLS),
                  validate=false, missingstring=["", "NaN", "missing"], silencewarnings=true)
    for c in _TIME_COLS
        if hasproperty(df, Symbol(c))
            df[!, Symbol(c * "_dt")] = parse_dt.(df[!, Symbol(c)])
        end
    end
    return df
end

"""Return the current forecast log as a DataFrame, reloading only when the file changes.
When the file is absent, serve the last good frame if cached, else an empty frame so the
log-backed endpoints degrade to `available=false` (never a 500). On a read failure mid-append,
serve the last good frame (raising only if none exists).

A parse failure is cached against the failing file identity. Serving the last good frame without
recording the failure left the cache key at the previous file, so every subsequent request re-ran the
full parse — under the monitor's row cap a multi-second parse, serialized behind `_LOG_LOCK`, for as
long as the unreadable file sat there. The identity is what re-arms the attempt: new bytes are always
re-read, and only the exact file that already failed is short-circuited. `loader` is injectable so
the caching contract can be tested against a counting loader."""
function get_log(path::AbstractString; loader = _load_log)
    lock(_LOG_LOCK) do
        c = _LOG_CACHE[]
        if !isfile(path)
            # Absent (fresh install, before the daemon's first write, or log rotation): fall back
            # to the cached frame, or an empty frame — do not throw, so /api/status et al. return
            # available=false rather than propagating a 500 that also leaks the log path.
            wanted = _canonical_missing_log_path(path)
            return c === nothing || c[1][1] != wanted ? DataFrame() : c[2]
        end
        # Resolve the file identity defensively: the file can be unlinked (log rotation/rewrite)
        # between the isfile check above and here, in which case realpath/stat throw. Degrade to
        # the absent-file branch instead of surfacing a 500 (and leaking the log path).
        key = try
            _log_file_identity(path)
        catch e
            e isa InterruptException && rethrow()
            wanted = _canonical_missing_log_path(path)
            return c === nothing || c[1][1] != wanted ? DataFrame() : c[2]
        end
        if c === nothing || c[1] != key
            # Negative cache: these exact bytes already failed to parse. Re-running the parse would
            # produce the same failure at the same cost while holding this lock, so the recorded
            # outcome is replayed instead — the cached frame when one exists for this file, and the
            # original exception when nothing was ever loaded from it.
            failure = _LOG_PARSE_FAILURE[]
            if failure !== nothing && failure.key == key
                (c !== nothing && c[1][1] == key[1]) && return c[2]
                throw(failure.error)
            end
            try
                loaded = loader(path)
                key == _log_file_identity(path) || error(
                    "forecast log changed while it was being loaded",
                )
                _LOG_CACHE[] = (key, loaded)
                _LOG_PARSE_FAILURE[] = nothing
            catch e
                e isa InterruptException && rethrow()
                # If the file vanished mid-load, degrade like the absent branch rather than 500.
                if !isfile(path)
                    wanted = _canonical_missing_log_path(path)
                    return c === nothing || c[1][1] != wanted ? DataFrame() : c[2]
                end
                _LOG_PARSE_FAILURE[] = (key = key, error = e, at = time(),
                                        error_type = string(nameof(typeof(e))))
                (c === nothing || c[1][1] != key[1]) && rethrow(e)
                @warn "log reload failed; serving cached copy" exception=(e, catch_backtrace())
                return c[2]
            end
        end
        return _LOG_CACHE[][2]
    end
end

"""Whether the current bytes of `path` parse, and how long a standing failure has held.

`get_log` degrades to the last good frame when a reload fails, which is what keeps the endpoints
answering through a mid-append read. Health must not inherit that silence: a payload assembled from a
cached frame while the file on disk cannot be parsed is a forecast from an unknown age presented as
the current one, and the container health probe reads exactly that verdict. Only the failure type is
reported — an exception message can carry the log path."""
function log_parse_state(path::AbstractString)
    failure, cached = lock(_LOG_LOCK) do
        (_LOG_PARSE_FAILURE[], _LOG_CACHE[])
    end
    readable = (readable=true, error_age_min=nothing, error_type=nothing, serving_cached_copy=false)
    failure === nothing && return readable
    key = try
        _log_file_identity(path)
    catch e
        e isa InterruptException && rethrow()
        return readable
    end
    failure.key == key || return readable
    return (readable=false,
            error_age_min=round((time() - failure.at) / 60; digits=1),
            error_type=failure.error_type,
            serving_cached_copy=cached !== nothing && cached[1][1] == key[1])
end

# Pick forecast columns with graceful fallbacks. hasproperty is checked BEFORE
# access so older CSV schemas fall back instead of throwing "column not found".
function _col(row, a::Symbol, b::Symbol)
    hasproperty(row, a) && getproperty(row, a) !== missing && return getproperty(row, a)
    hasproperty(row, b) ? getproperty(row, b) : missing
end
_audit_pred(row) = _col(row, :v2_pred_dst_nt,      :pred_dst_nt)
_audit_ci05(row) = _col(row, :v2_pred_dst_ci05_nt, :pred_dst_ci05_nt)
_audit_ci95(row) = _col(row, :v2_pred_dst_ci95_nt, :pred_dst_ci95_nt)

# Upgraded V2 is the product forecast. The live CSV keeps older product-column
# names, with fallback to older v2_/pred_* columns for legacy rows.
_v2_pred(row) = (x = _col(row, :served_pred_dst_nt,      :v2_pred_dst_nt);      x === missing ? _audit_pred(row) : x)
_v2_ci05(row) = (x = _col(row, :served_pred_dst_ci05_nt, :v2_pred_dst_ci05_nt); x === missing ? _audit_ci05(row) : x)
_v2_ci95(row) = (x = _col(row, :served_pred_dst_ci95_nt, :v2_pred_dst_ci95_nt); x === missing ? _audit_ci95(row) : x)
_rowget(row, name::Symbol) = hasproperty(row, name) ? getproperty(row, name) : missing

function _row_is_current_v2(row)
    model = _rowget(row, :model_version)
    served_model = _rowget(row, :sub_hourly_model_version)
    return model isa AbstractString && model == CURRENT_V2_MODEL_VERSION &&
           _is_accepted_served_model(served_model)
end

# Depth-safe alerting center. A fitted combination can blend toward persistence and therefore report a
# shallower storm than the stages it replaced; the published threat level is taken against the deepest
# of the served center, the static-stack center and the V2.1 operator center, so a conservative
# candidate cannot lower a warning either predecessor would have raised, while a deeper candidate still
# escalates one. Rows written before a stage existed carry no column for it and are simply skipped.
#
# The comparison itself is `v24_serving_depth_safe_center`, loaded from the shared definition the
# package serves under, so the published severity and the packaged serving contract are one rule.
_severity_partners(row) = (_rowget(row, :v2_2_stack_pred_dst_nt),
                           _rowget(row, :v2_1_served_pred_dst_nt))

function _severity_pred(row)
    served = _v2_pred(row)
    (served isa Real && !(served isa Bool)) || return served
    partners = Float64[Float64(v) for v in _severity_partners(row)
                       if v isa Real && !(v isa Bool)]
    isempty(partners) && return served
    return v24_serving_depth_safe_center(Float64(served), partners...)
end

# Predecessor watch edges of a row: the lower band edge the static-stack stage and the V2.1 operator
# would each have published for this issue, in the same strongest-to-weakest order as the centers. The
# engine forms them from the band machinery those stages served under and logs them per row.
_severity_partner_edges(row) = (_rowget(row, :v2_2_stack_ci05_dst_nt),
                                _rowget(row, :v2_1_served_ci05_dst_nt))

# Names the payload uses for the stage whose own edge is published, aligned with the tuple above.
const _SEVERITY_EDGE_PARTNERS = ("v2_2_stack", "v2_1_served")

# Depth-safe interval lower edge, and the stage whose own edge it is.
#
# The point level is taken on the depth-safe center, but a watch is assessed on the band edge, so a
# candidate that reports a shallower center than a stage it replaced could drop a watch tier that stage
# raised on the same physics. The edge is therefore the deepest of the served edge and the edge each
# predecessor would have published — the same minimum the center takes, applied to the edge itself.
#
# Shifting the served edge by the change in the center is not equivalent and is not depth-safe: the
# edge a predecessor would have published is its own center minus its OWN half-width, and the served
# half-width is a different number, so the shifted edge lands at the deepest partner center less the
# served half-width. Wherever the served band is the narrower of the two that is a shallower edge than
# the predecessor's own, by up to a whole storm tier. That shift is retained only for
# rows written before the predecessor edges were logged, where nothing deeper is on record, and such a
# row is disclosed as using the fallback rule.
function _severity_ci05_with_source(row)
    lower = _v2_ci05(row)
    (lower isa Real && !(lower isa Bool)) || return (lower, "unavailable")
    edge = Float64(lower)
    logged = Union{Nothing,Float64}[
        (v isa Real && !(v isa Bool) && isfinite(Float64(v))) ? Float64(v) : nothing
        for v in _severity_partner_edges(row)
    ]
    recorded = any(v -> v isa Real && !(v isa Bool), _severity_partner_edges(row))
    if recorded
        isfinite(edge) || return (lower, "served")
        finite = Float64[v for v in logged if v !== nothing]
        isempty(finite) && return (lower, "served")
        value = v24_serving_depth_safe_center(edge, finite...)
        value == edge && return (value, "served")
        hit = findfirst(v -> v !== nothing && v == value, logged)
        return (value, hit === nothing ? "served" : _SEVERITY_EDGE_PARTNERS[hit])
    end
    # Legacy row: no predecessor edge on record, so the earlier center-shift rule is kept unchanged.
    served = _v2_pred(row)
    (served isa Real && !(served isa Bool)) || return (lower, "served")
    partners = Float64[Float64(v) for v in _severity_partners(row)
                       if v isa Real && !(v isa Bool)]
    isempty(partners) && return (lower, "served")
    isfinite(edge) || return (lower, "served")
    shift = v24_serving_depth_safe_center(Float64(served), partners...) - Float64(served)
    isfinite(shift) || return (lower, "served")
    shift < 0.0 || return (edge, "served")
    return (edge + shift, "legacy_center_shift")
end

_severity_ci05(row) = first(_severity_ci05_with_source(row))
_severity_ci05_source(row) = last(_severity_ci05_with_source(row))

# Rows of the most recent forecast cycle, defined by ISSUE epoch (not driver vintage). When the
# L1 feed stalls across issue boundaries, several hourly cycles share one latest_solar_wind_utc;
# keying on that vintage would merge superseded issues into one payload (conflicting horizons per
# target, a stale issue/anchor, and threat minima computed over superseded forecasts). Per-row
# issue timestamps within one cycle may differ by seconds, so the product identity is the UTC issue
# hour and `_valid_live_cycle` separately bounds the within-cycle spread. This prevents adjacent-hour
# cycles from mixing while accepting the sequential-per-horizon writer and a short retry. Returns a
# sub-DataFrame sorted by target time.
const _CYCLE_TOL = Minute(5)   # sidecar matching tolerance; live-cycle grouping uses issue hour

_sort_by_target!(cyc) = (hasproperty(cyc, :target_time_utc_dt) && sort!(cyc, :target_time_utc_dt); cyc)

function _latest_cycle_uncached(df::DataFrame)
    nrow(df) == 0 && return df
    if hasproperty(df, :issue_time_utc_dt)
        iss = df.issue_time_utc_dt
        if !isempty(collect(skipmissing(iss)))
            tmax = maximum(skipmissing(iss))
            issue_hour = floor(tmax, Hour)
            keep = map(iss) do issue
                issue !== missing && floor(issue, Hour) == issue_hour
            end
            return _sort_by_target!(df[keep, :])
        end
    end
    # No usable issue time (legacy schema): fall back to the freshest solar-wind vintage.
    if hasproperty(df, :latest_solar_wind_utc_dt)
        sw = df.latest_solar_wind_utc_dt
        if !isempty(collect(skipmissing(sw)))
            newest = maximum(skipmissing(sw))
            return _sort_by_target!(df[coalesce.(sw .== newest, false), :])
        end
    end
    return df[1:0, :]   # nothing usable; downstream build_* handle the no-cycle case
end


function latest_cycle(df::DataFrame)
    key = _cached_log_key(df)
    key === nothing && return _latest_cycle_uncached(df)
    return lock(_LATEST_CYCLE_LOCK) do
        cached = _LATEST_CYCLE_CACHE[]
        if cached === nothing || cached[1] != key
            cached = (key, _latest_cycle_uncached(df))
            _LATEST_CYCLE_CACHE[] = cached
        end
        return cached[2]
    end
end

# How far back the serving fallback looks for a complete cycle when the newest issue hour is not one.
# Six issue hours is the daemon's own issuance dead-man window: past it the deployment is already in
# a declared outage, and the staleness gate rejects the cycle in any case.
const SERVING_FALLBACK_CYCLES = 6

function _cycle_at_issue_hour(df::DataFrame, hour::DateTime)
    keep = map(df.issue_time_utc_dt) do issue
        issue !== missing && floor(issue, Hour) == hour
    end
    return _sort_by_target!(df[keep, :])
end

"""Cycle the payload is served from: the newest issue hour, or the newest one that is complete.

A cycle can arrive incomplete or internally inconsistent — a partial write, a retry that duplicated a
horizon, an anchor that moved between rows — and serving it would publish a trajectory the engine
never issued as one. Rejecting it wholesale is what blanked every log-backed endpoint while a
complete cycle from the previous issue hour sat unread, so the newest complete cycle is served
instead and the fact that it is not the newest issue hour travels with the payload. The selection is
structural (no wall clock), so it caches with the log identity; the freshness gate is applied to the
chosen cycle by the callers, at request time."""
function _serving_cycle_uncached(df::DataFrame)
    newest = _latest_cycle_uncached(df)
    result = (cycle=newest, superseded=false, newest=newest)
    nrow(newest) == 0 && return result
    _valid_live_cycle(newest) && return result
    hasproperty(df, :issue_time_utc_dt) || return result
    issues = collect(skipmissing(df.issue_time_utc_dt))
    isempty(issues) && return result
    hours = sort(unique(floor.(issues, Hour)); rev=true)
    last_index = min(length(hours), SERVING_FALLBACK_CYCLES + 1)
    for index in 2:last_index
        candidate = _cycle_at_issue_hour(df, hours[index])
        _valid_live_cycle(candidate) &&
            return (cycle=candidate, superseded=true, newest=newest)
    end
    return result
end

function serving_cycle(df::DataFrame)
    key = _cached_log_key(df)
    key === nothing && return _serving_cycle_uncached(df)
    return lock(_SERVING_CYCLE_LOCK) do
        cached = _SERVING_CYCLE_CACHE[]
        if cached === nothing || cached[1] != key
            cached = (key, _serving_cycle_uncached(df))
            _SERVING_CYCLE_CACHE[] = cached
        end
        return cached[2]
    end
end

"""Cycle a live payload is built from, with its freshness and the superseded-cycle disclosure.

The fallback restores availability; it must never manufacture it. A complete cycle that is itself
outside the freshness window is not something to serve in place of the newest issue hour, so the
newest hour is reported exactly as it would be without the fallback and the endpoints degrade as
before."""
function _payload_cycle(df::DataFrame)
    served = serving_cycle(df)
    stale = _cycle_staleness(served.cycle)
    served.superseded || return (cycle=served.cycle, superseded=false, stale=stale)
    (stale.stale || stale.expired || stale.invalid_future) ||
        return (cycle=served.cycle, superseded=true, stale=stale)
    return (cycle=served.newest, superseded=false, stale=_cycle_staleness(served.newest))
end

# Verified rows: a real forecast (target strictly AFTER both the issue time and the last observed Dst — a
# 0-lead row is not a forecast and is what the locked comparison report excludes), with a FINITE observation
# and a finite V2 point + band. jnum(obs) rejects NaN/Inf observations that would otherwise reach the
# calibration summary and make JSON serialization throw.
function _verified_rows_uncached(df::DataFrame)
    nrow(df) == 0 && return view(df, Int[], :)
    (hasproperty(df, :target_time_utc_dt) &&
     hasproperty(df, :observation_dst_nt)) || return view(df, Int[], :)
    has_issue  = hasproperty(df, :issue_time_utc_dt)
    has_anchor = hasproperty(df, :latest_dst_time_utc_dt)
    (has_issue && has_anchor) || return view(df, Int[], :)
    keep = trues(nrow(df))
    for (i, r) in enumerate(eachrow(df))
        tt = _rowget(r, :target_time_utc_dt)
        issue = _rowget(r, :issue_time_utc_dt)
        anchor = _rowget(r, :latest_dst_time_utc_dt)
        valid_lead = tt isa DateTime && issue isa DateTime && anchor isa DateTime &&
                     tt > issue && tt > anchor
        lo = jnum(_v2_ci05(r))
        hi = jnum(_v2_ci95(r))
        keep[i] = _row_is_current_v2(r) && valid_lead &&
                  jnum(_rowget(r, :observation_dst_nt)) !== nothing &&
                  jnum(_v2_pred(r)) !== nothing && lo !== nothing && hi !== nothing && lo <= hi
    end
    return view(df, keep, :)
end


function verified_rows(df::DataFrame)
    key = _cached_log_key(df)
    key === nothing && return _verified_rows_uncached(df)
    return lock(_VERIFIED_ROWS_LOCK) do
        cached = _VERIFIED_ROWS_CACHE[]
        if cached === nothing || cached[1] != key
            cached = (key, _verified_rows_uncached(df))
            _VERIFIED_ROWS_CACHE[] = cached
        end
        return cached[2]
    end
end

# ---- calibration summary (computed from the log, not a static report) ------------------
# Honesty note: the *live* interval method (ACI) and the *verified* track record can differ.
# ACI was deployed after most verified rows were issued, so coverage is reported overall AND
# broken down by interval method, and the live method is reported separately. We never imply
# the historical coverage was achieved by a method that has no verified rows yet.
_src_label(s) = (s === missing || s === nothing || s == "") ? "legacy" : string(s)

function _wide_rmse(observed, predicted)
    n = length(observed)
    n == length(predicted) || throw(DimensionMismatch(
        "observed and predicted RMSE vectors must have equal lengths",
    ))
    n >= 1 || return nothing
    squared_sum = BigFloat(0)
    for index in eachindex(observed, predicted)
        difference = BigFloat(observed[index]) - BigFloat(predicted[index])
        squared_sum += difference^2
    end
    value = sqrt(squared_sum / n)
    value <= BigFloat(floatmax(Float64)) || return nothing
    converted = Float64(value)
    return isfinite(converted) ? converted : nothing
end

function _stable_rmse_or_nothing(observed, predicted)
    n = length(observed)
    n == length(predicted) || throw(DimensionMismatch(
        "observed and predicted RMSE vectors must have equal lengths",
    ))
    n >= 1 || return nothing
    scale = 0.0
    scaled_sum = 1.0
    for index in eachindex(observed, predicted)
        left = Float64(observed[index])
        right = Float64(predicted[index])
        isfinite(left) && isfinite(right) || return nothing
        difference = left - right
        isfinite(difference) || return _wide_rmse(observed, predicted)
        magnitude = abs(difference)
        iszero(magnitude) && continue
        if scale < magnitude
            scaled_sum = 1.0 + scaled_sum * (scale / magnitude)^2
            scale = magnitude
        else
            scaled_sum += (magnitude / scale)^2
        end
    end
    iszero(scale) && return 0.0
    value = scale * sqrt(scaled_sum / n)
    return isfinite(value) ? value : _wide_rmse(observed, predicted)
end

_prefer_metric(primary, fallback) = primary === nothing ? fallback : primary

function _compute_calibration_summary(df::DataFrame, cyc::DataFrame)
    # Live interval method = the method of the cycle this build publishes, read the way the forecast
    # payload reads it. Two readings were wrong here. Taking the newest issue hour described a cycle
    # the payloads need not be serving: during the superseded-cycle fallback the newest hour is the
    # incomplete one, so the panel reported "unknown" for a forecast that was on the page. And
    # requiring one common `interval_source` across the horizons is the rule per-row disclosure
    # replaced: a cycle whose super-learner stage acted on some horizons and not others carries two
    # sources by design, and the common-field reading of it is empty — published as the string
    # "nothing", which the page then showed as the live interval method.
    valid_cycle = _valid_live_cycle(cyc)
    # Every method the published cycle's horizons carry, and the one it is reported under (the
    # method the rows carrying its weakest served label were issued with). A cycle whose horizons
    # were served by one stage throughout has exactly one of each and they agree.
    live_sources = valid_cycle ? _cycle_interval_sources(cyc) : String[]
    reported_src = valid_cycle ? _cycle_interval_source(cyc) : nothing
    # Three distinguishable states, because they are three different facts: the method the published
    # cycle is reported under; nothing, when there is a published cycle whose horizons carry more
    # than one method and none of them is the one it is reported under (the set below is what the
    # page then states); and "unknown", when there is no publishable cycle to describe at all.
    live_src = valid_cycle ? (reported_src === nothing ? nothing : String(reported_src)) : "unknown"
    # Rows counted as "matured under the live interval method": the reported method when there is
    # one, otherwise any method the published cycle carries. Naming a single method that the cycle
    # does not have would attribute the count to a method that was never solely served.
    live_src_set = live_src === nothing ? Set(live_sources) : Set([live_src])
    # Verified rows accumulate across served pipelines. Pooling them would present a record earned by
    # the previous served label as the current product's record, so the count is broken out per label
    # and the current label's own count is reported separately.
    live_served = valid_cycle ? something(_cycle_served_model(cyc), "unknown") : "unknown"

    v = verified_rows(df)
    n = nrow(v)
    n == 0 && return (n_verified=0, coverage_90=nothing, rmse_nt=nothing,
                      v2_n_verified=0, v2_coverage_90=nothing, v2_rmse_nt=nothing,
                      frozen_tail_ablation_rmse_nt=nothing,
                      audit_baseline_rmse_nt=nothing,
                      rmse_sindy_v1_nt=nothing, rmse_persistence_nt=nothing,
                      rmse_burton_nt=nothing, rmse_burton_full_nt=nothing,
                      rmse_obrien_nt=nothing,
                      comparison_n_verified=0,
                      v2_matched_rmse_nt=nothing,
                      frozen_tail_ablation_matched_rmse_nt=nothing,
                      sindy_v1_matched_rmse_nt=nothing,
                      persistence_matched_rmse_nt=nothing,
                      burton_matched_rmse_nt=nothing,
                      burton_full_matched_rmse_nt=nothing,
                      obrien_matched_rmse_nt=nothing,
                      live_skill_min_verified=LIVE_SKILL_MIN_VERIFIED,
                      live_skill_mature=false,
                      live_skill_rows_remaining=LIVE_SKILL_MIN_VERIFIED,
                      interval_target_coverage=0.90,
                      served_interval_coverage_scope="empirical_only",
                      current_interval_source=live_src,
                      current_interval_sources=live_sources,
                      n_verified_current_source=0,
                      current_served_model=live_served, n_verified_current_served_model=0,
                      by_served_model=[],
                      deepest_obs_dst_nt=nothing, n_storm_verified=0, by_source=[])
    obs   = Float64.(v.observation_dst_nt)
    pred  = Float64.(_v2_pred.(eachrow(v)))
    ci05  = Float64.(_v2_ci05.(eachrow(v)))
    ci95  = Float64.(_v2_ci95.(eachrow(v)))
    audit_pred = [jnum(_audit_pred(r)) for r in eachrow(v)]
    inside = (obs .>= ci05) .& (obs .<= ci95)
    rmse(p) = _stable_rmse_or_nothing(obs, p)
    product_col_mask = falses(n)
    if all(c -> c in names(v), ("served_pred_dst_nt", "served_pred_dst_ci05_nt", "served_pred_dst_ci95_nt"))
        product_col_mask .= [jnum(v[i, :served_pred_dst_nt]) !== nothing &&
                             jnum(v[i, :served_pred_dst_ci05_nt]) !== nothing &&
                             jnum(v[i, :served_pred_dst_ci95_nt]) !== nothing for i in 1:n]
    end
    product_col_n = count(product_col_mask)
    product_col_cov = nothing
    product_col_rmse = nothing
    rmse_optional(values, mask=trues(n)) = begin
        valid = mask .& .!isnothing.(values)
        any(valid) || return nothing
        value = _stable_rmse_or_nothing(obs[valid], Float64.(values[valid]))
        return value === nothing ? nothing : jnum(round(value; digits=2))
    end
    audit_rmse = rmse_optional(audit_pred)
    if product_col_n > 0
        spred = Float64.(v[product_col_mask, :served_pred_dst_nt])
        slo = Float64.(v[product_col_mask, :served_pred_dst_ci05_nt])
        shi = Float64.(v[product_col_mask, :served_pred_dst_ci95_nt])
        sobs = obs[product_col_mask]
        sinside = (sobs .>= slo) .& (sobs .<= shi)
        product_col_cov = jnum(round(mean(sinside); digits=3))
        product_value = _stable_rmse_or_nothing(sobs, spred)
        product_col_rmse = product_value === nothing ? nothing :
            jnum(round(product_value; digits=2))
        audit_rmse = rmse_optional(audit_pred, product_col_mask)
    end
    product_mask = product_col_n > 0 ? product_col_mask : trues(n)
    product_n = count(product_mask)
    product_cov = product_col_n > 0 ? product_col_cov : jnum(round(mean(inside); digits=3))
    fallback_rmse = rmse(pred)
    product_rmse = product_col_n > 0 ? product_col_rmse :
        fallback_rmse === nothing ? nothing : jnum(round(fallback_rmse; digits=2))
    v1   = "v1_pred_dst_nt" in names(v) ? jnum.(v.v1_pred_dst_nt) : fill(nothing, n)
    pers = "persistence_dst_nt" in names(v) ? jnum.(v.persistence_dst_nt) : fill(nothing, n)
    burt = "burton_dst_nt" in names(v) ? jnum.(v.burton_dst_nt) : fill(nothing, n)
    burf = "burton_full_dst_nt" in names(v) ? jnum.(v.burton_full_dst_nt) : fill(nothing, n)
    obri = "obrien_dst_nt" in names(v) ? jnum.(v.obrien_dst_nt) : fill(nothing, n)

    # The displayed comparison is a genuinely matched cohort: a target enters only when
    # the served product, frozen-tail ablation, and every declared comparator are finite.
    # This prevents a method with missing hard cases from receiving an incomparable RMSE.
    comparison_mask = copy(product_mask)
    for values in (audit_pred, v1, pers, burt, burf, obri)
        comparison_mask .&= .!isnothing.(values)
    end
    comparison_n = count(comparison_mask)
    matched_rmse(values) = rmse_optional(values, comparison_mask)
    v2_values = jnum.(pred)
    live_skill_mature = comparison_n >= LIVE_SKILL_MIN_VERIFIED

    # per-method coverage breakdown
    srcs = "interval_source" in names(v) ? _src_label.(v.interval_source) : fill("unknown", n)
    product_srcs = srcs[product_mask]
    product_inside = inside[product_mask]
    by_source = NamedTuple[]
    for s in sort(unique(product_srcs))
        m = product_srcs .== s
        push!(by_source, (source=s, n=count(m),
                         coverage_90=round(mean(product_inside[m]); digits=3)))
    end
    n_live = count(in(live_src_set), product_srcs)

    served_labels = "sub_hourly_model_version" in names(v) ?
        _src_label.(v.sub_hourly_model_version) : fill("unknown", n)
    product_served = served_labels[product_mask]
    by_served_model = NamedTuple[]
    for label in sort(unique(product_served))
        m = product_served .== label
        push!(by_served_model,
              (served_model_version=label, product=served_product_name(label), n=count(m),
               coverage_90=round(mean(product_inside[m]); digits=3),
               rmse_nt=(value = _stable_rmse_or_nothing(obs[product_mask][m],
                                                       Float64.(pred[product_mask][m]));
                        value === nothing ? nothing : jnum(round(value; digits=2)))))
    end
    n_live_served = count(product_served .== live_served)

    return (n_verified=product_n,
            coverage_90=product_cov,
            rmse_nt=product_rmse,
            v2_n_verified=product_n,
            v2_coverage_90=product_cov,
            v2_rmse_nt=product_rmse,
            frozen_tail_ablation_rmse_nt=audit_rmse,
            # Deprecated compatibility alias. New clients should use the
            # explicit V2.1 frozen-tail-ablation field above.
            audit_baseline_rmse_nt=audit_rmse,
            rmse_sindy_v1_nt=rmse_optional(v1, product_mask),
            rmse_persistence_nt=rmse_optional(pers, product_mask),
            rmse_burton_nt=rmse_optional(burt, product_mask),
            rmse_burton_full_nt=rmse_optional(burf, product_mask),
            rmse_obrien_nt=rmse_optional(obri, product_mask),
            comparison_n_verified=comparison_n,
            v2_matched_rmse_nt=matched_rmse(v2_values),
            frozen_tail_ablation_matched_rmse_nt=matched_rmse(audit_pred),
            sindy_v1_matched_rmse_nt=matched_rmse(v1),
            persistence_matched_rmse_nt=matched_rmse(pers),
            burton_matched_rmse_nt=matched_rmse(burt),
            burton_full_matched_rmse_nt=matched_rmse(burf),
            obrien_matched_rmse_nt=matched_rmse(obri),
            live_skill_min_verified=LIVE_SKILL_MIN_VERIFIED,
            live_skill_mature=live_skill_mature,
            live_skill_rows_remaining=max(0, LIVE_SKILL_MIN_VERIFIED - comparison_n),
            interval_target_coverage=0.90,
            served_interval_coverage_scope="empirical_only",
            current_interval_source=live_src,
            # Every method the published cycle's horizons carry. A cycle whose stage changed between
            # horizons has more than one, and the page states the set rather than a single name.
            current_interval_sources=live_sources,
            n_verified_current_source=n_live,
            current_served_model=live_served,
            n_verified_current_served_model=n_live_served,
            by_served_model=by_served_model,
            deepest_obs_dst_nt=jnum(round(minimum(obs[product_mask]); digits=1)),
            n_storm_verified=count(obs[product_mask] .< -50),
            by_source=by_source)
end

_compute_calibration_summary(df::DataFrame) =
    _compute_calibration_summary(df, _payload_cycle(df).cycle)

function calibration_summary(df::DataFrame)
    # The published cycle is part of what this summary states, and the fallback that chooses it is
    # applied at request time, so the cached value is keyed on that cycle as well as on the log. The
    # selection itself is structural and memoised, so the identity is stable while the choice is.
    cyc = _payload_cycle(df).cycle
    key = _cached_log_key(df)
    key === nothing && return _compute_calibration_summary(df, cyc)
    return lock(_CALIBRATION_LOCK) do
        cached = _CALIBRATION_CACHE[]
        if cached === nothing || cached[1] != key || cached[3] !== cyc
            cached = (key, _compute_calibration_summary(df, cyc), cyc)
            _CALIBRATION_CACHE[] = cached
        end
        return cached[2]
    end
end

# ---- payload builders ------------------------------------------------------------------
# Recent observed Dst straight from the log's latest_dst columns (every row records the freshest Kyoto Dst at
# its issue). This runs to the forecast anchor with no extra network dependency, so the observed line is
# continuous up to where the forecast begins (the verified-row track lags by the verification delay). Later
# rows overwrite earlier ones at the same timestamp, so revised Dst values win.
function _recent_observed(df::DataFrame; hours::Real=48)
    nrow(df) == 0 && return NamedTuple[]
    (hasproperty(df, :latest_dst_time_utc_dt) && hasproperty(df, :latest_dst_nt)) || return NamedTuple[]
    cutoff = now(UTC) - Hour(round(Int, hours))
    seen = Dict{DateTime,Float64}()
    for r in eachrow(df)
        t = r.latest_dst_time_utc_dt; v = jnum(r.latest_dst_nt)
        (t === missing || v === nothing || t < cutoff) && continue
        seen[t] = v
    end
    out = [(target_utc=jdt(t), observed_dst_nt=v) for (t, v) in seen]
    sort!(out, by=x -> x.target_utc)
    return out
end

# Sub-hour MODEL trajectory (display only) written by the engine next to the log: V2 integrated
# at a sub-hour step. Not a validated sub-hour forecast (Dst is observed hourly; the ODE is hourly-fit) — it is the
# hourly model's own interpolation. Defensive: missing/unreadable/stale → [].
#
# Staleness gate (documented contract): the engine writes the sidecar in a try/catch AFTER
# appending the log row, so a failed trajectory computation leaves an old sidecar while the log
# advances — which would draw a sub-hour curve from a different cycle than the served hourly
# forecast. The sidecar records its own issue/anchor time; we serve it only when its issue time
# matches the current cycle's, and drop any point that predates the cycle anchor.
function _subhour_traj(log_path::AbstractString; cycle_issue=nothing)
    isempty(log_path) && return NamedTuple[]
    f = joinpath(dirname(log_path), "subhour_trajectory.json")
    isfile(f) || return NamedTuple[]
    try
        d = JSON3.read(read(f, String))
        if cycle_issue !== nothing && cycle_issue !== missing
            sc_issue = haskey(d, :issue_time_utc) ? parse_dt(get(d, :issue_time_utc, nothing)) : missing
            (sc_issue === missing || abs(sc_issue - cycle_issue) > _CYCLE_TOL) && return NamedTuple[]
        end
        anchor = haskey(d, :anchor_time_utc) ? parse_dt(get(d, :anchor_time_utc, nothing)) : missing
        out = NamedTuple[]
        for p in d.points
            t = parse_dt(String(p.t))
            t === missing && continue
            (anchor !== missing && t < anchor) && continue
            push!(out, (target_utc = String(p.t) * "Z", dst_nt = jnum(p.dst)))
        end
        return out
    catch e
        e isa InterruptException && rethrow()
        return NamedTuple[]
    end
end

# Staleness of a served cycle: age since the newest issue time, and whether every target hour is
# already in the past. A daemon crash or feed retirement freezes the log, so a cycle issued
# hours-to-days ago must not be served as the current operational status without a machine-readable
# flag. Threshold is tied to the hourly issue cadence.
const STALE_CYCLE_HOURS = 3.0   # hourly issue cadence; flag stale after ~3 missed issue cycles
const FUTURE_CYCLE_TOL_HOURS = 5 / 60  # tolerate only the documented intra-cycle clock spread

function _cycle_staleness(cyc::DataFrame)
    issue_max = (hasproperty(cyc, :issue_time_utc_dt) && !isempty(collect(skipmissing(cyc.issue_time_utc_dt)))) ?
                maximum(skipmissing(cyc.issue_time_utc_dt)) : missing
    age_hours = issue_max === missing ? nothing : (now(UTC) - issue_max) / Millisecond(3_600_000)
    invalid_future = age_hours !== nothing && age_hours < -FUTURE_CYCLE_TOL_HOURS
    stale = age_hours !== nothing && (age_hours > STALE_CYCLE_HOURS || invalid_future)
    tmax = (hasproperty(cyc, :target_time_utc_dt) && !isempty(collect(skipmissing(cyc.target_time_utc_dt)))) ?
           maximum(skipmissing(cyc.target_time_utc_dt)) : missing
    expired = tmax !== missing && tmax < now(UTC)
    return (age_hours=age_hours, stale=stale, expired=expired,
            invalid_future=invalid_future, issue_max=issue_max)
end

# A batch issues these four requested horizons sequentially. The log stores wall-clock lead in
# `horizon_hours`, so completeness is identified from the hourly target schedule instead.
const LIVE_CYCLE_HORIZONS = (1, 2, 3, 6)

function _common_cycle_field(cyc::DataFrame, name::Symbol)
    hasproperty(cyc, name) || return nothing
    values = [_rowget(r, name) for r in eachrow(cyc)]
    isempty(values) && return nothing
    any(x -> x === missing || x === nothing, values) && return nothing
    value = first(values)
    return all(x -> isequal(x, value), values) ? value : nothing
end

# The interval the super-learner stage serves under, and the per-row status that says the stage
# produced this row's center. The engine writes exactly this pair, and the two must agree row by row:
# the depth-stratified conformal band is calibrated on the V2.4e center, so a row carrying that
# source while its own status records a fallback would publish an interval nothing calibrated for the
# center it is drawn around.
const V2_4_INTERVAL_SOURCE = "v24_conformal_depth"
const V2_4_STATUS_OK = "ok"

_row_v2_4_acted(row) = (v = _rowget(row, :v24_status);
                        v isa AbstractString && String(v) == V2_4_STATUS_OK)

"""Per-row coherence of a cycle's interval disclosure with the stage that served each row.

V2.4 made `interval_source` a per-row field: a horizon whose super-learner stage acted carries the
study's depth-stratified conformal band, while a horizon whose stage could not act carries the band
the batch's own interval policy issued. Requiring one source across the four horizons therefore
rejects precisely the cycles that per-row disclosure exists to describe — a stage that healed or
failed between the horizons of one batch — and blanks the product over a degradation the log is
being honest about, exactly as accepting only one served label would.

Coherence is checked instead: a row records the V2.4e source when, and only when, its own status
records that the V2.4e stage produced its center, and every remaining row carries the one source the
batch issued them under. A cycle written before the per-row status existed carries no `v24_status`
column at all, and there the only statement the log can make is the batch one, so those cycles keep
the original common-source rule."""
function _coherent_interval_sources(cyc::DataFrame)
    hasproperty(cyc, :interval_source) || return false
    sources = String[]
    for r in eachrow(cyc)
        value = _rowget(r, :interval_source)
        (value isa AbstractString && !isempty(strip(value))) || return false
        push!(sources, String(value))
    end
    isempty(sources) && return false
    hasproperty(cyc, :v24_status) || return all(isequal(first(sources)), sources)
    batch = String[]
    for (index, r) in enumerate(eachrow(cyc))
        if _row_v2_4_acted(r)
            sources[index] == V2_4_INTERVAL_SOURCE || return false
        else
            sources[index] == V2_4_INTERVAL_SOURCE && return false
            push!(batch, sources[index])
        end
    end
    return isempty(batch) || all(isequal(first(batch)), batch)
end

"""Interval source the cycle is reported under, and every source its horizons carry.

A cycle served by one stage throughout has a single source and reports it. A cycle whose stage
changed between horizons is published under its weakest served label, so the reported source is read
from the rows carrying that label — the interval the reported product was actually issued with —
while the per-horizon sources travel with the horizons themselves."""
function _cycle_interval_source(cyc::DataFrame)
    common = _common_cycle_field(cyc, :interval_source)
    common === nothing || return String(common)
    label = _cycle_served_model(cyc)
    label === nothing && return nothing
    mask = [isequal(_rowget(r, :sub_hourly_model_version), label) for r in eachrow(cyc)]
    any(mask) || return nothing
    value = _common_cycle_field(cyc[mask, :], :interval_source)
    return value === nothing ? nothing : String(value)
end

_cycle_interval_sources(cyc::DataFrame) =
    sort(unique(String[String(v) for v in (_rowget(r, :interval_source) for r in eachrow(cyc))
                       if v isa AbstractString]))

"""Weakest served-pipeline label of a cycle, or `nothing` when a row carries an unknown label.

The stack stage is loaded per issuance and can heal or fail between the horizons of one cycle, so the
four rows may legitimately carry different accepted labels. Refusing such a cycle would blank the
dashboard and suppress its alerts over a disclosed per-row degradation, so a cycle is accepted when
every row carries an accepted label and the cycle is then reported under the weakest one: the label
that describes the least-capable stage any of its horizons was served by."""
function _cycle_served_model(cyc::DataFrame)
    hasproperty(cyc, :sub_hourly_model_version) || return nothing
    labels = [_rowget(r, :sub_hourly_model_version) for r in eachrow(cyc)]
    isempty(labels) && return nothing
    all(_is_accepted_served_model, labels) || return nothing
    for candidate in reverse(ACCEPTED_V2_SERVED_MODEL_VERSIONS)
        any(l -> isequal(l, candidate), labels) && return String(candidate)
    end
    return nothing
end

"""Driver-assumption token of the stage the cycle is reported under.

A cycle whose stack stage healed or failed between horizons carries more than one token, so the
common-field reading is empty and the payload would say the assumption was never recorded. The cycle
is published under its weakest served label, so the assumption is read from the rows carrying that
label: the sentence then describes the stage the reported product was actually served by."""
function _cycle_driver_assumption(cyc::DataFrame)
    hasproperty(cyc, :driver_assumption) || return nothing
    common = _common_cycle_field(cyc, :driver_assumption)
    common === nothing || return common
    label = _cycle_served_model(cyc)
    label === nothing && return nothing
    mask = [isequal(_rowget(r, :sub_hourly_model_version), label) for r in eachrow(cyc)]
    any(mask) || return nothing
    return _common_cycle_field(cyc[mask, :], :driver_assumption)
end

"""Validate one current forecast cycle as a complete, ordered, positive-lead trajectory."""
function _valid_live_cycle(cyc::DataFrame)
    nrow(cyc) == length(LIVE_CYCLE_HORIZONS) || return false
    model = _common_cycle_field(cyc, :model_version)
    served_model = _cycle_served_model(cyc)
    anchor = _common_cycle_field(cyc, :latest_dst_time_utc_dt)
    anchor_dst = jnum(_common_cycle_field(cyc, :latest_dst_nt))
    vintage = _common_cycle_field(cyc, :latest_solar_wind_utc_dt)
    model == CURRENT_V2_MODEL_VERSION || return false
    served_model === nothing && return false
    _coherent_interval_sources(cyc) || return false
    anchor isa DateTime && anchor_dst !== nothing && vintage isa DateTime || return false

    issues = DateTime[]
    targets = DateTime[]
    for r in eachrow(cyc)
        issue = _rowget(r, :issue_time_utc_dt)
        target = _rowget(r, :target_time_utc_dt)
        horizon = jnum(_rowget(r, :horizon_hours))
        pred = jnum(_v2_pred(r))
        lo = jnum(_v2_ci05(r))
        hi = jnum(_v2_ci95(r))
        issue isa DateTime && target isa DateTime || return false
        target > issue || return false
        target > anchor || return false
        horizon !== nothing && horizon > 0 || return false
        lead_hours = (target - issue) / Millisecond(3_600_000)
        abs(horizon - lead_hours) <= 0.05 || return false
        pred !== nothing && lo !== nothing && hi !== nothing || return false
        lo <= pred <= hi || return false
        push!(issues, issue)
        push!(targets, target)
    end
    issue_hour = floor(first(issues), Hour)
    all(issue -> floor(issue, Hour) == issue_hour, issues) || return false
    maximum(issues) - minimum(issues) <= _CYCLE_TOL || return false
    expected_targets = [issue_hour + Hour(h) for h in LIVE_CYCLE_HORIZONS]
    return targets == expected_targets
end

"""Forecast trajectory of the most recent cycle: anchor + per-horizon point and 90% band."""
function build_forecast(df::DataFrame, log_path::AbstractString="")
    served = _payload_cycle(df)
    cyc = served.cycle
    nrow(cyc) == 0 && return (available=false, issue_time_utc=nothing, horizons=[])
    stale = served.stale
    if stale.invalid_future || stale.expired || stale.stale
        reason = stale.invalid_future ? "future issue time" :
                 stale.expired ? "expired targets" : "stale issue time"
        return (available=false, issue_time_utc=jdt(stale.issue_max), horizons=[],
                stale=true, expired=stale.expired,
                invalid_future=stale.invalid_future,
                age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
                message="Latest forecast cycle is unavailable: $reason.")
    end
    _valid_live_cycle(cyc) ||
        return (available=false, issue_time_utc=jdt(stale.issue_max), horizons=[],
                stale=false, expired=false, invalid_future=false,
                message="Latest forecast cycle has incomplete or inconsistent rows.")
    horizons = NamedTuple[]
    for r in eachrow(cyc)
        target = _rowget(r, :target_time_utc_dt)
        pred = jnum(_v2_pred(r))
        lo = jnum(_v2_ci05(r))
        hi = jnum(_v2_ci95(r))
        push!(horizons, (target_utc=jdt(target),
                         horizon_hours=jnum(_rowget(r, :horizon_hours)),
                         pred_dst_nt=pred,
                         ci05_dst_nt=lo,
                         ci95_dst_nt=hi,
                         # Which interval this horizon's band came from. The super-learner stage acts
                         # per horizon, so a cycle can carry the study's depth-stratified conformal
                         # band on the horizons it served and the batch's own policy on the rest;
                         # the band a reader sees is only described by the row that produced it.
                         interval_source=(v = _rowget(r, :interval_source);
                                          v isa AbstractString ? String(v) : nothing),
                         # Alerting depth of this horizon and the V2.1 center it is taken against, so
                         # an operator can see why a warning is deeper than the served center.
                         severity_dst_nt=jnum(_severity_pred(r)),
                         severity_ci05_dst_nt=jnum(_severity_ci05(r)),
                         # Which stage's own band edge the watch is assessed on: the served product,
                         # one of the two predecessors, or the legacy center-shift rule for a row
                         # written before the predecessor edges were logged.
                         severity_ci05_source=_severity_ci05_source(r),
                         v2_1_served_pred_dst_nt=jnum(_rowget(r, :v2_1_served_pred_dst_nt)),
                         v2_2_stack_pred_dst_nt=jnum(_rowget(r, :v2_2_stack_pred_dst_nt)),
                         v2_1_served_ci05_dst_nt=jnum(_rowget(r, :v2_1_served_ci05_dst_nt)),
                         v2_2_stack_ci05_dst_nt=jnum(_rowget(r, :v2_2_stack_ci05_dst_nt)),
                         served_model_version=(v = _rowget(r, :sub_hourly_model_version);
                                               v isa AbstractString ? String(v) : nothing),
                         v2_2_status=(v = _rowget(r, :v2_2_status);
                                      v isa AbstractString ? String(v) : nothing),
                         # Per-row served-stage disclosure of the super-learner: `ok` when this
                         # horizon's center is the V2.4e center, otherwise the reason it fell back.
                         v24_status=(v = _rowget(r, :v24_status);
                                     v isa AbstractString ? String(v) : nothing),
                         v24_pred_dst_nt=jnum(_rowget(r, :v24_pred_dst_nt)),
                         v24_guard_applied=(v = _rowget(r, :v24_guard_applied);
                                            v isa Bool ? v : nothing),
                         # The physical +50/-2000 nT projection of the served center. Separate from the
                         # guard, and the reason the served center can differ from the combination's
                         # own output on a row where no guard acted.
                         v24_projection_applied=(v = _rowget(r, :v24_projection_applied);
                                                 v isa Bool ? v : nothing),
                         v24_regime_cell=(v = _rowget(r, :v24_regime_cell);
                                          v isa AbstractString ? String(v) : nothing),
                         frozen_tail_ablation_dst_nt=jnum(_audit_pred(r)),
                         audit_baseline_dst_nt=jnum(_audit_pred(r))))
    end
    issue = stale.issue_max
    cycle_issue = issue === missing ? nothing : issue
    return (available=true,
            issue_time_utc=jdt(issue),
            latest_solar_wind_utc=jdt(_common_cycle_field(cyc, :latest_solar_wind_utc_dt)),
            anchor_dst_nt=jnum(_common_cycle_field(cyc, :latest_dst_nt)),
            anchor_dst_time_utc=jdt(_common_cycle_field(cyc, :latest_dst_time_utc_dt)),
            interval_source=_cycle_interval_source(cyc),
            interval_sources=_cycle_interval_sources(cyc),
            # True when the newest issue hour was not a complete cycle and this payload is the
            # newest one that was. Availability is restored; the incompleteness is not masked, and
            # the health endpoint continues to report the newest issue hour as incomplete.
            superseded_cycle_incomplete=served.superseded,
            model_version=string(_common_cycle_field(cyc, :model_version)),
            served_model_version=string(_cycle_served_model(cyc)),
            served_product=served_product_name(_cycle_served_model(cyc)),
            stale=stale.stale, expired=stale.expired, invalid_future=stale.invalid_future,
            age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
            recent_observed=_recent_observed(df),
            subhour_trajectory=_subhour_traj(log_path; cycle_issue=cycle_issue),
            horizons=horizons)
end

"""Storm-time replay summary: serves the offline replay report and its scored-row provenance.
Defensive (never throws): missing/unreadable artifacts return available=false."""
function build_storm_replay(log_path::AbstractString; evidence_dir=nothing)
    dir = evidence_dir === nothing ? dirname(log_path) : String(evidence_dir)
    report = joinpath(dir, "storm_replay_report.md")
    scored = joinpath(dir, "storm_replay_scored.csv")
    isfile(report) || return (available=false,
                              reason="no storm-replay report; run validation/operational/storm_replay.jl")
    md = try
        read(report, String)
    catch e
        e isa InterruptException && rethrow()
        return (available=false, reason="report unreadable")
    end
    age = round((time() - mtime(report)) / 60; digits=1)
    n = 0; storms = String[]
    if isfile(scored)
        try
            df = CSV.read(scored, DataFrame)
            n = nrow(df)
            "storm" in names(df) && (storms = unique(string.(skipmissing(df.storm))))
        catch e
            e isa InterruptException && rethrow()
        end
    end
    return (available=true, report_age_min=age, n_scored=n, storms=storms, report_markdown=md)
end

# Number of trailing issue cycles the served/shadow stage health is summarized over. One cycle per
# hour, so this is a day of issuance: long enough that a single transient load failure is visible as a
# rate rather than as a verdict, short enough that a healed deployment clears it within a day.
const SERVED_HEALTH_CYCLES = 24

"""Trailing served- and shadow-stage health of the log, grouped by issue hour.

The served stack and the shadow deployment are both loaded per issuance and both fail closed, so a
degradation shows up as rows that carry the fallback label or an unavailable shadow status rather than
as a missing log. Summarizing them here is what lets the health endpoint state which product is
actually being served instead of only whether a file is fresh."""
function build_served_health(df::DataFrame, cycles::Integer=SERVED_HEALTH_CYCLES)
    empty_result = (cycles_considered=0, pre_stage_cycles_excluded=0, served_model_version=nothing,
                    served_product=nothing, served_fallback_cycles=0, served_fallback_rate=nothing,
                    served_fallback_window_cycles=Int(cycles),
                    window_start_utc=nothing, window_end_utc=nothing, window_span_hours=nothing,
                    served_fallback_cycles_24h=0, served_fallback_window_24h_cycles=0,
                    served_fallback_rate_24h=nothing,
                    newest_cycle_is_fallback=nothing, served_stack_cycles=0,
                    served_v2_1_cycles=0, shadow_model_version=nothing,
                    shadow_cycles_considered=0, shadow_available_cycles=0,
                    shadow_available_rate=nothing, shadow_e_layer_cycles=0,
                    shadow_e_layer_rate=nothing)
    nrow(df) == 0 && return empty_result
    hasproperty(df, :issue_time_utc_dt) || return empty_result
    hours = [issue === missing ? missing : floor(issue, Hour) for issue in df.issue_time_utc_dt]
    keys_present = sort(unique(collect(skipmissing(hours))))
    isempty(keys_present) && return empty_result
    window = keys_present[max(1, length(keys_present) - Int(cycles) + 1):end]
    # The window is the last `cycles` issue hours PRESENT IN THE LOG, which is a day of issuance only
    # while issuance is unbroken. After an outage those hours can span days, and the rate would be
    # described as a trailing day while summarizing cycles far older than that. The window's own
    # bounds and span are published so the rate cannot be read as something it is not, and a
    # wall-clock-bounded rate over the last 24 h of issuance is reported beside it.
    window_start = first(window)
    window_end = last(window)
    window_span = round((window_end - window_start) / Millisecond(3_600_000); digits=1)
    day_cutoff = window_end - Hour(24)
    fallback_24h = 0
    staged_24h = 0
    fallback = 0
    staged = 0
    pre_stage = 0
    stack_served = 0
    v2_1_served = 0
    shadow_staged = 0
    shadow_ok = 0
    e_layer = 0
    newest_fallback = nothing
    shadow_labels = String[]
    for (index, hour) in enumerate(window)
        rows = df[coalesce.(hours .== hour, false), :]
        labels = [_rowget(r, :sub_hourly_model_version) for r in eachrow(rows)]
        # Every predicate here is written with `isequal`/`isa` rather than `==`, because a log that
        # was extended with the shadow columns while it was already being appended to carries
        # `missing` in those columns for its earlier rows. `==` against `missing` is three-valued, so
        # `any` would return `missing` and the health summary would throw instead of reporting the
        # trailing window that spans the schema change.
        #
        # Only cycles issued by the current served stage (they carry `v24_status`) enter the
        # fallback rate; rows written before the stage existed are disclosed as excluded rather
        # than counted as fallbacks, mirroring the readiness audit's transition rule.
        cycle_staged = any(r -> _rowget(r, :v24_status) isa AbstractString, eachrow(rows))
        if cycle_staged
            staged += 1
            is_fallback = any(label -> !isequal(label, CURRENT_V2_SERVED_MODEL_VERSION), labels)
            is_fallback && (fallback += 1)
            if hour >= day_cutoff
                staged_24h += 1
                is_fallback && (fallback_24h += 1)
            end
            any(label -> isequal(label, STACK_V2_SERVED_MODEL_VERSION), labels) &&
                (stack_served += 1)
            any(label -> isequal(label, PREVIOUS_V2_SERVED_MODEL_VERSION), labels) &&
                (v2_1_served += 1)
            index == length(window) && (newest_fallback = is_fallback)
        else
            pre_stage += 1
        end
        statuses = [_rowget(r, :v23_status) for r in eachrow(rows)]
        if any(st -> st isa AbstractString, statuses)
            shadow_staged += 1
            any(st -> st isa AbstractString && startswith(st, "ok"), statuses) && (shadow_ok += 1)
            applied = [_rowget(r, :v23_e_layer_applied) for r in eachrow(rows)]
            any(flag -> flag === true || isequal(flag, 1), applied) && (e_layer += 1)
        end
        for label in (_rowget(r, :v23_shadow_model_version) for r in eachrow(rows))
            label isa AbstractString && push!(shadow_labels, String(label))
        end
    end
    newest = df[coalesce.(hours .== last(keys_present), false), :]
    served_label = _cycle_served_model(newest)
    return (cycles_considered=staged,
            pre_stage_cycles_excluded=pre_stage,
            served_model_version=served_label,
            served_product=served_label === nothing ? nothing : served_product_name(served_label),
            served_fallback_cycles=fallback,
            served_fallback_rate=staged == 0 ? nothing : round(fallback / staged; digits=4),
            # What the rate above is measured over: the requested cycle count, the issue hours it
            # actually spans, and the same rate restricted to the trailing 24 h of issuance.
            served_fallback_window_cycles=Int(cycles),
            window_start_utc=jdt(window_start),
            window_end_utc=jdt(window_end),
            window_span_hours=window_span,
            served_fallback_cycles_24h=fallback_24h,
            served_fallback_window_24h_cycles=staged_24h,
            served_fallback_rate_24h=staged_24h == 0 ? nothing :
                round(fallback_24h / staged_24h; digits=4),
            newest_cycle_is_fallback=newest_fallback,
            served_stack_cycles=stack_served,
            served_v2_1_cycles=v2_1_served,
            shadow_model_version=isempty(shadow_labels) ? nothing : first(sort(unique(shadow_labels))),
            shadow_cycles_considered=shadow_staged,
            shadow_available_cycles=shadow_ok,
            shadow_available_rate=shadow_staged == 0 ? nothing : round(shadow_ok / shadow_staged; digits=4),
            shadow_e_layer_cycles=e_layer,
            shadow_e_layer_rate=shadow_staged == 0 ? nothing : round(e_layer / shadow_staged; digits=4))
end

"""Threat status: current observation and most negative edge among the latest 90% intervals."""
function build_status(df::DataFrame)
    served = _payload_cycle(df)
    cyc = served.cycle
    cal = calibration_summary(df)
    if nrow(cyc) == 0
        return (generated_utc=jdt(now(UTC)), available=false,
                message="No forecast rows in log yet.", calibration=cal)
    end
    stale = served.stale
    if stale.invalid_future
        return (generated_utc=jdt(now(UTC)), available=false, stale=true, expired=false,
                invalid_future=true,
                age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
                forecast_issue_utc=jdt(stale.issue_max),
                message="Latest forecast cycle has an issue time too far in the future.",
                calibration=cal)
    end
    if stale.expired
        # Every target hour is in the past: the daemon has stopped issuing (crash or feed
        # retirement). Serve available=false with an explicit stale/expired flag so build_alerts
        # suppresses forecast/watch alerts and the dashboard does not present an expired forecast
        # as the current threat.
        return (generated_utc=jdt(now(UTC)), available=false, stale=true, expired=true,
                age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
                forecast_issue_utc=jdt(stale.issue_max),
                message="Latest forecast cycle expired; all target hours are in the past.",
                calibration=cal)
    end
    if stale.stale
        return (generated_utc=jdt(now(UTC)), available=false, stale=true,
                expired=false, invalid_future=false,
                age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
                forecast_issue_utc=jdt(stale.issue_max),
                message="Latest forecast cycle is stale.", calibration=cal)
    end
    if !_valid_live_cycle(cyc)
        return (generated_utc=jdt(now(UTC)), available=false, stale=stale.stale,
                expired=false, invalid_future=false,
                age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
                forecast_issue_utc=jdt(stale.issue_max),
                message="Latest forecast cycle has incomplete or invalid point/interval rows.",
                calibration=cal)
    end
    preds = [Float64(_severity_pred(r)) for r in eachrow(cyc)]
    # Watch edge on the depth-safe edge, not on the served band as issued: a candidate can report a
    # shallower center and a narrower band than the stages it replaced, and a watch assessed on that
    # band could drop a tier the previous product raised on the same physics. `_severity_ci05` takes
    # the deepest of the served edge and each predecessor's own logged edge.
    edges = [_severity_ci05_with_source(r) for r in eachrow(cyc)]
    lbs = [Float64(first(e)) for e in edges]
    point_min = minimum(preds)
    interval_lower_edge_min = minimum(lbs)
    interval_lower_edge_source = last(edges[argmin(lbs)])
    lvl_pt, lbl_pt = dst_threat_level(point_min)
    lvl_wc, lbl_wc = dst_threat_level(interval_lower_edge_min)
    # Reported threat level is the point-forecast level; a "watch" flag fires when the
    # lower edge of a displayed 90%-target interval reaches a stronger storm tier than the
    # point forecast. This does not turn the marginal interval into a one-sided bound.
    watch = lvl_wc > lvl_pt
    horizon_max = (h = filter(!isnothing,
        [jnum(_rowget(r, :horizon_hours)) for r in eachrow(cyc)]);
        isempty(h) ? nothing : maximum(h))
    return (generated_utc=jdt(now(UTC)),
            available=true,
            stale=stale.stale, expired=false,
            # The newest issue hour was not a complete cycle; this is the newest one that was, and
            # its true age is reported below. `/api/health` still calls the newest hour incomplete.
            superseded_cycle_incomplete=served.superseded,
            age_hours=(stale.age_hours === nothing ? nothing : round(stale.age_hours; digits=2)),
            latest_observation=(dst_nt=jnum(_common_cycle_field(cyc, :latest_dst_nt)),
                                time_utc=jdt(_common_cycle_field(cyc, :latest_dst_time_utc_dt))),
            latest_solar_wind_utc=jdt(_common_cycle_field(cyc, :latest_solar_wind_utc_dt)),
            forecast_issue_utc=jdt(stale.issue_max),
            threat=(level=lvl_pt, label=lbl_pt, watch=watch,
                    watch_level=lvl_wc, watch_label=lbl_wc,
                    point_min_dst_nt=point_min,
                    interval_lower_edge_min_dst_nt=interval_lower_edge_min,
                    # The stage whose own band edge the published watch edge came from.
                    interval_lower_edge_source=interval_lower_edge_source,
                    # Deprecated value-equivalent aliases for already-loaded dashboards and API
                    # clients. New code and all visible labels use interval_lower_edge_min_dst_nt.
                    lower_bound_min_dst_nt=interval_lower_edge_min,
                    worst_credible_dst_nt=interval_lower_edge_min,
                    basis="Dst storm-intensity scale (-30/-50/-100/-200 nT)"),
            lead_time=(forecast_horizon_hours=horizon_max,
                       driver_assumption=driver_assumption_text(
                           _cycle_driver_assumption(cyc)),
                       physical_upstream_lead_min=[30, 60],
                       note="Genuine upstream lead for new severity is the L1 advection time (~30-60 min). " *
                            "Multi-day lead requires CME eruption/propagation models, not yet in this system."),
            calibration=cal,
            model_version=string(_common_cycle_field(cyc, :model_version)),
            served_model_version=string(_cycle_served_model(cyc)),
            served_product=served_product_name(_cycle_served_model(cyc)),
            served_model_versions=sort(unique(String[
                String(v) for v in (_rowget(r, :sub_hourly_model_version) for r in eachrow(cyc))
                if v isa AbstractString
            ])))
end

function _build_history_uncached(df::DataFrame, hours::Real, reference_time::DateTime)
    v = verified_rows(df)
    nrow(v) == 0 && return (rows=[], coverage_90=nothing, rmse_nt=nothing, hours=hours)
    cutoff = reference_time - Hour(round(Int, hours))
    rows = NamedTuple[]
    for r in eachrow(v)
        t = _rowget(r, :target_time_utc_dt)
        (t === missing || t < cutoff) && continue
        obs = jnum(_rowget(r, :observation_dst_nt)); p = jnum(_v2_pred(r))
        lo = jnum(_v2_ci05(r)); hi = jnum(_v2_ci95(r))
        inside = (obs !== nothing && lo !== nothing && hi !== nothing) ? (obs >= lo && obs <= hi) : nothing
        push!(rows, (target_utc=jdt(t), horizon_hours=jnum(_rowget(r, :horizon_hours)),
                     observed_dst_nt=obs, pred_dst_nt=p, ci05_dst_nt=lo, ci95_dst_nt=hi,
                     frozen_tail_ablation_dst_nt=jnum(_audit_pred(r)),
                     audit_baseline_dst_nt=jnum(_audit_pred(r)),
                     inside_90ci=inside))
    end
    sort!(rows, by=x -> something(x.target_utc, ""))
    # Stats must match the WINDOW the rows show, not the whole log. Compute coverage/RMSE over
    # the windowed rows; also expose the (more robust) all-log figures under explicit names.
    flags = [r.inside_90ci for r in rows if r.inside_90ci !== nothing]
    paired = [(r.observed_dst_nt, r.pred_dst_nt) for r in rows
              if r.observed_dst_nt !== nothing && r.pred_dst_nt !== nothing]
    win_cov = isempty(flags) ? nothing : round(count(flags) / length(flags); digits=3)
    window_value = isempty(paired) ? nothing : _stable_rmse_or_nothing(
        first.(paired), last.(paired),
    )
    win_rmse = window_value === nothing ? nothing : round(window_value; digits=2)
    cal = calibration_summary(df)
    return (rows=rows, hours=hours, n=length(rows),
            coverage_90=win_cov, rmse_nt=win_rmse,
            coverage_90_all=_prefer_metric(cal.v2_coverage_90, cal.coverage_90),
            rmse_nt_all=_prefer_metric(cal.v2_rmse_nt, cal.rmse_nt))
end


"""Recent verified track record (observed vs predicted with band) for the last `hours`."""
function build_history(df::DataFrame, hours::Real=72)
    reference_time = now(UTC)
    log_key = _cached_log_key(df)
    log_key === nothing && return _build_history_uncached(df, hours, reference_time)
    key = (log_key, Float64(hours), floor(reference_time, Hour))
    return lock(_HISTORY_LOCK) do
        cached = _HISTORY_CACHE[]
        if cached === nothing || cached[1] != key
            cached = (key, _build_history_uncached(df, hours, reference_time))
            _HISTORY_CACHE[] = cached
        end
        return cached[2]
    end
end

# Whole-nanotesla text of an alerting depth. `round(x; digits=0)` keeps the value a Float64, so the
# webhook and the API read "-120.0 nT" for a depth the dashboard shows as "-120": the same number
# written two ways across the surfaces an operator reads together. Ties round away from zero, which
# is what the dashboard's own fixed-decimal formatting does.
#
# Total by construction. This runs inside the alert builder, so a value it cannot render must
# degrade to the value written out, never to an exception: a corrupt depth in the log would
# otherwise take `/api/alerts` and the webhook down instead of showing an obviously wrong number.
# `round(Int, ...)` throws on a finite magnitude no Int64 can hold, so the machine range is checked
# before the conversion rather than after it.
function _alert_depth_nt(x::Real)
    isfinite(x) || return string(x)
    rounded = round(float(x), RoundNearestTiesAway)
    # Exact mixed-type comparison: a Float64 outside the Int64 range fails it, and one inside is
    # integral and converts exactly.
    (typemin(Int) <= rounded <= typemax(Int)) || return string(x)
    return string(Int(rounded))
end

"""Active alert summary derived from the current threat status."""
function build_alerts(df::DataFrame, st=build_status(df))
    getproperty(st, :available) == false && return (active=false, alerts=[], generated_utc=st.generated_utc)
    th = st.threat
    alerts = NamedTuple[]
    if th.level >= 1
        push!(alerts, (severity=th.label, level=th.level, kind="forecast",
                       message="Forecast Dst reaches $(_alert_depth_nt(th.point_min_dst_nt)) nT " *
                               "($(th.label)) within the next $(round(something(st.lead_time.forecast_horizon_hours, 0.0); digits=1)) h."))
    end
    if th.watch && th.watch_level > th.level
        push!(alerts, (severity=th.watch_label, level=th.watch_level, kind="watch",
                       message="A displayed 90% target interval extends to " *
                               "$(_alert_depth_nt(th.interval_lower_edge_min_dst_nt)) nT " *
                               "($(th.watch_label) range) at one or more horizons."))
    end
    return (active=!isempty(alerts), generated_utc=st.generated_utc,
            threat_level=th.level, threat_label=th.label, alerts=alerts)
end
