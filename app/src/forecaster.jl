# forecaster.jl — offline dB/dt replay using the exported ridge model and historical
# normalized-residual sample.
#
# Replays a next-30-minute maximum from a fixed historical residual sample: a point estimate,
# an empirical 0.90 residual-quantile upper estimate, and empirical exceedance scores at four
# threshold magnitudes. The artifact was trained with bow-shock-shifted OMNI drivers; the
# available live feed is unshifted L1, so callers must explicitly request offline replay.
#
# The residual distribution is fixed from the historical calibration sample; it is not updated
# online, is not a per-issue coverage guarantee, and is not served by the package API.

using JSON3, Statistics, Dates

const _FORECASTER = Dict{String,Any}()
const _FORECASTER_LOCK = ReentrantLock()
const FORECASTER_STATIONS = ("FRD", "CMO")

function _valid_forecaster_threshold(x::Real)
    x isa Bool && return false
    isfinite(x) && x > 0 && isinteger(x) || return false
    try
        Int(x)
        return true
    catch e
        e isa InterruptException && rethrow()
        return false
    end
end

function _valid_forecaster_artifact(a, station::AbstractString)
    required = (:artifact_schema_version, :station, :features, :mu, :sigma, :beta,
                :rn_calib, :thresholds, :training_ground_data_product,
                :live_ground_data_product, :training_driver_product,
                :training_driver_propagation, :available_live_driver_product,
                :driver_time_reference_aligned, :serving_enabled, :serving_blocker,
                :forecast_horizon_minutes, :issue_cadence_minutes,
                :overlapping_targets)
    all(key -> haskey(a, key), required) || return false
    a.artifact_schema_version == 3 || return false
    string(a.station) == station || return false
    string(a.training_ground_data_product) == "quasi-definitive" || return false
    string(a.live_ground_data_product) == "adjusted (provisional)" || return false
    string(a.training_driver_product) == "OMNI_HRO_1MIN" || return false
    string(a.training_driver_propagation) == "bow-shock-nose time shifted" || return false
    string(a.available_live_driver_product) ==
        "NOAA SWPC real-time solar wind at L1" || return false
    a.driver_time_reference_aligned === false || return false
    a.serving_enabled === false || return false
    !isempty(strip(string(a.serving_blocker))) || return false
    a.forecast_horizon_minutes == 30 || return false
    a.issue_cadence_minutes == 1 || return false
    a.overlapping_targets === true || return false
    features = string.(a.features)
    features == ["dbdt_now", "dbdt_mean30", "dbdt_max30", "dbdt_std30",
                 "V", "Bz", "Bs", "VBs"] || return false
    μ = try Float64.(a.mu) catch e; e isa InterruptException && rethrow(); return false end
    σ = try Float64.(a.sigma) catch e; e isa InterruptException && rethrow(); return false end
    β = try Float64.(a.beta) catch e; e isa InterruptException && rethrow(); return false end
    rn = try Float64.(a.rn_calib) catch e; e isa InterruptException && rethrow(); return false end
    raw_thresholds = try collect(a.thresholds) catch e; e isa InterruptException && rethrow(); return false end
    all(x -> x isa Real && !(x isa Bool), raw_thresholds) || return false
    thresholds = try Float64.(raw_thresholds) catch e; e isa InterruptException && rethrow(); return false end
    length(μ) == 8 && length(σ) == 8 && length(β) == 9 || return false
    !isempty(rn) && !isempty(thresholds) || return false
    all(isfinite, μ) && all(x -> isfinite(x) && x > 0, σ) && all(isfinite, β) || return false
    all(isfinite, rn) && issorted(rn) || return false
    all(_valid_forecaster_threshold, thresholds) || return false
    return true
end

function load_forecaster(; station::AbstractString = "FRD")
    station = uppercase(strip(String(station)))
    station in FORECASTER_STATIONS || return nothing
    # Guard the check-then-load-then-store like the sibling caches (_DBDT/_SWPC/_NET/_LOG):
    # the file read between the haskey miss and the store is a yield point, so two concurrent
    # first-time pollers could otherwise both load (and, multi-threaded, race the setindex!).
    lock(_FORECASTER_LOCK) do
        haskey(_FORECASTER, station) && return _FORECASTER[station]
        path = joinpath(@__DIR__, "..", "models", "forecaster_$(station).json")
        isfile(path) || return nothing
        try
            artifact = JSON3.read(read(path, String))
            _valid_forecaster_artifact(artifact, station) || begin
                @warn "forecaster artifact failed schema validation" station
                return nothing
            end
            _FORECASTER[station] = artifact
        catch e
            e isa InterruptException && rethrow()
            @warn "forecaster artifact load failed" station exception=e; return nothing
        end
        return get(_FORECASTER, station, nothing)
    end
end

# Features (must match export): dbdt_now, dbdt_mean30, dbdt_max30, dbdt_std30, V, Bz, Bs, VBs.
function forecast_dbdt(dbdt_recent::AbstractVector, V, Bz;
                       station::AbstractString = "FRD", offline_replay::Bool=false)
    offline_replay || return nothing
    station = uppercase(strip(String(station)))
    a = load_forecaster(; station=station)
    (a === nothing || V === nothing || Bz === nothing) && return nothing
    length(dbdt_recent) >= 30 || return nothing
    w = Vector{Float64}(undef, 30)
    source_start = length(dbdt_recent) - 29
    for (destination, source) in enumerate(source_start:length(dbdt_recent))
        value = jnum(dbdt_recent[source])
        (value === nothing || value < 0) && return nothing
        w[destination] = value
    end
    dnow = w[end]; dmean = mean(w); dmax = maximum(w); dstd = length(w) > 1 ? std(w) : 0.0
    Vf = jnum(V); Bzf = jnum(Bz)
    (Vf === nothing || Bzf === nothing || Vf <= 0 || Vf > 5_000 || abs(Bzf) > 1_000) && return nothing
    Bs = max(-Bzf, 0.0); VBs = Vf * Bs
    x = Float64[dnow, dmean, dmax, dstd, Vf, Bzf, Bs, VBs]
    μ = Float64.(a.mu); σ = Float64.(a.sigma); β = Float64.(a.beta)
    length(x) == length(μ) || return nothing
    zf = (x .- μ) ./ σ
    all(isfinite, zf) || return nothing
    ẑ = β[1] + sum(β[2:end] .* zf)               # bias + standardized features (log1p target)
    isfinite(ẑ) || return nothing
    s = log1p(dmax) + 1.0                         # conformal scale
    rn = Float64.(a.rn_calib); n = length(rn)
    # rn_calib[i] holds the (i-1)/(n-1) quantile of the conformal residuals; interpolate to the
    # exact p=0.90 quantile so the live bound matches export_forecaster.jl's quantile(rnca, 0.90)
    # rather than landing one grid step low at rn[round(0.9n)].
    fpos = 0.90 * (n - 1) + 1
    ilo = clamp(floor(Int, fpos), 1, n); ihi = clamp(ilo + 1, 1, n)
    q90 = rn[ilo] + (fpos - ilo) * (rn[ihi] - rn[ilo])
    zcap = log(2001.0)                            # numerical safety cap: ridge extrapolation in log
    point = expm1(min(ẑ, zcap))                   # space + expm1 can blow up out of support
    ub90 = expm1(min(ẑ + q90 * s, zcap))
    exc = [(threshold = Int(thr),
            empirical_score = round(count(>( (log1p(thr) - ẑ) / s ), rn) / n; digits=3))
           for thr in a.thresholds]
    # Reliability guard: the log-space ridge extrapolates catastrophically when replay
    # features sit far outside the calibration support, and the expm1 then saturates at the
    # ~2000 nT/min cap with ~1.0 exceedance. Flag (do not hide) such forecasts so the
    # diagnostics can mark the estimate out of range instead of treating the cap as evidence.
    Z_SUPPORT = 6.0                                # |standardized feature| beyond ~6σ ⇒ extrapolation
    out_of_range = any(>(Z_SUPPORT), abs.(zf))
    saturated = (ẑ >= zcap) || (ẑ + q90 * s >= zcap)
    return (point_dbdt = round(max(point, 0.0); digits=2),
            ub90_dbdt = round(max(ub90, 0.0); digits=2),
            exceedance = exc, station = station,
            horizon_min = 30, source = "offline ridge + historical residual sample",
            training_ground_data_product = string(a.training_ground_data_product),
            live_ground_data_product = string(a.live_ground_data_product),
            training_driver_product = string(a.training_driver_product),
            training_driver_propagation = string(a.training_driver_propagation),
            driver_time_reference_aligned = false, serving_enabled = false,
            reliable = !(out_of_range || saturated),
            out_of_validated_range = out_of_range, saturated = saturated)
end

"""Forecast from timestamped dB/dt observations only when the last 30 samples are consecutive."""
function forecast_dbdt_observations(series::AbstractVector, V, Bz;
                                    station::AbstractString = "FRD",
                                    offline_replay::Bool=false)
    offline_replay || return nothing
    length(series) >= 30 || return nothing
    rows = series[(end - 29):end]
    times = Vector{DateTime}(undef, 30)
    values = Vector{Float64}(undef, 30)
    for index in eachindex(rows)
        timestamp = parse_dt(get(rows[index], :t, nothing))
        value = jnum(get(rows[index], :dbdt, nothing))
        (timestamp === missing || value === nothing || value < 0) && return nothing
        times[index] = timestamp
        values[index] = value
    end
    all(index -> times[index] - times[index - 1] == Minute(1), 2:30) ||
        return nothing
    return forecast_dbdt(values, V, Bz; station=station, offline_replay=true)
end
