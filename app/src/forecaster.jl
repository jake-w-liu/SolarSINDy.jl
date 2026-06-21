# forecaster.jl — live calibrated dB/dt FORECAST using the exported paper3 conformal model.
#
# Turns the live dB/dt history (USGS) + live solar wind (SWPC) into a calibrated forecast of
# the next-30-min MAX dB/dt: a point forecast, a 90% conformal upper bound, and exceedance
# probabilities P(max dB/dt > threshold) at the Pulkkinen levels. Uses the artifact exported by
# paper3/scripts/export_forecaster.jl (ridge point model + empirical conformal residual CDF).
#
# This is the paper3 method applied live — a FORECAST, distinct from the observed dB/dt nowcast.

using JSON3, Statistics

const _FORECASTER = Dict{String,Any}()

function load_forecaster(; station::AbstractString = "FRD")
    haskey(_FORECASTER, station) && return _FORECASTER[station]
    path = joinpath(@__DIR__, "..", "models", "forecaster_$(station).json")
    isfile(path) || return nothing
    try
        _FORECASTER[station] = JSON3.read(read(path, String))
    catch e
        @warn "forecaster artifact load failed" station exception=e; return nothing
    end
    return get(_FORECASTER, station, nothing)
end

# Features (must match export): dbdt_now, dbdt_mean30, dbdt_max30, dbdt_std30, V, Bz, Bs, VBs.
function forecast_dbdt(dbdt_recent::AbstractVector, V, Bz; station::AbstractString = "FRD")
    a = load_forecaster(; station=station)
    (a === nothing || V === nothing || Bz === nothing) && return nothing
    vals = Float64[x for x in dbdt_recent if x !== nothing && isfinite(x)]
    length(vals) < 5 && return nothing
    w = vals[max(1, end-29):end]                 # trailing 30 min
    dnow = w[end]; dmean = mean(w); dmax = maximum(w); dstd = length(w) > 1 ? std(w) : 0.0
    Vf = Float64(V); Bzf = Float64(Bz); Bs = max(-Bzf, 0.0); VBs = Vf * Bs
    x = Float64[dnow, dmean, dmax, dstd, Vf, Bzf, Bs, VBs]
    μ = Float64.(a.mu); σ = Float64.(a.sigma); β = Float64.(a.beta)
    length(x) == length(μ) || return nothing
    zf = (x .- μ) ./ σ
    ẑ = β[1] + sum(β[2:end] .* zf)               # bias + standardized features (log1p target)
    s = log1p(dmax) + 1.0                         # conformal scale
    rn = Float64.(a.rn_calib); n = length(rn)
    # rn_calib[i] holds the (i-1)/(n-1) quantile of the conformal residuals; interpolate to the
    # exact p=0.90 quantile so the live bound matches export_forecaster.jl's quantile(rnca, 0.90)
    # rather than landing one grid step low at rn[round(0.9n)].
    fpos = 0.90 * (n - 1) + 1
    ilo = clamp(floor(Int, fpos), 1, n); ihi = clamp(ilo + 1, 1, n)
    q90 = rn[ilo] + (fpos - ilo) * (rn[ihi] - rn[ilo])
    zcap = log(2001.0)                            # physical cap ~2000 nT/min: ridge extrapolation in
    point = expm1(min(ẑ, zcap))                   # log space + expm1 can blow up on out-of-range input
    ub90 = expm1(min(ẑ + q90 * s, zcap))
    exc = [(threshold = Int(thr),
            prob = round(count(>( (log1p(thr) - ẑ) / s ), rn) / n; digits=3))
           for thr in a.thresholds]
    return (point_dbdt = round(max(point, 0.0); digits=2),
            ub90_dbdt = round(ub90; digits=2),
            exceedance = exc, station = station,
            horizon_min = 30, source = "paper3 online conformal")
end
