# Split-conformal predictive intervals for the operational Dst forecaster.
#
# Distribution-free, finite-sample coverage on top of any point forecast. The
# nonconformity score is the absolute residual |obs - point|; the interval is
# point ± Q̂, where Q̂ is the finite-sample-corrected (1-α) empirical quantile of
# calibration residuals. Quantiles are STRATIFIED by forecast horizon and by
# geomagnetic activity regime, because storm-time residuals are heavier-tailed
# than quiet-time residuals and longer horizons carry larger error — a single
# pooled interval would over-cover quiet/short cases and under-cover
# storm/long-lead cases.
#
# Reference: split conformal prediction (Vovk, Gammerman & Shafer, 2005;
# Lei et al., 2018). The marginal guarantee is P(|Y - Ŷ| ≤ Q̂) ≥ k/(n+1) with
# k = ⌈(n+1)(1-α)⌉, achieved exactly when calibration and test residuals are
# exchangeable.

"""
    ConformalStratum

Per-stratum calibration record: the stratum label, the calibration count `n`,
the conformal half-width, and the guaranteed coverage floor `k/(n+1)` actually
achieved by that half-width (≤ the nominal level when `n` is small).
"""
struct ConformalStratum
    key::Symbol
    n::Int
    half_width::Float64
    coverage_floor::Float64
end

"""
    ConformalCalibration

Stratified split-conformal calibration. Holds one [`ConformalStratum`](@ref) per
occupied (horizon-bin × activity-regime) cell plus a pooled global fallback used
when a stratum has fewer than `min_stratum_n` calibration points.
"""
struct ConformalCalibration
    coverage::Float64                      # nominal target, e.g. 0.90
    horizon_edges::Vector{Float64}         # ascending lead-time bin edges [hr]
    activity_threshold_nt::Float64         # latest Dst ≤ threshold ⇒ :disturbed
    min_stratum_n::Int                     # below this, use the global fallback
    strata::Dict{Symbol,ConformalStratum}
    global_stratum::ConformalStratum
end

# Defaults are operational choices, justified rather than arbitrary:
# - horizon edges separate the common operational lead times 1, 2, 3, and >=6 h
#   into distinct bins ([0,1.5), [1.5,2.5), [2.5,4.5), [4.5,Inf)). An earlier
#   [0,1.5,3.5,Inf) default merged the 2 h and 3 h leads, which under-covered the
#   3 h stratum (it borrowed the narrower 2 h half-width); offline OMNI replay
#   coverage testing surfaced this, hence the finer binning.
const CONFORMAL_HORIZON_EDGES = [0.0, 1.5, 2.5, 4.5, Inf]
# - activity split at −30 nT: below quiet baseline, capturing pre-storm/active
#   conditions where residuals fatten, but above the −50 nT storm threshold.
const CONFORMAL_ACTIVITY_THRESHOLD_NT = -30.0
# - need enough points for a stable (1-α) quantile: ⌈(20+1)·0.9⌉ = 19 ≤ 20.
const CONFORMAL_MIN_STRATUM_N = 20

"""
    _conformal_quantile(residuals, coverage)

Finite-sample split-conformal half-width: the `k`-th smallest absolute residual
with `k = ⌈(n+1)·coverage⌉`, clamped to `n` (using the max residual when the
sample is too small to reach the nominal level). Returns `(half_width,
coverage_floor)` where `coverage_floor = k/(n+1)` is the guaranteed marginal
coverage actually delivered (never overstated).
"""
function _conformal_quantile(residuals::AbstractVector{<:Real}, coverage::Real)
    finite = Float64[abs(r) for r in residuals if isfinite(r)]
    n = length(finite)
    n >= 1 || throw(ArgumentError("need ≥ 1 finite residual for a conformal quantile"))
    sorted = sort(finite)
    k = ceil(Int, (n + 1) * coverage)
    k_eff = clamp(k, 1, n)
    return sorted[k_eff], k_eff / (n + 1)
end

"""
    _activity_regime(latest_dst, threshold)

Causal activity label from the issue-time observed Dst: `:disturbed` when
`latest_dst ≤ threshold`, else `:quiet`. Non-finite Dst defaults to `:disturbed`
(the wider, safer interval).
"""
function _activity_regime(latest_dst::Real, threshold::Real)
    isfinite(latest_dst) || return :disturbed
    return latest_dst <= threshold ? :disturbed : :quiet
end

"""
    _horizon_bin(horizon, edges)

Index of the half-open lead-time bin `[edges[i], edges[i+1])` containing
`horizon` (clamped into range). `edges` must be ascending with ≥ 2 entries.
"""
function _horizon_bin(horizon::Real, edges::AbstractVector{<:Real})
    length(edges) >= 2 || throw(ArgumentError("horizon_edges needs ≥ 2 entries"))
    h = isfinite(horizon) ? Float64(horizon) : Float64(edges[end])
    for i in 1:(length(edges) - 1)
        (h >= edges[i] && h < edges[i + 1]) && return i
    end
    return h < edges[1] ? 1 : length(edges) - 1
end

_stratum_key(hbin::Int, regime::Symbol) = Symbol("h", hbin, "_", regime)

"""
    fit_conformal(points, observations, horizons, latest_dsts; kwargs...)

Fit a stratified split-conformal calibration from paired point forecasts and
observations. `horizons` are lead times [hr]; `latest_dsts` are the issue-time
observed Dst values used to assign the activity regime. Rows with any non-finite
entry are dropped.

Keyword arguments: `coverage` (default 0.90), `horizon_edges`,
`activity_threshold_nt`, `min_stratum_n`.
"""
function fit_conformal(points::AbstractVector{<:Real},
                       observations::AbstractVector{<:Real},
                       horizons::AbstractVector{<:Real},
                       latest_dsts::AbstractVector{<:Real};
                       coverage::Real=0.90,
                       horizon_edges::AbstractVector{<:Real}=CONFORMAL_HORIZON_EDGES,
                       activity_threshold_nt::Real=CONFORMAL_ACTIVITY_THRESHOLD_NT,
                       min_stratum_n::Int=CONFORMAL_MIN_STRATUM_N)
    n = length(points)
    (length(observations) == n && length(horizons) == n && length(latest_dsts) == n) ||
        throw(DimensionMismatch("points, observations, horizons, latest_dsts must have equal length"))
    0 < coverage < 1 || throw(ArgumentError("coverage must lie in (0, 1)"))
    issorted(horizon_edges) || throw(ArgumentError("horizon_edges must be ascending"))
    min_stratum_n >= 1 || throw(ArgumentError("min_stratum_n must be ≥ 1"))

    edges = collect(Float64, horizon_edges)
    # Collect residuals per stratum and globally.
    by_key = Dict{Symbol,Vector{Float64}}()
    global_res = Float64[]
    for i in 1:n
        p, o, h, d = points[i], observations[i], horizons[i], latest_dsts[i]
        (isfinite(p) && isfinite(o) && isfinite(h)) || continue
        r = abs(o - p)
        push!(global_res, r)
        key = _stratum_key(_horizon_bin(h, edges), _activity_regime(d, activity_threshold_nt))
        push!(get!(by_key, key, Float64[]), r)
    end
    isempty(global_res) && throw(ArgumentError("no finite calibration rows for conformal fit"))

    gqw, gcf = _conformal_quantile(global_res, coverage)
    global_stratum = ConformalStratum(:global, length(global_res), gqw, gcf)

    strata = Dict{Symbol,ConformalStratum}()
    for (key, res) in by_key
        qw, cf = _conformal_quantile(res, coverage)
        strata[key] = ConformalStratum(key, length(res), qw, cf)
    end
    return ConformalCalibration(Float64(coverage), edges, Float64(activity_threshold_nt),
                                min_stratum_n, strata, global_stratum)
end

"""
    conformal_stratum(cal, horizon, latest_dst)

Resolve the [`ConformalStratum`](@ref) used for a `(horizon, latest_dst)` query:
the matching cell when it has `≥ min_stratum_n` calibration points, otherwise the
pooled global fallback.
"""
function conformal_stratum(cal::ConformalCalibration, horizon::Real, latest_dst::Real)
    key = _stratum_key(_horizon_bin(horizon, cal.horizon_edges),
                       _activity_regime(latest_dst, cal.activity_threshold_nt))
    s = get(cal.strata, key, nothing)
    return (s !== nothing && s.n >= cal.min_stratum_n) ? s : cal.global_stratum
end

"""
    conformal_halfwidth(cal, horizon, latest_dst)

Conformal interval half-width for a query, from the resolved stratum.
"""
conformal_halfwidth(cal::ConformalCalibration, horizon::Real, latest_dst::Real) =
    conformal_stratum(cal, horizon, latest_dst).half_width

"""
    conformal_interval(cal, point, horizon, latest_dst)

`(lo, hi)` conformal interval centered on `point`.
"""
function conformal_interval(cal::ConformalCalibration, point::Real,
                            horizon::Real, latest_dst::Real)
    hw = conformal_halfwidth(cal, horizon, latest_dst)
    return (Float64(point) - hw, Float64(point) + hw)
end

"""
    write_conformal_calibration(path, cal)

Persist a [`ConformalCalibration`](@ref) to CSV. Row 1 is a `__meta__` record
holding the nominal coverage, horizon edges, activity threshold, and
`min_stratum_n`; the remaining rows are one per stratum (including `global`).
"""
function write_conformal_calibration(path::String, cal::ConformalCalibration)
    dir = dirname(path)
    !isempty(dir) && mkpath(dir)
    rows = NamedTuple[]
    push!(rows, (
        stratum="__meta__",
        n=0,
        half_width=cal.coverage,
        coverage_floor=cal.activity_threshold_nt,
        horizon_edges=join(string.(cal.horizon_edges), ";"),
        min_stratum_n=cal.min_stratum_n,
    ))
    function _push_stratum(s::ConformalStratum)
        push!(rows, (
            stratum=String(s.key),
            n=s.n,
            half_width=s.half_width,
            coverage_floor=s.coverage_floor,
            horizon_edges="",
            min_stratum_n=cal.min_stratum_n,
        ))
    end
    _push_stratum(cal.global_stratum)
    for key in sort(collect(keys(cal.strata)))
        _push_stratum(cal.strata[key])
    end
    CSV.write(path, DataFrame(rows))
    return path
end

"""
    read_conformal_calibration(path)

Load a [`ConformalCalibration`](@ref) written by
[`write_conformal_calibration`](@ref).
"""
function read_conformal_calibration(path::String)
    df = CSV.read(path, DataFrame)
    nrow(df) >= 2 || throw(ArgumentError("conformal calibration file has no strata: $path"))
    String(df.stratum[1]) == "__meta__" ||
        throw(ArgumentError("first conformal row must be __meta__"))
    coverage = Float64(df.half_width[1])
    activity_threshold = Float64(df.coverage_floor[1])
    edges = parse.(Float64, split(String(df.horizon_edges[1]), ";"))
    min_stratum_n = Int(df.min_stratum_n[1])

    strata = Dict{Symbol,ConformalStratum}()
    global_stratum = nothing
    for i in 2:nrow(df)
        key = Symbol(String(df.stratum[i]))
        s = ConformalStratum(key, Int(df.n[i]), Float64(df.half_width[i]),
                             Float64(df.coverage_floor[i]))
        if key == :global
            global_stratum = s
        else
            strata[key] = s
        end
    end
    global_stratum === nothing &&
        throw(ArgumentError("conformal calibration missing :global stratum: $path"))
    return ConformalCalibration(coverage, edges, activity_threshold,
                                min_stratum_n, strata, global_stratum)
end

"""
    conformal_coverage(cal, points, observations, horizons, latest_dsts)

Empirical fraction of observations falling inside the conformal interval — for
validating coverage on a held-out set. Rows with non-finite point/obs/horizon are
skipped; returns `NaN` if none remain.
"""
function conformal_coverage(cal::ConformalCalibration,
                            points::AbstractVector{<:Real},
                            observations::AbstractVector{<:Real},
                            horizons::AbstractVector{<:Real},
                            latest_dsts::AbstractVector{<:Real})
    n = length(points)
    (length(observations) == n && length(horizons) == n && length(latest_dsts) == n) ||
        throw(DimensionMismatch("inputs must have equal length"))
    hits = 0
    total = 0
    for i in 1:n
        (isfinite(points[i]) && isfinite(observations[i]) && isfinite(horizons[i])) || continue
        lo, hi = conformal_interval(cal, points[i], horizons[i], latest_dsts[i])
        total += 1
        (lo <= observations[i] <= hi) && (hits += 1)
    end
    return total == 0 ? NaN : hits / total
end
