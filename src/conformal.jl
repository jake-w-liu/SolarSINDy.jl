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
occupied (horizon-bin × activity-regime) cell plus a pooled global fallback. An
under-populated cell (fewer than `min_stratum_n` calibration points) resolves via
the monotone-safe fallback in [`conformal_stratum`](@ref) — the widest of the
global band and the strata the query dominates — not blindly to the (possibly
narrower) global band.
"""
struct ConformalCalibration
    coverage::Float64                      # nominal target, e.g. 0.90
    horizon_edges::Vector{Float64}         # ascending lead-time bin edges [hr]
    activity_threshold_nt::Float64         # latest Dst ≤ threshold ⇒ :disturbed
    min_stratum_n::Int                     # below this, use the global fallback
    max_horizon::Float64                   # largest calibrated lead time [hr]; queries
                                           # beyond it fall back to the global band
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

function _validate_conformal_edges(edges::AbstractVector{<:Real})
    length(edges) >= 2 || throw(ArgumentError("horizon_edges needs ≥ 2 entries"))
    all(x -> !isnan(x) && x >= 0, edges) ||
        throw(ArgumentError("horizon_edges must be nonnegative and not NaN"))
    all(i -> edges[i] < edges[i + 1], 1:(length(edges) - 1)) ||
        throw(ArgumentError("horizon_edges must be strictly ascending"))
    all(isfinite, @view(edges[1:end-1])) ||
        throw(ArgumentError("only the final horizon edge may be infinite"))
    return nothing
end

_stratum_key(hbin::Int, regime::Symbol) = Symbol("h", hbin, "_", regime)

"""
    fit_conformal(points, observations, horizons, latest_dsts; kwargs...)

Fit a stratified split-conformal calibration from paired point forecasts and
observations. `horizons` are lead times [hr]; `latest_dsts` are the issue-time
observed Dst values used to assign the activity regime. A row is dropped only
when its point, observation, or horizon is non-finite; a non-finite
`latest_dst` is retained and assigned the wider `:disturbed` regime (the safe
default, since the activity label is then unknown).

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
    coverage64 = Float64(coverage)
    isfinite(coverage64) && 0 < coverage64 < 1 ||
        throw(ArgumentError("coverage must be finite and lie in (0, 1)"))
    edges = collect(Float64, horizon_edges)
    _validate_conformal_edges(edges)
    activity_threshold64 = Float64(activity_threshold_nt)
    isfinite(activity_threshold64) ||
        throw(ArgumentError("activity_threshold_nt must be finite"))
    min_stratum_n >= 1 || throw(ArgumentError("min_stratum_n must be ≥ 1"))

    any(h -> isfinite(h) && h < 0, horizons) &&
        throw(ArgumentError("calibration horizons must be nonnegative"))

    # Collect residuals per stratum and globally, and track the largest calibrated
    # lead time so out-of-support queries can be detected later.
    by_key = Dict{Symbol,Vector{Float64}}()
    global_res = Float64[]
    max_horizon = 0.0
    for i in 1:n
        p, o, h, d = points[i], observations[i], horizons[i], latest_dsts[i]
        (isfinite(p) && isfinite(o) && isfinite(h)) || continue
        p64 = Float64(p)
        o64 = Float64(o)
        h64 = Float64(h)
        all(isfinite, (p64, o64, h64)) || throw(ArgumentError(
            "finite conformal inputs exceed the supported Float64 range",
        ))
        # Convert before subtraction: fixed-width integer endpoints can otherwise
        # wrap to a tiny residual and produce a dangerously narrow interval.
        r = abs(o64 - p64)
        isfinite(r) || throw(ArgumentError(
            "conformal residual exceeds the supported Float64 range",
        ))
        push!(global_res, r)
        max_horizon = max(max_horizon, h64)
        key = _stratum_key(_horizon_bin(h64, edges),
                           _activity_regime(d, activity_threshold64))
        push!(get!(by_key, key, Float64[]), r)
    end
    isempty(global_res) && throw(ArgumentError("no finite calibration rows for conformal fit"))

    gqw, gcf = _conformal_quantile(global_res, coverage64)
    global_stratum = ConformalStratum(:global, length(global_res), gqw, gcf)

    strata = Dict{Symbol,ConformalStratum}()
    for (key, res) in by_key
        qw, cf = _conformal_quantile(res, coverage64)
        strata[key] = ConformalStratum(key, length(res), qw, cf)
    end
    return ConformalCalibration(coverage64, edges, activity_threshold64,
                                min_stratum_n, max_horizon, strata, global_stratum)
end

# Widest (largest half-width) stratum among `candidate_keys` that meets the
# population floor, or the pooled global band when none qualifies or is wider.
# Used so an under-populated cell never resolves to a band NARROWER than a
# better-populated stratum that the query dominates (regime/lead monotonicity).
function _widest_stratum(cal::ConformalCalibration, candidate_keys)
    best = cal.global_stratum
    for k in candidate_keys
        s = get(cal.strata, k, nothing)
        (s === nothing || s.n < cal.min_stratum_n) && continue
        s.half_width > best.half_width && (best = s)
    end
    return best
end

"""
    conformal_stratum(cal, horizon, latest_dst)

Resolve the [`ConformalStratum`](@ref) used for a `(horizon, latest_dst)` query.
The matching cell is used when it has `≥ min_stratum_n` calibration points.
Otherwise the fallback is MONOTONE-SAFE: because storm-time residuals are
heavier-tailed than quiet-time residuals and longer leads carry larger error, an
under-populated cell must never receive a band narrower than a better-populated
stratum that the query dominates. The fallback therefore returns the widest of the
pooled global band, the same-lead quiet counterpart (for a disturbed query), and
any populated disturbed stratum at an equal-or-shorter lead — never blindly the
(possibly narrower) global band. This preserves `disturbed ≥ quiet` at a fixed lead
even when the disturbed stratum is sparse.

A query horizon beyond the largest calibrated lead time (`max_horizon`) is
out-of-support: the top horizon bin `[edges[end-1], Inf)` was only ever fit on
horizons up to `max_horizon`, so reusing a narrower band for a longer query would
silently understate the interval. Such queries use the widest populated stratum
dominated by the query (plus the global pool).
"""
function conformal_stratum(cal::ConformalCalibration, horizon::Real, latest_dst::Real)
    horizon_float = Float64(horizon)
    isfinite(horizon_float) && horizon_float >= 0 || throw(ArgumentError(
        "forecast horizon must be finite, nonnegative, and representable as Float64",
    ))
    hbin = _horizon_bin(horizon_float, cal.horizon_edges)
    regime = _activity_regime(latest_dst, cal.activity_threshold_nt)
    # Out-of-support queries may never become narrower merely because they cross
    # max_horizon. Include every populated same-regime shorter-lead cell; for a
    # disturbed query also include quiet cells, which it dominates by regime.
    if horizon_float > cal.max_horizon
        candidates = Symbol[]
        for b in 1:(length(cal.horizon_edges) - 1)
            push!(candidates, _stratum_key(b, :quiet))
            regime === :disturbed && push!(candidates, _stratum_key(b, :disturbed))
        end
        return _widest_stratum(cal, candidates)
    end
    key = _stratum_key(hbin, regime)
    s = get(cal.strata, key, nothing)
    (s !== nothing && s.n >= cal.min_stratum_n) && return s
    # Under-populated cell: fall back monotone-safely rather than to the pooled
    # global band, which in a real sidecar can be NARROWER than the same-lead quiet
    # band (observed inversion: global 13.98 nT < 6 h quiet 16.51 nT). Every query
    # is guarded by populated same-regime cells at equal-or-shorter leads; a
    # disturbed query additionally dominates the corresponding quiet cells.
    candidates = Symbol[]
    for b in 1:hbin
        push!(candidates, _stratum_key(b, regime))
        regime === :disturbed && push!(candidates, _stratum_key(b, :quiet))
    end
    return _widest_stratum(cal, candidates)
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
    point_float = Float64(point)
    isfinite(point_float) || throw(ArgumentError(
        "conformal interval point must be finite and representable as Float64",
    ))
    hw = conformal_halfwidth(cal, horizon, latest_dst)
    return (point_float - hw, point_float + hw)
end

# ---------------------------------------------------------------------------
# Adaptive Conformal Inference (ACI; Gibbs & Candès, 2021)
#
# Split conformal assumes calibration and test residuals are exchangeable. Under
# distribution shift (e.g. rising solar activity across a chronological split)
# that assumption fails and static intervals drift off nominal. ACI restores
# long-run coverage by adapting the effective miscoverage rate online:
#
#   half-width_t = empirical (1 - α_t) quantile of the residual history
#   α_{t+1}      = α_t + γ (α* - err_t),   err_t = 1 if y_t fell outside, else 0
#
# with target miscoverage α* = 1 - coverage. A miss lowers α_t (widening the next
# interval); a hit raises it (tightening).
#
# Coverage guarantee — scope. The Gibbs & Candès (2021) distribution-free bound
# |mean(err) - α*| ≤ (α_1 + γ)/(γ T) → α* holds for the IDEALIZED recursion in
# which α_t is UNCLAMPED and the prediction set SATURATES to the whole real line
# whenever α_t ≤ 0 (and to the empty set when α_t ≥ 1). The telescoping proof needs
# exactly that: during a miss burst α_t must be allowed to go negative and the band
# to become infinite so the accumulated miscoverage "debt" is repaid later.
#
# This implementation deliberately trades that worst-case guarantee for BOUNDED,
# operationally usable bands: α_t is clamped to [0, 1] (line ~310) and the widest
# band is the empirical sample maximum (`_empirical_halfwidth` at level ≥ 1), never
# an infinite interval. Under sustained, unprecedented drift the clamp discards
# miscoverage debt, so the (α_1 + γ)/(γ T) bound is NOT guaranteed as written;
# realized coverage is instead validated empirically (unit tests + live
# monitoring). Merely widening the α_t range would not restore the bound while the
# band is still capped at the sample max — restoring the theorem would require
# serving an infinite interval whenever α_t ≤ 0, which is useless for a forecast
# product. The bounded-band behavior is the intended, safer operational choice.
# ---------------------------------------------------------------------------

"""
    AdaptiveConformal

Online ACI state for a single forecast stream: target coverage `1-α*`, learning
rate `gamma`, current miscoverage `alpha_t`, the trailing absolute-residual
`window`, the residual `history`, a reusable order-statistic buffer, and a
`warmup` count of initial steps that use the widest available band before
adaptation is trusted.
"""
mutable struct AdaptiveConformal
    target_coverage::Float64
    gamma::Float64
    alpha_t::Float64
    window::Int
    history::Vector{Float64}
    scratch::Vector{Float64}
    warmup::Int
end

const _DEFAULT_ACI_WINDOW = 500

"""
    init_adaptive_conformal(; target_coverage=0.90, gamma=0.02, window=500, warmup=20)

Construct an [`AdaptiveConformal`](@ref) stream with miscoverage initialised to
`1 - target_coverage`. The finite default window bounds steady-state memory;
an explicit larger window remains available for offline sensitivity analyses.
"""
function init_adaptive_conformal(; target_coverage::Real=0.90, gamma::Real=0.02,
                                 window::Integer=_DEFAULT_ACI_WINDOW,
                                 warmup::Integer=20)
    isfinite(target_coverage) && 0 < target_coverage < 1 ||
        throw(ArgumentError("target_coverage must be finite and lie in (0, 1)"))
    isfinite(gamma) && gamma > 0 || throw(ArgumentError("gamma must be finite and positive"))
    window >= 1 || throw(ArgumentError("window must be ≥ 1"))
    warmup >= 0 || throw(ArgumentError("warmup must be ≥ 0"))
    window >= warmup || throw(ArgumentError("window must be ≥ warmup"))
    window_int = Int(window)
    initial_capacity = min(window_int, _DEFAULT_ACI_WINDOW)
    history = sizehint!(Float64[], initial_capacity)
    scratch = sizehint!(Float64[], initial_capacity)
    return AdaptiveConformal(Float64(target_coverage), Float64(gamma),
                             1.0 - Float64(target_coverage), window_int,
                             history, scratch, Int(warmup))
end

function _validate_adaptive_conformal(ac::AdaptiveConformal)
    isfinite(ac.target_coverage) && 0 < ac.target_coverage < 1 ||
        throw(ArgumentError(
            "adaptive conformal target coverage must remain finite and lie in (0, 1)",
        ))
    isfinite(ac.gamma) && ac.gamma > 0 || throw(ArgumentError(
        "adaptive conformal learning rate must remain finite and positive",
    ))
    isfinite(ac.alpha_t) && 0 <= ac.alpha_t <= 1 || throw(ArgumentError(
        "adaptive conformal alpha must remain finite and lie in [0, 1]",
    ))
    ac.window >= 1 || throw(ArgumentError(
        "adaptive conformal window must remain positive",
    ))
    0 <= ac.warmup <= ac.window || throw(ArgumentError(
        "adaptive conformal warmup must remain between zero and the window",
    ))
    length(ac.history) <= ac.window || throw(ArgumentError(
        "adaptive conformal history exceeds its configured window",
    ))
    length(ac.scratch) <= ac.window || throw(ArgumentError(
        "adaptive conformal scratch buffer exceeds its configured window",
    ))
    ac.scratch !== ac.history || throw(ArgumentError(
        "adaptive conformal history and scratch buffers must not alias",
    ))
    all(value -> isfinite(value) && value >= 0, ac.history) ||
        throw(ArgumentError(
            "adaptive conformal history must contain finite nonnegative residuals",
        ))
    all(value -> isfinite(value) && value >= 0, ac.scratch) ||
        throw(ArgumentError(
            "adaptive conformal scratch buffer must contain finite nonnegative residuals",
        ))
    return ac
end

# Finite-sample empirical half-width at coverage `level` (0 if level ≤ 0; the
# sample max — the widest the sample supports — when level is too high for n).
# Non-finite history entries are filtered out so a stray NaN/Inf residual cannot
# poison the quantile; with no finite entry the band falls back to the widest
# possible (Inf), mirroring the empty-history case.
function _empirical_halfwidth!(scratch::Vector{Float64},
                               history::AbstractVector{<:Real}, level::Real)
    empty!(scratch)
    for h in history
        isfinite(h) && push!(scratch, Float64(h))
    end
    n = length(scratch)
    n == 0 && return Inf
    level <= 0 && return 0.0
    k = ceil(Int, (n + 1) * level)
    k > n && return maximum(scratch)
    sort!(scratch; alg=QuickSort)
    return scratch[clamp(k, 1, n)]
end

_empirical_halfwidth(history::AbstractVector{<:Real}, level::Real) =
    _empirical_halfwidth!(Float64[], history, level)

function _widest_finite(history::AbstractVector{<:Real})
    widest = -Inf
    for h in history
        isfinite(h) && (widest = max(widest, Float64(h)))
    end
    return widest == -Inf ? Inf : widest
end

"""
    adaptive_conformal_step!(ac, point, observed)

Process one online step: form the interval around `point` from the current
residual history at level `1 - α_t`, score coverage against `observed`, then
append the new absolute residual (respecting the trailing window) and update
`α_t`. Returns `(lo, hi, half_width, covered, alpha)`. The interval is formed
BEFORE the observation enters the history (causal).

A non-finite `point`/`observed` (or a non-finite residual) is treated as a
missing/gap step, mirroring the split-path contract that drops non-finite rows:
the interval is still reported from the current finite history, but the residual
is NOT pushed and `α_t` is NOT updated, so a single NaN cannot poison the stream.
"""
function _conformal_stream_value(value::Real, label::AbstractString)
    converted = Float64(value)
    isfinite(value) && !isfinite(converted) && throw(ArgumentError(
        "$label exceeds the supported Float64 range",
    ))
    return converted
end

function _adaptive_conformal_step!(ac::AdaptiveConformal, point::Float64,
                                   observed::Float64)
    level = clamp(1.0 - ac.alpha_t, 0.0, 1.0)
    hw = length(ac.history) < ac.warmup ?
         _widest_finite(ac.history) :
         _empirical_halfwidth!(ac.scratch, ac.history, level)
    lo = point - hw
    hi = point + hw
    r = abs(observed - point)
    isfinite(point) && isfinite(observed) && !isfinite(r) &&
        throw(ArgumentError("adaptive conformal residual exceeds the supported range"))
    # Gap step: a non-finite residual carries no coverage information. Report the
    # interval from existing history but skip the history push and α_t update.
    if !isfinite(r)
        return (lo=lo, hi=hi, half_width=hw, covered=false, alpha=ac.alpha_t)
    end
    covered = r <= hw
    err = covered ? 0.0 : 1.0
    α_target = 1.0 - ac.target_coverage
    ac.alpha_t = clamp(ac.alpha_t + ac.gamma * (α_target - err), 0.0, 1.0)
    push!(ac.history, r)
    length(ac.history) > ac.window && popfirst!(ac.history)
    return (lo=lo, hi=hi, half_width=hw, covered=covered, alpha=ac.alpha_t)
end


function adaptive_conformal_step!(ac::AdaptiveConformal, point::Real, observed::Real)
    _validate_adaptive_conformal(ac)
    point_float = _conformal_stream_value(point, "adaptive conformal point")
    observed_float = _conformal_stream_value(observed, "adaptive conformal observation")
    return _adaptive_conformal_step!(ac, point_float, observed_float)
end

"""
    run_adaptive_conformal(points, observations; kwargs...)

Run ACI over a time-ordered stream. Returns `(lo, hi, covered, alpha,
coverage)` where the per-step vectors span all steps and `coverage` is the
realized coverage over post-warmup steps only (the operationally meaningful
figure). Keyword arguments are forwarded to [`init_adaptive_conformal`](@ref).
"""
function run_adaptive_conformal(points::AbstractVector{<:Real},
                                observations::AbstractVector{<:Real}; kwargs...)
    length(points) == length(observations) ||
        throw(DimensionMismatch("points and observations must have equal length"))
    ac = init_adaptive_conformal(; kwargs...)
    n = length(points)
    lo = Vector{Float64}(undef, n); hi = Vector{Float64}(undef, n)
    covered = Vector{Bool}(undef, n); alpha = Vector{Float64}(undef, n)
    post_hits = 0; post_total = 0
    for t in 1:n
        warm = length(ac.history) < ac.warmup
        point = _conformal_stream_value(points[t], "adaptive conformal point")
        observed = _conformal_stream_value(
            observations[t], "adaptive conformal observation",
        )
        # A non-finite point/obs is a gap step (no residual pushed); it carries no
        # coverage information and must not be scored as a miss.
        gap = !(isfinite(point) && isfinite(observed))
        s = _adaptive_conformal_step!(ac, point, observed)
        lo[t] = s.lo; hi[t] = s.hi; covered[t] = s.covered; alpha[t] = s.alpha
        if !warm && !gap
            post_total += 1
            s.covered && (post_hits += 1)
        end
    end
    coverage = post_total == 0 ? NaN : post_hits / post_total
    return (; lo, hi, covered, alpha, coverage)
end

"""
    write_conformal_calibration(path, cal)

Persist a [`ConformalCalibration`](@ref) to CSV. Row 1 is a `__meta__` record
holding the nominal coverage, horizon edges, activity threshold, `min_stratum_n`,
and the largest calibrated horizon; the remaining rows are one per stratum
(including `global`). The `max_horizon` column is new; readers default it to
`Inf` when absent so older sidecars still load.
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
        max_horizon=cal.max_horizon,
    ))
    function _push_stratum(s::ConformalStratum)
        push!(rows, (
            stratum=String(s.key),
            n=s.n,
            half_width=s.half_width,
            coverage_floor=s.coverage_floor,
            horizon_edges="",
            min_stratum_n=cal.min_stratum_n,
            max_horizon=cal.max_horizon,
        ))
    end
    _push_stratum(cal.global_stratum)
    for key in sort(collect(keys(cal.strata)))
        _push_stratum(cal.strata[key])
    end
    cal.global_stratum.key == :global ||
        throw(ArgumentError("global conformal stratum must have key :global"))
    _validate_conformal_edges(cal.horizon_edges)
    isfinite(cal.coverage) && 0 < cal.coverage < 1 ||
        throw(ArgumentError("conformal coverage must be finite and lie in (0, 1)"))
    isfinite(cal.activity_threshold_nt) ||
        throw(ArgumentError("conformal activity threshold must be finite"))
    isfinite(cal.max_horizon) && cal.max_horizon >= 0 ||
        throw(ArgumentError("conformal max_horizon must be finite and nonnegative"))
    cal.min_stratum_n >= 1 ||
        throw(ArgumentError("conformal min_stratum_n must be at least 1"))
    _write_selection_csv(path, rows)
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
    # max_horizon is a newer field; legacy sidecars without the column default to
    # Inf, which disables the out-of-support guard and preserves prior behavior.
    has_max_horizon = hasproperty(df, :max_horizon)
    max_horizon = has_max_horizon ? Float64(df.max_horizon[1]) : Inf

    strata = Dict{Symbol,ConformalStratum}()
    global_stratum = nothing
    seen_keys = Set{Symbol}()
    for i in 2:nrow(df)
        key = Symbol(String(df.stratum[i]))
        key in seen_keys &&
            throw(ArgumentError("corrupt conformal calibration: duplicate stratum $key: $path"))
        push!(seen_keys, key)
        s = ConformalStratum(key, Int(df.n[i]), Float64(df.half_width[i]),
                             Float64(df.coverage_floor[i]))
        s.n >= 1 || throw(ArgumentError("corrupt conformal calibration: stratum $key has n < 1: $path"))
        isfinite(s.half_width) && s.half_width >= 0 ||
            throw(ArgumentError("corrupt conformal calibration: stratum $key has invalid half-width: $path"))
        isfinite(s.coverage_floor) && 0 <= s.coverage_floor <= 1 ||
            throw(ArgumentError("corrupt conformal calibration: stratum $key has invalid coverage floor: $path"))
        if key == :global
            global_stratum = s
        else
            strata[key] = s
        end
    end
    global_stratum === nothing &&
        throw(ArgumentError("conformal calibration missing :global stratum: $path"))
    # Re-assert the same invariants fit_conformal enforces, so a corrupt or
    # hand-edited sidecar cannot load into an inconsistent calibration.
    isfinite(coverage) && 0 < coverage < 1 ||
        throw(ArgumentError("corrupt conformal calibration: coverage must lie in (0, 1): $path"))
    isfinite(activity_threshold) ||
        throw(ArgumentError("corrupt conformal calibration: activity threshold must be finite: $path"))
    length(edges) >= 2 ||
        throw(ArgumentError("corrupt conformal calibration: at least two horizon edges are required: $path"))
    try
        _validate_conformal_edges(edges)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("corrupt conformal calibration: $(err.msg): $path"))
    end
    has_max_horizon && !(isfinite(max_horizon) && max_horizon >= 0) &&
        throw(ArgumentError("corrupt conformal calibration: max_horizon must be finite and nonnegative: $path"))
    min_stratum_n >= 1 || throw(ArgumentError("corrupt conformal calibration: min_stratum_n must be ≥ 1: $path"))
    return ConformalCalibration(coverage, edges, activity_threshold,
                                min_stratum_n, max_horizon, strata, global_stratum)
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
