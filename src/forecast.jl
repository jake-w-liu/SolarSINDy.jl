# Rolling forecaster with ensemble uncertainty quantification

"""
    ForecastResult

Single-step forecast output with ensemble statistics.
"""
struct ForecastResult
    t::DateTime
    dst_predicted::Float64       # primary model prediction [nT]
    dst_median::Float64          # ensemble median [nT]
    dst_ci_05::Float64           # 5th percentile (worst case) [nT]
    dst_ci_95::Float64           # 95th percentile (best case) [nT]
    dst_observed::Float64        # observed Dst* (NaN if unavailable) [nT]
end

function _forecast_step_period(dt::Real)
    isfinite(dt) && dt > 0 ||
        throw(ArgumentError("forecast state dt must be finite and positive"))
    milliseconds_float = Float64(dt) * 3_600_000.0
    isfinite(milliseconds_float) && milliseconds_float <= typemax(Int) ||
        throw(ArgumentError("forecast state dt is outside the DateTime range"))
    milliseconds = round(Int, milliseconds_float)
    milliseconds >= 1 ||
        throw(ArgumentError("forecast state dt must be at least one millisecond"))
    tolerance = 8eps(max(1.0, abs(milliseconds_float)))
    isapprox(milliseconds_float, milliseconds; rtol=0.0, atol=tolerance) ||
        throw(ArgumentError("forecast state dt must be representable to millisecond precision"))
    return Millisecond(milliseconds)
end

"""
    ForecastState

Mutable state for rolling one-step-ahead forecasting.
"""
mutable struct ForecastState
    t_current::DateTime
    dst_current::Float64
    lib::CandidateLibrary
    ξ_primary::Vector{Float64}
    ξ_ensemble::Matrix{Float64}  # n_ensemble × n_terms
    dt::Float64
    history::Vector{ForecastResult}
    function ForecastState(t_current::DateTime, dst_current::Float64,
                           lib::CandidateLibrary, ξ_primary::Vector{Float64},
                           ξ_ensemble::Matrix{Float64}, dt::Float64,
                           history::Vector{ForecastResult})
        isfinite(dst_current) || throw(ArgumentError("forecast state Dst* must be finite"))
        _forecast_step_period(dt)
        length(ξ_primary) == length(lib) ||
            throw(DimensionMismatch("primary coefficient count does not match library"))
        size(ξ_ensemble, 2) == length(lib) ||
            throw(DimensionMismatch("ensemble coefficient count does not match library"))
        size(ξ_ensemble, 1) >= 1 ||
            throw(ArgumentError("forecast state requires at least one ensemble member"))
        all(isfinite, ξ_primary) || throw(ArgumentError("primary coefficients must be finite"))
        all(isfinite, ξ_ensemble) || throw(ArgumentError("ensemble coefficients must be finite"))
        return new(t_current, dst_current, lib, ξ_primary, ξ_ensemble, dt, history)
    end
end

ForecastState(t_current::DateTime, dst_current::Real, lib::CandidateLibrary,
              ξ_primary::AbstractVector{<:Real}, ξ_ensemble::AbstractMatrix{<:Real},
              dt::Real, history::Vector{ForecastResult}=ForecastResult[]) =
    ForecastState(t_current, Float64(dst_current), lib, collect(Float64, ξ_primary),
                  Matrix{Float64}(ξ_ensemble), Float64(dt), history)

"""
    OperationalV2Calibration

Small causal post-processing layer for a V2 forecast.

The correction is fit from prior replay/live rows only:

```
Dst_v2 = Dst_v1 + β₀ + Σ βⱼ zⱼ
```

where `zⱼ` are standardized issue-time features, including derived causal
coupling features in the default configuration. The interval scale inflates the
v1 ensemble interval from calibration residuals. This type deliberately does
not alter `ForecastState` or v1 SINDy coefficients.

When mandatory baseline columns are present during calibration, v2 also stores
guarded component metrics and optional convex ensemble weights over corrected
SINDy, uncorrected SINDy v1, persistence, Burton, BurtonFull, and
O'Brien--McPherron. A fallback component or ensemble is deployed only after
chronological validation in the operational workflow.
"""
struct OperationalV2Calibration
    feature_names::Vector{Symbol}
    feature_mean::Vector{Float64}
    feature_scale::Vector{Float64}
    coefficients::Vector{Float64}
    interval_scale::Float64
    label::String
    selector_names::Vector{Symbol}
    selector_rmse::Vector{Float64}
    selector_mae::Vector{Float64}
    selector_half_width::Vector{Float64}
    selector_weights::Vector{Float64}
    selected_component::Symbol
    guard_margin_nt::Float64


    function OperationalV2Calibration(feature_names::Vector{Symbol},
                                      feature_mean::Vector{Float64},
                                      feature_scale::Vector{Float64},
                                      coefficients::Vector{Float64},
                                      interval_scale::Float64,
                                      label::AbstractString;
                                      selector_names::Vector{Symbol}=Symbol[:v2],
                                      selector_rmse::Vector{Float64}=Float64[NaN],
                                      selector_mae::Vector{Float64}=Float64[NaN],
                                      selector_half_width::Vector{Float64}=Float64[0.0],
                                      selector_weights::Union{Nothing,Vector{Float64}}=nothing,
                                      selected_component::Symbol=:v2,
                                      guard_margin_nt::Real=0.0)
        n = length(feature_names)
        length(Set(feature_names)) == length(feature_names) ||
            throw(ArgumentError("feature_names must not contain duplicates"))
        length(feature_mean) == n ||
            throw(DimensionMismatch("feature_mean length $(length(feature_mean)) != $n"))
        length(feature_scale) == n ||
            throw(DimensionMismatch("feature_scale length $(length(feature_scale)) != $n"))
        length(coefficients) == n + 1 ||
            throw(DimensionMismatch("coefficients length $(length(coefficients)) != $(n + 1)"))
        all(isfinite, feature_mean) || throw(ArgumentError("feature means must be finite"))
        all(isfinite, feature_scale) || throw(ArgumentError("feature scales must be finite"))
        all(>(0.0), feature_scale) || throw(ArgumentError("feature scales must be positive"))
        all(isfinite, coefficients) || throw(ArgumentError("coefficients must be finite"))
        isfinite(interval_scale) && interval_scale > 0 ||
            throw(ArgumentError("interval_scale must be positive and finite"))
        n_selector = length(selector_names)
        n_selector > 0 || throw(ArgumentError("at least one selector component is required"))
        length(Set(selector_names)) == n_selector ||
            throw(ArgumentError("selector_names must not contain duplicates"))
        length(selector_rmse) == n_selector ||
            throw(DimensionMismatch("selector_rmse length $(length(selector_rmse)) != $n_selector"))
        length(selector_mae) == n_selector ||
            throw(DimensionMismatch("selector_mae length $(length(selector_mae)) != $n_selector"))
        length(selector_half_width) == n_selector ||
            throw(DimensionMismatch("selector_half_width length $(length(selector_half_width)) != $n_selector"))
        all(x -> isfinite(x) || isnan(x), selector_rmse) ||
            throw(ArgumentError("selector RMSE values must be finite or NaN"))
        all(x -> isfinite(x) || isnan(x), selector_mae) ||
            throw(ArgumentError("selector MAE values must be finite or NaN"))
        all(isfinite, selector_half_width) ||
            throw(ArgumentError("selector interval half-widths must be finite"))
        all(>=(0.0), selector_half_width) ||
            throw(ArgumentError("selector interval half-widths must be nonnegative"))
        selected_component == :ensemble || selected_component in selector_names ||
            throw(ArgumentError("selected_component must be :ensemble or appear in selector_names"))
        weights = if selector_weights === nothing
            w = zeros(n_selector)
            if selected_component == :ensemble
                w .= 1.0 / n_selector
            else
                w[findfirst(==(selected_component), selector_names)] = 1.0
            end
            w
        else
            copy(selector_weights)
        end
        length(weights) == n_selector ||
            throw(DimensionMismatch("selector_weights length $(length(weights)) != $n_selector"))
        all(isfinite, weights) || throw(ArgumentError("selector weights must be finite"))
        all(>=(0.0), weights) || throw(ArgumentError("selector weights must be nonnegative"))
        weight_sum = sum(weights)
        weight_sum > 0 || throw(ArgumentError("selector weights must have positive sum"))
        if isfinite(weight_sum)
            weights ./= weight_sum
        else
            weight_scale = maximum(weights)
            isfinite(weight_scale) && weight_scale > 0 || throw(ArgumentError(
                "selector weights cannot be normalized within the supported range",
            ))
            weights ./= weight_scale
            scaled_sum = sum(weights)
            isfinite(scaled_sum) && scaled_sum > 0 || throw(ArgumentError(
                "selector weights cannot be normalized within the supported range",
            ))
            weights ./= scaled_sum
        end
        all(isfinite, weights) && sum(weights) > 0 || throw(ArgumentError(
            "normalized selector weights must be finite with positive sum",
        ))
        isfinite(Float64(guard_margin_nt)) && Float64(guard_margin_nt) >= 0 ||
            throw(ArgumentError("guard_margin_nt must be finite and nonnegative"))
        return new(feature_names, feature_mean, feature_scale, coefficients,
                   interval_scale, String(label), selector_names, selector_rmse,
                   selector_mae, selector_half_width, weights, selected_component,
                   Float64(guard_margin_nt))
    end
end

const DEFAULT_OPERATIONAL_V2_FEATURES = [
    :latest_dst_nt,
    :V_kms,
    :Bz_nt,
    :By_nt,
    :n_cm3,
    :Pdyn_npa,
    :Bsouth_nt,
    :VBsouth_mvm,
    :Bperp_nt,
    :clock_angle_sin2,
    :sqrt_Pdyn_npa,
]

const OPERATIONAL_V2_MEMORY_FEATURES = [
    :dst_delta_1h_nt,
    :dst_delta_3h_nt,
    :dst_delta_6h_nt,
    :Bz_delta_1h_nt,
    :VBsouth_delta_1h_mvm,
    :VBsouth_mean_3h_mvm,
    :VBsouth_mean_6h_mvm,
    :Bsouth_mean_3h_nt,
    :Bsouth_mean_6h_nt,
]

const OPERATIONAL_V2_EXPERT_FEATURES = [
    :baseline_spread_nt,
    :v1_minus_persistence_nt,
    :obrien_minus_v1_nt,
    :burton_minus_v1_nt,
]

const OPERATIONAL_V2_STORM_PHASE_FEATURES = [
    :main_phase_pressure_nt,
    :main_phase_pressure_6h_nt,
    :coupling_active_mvm,
    :coupling_active_6h_mvm,
    :recovery_pressure_nt,
    :main_phase_recovery_pressure,
    :horizon_coupling_pressure,
    :storm_guard_active,
    :storm_guard_minus_v1_nt,
]

const OPERATIONAL_V2_BASELINE_COLUMNS = Pair{Symbol,Symbol}[
    :persistence => :persistence_dst_nt,
    :burton => :burton_dst_nt,
    :burton_full => :burton_full_dst_nt,
    :obrien => :obrien_dst_nt,
    :storm_guard => :storm_guard_dst_nt,
]

function default_operational_v2_calibration(;
        feature_names::Vector{Symbol}=copy(DEFAULT_OPERATIONAL_V2_FEATURES),
        interval_scale::Real=1.0,
        label::AbstractString="uncalibrated_v2",
    )
    n = length(feature_names)
    return OperationalV2Calibration(
        feature_names,
        zeros(n),
        ones(n),
        zeros(n + 1),
        Float64(interval_scale),
        label,
    )
end

"""
    operational_v2_feature_tuple(latest_dst, V, Bz, By, n, Pdyn)

Return the causal issue-time feature tuple used by the default operational-v2
calibration. The derived features are computed only from quantities available at
issue time.
"""
function operational_v2_feature_tuple(latest_dst::Real, V::Real, Bz::Real,
                                      By::Real, n::Real, Pdyn::Real)
    latest = Float64(latest_dst)
    speed = Float64(V)
    bz = Float64(Bz)
    by = Float64(By)
    density = Float64(n)
    pdyn = Float64(Pdyn)
    bsouth = max(-bz, 0.0)
    bperp = hypot(by, bz)
    clock_angle_sin2 = iszero(bperp) ? 0.0 : clamp((1.0 - bz / bperp) / 2.0, 0.0, 1.0)
    return (
        latest_dst_nt=latest,
        V_kms=speed,
        Bz_nt=bz,
        By_nt=by,
        n_cm3=density,
        Pdyn_npa=pdyn,
        Bsouth_nt=bsouth,
        VBsouth_mvm=1e-3 * speed * bsouth,
        Bperp_nt=bperp,
        clock_angle_sin2=clock_angle_sin2,
        sqrt_Pdyn_npa=sqrt(max(pdyn, 0.0)),
    )
end

function _column_float_or_nan(df::DataFrame, col::Symbol)
    return Float64[ismissing(x) ? NaN : Float64(x) for x in df[!, col]]
end

# Scalar form: a `missing` cell (e.g. a not-yet-verified observation) becomes NaN, which
# propagates harmlessly through scoring rather than throwing `MethodError(Float64, missing)`.
_cell_float_or_nan(x) = ismissing(x) ? NaN : Float64(x)

function _chronological_order(df::DataFrame)
    for col in (:issue_time_utc, :latest_dst_time_utc, :target_time_utc)
        String(col) in names(df) || continue
        values = DateTime[]
        ok = true
        for x in df[!, col]
            if ismissing(x)
                ok = false
                break
            end
            push!(values, x isa DateTime ? x : DateTime(String(x)))
        end
        ok && return sortperm(values)
    end
    return collect(1:nrow(df))
end

function _lagged_difference(values::Vector{Float64}, order::Vector{Int}, lag::Int)
    out = zeros(length(values))
    for k in eachindex(order)
        idx = order[k]
        out[idx] = k > lag ? values[idx] - values[order[k - lag]] : 0.0
    end
    return out
end

function _rolling_mean(values::Vector{Float64}, order::Vector{Int}, width::Int)
    out = similar(values)
    for k in eachindex(order)
        lo = max(1, k - width + 1)
        idxs = order[lo:k]
        vals = values[idxs]
        finite = vals[isfinite.(vals)]
        out[order[k]] = isempty(finite) ? 0.0 : mean(finite)
    end
    return out
end

# Per-row anchor DateTime from the issue/latest time column, or `nothing` when no
# usable time axis exists (e.g. the single-row serving feature frame).
function _anchor_times(df::DataFrame)
    for col in (:issue_time_utc, :latest_dst_time_utc)
        String(col) in names(df) || continue
        times = Vector{DateTime}(undef, nrow(df))
        ok = true
        for (i, x) in enumerate(df[!, col])
            if ismissing(x)
                ok = false
                break
            end
            times[i] = x isa DateTime ? x : DateTime(String(x))
        end
        ok && return times
    end
    return nothing
end

# Mean of `v` over the hourly window {t, t-1h, ..., t-(w-1)h} via the anchor
# time→row map `idx_at`; a missing/non-finite window hour falls back to `current`,
# mirroring the live rolling mean's current-driver fallback.
function _window_time_mean(v::Vector{Float64}, idx_at::Dict{DateTime,Int},
                           t::DateTime, w::Int, current::Float64)
    s = 0.0
    for h in 0:w-1
        j = get(idx_at, t - Hour(h), nothing)
        s += (j === nothing || !isfinite(v[j])) ? current : v[j]
    end
    return s / w
end

# Memory features (Dst/Bz/VBsouth deltas and VBsouth/Bsouth rolling means)
# computed by TIMESTAMP, so tied multi-horizon issue times and gap-skipped anchors
# resolve to the true hourly differences and every horizon row of one anchor gets
# identical values. Mirrors _live_v2_memory_features: memory is neutral (0) unless
# the Dst history at t-1h and t-3h is present. Falls back to the row-offset (neutral
# for a lone row) computation when the frame has no usable time axis.
function _add_memory_features!(df::DataFrame, latest::Vector{Float64}, bz::Vector{Float64})
    vbsouth = Float64.(df.VBsouth_mvm)
    bsouth = Float64.(df.Bsouth_nt)
    anchor_times = _anchor_times(df)
    if anchor_times === nothing
        order = _chronological_order(df)
        _maybe_add_column!(df, :dst_delta_1h_nt, _lagged_difference(latest, order, 1))
        _maybe_add_column!(df, :dst_delta_3h_nt, _lagged_difference(latest, order, 3))
        _maybe_add_column!(df, :dst_delta_6h_nt, _lagged_difference(latest, order, 6))
        _maybe_add_column!(df, :Bz_delta_1h_nt, _lagged_difference(bz, order, 1))
        _maybe_add_column!(df, :VBsouth_delta_1h_mvm, _lagged_difference(vbsouth, order, 1))
        _maybe_add_column!(df, :VBsouth_mean_3h_mvm, _rolling_mean(vbsouth, order, 3))
        _maybe_add_column!(df, :VBsouth_mean_6h_mvm, _rolling_mean(vbsouth, order, 6))
        _maybe_add_column!(df, :Bsouth_mean_3h_nt, _rolling_mean(bsouth, order, 3))
        _maybe_add_column!(df, :Bsouth_mean_6h_nt, _rolling_mean(bsouth, order, 6))
        return df
    end
    idx_at = Dict{DateTime,Int}()
    for i in 1:nrow(df)
        get!(idx_at, anchor_times[i], i)   # first occurrence per anchor time
    end
    at(v, t) = (j = get(idx_at, t, nothing); j === nothing ? NaN : v[j])
    n = nrow(df)
    dst_d1 = zeros(n); dst_d3 = zeros(n); dst_d6 = zeros(n)
    bz_d1 = zeros(n); vb_d1 = zeros(n)
    vb_m3 = zeros(n); vb_m6 = zeros(n); bs_m3 = zeros(n); bs_m6 = zeros(n)
    for i in 1:n
        t = anchor_times[i]
        l1 = at(latest, t - Hour(1))
        l3 = at(latest, t - Hour(3))
        (isfinite(latest[i]) && isfinite(l1) && isfinite(l3)) || continue  # neutral memory
        l6 = at(latest, t - Hour(6))
        dst_d1[i] = latest[i] - l1
        dst_d3[i] = latest[i] - l3
        dst_d6[i] = isfinite(l6) ? latest[i] - l6 : 0.0
        bz1 = at(bz, t - Hour(1))
        bz_d1[i] = (isfinite(bz[i]) && isfinite(bz1)) ? bz[i] - bz1 : 0.0
        vbp = at(vbsouth, t - Hour(1))
        vb_d1[i] = (isfinite(vbsouth[i]) && isfinite(vbp)) ? vbsouth[i] - vbp : 0.0
        vb_m3[i] = _window_time_mean(vbsouth, idx_at, t, 3, vbsouth[i])
        vb_m6[i] = _window_time_mean(vbsouth, idx_at, t, 6, vbsouth[i])
        bs_m3[i] = _window_time_mean(bsouth, idx_at, t, 3, bsouth[i])
        bs_m6[i] = _window_time_mean(bsouth, idx_at, t, 6, bsouth[i])
    end
    _maybe_add_column!(df, :dst_delta_1h_nt, dst_d1)
    _maybe_add_column!(df, :dst_delta_3h_nt, dst_d3)
    _maybe_add_column!(df, :dst_delta_6h_nt, dst_d6)
    _maybe_add_column!(df, :Bz_delta_1h_nt, bz_d1)
    _maybe_add_column!(df, :VBsouth_delta_1h_mvm, vb_d1)
    _maybe_add_column!(df, :VBsouth_mean_3h_mvm, vb_m3)
    _maybe_add_column!(df, :VBsouth_mean_6h_mvm, vb_m6)
    _maybe_add_column!(df, :Bsouth_mean_3h_nt, bs_m3)
    _maybe_add_column!(df, :Bsouth_mean_6h_nt, bs_m6)
    return df
end

function _column_float_or_default(df::DataFrame, col::Symbol, fallback::Vector{Float64})
    length(fallback) == nrow(df) ||
        throw(DimensionMismatch("fallback length $(length(fallback)) != $(nrow(df))"))
    String(col) in names(df) || return copy(fallback)
    return Float64[
        ismissing(df[i, col]) ? fallback[i] : Float64(df[i, col])
        for i in 1:nrow(df)
    ]
end

function _storm_guard_prediction(latest_dst::Real, pred_dst::Real, guard_drop::Real)
    latest = Float64(latest_dst)
    pred = Float64(pred_dst)
    isfinite(latest) || return pred
    isfinite(pred) || return latest
    drop = isfinite(Float64(guard_drop)) ? max(Float64(guard_drop), 0.0) : 0.0
    return clamp(min(pred, latest - drop), -2000.0, 50.0)
end

function _maybe_add_column!(df::DataFrame, col::Symbol, values::Vector{Float64})
    String(col) in names(df) && return df
    df[!, col] = values
    return df
end

function add_operational_v2_features!(df::DataFrame)
    base = [:latest_dst_nt, :V_kms, :Bz_nt, :By_nt, :n_cm3, :Pdyn_npa]
    all(String(c) in names(df) for c in base) || return df

    latest = _column_float_or_nan(df, :latest_dst_nt)
    speed = _column_float_or_nan(df, :V_kms)
    bz = _column_float_or_nan(df, :Bz_nt)
    by = _column_float_or_nan(df, :By_nt)
    density = _column_float_or_nan(df, :n_cm3)
    pdyn = _column_float_or_nan(df, :Pdyn_npa)

    features = [
        operational_v2_feature_tuple(latest[i], speed[i], bz[i], by[i], density[i], pdyn[i])
        for i in eachindex(latest)
    ]
    df[!, :Bsouth_nt] = [f.Bsouth_nt for f in features]
    df[!, :VBsouth_mvm] = [f.VBsouth_mvm for f in features]
    df[!, :Bperp_nt] = [f.Bperp_nt for f in features]
    df[!, :clock_angle_sin2] = [f.clock_angle_sin2 for f in features]
    df[!, :sqrt_Pdyn_npa] = [f.sqrt_Pdyn_npa for f in features]

    _add_memory_features!(df, latest, bz)
    if all(String(c) in names(df) for c in (:pred_dst_nt, :persistence_dst_nt))
        v1 = _column_float_or_nan(df, :pred_dst_nt)
        persistence = _column_float_or_nan(df, :persistence_dst_nt)
        _maybe_add_column!(df, :v1_minus_persistence_nt, v1 .- persistence)
        expert_cols = Symbol[:pred_dst_nt, :persistence_dst_nt]
        String(:obrien_dst_nt) in names(df) && push!(expert_cols, :obrien_dst_nt)
        String(:burton_dst_nt) in names(df) && push!(expert_cols, :burton_dst_nt)
        String(:burton_full_dst_nt) in names(df) && push!(expert_cols, :burton_full_dst_nt)
        spread = zeros(nrow(df))
        for i in 1:nrow(df)
            vals = Float64[
                Float64(df[i, c])
                for c in expert_cols
                if !ismissing(df[i, c]) && isfinite(Float64(df[i, c]))
            ]
            spread[i] = isempty(vals) ? 0.0 : maximum(vals) - minimum(vals)
        end
        _maybe_add_column!(df, :baseline_spread_nt, spread)
    end
    if all(String(c) in names(df) for c in (:obrien_dst_nt, :pred_dst_nt))
        _maybe_add_column!(
            df,
            :obrien_minus_v1_nt,
            _column_float_or_nan(df, :obrien_dst_nt) .- _column_float_or_nan(df, :pred_dst_nt),
        )
    end
    if all(String(c) in names(df) for c in (:burton_dst_nt, :pred_dst_nt))
        _maybe_add_column!(
            df,
            :burton_minus_v1_nt,
            _column_float_or_nan(df, :burton_dst_nt) .- _column_float_or_nan(df, :pred_dst_nt),
        )
    end
    dst_delta_1h = _column_float_or_nan(df, :dst_delta_1h_nt)
    dst_delta_3h = _column_float_or_nan(df, :dst_delta_3h_nt)
    dst_delta_6h = _column_float_or_nan(df, :dst_delta_6h_nt)
    vb = Float64.(df.VBsouth_mvm)
    vb6 = Float64.(df.VBsouth_mean_6h_mvm)
    sqrt_pdyn = Float64.(df.sqrt_Pdyn_npa)
    horizon = _column_float_or_default(df, :model_step_hours, ones(nrow(df)))
    horizon = max.(horizon, 1.0)
    main_phase_pressure = max.(-dst_delta_1h, 0.0) .* sqrt_pdyn
    main_phase_pressure_6h = max.(-dst_delta_6h, 0.0) .* sqrt_pdyn
    coupling_active = [
        (vb[i] > 0.0 || dst_delta_1h[i] < 0.0) ? max(vb[i], 0.0) : 0.0
        for i in 1:nrow(df)
    ]
    coupling_active_6h = [
        (vb6[i] > 0.0 || dst_delta_6h[i] < 0.0) ? max(vb6[i], 0.0) : 0.0
        for i in 1:nrow(df)
    ]
    recovery_pressure = max.(dst_delta_3h, 0.0) .* sqrt_pdyn
    main_phase_recovery_pressure = main_phase_pressure_6h .- recovery_pressure
    horizon_coupling_pressure = horizon .* coupling_active .* sqrt_pdyn
    storm_guard_active = Float64.((coupling_active .> 0.0) .| (main_phase_pressure .> 0.0))
    _maybe_add_column!(df, :main_phase_pressure_nt, main_phase_pressure)
    _maybe_add_column!(df, :main_phase_pressure_6h_nt, main_phase_pressure_6h)
    _maybe_add_column!(df, :coupling_active_mvm, coupling_active)
    _maybe_add_column!(df, :coupling_active_6h_mvm, coupling_active_6h)
    _maybe_add_column!(df, :recovery_pressure_nt, recovery_pressure)
    _maybe_add_column!(df, :main_phase_recovery_pressure, main_phase_recovery_pressure)
    _maybe_add_column!(df, :horizon_coupling_pressure, horizon_coupling_pressure)
    _maybe_add_column!(df, :storm_guard_active, storm_guard_active)
    if String(:pred_dst_nt) in names(df)
        pred = _column_float_or_default(df, :pred_dst_nt, latest)
        if !(String(:storm_guard_dst_nt) in names(df))
            df[!, :storm_guard_dst_nt] = [
                _storm_guard_prediction(latest[i], pred[i], horizon_coupling_pressure[i])
                for i in 1:nrow(df)
            ]
        end
        storm_guard = _column_float_or_default(df, :storm_guard_dst_nt, pred)
        _maybe_add_column!(df, :storm_guard_minus_v1_nt, storm_guard .- pred)
    end
    return df
end

function _require_columns(df::DataFrame, cols)
    missing_cols = [String(c) for c in cols if !(String(c) in names(df))]
    isempty(missing_cols) || throw(ArgumentError(
        "missing required calibration column(s): $(join(missing_cols, ", "))"
    ))
    return nothing
end

function _finite_rows(df::DataFrame, cols)
    mask = trues(nrow(df))
    for c in cols
        # Map `missing` → NaN BEFORE the finiteness test: a `missing` (e.g. a not-yet-verified
        # observation row, normal in live replay) must be dropped here, not crash the conversion.
        mask .&= isfinite.(_column_float_or_nan(df, Symbol(c)))
    end
    return df[mask, :]
end

function _standardized_design(df::DataFrame, feature_names::Vector{Symbol},
                              feature_mean::Vector{Float64},
                              feature_scale::Vector{Float64})
    n_features = length(feature_names)
    x = ones(nrow(df), n_features + 1)
    for (j, name) in enumerate(feature_names)
        x[:, j + 1] = (Float64.(df[!, name]) .- feature_mean[j]) ./ feature_scale[j]
    end
    return x
end

function _quantile_sorted(v::Vector{Float64}, q::Real)
    isempty(v) && throw(ArgumentError("cannot compute quantile of empty vector"))
    sorted = sort(v)
    idx = clamp(ceil(Int, Float64(q) * length(sorted)), 1, length(sorted))
    return sorted[idx]
end

function _component_index(cal::OperationalV2Calibration, component::Symbol)
    idx = findfirst(==(component), cal.selector_names)
    idx === nothing && throw(ArgumentError("unknown v2 selector component: $component"))
    return idx
end

function _selector_metric_columns(df::DataFrame)
    out = Pair{Symbol,Symbol}[]
    for spec in OPERATIONAL_V2_BASELINE_COLUMNS
        String(last(spec)) in names(df) && push!(out, spec)
    end
    return out
end

function _candidate_selector_stats(clean::DataFrame, corrected::Vector{Float64},
                                   interval_coverage::Real,
                                   guard_margin_nt::Real)
    obs = Float64.(clean.observation_dst_nt)
    names_out = Symbol[:v2, :sindy_v1]
    preds = [corrected, Float64.(clean.pred_dst_nt)]

    for (component, col) in _selector_metric_columns(clean)
        values = Float64.(clean[!, col])
        all(isfinite, values) || continue
        push!(names_out, component)
        push!(preds, values)
    end

    rmse_vals = Float64[]
    mae_vals = Float64[]
    half_widths = Float64[]
    for p in preds
        residuals = obs .- p
        all(isfinite, residuals) || throw(ArgumentError(
            "selector residuals exceed the supported range",
        ))
        push!(rmse_vals, rmse(p, obs))
        push!(mae_vals, mae(p, obs))
        push!(half_widths, _quantile_sorted(abs.(collect(residuals)), interval_coverage))
    end

    best_idx = argmin(mae_vals)
    selected_idx = if best_idx == 1 || mae_vals[best_idx] + Float64(guard_margin_nt) < mae_vals[1]
        best_idx
    else
        1
    end

    return (
        selector_names=names_out,
        selector_rmse=rmse_vals,
        selector_mae=mae_vals,
        selector_half_width=half_widths,
        selected_component=names_out[selected_idx],
    )
end

function _component_value(baselines, component::Symbol)
    baselines === nothing && throw(ArgumentError(
        "v2 selector component $component requires baseline predictions"
    ))
    value = if baselines isa NamedTuple
        haskey(baselines, component) || throw(ArgumentError(
            "missing v2 selector baseline: $component"
        ))
        getfield(baselines, component)
    else
        haskey(baselines, component) || throw(ArgumentError(
            "missing v2 selector baseline: $component"
        ))
        baselines[component]
    end
    ismissing(value) && throw(ArgumentError("v2 selector baseline $component is missing"))
    return Float64(value)
end

function _selected_component_prediction(component::Symbol, corrected_center::Float64,
                                        original_center::Float64,
                                        cal::OperationalV2Calibration,
                                        baselines)
    component == :v2 && return corrected_center
    component == :sindy_v1 && return original_center
    if component == :ensemble
        prediction = 0.0
        for (name, weight) in zip(cal.selector_names, cal.selector_weights)
            iszero(weight) && continue
            value = if name == :v2
                corrected_center
            elseif name == :sindy_v1
                original_center
            else
                _component_value(baselines, name)
            end
            isfinite(value) ||
                throw(ArgumentError("positive-weight v2 selector component $name is not finite"))
            prediction += weight * value
        end
        return prediction
    end
    value = _component_value(baselines, component)
    isfinite(value) || throw(ArgumentError("v2 selector baseline $component is not finite"))
    return value
end

function _join_symbols(v::Vector{Symbol})
    return join(String.(v), ";")
end

function _join_floats(v::Vector{Float64})
    return join(string.(v), ";")
end

function _split_symbols(s)
    text = String(s)
    isempty(text) && return Symbol[]
    return Symbol.(split(text, ";"))
end

function _split_floats(s)
    s isa Real && return Float64[Float64(s)]
    text = String(s)
    isempty(text) && return Float64[]
    return parse.(Float64, split(text, ";"))
end

"""
    fit_operational_v2_calibration(df; kwargs...)

Fit a causal residual-correction and interval-inflation layer from prior replay
or locked-live rows. Required columns are `pred_dst_nt`,
`observation_dst_nt`, `pred_dst_ci05_nt`, `pred_dst_ci95_nt`, and the selected
issue-time features.
"""
function fit_operational_v2_calibration(df::DataFrame;
        feature_names::Vector{Symbol}=copy(DEFAULT_OPERATIONAL_V2_FEATURES),
        ridge::Real=100.0,
        interval_coverage::Real=0.90,
        label::AbstractString="operational_v2",
        guard_margin_nt::Real=0.5,
    )
    ridge >= 0 || throw(ArgumentError("ridge must be nonnegative"))
    0 < interval_coverage < 1 ||
        throw(ArgumentError("interval_coverage must lie in (0, 1)"))
    isfinite(Float64(guard_margin_nt)) && Float64(guard_margin_nt) >= 0 ||
        throw(ArgumentError("guard_margin_nt must be finite and nonnegative"))
    length(Set(feature_names)) == length(feature_names) ||
        throw(ArgumentError("feature_names must not contain duplicates"))
    required = vcat(
        [:pred_dst_nt, :observation_dst_nt, :pred_dst_ci05_nt, :pred_dst_ci95_nt],
        feature_names,
    )
    prepared = add_operational_v2_features!(copy(df))
    _require_columns(prepared, required)
    clean = _finite_rows(prepared, required)
    nrow(clean) > length(feature_names) + 1 ||
        throw(ArgumentError("not enough finite rows to fit V2 calibration"))

    μ = Float64[mean(Float64.(clean[!, c])) for c in feature_names]
    σ = Float64[std(Float64.(clean[!, c])) for c in feature_names]
    σ = Float64[s == 0.0 ? 1.0 : s for s in σ]
    x = _standardized_design(clean, feature_names, μ, σ)
    y = Float64.(clean.observation_dst_nt) .- Float64.(clean.pred_dst_nt)
    λ = Float64(ridge)
    β = if iszero(λ)
        x \ y
    else
        penalty = Matrix{Float64}(I, size(x, 2), size(x, 2))
        penalty[1, 1] = 0.0
        (x' * x + λ * penalty) \ (x' * y)
    end

    corrected = Float64.(clean.pred_dst_nt) .+ x * β
    half_width = abs.(Float64.(clean.pred_dst_ci95_nt) .-
                      Float64.(clean.pred_dst_ci05_nt)) ./ 2
    ratios = abs.(Float64.(clean.observation_dst_nt) .- corrected) ./
             max.(half_width, eps(Float64))
    interval_scale = max(1.0, _quantile_sorted(collect(ratios), interval_coverage))
    selector = _candidate_selector_stats(
        clean,
        collect(corrected),
        interval_coverage,
        guard_margin_nt,
    )

    return OperationalV2Calibration(
        feature_names,
        μ,
        σ,
        collect(β),
        interval_scale,
        label,
        selector_names=selector.selector_names,
        selector_rmse=selector.selector_rmse,
        selector_mae=selector.selector_mae,
        selector_half_width=selector.selector_half_width,
        selected_component=selector.selected_component,
        guard_margin_nt=guard_margin_nt,
    )
end

function operational_v2_correction(cal::OperationalV2Calibration,
                                   features::NamedTuple)
    correction = cal.coefficients[1]
    for (j, name) in enumerate(cal.feature_names)
        haskey(features, name) ||
            throw(ArgumentError("missing v2 feature: $(String(name))"))
        value = Float64(getfield(features, name))
        isfinite(value) || throw(ArgumentError(
            "v2 feature $(String(name)) must be finite and representable as Float64",
        ))
        centered = value - cal.feature_mean[j]
        isfinite(centered) || throw(ArgumentError(
            "v2 feature $(String(name)) centered value exceeds Float64 range",
        ))
        contribution = cal.coefficients[j + 1] *
                       (centered / cal.feature_scale[j])
        isfinite(contribution) || throw(ArgumentError(
            "v2 feature $(String(name)) contribution became non-finite",
        ))
        correction += contribution
        isfinite(correction) || throw(ArgumentError(
            "v2 correction became non-finite",
        ))
    end
    return correction
end

"""
    operational_v2_predict(cal, pred_dst, ci05, ci95, features)

Apply a fitted v2 residual correction and interval inflation to a v1 Dst
forecast. `features` must contain the calibration feature names. If the
calibration selected a baseline component, pass `baselines` with matching
component names such as `obrien` or `burton_full`.
"""
function operational_v2_predict(cal::OperationalV2Calibration,
                                pred_dst::Real,
                                ci05::Real,
                                ci95::Real,
                                features::NamedTuple;
                                baselines=nothing)
    pred64 = Float64(pred_dst)
    lower64 = Float64(ci05)
    upper64 = Float64(ci95)
    all(isfinite, (pred64, lower64, upper64)) || throw(ArgumentError(
        "v2 point and interval inputs must be finite and representable as Float64",
    ))
    correction = operational_v2_correction(cal, features)
    # Clamp the corrected center to the physical Dst range used by every other
    # forecast path (v1 single/multi-step, baselines, storm guard). On valid
    # operational data the corrected center stays well inside [-2000, 50] (max
    # observed prediction ~46 nT), so this is a no-op there; it only guards the
    # selector :v2 path against a pathological positive correction.
    unbounded_center = pred64 + correction
    isfinite(unbounded_center) || throw(ArgumentError(
        "v2 corrected center exceeds the supported Float64 range",
    ))
    corrected_center = clamp(unbounded_center, -2000.0, 50.0)
    original_center = pred64
    # Re-clamp the selected center: the :ensemble path is a weighted sum of
    # components and could otherwise land outside the physical range.
    selected_center = _selected_component_prediction(
        cal.selected_component,
        corrected_center,
        original_center,
        cal,
        baselines,
    )
    isfinite(selected_center) || throw(ArgumentError(
        "v2 selected center became non-finite",
    ))
    center = clamp(selected_center, -2000.0, 50.0)
    span = upper64 - lower64
    half_width = isfinite(span) ? abs(span) / 2 : abs(upper64 / 2 - lower64 / 2)
    isfinite(half_width) || throw(ArgumentError(
        "v2 interval half-width exceeds the supported Float64 range",
    ))
    selector_half_width = if cal.selected_component == :ensemble
        sum(cal.selector_weights .* cal.selector_half_width)
    else
        cal.selector_half_width[_component_index(cal, cal.selected_component)]
    end
    inflated = max(cal.interval_scale * half_width, selector_half_width)
    isfinite(inflated) || throw(ArgumentError(
        "v2 inflated interval half-width exceeds the supported Float64 range",
    ))
    ci05_out = center - inflated
    ci95_out = center + inflated
    all(isfinite, (ci05_out, ci95_out)) || throw(ArgumentError(
        "v2 interval endpoints exceed the supported Float64 range",
    ))
    return (
        pred_dst=center,
        ci05_dst=ci05_out,
        ci95_dst=ci95_out,
        correction=correction,
        interval_scale=cal.interval_scale,
        label=cal.label,
        selected_component=String(cal.selected_component),
        selected_component_pred=center,
        corrected_sindy_pred=corrected_center,
        selector_half_width=selector_half_width,
    )
end

function _row_baselines(df::DataFrame, row_idx::Int)
    baselines = Dict{Symbol,Float64}()
    for (component, col) in OPERATIONAL_V2_BASELINE_COLUMNS
        String(col) in names(df) || continue
        value = df[row_idx, col]
        if !ismissing(value)
            baselines[component] = Float64(value)
        end
    end
    return baselines
end

function score_operational_v2(df::DataFrame, cal::OperationalV2Calibration)
    out = add_operational_v2_features!(copy(df))
    _require_columns(out, vcat(
        [:pred_dst_nt, :pred_dst_ci05_nt, :pred_dst_ci95_nt, :observation_dst_nt],
        cal.feature_names,
    ))
    v2_pred = Vector{Float64}(undef, nrow(out))
    v2_ci05 = Vector{Float64}(undef, nrow(out))
    v2_ci95 = Vector{Float64}(undef, nrow(out))
    v2_corr = Vector{Float64}(undef, nrow(out))
    v2_corrected = Vector{Float64}(undef, nrow(out))
    v2_residual = Vector{Float64}(undef, nrow(out))
    v2_in_ci = Vector{Bool}(undef, nrow(out))
    v2_component = Vector{String}(undef, nrow(out))
    v2_component_pred = Vector{Float64}(undef, nrow(out))
    needs_baselines = if cal.selected_component == :ensemble
        any(name != :v2 && name != :sindy_v1 && !iszero(weight)
            for (name, weight) in zip(cal.selector_names, cal.selector_weights))
    else
        cal.selected_component != :v2 && cal.selected_component != :sindy_v1
    end

    for i in 1:nrow(out)
        features = NamedTuple{Tuple(cal.feature_names)}(
            Tuple(_cell_float_or_nan(out[i, c]) for c in cal.feature_names)
        )
        pred = operational_v2_predict(
            cal,
            _cell_float_or_nan(out[i, :pred_dst_nt]),
            _cell_float_or_nan(out[i, :pred_dst_ci05_nt]),
            _cell_float_or_nan(out[i, :pred_dst_ci95_nt]),
            features,
            baselines=needs_baselines ? _row_baselines(out, i) : nothing,
        )
        obs = _cell_float_or_nan(out[i, :observation_dst_nt])
        v2_pred[i] = pred.pred_dst
        v2_ci05[i] = pred.ci05_dst
        v2_ci95[i] = pred.ci95_dst
        v2_corr[i] = pred.correction
        v2_corrected[i] = pred.corrected_sindy_pred
        v2_residual[i] = obs - pred.pred_dst
        v2_in_ci[i] = min(pred.ci05_dst, pred.ci95_dst) <= obs <=
                      max(pred.ci05_dst, pred.ci95_dst)
        v2_component[i] = pred.selected_component
        v2_component_pred[i] = pred.selected_component_pred
    end

    out[!, :v2_pred_dst_nt] = v2_pred
    out[!, :v2_pred_dst_ci05_nt] = v2_ci05
    out[!, :v2_pred_dst_ci95_nt] = v2_ci95
    out[!, :v2_correction_dst_nt] = v2_corr
    out[!, :v2_corrected_sindy_pred_nt] = v2_corrected
    out[!, :v2_residual_dst_nt] = v2_residual
    out[!, :v2_observed_in_90ci] = v2_in_ci
    out[!, :v2_calibration_label] = fill(cal.label, nrow(out))
    out[!, :v2_selected_component] = v2_component
    out[!, :v2_selected_component_pred_nt] = v2_component_pred
    return out
end

function write_operational_v2_calibration(path::String,
                                          cal::OperationalV2Calibration)
    dir = dirname(path)
    !isempty(dir) && mkpath(dir)
    rows = NamedTuple[]
    selector_components = _join_symbols(cal.selector_names)
    selector_rmse = _join_floats(cal.selector_rmse)
    selector_mae = _join_floats(cal.selector_mae)
    selector_half_width = _join_floats(cal.selector_half_width)
    selector_weights = _join_floats(cal.selector_weights)
    push!(rows, (
        feature="intercept",
        feature_mean=0.0,
        feature_scale=1.0,
        coefficient=cal.coefficients[1],
        interval_scale=cal.interval_scale,
        label=cal.label,
        selected_component=String(cal.selected_component),
        guard_margin_nt=cal.guard_margin_nt,
        selector_components=selector_components,
        selector_rmse_nt=selector_rmse,
        selector_mae_nt=selector_mae,
        selector_half_width_nt=selector_half_width,
        selector_weights=selector_weights,
    ))
    for (j, name) in enumerate(cal.feature_names)
        push!(rows, (
            feature=String(name),
            feature_mean=cal.feature_mean[j],
            feature_scale=cal.feature_scale[j],
            coefficient=cal.coefficients[j + 1],
            interval_scale=cal.interval_scale,
            label=cal.label,
            selected_component=String(cal.selected_component),
            guard_margin_nt=cal.guard_margin_nt,
            selector_components=selector_components,
            selector_rmse_nt=selector_rmse,
            selector_mae_nt=selector_mae,
            selector_half_width_nt=selector_half_width,
            selector_weights=selector_weights,
        ))
    end
    _write_selection_csv(path, rows)
    return path
end

function read_operational_v2_calibration(path::String)
    df = CSV.read(path, DataFrame)
    _require_columns(df, [:feature, :feature_mean, :feature_scale,
                          :coefficient, :interval_scale, :label])
    nrow(df) >= 1 || throw(ArgumentError("empty V2 calibration: $path"))
    String(df.feature[1]) == "intercept" ||
        throw(ArgumentError("first calibration row must be feature=intercept"))
    feature_names = Symbol.(String.(df.feature[2:end]))
    length(Set(feature_names)) == length(feature_names) ||
        throw(ArgumentError("V2 calibration contains duplicate feature names: $path"))
    feature_mean = Float64.(df.feature_mean[2:end])
    feature_scale = Float64.(df.feature_scale[2:end])
    coefficients = Float64.(df.coefficient)
    interval_scale = Float64(df.interval_scale[1])
    label = String(df.label[1])
    selector_names = String(:selector_components) in names(df) ?
        _split_symbols(df[1, :selector_components]) : Symbol[:v2]
    selector_rmse = String(:selector_rmse_nt) in names(df) ?
        _split_floats(df[1, :selector_rmse_nt]) : fill(NaN, length(selector_names))
    selector_mae = String(:selector_mae_nt) in names(df) ?
        _split_floats(df[1, :selector_mae_nt]) : fill(NaN, length(selector_names))
    selector_half_width = String(:selector_half_width_nt) in names(df) ?
        _split_floats(df[1, :selector_half_width_nt]) : zeros(length(selector_names))
    selector_weights = String(:selector_weights) in names(df) ?
        _split_floats(df[1, :selector_weights]) : nothing
    selected_component = String(:selected_component) in names(df) ?
        Symbol(String(df[1, :selected_component])) : :v2
    guard_margin_nt = String(:guard_margin_nt) in names(df) ?
        Float64(df[1, :guard_margin_nt]) : 0.0
    return OperationalV2Calibration(
        collect(feature_names),
        collect(feature_mean),
        collect(feature_scale),
        collect(coefficients),
        interval_scale,
        label,
        selector_names=collect(selector_names),
        selector_rmse=collect(selector_rmse),
        selector_mae=collect(selector_mae),
        selector_half_width=collect(selector_half_width),
        selector_weights=selector_weights === nothing ? nothing : collect(selector_weights),
        selected_component=selected_component,
        guard_margin_nt=guard_margin_nt,
    )
end

"""
    init_forecast(; coefficients_csv, ensemble_csv, t0, dst0, dt=1.0)

Initialise a ForecastState from saved discovery results.

- `coefficients_csv`: path to real_sindy_discovery_coefficients.csv
- `ensemble_csv`: path to real_ensemble_inclusion.csv (marginal per-term
  conditional nonzero empirical intervals; legacy `ci_025`/`ci_975` columns
  are also accepted)
- `draws_csv`: path to the joint posterior-draws artifact
  (real_sindy_ensemble_draws.csv; columns = library term names, one draw per row).
  When present it is resampled and recentered on the deployed coefficients;
  otherwise the ensemble falls back to marginal per-term sampling with a warning.
- `t0`: initial DateTime (UTC)
- `dst0`: initial Dst* value [nT]
"""
# Build the coefficient uncertainty ensemble by resampling rows of a joint-draws
# artifact (columns = library term names, one posterior draw per row), recentered
# on the deployed point coefficients so it describes the model actually issued.
# Only active (deployed-nonzero) terms carry spread; inactive terms stay zero.
# Returns the n_ensemble×n_terms matrix, or `nothing` if the artifact carries no
# usable active-term column (signalling the caller to fall back to marginal draws).
function _ensemble_from_joint_draws(path::String, term_names::Vector{String},
                                    ξ_primary::Vector{Float64}, n_ensemble::Int, rng)
    draws = CSV.read(path, DataFrame)
    D = nrow(draws)
    D >= 1 || return nothing
    colnames = names(draws)
    colmap = Tuple{Int,Symbol}[]           # (library index, draw column) for active terms
    for (idx, term) in enumerate(term_names)
        (ξ_primary[idx] == 0.0 || !(term in colnames)) && continue
        push!(colmap, (idx, Symbol(term)))
    end
    isempty(colmap) && return nothing
    missing_active = [term_names[idx] for idx in eachindex(term_names)
                      if ξ_primary[idx] != 0.0 &&
                         all(first(pair) != idx for pair in colmap)]
    isempty(missing_active) ||
        @warn "Joint-draws artifact omits active terms; omitted terms remain fixed at their point coefficients" missing_active maxlog=1
    valid_rows = [r for r in 1:D if all(begin
        value = draws[r, c]
        !ismissing(value) && isfinite(Float64(value))
    end for (_, c) in colmap)]
    isempty(valid_rows) && return nothing
    μ = Dict{Symbol,Float64}()             # per-term draw mean, for recentering on ξ_primary
    for (_, c) in colmap
        μ[c] = mean(Float64(draws[r, c]) for r in valid_rows)
    end
    ξ_ensemble = repeat(reshape(ξ_primary, 1, :), n_ensemble, 1)
    for i in 1:n_ensemble
        r = rand(rng, valid_rows)          # resample a complete joint draw
        for (idx, c) in colmap
            ξ_ensemble[i, idx] = ξ_primary[idx] + (Float64(draws[r, c]) - μ[c])
        end
    end
    return ξ_ensemble
end

function init_forecast(; coefficients_csv::String,
                         ensemble_csv::String,
                         draws_csv::String=joinpath(dirname(coefficients_csv),
                                                    "real_sindy_ensemble_draws.csv"),
                         t0::DateTime,
                         dst0::Float64,
                         dt::Float64=1.0)
    # Load primary coefficients
    coef_df = CSV.read(coefficients_csv, DataFrame)
    all(c -> String(c) in names(coef_df), (:term, :coefficient)) ||
        throw(ArgumentError("coefficient CSV must contain term and coefficient columns"))
    # Legacy deployed artifacts contain the now-removed exact pressure proxy
    # `n*V^2`. Opt into that 21-term representation only when the artifact
    # explicitly requires it; new/canonical artifacts use the identifiable
    # default library.
    # CSV.jl may infer a one-row intercept-only `term` column as `Int64` for the
    # textual term `"1"`.  Normalize through `string`, which is defined for both
    # numeric and textual cells, instead of requiring an existing String value.
    coefficient_terms = string.(coef_df.term)
    lib = build_solar_wind_library(
        include_redundant_n_v2=("n*V^2" in coefficient_terms),
    )
    term_names = get_term_names(lib)

    # Map CSV terms to library indices
    ξ_primary = zeros(length(lib))
    seen_terms = Set{String}()
    for row in eachrow(coef_df)
        term = string(row.term)
        term in seen_terms && throw(ArgumentError("duplicate coefficient term: $term"))
        push!(seen_terms, term)
        idx = findfirst(==(term), term_names)
        idx === nothing && throw(ArgumentError("unknown coefficient term: $term"))
        coefficient = Float64(row.coefficient)
        isfinite(coefficient) || throw(ArgumentError("non-finite coefficient for term $term"))
        ξ_primary[idx] = coefficient
    end

    # Build the uncertainty ensemble AROUND the deployed point coefficients
    # (ξ_primary), so the ensemble describes the model that is actually issued.
    # Per-term spread comes from the empirical interval width in the ensemble CSV,
    # mapped to an approximate Gaussian 95% interval (±z·σ, z=1.96).
    # Every active primary term is perturbed (terms with no interval / zero width keep
    # ξ_primary exactly); terms absent from ξ_primary stay zero. This fixes the
    # prior defects where the ensemble was centered on a different fit (the
    # ensemble CSV medians) and silently dropped active terms with π<0.9.
    n_ensemble = 500
    n_terms = length(lib)
    rng = MersenneTwister(42)
    z95 = 1.959963984540054   # standard normal 97.5th percentile

    # Preferred: resample rows of the joint posterior-draws artifact (preserves
    # cross-term covariance), recentered on the deployed point coefficients.
    ξ_ensemble = isfile(draws_csv) ?
        _ensemble_from_joint_draws(draws_csv, term_names, ξ_primary, n_ensemble, rng) : nothing

    if ξ_ensemble === nothing
        isfile(draws_csv) &&
            @warn "Joint-draws artifact present but unusable; using marginal per-term ensemble" draws_csv maxlog=1
        isfile(draws_csv) ||
            @warn "Joint-draws artifact not found; using marginal per-term ensemble" draws_csv maxlog=1
        # Fallback: marginal per-term Gaussian spread from the empirical interval width
        # (±z·σ, z=1.96 → σ = width/(2z)). Only active terms are perturbed.
        isfile(ensemble_csv) ||
            throw(ArgumentError("ensemble coefficient artifact not found: $ensemble_csv"))
        ens_df = CSV.read(ensemble_csv, DataFrame)
        available = Set(names(ens_df))
        "term" in available || throw(ArgumentError(
            "ensemble CSV must contain a term column",
        ))
        interval_columns = if all(in(available),
                                  ("conditional_nonzero_empirical_q025",
                                   "conditional_nonzero_empirical_q975"))
            (:conditional_nonzero_empirical_q025,
             :conditional_nonzero_empirical_q975)
        elseif all(in(available), ("ci_025", "ci_975"))
            (:ci_025, :ci_975)
        else
            throw(ArgumentError(
                "ensemble CSV must contain canonical conditional-nonzero " *
                "q025/q975 columns or legacy ci_025/ci_975 columns",
            ))
        end
        nrow(ens_df) >= 1 || throw(ArgumentError("ensemble CSV contains no coefficient rows"))
        σ_term = zeros(n_terms)
        seen_ensemble_terms = Set{String}()
        for row in eachrow(ens_df)
            term = string(row.term)
            term in seen_ensemble_terms &&
                throw(ArgumentError("duplicate ensemble coefficient term: $term"))
            push!(seen_ensemble_terms, term)
            idx = findfirst(==(term), term_names)
            idx === nothing && throw(ArgumentError("unknown ensemble coefficient term: $term"))
            lo = Float64(row[interval_columns[1]])
            hi = Float64(row[interval_columns[2]])
            if isnan(lo) && isnan(hi)
                σ_term[idx] = 0.0
                continue
            end
            isfinite(lo) && isfinite(hi) && lo <= hi ||
                throw(ArgumentError("invalid ensemble interval for term $term"))
            width = hi - lo
            σ_term[idx] = width > 0 ? width / (2 * z95) : 0.0
        end
        ξ_ensemble = zeros(n_ensemble, n_terms)
        for idx in 1:n_terms
            ξ_primary[idx] == 0.0 && continue   # only active (deployed) terms get spread
            σ = σ_term[idx]
            for i in 1:n_ensemble
                ξ_ensemble[i, idx] = ξ_primary[idx] + (σ > 0 ? σ * randn(rng) : 0.0)
            end
        end
    end

    return ForecastState(t0, dst0, lib, ξ_primary, ξ_ensemble, dt,
                         ForecastResult[])
end

"""
    _evaluate_point(lib, dst_star, V, Bz, By, n, Pdyn)

Evaluate library at a single space-time point. Returns 1×p matrix.
"""
function _evaluate_point(lib::CandidateLibrary,
                         dst_star::Float64, V::Float64,
                         Bz::Float64, By::Float64,
                         n_density::Float64, Pdyn::Float64)
    θ = Vector{Float64}(undef, length(lib))
    _evaluate_point_vector!(θ, lib, dst_star, V, Bz, By, n_density, Pdyn)
    return reshape(θ, 1, :)
end

function _point_data(dst_star::Float64, V::Float64, Bz::Float64, By::Float64,
                     n_density::Float64, Pdyn::Float64, Bs::Float64,
                     theta_c::Float64, BT::Float64)
    return Dict{String,Vector{Float64}}(
        "V"        => [V],
        "Bs"       => [Bs],
        "n"        => [n_density],
        "Pdyn"     => [Pdyn],
        "Dst_star" => [dst_star],
        "theta_c"  => [theta_c],
        "BT"       => [BT],
        "By"       => [By],
        "Bz"       => [Bz],
    )
end

function _evaluate_point_vector!(θ::Vector{Float64},
                                 lib::CandidateLibrary,
                                 dst_star::Float64, V::Float64,
                                 Bz::Float64, By::Float64,
                                 n_density::Float64, Pdyn::Float64)
    _validate_candidate_library(lib)
    return _evaluate_point_vector_unchecked!(
        θ, lib, dst_star, V, Bz, By, n_density, Pdyn,
    )
end

# Internal hot-loop kernel. Public callers validate the immutable semantic
# contract once before entering their loop; immutable tuples are used here so a
# concurrent mutation of the legacy public vectors cannot change equations
# between iterations.
function _evaluate_point_vector_unchecked!(θ::Vector{Float64},
                                           lib::CandidateLibrary,
                                           dst_star::Float64, V::Float64,
                                           Bz::Float64, By::Float64,
                                           n_density::Float64, Pdyn::Float64)
    n_terms = _validate_candidate_library_structure(lib)
    length(θ) == n_terms || throw(DimensionMismatch(
        "θ length $(length(θ)) != library length $n_terms",
    ))
    Bs = max(-Bz, 0.0)
    theta_c = atan(abs(By), Bz)
    BT = hypot(By, Bz)
    sin_half = sin(theta_c / 2)
    fallback_data = nothing

    for j in eachindex(lib._contract_term_codes)
        code = lib._contract_term_codes[j]
        if code == TERM_ONE
            θ[j] = 1.0
        elseif code == TERM_V
            θ[j] = V
        elseif code == TERM_BS
            θ[j] = Bs
        elseif code == TERM_N
            θ[j] = n_density
        elseif code == TERM_PDYN
            θ[j] = Pdyn
        elseif code == TERM_DST_STAR
            θ[j] = dst_star
        elseif code == TERM_V2
            θ[j] = V^2
        elseif code == TERM_BS2
            θ[j] = Bs^2
        elseif code == TERM_N2
            θ[j] = n_density^2
        elseif code == TERM_V_BS
            θ[j] = V * Bs
        elseif code == TERM_N_V
            θ[j] = n_density * V
        elseif code == TERM_N_BS
            θ[j] = n_density * Bs
        elseif code == TERM_PDYN_BS
            θ[j] = Pdyn * Bs
        elseif code == TERM_N_V_BS
            θ[j] = n_density * V * Bs
        elseif code == TERM_N_V2
            θ[j] = n_density * V^2
        elseif code == TERM_SIN_HALF
            θ[j] = sin_half
        elseif code == TERM_SIN_HALF2
            θ[j] = sin_half^2
        elseif code == TERM_SIN_HALF4
            θ[j] = sin_half^4
        elseif code == TERM_SIN_HALF_8_3
            θ[j] = sin_half^(8/3)
        elseif code == TERM_V_SIN_HALF2
            θ[j] = V * sin_half^2
        elseif code == TERM_NEWELL
            θ[j] = _newell_coupling_value(V, BT, sin_half)
        else
            if fallback_data === nothing
                fallback_data = _point_data(dst_star, V, Bz, By, n_density, Pdyn, Bs, theta_c, BT)
            end
            value = lib._contract_funcs[j](fallback_data)
            value isa AbstractVector || throw(ArgumentError(
                "custom point-evaluation term $(lib._contract_names[j]) must return a vector",
            ))
            length(value) == 1 || throw(DimensionMismatch(
                "custom point-evaluation term $(lib._contract_names[j]) returned $(length(value)) values; expected 1"
            ))
            raw_value = value[1]
            raw_value isa Real && !(raw_value isa Bool) && isfinite(raw_value) ||
                throw(ArgumentError(
                    "custom point-evaluation term $(lib._contract_names[j]) returned a non-finite or non-real value",
                ))
            converted = Float64(raw_value)
            isfinite(converted) || throw(ArgumentError(
                "custom point-evaluation term $(lib._contract_names[j]) exceeds the supported Float64 range",
            ))
            θ[j] = converted
        end
    end

    all(isfinite, θ) ||
        throw(ArgumentError("candidate-library point evaluation produced a non-finite value"))

    return θ
end

function _dot_ensemble_row(θ::Vector{Float64}, ξ_ensemble::Matrix{Float64}, i::Int)
    s = 0.0
    @inbounds @simd for j in eachindex(θ)
        s += θ[j] * ξ_ensemble[i, j]
    end
    return s
end

function _clamped_finite_derivative(value::Real, label::AbstractString)
    isfinite(value) || throw(ArgumentError("$label derivative is non-finite"))
    return clamp(Float64(value), -200.0, 200.0)
end

function _validate_forecast_step_time(state::ForecastState, t::DateTime)
    elapsed = t - state.t_current
    elapsed > Millisecond(0) ||
        throw(ArgumentError("forecast step time must be later than the current state time"))
    expected = _forecast_step_period(state.dt)
    elapsed == expected ||
        throw(ArgumentError(
            "forecast step elapsed time ($(Dates.value(elapsed)) ms) does not match " *
            "state.dt ($(state.dt) h)"
        ))
    return nothing
end

function _validate_forecast_state(state::ForecastState)
    n_terms = _validate_candidate_library(state.lib)
    length(state.ξ_primary) == n_terms || throw(DimensionMismatch(
        "primary coefficient count does not match the mutable forecast library",
    ))
    size(state.ξ_ensemble, 2) == n_terms || throw(DimensionMismatch(
        "ensemble coefficient count does not match the mutable forecast library",
    ))
    size(state.ξ_ensemble, 1) >= 1 || throw(ArgumentError(
        "forecast state requires at least one ensemble member",
    ))
    all(isfinite, state.ξ_primary) ||
        throw(ArgumentError("primary coefficients must remain finite"))
    all(isfinite, state.ξ_ensemble) ||
        throw(ArgumentError("ensemble coefficients must remain finite"))
    isfinite(state.dst_current) ||
        throw(ArgumentError("forecast state Dst* must remain finite"))
    _forecast_step_period(state.dt)
    return nothing
end

"""
    step_forecast!(state, t, V, Bz, By, n, Pdyn; dst_observed=NaN)

Advance forecast by one time step using solar wind inputs.
Updates state.dst_current and appends to state.history.

Returns ForecastResult.
"""
function step_forecast!(state::ForecastState,
                        t::DateTime,
                        V::Float64, Bz::Float64, By::Float64,
                        n_density::Float64, Pdyn::Float64;
                        dst_observed::Float64=NaN)
    all(isfinite, (V, Bz, By, n_density, Pdyn)) ||
        throw(ArgumentError("forecast drivers must be finite"))
    V >= 0 && n_density >= 0 && Pdyn >= 0 ||
        throw(ArgumentError("V, density, and dynamic pressure must be nonnegative"))
    (isnan(dst_observed) || isfinite(dst_observed)) ||
        throw(ArgumentError("dst_observed must be finite or NaN"))
    _validate_forecast_state(state)
    _validate_forecast_step_time(state, t)
    θ_k = Vector{Float64}(undef, length(state.lib))
    _evaluate_point_vector_unchecked!(θ_k, state.lib, state.dst_current,
                                      V, Bz, By, n_density, Pdyn)

    # Primary prediction
    dDst = _clamped_finite_derivative(dot(θ_k, state.ξ_primary), "primary")
    dst_next = clamp(state.dst_current + state.dt * dDst, -2000.0, 50.0)

    # Ensemble predictions
    n_ens = size(state.ξ_ensemble, 1)
    dst_ens = Vector{Float64}(undef, n_ens)
    for i in 1:n_ens
        dDst_i = _clamped_finite_derivative(
            _dot_ensemble_row(θ_k, state.ξ_ensemble, i), "ensemble member $i")
        dst_ens[i] = clamp(state.dst_current + state.dt * dDst_i, -2000.0, 50.0)
    end
    sort!(dst_ens)

    i50 = clamp(ceil(Int, 0.50 * n_ens), 1, n_ens)   # ensemble percentile indices, ordering-enforced: i05<=i50<=i95
    i05 = clamp(ceil(Int, 0.05 * n_ens), 1, i50)
    i95 = clamp(ceil(Int, 0.95 * n_ens), i50, n_ens)
    result = ForecastResult(
        t,
        dst_next,
        dst_ens[i50],   # median
        dst_ens[i05],   # 5th percentile
        dst_ens[i95],   # 95th percentile
        dst_observed,
    )

    # Update state
    if isfinite(dst_observed)
        state.dst_current = dst_observed   # anchor to observation when available
    else
        state.dst_current = dst_next
    end
    state.t_current = t
    push!(state.history, result)

    return result
end

"""
    forecast_ahead(state, V, Bz, By, n, Pdyn, n_steps)

Multi-step forecast assuming persistence of current solar wind conditions. Each
step advances by `state.dt` hours. Does NOT modify state. Returns
`Vector{ForecastResult}`.
"""
function forecast_ahead(state::ForecastState,
                        V::Float64, Bz::Float64, By::Float64,
                        n_density::Float64, Pdyn::Float64,
                        n_steps::Int)
    n_steps >= 0 || throw(ArgumentError("n_steps must be nonnegative, got $n_steps"))
    all(isfinite, (V, Bz, By, n_density, Pdyn)) ||
        throw(ArgumentError("forecast drivers must be finite"))
    V >= 0 && n_density >= 0 && Pdyn >= 0 ||
        throw(ArgumentError("V, density, and dynamic pressure must be nonnegative"))
    _validate_forecast_state(state)
    results = ForecastResult[]
    dst_curr = state.dst_current
    t_curr = state.t_current

    n_ens = size(state.ξ_ensemble, 1)
    dst_ens_curr = fill(dst_curr, n_ens)
    θ_k = Vector{Float64}(undef, length(state.lib))

    # DateTime has millisecond resolution, so the timestamp step must use the
    # same exactly representable period validated by ForecastState.
    step = _forecast_step_period(state.dt)
    for _ in 1:n_steps
        t_next = t_curr + step

        _evaluate_point_vector_unchecked!(
            θ_k, state.lib, dst_curr, V, Bz, By, n_density, Pdyn,
        )

        # Primary
        dDst = _clamped_finite_derivative(dot(θ_k, state.ξ_primary), "primary")
        dst_next = clamp(dst_curr + state.dt * dDst, -2000.0, 50.0)

        # Ensemble (each starts from its own previous prediction)
        for i in 1:n_ens
            _evaluate_point_vector_unchecked!(
                θ_k, state.lib, dst_ens_curr[i], V, Bz, By, n_density, Pdyn,
            )
            dDst_i = _clamped_finite_derivative(
                _dot_ensemble_row(θ_k, state.ξ_ensemble, i), "ensemble member $i")
            dst_ens_curr[i] = clamp(dst_ens_curr[i] + state.dt * dDst_i, -2000.0, 50.0)
        end
        sorted_ens = sort(dst_ens_curr)
        j50 = clamp(ceil(Int, 0.50 * n_ens), 1, n_ens)   # ordering-enforced ensemble percentile indices
        j05 = clamp(ceil(Int, 0.05 * n_ens), 1, j50)
        j95 = clamp(ceil(Int, 0.95 * n_ens), j50, n_ens)
        push!(results, ForecastResult(
            t_next, dst_next,
            sorted_ens[j50],
            sorted_ens[j05],
            sorted_ens[j95],
            NaN,
        ))

        dst_curr = dst_next
        t_curr = t_next
    end

    return results
end
