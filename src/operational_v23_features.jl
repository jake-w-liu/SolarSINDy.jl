# operational_v23_features.jl — Operational V2.3 analog-driver-continuation (ADC) issue-time features.
#
# The V2.3 study replaces the served V2.1 heuristic tail (an exponential relaxation of the last
# L1-known driver) with a driver continuation drawn from the observed solar-wind archive. Selecting
# that continuation needs a fixed, causal description of the solar-wind and ring-current state at
# issue time. This file defines that description and nothing else: it does not forecast, does not
# search, and does not touch the SINDy core.
#
# Time convention (hourly OMNI; the record tagged τ covers the at-Earth interval [τ, τ+1)):
#   * issue time t is on the hour and the issue-time Dst is Dst(t) (anchor lag a = 0);
#   * the freshest driver record available at issue t is the record tagged t−1, so driver feature
#     lag j (j = 0 … 6) is the record tagged t−1−j;
#   * six-hour means and standard deviations therefore span records t−1 … t−6, the three-hour VBs
#     mean spans t−1 … t−3, and the six-hour speed increment uses records t−1 and t−7.
# The history window for issue t is complete when records t−1 … t−7 are all present in the causal
# forward-filled hourly frame with finite, physically admissible driver values and Dst(t), Dst(t−1)
# are both finite. Incomplete windows return NaN features and `ok = false`; they are never imputed,
# because an imputed analog key would silently change which archive origins are retrieved.

"""
    V23_FEATURE_NAMES

Ordered names of the 18 issue-time features that key the V2.3 analog search. The order is fixed:
every feature matrix, weight vector, and standardisation tuple in the study is interpreted in this
order, so it must not be permuted.
"""
const V23_FEATURE_NAMES = (
    :bz0, :bz1, :bz2, :bz_mean6, :bz_sd6,
    :by0, :bperp0, :bperp_mean6,
    :v0, :dv6,
    :logn0, :logn_mean6, :pdyn0,
    :vbs0, :vbs_mean3, :south_run,
    :dst0, :ddst1,
)

"Number of V2.3 issue-time features."
const V23_FEATURE_COUNT = length(V23_FEATURE_NAMES)

"Driver-record lags (in hours before the issue time) that must be present for a complete history."
const V23_HISTORY_LAGS_H = 7

"Cap on the consecutive-southward run-length feature (hours)."
const V23_SOUTH_RUN_CAP_H = 12

"Features carrying the magnetic-heavy distance weight in `V23_WEIGHTS.magnetic`."
const V23_MAGNETIC_WEIGHT_FEATURES = (
    :bz0, :bz1, :bz2, :bz_mean6, :bz_sd6,
    :bperp0, :bperp_mean6, :vbs0, :vbs_mean3, :south_run,
)

"""
    V23_WEIGHTS

The two preregistered distance-weight sets for the V2.3 analog search, in `V23_FEATURE_NAMES`
order. `uniform` weights every standardised feature equally; `magnetic` doubles the weight of the
southward-field, transverse-field, coupling, and run-length features, which carry the storm-driving
information the tail continuation has to reproduce. Use [`v23_weights`](@ref) to obtain a private
copy; the stored vectors are shared state and must not be mutated.
"""
const V23_WEIGHTS = (
    uniform = ones(Float64, V23_FEATURE_COUNT),
    magnetic = Float64[
        name in V23_MAGNETIC_WEIGHT_FEATURES ? 2.0 : 1.0 for name in V23_FEATURE_NAMES
    ],
)

"""
    v23_weights(name::Symbol) -> Vector{Float64}

Return a fresh copy of the preregistered weight set `name` (`:uniform` or `:magnetic`).
"""
function v23_weights(name::Symbol)
    haskey(V23_WEIGHTS, name) || throw(ArgumentError(
        "unknown V2.3 weight set $(repr(name)); expected one of " *
        "$(join(repr.(keys(V23_WEIGHTS)), ", "))",
    ))
    return copy(getproperty(V23_WEIGHTS, name))
end

"""
    v23_feature_index(name::Symbol) -> Int

Column index of feature `name` in a V2.3 feature matrix.
"""
function v23_feature_index(name::Symbol)
    idx = findfirst(==(name), V23_FEATURE_NAMES)
    idx === nothing && throw(ArgumentError(
        "unknown V2.3 feature $(repr(name)); expected one of " *
        "$(join(repr.(V23_FEATURE_NAMES), ", "))",
    ))
    return idx
end

const _V23_FRAME_COLUMNS = (:time_utc, :V, :Bz, :By, :n, :Pdyn, :Dst)

"Coerce a frame time column to `Vector{DateTime}` without reordering it."
function _v23_datetimes(values)
    isempty(values) && return DateTime[]
    first_value = first(values)
    first_value isa DateTime && return collect(DateTime, values)
    first_value isa AbstractString && return DateTime.(strip.(String.(values)))
    throw(ArgumentError(
        "frame times must be DateTime or ISO-8601 strings, got $(typeof(first_value))",
    ))
end

"""
    _v23_frame_columns(frame) -> NamedTuple

Extract and validate the hourly driver/Dst frame used by the V2.3 features. `frame` is any object
exposing the columns `time_utc, V, Bz, By, n, Pdyn, Dst` as properties (a `DataFrame` or a
`NamedTuple` of vectors). Returns the columns as plain `Vector`s plus a time → row index map; a
duplicated timestamp is rejected because it would make the lag lookups ambiguous.
"""
function _v23_frame_columns(frame)
    for column in _V23_FRAME_COLUMNS
        hasproperty(frame, column) || throw(ArgumentError(
            "hourly frame is missing the $(column) column; required columns are " *
            "$(join(String.(_V23_FRAME_COLUMNS), ", "))",
        ))
    end
    times = _v23_datetimes(getproperty(frame, :time_utc))
    columns = map(_V23_FRAME_COLUMNS[2:end]) do column
        values = getproperty(frame, column)
        length(values) == length(times) || throw(ArgumentError(
            "hourly frame column $(column) has $(length(values)) rows but the time column has " *
            "$(length(times))",
        ))
        Float64[v === missing ? NaN : Float64(v) for v in values]
    end
    index = Dict{DateTime,Int}()
    sizehint!(index, length(times))
    for (row, t) in enumerate(times)
        haskey(index, t) && throw(ArgumentError("hourly frame repeats timestamp $t"))
        index[t] = row
    end
    return (
        times = times, V = columns[1], Bz = columns[2], By = columns[3],
        n = columns[4], Pdyn = columns[5], Dst = columns[6], index = index,
    )
end

"A driver record is usable when every component is finite and the density is strictly positive."
@inline function _v23_driver_usable(f, row::Int)
    return isfinite(f.V[row]) && isfinite(f.Bz[row]) && isfinite(f.By[row]) &&
           isfinite(f.n[row]) && isfinite(f.Pdyn[row]) && f.n[row] > 0.0
end

"Rectified coupling proxy VBs = V·max(0, −Bz)/1000 in mV/m."
@inline _v23_rectified_vbs(V::Float64, Bz::Float64) = V * max(0.0, -Bz) / 1000.0

"""
    v23_feature_matrix(frame, issue_times) -> (X::Matrix{Float64}, ok::BitVector)

Build the 18-column V2.3 issue-time feature matrix for `issue_times` from the hourly `frame`
(columns `time_utc, V, Bz, By, n, Pdyn, Dst`). Row `i` of `X` holds the features for
`issue_times[i]` in [`V23_FEATURE_NAMES`](@ref) order; `ok[i]` is `true` only when the issue has a
complete causal history, meaning records `t−1 … t−7` are present with finite, positive-density
driver values and `Dst(t)`, `Dst(t−1)` are finite. Rows with `ok[i] = false` are filled with `NaN`
so that a downstream consumer cannot silently treat an incomplete window as data.

All features are strictly causal: every driver value comes from a record tagged at or before
`t−1`, and the only issue-hour information used is the issue-time Dst, which is known at `t` by the
V2.1 anchor convention.
"""
function v23_feature_matrix(frame, issue_times)
    f = _v23_frame_columns(frame)
    queries = _v23_datetimes(issue_times)
    nq = length(queries)
    X = fill(NaN, nq, V23_FEATURE_COUNT)
    ok = falses(nq)
    lag_rows = Vector{Int}(undef, V23_HISTORY_LAGS_H)
    for (q, t) in enumerate(queries)
        issue_row = get(f.index, t, 0)
        issue_row == 0 && continue
        isfinite(f.Dst[issue_row]) || continue
        complete = true
        for lag in 1:V23_HISTORY_LAGS_H
            row = get(f.index, t - Hour(lag), 0)
            if row == 0 || !_v23_driver_usable(f, row)
                complete = false
                break
            end
            lag_rows[lag] = row
        end
        complete || continue
        isfinite(f.Dst[lag_rows[1]]) || continue

        r1 = lag_rows[1]
        bz_sum = 0.0
        bperp_sum = 0.0
        logn_sum = 0.0
        for lag in 1:6
            row = lag_rows[lag]
            bz_sum += f.Bz[row]
            bperp_sum += hypot(f.By[row], f.Bz[row])
            logn_sum += log(f.n[row])
        end
        bz_mean6 = bz_sum / 6
        variance_sum = 0.0
        for lag in 1:6
            variance_sum += abs2(f.Bz[lag_rows[lag]] - bz_mean6)
        end
        # Population (uncorrected) standard deviation: the six records are the whole window being
        # described, not a sample drawn from a larger one.
        bz_sd6 = sqrt(variance_sum / 6)
        vbs_sum3 = 0.0
        for lag in 1:3
            row = lag_rows[lag]
            vbs_sum3 += _v23_rectified_vbs(f.V[row], f.Bz[row])
        end
        south_run = 0
        for lag in 1:V23_SOUTH_RUN_CAP_H
            row = get(f.index, t - Hour(lag), 0)
            (row != 0 && isfinite(f.Bz[row]) && f.Bz[row] < 0.0) || break
            south_run += 1
        end

        X[q, 1] = f.Bz[r1]
        X[q, 2] = f.Bz[lag_rows[2]]
        X[q, 3] = f.Bz[lag_rows[3]]
        X[q, 4] = bz_mean6
        X[q, 5] = bz_sd6
        X[q, 6] = f.By[r1]
        X[q, 7] = hypot(f.By[r1], f.Bz[r1])
        X[q, 8] = bperp_sum / 6
        X[q, 9] = f.V[r1]
        X[q, 10] = f.V[r1] - f.V[lag_rows[7]]
        X[q, 11] = log(f.n[r1])
        X[q, 12] = logn_sum / 6
        X[q, 13] = f.Pdyn[r1]
        X[q, 14] = _v23_rectified_vbs(f.V[r1], f.Bz[r1])
        X[q, 15] = vbs_sum3 / 3
        X[q, 16] = Float64(south_run)
        X[q, 17] = f.Dst[issue_row]
        X[q, 18] = f.Dst[issue_row] - f.Dst[lag_rows[1]]
        ok[q] = true
    end
    return X, ok
end

"""
    v23_feature_stats(X; sd_floor = 1e-9) -> (mean, sd)

Column means and standard deviations of the finite rows of the V2.3 feature matrix `X`, used to
standardise features before the analog distance is evaluated. `sd` is the corrected (sample)
standard deviation, floored at `sd_floor` so that a feature that is constant on the fitting
partition cannot inflate the standardised distance.
"""
function v23_feature_stats(X::AbstractMatrix{<:Real}; sd_floor::Real = 1e-9)
    size(X, 2) == V23_FEATURE_COUNT || throw(DimensionMismatch(
        "V2.3 feature matrix must have $(V23_FEATURE_COUNT) columns, got $(size(X, 2))",
    ))
    isfinite(sd_floor) && sd_floor > 0 || throw(ArgumentError("sd_floor must be finite and positive"))
    rows = [i for i in axes(X, 1) if all(isfinite, view(X, i, :))]
    length(rows) >= 2 || throw(ArgumentError(
        "V2.3 feature standardisation needs at least two complete rows, got $(length(rows))",
    ))
    finite = Float64.(view(X, rows, :))
    μ = vec(mean(finite; dims = 1))
    σ = max.(vec(std(finite; dims = 1)), Float64(sd_floor))
    return (mean = μ, sd = σ)
end

"""
    v23_standardize(X, μ, σ) -> Matrix{Float64}

Standardise the V2.3 feature matrix `X` with the column means `μ` and standard deviations `σ`
produced by [`v23_feature_stats`](@ref). Incomplete rows stay non-finite, so they remain
recognisable after standardisation.
"""
function v23_standardize(X::AbstractMatrix{<:Real}, μ::AbstractVector{<:Real},
                         σ::AbstractVector{<:Real})
    size(X, 2) == V23_FEATURE_COUNT || throw(DimensionMismatch(
        "V2.3 feature matrix must have $(V23_FEATURE_COUNT) columns, got $(size(X, 2))",
    ))
    length(μ) == V23_FEATURE_COUNT || throw(DimensionMismatch(
        "standardisation mean must have $(V23_FEATURE_COUNT) entries, got $(length(μ))",
    ))
    length(σ) == V23_FEATURE_COUNT || throw(DimensionMismatch(
        "standardisation sd must have $(V23_FEATURE_COUNT) entries, got $(length(σ))",
    ))
    all(isfinite, μ) || throw(ArgumentError("standardisation mean must be finite"))
    all(s -> isfinite(s) && s > 0, σ) || throw(ArgumentError(
        "standardisation sd must be finite and positive",
    ))
    return (Float64.(X) .- transpose(Float64.(μ))) ./ transpose(Float64.(σ))
end
