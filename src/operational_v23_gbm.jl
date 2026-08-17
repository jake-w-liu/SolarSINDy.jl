# Boosted-model primitives for the Operational V2.3 driver-continuation study.
#
# Three responsibilities, all issue-time causal:
#   * a fully pinned EvoTrees wrapper (fit / predict / save / load) shared by the
#     GDC driver models (plan section 4, T2), the direct-GBM comparator (plan
#     section 5) and the E2 residual layer (plan section 4);
#   * the direct-GBM feature block, which extends the 18 ADC features with the
#     Dst and VBs lag ladder required by plan section 5;
#   * the GDC regression targets: driver variables at record t+k-1 expressed as
#     increments relative to record t-1.
#
# Hourly-record conventions (OMNI record tagged tau covers [tau, tau+1)):
#   * issue t: the latest known Dst is the record tagged t, while the latest
#     known driver record is t-1;
#   * ADC feature lag j (j = 0..6) is driver record t-1-j, so VBs lag L is
#     record t-1-L;
#   * Dst lag j is the record tagged t-j;
#   * future(k) is record t+k-1, which is what the GDC models regress on.

import EvoTrees

"EvoTrees release pinned for every V2.3 boosted artifact."
const V23_GBM_EVOTREES_VERSION = "0.18.7"

"Preregistered seed for every V2.3 boosted fit (plan section 4)."
const V23_GBM_DEFAULT_SEED = 22022026

"Dst lag hours appended to the ADC block for the direct-GBM comparator."
const V23_DIRECT_DST_LAG_HOURS = (1, 2, 3, 4, 5, 6, 12, 24)

"VBs lag steps appended to the ADC block; lag L is driver record t-1-L."
const V23_DIRECT_VBS_LAG_STEPS = (1, 2, 3)

"Column names of the direct-GBM extra block, in matrix order."
const V23_DIRECT_EXTRA_FEATURE_NAMES = (
    "dst_lag1", "dst_lag2", "dst_lag3", "dst_lag4", "dst_lag5", "dst_lag6",
    "dst_lag12", "dst_lag24", "vbs_lag1", "vbs_lag2", "vbs_lag3",
)

"Total direct-GBM feature count: 18 ADC features plus the 11 extra lags."
const V23_DIRECT_FEATURE_COUNT = 29

const _V23_FRAME_COLUMNS = (:time_utc, :V, :Bz, :By, :n, :Pdyn, :Dst)

"""
    _v23_frame_index(frame) -> Dict{DateTime,Int}

Map each hourly record tag to its row. Duplicate tags are rejected because a
duplicate would make the lag lookup order-dependent and therefore
non-reproducible.
"""
function _v23_frame_index(frame)
    for column in _V23_FRAME_COLUMNS
        hasproperty(frame, column) || throw(ArgumentError(
            "hourly frame is missing column :$column",
        ))
    end
    times = getproperty(frame, :time_utc)
    index = Dict{DateTime,Int}()
    sizehint!(index, length(times))
    for i in eachindex(times)
        tag = times[i]
        tag isa DateTime || throw(ArgumentError(
            "hourly frame time_utc must hold DateTime values",
        ))
        haskey(index, tag) && throw(ArgumentError(
            "hourly frame has a duplicate time_utc entry at $tag",
        ))
        index[tag] = i
    end
    return index
end

"Finite value of `column` at record tag `tag`, or NaN when absent/missing/non-finite."
@inline function _v23_record_value(column, index::Dict{DateTime,Int}, tag::DateTime)
    row = get(index, tag, 0)
    row == 0 && return NaN
    value = column[row]
    (value isa Real && !(value isa Bool)) || return NaN
    numeric = Float64(value)
    return isfinite(numeric) ? numeric : NaN
end

"Southward coupling proxy VBs = V * max(0, -Bz) / 1000 in mV/m."
@inline function _v23_vbs(speed::Float64, bz::Float64)
    (isnan(speed) || isnan(bz)) && return NaN
    return speed * max(0.0, -bz) / 1000.0
end

"""
    v23_direct_extra_features(frame, issue_times) -> (Xextra, ok)

Build the direct-GBM extra block: Dst at records t-1..t-6, t-12, t-24 followed
by VBs at driver records t-2, t-3, t-4 (VBs lags 1..3). `frame` is the hourly
driver/Dst frame (`time_utc`, `V`, `Bz`, `By`, `n`, `Pdyn`, `Dst`; NaN marks a
missing value).

Every column uses records strictly earlier than the issue hour, so the block is
invariant to any frame value at or after the issue time. Rows whose required
records are absent or non-finite are returned as all-NaN with `ok = false`.
"""
function v23_direct_extra_features(frame, issue_times::AbstractVector{DateTime})
    index = _v23_frame_index(frame)
    dst_column = getproperty(frame, :Dst)
    speed_column = getproperty(frame, :V)
    bz_column = getproperty(frame, :Bz)
    rows = length(issue_times)
    cols = length(V23_DIRECT_EXTRA_FEATURE_NAMES)
    X = fill(NaN, rows, cols)
    ok = falses(rows)
    buffer = Vector{Float64}(undef, cols)
    for i in 1:rows
        issue = issue_times[i]
        complete = true
        slot = 0
        for lag in V23_DIRECT_DST_LAG_HOURS
            slot += 1
            value = _v23_record_value(dst_column, index, issue - Hour(lag))
            buffer[slot] = value
            complete &= !isnan(value)
        end
        for lag in V23_DIRECT_VBS_LAG_STEPS
            slot += 1
            record = issue - Hour(1 + lag)
            value = _v23_vbs(
                _v23_record_value(speed_column, index, record),
                _v23_record_value(bz_column, index, record),
            )
            buffer[slot] = value
            complete &= !isnan(value)
        end
        complete || continue
        ok[i] = true
        for j in 1:cols
            X[i, j] = buffer[j]
        end
    end
    return X, ok
end

"""
    v23_direct_feature_names() -> Vector

Ordered names of the 29 direct-GBM columns: the ADC schema `V23_FEATURE_NAMES`
followed by `V23_DIRECT_EXTRA_FEATURE_NAMES`, converted to the element type used
by the ADC schema.
"""
function v23_direct_feature_names()
    isdefined(@__MODULE__, :V23_FEATURE_NAMES) || throw(ArgumentError(
        "v23_direct_feature_names requires V23_FEATURE_NAMES from operational_v23_features.jl",
    ))
    base = collect(V23_FEATURE_NAMES)
    names = eltype(base) <: Symbol ?
        vcat(base, Symbol.(collect(V23_DIRECT_EXTRA_FEATURE_NAMES))) :
        vcat(String.(base), String.(collect(V23_DIRECT_EXTRA_FEATURE_NAMES)))
    return names
end

"""
    v23_direct_features(frame, issue_times) -> (X, ok)

Direct-GBM design matrix: the 18 ADC features of `v23_feature_matrix` followed by
the 11 columns of `v23_direct_extra_features`, 29 columns in total. A row is
usable only when both blocks are complete; unusable rows are all-NaN.
"""
function v23_direct_features(frame, issue_times::AbstractVector{DateTime})
    isdefined(@__MODULE__, :v23_feature_matrix) || throw(ArgumentError(
        "v23_direct_features requires v23_feature_matrix from operational_v23_features.jl",
    ))
    adc, adc_ok = v23_feature_matrix(frame, issue_times)
    extra, extra_ok = v23_direct_extra_features(frame, issue_times)
    size(adc, 1) == size(extra, 1) || throw(DimensionMismatch(
        "ADC and extra feature blocks disagree on row count",
    ))
    X = hcat(Matrix{Float64}(adc), extra)
    size(X, 2) == V23_DIRECT_FEATURE_COUNT || throw(ArgumentError(
        "direct-GBM design matrix must have $V23_DIRECT_FEATURE_COUNT columns, " *
        "found $(size(X, 2))",
    ))
    ok = BitVector(undef, length(adc_ok))
    for i in eachindex(ok)
        ok[i] = adc_ok[i] & extra_ok[i]
        ok[i] && continue
        for j in axes(X, 2)
            X[i, j] = NaN
        end
    end
    return X, ok
end

"""
    v23_gdc_targets(frame, issue_times, k) -> (bz, by, dlogv, dlogn)

Boosted driver-continuation targets for model step `k`: `Bz` and `By` at record
t+k-1, and the log increments of `V` and `n` from record t-1 to record t+k-1.
Entries whose records are absent, non-finite, or non-positive where a logarithm
is required are returned as NaN; finiteness is the row-usability flag.
"""
function v23_gdc_targets(frame, issue_times::AbstractVector{DateTime}, k::Integer)
    Int(k) >= 1 || throw(ArgumentError("GDC target step k must be at least 1"))
    index = _v23_frame_index(frame)
    speed_column = getproperty(frame, :V)
    bz_column = getproperty(frame, :Bz)
    by_column = getproperty(frame, :By)
    density_column = getproperty(frame, :n)
    rows = length(issue_times)
    bz = fill(NaN, rows)
    by = fill(NaN, rows)
    dlogv = fill(NaN, rows)
    dlogn = fill(NaN, rows)
    offset = Hour(Int(k) - 1)
    for i in 1:rows
        issue = issue_times[i]
        base = issue - Hour(1)
        target = issue + offset
        bz[i] = _v23_record_value(bz_column, index, target)
        by[i] = _v23_record_value(by_column, index, target)
        speed_base = _v23_record_value(speed_column, index, base)
        speed_target = _v23_record_value(speed_column, index, target)
        density_base = _v23_record_value(density_column, index, base)
        density_target = _v23_record_value(density_column, index, target)
        if speed_base > 0 && speed_target > 0
            dlogv[i] = log(speed_target) - log(speed_base)
        end
        if density_base > 0 && density_target > 0
            dlogn[i] = log(density_target) - log(density_base)
        end
    end
    return (bz, by, dlogv, dlogn)
end

function _v23_gbm_validate_matrix(X::AbstractMatrix)
    size(X, 1) >= 2 || throw(ArgumentError(
        "a V2.3 boosted fit requires at least two training rows",
    ))
    size(X, 2) >= 1 || throw(ArgumentError(
        "a V2.3 boosted fit requires at least one feature column",
    ))
    matrix = Matrix{Float64}(X)
    all(isfinite, matrix) || throw(ArgumentError(
        "V2.3 boosted feature matrices must be finite",
    ))
    return matrix
end

"""
    v23_fit_gbm(X, y; max_depth, nrounds, eta=0.05, min_weight=64,
                seed=$(V23_GBM_DEFAULT_SEED), nbins=64, lambda=1.0, ...)

Fit one deterministic CPU EvoTrees squared-error regressor. Every stochastic
knob is pinned (`bagging_size=1`, `rowsample=1`, `colsample=1`, explicit seed),
so repeated fits on identical inputs are bit-identical. `L2`, `gamma`,
`rowsample` and `colsample` are exposed only so the configuration is fully
recorded; their defaults are the EvoTrees $(V23_GBM_EVOTREES_VERSION) defaults.
"""
function v23_fit_gbm(
        X::AbstractMatrix,
        y::AbstractVector;
        max_depth::Integer,
        nrounds::Integer,
        eta::Real=0.05,
        min_weight::Real=64,
        seed::Integer=V23_GBM_DEFAULT_SEED,
        nbins::Integer=64,
        lambda::Real=1.0,
        L2::Real=1.0,
        gamma::Real=0.0,
        rowsample::Real=1.0,
        colsample::Real=1.0,
        feature_names=nothing)
    version = string(Base.pkgversion(EvoTrees))
    version == V23_GBM_EVOTREES_VERSION || throw(ArgumentError(
        "V2.3 boosted models require EvoTrees $V23_GBM_EVOTREES_VERSION, found $version",
    ))
    matrix = _v23_gbm_validate_matrix(X)
    length(y) == size(matrix, 1) || throw(DimensionMismatch(
        "V2.3 boosted target length must equal the training row count",
    ))
    target = Vector{Float64}(y)
    all(isfinite, target) || throw(ArgumentError(
        "V2.3 boosted targets must be finite",
    ))
    1 <= Int(max_depth) <= 12 || throw(ArgumentError(
        "V2.3 boosted max_depth must be between 1 and 12",
    ))
    1 <= Int(nrounds) <= 2048 || throw(ArgumentError(
        "V2.3 boosted nrounds must be between 1 and 2048",
    ))
    2 <= Int(nbins) <= 255 || throw(ArgumentError(
        "V2.3 boosted nbins must be between 2 and 255",
    ))
    Int(seed) >= 0 || throw(ArgumentError("V2.3 boosted seed must be nonnegative"))
    Float64(eta) > 0 || throw(ArgumentError("V2.3 boosted eta must be positive"))
    Float64(min_weight) >= 0 || throw(ArgumentError(
        "V2.3 boosted min_weight must be nonnegative",
    ))
    Float64(lambda) >= 0 || throw(ArgumentError("V2.3 boosted lambda must be nonnegative"))
    Float64(L2) >= 0 || throw(ArgumentError("V2.3 boosted L2 must be nonnegative"))
    0 < Float64(rowsample) <= 1 || throw(ArgumentError(
        "V2.3 boosted rowsample must lie in (0, 1]",
    ))
    0 < Float64(colsample) <= 1 || throw(ArgumentError(
        "V2.3 boosted colsample must lie in (0, 1]",
    ))
    names = feature_names === nothing ?
        ["x$(j)" for j in 1:size(matrix, 2)] : collect(String.(feature_names))
    length(names) == size(matrix, 2) || throw(DimensionMismatch(
        "V2.3 boosted feature names must match the training column count",
    ))
    allunique(names) || throw(ArgumentError(
        "V2.3 boosted feature names must be unique",
    ))
    config = EvoTrees.EvoTreeRegressor(
        loss=:mse, metric=:rmse, nrounds=Int(nrounds), bagging_size=1,
        early_stopping_rounds=typemax(Int), L2=Float64(L2),
        lambda=Float64(lambda), gamma=Float64(gamma), eta=Float64(eta),
        max_depth=Int(max_depth), min_weight=Float64(min_weight),
        rowsample=Float64(rowsample), colsample=Float64(colsample),
        nbins=Int(nbins), seed=Int(seed), tree_type=:binary, device=:cpu,
    )
    return EvoTrees.fit(
        config; x_train=matrix, y_train=target, feature_names=names, verbosity=0,
    )
end

"""
    v23_predict(model, X) -> Vector{Float64}

Score a fitted V2.3 boosted regressor. Columns are matched positionally against
the frozen training schema; EvoTrees accumulates in `Float32`, and the result is
widened exactly to `Float64`.
"""
function v23_predict(model, X::AbstractMatrix)
    model isa EvoTrees.EvoTree{EvoTrees.MSE,1} || throw(ArgumentError(
        "v23_predict expects a scalar EvoTrees squared-error regressor",
    ))
    expected = length(model.info[:feature_names])
    size(X, 2) == expected || throw(DimensionMismatch(
        "v23_predict expects $expected feature columns, received $(size(X, 2))",
    ))
    matrix = Matrix{Float64}(X)
    all(isfinite, matrix) || throw(ArgumentError(
        "v23_predict feature matrices must be finite",
    ))
    return Float64.(EvoTrees.predict(model, matrix))
end

"""
    v23_save(model, path) -> path

Persist a fitted V2.3 boosted regressor through the EvoTrees BSON writer.
"""
function v23_save(model, path::AbstractString)
    model isa EvoTrees.EvoTree || throw(ArgumentError(
        "v23_save expects a fitted EvoTrees model",
    ))
    islink(path) && throw(ArgumentError(
        "refusing to write a V2.3 boosted model through a symlink: $path",
    ))
    directory = dirname(abspath(path))
    isdir(directory) || mkpath(directory)
    EvoTrees.save(model, path)
    return path
end

"""
    v23_load(path) -> model

Read a V2.3 boosted regressor written by [`v23_save`](@ref).
"""
function v23_load(path::AbstractString)
    islink(path) && throw(ArgumentError(
        "refusing to read a V2.3 boosted model through a symlink: $path",
    ))
    isfile(path) || throw(ArgumentError(
        "V2.3 boosted model file not found: $path",
    ))
    return EvoTrees.load(path)
end
