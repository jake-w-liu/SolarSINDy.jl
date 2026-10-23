module DBDTCausalForecaster

using Dates, Statistics, LinearAlgebra, Random, JSON3

export ground_features, ground_cases, feature_matrix, fit_bins, apply_bins,
       fit_tree, predict_tree, fit_boost, predict_boost, fit_forest, predict_forest,
       fit_ridge, predict_ridge, pinball_loss, issue_features, export_model,
       load_model, predict_model, forecast_ground

const FEATURE_NAMES = ["log_now", "log_mean30", "log_max30", "log_std30",
    "now_over_max", "mean_over_max", "std_over_mean", "utc_day_sin",
    "utc_day_cos", "utc_year_sin", "utc_year_cos"]

function issue_features(mag::AbstractDict, t::DateTime)
    history = Float64[]
    for k in 29:-1:0
        current = t - Minute(k); previous = current - Minute(1)
        haskey(mag, current) && haskey(mag, previous) ||
            throw(ArgumentError("incomplete causal ground history"))
        before, after = mag[previous], mag[current]
        length(before) == length(after) == 2 ||
            throw(ArgumentError("each ground vector must have X and Y"))
        all(isfinite, before) && all(isfinite, after) ||
            throw(ArgumentError("ground vectors must be finite"))
        push!(history, hypot(after[1] - before[1], after[2] - before[2]))
    end
    return ground_features(t, history)
end

function ground_features(t::DateTime, history::AbstractVector{<:Real})
    length(history) == 30 || throw(ArgumentError("exactly 30 rates are required"))
    Dates.value(t - DateTime(1970)) % 60_000 == 0 ||
        throw(ArgumentError("issue time must be a whole UTC minute"))
    all(x -> isfinite(x) && x >= 0, history) ||
        throw(ArgumentError("rates must be finite and nonnegative"))
    now = Float64(last(history)); avg = mean(history)
    peak = maximum(history); deviation = std(history)
    daily = 2pi * (60hour(t) + minute(t)) / 1440
    seasonal = 2pi * (dayofyear(t) - 1) / daysinyear(t)
    return [log1p(now), log1p(avg), log1p(peak), log1p(deviation),
            now / (1 + peak), avg / (1 + peak), deviation / (1 + avg),
            sin(daily), cos(daily), sin(seasonal), cos(seasonal)]
end

"""Construct ground-only cases; histories end at issue time and targets start t+1."""
function ground_cases(mag::AbstractDict; horizon::Int=30)
    horizon == 30 || throw(ArgumentError("the frozen forecast horizon is 30 minutes"))
    rates = Dict{DateTime,Float64}()
    for t in sort!(collect(keys(mag)))
        Dates.value(t - DateTime(1970)) % 60_000 == 0 ||
            throw(ArgumentError("ground source contains an irregular UTC timestamp"))
        previous = t - Minute(1)
        haskey(mag, previous) || continue
        before, after = mag[previous], mag[t]
        all(isfinite, before) && all(isfinite, after) || continue
        rate = hypot(after[1] - before[1], after[2] - before[2])
        isfinite(rate) || continue
        rates[t] = rate
    end
    times = DateTime[]; targets = Float64[]; features = Vector{Float64}[]
    for t in sort!(collect(keys(rates)))
        history = [get(rates, t - Minute(k), NaN) for k in 29:-1:0]
        future = [get(rates, t + Minute(k), NaN) for k in 1:horizon]
        all(isfinite, history) && all(isfinite, future) || continue
        push!(times, t); push!(targets, maximum(future))
        push!(features, ground_features(t, history))
    end
    matrix = isempty(features) ? zeros(0, length(FEATURE_NAMES)) :
             reduce(hcat, features)'
    return (; times, targets, features=Matrix(matrix))
end

function feature_matrix(times, now, avg, peak, deviation)
    n = length(times)
    all(length(v) == n for v in (now, avg, peak, deviation)) ||
        throw(DimensionMismatch("feature column lengths differ"))
    result = Matrix{Float64}(undef, n, length(FEATURE_NAMES))
    for i in 1:n
        t = times[i]
        Dates.value(t - DateTime(1970)) % 60_000 == 0 ||
            throw(ArgumentError("issue time must be a whole UTC minute"))
        all(x -> isfinite(x) && x >= 0, (now[i], avg[i], peak[i], deviation[i])) ||
            throw(ArgumentError("ground summaries must be finite and nonnegative"))
        daily = 2pi * (60hour(t) + minute(t)) / 1440
        seasonal = 2pi * (dayofyear(t) - 1) / daysinyear(t)
        result[i, :] = [log1p(now[i]), log1p(avg[i]), log1p(peak[i]),
            log1p(deviation[i]), now[i] / (1 + peak[i]), avg[i] / (1 + peak[i]),
            deviation[i] / (1 + avg[i]), sin(daily), cos(daily),
            sin(seasonal), cos(seasonal)]
    end
    return result
end

function check_matrix(X)
    ndims(X) == 2 && size(X, 1) > 0 && size(X, 2) > 0 ||
        throw(ArgumentError("features must be a nonempty matrix"))
    all(isfinite, X) || throw(ArgumentError("features must be finite"))
end

function fit_bins(X::AbstractMatrix; nbins::Int=64)
    check_matrix(X)
    2 <= nbins <= 255 || throw(ArgumentError("nbins must be in 2:255"))
    return [unique!(sort!(quantile(view(X, :, j), collect(1:nbins-1) ./ nbins)))
            for j in axes(X, 2)]
end

function apply_bins(X::AbstractMatrix, cuts)
    check_matrix(X)
    size(X, 2) == length(cuts) || throw(DimensionMismatch("cut-point dimension mismatch"))
    result = Matrix{UInt8}(undef, size(X))
    for j in axes(X, 2)
        issorted(cuts[j]) && length(cuts[j]) < 255 ||
            throw(ArgumentError("invalid bin cut points"))
        for i in axes(X, 1)
            result[i, j] = searchsortedfirst(cuts[j], X[i, j])
        end
    end
    return result
end

struct HistogramTree
    feature::Vector{Int}
    threshold::Vector{UInt8}
    left::Vector{Int}
    right::Vector{Int}
    value::Vector{Float64}
end

function fit_tree(B::AbstractMatrix{UInt8}, y::AbstractVector{<:Real};
                  maxdepth::Int=3, minleaf::Int=64,
                  rows=collect(axes(B, 1)), mtry::Int=size(B, 2),
                  rng=Random.MersenneTwister(20171001))
    size(B, 1) == length(y) || throw(DimensionMismatch("target length mismatch"))
    !isempty(rows) && all(i -> 1 <= i <= length(y), rows) ||
        throw(ArgumentError("invalid training row indices"))
    all(isfinite, y) || throw(ArgumentError("targets must be finite"))
    maxdepth >= 0 && minleaf > 0 && 1 <= mtry <= size(B, 2) ||
        throw(ArgumentError("invalid tree configuration"))
    all(>(0x00), B) || throw(ArgumentError("bins must be positive"))
    tree = HistogramTree(Int[], UInt8[], Int[], Int[], Float64[])
    function grow(indices, depth)
        node = length(tree.value) + 1
        # Center before reduction so a large shared offset cannot erase a small mean.
        anchor = Float64(y[first(indices)])
        center = anchor + mean(y[i] - anchor for i in indices)
        push!(tree.feature, 0); push!(tree.threshold, 0x00)
        push!(tree.left, 0); push!(tree.right, 0); push!(tree.value, center)
        (depth >= maxdepth || length(indices) < 2minleaf) && return node
        candidates = mtry == size(B, 2) ? collect(axes(B, 2)) :
                     sort!(randperm(rng, size(B, 2))[1:mtry])
        best_gain = 0.0; best_feature = 0; best_threshold = 0
        for feature in candidates
            counts = zeros(Int, 255); sums = zeros(255)
            total = 0.0
            for i in indices
                bin = B[i, feature]
                residual = y[i] - center
                counts[bin] += 1; sums[bin] += residual; total += residual
            end
            left_count = 0; left_sum = 0.0
            for threshold in 1:254
                left_count += counts[threshold]; left_sum += sums[threshold]
                right_count = length(indices) - left_count
                (left_count < minleaf || right_count < minleaf) && continue
                gain = left_sum^2 / left_count + (total - left_sum)^2 / right_count -
                       total^2 / length(indices)
                if gain > best_gain
                    best_gain = gain; best_feature = feature; best_threshold = threshold
                end
            end
        end
        best_feature == 0 && return node
        lower = [i for i in indices if B[i, best_feature] <= best_threshold]
        upper = [i for i in indices if B[i, best_feature] > best_threshold]
        tree.feature[node] = best_feature; tree.threshold[node] = best_threshold
        tree.left[node] = grow(lower, depth + 1)
        tree.right[node] = grow(upper, depth + 1)
        return node
    end
    grow(rows, 0)
    return tree
end

function predict_tree(tree::HistogramTree, B::AbstractMatrix{UInt8})
    maximum(tree.feature; init=0) <= size(B, 2) ||
        throw(DimensionMismatch("tree feature dimension mismatch"))
    return [begin
        node = 1
        while tree.feature[node] != 0
            node = B[i, tree.feature[node]] <= tree.threshold[node] ?
                   tree.left[node] : tree.right[node]
        end
        tree.value[node]
    end for i in axes(B, 1)]
end

struct BoostedModel
    intercept::Float64
    rate::Float64
    trees::Vector{HistogramTree}
end

function fit_boost(B, y; maxdepth=3, ntrees=100, rate=0.05, minleaf=64,
                   checkpoints=Int[], callback=(n, model) -> nothing)
    ntrees > 0 && isfinite(rate) && 0 < rate <= 1 ||
        throw(ArgumentError("invalid boosting configuration"))
    model = BoostedModel(mean(y), rate, HistogramTree[])
    prediction = fill(model.intercept, length(y))
    for k in 1:ntrees
        tree = fit_tree(B, y .- prediction; maxdepth, minleaf)
        push!(model.trees, tree)
        prediction .+= rate .* predict_tree(tree, B)
        k in checkpoints && callback(k, model)
    end
    return model
end

function predict_boost(model::BoostedModel, B)
    result = fill(model.intercept, size(B, 1))
    for tree in model.trees
        result .+= model.rate .* predict_tree(tree, B)
    end
    return result
end

struct ForestModel
    trees::Vector{HistogramTree}
end

function fit_forest(B, y; maxdepth=9, ntrees=64, sample_rows=20_000,
                    minleaf=32, seed=20171001)
    ntrees > 0 && sample_rows > 0 || throw(ArgumentError("invalid forest configuration"))
    rng = MersenneTwister(seed)
    trees = HistogramTree[]
    for _ in 1:ntrees
        rows = rand(rng, axes(B, 1), min(sample_rows, size(B, 1)))
        push!(trees, fit_tree(B, y; maxdepth, minleaf, rows,
                             mtry=max(1, isqrt(size(B, 2))), rng))
    end
    return ForestModel(trees)
end

function predict_forest(model::ForestModel, B)
    result = zeros(size(B, 1))
    for tree in model.trees
        result .+= predict_tree(tree, B)
    end
    return result ./ length(model.trees)
end

struct RidgeModel
    mean::Vector{Float64}
    scale::Vector{Float64}
    coefficients::Vector{Float64}
end

function fit_ridge(X, y; penalty=1.0)
    check_matrix(X)
    size(X, 1) == length(y) || throw(DimensionMismatch("target length mismatch"))
    all(isfinite, y) && isfinite(penalty) && penalty >= 0 ||
        throw(ArgumentError("invalid targets or ridge penalty"))
    center = vec(mean(X; dims=1)); scale = vec(std(X; dims=1))
    scale[.!isfinite.(scale) .| (scale .== 0)] .= 1
    Z = hcat(ones(size(X, 1)), (X .- center') ./ scale')
    regularizer = Diagonal(vcat(0.0, fill(Float64(penalty), size(X, 2))))
    coefficients = (Z' * Z + regularizer) \ (Z' * y)
    return RidgeModel(center, scale, coefficients)
end

function predict_ridge(model::RidgeModel, X)
    check_matrix(X)
    size(X, 2) == length(model.mean) || throw(DimensionMismatch("ridge dimension mismatch"))
    return hcat(ones(size(X, 1)), (X .- model.mean') ./ model.scale') *
           model.coefficients
end

function pinball_loss(y, upper; probability=0.9)
    length(y) == length(upper) > 0 || throw(DimensionMismatch("pinball arrays differ or are empty"))
    0 < probability < 1 || throw(ArgumentError("probability must be between zero and one"))
    return mean(eachindex(y)) do i
        delta = y[i] - upper[i]
        delta >= 0 ? probability * delta : (probability - 1) * delta
    end
end

tree_payload(tree) = Dict("feature" => tree.feature, "threshold" => Int.(tree.threshold),
    "left" => tree.left, "right" => tree.right, "value" => tree.value)

function restore_tree(payload)
    tree = HistogramTree(Int.(payload.feature), UInt8.(payload.threshold),
                         Int.(payload.left), Int.(payload.right), Float64.(payload.value))
    n = length(tree.value)
    n > 0 && all(length(v) == n for v in
        (tree.feature, tree.threshold, tree.left, tree.right)) && all(isfinite, tree.value) ||
        throw(ArgumentError("malformed serialized tree"))
    visited = falses(n)
    function visit(node)
        1 <= node <= n && !visited[node] || throw(ArgumentError("invalid tree edge or cycle"))
        visited[node] = true
        if tree.feature[node] == 0
            tree.left[node] == tree.right[node] == 0 ||
                throw(ArgumentError("leaf has child edges"))
        else
            1 <= tree.feature[node] <= length(FEATURE_NAMES) && tree.threshold[node] > 0 ||
                throw(ArgumentError("invalid tree split"))
            visit(tree.left[node]); visit(tree.right[node])
        end
    end
    visit(1)
    all(visited) || throw(ArgumentError("tree has unreachable nodes"))
    return tree
end

function export_model(path, kind, model, cuts, sorted_scores; station)
    kind in ("ridge", "boost", "forest") || throw(ArgumentError("unsupported model"))
    !isempty(sorted_scores) && issorted(sorted_scores) && all(isfinite, sorted_scores) ||
        throw(ArgumentError("invalid calibration scores"))
    payload = if kind == "ridge"
        Dict("mean" => model.mean, "scale" => model.scale,
             "coefficients" => model.coefficients)
    elseif kind == "boost"
        Dict("intercept" => model.intercept, "rate" => model.rate,
             "trees" => tree_payload.(model.trees))
    else
        Dict("trees" => tree_payload.(model.trees))
    end
    artifact = Dict("schema" => "dbdt_ground_only_v1", "station" => station,
        "ground_product" => "quasi-definitive", "units" => "nT/min",
        "target_horizon_minutes" => 30, "history_minutes" => 30,
        "feature_names" => FEATURE_NAMES, "kind" => kind, "model" => payload,
        "cuts" => cuts, "normalized_scores" => sorted_scores,
        "normalization" => "1+log1p(trailing30max)", "nominal_coverage" => 0.9,
        "fit_end_exclusive_utc" => "2017-10-01T00:00:00",
        "calibration_end_exclusive_utc" => "2018-01-01T00:00:00",
        "live_product_transfer_validated" => false)
    open(io -> JSON3.write(io, artifact), path, "w")
    return path
end

function load_model(path)
    artifact = JSON3.read(read(path, String))
    artifact.schema == "dbdt_ground_only_v1" && artifact.units == "nT/min" &&
        artifact.target_horizon_minutes == artifact.history_minutes == 30 &&
        String.(artifact.feature_names) == FEATURE_NAMES ||
        throw(ArgumentError("incompatible forecast artifact"))
    kind = String(artifact.kind)
    payload = artifact.model
    model = if kind == "ridge"
        center = Float64.(payload.mean); scale = Float64.(payload.scale)
        coefficients = Float64.(payload.coefficients)
        length(center) == length(scale) == length(FEATURE_NAMES) &&
            length(coefficients) == length(center) + 1 && all(isfinite, center) &&
            all(x -> isfinite(x) && x > 0, scale) && all(isfinite, coefficients) ||
            throw(ArgumentError("malformed ridge artifact"))
        RidgeModel(center, scale, coefficients)
    elseif kind == "boost"
        intercept = Float64(payload.intercept); rate = Float64(payload.rate)
        isfinite(intercept) && isfinite(rate) && 0 < rate <= 1 ||
            throw(ArgumentError("malformed boosting artifact"))
        trees = restore_tree.(payload.trees)
        isempty(trees) && throw(ArgumentError("empty boosted ensemble"))
        BoostedModel(intercept, rate, trees)
    elseif kind == "forest"
        trees = restore_tree.(payload.trees)
        isempty(trees) && throw(ArgumentError("empty forest"))
        ForestModel(trees)
    else
        throw(ArgumentError("unsupported serialized model"))
    end
    cuts = [Float64.(x) for x in artifact.cuts]
    length(cuts) == length(FEATURE_NAMES) && all(x -> issorted(x) &&
        all(isfinite, x) && length(x) < 255, cuts) ||
        throw(ArgumentError("invalid serialized cut points"))
    scores = Float64.(artifact.normalized_scores)
    !isempty(scores) && issorted(scores) && all(isfinite, scores) ||
        throw(ArgumentError("invalid serialized calibration scores"))
    return (; kind, model, cuts, scores, station=String(artifact.station),
            ground_product=String(artifact.ground_product))
end

function predict_model(bundle, X)
    return bundle.kind == "ridge" ? predict_ridge(bundle.model, X) :
        bundle.kind == "boost" ? predict_boost(bundle.model, apply_bins(X, bundle.cuts)) :
        bundle.kind == "forest" ? predict_forest(bundle.model, apply_bins(X, bundle.cuts)) :
        throw(ArgumentError("unsupported prediction model"))
end

function forecast_ground(bundle, mag, t; station, ground_product)
    String(station) == bundle.station && String(ground_product) == bundle.ground_product ||
        throw(ArgumentError("station/product differs from validated artifact"))
    features = issue_features(mag, t)
    center = only(predict_model(bundle, reshape(features, 1, :)))
    k = ceil(Int, (length(bundle.scores) + 1) * 0.9)
    quantile = k > length(bundle.scores) ? Inf : bundle.scores[k]
    upper = expm1(center + quantile * (1 + features[3]))
    return (; issue_time=t, target_start=t + Minute(1), target_end=t + Minute(30),
             point=max(0.0, expm1(center)), upper, station=bundle.station,
             ground_product=bundle.ground_product, units="nT/min")
end

end
