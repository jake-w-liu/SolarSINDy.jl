"""Retained numerical algorithms for the original retrospective baseline families."""
module RetainedClassicTraining
using Statistics,LinearAlgebra,Random
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


end
