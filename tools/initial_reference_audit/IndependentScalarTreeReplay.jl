module IndependentScalarTreeReplay

# Data-only numerical oracle. No author code, model loader or ML dependency.
export Model, predict_delta
need(ok, message) = ok || throw(ArgumentError(message))
isint(x) = x isa Integer && !(x isa Bool)
isnumber(x) = x isa Real && !(x isa Bool) && isfinite(x)

struct Tree
    feature::Vector{Int}
    condition::Vector{Int}
    prediction::Vector{Float32}
    split::BitVector
end

struct Model
    edges::Vector{Vector{Float64}}
    trees::Vector{Tree}
    function Model(raw_edges, raw_trees; max_depth, nrounds)
        need(isint(max_depth) && 1 <= max_depth <= 6, "maximum depth outside 1:6")
        need(isint(nrounds) && 0 <= nrounds <= 400, "round count outside 0:400")
        need(length(raw_edges) == 29, "exactly 29 feature edge vectors required")
        need(length(raw_trees) == nrounds + 1, "bias plus exact round count required")
        edges = Vector{Float64}[]
        bins = nothing
        for values in raw_edges
            need(1 <= length(values) <= 254 && all(isnumber, values), "edge count/finite values")
            edge = Float64.(values)
            need(all(isfinite, edge) && issorted(edge), "finite ordered Float64 edges required")
            bins === nothing && (bins = length(edge) + 1)
            need(length(edge) + 1 == bins, "common bin count required")
            push!(edges, edge)
        end
        trees = Tree[]
        for (index, raw) in enumerate(raw_trees)
            need(Set(propertynames(raw)) == Set((:feature, :condition, :prediction, :split)),
                 "exact plain tree field set required")
            nodes = index == 1 ? 1 : 2^max_depth - 1
            need(all(length(getproperty(raw, key)) == nodes for key in propertynames(raw)),
                 "tree heap dimensions differ")
            need(all(x -> isint(x) && 0 <= x <= 29, raw.feature), "stored feature index")
            need(all(x -> isint(x) && 0 <= x < bins, raw.condition), "stored condition bin")
            need(all(x -> x isa Bool, raw.split), "Boolean split flags required")
            need(all(isnumber, raw.prediction), "finite scalar leaf values required")
            prediction = Float32.(raw.prediction)
            need(all(isfinite, prediction) && all(a == b for (a, b) in zip(prediction, raw.prediction)),
                 "leaf values must represent exact finite Float32 data")
            feature, condition, split = Int.(raw.feature), Int.(raw.condition), BitVector(raw.split)
            reached = falses(nodes)
            pending = [1]
            while !isempty(pending)
                node = pop!(pending)
                reached[node] = true
                if split[node]
                    need(index != 1, "bias tree cannot split")
                    need(1 <= feature[node] <= 29 && 1 <= condition[node] < bins,
                         "reachable numeric split index/bin")
                    need(2node + 1 <= nodes, "split children exceed heap")
                    push!(pending, 2node, 2node + 1)
                end
            end
            need(all(reached[node] || !split[node] for node in 1:nodes), "unreachable split node")
            push!(trees, Tree(feature, condition, prediction, split))
        end
        new(edges, trees)
    end
end

function predict_delta(model::Model, values::AbstractVector{Float64})
    need(length(values) == 29 && all(isfinite, values), "29 finite Float64 values required")
    bins = [searchsortedfirst(edge, value) for (edge, value) in zip(model.edges, values)]
    total = 0.0f0
    for tree in model.trees
        node = 1
        while tree.split[node]
            node = bins[tree.feature[node]] <= tree.condition[node] ? 2node : 2node + 1
        end
        total = Float32(total + tree.prediction[node])
        need(isfinite(total), "nonfinite accumulated tree prediction")
    end
    Float64(total)
end

end
