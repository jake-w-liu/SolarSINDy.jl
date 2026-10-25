"""Independent scalar numeric replay. No author-code or model-loader dependency."""
module MatchedScalarTreeReplay342
export Tree,Model,predict_delta,predict_deltas
need(ok,message)=ok||throw(ArgumentError(message))
isint(x)=x isa Integer&&!(x isa Bool)
isnumber(x)=x isa Real&&!(x isa Bool)&&isfinite(x)

# Wrapper types never depend on edge count, depth, rounds or source length.
# Abstract-length typed tuples retain immutable data without payload-type escape.
"Immutable finite scalar Float32 tree with exact feature/condition/prediction/split fields, bounded to63 heap nodes."
struct Tree
    feature::Tuple{Vararg{Int}}
    condition::Tuple{Vararg{Int}}
    prediction::Tuple{Vararg{Float32}}
    split::Tuple{Vararg{Bool}}
    function Tree(feature,condition,prediction,split)
        n=length(feature);need(1<=n<=63&&length(condition)==length(prediction)==length(split)==n,"bounded matching tree fields required")
        need(all(x->isint(x)&&0<=x<=342,feature),"stored feature index")
        need(all(x->isint(x)&&0<=x<=254,condition),"stored condition index")
        need(all(x->x isa Bool,split)&&all(isnumber,prediction),"tree flag/finite leaf types differ")
        leaves=Float32.(prediction)
        need(all(isfinite,leaves)&&all(a==b for (a,b) in zip(leaves,prediction)),"exact finite Float32 leaves required")
        new(Tuple(Int.(feature)),Tuple(Int.(condition)),Tuple(leaves),Tuple(split))
    end
end
"Immutable342-feature numeric replay model; depth1:6, rounds0:400, common2:255 bins, bias plus exact ordered trees."
struct Model
    edges::Tuple{Vararg{Tuple{Vararg{Float64}}}}
    trees::Tuple{Vararg{Tree}}
    max_depth::Int
    nrounds::Int
    nbins::Int
    function Model(raw_edges,raw_trees;max_depth,nrounds)
        need(isint(max_depth)&&1<=max_depth<=6,"depth outside1:6")
        need(isint(nrounds)&&0<=nrounds<=400,"rounds outside0:400")
        need(length(raw_edges)==342&&length(raw_trees)==nrounds+1,"342edges and bias+exactrounds required")
        edges=Tuple{Vararg{Float64}}[];bins=0
        for values in raw_edges
            need(1<=length(values)<=254&&all(isnumber,values),"finite bounded edges required")
            edge=Float64.(values);need(all(isfinite,edge)&&issorted(edge),"ordered finite Float64 edges required")
            bins==0&&(bins=length(edge)+1)
            need(length(edge)+1==bins,"common numeric bin count required")
            push!(edges,Tuple(edge))
        end
        trees=Tree[]
        for (i,raw) in enumerate(raw_trees)
            need((raw isa NamedTuple||raw isa Tree)&&Set(propertynames(raw))==Set((:feature,:condition,:prediction,:split)),"plain exact four-field tree required")
            nodes=i==1 ? 1 : 2^Int(max_depth)-1
            need(all(length(getproperty(raw,key))==nodes for key in propertynames(raw)),"tree heap geometry differs")
            tree=Tree(raw.feature,raw.condition,raw.prediction,raw.split)
            need(all(c->c<bins,tree.condition),"condition exceeds numeric bins")
            reached=falses(nodes);pending=Int[1]
            while !isempty(pending)
                node=pop!(pending);reached[node]=true
                if tree.split[node]
                    need(i!=1&&1<=tree.feature[node]<=342&&1<=tree.condition[node]<bins,"reachable numeric split differs")
                    need(2node+1<=nodes,"split children exceed heap")
                    push!(pending,2node,2node+1)
                end
            end
            need(all(reached[j]||!tree.split[j] for j in 1:nodes),"unreachable split node")
            push!(trees,tree)
        end
        new(Tuple(edges),Tuple(trees),Int(max_depth),Int(nrounds),bins)
    end
end
function _bin(edge,x::Float64)
    Base.@nospecialize edge
    lo=1;hi=length(edge)+1
    while lo<hi
        mid=lo+((hi-lo)>>>1)
        isless(edge[mid],x) ? (lo=mid+1) : (hi=mid)
    end
    lo
end
function _with_bins(model::Model,bins)
    total=0.0f0
    for tree in model.trees
        node=1
        while tree.split[node]
            node=bins[tree.feature[node]]<=tree.condition[node] ? 2node : 2node+1
        end
        total=Float32(total+tree.prediction[node])
        need(isfinite(total),"nonfinite ordered Float32 accumulation")
    end
    Float64(total)
end
"Replay one finite Float64 feature vector using Float64 edges and ordered Float32 tree accumulation, widened to Float64."
function predict_delta(model::Model,values::AbstractVector{Float64})
    need(length(values)==342&&all(isfinite,values),"342 finite Float64 input values required")
    Base.require_one_based_indexing(values)
    bins=Int[_bin(model.edges[j],values[j]) for j in 1:342]
    _with_bins(model,bins)
end
"Replay a finite Float64 matrix with342 columns and0:500000 rows, reusing a bin buffer."
function predict_deltas(model::Model,X::AbstractMatrix{Float64})
    need(size(X,2)==342&&size(X,1)<=500_000&&all(isfinite,X),"bounded finite342matrix required")
    Base.require_one_based_indexing(X);result=Vector{Float64}(undef,size(X,1));bins=Vector{Int}(undef,342)
    edges=Tuple{Vararg{Float64}}[edge for edge in model.edges]
    for i in axes(X,1)
        for j in 1:342;bins[j]=_bin(edges[j],X[i,j]);end
        result[i]=_with_bins(model,bins)
    end
    result
end
end
