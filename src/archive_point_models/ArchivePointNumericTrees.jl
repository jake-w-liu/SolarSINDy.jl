"""Bounded numeric tree values; no source, archival fit or publication authority."""
module ArchivePointNumericTrees
using SHA
import ..CompletedSourceDBDT, ..EvoTrees
const R=CompletedSourceDBDT.CompletedRefitArtifacts
const E=EvoTrees
export NumericTree,NumericModel,predict,from_native,encode_model,decode_model
need(ok,message)=ok || throw(ArgumentError(message))
isint(x)=x isa Integer && !(x isa Bool)
number(x)=x isa Real && !(x isa Bool) && isfinite(x)
sequence(x)=x isa AbstractVector || x isa Tuple
const MAX_ROWS=600000
const MAX_BYTES=8_388_608
const MODES=(:ground11,:mixed8)
width(mode)=mode===:ground11 ? 11 : mode===:mixed8 ? 8 : throw(ArgumentError("point tree mode differs"))
bits(x::Float64)=string(reinterpret(UInt64,x);base=16,pad=16)
bits(x::Float32)=string(reinterpret(UInt32,x);base=16,pad=8)
function unbits(value,::Type{T}) where T<:Union{Float32,Float64}
    n=T===Float64 ? 16 : 8
    need(value isa AbstractString && ncodeunits(value)==n && occursin(r"^[0-9a-f]+$",value),"point tree bit syntax")
    result=T===Float64 ? reinterpret(Float64,parse(UInt64,value;base=16)) : reinterpret(Float32,parse(UInt32,value;base=16))
    need(isfinite(result),"point tree nonfinite encoded number");result
end
"""Own a bounded scalar Float32 heap tree; a value carries no fit authority."""
struct NumericTree
    feature::Tuple{Vararg{Int}}
    condition::Tuple{Vararg{Int}}
    prediction::Tuple{Vararg{Float32}}
    split::Tuple{Vararg{Bool}}
    function NumericTree(feature,condition,prediction,split)
        need(all(sequence,(feature,condition,prediction,split)),"point tree sequences required")
        n=length(feature);need(1<=n<=63 && length(condition)==length(prediction)==length(split)==n,"point tree bounded field lengths")
        need(all(x->isint(x)&&0<=x<=11,feature) && all(x->isint(x)&&0<=x<=254,condition),"point tree integer field")
        need(all(x->x isa Bool,split) && all(number,prediction),"point tree flag or finite leaf")
        values=Float32.(prediction)
        need(all(isfinite,values) && all(a==b for (a,b) in zip(values,prediction)),"point tree exact Float32 leaves")
        new(Tuple(Int.(feature)),Tuple(Int.(condition)),Tuple(values),Tuple(split))
    end
end
"""Own an eight- or eleven-column numeric ensemble with explicit bounded geometry."""
struct NumericModel
    mode::Symbol
    edges::Tuple{Vararg{Tuple{Vararg{Float64}}}}
    trees::Tuple{Vararg{NumericTree}}
    depth::Int
    rounds::Int
    bins::Int
    function NumericModel(mode,raw_edges,raw_trees;depth,rounds,bins)
        p=width(mode)
        need(isint(depth)&&1<=depth<=6 && isint(rounds)&&0<=rounds<=400 && isint(bins)&&2<=bins<=255,"point tree configuration bound")
        need(sequence(raw_edges)&&sequence(raw_trees),"point tree sequences required")
        need(length(raw_edges)==p && length(raw_trees)==rounds+1,"point tree layout or round count")
        edges=Tuple{Vararg{Float64}}[]
        for raw in raw_edges
            need(sequence(raw) && length(raw)==bins-1 && all(number,raw),"point tree finite edge geometry")
            value=Float64.(raw);need(all(isfinite,value)&&issorted(value),"point tree edge order")
            push!(edges,Tuple(value))
        end
        trees=NumericTree[]
        for (index,raw) in enumerate(raw_trees)
            need(raw isa NumericTree,"point tree numeric value required")
            n=index==1 ? 1 : 2^Int(depth)-1
            need(length(raw.feature)==n && all(f->f<=p,raw.feature) && all(c->c<bins,raw.condition),"point tree heap or column geometry")
            reached=falses(n);pending=Int[1]
            while !isempty(pending)
                node=pop!(pending);reached[node]=true
                if raw.split[node]
                    need(index!=1 && 1<=raw.feature[node]<=p && 1<=raw.condition[node]<bins && 2node+1<=n,"point tree reachable split")
                    push!(pending,2node,2node+1)
                end
            end
            need(all(reached[j] || !raw.split[j] for j in 1:n),"point tree unreachable split")
            push!(trees,raw)
        end
        new(mode,Tuple(edges),Tuple(trees),Int(depth),Int(rounds),Int(bins))
    end
end
function bin(edge,value::Float64)
    Base.@nospecialize edge
    lo=1;hi=length(edge)+1
    while lo<hi
        mid=lo+((hi-lo)>>>1)
        isless(edge[mid],value) ? (lo=mid+1) : (hi=mid)
    end
    lo
end
function _prediction(model,bins)
    total=0.0f0
    for tree in model.trees
        node=1
        while tree.split[node]
            node=bins[tree.feature[node]]<=tree.condition[node] ? 2node : 2node+1
        end
        total=Float32(total+tree.prediction[node]);need(isfinite(total),"point tree prediction overflow")
    end
    Float64(total)
end
"""Predict bounded finite rows with Float64 bins and ordered Float32 tree sums."""
function predict(model::NumericModel,X::AbstractMatrix{Float64})
    p=width(model.mode);need(size(X,2)==p && size(X,1)<=MAX_ROWS && all(isfinite,X),"point tree finite matrix bound")
    Base.require_one_based_indexing(X)
    result=Vector{Float64}(undef,size(X,1));bins=Vector{Int}(undef,p)
    edges=Tuple{Vararg{Float64}}[edge for edge in model.edges]
    for row in axes(X,1)
        for column in 1:p;bins[column]=bin(edges[column],X[row,column]);end
        result[row]=_prediction(model,bins)
    end
    result
end
"""Copy a pinned-version scalar MSE native ensemble into numeric values.

This conversion does not certify training inputs, source provenance or recipes.
"""
function from_native(native;mode,depth,rounds,bins)
    need(string(Base.pkgversion(E))=="0.18.7","point tree EvoTrees version differs")
    need(native isa E.EvoTree{E.MSE,1},"point tree scalar MSE native model required")
    need(isint(depth)&&1<=depth<=6 && isint(rounds)&&0<=rounds<=400 && isint(bins)&&2<=bins<=255,"point tree native configuration bound")
    need(length(native.trees)==rounds+1,"point tree native round geometry")
    info=native.info;p=width(mode)
    need(length(info[:feature_names])==p && length(info[:feattypes])==p && all(x->x===true,info[:feattypes]),"point tree ordered numeric native schema")
    need(info[:nrounds]==rounds && length(info[:featbins])==p && all(x->x==bins,info[:featbins]),"point tree native recipe differs")
    trees=NumericTree[]
    for (index,raw) in enumerate(native.trees)
        nodes=index==1 ? 1 : 2^Int(depth)-1
        need(length(raw.feat)==length(raw.cond_bin)==length(raw.split)==nodes && size(raw.pred)==(1,nodes),"point tree native scalar geometry")
        push!(trees,NumericTree(copy(raw.feat),Int.(raw.cond_bin),vec(copy(raw.pred)),copy(raw.split)))
    end
    NumericModel(mode,info[:edges],trees;depth,rounds,bins)
end
function object(model)
    Dict("schema"=>"archive_point_numeric_trees_v1","mode"=>String(model.mode),"depth"=>model.depth,"rounds"=>model.rounds,"bins"=>model.bins,
        "edges"=>[[bits(x) for x in edge] for edge in model.edges],
        "trees"=>[Dict("feature"=>collect(tree.feature),"condition"=>collect(tree.condition),
            "prediction_bits"=>[bits(x) for x in tree.prediction],"split"=>collect(tree.split)) for tree in model.trees])
end
"""Encode bounded numeric values as canonical JSON with explicit floating bits."""
encode_model(model::NumericModel)=R.canonical_bytes(object(model);limit=MAX_BYTES)
"""Reconstruct exact bounded numeric values from owned canonical bytes; no fit authority."""
function decode_model(bytes)
    need(bytes isa AbstractVector{UInt8} && length(bytes)<=MAX_BYTES,"point tree codec byte bound")
    value=R.decode_canonical(copy(bytes);limit=MAX_BYTES)
    need(value isa AbstractDict && Set(keys(value))==Set(("schema","mode","depth","rounds","bins","edges","trees")),"point tree codec fields")
    need(value["schema"]=="archive_point_numeric_trees_v1" && value["mode"] in ("ground11","mixed8"),"point tree codec schema")
    mode=Symbol(value["mode"]);p=width(mode);depth=value["depth"];rounds=value["rounds"];bins=value["bins"]
    need(isint(depth)&&1<=depth<=6 && isint(rounds)&&0<=rounds<=400 && isint(bins)&&2<=bins<=255,"point tree codec configuration")
    need(value["edges"] isa AbstractVector && length(value["edges"])==p && value["trees"] isa AbstractVector && length(value["trees"])==rounds+1,"point tree codec geometry")
    edges=Vector{Float64}[];trees=NumericTree[]
    for edge in value["edges"]
        need(edge isa AbstractVector && length(edge)==bins-1,"point tree codec edge length")
        push!(edges,[unbits(x,Float64) for x in edge])
    end
    for (index,tree) in enumerate(value["trees"])
        need(tree isa AbstractDict && Set(keys(tree))==Set(("feature","condition","prediction_bits","split")),"point tree codec tree fields")
        nodes=index==1 ? 1 : 2^Int(depth)-1
        need(all(tree[key] isa AbstractVector && length(tree[key])==nodes for key in keys(tree)),"point tree codec tree length")
        push!(trees,NumericTree(tree["feature"],tree["condition"],[unbits(x,Float32) for x in tree["prediction_bits"]],tree["split"]))
    end
    model=NumericModel(mode,edges,trees;depth,rounds,bins)
    need(encode_model(model)==bytes,"point tree codec reconstruction differs");model
end
end
