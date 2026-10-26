"""Bounded immutable values for the retained ridge, histogram boost and forest baselines."""
module ArchiveClassicPointModels
using SHA,LinearAlgebra
import ..CompletedSourceDBDT
const R=CompletedSourceDBDT.CompletedRefitArtifacts
include("classic_training.jl")
const T=RetainedClassicTraining
export Recipe,NumericTree,NumericModel,fit_numeric,predict,encode_model,decode_model
need(ok,message)=ok || throw(ArgumentError(message))
isint(x)=x isa Integer && !(x isa Bool)
number(x)=x isa Real && !(x isa Bool) && isfinite(x)
sequence(x)=x isa AbstractVector || x isa Tuple
const MAX_ROWS=600000
const MAX_BYTES=8_388_608
width(mode)=mode===:ground11 ? 11 : mode===:mixed8 ? 8 : throw(ArgumentError("classic model layout differs"))
bits(x::Float64)=string(reinterpret(UInt64,x);base=16,pad=16)
function unbits(value)
    need(value isa AbstractString && occursin(r"\A[0-9a-f]{16}\z",value),"classic model bit syntax")
    result=reinterpret(Float64,parse(UInt64,value;base=16));need(isfinite(result),"classic model nonfinite encoded number");result
end
"""Bind an explicit retained baseline recipe; the value grants no archival fit permission."""
struct Recipe
    kind::Symbol
    settings::NamedTuple
    json::String
    sha256::String
    function Recipe(;kind,bins=64,depth=nothing,trees=nothing,penalty=nothing,rate=nothing,
                    minleaf=nothing,sample_rows=nothing,seed=nothing)
        need(kind in (:ridge,:boost,:forest),"classic recipe family")
        depth=depth===nothing ? (kind===:ridge ? 0 : kind===:boost ? 3 : 9) : depth
        trees=trees===nothing ? (kind===:ridge ? 0 : kind===:boost ? 100 : 64) : trees
        penalty=penalty===nothing ? (kind===:ridge ? 1.0 : 0.0) : penalty
        rate=rate===nothing ? (kind===:boost ? 0.05 : 0.0) : rate
        minleaf=minleaf===nothing ? (kind===:boost ? 64 : kind===:forest ? 32 : 1) : minleaf
        sample_rows=sample_rows===nothing ? (kind===:forest ? 20000 : 0) : sample_rows
        seed=seed===nothing ? (kind===:forest ? 20171001 : 0) : seed
        need(isint(bins)&&2<=bins<=255 && isint(depth)&&0<=depth<=9 && isint(trees)&&0<=trees<=400 &&
            isint(minleaf)&&1<=minleaf<=MAX_ROWS && isint(sample_rows)&&0<=sample_rows<=MAX_ROWS &&
            isint(seed)&&0<=seed<=2147483647,"classic recipe integer bounds")
        need(number(penalty)&&0<=penalty<=1e6 && number(rate)&&0<=rate<=1,"classic recipe numerical bounds")
        if kind===:ridge
            need(depth==trees==sample_rows==seed==0 && minleaf==1 && rate==0,"classic ridge unused parameters")
        elseif kind===:boost
            need(depth<=4 && trees>=1 && penalty==0 && 0<rate<=1 && sample_rows==seed==0,"classic boost parameters")
        else
            need(1<=trees<=128 && penalty==rate==0 && sample_rows>=1,"classic forest parameters")
        end
        settings=(;bins=Int(bins),depth=Int(depth),trees=Int(trees),penalty=Float64(penalty),rate=Float64(rate),
            minleaf=Int(minleaf),sample_rows=Int(sample_rows),seed=Int(seed))
        object=Dict("schema"=>"archive_classic_recipe_v1","kind"=>String(kind),
            "parameters"=>Dict(String(k)=>(v isa AbstractFloat ? bits(v) : v) for (k,v) in pairs(settings)))
        bytes=R.canonical_bytes(object;limit=16384)
        new(kind,settings,String(copy(bytes)),bytes2hex(sha256(bytes)))
    end
end
function _recipe(value)
    need(value isa AbstractDict && Set(keys(value))==Set(("schema","kind","parameters")) &&
        value["schema"]=="archive_classic_recipe_v1" && value["kind"] in ("ridge","boost","forest"),"classic recipe codec fields")
    p=value["parameters"]
    need(p isa AbstractDict && Set(keys(p))==Set(("bins","depth","trees","penalty","rate","minleaf","sample_rows","seed")),"classic recipe parameter fields")
    result=Recipe(;kind=Symbol(value["kind"]),bins=p["bins"],depth=p["depth"],trees=p["trees"],
        penalty=unbits(p["penalty"]),rate=unbits(p["rate"]),minleaf=p["minleaf"],sample_rows=p["sample_rows"],seed=p["seed"])
    need(Vector{UInt8}(codeunits(result.json))==R.canonical_bytes(value;limit=16384),"classic recipe reconstruction differs");result
end
"""Own bounded retained-tree arrays; model construction checks reachability and depth."""
struct NumericTree
    feature::Tuple{Vararg{Int}}
    threshold::Tuple{Vararg{UInt8}}
    left::Tuple{Vararg{Int}}
    right::Tuple{Vararg{Int}}
    value::Tuple{Vararg{Float64}}
    function NumericTree(feature,threshold,left,right,value)
        need(all(sequence,(feature,threshold,left,right,value)),"classic tree sequences")
        n=length(value);need(1<=n<=1023 && all(length(x)==n for x in (feature,threshold,left,right)),"classic tree bounded lengths")
        need(all(x->isint(x)&&0<=x<=11,feature) && all(x->isint(x)&&0<=x<=254,threshold) &&
            all(x->isint(x)&&0<=x<=n,left) && all(x->isint(x)&&0<=x<=n,right) && all(number,value),"classic tree field domain")
        converted=Float64.(value);need(all(isfinite,converted),"classic tree Float64 range")
        new(Tuple(Int.(feature)),Tuple(UInt8.(threshold)),Tuple(Int.(left)),Tuple(Int.(right)),Tuple(converted))
    end
end
function _tree_geometry(tree,cuts,depth)
    n=length(tree.value);need(n<=2^(depth+1)-1,"classic tree depth geometry");seen=falses(n)
    function visit(node,level)
        need(1<=node<=n && !seen[node] && level<=depth,"classic tree edge,cycle or depth")
        seen[node]=true;f=tree.feature[node]
        if f==0
            need(tree.left[node]==tree.right[node]==0 && tree.threshold[node]==0,"classic leaf unused fields")
        else
            need(1<=f<=length(cuts) && 1<=tree.threshold[node]<=length(cuts[f]),"classic split geometry")
            visit(tree.left[node],level+1);visit(tree.right[node],level+1)
        end
    end
    visit(1,0);need(all(seen),"classic tree unreachable node");nothing
end
"""Own a retained baseline's recipe, cut points and finite numeric model values."""
struct NumericModel
    recipe::Recipe
    mode::Symbol
    cuts::Tuple{Vararg{Tuple{Vararg{Float64}}}}
    trees::Tuple{Vararg{NumericTree}}
    intercept::Float64
    center::Tuple{Vararg{Float64}}
    scale::Tuple{Vararg{Float64}}
    coefficients::Tuple{Vararg{Float64}}
    function NumericModel(recipe::Recipe,mode,cuts,trees;intercept=0.0,center=(),scale=(),coefficients=())
        p=width(mode);need(sequence(cuts)&&length(cuts)==p && all(sequence,cuts),"classic model cut layout")
        owned=Tuple{Vararg{Float64}}[]
        for raw in cuts
            need(length(raw)<=recipe.settings.bins-1 && all(number,raw),"classic cut bound or number")
            values=Float64.(raw);need(all(isfinite,values)&&issorted(values)&&allunique(values),"classic cut order or Float64 range")
            push!(owned,Tuple(values))
        end
        need(sequence(trees)&&all(t->t isa NumericTree,trees) && number(intercept),"classic model tree or intercept")
        need(all(sequence,(center,scale,coefficients)) && all(x->all(number,x),(center,scale,coefficients)),"classic model finite regression values")
        regression=map(x->Float64.(x),(center,scale,coefficients))
        need(all(x->all(isfinite,x),regression) && isfinite(Float64(intercept)),"classic model Float64 range")
        if recipe.kind===:ridge
            need(isempty(trees) && intercept==0 && length(center)==length(scale)==p && length(coefficients)==p+1 && all(>(0),scale),"classic ridge model geometry")
        else
            need(length(trees)==recipe.settings.trees && isempty(center)&&isempty(scale)&&isempty(coefficients),"classic ensemble model geometry")
            recipe.kind===:forest && need(intercept==0,"classic forest unused intercept")
            for tree in trees;_tree_geometry(tree,owned,recipe.settings.depth);end
        end
        new(recipe,mode,Tuple(owned),Tuple(trees),Float64(intercept),Tuple(regression[1]),Tuple(regression[2]),Tuple(regression[3]))
    end
end
function _matrix(X;minimum=0)
    need(X isa AbstractMatrix{Float64} && minimum<=size(X,1)<=MAX_ROWS && size(X,2) in (8,11) && all(isfinite,X),"classic finite matrix bound")
    Base.require_one_based_indexing(X);nothing
end
function _copy_native(native,cuts,recipe,mode)
    if recipe.kind===:ridge
        need(native isa T.RidgeModel,"classic native ridge type")
        return NumericModel(recipe,mode,cuts,();center=copy(native.mean),scale=copy(native.scale),coefficients=copy(native.coefficients))
    end
    need(recipe.kind===:boost ? native isa T.BoostedModel : native isa T.ForestModel,"classic native ensemble type")
    trees=[NumericTree(copy(t.feature),Int.(t.threshold),copy(t.left),copy(t.right),copy(t.value)) for t in native.trees]
    NumericModel(recipe,mode,cuts,trees;intercept=recipe.kind===:boost ? native.intercept : 0.0)
end
"""Fit a bounded retained recipe and return an immutable copy plus native diagnostics.

This low-level numerical routine grants no source, archival fit or publication
permission. The source-bound fitting adapter supplies those separate checks.
"""
function fit_numeric(X::AbstractMatrix{Float64},y::AbstractVector{Float64},recipe::Recipe)
    _matrix(X;minimum=2);need(length(y)==size(X,1) && all(isfinite,y),"classic finite target bound")
    Base.require_one_based_indexing(y);s=recipe.settings;cuts=T.fit_bins(X;nbins=s.bins)
    native=if recipe.kind===:ridge
        T.fit_ridge(X,y;penalty=s.penalty)
    else
        bins=T.apply_bins(X,cuts)
        recipe.kind===:boost ? T.fit_boost(bins,y;maxdepth=s.depth,ntrees=s.trees,rate=s.rate,minleaf=s.minleaf) :
            T.fit_forest(bins,y;maxdepth=s.depth,ntrees=s.trees,minleaf=s.minleaf,sample_rows=s.sample_rows,seed=s.seed)
    end
    mode=size(X,2)==11 ? :ground11 : :mixed8
    model=_copy_native(native,cuts,recipe,mode)
    author=recipe.kind===:ridge ? T.predict_ridge(native,X) : recipe.kind===:boost ?
        T.predict_boost(native,T.apply_bins(X,cuts)) : T.predict_forest(native,T.apply_bins(X,cuts))
    need(isequal(predict(model,X),author),"classic immutable replay differs from native")
    (;model,native,cuts,training_rows=length(y))
end
function _tree_predict(tree,B)
    result=Vector{Float64}(undef,size(B,1))
    for row in axes(B,1)
        node=1
        while tree.feature[node]!=0
            node=B[row,tree.feature[node]]<=tree.threshold[node] ? tree.left[node] : tree.right[node]
        end
        result[row]=tree.value[node]
    end
    result
end
"""Reproduce retained Float64 prediction operations from bounded immutable values."""
function predict(model::NumericModel,X::AbstractMatrix{Float64})
    _matrix(X);need(size(X,2)==width(model.mode),"classic prediction layout")
    size(X,1)==0 && return Float64[]
    result=if model.recipe.kind===:ridge
        hcat(ones(size(X,1)),(X .- collect(model.center)') ./ collect(model.scale)')*collect(model.coefficients)
    else
        B=T.apply_bins(X,[collect(c) for c in model.cuts])
        prediction=fill(model.intercept,size(X,1))
        for tree in model.trees
            if model.recipe.kind===:boost;prediction .+= model.recipe.settings.rate .* _tree_predict(tree,B)
            else;prediction .+= _tree_predict(tree,B)
            end
        end
        model.recipe.kind===:forest ? prediction ./ length(model.trees) : prediction
    end
    need(all(isfinite,result),"classic prediction overflow");result
end
function _object(model)
    Dict("schema"=>"archive_classic_point_model_v1","recipe"=>R.decode_canonical(Vector{UInt8}(codeunits(model.recipe.json));limit=16384),
        "mode"=>String(model.mode),"cuts"=>[[bits(x) for x in c] for c in model.cuts],
        "trees"=>[Dict("feature"=>collect(t.feature),"threshold"=>Int.(t.threshold),"left"=>collect(t.left),
            "right"=>collect(t.right),"value"=>[bits(x) for x in t.value]) for t in model.trees],
        "intercept"=>bits(model.intercept),"center"=>[bits(x) for x in model.center],
        "scale"=>[bits(x) for x in model.scale],"coefficients"=>[bits(x) for x in model.coefficients])
end
"""Encode a bounded retained numeric model as canonical floating-bit JSON."""
encode_model(model::NumericModel)=R.canonical_bytes(_object(model);limit=MAX_BYTES)
"""Decode owned canonical bytes, rejecting malformed geometry and arbitrary objects."""
function decode_model(bytes)
    need(bytes isa AbstractVector{UInt8} && length(bytes)<=MAX_BYTES,"classic codec byte bound")
    value=R.decode_canonical(copy(bytes);limit=MAX_BYTES)
    need(value isa AbstractDict && Set(keys(value))==Set(("schema","recipe","mode","cuts","trees","intercept","center","scale","coefficients")) &&
        value["schema"]=="archive_classic_point_model_v1" && value["mode"] in ("ground11","mixed8"),"classic codec fields")
    recipe=_recipe(value["recipe"]);cuts=value["cuts"];trees=value["trees"]
    need(cuts isa AbstractVector && length(cuts)==width(Symbol(value["mode"])) &&
        all(c->c isa AbstractVector && length(c)<=recipe.settings.bins-1,cuts),"classic codec cut geometry")
    need(trees isa AbstractVector && length(trees)==recipe.settings.trees,"classic codec tree count")
    decoded=NumericTree[]
    for tree in trees
        need(tree isa AbstractDict && Set(keys(tree))==Set(("feature","threshold","left","right","value")),"classic codec tree fields")
        need(all(tree[k] isa AbstractVector && 1<=length(tree[k])<=1023 for k in keys(tree)),"classic codec tree lengths")
        push!(decoded,NumericTree(tree["feature"],tree["threshold"],tree["left"],tree["right"],[unbits(x) for x in tree["value"]]))
    end
    need(all(value[k] isa AbstractVector && length(value[k])<=12 for k in ("center","scale","coefficients")),"classic codec regression geometry")
    model=NumericModel(recipe,Symbol(value["mode"]),[[unbits(x) for x in c] for c in cuts],decoded;
        intercept=unbits(value["intercept"]),center=[unbits(x) for x in value["center"]],
        scale=[unbits(x) for x in value["scale"]],coefficients=[unbits(x) for x in value["coefficients"]])
    need(encode_model(model)==bytes,"classic codec reconstruction differs");model
end
end
