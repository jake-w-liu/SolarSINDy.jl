"""The locked mixed8 paper ridge, distinct from the generic retained ridge.

These numerical values grant no case access, fit authority or scientific
qualification. Actual-source fitting belongs to ArchiveSourceBoundPointFit.
"""
module ArchiveLockedMixed8Ridge
using SHA,LinearAlgebra
import ..CompletedSourceDBDT, ..ArchiveClassicPointModels
const R=CompletedSourceDBDT.CompletedRefitArtifacts
const C=ArchiveClassicPointModels
export Recipe,NumericModel,fit_numeric,predict,encode_model,decode_model
const FAMILY="locked_mixed8_ridge_log1p_v1"
const RECIPE_SCHEMA="archive_locked_mixed8_ridge_recipe_v1"
const MODEL_SCHEMA="archive_locked_mixed8_ridge_model_v1"
const FEATURES=("dbdt_now","dbdt_mean30","dbdt_max30","dbdt_std30","V","Bz","Bs","VBs")
const TARGET="log1p_rate_reference_1_nt_per_min"
const STD_RULE="sample_std_plus_1e_minus9"
const EPSILON=1e-9
const MAX_ROWS=600000
const MAX_BYTES=16384
need(ok,message)=ok || throw(ArgumentError(message))
number(x)=x isa Real && !(x isa Bool) && isfinite(x)
sequence(x)=x isa AbstractVector || x isa Tuple

"""Bind only the locked penalty-one, all-coefficient mixed8 convention."""
struct Recipe
    settings::NamedTuple
    json::String
    sha256::String
    function Recipe(;penalty=1.0)
        need(number(penalty) && penalty==1.0,"locked mixed8 ridge requires penalty one")
        settings=(;penalty=1.0,epsilon=EPSILON,intercept_penalty=true,std_rule=STD_RULE)
        object=Dict("schema"=>RECIPE_SCHEMA,"family"=>FAMILY,"layout"=>"mixed8",
            "feature_names"=>collect(FEATURES),"target_transform"=>TARGET,
            "penalty"=>C.bits(settings.penalty),"intercept_penalty"=>true,
            "std_rule"=>STD_RULE,"epsilon"=>C.bits(EPSILON))
        bytes=R.canonical_bytes(object;limit=MAX_BYTES)
        new(settings,String(copy(bytes)),bytes2hex(sha256(bytes)))
    end
end
function _recipe(value)
    fields=Set(("schema","family","layout","feature_names","target_transform","penalty",
                "intercept_penalty","std_rule","epsilon"))
    need(value isa AbstractDict && Set(keys(value))==fields,"locked mixed8 recipe fields")
    need(value["schema"]==RECIPE_SCHEMA && value["family"]==FAMILY && value["layout"]=="mixed8" &&
        value["target_transform"]==TARGET && value["intercept_penalty"]===true &&
        value["std_rule"]==STD_RULE,"locked mixed8 recipe convention")
    names=value["feature_names"]
    need(names isa AbstractVector && length(names)==8 && all(x->x isa AbstractString,names) &&
        Tuple(names)==FEATURES,"locked mixed8 ordered features")
    result=Recipe(;penalty=C.unbits(value["penalty"]))
    need(C.unbits(value["epsilon"])==EPSILON &&
        R.canonical_bytes(value;limit=MAX_BYTES)==Vector{UInt8}(codeunits(result.json)),
        "locked mixed8 recipe reconstruction differs")
    result
end

"""Own eight means/scales and nine coefficients under the locked convention."""
struct NumericModel
    recipe::Recipe
    mode::Symbol
    center::NTuple{8,Float64}
    scale::NTuple{8,Float64}
    coefficients::NTuple{9,Float64}
    function NumericModel(recipe::Recipe,mode,center,scale,coefficients)
        need(mode===:mixed8,"locked ridge model requires mixed8 layout")
        need(all(sequence,(center,scale,coefficients)) && length(center)==length(scale)==8 &&
            length(coefficients)==9 && all(values->all(number,values),(center,scale,coefficients)),
            "locked mixed8 model regression geometry")
        means=Float64.(center);scales=Float64.(scale);weights=Float64.(coefficients)
        need(all(isfinite,means) && all(x->isfinite(x)&&x>=EPSILON,scales) && all(isfinite,weights),
            "locked mixed8 model Float64 domain")
        new(recipe,mode,Tuple(means),Tuple(scales),Tuple(weights))
    end
end
function _matrix(X;minimum=0)
    need(X isa AbstractMatrix{Float64} && minimum<=size(X,1)<=MAX_ROWS && size(X,2)==8 && all(isfinite,X),
        "locked mixed8 finite matrix bound")
    Base.require_one_based_indexing(X);nothing
end
function _math_recipe(recipe::Recipe)
    Dict{String,Any}("penalty"=>R.float_bits(recipe.settings.penalty),
        "intercept_penalty"=>true,"std_rule"=>STD_RULE)
end

"""Fit the exact locked standardization and all-coefficient penalty-one solve.

The existing pure preprocessing/statistics helpers provide no synthetic or
actual-source admission. The explicit Diagonal adds the same unit penalty to
all nine Gram diagonal entries as the historical UniformScaling expression;
native operation parity is covered by the separate historical-reference test.
"""
function fit_numeric(X::AbstractMatrix{Float64},y::AbstractVector{Float64},recipe::Recipe)
    _matrix(X;minimum=2)
    need(length(y)==size(X,1) && all(isfinite,y),"locked mixed8 finite target bound")
    Base.require_one_based_indexing(y)
    owned=Matrix{Float64}(X);targets=copy(y);math=_math_recipe(recipe)
    pre,Z=R._ridge_preprocess(owned,math)
    A,b,weights=R._normal_statistics(Z,targets,math)
    beta=A\b
    need(all(isfinite,beta),"locked mixed8 ridge solve overflow")
    means=R.read_float_bits.(pre["mean"]);scales=R.read_float_bits.(pre["scale"])
    model=NumericModel(recipe,:mixed8,means,scales,beta)
    native=(;mean=copy(means),scale=copy(scales),coefficients=copy(beta))
    author=hcat(ones(size(X,1)),(X.-native.mean')./native.scale')*native.coefficients
    need(all(isfinite,author) && isequal(predict(model,X),author),"locked mixed8 numeric replay differs")
    (;model,native,training_rows=length(y),statistics=(;A,b,weights))
end

"""Predict from immutable locked values without targets or fit authority."""
function predict(model::NumericModel,X::AbstractMatrix{Float64})
    _matrix(X)
    size(X,1)==0 && return Float64[]
    Z=hcat(ones(size(X,1)),(X.-collect(model.center)')./collect(model.scale)')
    need(all(isfinite,Z),"locked mixed8 prediction normalization overflow")
    result=Z*collect(model.coefficients)
    need(all(isfinite,result),"locked mixed8 prediction overflow")
    result
end
function _object(model::NumericModel)
    Dict("schema"=>MODEL_SCHEMA,"family"=>FAMILY,"mode"=>"mixed8",
        "recipe"=>R.decode_canonical(Vector{UInt8}(codeunits(model.recipe.json));limit=MAX_BYTES),
        "center"=>[C.bits(x) for x in model.center],"scale"=>[C.bits(x) for x in model.scale],
        "coefficients"=>[C.bits(x) for x in model.coefficients])
end
encode_model(model::NumericModel)=R.canonical_bytes(_object(model);limit=MAX_BYTES)
function decode_model(bytes)
    need(bytes isa AbstractVector{UInt8} && length(bytes)<=MAX_BYTES,"locked mixed8 model byte bound")
    value=R.decode_canonical(copy(bytes);limit=MAX_BYTES)
    need(value isa AbstractDict && Set(keys(value))==Set(("schema","family","mode","recipe","center","scale","coefficients")) &&
        value["schema"]==MODEL_SCHEMA && value["family"]==FAMILY && value["mode"]=="mixed8",
        "locked mixed8 model codec fields")
    recipe=_recipe(value["recipe"])
    for (name,width) in (("center",8),("scale",8),("coefficients",9))
        need(value[name] isa AbstractVector && length(value[name])==width,"locked mixed8 model codec geometry")
    end
    model=NumericModel(recipe,:mixed8,C.unbits.(value["center"]),C.unbits.(value["scale"]),C.unbits.(value["coefficients"]))
    need(encode_model(model)==bytes,"locked mixed8 model reconstruction differs")
    model
end
end
