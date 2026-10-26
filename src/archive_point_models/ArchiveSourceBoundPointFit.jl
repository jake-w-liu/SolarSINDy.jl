"""Source-bound fixed-recipe point fits with separate archival admission and data-only publication."""
module ArchiveSourceBoundPointFit
using Dates,SHA,LinearAlgebra
import ..ArchivePointNumericTrees, ..ArchiveClassicPointModels, ..ArchiveCatalogCaseJob, ..ArchiveCatalogCaseReader
import ..ArchiveCatalogCaseAuthority, ..ArchiveCatalogSourceJob, ..ArchiveCatalogSnapshotFilesystem, ..CompletedSourceDBDT, ..MatchedGBMPrimitives342
const ROOT=parentmodule(@__MODULE__)
const C=ArchiveClassicPointModels
const B=ArchiveCatalogCaseReader
const A=ArchiveCatalogCaseAuthority
const J=ArchiveCatalogSourceJob
const FS=ArchiveCatalogSnapshotFilesystem
const Q=ArchiveCatalogCaseJob
const R=CompletedSourceDBDT.CompletedRefitArtifacts
const P=ArchivePointNumericTrees
export BoostRecipe,ExpectedFit,BoundFit,FitAuthority,prepare_expected,fit_generated,fit_archival,predict_log,receipt_bytes,publish_fit,read_published
need(ok,message)=ok || throw(ArgumentError(message))
isint(x)=x isa Integer && !(x isa Bool)
hex(x)=x isa AbstractString && occursin(r"\A[0-9a-f]{64}\z",x)
digest(x)=bytes2hex(sha256(x))
bits(x::Float64)=string(reinterpret(UInt64,x);base=16,pad=16)
clock(t)=Dates.format(t,dateformat"yyyy-mm-ddTHH:MM:SS.sss")*"Z"
const LIBRARY_SHA="f0d3511130a9a84ba4bce823bd47055fba2e11d368db3a0376fb48951e4ecd8f"
"""Explicit bounded CPU EvoTrees MSE recipe for a log1p rate target."""
struct BoostRecipe
    settings::NamedTuple
    json::String
    sha256::String
    function BoostRecipe(;depth,rounds,bins,seed,eta,min_weight,lambda,L2)
        need(isint(depth)&&1<=depth<=6 && isint(rounds)&&1<=rounds<=400 && isint(bins)&&2<=bins<=255,"fit recipe integer geometry")
        need(isint(seed)&&0<=seed<=2147483647,"fit recipe seed")
        floats=(eta,min_weight,lambda,L2)
        need(all(x->x isa Real && !(x isa Bool) && isfinite(x),floats),"fit recipe finite numbers")
        e,w,l,r=Float64.(floats)
        need(0<e<=1 && 0<=w<=600000 && 0<=l<=1e6 && 0<=r<=1e6,"fit recipe numerical bounds")
        settings=(;max_depth=Int(depth),nrounds=Int(rounds),nbins=Int(bins),seed=Int(seed),eta=e,min_weight=w,lambda=l,L2=r,
            gamma=0.0,rowsample=1.0,colsample=1.0)
        object=Dict("schema"=>"source_bound_boost_recipe_v1","family"=>"cpu_scalar_mse_log1p",
            "parameters"=>Dict(String(k)=>(v isa AbstractFloat ? bits(v) : v) for (k,v) in pairs(settings)))
        bytes=R.canonical_bytes(object;limit=16384);new(settings,String(copy(bytes)),digest(bytes))
    end
end
"""External context, recipe, source-closure and pinned library digests for one fit."""
struct ExpectedFit
    context_sha256::String
    recipe_sha256::String
    source_sha256::String
    library_sha256::String
    function ExpectedFit(context,recipe,source,library)
        need(all(hex,(context,recipe,source,library)) && library==LIBRARY_SHA,"expected fit pins")
        new(String(context),String(recipe),String(source),String(library))
    end
end
function _source_files()
    paths=Pair{String,String}[];root=joinpath(pkgdir(ROOT),"src");J._ordinary(root;directory=true)
    for (directory,dirs,files) in walkdir(root;follow_symlinks=false)
        J._ordinary(directory;directory=true)
        for name in dirs;J._ordinary(joinpath(directory,name);directory=true);end
        for name in files
            endswith(name,".jl") || continue
            path=joinpath(directory,name);J._ordinary(path)
            push!(paths,"SolarSINDy/src/"*replace(relpath(path,root),'\\'=>'/')=>path)
            need(length(paths)<=1024,"fit source closure bound")
        end
    end
    Tuple(sort!(paths;by=first))
end
const _FILES=_source_files()
const _SOURCE_PINS=Tuple(digest(read(last(p))) for p in _FILES)
function source_identity()
    need(_source_files()==_FILES && all(digest(read(last(p)))==h for (p,h) in zip(_FILES,_SOURCE_PINS)),"fit loaded source changed")
    need(MatchedGBMPrimitives342._trusted_evo_sha()==LIBRARY_SHA,"fit library source differs")
    entries=[Dict("path"=>first(p),"sha256"=>h) for (p,h) in zip(_FILES,_SOURCE_PINS)]
    digest(R.canonical_bytes(Dict("schema"=>"source_bound_fit_source_closure_v1","files"=>entries);limit=262144))
end
const PointRecipe=Union{BoostRecipe,C.Recipe}
const PointModel=Union{P.NumericModel,C.NumericModel}
family(recipe::BoostRecipe)="evotrees_cpu_mse_log1p"
family(recipe::C.Recipe)="retained_"*String(recipe.kind)*"_log1p"
model_bytes(model::P.NumericModel)=P.encode_model(model)
model_bytes(model::C.NumericModel)=C.encode_model(model)
model_predict(model::P.NumericModel,X)=P.predict(model,X)
model_predict(model::C.NumericModel,X)=C.predict(model,X)
struct _ArchivalBinding
    json::String
    admission_sha256::String
    input_sha256::String
end
function _update(ctx,values::AbstractVector{Float64})
    SHA.update!(ctx,reinterpret(UInt8,htol.(reinterpret(UInt64,values))))
end
function _data_sha(arrays)
    X=arrays.features;n,p=size(X)
    need(2<=n<=600000 && p in (8,11) && length(arrays.rows)==length(arrays.issues)==length(arrays.targets)==length(arrays.raw_now)==length(arrays.scales)==n,"fit matrix geometry")
    need(all(isfinite,X) && all(x->isfinite(x)&&x>=0,arrays.targets) && all(isfinite,arrays.raw_now) && all(x->isfinite(x)&&x>0,arrays.scales),"fit finite input domain")
    need(issorted(arrays.rows)&&allunique(arrays.rows)&&issorted(arrays.issues)&&allunique(arrays.issues),"fit chronological unique rows")
    y=log1p.(arrays.targets);need(all(isfinite,y),"fit transformed target domain")
    ctx=SHA.SHA2_256_CTX();SHA.update!(ctx,codeunits("SOURCE_BOUND_POINT_INPUT_V1\n"))
    SHA.update!(ctx,reinterpret(UInt8,htol.(UInt64[n,p])))
    SHA.update!(ctx,reinterpret(UInt8,htol.(UInt64.(arrays.rows))))
    SHA.update!(ctx,reinterpret(UInt8,htol.(reinterpret(UInt64,Int64[Dates.value(t) for t in arrays.issues]))))
    for column in axes(X,2);_update(ctx,collect(view(X,:,column)));end
    for values in (arrays.targets,y,arrays.raw_now,arrays.scales);_update(ctx,values);end
    bytes2hex(SHA.digest!(ctx)),y
end
function _prepare(value::Q.VerifiedCases,index,recipe::PointRecipe,start,stop,asof,binding=nothing)
    actual=binding isa _ArchivalBinding
    need((actual || binding===nothing) && value.actual_source_authority_granted===actual && value.actual_fit_performed===false,"actual archival fit authority required")
    request=R.decode_canonical(Vector{UInt8}(codeunits(value.request_json));limit=131072)
    need(request["schema"]==(actual ? "archive_catalog_case_request_v1" : "generated_catalog_case_request_v1") && request["evidence_mode"]==(actual ? "actual_archival_cache" : "generated"),"fit case evidence context")
    need(digest(codeunits(value.request_json))==value.request_sha256,"fit case request binding")
    need(isint(index)&&1<=index<=length(request["tables"]),"fit table index")
    need(all(t->t isa DateTime,(start,stop,asof)) && start<stop<=asof,"fit split clocks")
    source=source_identity()
    arrays=Q.split_arrays(value,index;start,stop,asof)
    input,y=_data_sha(arrays)
    actual && need(input==binding.input_sha256,"archival fit input differs from independent grant")
    object=Dict("schema"=>(actual ? "source_bound_archival_fit_context_v1" : "source_bound_generated_fit_context_v1"),"evidence_mode"=>request["evidence_mode"],
        "case_request_sha256"=>value.request_sha256,"case_manifest_sha256"=>value.manifest_sha256,"case_complete_sha256"=>value.complete_sha256,
        "table_index"=>Int(index),"table"=>request["tables"][index],"role"=>request["role"],"layout"=>request["layout"],
        "start_inclusive"=>clock(start),"stop_exclusive"=>clock(stop),"target_asof"=>clock(asof),"label_ready_before"=>clock(stop),
        "rows"=>length(y),"first_row"=>first(arrays.rows),"last_row"=>last(arrays.rows),
        "input_sha256"=>input,"target_transform"=>"log1p_rate_reference_1_nt_per_min",
        "recipe_sha256"=>recipe.sha256,"source_sha256"=>source,"library_sha256"=>LIBRARY_SHA,"runtime"=>string(VERSION))
    actual && (object["case_read_admission_sha256"]=binding.admission_sha256)
    bytes=R.canonical_bytes(object;limit=131072)
    (;arrays,y,context=String(copy(bytes)),context_sha256=digest(bytes),source,input)
end
"""Derive pre-fit context pins from a verified generated case and an explicit strict split."""
function prepare_expected(value::Q.VerifiedCases,index,recipe::PointRecipe;start,stop,asof)
    prepared=_prepare(value,index,recipe,start,stop,asof)
    ExpectedFit(prepared.context_sha256,recipe.sha256,prepared.source,LIBRARY_SHA)
end
struct _FitToken end
"""Own an immutable numeric model, its independently pinned context and fit receipt."""
struct BoundFit
    model::PointModel
    expected::ExpectedFit
    context::String
    recipe::String
    receipt::String
    function BoundFit(model,expected,context,recipe,receipt,::_FitToken)
        new(model,expected,context,recipe,receipt)
    end
end
function _matches(expected,prepared,recipe)
    need(expected.context_sha256==prepared.context_sha256 && expected.recipe_sha256==recipe.sha256 &&
        expected.source_sha256==prepared.source && expected.library_sha256==LIBRARY_SHA,"independent expected fit binding differs")
end
function _native_fit(X,y,recipe::BoostRecipe)
    native=getproperty(ROOT,:gbm_fit)(X,y;recipe.settings...);s=recipe.settings
    model=P.from_native(native;mode=size(X,2)==11 ? :ground11 : :mixed8,depth=s.max_depth,rounds=s.nrounds,bins=s.nbins)
    (;model,native,author=getproperty(ROOT,:gbm_predict)(native,X))
end
function _native_fit(X,y,recipe::C.Recipe)
    result=C.fit_numeric(X,y,recipe)
    s=recipe.settings;T=C.RetainedClassicTraining
    author=recipe.kind===:ridge ? T.predict_ridge(result.native,X) : recipe.kind===:boost ?
        T.predict_boost(result.native,T.apply_bins(X,result.cuts)) : T.predict_forest(result.native,T.apply_bins(X,result.cuts))
    (;model=result.model,native=result.native,author)
end
function _fit_prepared(value,index,recipe,expected,start,stop,asof,fitter,binding;attempt_sha256=nothing)
    actual=binding isa _ArchivalBinding
    actual ? J._sha(attempt_sha256) : need(attempt_sha256===nothing,"generated fit cannot consume an archival attempt")
    prepared=_prepare(value,index,recipe,start,stop,asof,binding);_matches(expected,prepared,recipe)
    X=copy(prepared.arrays.features);y=copy(prepared.y)
    fitted=fitter(X,y,recipe)
    need(isequal(X,prepared.arrays.features) && isequal(y,prepared.y),"native fitter mutated its owned inputs")
    need(fitted isa NamedTuple && all(k->hasproperty(fitted,k),(:model,:native,:author)),"fit native result fields")
    need(fitted.model isa PointModel,"fit numeric model type")
    need(recipe isa BoostRecipe ? fitted.model isa P.NumericModel && fitted.model.depth==recipe.settings.max_depth &&
        fitted.model.rounds==recipe.settings.nrounds && fitted.model.bins==recipe.settings.nbins :
        fitted.model isa C.NumericModel && fitted.model.recipe.json==recipe.json,"fit numeric model recipe differs")
    reconstructed=model_predict(fitted.model,X);author=fitted.author
    need(author isa Vector{Float64} && isequal(reinterpret(UInt64,reconstructed),reinterpret(UInt64,author)),"fit numeric replay differs from native prediction")
    after=_prepare(value,index,recipe,start,stop,asof,binding)
    _matches(expected,after,recipe);need(after.input==prepared.input && after.context==prepared.context,"fit case input changed")
    actual=binding isa _ArchivalBinding;encoded=model_bytes(fitted.model)
    receipt=Dict("schema"=>"source_bound_point_fit_receipt_v1","complete"=>true,"family"=>family(recipe),
        "context_sha256"=>expected.context_sha256,"recipe_sha256"=>recipe.sha256,"source_sha256"=>prepared.source,
        "library_sha256"=>LIBRARY_SHA,"input_sha256"=>prepared.input,"model_sha256"=>digest(encoded),
        "training_rows"=>length(y),"all_training_native_predictions_exact"=>true,
        "actual_source_access"=>actual,"actual_fit_performed"=>actual,"software_qualification"=>false)
    if actual
        authorization=R.decode_canonical(Vector{UInt8}(codeunits(binding.json));limit=16384)
        authorization["fit_attempt_sha256"]=J._sha(attempt_sha256)
        receipt["authorization"]=authorization
    else
        need(attempt_sha256===nothing,"generated fit cannot consume an archival attempt")
    end
    bytes=R.canonical_bytes(receipt;limit=65536)
    capsule=BoundFit(fitted.model,expected,prepared.context,recipe.json,String(copy(bytes)),_FitToken())
    (;capsule,native=fitted.native,training_rows=length(y))
end
function _fit(value::Q.VerifiedCases,index,recipe::PointRecipe,expected::ExpectedFit,start,stop,asof,fitter)
    _fit_prepared(value,index,recipe,expected,start,stop,asof,fitter,nothing)
end
"""Fit verified generated cases on their prebound split; no archival permission is consumed."""
fit_generated(value::Q.VerifiedCases,index,recipe::PointRecipe,expected::ExpectedFit;start,stop,asof)=
    _fit(value,index,recipe,expected,start,stop,asof,_native_fit)
"""Predict the log1p rate target from an immutable numeric model after checking its receipt."""
function predict_log(value::BoundFit,X::AbstractMatrix{Float64})
    _check_capsule(value)
    model_predict(value.model,X)
end
"""Return owned canonical receipt bytes; reading this receipt grants no additional fit."""
receipt_bytes(value::BoundFit)=Vector{UInt8}(codeunits(value.receipt))
include("point_fit_authority.jl")
include("point_fit_publication.jl")
end
