const _FIT_FILES=("context.json","recipe.json","model.json","receipt.json","COMPLETE.json")
const _FIT_LIMITS=(131072,16384,8_388_608,65536,4096)
function _recipe_bytes(bytes)
    value=R.decode_canonical(bytes;limit=16384)
    need(value isa AbstractDict,"fit recipe must be an object")
    if get(value,"schema",nothing)=="archive_classic_recipe_v1"
        return C._recipe(value)
    end
    J._object(value,Set(("schema","family","parameters")))
    need(value["schema"]=="source_bound_boost_recipe_v1" && value["family"]=="cpu_scalar_mse_log1p","fit recipe codec schema")
    p=value["parameters"]
    J._object(p,Set(("max_depth","nrounds","nbins","seed","eta","min_weight","lambda","L2","gamma","rowsample","colsample")))
    result=BoostRecipe(depth=p["max_depth"],rounds=p["nrounds"],bins=p["nbins"],seed=p["seed"],
        eta=C.unbits(p["eta"]),min_weight=C.unbits(p["min_weight"]),lambda=C.unbits(p["lambda"]),L2=C.unbits(p["L2"]))
    need(Vector{UInt8}(codeunits(result.json))==bytes,"fit recipe codec reconstruction differs");result
end
function _fit_clock(value)
    need(value isa AbstractString && occursin(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z\z",value),"fit context clock syntax")
    parsed=try;DateTime(value[1:end-1]);catch;throw(ArgumentError("fit context clock value"));end
    need(clock(parsed)==value,"fit context canonical clock");parsed
end
function _check_capsule(value::BoundFit)
    expected=value.expected
    context_bytes=Vector{UInt8}(codeunits(value.context));recipe_bytes=Vector{UInt8}(codeunits(value.recipe))
    receipt_bytes=Vector{UInt8}(codeunits(value.receipt));encoded=model_bytes(value.model)
    context=R.decode_canonical(context_bytes;limit=131072);recipe=_recipe_bytes(recipe_bytes)
    receipt=R.decode_canonical(receipt_bytes;limit=65536)
    need(context isa AbstractDict && receipt isa AbstractDict,"fit capsule metadata objects required")
    actual=get(context,"evidence_mode",nothing)=="actual_archival_cache"
    context_fields=Set(("schema","evidence_mode","case_request_sha256","case_manifest_sha256","case_complete_sha256",
        "table_index","table","role","layout","start_inclusive","stop_exclusive","target_asof","label_ready_before",
        "rows","first_row","last_row","input_sha256","target_transform","recipe_sha256","source_sha256","library_sha256","runtime"))
    actual && push!(context_fields,"case_read_admission_sha256")
    J._object(context,context_fields)
    need(context["schema"]==(actual ? "source_bound_archival_fit_context_v1" : "source_bound_generated_fit_context_v1") &&
        context["evidence_mode"]==(actual ? "actual_archival_cache" : "generated"),"fit context evidence schema")
    need(digest(context_bytes)==expected.context_sha256 && digest(recipe_bytes)==expected.recipe_sha256==recipe.sha256 &&
        context["recipe_sha256"]==recipe.sha256 && context["source_sha256"]==expected.source_sha256 &&
        context["library_sha256"]==expected.library_sha256==LIBRARY_SHA,"fit capsule expected binding differs")
    for key in ("case_request_sha256","case_manifest_sha256","case_complete_sha256","input_sha256");J._sha(context[key]);end
    need(context["role"] in ("primary","independent") && context["layout"] in ("ground-only","mixed") &&
        context["runtime"] in ("1.12.6","1.13.1") && context["target_transform"]=="log1p_rate_reference_1_nt_per_min",
        "fit context native or target convention")
    need(isint(context["table_index"]) && 1<=context["table_index"]<=16 && isint(context["rows"]) &&
        2<=context["rows"]<=600000 && isint(context["first_row"]) && isint(context["last_row"]) &&
        1<=context["first_row"]<=context["last_row"]<=600000 && context["rows"]<=context["last_row"]-context["first_row"]+1,
        "fit context row geometry")
    table=context["table"];J._object(table,Set(("id","station","start_inclusive","stop_exclusive","label_asof")))
    J._text(table["id"];limit=128);need(table["station"] in ("CMO","FRD"),"fit context station")
    table_start=J._clock(table["start_inclusive"]);table_stop=J._clock(table["stop_exclusive"]);table_asof=J._clock(table["label_asof"])
    start=_fit_clock(context["start_inclusive"]);stop=_fit_clock(context["stop_exclusive"]);asof=_fit_clock(context["target_asof"])
    need(table_start<=start<stop<=table_stop && stop<=asof<=table_asof && context["label_ready_before"]==context["stop_exclusive"],
        "fit context split or readiness bounds")
    actual && need(DateTime(2017)<=start<stop<=DateTime(2018),"archival published fit must retain 2017 split")
    need(value.model.mode===(context["layout"]=="ground-only" ? :ground11 : :mixed8),"fit capsule model layout")
    if recipe isa BoostRecipe
        need(value.model isa P.NumericModel && value.model.depth==recipe.settings.max_depth &&
            value.model.rounds==recipe.settings.nrounds && value.model.bins==recipe.settings.nbins,"fit capsule EvoTrees recipe")
    else
        need(value.model isa C.NumericModel && value.model.recipe.json==recipe.json,"fit capsule retained recipe")
    end
    fields=Set(("schema","complete","family","context_sha256","recipe_sha256","source_sha256","library_sha256",
        "input_sha256","model_sha256","training_rows","all_training_native_predictions_exact",
        "actual_source_access","actual_fit_performed","software_qualification"))
    actual && push!(fields,"authorization");J._object(receipt,fields)
    need(receipt["schema"]=="source_bound_point_fit_receipt_v1" && receipt["complete"]===true &&
        receipt["family"]==family(recipe) && receipt["context_sha256"]==expected.context_sha256 &&
        receipt["recipe_sha256"]==expected.recipe_sha256 && receipt["source_sha256"]==expected.source_sha256 &&
        receipt["library_sha256"]==expected.library_sha256 && receipt["input_sha256"]==context["input_sha256"] &&
        receipt["model_sha256"]==digest(encoded) && isint(receipt["training_rows"]) && receipt["training_rows"]==context["rows"] &&
        receipt["all_training_native_predictions_exact"]===true && receipt["actual_source_access"]===actual &&
        receipt["actual_fit_performed"]===actual && receipt["software_qualification"]===false,"fit receipt binding differs")
    if actual
        authorization=receipt["authorization"]
        J._object(authorization,Set(("schema","operation","head","grant_sha256","source_proof_sha256","qualification_sha256",
            "case_read_admission_sha256","input_sha256","actual_model_fit_authorized","expected_model_fits",
            "dataset_discovery","provider_receipts_available","fit_attempt_sha256")))
        need(authorization["schema"]=="archive_point_fit_admission_v1" && authorization["operation"]=="fixed_recipe_2017_point_fit_only" &&
            authorization["actual_model_fit_authorized"]===true && isint(authorization["expected_model_fits"]) &&
            authorization["expected_model_fits"]==1 && authorization["dataset_discovery"]===false &&
            authorization["provider_receipts_available"]===false && authorization["input_sha256"]==context["input_sha256"] &&
            authorization["case_read_admission_sha256"]==J._sha(context["case_read_admission_sha256"]),"published archival fit admission differs")
        need(authorization["head"] isa String && occursin(r"\A[0-9a-f]{40}\z",authorization["head"]),"published fit source head")
        for key in ("grant_sha256","source_proof_sha256","qualification_sha256","fit_attempt_sha256");J._sha(authorization[key]);end
    end
    need(source_identity()==expected.source_sha256,"fit prediction loaded source differs")
    (;context_bytes,recipe_bytes,model_bytes=encoded,receipt_bytes,context,receipt,actual)
end
"""Publish one source-bound model to a fresh ordinary five-file directory.

The completion marker is written last. Existing paths are never overwritten;
partial failures remain available for diagnosis. Publication grants no new fit.
"""
function publish_fit(value::BoundFit,directory)
    checked=_check_capsule(value)
    need(directory isa AbstractString && !isempty(directory),"fit publication path")
    path=abspath(directory);J._ordinary(dirname(path);directory=true)
    need(!ispath(path) && !islink(path),"fit publication destination must be fresh")
    mkdir(path);J._ordinary(path;directory=true)
    payloads=(checked.context_bytes,checked.recipe_bytes,checked.model_bytes,checked.receipt_bytes)
    for (name,bytes,limit) in zip(_FIT_FILES[1:4],payloads,_FIT_LIMITS[1:4])
        need(length(bytes)<=limit,"fit publication member bound");R._exclusive_write(joinpath(path,name),bytes)
    end
    marker=R.canonical_bytes(Dict("schema"=>"source_bound_point_fit_complete_v1",
        "context_sha256"=>digest(payloads[1]),"recipe_sha256"=>digest(payloads[2]),
        "model_sha256"=>digest(payloads[3]),"receipt_sha256"=>digest(payloads[4]),
        "members"=>5,"bytes_without_marker"=>sum(length,payloads),"actual_fit_performed"=>checked.actual);limit=4096)
    _check_capsule(value)
    R._exclusive_write(joinpath(path,"COMPLETE.json"),marker)
    (;context_sha256=digest(payloads[1]),recipe_sha256=digest(payloads[2]),model_sha256=digest(payloads[3]),
      receipt_sha256=digest(payloads[4]),complete_sha256=digest(marker),actual_fit_performed=checked.actual)
end
"""Read a complete data-only fit with outside context, model and receipt pins.

Every member is canonical and bounded. The loaded numerical source must match
the fit's expected source closure; an artifact supplies no permission to refit.
"""
function read_published(directory;expected::ExpectedFit,model_sha256,receipt_sha256,complete_sha256)
    J._ordinary(directory;directory=true);names=readdir(directory)
    need(Set(names)==Set(_FIT_FILES) && length(names)==5,"fit publication fileset differs")
    payloads=Vector{UInt8}[];identities=Any[]
    for (name,limit) in zip(_FIT_FILES,_FIT_LIMITS)
        text,identity=FS._read(joinpath(directory,name),limit)
        push!(payloads,Vector{UInt8}(codeunits(text)));push!(identities,identity)
    end
    pins=(expected.context_sha256,expected.recipe_sha256,J._sha(model_sha256),J._sha(receipt_sha256),J._sha(complete_sha256))
    need(all(digest(bytes)==pin for (bytes,pin) in zip(payloads,pins)),"fit publication external pin differs")
    marker=R.decode_canonical(payloads[5];limit=4096)
    J._object(marker,Set(("schema","context_sha256","recipe_sha256","model_sha256","receipt_sha256","members","bytes_without_marker","actual_fit_performed")))
    need(marker["schema"]=="source_bound_point_fit_complete_v1" && isint(marker["members"]) && marker["members"]==5 &&
        isint(marker["bytes_without_marker"]) && marker["bytes_without_marker"]==sum(length,payloads[1:4]) &&
        all(marker[key]==pin for (key,pin) in zip(("context_sha256","recipe_sha256","model_sha256","receipt_sha256"),pins[1:4])),
        "fit publication completion binding differs")
    recipe=_recipe_bytes(payloads[2]);model=recipe isa BoostRecipe ? P.decode_model(payloads[3]) : C.decode_model(payloads[3])
    value=BoundFit(model,expected,String(copy(payloads[1])),String(copy(payloads[2])),String(copy(payloads[4])),_FitToken())
    checked=_check_capsule(value)
    need(marker["actual_fit_performed"]===checked.actual,"fit publication evidence flag differs")
    need(readdir(directory)==names && all(J._identity(joinpath(directory,name))==identity for (name,identity) in zip(_FIT_FILES,identities)),
        "fit publication changed during read")
    value
end
