"""Fixed descriptor-only worker; its Python owner admits resources before Julia.

Only pinned JSON headers are read. No CaseReader, case shards, target body,
model binary, ExpectedFit, source admission or fitting operation is invoked.
"""
module CandidateDescriptorNativePreview
using SolarSINDy,Dates,SHA
include("candidate_validation.jl")
const N=SourceBoundCandidateValidation
const R=N.R
const J=N.J
need(ok,message)=ok || throw(ArgumentError(message))
function object(value,names)
    J._object(value,Set(names));value
end
function metadata(ref;limit=131072)
    object(ref,("path","sha256"));J._ordinary(ref["path"])
    bytes=R.bounded_read(ref["path"];limit)
    need(N.digest(bytes)==J._sha(ref["sha256"]),"preview metadata outside SHA")
    R.decode_canonical(bytes;limit)
end
function authority(value)
    names=String.(fieldnames(N.InputPreparationAuthority));object(value,names)
    N.InputPreparationAuthority((String(value[name]) for name in names)...)
end
function preview_known_metadata(value)
    object(value,("stage_key","stage_operation","candidate_id","table_index","station","case_request","manifest",
        "complete","admission","point","selection_sha256","selection"))
    request=metadata(value["case_request"]);manifest=metadata(value["manifest"];limit=1048576)
    complete=metadata(value["complete"];limit=4096);admission=metadata(value["admission"])
    need(request["schema"]=="archive_catalog_case_request_v1" && request["evidence_mode"]=="actual_archival_cache" &&
        request["layout"]=="ground-only" && request["runtime"]==string(VERSION),"preview known case header")
    index=value["table_index"]
    need(N.F.isint(index) && 1<=index<=length(request["tables"])<=16,"preview table index")
    table=request["tables"][index]
    need(table["station"]==value["station"] && manifest["schema"]=="archive_catalog_case_artifact_v1" &&
        manifest["request_sha256"]==value["case_request"]["sha256"] && manifest["layout"]==request["layout"] &&
        manifest["role"]==request["role"] && manifest["runtime"]==request["runtime"] &&
        manifest["tables"][index]["declaration"]==table && complete["schema"]=="archive_catalog_case_complete_v1" &&
        complete["manifest_sha256"]==value["manifest"]["sha256"],"preview pinned header/manifest/COMPLETE lineage")
    need(admission["schema"]=="archive_catalog_case_read_admission_v1" && admission["evidence_mode"]=="actual_archival_cache" &&
        admission["request_sha256"]==value["case_request"]["sha256"] &&
        admission["artifact_manifest_sha256"]==value["manifest"]["sha256"] &&
        admission["artifact_complete_sha256"]==value["complete"]["sha256"],"preview admission metadata lineage")
    op=value["stage_operation"]
    need(op in ("candidate-input","candidate-fit","candidate-validation","independent-verify","selected-refit-input","selected-refit"),"preview fixed stage")
    selected=op in ("selected-refit-input","selected-refit")
    scoring=op in ("candidate-validation","independent-verify")
    split=selected ? N.CandidateSplit(DateTime(2017),DateTime(2017,10,1),DateTime(2018)) :
        scoring ? N.CandidateSplit(DateTime(2017,7,1),DateTime(2017,10,1),DateTime(2018)) :
        N.CandidateSplit(DateTime(2017),DateTime(2017,7,1),DateTime(2018))
    operation=selected ? "selected_R2_2017_refit_input_only" : scoring ? "fixed_candidate_2017_validation_only" : "fixed_candidate_2017_fit_input_only"
    id=value["candidate_id"];recipe=N.recipe_for(id);point=nothing
    stage_parts=Base.split(value["stage_key"],'/')
    need(length(stage_parts)==3 && stage_parts[1]==value["station"] && stage_parts[3]==op &&
        stage_parts[2]==(selected ? String(recipe.kind) : id),"preview fixed stage/station/candidate binding")
    need(selected ? value["selection_sha256"]!==nothing : value["selection_sha256"]===nothing,"preview selected lineage presence")
    if selected
        selection=value["selection"];object(selection,("family","result","receipt"))
        result=metadata(selection["result"];limit=1048576);owner=metadata(selection["receipt"];limit=262144)
        choice=result["selection"]
        need(owner["schema"]=="candidate_native_stage_evidence_v1" && owner["complete"]===true && owner["operation"]=="select-R2" &&
            owner["internal_result_sha256"]==selection["result"]["sha256"] && owner["outputs"]==result && result["complete"]===true &&
            result["operation"]=="select-R2" && choice["station"]==value["station"] &&
            choice["selection_evidence_sha256"]==value["selection_sha256"] && selection["family"]==String(recipe.kind),
            "preview finished selection metadata binding")
        winner=only(filter(x->x["family"]==selection["family"],choice["family_winners"]))
        need(winner["candidate_id"]==id,"preview finished family winner differs")
    else
        need(value["selection"]===nothing,"preview unused selection metadata")
    end
    if scoring
        p=value["point"];object(p,("descriptor","context","receipt"));point=p["descriptor"]
        object(point,("context_sha256","recipe_sha256","model_sha256","receipt_sha256","source_sha256"))
        context=metadata(p["context"]);receipt=metadata(p["receipt"];limit=65536)
        need(p["context"]["sha256"]==point["context_sha256"] && p["receipt"]["sha256"]==point["receipt_sha256"] &&
            context["case_request_sha256"]==value["case_request"]["sha256"] && context["table_index"]==index &&
            context["table"]["station"]==value["station"] && context["recipe_sha256"]==recipe.sha256 &&
            context["start_inclusive"]==N.F.clock(DateTime(2017)) && context["stop_exclusive"]==N.F.clock(DateTime(2017,7,1)) &&
            N.F.isint(context["rows"]) && context["rows"]>=1000 && receipt["actual_fit_performed"]===true &&
            receipt["context_sha256"]==point["context_sha256"] && receipt["model_sha256"]==point["model_sha256"] &&
            point["source_sha256"]==context["source_sha256"]==receipt["source_sha256"]==N.F.source_identity() &&
            receipt["recipe_sha256"]==point["recipe_sha256"]==context["recipe_sha256"] &&
            receipt["actual_source_access"]===true,
            "preview completed point metadata binding")
    else
        need(value["point"]===nothing,"preview unused point metadata")
    end
    facts=Dict{String,Any}("operation"=>operation,"case_request_sha256"=>value["case_request"]["sha256"],
        "case_manifest_sha256"=>value["manifest"]["sha256"],"case_complete_sha256"=>value["complete"]["sha256"],
        "table_index"=>Int(index),"station"=>table["station"],"role"=>request["role"],"candidate_id"=>String(id),
        "recipe_sha256"=>recipe.sha256,"grid_sha256"=>N.M.grid_identity(:R2),"split"=>N.split_object(split),"runtime"=>string(VERSION),
        "policy_sha256"=>N.policy_identity(),"job_sha256"=>N.job_identity(),"point_source_sha256"=>N.F.source_identity(),
        "actual_source_access"=>true,"case_read_admission_sha256"=>value["admission"]["sha256"],
        "selection_sha256"=>value["selection_sha256"],"point"=>point)
    descriptor=N.descriptor_from_metadata(facts)
    Dict("schema"=>"candidate_descriptor_preview_data_v1","descriptor"=>descriptor,
        "descriptor_sha256"=>N.digest(N.canonical(descriptor)),"descriptor_only"=>true,"metadata_unverified"=>true,
        "authority_granted"=>false,"actual_source_bodies_read"=>false,"actual_fits"=>0)
end
function main(args)
    need(length(args)==6 && args[1]=="--request" && args[3]=="--request-sha256" && args[5]=="--argv","fixed preview worker arguments")
    path=args[2];bytes=R.bounded_read(path;limit=1048576);need(N.digest(bytes)==J._sha(args[4]),"preview request outside SHA")
    request=R.decode_canonical(bytes;limit=1048576)
    object(request,("schema","campaign","stage_key","execution_directory","metadata"))
    need(request["schema"]=="candidate_descriptor_preview_request_v1" && abspath(PROGRAM_FILE)==abspath(@__FILE__),"fixed preview entry")
    need(request["stage_key"]==request["metadata"]["stage_key"],"preview outer/inner stage differs")
    config=metadata(request["campaign"];limit=1048576);q=authority(config["preview_authority"])
    grant=N.A._pinned(q.grant_path,q.grant_sha256;canonical=true,limit=131072);N._whole_qualification(q,grant)
    argv=R.decode_canonical(R.bounded_read(args[6];limit=65536);limit=65536);runtime=R._inspect_runtime(argv)
    result=preview_known_metadata(request["metadata"])
    need(R.bounded_read(path;limit=1048576)==bytes,"preview request changed")
    result["request_sha256"]=args[4];result["runtime"]=runtime;result["job_sha256"]=N.job_identity()
    R._exclusive_write(joinpath(request["execution_directory"],"PREVIEW_RESULT.json"),R.canonical_bytes(result;limit=131072))
    println("completed descriptor-only metadata preview; no source-body reads or fits")
end
end
if abspath(PROGRAM_FILE)==(@__FILE__);CandidateDescriptorNativePreview.main(ARGS);end
