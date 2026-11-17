"""Offline completed-source candidate fitting and independent evaluation."""
module CandidateNativeStage
using SolarSINDy,Dates,SHA,LinearAlgebra
include("candidate_validation.jl")
const N=SourceBoundCandidateValidation
const F=N.F;const B=N.B;const J=N.J;const R=N.R
const OPERATIONS=("candidate-input","candidate-fit","candidate-validation","independent-verify","select-R2","selected-refit-input","selected-refit","campaign")
need(ok,message)=ok || throw(ArgumentError(message))
object(value,names)=J._object(value,Set(names))
kwargs(value)=(; (Symbol(k)=>v for (k,v) in value)...)
pins_data(p)=Dict(String(k)=>v for (k,v) in pairs(p))
expected_data(e)=Dict("context_sha256"=>e.context_sha256,"recipe_sha256"=>e.recipe_sha256,"source_sha256"=>e.source_sha256,"library_sha256"=>e.library_sha256)
fit_data(p)=Dict("directory"=>p.directory,"expected"=>expected_data(p.point.expected),"pins"=>pins_data(p.pins))
function fit_input(value)
    object(value,("directory","pins"));N.read_fit_input(value["directory"];kwargs(value["pins"])...)
end
function fit_publication(value)
    object(value,("directory","expected","pins"));e=value["expected"]
    object(e,("context_sha256","recipe_sha256","source_sha256","library_sha256"))
    expected=F.ExpectedFit(e["context_sha256"],e["recipe_sha256"],e["source_sha256"],e["library_sha256"])
    p=value["pins"];object(p,("context_sha256","recipe_sha256","model_sha256","receipt_sha256","complete_sha256","actual_fit_performed"))
    need(p["context_sha256"]==expected.context_sha256 && p["recipe_sha256"]==expected.recipe_sha256 && p["actual_fit_performed"]===true,
        "stage fit outside pins differ")
    point=F.read_published(value["directory"];expected,model_sha256=p["model_sha256"],receipt_sha256=p["receipt_sha256"],complete_sha256=p["complete_sha256"])
    need(F._check_capsule(point).actual,"actual fit artifact required")
    N.FitPublication(point,value["directory"],kwargs(p))
end
function validation_pins(value)
    names=("request_sha256","context_sha256","metric_sha256","receipt_sha256","manifest_sha256","complete_sha256")
    object(value,names);N.ValidationPins((J._sha(value[name]) for name in names)...)
end
function case_reader(value)
    object(value,("request_path","directory","request_sha256","manifest_sha256","complete_sha256"))
    read=B.read_reverified(value["request_path"],value["directory"];request_sha256=value["request_sha256"],
        manifest_sha256=value["manifest_sha256"],complete_sha256=value["complete_sha256"])
    need(read.cases.actual_source_authority_granted,"actual case artifact required");read
end
function checked_selection(cases,index,case_ref,rows)
    need(rows isa AbstractVector && length(rows)==16,"stage requires complete fixed16 candidates")
    checked=N.VerifiedCandidate[];score=N.CandidateSplit(DateTime(2017,7,1),DateTime(2017,10,1),DateTime(2018))
    for (spec,row) in zip(N.S.POINT_GRID,rows)
        object(row,("candidate_id","fit","validation","witness"));need(row["candidate_id"]==spec.id,"stage candidate grid/order differs")
        fit=fit_publication(row["fit"]);v=row["validation"];object(v,("directory","pins"));w=row["witness"];object(w,("directory","pins"))
        candidate=N.read_completed_candidate(cases,index,spec.id,score,fit,v["directory"],validation_pins(v["pins"]),w["directory"],kwargs(w["pins"]);
            request_path=case_ref["request_path"],case_directory=case_ref["directory"])
        push!(checked,candidate)
    end
    N.choose_verified_R2(checked)
end
function selection_data(selection)
    d=selection.decision
    Dict("schema"=>"completed_source_R2_selection_v2","selection_evidence_sha256"=>selection.evidence_sha256,
        "decision_sha256"=>d.decision_sha256,"station"=>d.station,"primary_candidate_id"=>d.primary_candidate_id,
        "primary_family"=>String(d.primary_family),"family_winners"=>[Dict("family"=>String(w.family),"candidate_id"=>w.candidate_id,
        "loss_bits"=>N.C.bits(w.selection_log_mse)) for w in d.family_winners])
end
function run_campaign(cases,index,root)
    train=N.CandidateSplit(DateTime(2017),DateTime(2017,7,1),DateTime(2018))
    score=N.CandidateSplit(DateTime(2017,7,1),DateTime(2017,10,1),DateTime(2018))
    refit=N.CandidateSplit(DateTime(2017),DateTime(2017,10,1),DateTime(2018))
    verified=N.VerifiedCandidate[];rows=Any[]
    for spec in N.S.POINT_GRID
        inputdir=joinpath(root,spec.id*"-input");p=N.prepare_candidate_fit_input_archival(cases,index,spec.id,train,inputdir)
        fit=N.fit_candidate_archival(cases,index,spec.id,train,p.input;directory=joinpath(root,spec.id*"-point"))
        scored=N.validate_candidate_archival(cases,index,spec.id,score,fit.point,joinpath(root,spec.id*"-validation"))
        witness=joinpath(root,spec.id*"-independent")
        candidate=N.verify_candidate_validation_archival(cases,index,spec.id,score,fit,scored.artifact.directory,scored.pins;witness_directory=witness)
        push!(verified,candidate)
        push!(rows,Dict("candidate_id"=>spec.id,"fit"=>fit_data(fit),"validation"=>Dict("directory"=>scored.artifact.directory,"pins"=>pins_data(scored.pins)),
            "witness"=>Dict("directory"=>witness,"pins"=>Dict("manifest_sha256"=>N.digest(read(joinpath(witness,"manifest.json"))),
                "complete_sha256"=>N.digest(read(joinpath(witness,"COMPLETE.json")))))))
    end
    selection=N.choose_verified_R2(verified);refits=Any[]
    for winner in selection.decision.family_winners
        family=winner.family;input=N.prepare_selected_refit_input_archival(selection,cases,index,family,refit,joinpath(root,String(family)*"-refit-input"))
        fitted=N.refit_selected_archival(selection,cases,index,family,refit,input.input;directory=joinpath(root,String(family)*"-selected-refit"))
        push!(refits,Dict("family"=>String(family),"candidate_id"=>fitted.candidate_id,"fit"=>fit_data(fitted.fit),
            "binding"=>N.decode(Vector{UInt8}(codeunits(fitted.binding_json)))))
    end
    Dict("candidates"=>rows,"selection"=>selection_data(selection),"selected_refits"=>refits,"actual_fits"=>19)
end
function dispatch(request)
    op=request["operation"];need(op in OPERATIONS,"unsupported candidate operation")
    root=request["execution_directory"];J._ordinary(root;directory=true);x=request["inputs"]
    cases=case_reader(request["case"]);index=request["table_index"];N.case_metadata(cases.cases,index;actual=true)
    train=N.CandidateSplit(DateTime(2017),DateTime(2017,7,1),DateTime(2018))
    score=N.CandidateSplit(DateTime(2017,7,1),DateTime(2017,10,1),DateTime(2018))
    refit=N.CandidateSplit(DateTime(2017),DateTime(2017,10,1),DateTime(2018));output=joinpath(root,"output");id=request["candidate_id"]
    result=if op=="campaign"
        object(x,());need(id===nothing,"campaign candidate id must be null");run_campaign(cases,index,root)
    elseif op=="candidate-input"
        object(x,());p=N.prepare_candidate_fit_input_archival(cases,index,id,train,output)
        Dict("input"=>Dict("directory"=>output,"pins"=>pins_data(p.pins)),"expected_fit"=>expected_data(N.expected_candidate_fit(p.input)))
    elseif op=="candidate-fit"
        object(x,("input",));p=N.fit_candidate_archival(cases,index,id,train,fit_input(x["input"]);directory=output)
        Dict("fit"=>fit_data(p))
    elseif op=="candidate-validation"
        object(x,("fit",));fit=fit_publication(x["fit"]);p=N.validate_candidate_archival(cases,index,id,score,fit.point,output)
        Dict("validation"=>Dict("directory"=>output,"pins"=>pins_data(p.pins)))
    elseif op=="independent-verify"
        object(x,("fit","validation"));fit=fit_publication(x["fit"]);v=x["validation"];object(v,("directory","pins"))
        witness=joinpath(root,"independent-verification")
        candidate=N.verify_candidate_validation_archival(cases,index,id,score,fit,v["directory"],validation_pins(v["pins"]);witness_directory=witness)
        Dict("witness"=>Dict("directory"=>witness,"pins"=>Dict("manifest_sha256"=>N.digest(read(joinpath(witness,"manifest.json"))),
            "complete_sha256"=>N.digest(read(joinpath(witness,"COMPLETE.json"))))),"witness_sha256"=>candidate.witness_sha256)
    else
        object(x,op=="select-R2" ? ("candidates",) : op=="selected-refit-input" ? ("candidates","family") : ("candidates","family","input"))
        selection=checked_selection(cases,index,request["case"],x["candidates"])
        if op=="select-R2"
            Dict("selection"=>selection_data(selection))
        elseif op=="selected-refit-input"
            p=N.prepare_selected_refit_input_archival(selection,cases,index,Symbol(x["family"]),refit,output)
            Dict("selection"=>selection_data(selection),"input"=>Dict("directory"=>output,"pins"=>pins_data(p.pins)))
        else
            fitted=N.refit_selected_archival(selection,cases,index,Symbol(x["family"]),refit,fit_input(x["input"]);directory=output)
            Dict("selection"=>selection_data(selection),"fit"=>fit_data(fitted.fit),"binding"=>N.decode(Vector{UInt8}(codeunits(fitted.binding_json))))
        end
    end
    merge(result,Dict("actual_source_access"=>true,"actual_fits"=>get(result,"actual_fits",op in ("candidate-fit","selected-refit") ? 1 : 0)))
end
function main(args)
    need(length(args)==4 && args[1]=="--request" && args[3]=="--request-sha256","candidate worker requires --request PATH --request-sha256 SHA")
    path=args[2];pin=J._sha(args[4]);J._ordinary(path);bytes=R.bounded_read(path;limit=1048576)
    need(N.digest(bytes)==pin,"candidate worker request outside pin");request=N.decode(bytes;limit=1048576)
    object(request,("schema","evidence_mode","operation","execution_directory","case","table_index","candidate_id","inputs"))
    need(request["schema"]=="completed_source_candidate_request_v2" && request["evidence_mode"]=="actual_archival_cache","ordinary actual candidate request required")
    need(BLAS.get_num_threads()==1,"candidate operations require one BLAS thread")
    started=time_ns();observed=@timed dispatch(request);result=observed.value
    need(R.bounded_read(path;limit=1048576)==bytes,"candidate request changed")
    result["schema"]="completed_source_candidate_result_v2";result["complete"]=true;result["operation"]=request["operation"]
    result["request_sha256"]=pin;result["job_sha256"]=N.job_identity()
    result["runtime_observations"]=Dict("julia_version"=>string(VERSION),"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
        "elapsed_ns"=>Int(time_ns()-started),"allocated_bytes"=>observed.bytes,"available_physical_bytes"=>Sys.free_memory(),
        "peak_rss_bytes"=>Int(Sys.maxrss()))
    R._exclusive_write(joinpath(request["execution_directory"],"JULIA_STAGE_RESULT.json"),N.canonical(result;limit=1048576))
    println("completed candidate operation ",request["operation"])
end
end
if abspath(PROGRAM_FILE)==(@__FILE__);CandidateNativeStage.main(ARGS);end
