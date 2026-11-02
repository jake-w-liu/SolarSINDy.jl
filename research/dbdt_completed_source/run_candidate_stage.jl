"""Fixed native stage dispatcher. No caller code, Serialization or grant creation."""
module CandidateNativeStage
using SolarSINDy,Dates,SHA
include("candidate_validation.jl")
const N=SourceBoundCandidateValidation
const R=N.R
const F=N.F
const A=N.A
const B=N.B
const J=N.J
const OPERATIONS=("inspect-runtime","generated-chain","candidate-input","candidate-fit","candidate-validation",
    "independent-verify","select-R2","selected-refit-input","selected-refit")
need(ok,message)=ok || throw(ArgumentError(message))
function object(value,names)
    J._object(value,Set(names));value
end
kwargs(value)=(; (Symbol(k)=>v for (k,v) in value)...)
function authority(::Type{T},value) where T
    object(value,String.(fieldnames(T)))
    T((String(value[String(name)]) for name in fieldnames(T))...)
end
function native(value)
    object(value,("receipt_path","receipt_sha256","raw_path","raw_sha256"))
    N.NativeEvidence(value["receipt_path"],J._sha(value["receipt_sha256"]),value["raw_path"],J._sha(value["raw_sha256"]))
end
function fit_input(value)
    object(value,("directory","pins"));N.read_fit_input(value["directory"];kwargs(value["pins"])...)
end
function fit_publication(value)
    object(value,("directory","expected","pins"));e=value["expected"]
    object(e,("context_sha256","recipe_sha256","source_sha256","library_sha256"))
    expected=F.ExpectedFit(e["context_sha256"],e["recipe_sha256"],e["source_sha256"],e["library_sha256"])
    p=value["pins"];object(p,("context_sha256","recipe_sha256","model_sha256","receipt_sha256","complete_sha256","actual_fit_performed"))
    need(p["context_sha256"]==expected.context_sha256 && p["recipe_sha256"]==expected.recipe_sha256,"stage fit outside pins differ")
    point=F.read_published(value["directory"];expected,model_sha256=p["model_sha256"],receipt_sha256=p["receipt_sha256"],complete_sha256=p["complete_sha256"])
    need(F._check_capsule(point).actual===p["actual_fit_performed"],"stage fit mode differs")
    N.FitPublication(point,value["directory"],kwargs(p))
end
function validation_pins(value)
    object(value,("request_sha256","context_sha256","metric_sha256","receipt_sha256","manifest_sha256","complete_sha256"))
    N.ValidationPins((value[name] for name in ("request_sha256","context_sha256","metric_sha256","receipt_sha256","manifest_sha256","complete_sha256"))...)
end
function case_reader(value)
    object(value,("request_path","directory","request_sha256","manifest_sha256","complete_sha256","producer","reader","admission_sha256"))
    producer=authority(A.Authority,value["producer"]);reader=authority(B.ReaderAuthority,value["reader"])
    read=B.read_reverified(value["request_path"],value["directory"];request_sha256=value["request_sha256"],
        manifest_sha256=value["manifest_sha256"],complete_sha256=value["complete_sha256"],producer,reader)
    need(read.cases.actual_source_authority_granted && N.digest(B.admission_bytes(read))==value["admission_sha256"],"stage existing case reader admission differs")
    read
end
function checked_selection(cases,index,case_ref,rows)
    need(rows isa AbstractVector && length(rows)==16,"stage requires complete fixed16 candidates")
    checked=N.VerifiedCandidate[];owners=N.NativeEvidence[]
    for (spec,row) in zip(N.S.POINT_GRID,rows)
        object(row,("candidate_id","fit","fit_authority","fit_native","validation","validation_authority","validation_native",
            "verification_authority","verification_native","witness"))
        need(row["candidate_id"]==spec.id,"stage candidate grid/order differs")
        fit=fit_publication(row["fit"]);v=row["validation"];object(v,("directory","pins"));w=row["witness"];object(w,("directory","pins"))
        p=validation_pins(v["pins"]);owner=native(row["verification_native"])
        candidate=N.read_completed_candidate(cases,index,spec.id,N.CandidateSplit(DateTime(2017,7,1),DateTime(2017,10,1),DateTime(2018)),
            fit,v["directory"],p,w["directory"],kwargs(w["pins"]);request_path=case_ref["request_path"],case_directory=case_ref["directory"],
            verification_authority=authority(N.IndependentReadAuthority,row["verification_authority"]),verification_native=owner,
            producer=authority(F.FitAuthority,row["fit_authority"]),native=native(row["fit_native"]),
            validation_producer=authority(N.CandidateValidationAuthority,row["validation_authority"]),validation_native=native(row["validation_native"]))
        push!(checked,candidate);push!(owners,owner)
    end
    N.choose_verified_R2(checked;independent_native=owners)
end
expected_data(e)=Dict("context_sha256"=>e.context_sha256,"recipe_sha256"=>e.recipe_sha256,"source_sha256"=>e.source_sha256,"library_sha256"=>e.library_sha256)
pins_data(p)=Dict(String(k)=>v for (k,v) in pairs(p))
function selection_data(selection)
    d=selection.decision
    Dict("schema"=>"completed_native_R2_selection_data_v1","selection_evidence_sha256"=>selection.evidence_sha256,
        "decision_sha256"=>d.decision_sha256,"station"=>d.station,"primary_candidate_id"=>d.primary_candidate_id,
        "primary_family"=>String(d.primary_family),"family_winners"=>[Dict("family"=>String(w.family),"candidate_id"=>w.candidate_id,
        "loss_bits"=>N.C.bits(w.selection_log_mse)) for w in d.family_winners],
        "candidate_witnesses"=>[Dict("candidate_id"=>c.candidate_id,"witness_sha256"=>c.witness_sha256) for c in selection.candidates],
        "grants_new_fit"=>false)
end
function dispatch(request)
    op=request["operation"];need(op in OPERATIONS,"unsupported fixed native stage")
    root=request["execution_directory"];J._ordinary(root;directory=true);x=request["inputs"]
    if op=="inspect-runtime"
        need(request["evidence_mode"]=="generated" && isempty(x),"inspection accepts no source inputs")
        return Dict("actual_source_access"=>false,"actual_fits"=>0)
    elseif op=="generated-chain"
        need(request["evidence_mode"]=="generated" && isempty(x),"generated native control accepts no caller data")
        include(joinpath(pkgdir(SolarSINDy),"test","candidate_validation","generated_fixture.jl"))
        destination=joinpath(root,"generated");N._fresh_directory(destination)
        result=Base.invokelatest(CandidateValidationGeneratedFixture.run_chain,N,destination)
        need(length(result.stations)==2 && all(s->length(s.verified)==16&&length(s.refits)==3,result.stations),"generated native chain incomplete")
        return Dict("actual_source_access"=>false,"actual_fits"=>0,"generated_candidate_fits"=>32,"generated_selected_refits"=>6)
    end
    need(request["evidence_mode"]=="actual_archival_cache","actual operation needs actual request")
    # The fixed parent has already checked every supplied permit and current
    # whole-source qualification BEFORE this process starts. Recheck the current
    # loaded Julia closure before CaseReader recomputes any source body.
    kind=op in ("candidate-input","selected-refit-input") ? N.InputPreparationAuthority : op=="candidate-fit" ? F.FitAuthority :
        op=="candidate-validation" ? N.CandidateValidationAuthority : op=="selected-refit" ? N.SelectedRefitAuthority : N.IndependentReadAuthority
    q=authority(kind,request["qualification_authority"])
    grant=A._pinned(q.grant_path,q.grant_sha256;canonical=true,limit=131072);N._whole_qualification(q,grant)
    cases=case_reader(request["case"]);index=request["table_index"];N.case_metadata(cases.cases,index;actual=true)
    output=joinpath(root,"output");id=request["candidate_id"]
    train=N.CandidateSplit(DateTime(2017),DateTime(2017,7,1),DateTime(2018))
    score=N.CandidateSplit(DateTime(2017,7,1),DateTime(2017,10,1),DateTime(2018))
    refit=N.CandidateSplit(DateTime(2017),DateTime(2017,10,1),DateTime(2018))
    result=if op=="candidate-input"
        object(x,("authority",));a=authority(N.InputPreparationAuthority,x["authority"])
        p=N.prepare_candidate_fit_input_archival(cases,index,id,train,output;authority=a)
        Dict("input"=>Dict("directory"=>output,"pins"=>pins_data(p.pins)),"expected_fit"=>expected_data(N.expected_candidate_fit(p.input)))
    elseif op=="candidate-fit"
        object(x,("input","fit_authority"));a=authority(F.FitAuthority,x["fit_authority"])
        p=N.fit_candidate_archival(cases,index,id,train,fit_input(x["input"]);directory=output,authority=a)
        Dict("point"=>N.point_descriptor(p.point),"fit"=>Dict("directory"=>output,"expected"=>expected_data(p.point.expected),"pins"=>pins_data(p.pins)))
    elseif op=="candidate-validation"
        object(x,("fit","authority","fit_authority","fit_native"));fit=fit_publication(x["fit"])
        p=N.validate_candidate_archival(cases,index,id,score,fit.point,output;authority=authority(N.CandidateValidationAuthority,x["authority"]),
            producer=authority(F.FitAuthority,x["fit_authority"]),native=native(x["fit_native"]))
        Dict("validation"=>Dict("directory"=>output,"pins"=>pins_data(p.pins)),"output_context_sha256"=>p.pins.context_sha256,
            "output_complete_sha256"=>p.pins.complete_sha256)
    elseif op=="independent-verify"
        object(x,("fit","validation","authority","fit_authority","fit_native","validation_authority","validation_native"))
        fit=fit_publication(x["fit"]);v=x["validation"];object(v,("directory","pins"));vp=validation_pins(v["pins"])
        candidate=N.verify_candidate_validation_archival(cases,index,id,score,fit,v["directory"],vp;
            authority=authority(N.IndependentReadAuthority,x["authority"]),producer=authority(F.FitAuthority,x["fit_authority"]),native=native(x["fit_native"]),
            validation_producer=authority(N.CandidateValidationAuthority,x["validation_authority"]),validation_native=native(x["validation_native"]))
        witness=joinpath(root,"independent-verification")
        p=Dict("manifest_sha256"=>N.digest(read(joinpath(witness,"manifest.json"))),"complete_sha256"=>N.digest(read(joinpath(witness,"COMPLETE.json"))))
        Dict("witness"=>Dict("directory"=>witness,"pins"=>p),"output_witness_sha256"=>candidate.witness_sha256,
            "output_manifest_sha256"=>p["manifest_sha256"],"output_complete_sha256"=>p["complete_sha256"],
            "input_descriptor_sha256"=>N.digest(codeunits(candidate.descriptor_json)))
    else
        allowed=op=="select-R2" ? ("candidates",) : op=="selected-refit-input" ? ("candidates","family","authority") :
            ("candidates","family","input","authority","fit_authority")
        object(x,allowed);selection=checked_selection(cases,index,request["case"],x["candidates"])
        if op=="select-R2"
            Dict("selection"=>selection_data(selection))
        elseif op=="selected-refit-input"
            family=Symbol(x["family"])
            p=N.prepare_selected_refit_input_archival(selection,cases,index,family,refit,output;authority=authority(N.InputPreparationAuthority,x["authority"]))
            Dict("selection"=>selection_data(selection),"input"=>Dict("directory"=>output,"pins"=>pins_data(p.pins)),
                "expected_fit"=>expected_data(N.expected_candidate_fit(p.input)))
        else
            family=Symbol(x["family"])
            p=N.refit_selected_archival(selection,cases,index,family,refit,fit_input(x["input"]);directory=output,
                authority=authority(N.SelectedRefitAuthority,x["authority"]),fit_authority=authority(F.FitAuthority,x["fit_authority"]))
            binding=Vector{UInt8}(codeunits(p.binding_json));R._exclusive_write(joinpath(root,"selected-binding.json"),binding)
            Dict("selection"=>selection_data(selection),"point"=>N.point_descriptor(p.fit.point),
                "fit"=>Dict("directory"=>output,"expected"=>expected_data(p.fit.point.expected),"pins"=>pins_data(p.fit.pins)),"selected_binding_sha256"=>N.digest(binding))
        end
    end
    merge(result,Dict("actual_source_access"=>true,"actual_fits"=>op in ("candidate-fit","selected-refit") ? 1 : 0,
        "case_reader_reconstruction_separate_from_split_budget"=>true,"case_scheduled_rows_reconstructed"=>sum(length,cases.cases.features)))
end
function main(args)
    need(length(args)==6 && args[1]=="--request" && args[3]=="--request-sha256" && args[5]=="--argv","fixed candidate worker arguments required")
    path=args[2];pin=J._sha(args[4]);bytes=R.bounded_read(path;limit=1048576);need(N.digest(bytes)==pin,"candidate worker request outside pin")
    request=R.decode_canonical(bytes;limit=1048576)
    object(request,("schema","evidence_mode","operation","execution_directory","source_proof","qualification","qualification_authority",
        "julia","runtime","case","table_index","candidate_id","inputs"))
    need(request["schema"]=="candidate_native_stage_request_v1","candidate worker request schema")
    argv=R.decode_canonical(R.bounded_read(args[6];limit=65536);limit=65536);runtime=R._inspect_runtime(argv)
    need(abspath(PROGRAM_FILE)==abspath(@__FILE__),"candidate worker entry mismatch")
    result=dispatch(request)
    need(R.bounded_read(path;limit=1048576)==bytes,"candidate worker request changed")
    result["schema"]="candidate_native_stage_internal_v1";result["complete"]=true;result["operation"]=request["operation"]
    result["request_sha256"]=pin;result["runtime"]=runtime;result["job_sha256"]=N.job_identity()
    R._exclusive_write(joinpath(request["execution_directory"],"JULIA_STAGE_RESULT.json"),R.canonical_bytes(result;limit=1048576))
    println("completed fixed candidate stage ",request["operation"])
end
end
if abspath(PROGRAM_FILE)==(@__FILE__);CandidateNativeStage.main(ARGS);end
