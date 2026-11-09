"""Fixed B1 actual entry and a separate, in-process generated correctness route."""
module BInitialNativeStage
using SolarSINDy,Dates,SHA,LinearAlgebra
include("ActualBChain.jl")
const B=ActualBChain
const F=B.BInitialPointFits
const T=SolarSINDy.DurationPointTraining
const P=SolarSINDy.DurationPointArtifacts
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const REPO=normpath(joinpath(@__DIR__,"..",".."))
need(ok,msg)=T._need(ok,msg)
function _assembled_result(stage,verified;actual,fit_elapsed_ns,readback_elapsed_ns,worker_elapsed_ns)
    need(verified.complete===true&&verified.model_count==36&&verified.certificate_count==36&&verified.hazard_fit_calls==0&&verified.gain_fit_calls==0&&verified.raw_replay_A_streams==9&&verified.selected_index===nothing,"B native strict readback scope differs")
    completion,bytes=P._json(joinpath(stage.directory,"completion.json");fields=P.COMPLETION_FIELDS)
    need(T._sha(bytes)==stage.completion_sha256&&stage.gain_fit_count==36&&completion.provisional_selection.selected_index===nothing,"B native completion/selection differs")
    for (k,v) in ((:initial_hazard_fit_count,2),(:initial_gain_fit_count,36),(:state_schedule_count,9),(:beta_design_count,18),(:final_fit_count,0),(:calibration_fit_count,0),(:output_file_count,423))
        need(T._int(getproperty(completion,k),0,typemax(Int64))==v,"B native fixed output count differs")
    end
    for k in (:exposed_development_scoring,:joint_primary_selection,:scientific_verdict,:serving_promotion);need(getproperty(completion,k)===false,"B native exceeds L0 scope");end
    rss=Sys.maxrss();need(rss isa Integer&&!(rss isa Bool)&&rss>=0,"native peak RSS measurement unavailable")
    Dict{String,Any}("schema"=>"b_initial_l0_internal_result_v2","complete"=>true,"evidence_mode"=>(actual ? "archival_assumed_completion_research" : "synthetic_only"),
        "actual_source_access"=>actual,"initial_hazard_fit_count"=>2,"initial_gain_fit_count"=>36,"state_schedule_count"=>9,"beta_design_count"=>18,"final_fit_count"=>0,"calibration_fit_count"=>0,
        "selected_index"=>nothing,"scientific_verdict"=>false,"serving_promotion"=>false,"stage_directory"=>stage.directory,"completion_sha256"=>stage.completion_sha256,
        "output_sha256"=>String(completion.output_sha256),"output_file_count"=>Int(completion.output_file_count),"output_directory_count"=>Int(completion.output_directory_count),
        "verified_model_count"=>verified.model_count,"verified_certificate_count"=>verified.certificate_count,"verified_hazard_fit_calls"=>verified.hazard_fit_calls,"verified_gain_fit_calls"=>verified.gain_fit_calls,"verified_raw_replay_A_streams"=>verified.raw_replay_A_streams,
        "fit_elapsed_ns"=>fit_elapsed_ns,"readback_elapsed_ns"=>readback_elapsed_ns,"worker_elapsed_ns"=>worker_elapsed_ns,"julia_peak_rss_bytes"=>rss,"julia_peak_rss_measured"=>true)
end
function run_generated(root::AbstractString)
    need(isabspath(root)&&isdir(root)&&P._ordinary(root)==realpath(root),"canonical owned generated root required")
    if !isdefined(@__MODULE__,:BPointSyntheticFixture)
        include(joinpath(REPO,"examples","support","b_point_generated_fixture.jl"))
    end
    # Use exactly the fixture's own nominal ActualBChain types and executor.
    G=BPointSyntheticFixture
    Base.invokelatest(G.with_single_blas, function ()
        worker=time_ns()
        admission,reference=Base.invokelatest(G.fixture,joinpath(root,"input"))
        start=time_ns();stage=Base.invokelatest(G.B.BInitialPointFits.run_initial_l0,admission,reference,joinpath(root,"stage");chunk_hours=7);fit_elapsed_ns=Int(time_ns()-start)
        start=time_ns();verified=Base.invokelatest(G.B.BInitialPointFits.verify_initial_l0,stage.directory,admission,reference;expected_completion_sha256=stage.completion_sha256,expected_sources=P.source_identity(),chunk_hours=2);readback_elapsed_ns=Int(time_ns()-start)
        result=_assembled_result(stage,verified;actual=false,fit_elapsed_ns,readback_elapsed_ns,worker_elapsed_ns=Int(time_ns()-worker))
        (;admission,reference,stage,verified,result)
    end)
end
function _reference_pins(ref)
    d=F._v2_canonical_ref(ref)
    names=(:completion_sha256,:audit_sha256,:grid_sha256,:tuning_sha256,:prediction_sha256,:feature_contract_sha256,:tuning_loss_contract_sha256,:baseline_qualification_sha256,:baseline_sources,:baseline_qualified_head,:feature_contract_member,:tuning_loss_contract_member)
    T._fields(d,names)
    sources=Pair{String,String}[String(row.name)=>String(row.sha256) for row in d.baseline_sources]
    B.BSelection.BReferencePins(; (name=>(name==:baseline_sources ? sources : String(getproperty(d,name))) for name in names)...)
end
function reconstruct_inputs_v2(input,reference_spec;expected_evidence_mode::Symbol,expected_input_prefix_sha256=nothing,expected_sources=P.source_identity())
    need(expected_evidence_mode in (:synthetic_only,:archival_assumed_completion_research),"B v2 expected evidence mode differs")
    T._fields(input,(:envelope,:roots,:independent_admission,:expected_view_id,:expected_admitted_seal))
    T._fields(input.envelope,(:path,:sha256));T._fields(input.independent_admission,(:path,:sha256))
    T._fields(reference_spec,(:completion_path,:roots,:pins))
    independent=(;path=String(input.independent_admission.path),sha256=String(input.independent_admission.sha256))
    rootmap=Dict(String(k)=>String(v) for (k,v) in pairs(input.roots))
    info=B.BInputs._joined_metadata_v2(input.envelope.path;expected_envelope_sha256=input.envelope.sha256,
        roots=rootmap,expected_sources,independent_admission=independent,expected_view_id=input.expected_view_id,
        expected_input_prefix_sha256,reference_pins=reference_spec.pins)
    trusted=_reference_pins(reference_spec.pins)
    reference_roots=B.BInputs._roots_v2(Dict(String(k)=>String(v) for (k,v) in pairs(reference_spec.roots)))
    completion=B.BInputs._ordinary_input_v2(reference_spec.completion_path)
    need(Symbol(info.data.evidence_mode)==expected_evidence_mode&&completion==B.BInputs._role_v2(info.members,"reference_completion").absolute&&dirname(completion) in values(reference_roots)&&
        trusted.completion_sha256==info.data.reference_completion_sha256&&trusted.audit_sha256==info.data.reference_audit_sha256,"B v2 supplied reference is not the joined reference")
    admission=B.BInputs.admit_prefix_v2(input.envelope.path;expected_envelope_sha256=input.envelope.sha256,
        roots=rootmap,expected_sources,independent_admission=independent,expected_view_id=input.expected_view_id,
        expected_input_prefix_sha256,reference_pins=reference_spec.pins)
    need(admission.view.evidence_mode==expected_evidence_mode&&admission.view.view_id==input.expected_view_id&&admission.admission_id==input.expected_admitted_seal,"B v2 reconstructed source/join differs")
    reference=B.BInputs.admit_reference(reference_spec.completion_path;trusted_pins=trusted,
        roots=reference_roots,view=admission.view,expected_sources)
    m=reference.metadata;a=admission.metadata
    need(m.evidence_mode==a.evidence_mode&&m.source_sha256==a.source_sha256&&m.delay_hours==a.delay_hours&&
        m.training_boundary==a.training_boundary&&m.validation_start==a.validation_start&&m.validation_end==a.validation_end&&
        m.training_history_start==a.prefix_start&&m.reference_kind==a.reference_kind&&
        m.reference_completion_sha256==a.reference_completion_sha256&&m.audit_completion_sha256==a.reference_audit_sha256&&
        reference.audit!==nothing&&reference.audit.view.view_id==admission.view.view_id,"B v2 admitted reference/join pair differs")
    B.BInputs.recheck(admission)
    (;admission,reference)
end
function main(args)
    worker=time_ns()
    need(length(args)==6&&args[1]=="--request"&&args[3]=="--request-sha256"&&args[5]=="--argv","fixed actual B1 worker argv required")
    path=args[2];sha=args[4];need(T._hex(sha)&&get(ENV,"SOLARSINDY_B_V2_REQUEST_PATH","")==path,"fixed B1 request environment differs")
    r,body=P._json(path;fields=F.OWNED_V2_REQUEST);need(T._sha(body)==sha,"B1 outside request pin differs")
    permit=F.OwnedInitialPointExecutionPermitV2(String(r.permit.path),String(r.permit.sha256),String(r.intent.path),String(r.intent.sha256))
    checked=F._v2_precheck(permit;argv_path=args[6]);intent=checked.intent;root=String(r.execution_directory)
    argv,_=P._json(args[6]);runtime=R._inspect_runtime(collect(argv));expected=F._v2_ref(r.runtime)
    need(all(x->x isa Integer&&!(x isa Bool),(expected.blas_threads,expected.omp_threads,expected.rng_seed)),"B1 inspected runtime integer grammar")
    need(Dict(k=>v for (k,v) in pairs(runtime) if k!="argv")==Dict(String(k)=>v for (k,v) in pairs(expected) if String(k)!="argv"),"B1 inspected runtime changed")
    # All supplied native authority/source/runtime checks above precede decoding.
    reconstructed=reconstruct_inputs_v2(intent.input,intent.reference;expected_evidence_mode=:archival_assumed_completion_research,expected_input_prefix_sha256=checked.expected_input_prefix_sha256)
    admission=reconstructed.admission;reference=reconstructed.reference
    output=joinpath(root,"output");start=time_ns()
    stage=F.run_initial_l0(admission,reference,output;execution_permit=permit,chunk_hours=24);fit_elapsed_ns=Int(time_ns()-start)
    start=time_ns();verified=F.verify_initial_l0(output,admission,reference;expected_completion_sha256=stage.completion_sha256,expected_sources=P.source_identity(),execution_permit=permit,chunk_hours=7);readback_elapsed_ns=Int(time_ns()-start)
    result=_assembled_result(stage,verified;actual=true,fit_elapsed_ns,readback_elapsed_ns,worker_elapsed_ns=Int(time_ns()-worker))
    need(P._bound(path)==body,"B1 request changed during numerical work");F._permission_recheck(checked.permission)
    result["request_sha256"]=sha;result["runtime"]=runtime;result["worker_pid"]=getpid()
    result["job_sha256"]=T._id((;driver_sha256=P._file_sha(joinpath(@__DIR__,"run_initial_l0_native.py"))[1],entry_sha256=P._file_sha(@__FILE__)[1],owner_sha256=F.OWNED_V2_OWNER,legacy_owner_sha256=F.OWNED_V2_LEGACY,argv=collect(argv),request_sha256=sha))
    P._exclusive(joinpath(root,"B1_JULIA_RESULT.json"),T._bytes(result))
    println("B1_INITIAL_L0_COMPLETE;GAINS=36;READBACK_MODELS=36;FINAL=0;CALIBRATION=0;PROMOTION=false")
    nothing
end
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__);BInitialNativeStage.main(ARGS);end