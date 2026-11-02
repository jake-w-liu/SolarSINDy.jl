"""Fixed candidate numerical production and independently checked R2 selection.

Generated and archival entry points are distinct. Data digests and parsed
selection metadata never confer source access, fitting, or selection authority.
"""
module SourceBoundCandidateValidation
using Dates,SHA,Statistics,LinearAlgebra,SolarSINDy
include("source_bound_calibration.jl")
const K=SourceBoundCalibration
const F=SolarSINDy.ArchiveSourceBoundPointFit
const C=SolarSINDy.ArchiveClassicPointModels
const Q=SolarSINDy.ArchiveCatalogCaseJob
const B=SolarSINDy.ArchiveCatalogCaseReader
const A=SolarSINDy.ArchiveCatalogCaseAuthority
const J=SolarSINDy.ArchiveCatalogSourceJob
const FS=SolarSINDy.ArchiveCatalogSnapshotFilesystem
const V=SolarSINDy.ArchiveCatalogCases
const S=SolarSINDy.CompletedSelectionPrimitives
const M=SolarSINDy.ArchiveSelectedPointBindings
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
export CandidateSplit,InputPreparationAuthority,CandidateValidationAuthority,IndependentReadAuthority,
    prepare_candidate_fit_input_generated,prepare_candidate_fit_input_archival,read_fit_input,
    expected_candidate_fit,fit_candidate_generated,fit_candidate_archival,
    validate_candidate_generated,validate_candidate_archival,read_validation,
    verify_candidate_validation_generated,verify_candidate_validation_archival,
    choose_verified_R2,refit_selected_generated,refit_selected_archival,
    prepare_selected_refit_input_archival,SelectedRefitAuthority,NativeEvidence,FitPublication,ValidationPins,
    read_completed_candidate,publish_generated_verification,read_completed_generated_candidate
need(ok,message)=ok || throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(bytes))
canonical(value;limit=131072)=R.canonical_bytes(value;limit)
decode(bytes;limit=131072)=R.decode_canonical(bytes;limit)
const MAX_ROWS=600000
const MAX_INPUT_BYTES=462848
const MAX_VALIDATION_BYTES=36413696
const WORKER_NAMES=("candidate_validation.jl","candidate_validation_authority.jl",
    "candidate_validation_codec.jl","candidate_validation_independent.jl","candidate_descriptor_metadata.jl")
const WORKER_PINS=Tuple(digest(read(joinpath(@__DIR__,name))) for name in WORKER_NAMES)
function job_identity()
    root=realpath(pkgdir(SolarSINDy))
    for (name,pin) in zip(WORKER_NAMES,WORKER_PINS)
        path=joinpath(@__DIR__,name);J._ordinary(path)
        need(J._same(realpath(path),joinpath(root,"research","dbdt_completed_source",name)) && digest(read(path))==pin,"candidate loaded worker changed")
    end
    digest(canonical(Dict("schema"=>"candidate_validation_job_closure_v1","workers"=>[
        Dict("path"=>"research/dbdt_completed_source/"*name,"sha256"=>pin) for (name,pin) in zip(WORKER_NAMES,WORKER_PINS)],
        "calibration_job_sha256"=>K.job_identity());limit=16384))
end
"""Explicit generated split; actual entry points additionally enforce locked2017 dates."""
struct CandidateSplit
    start::DateTime
    stop::DateTime
    asof::DateTime
    first_issue::DateTime
    function CandidateSplit(start,stop,asof;first_issue=stop)
        need(all(t->t isa DateTime,(start,stop,asof,first_issue)) && start<stop<=asof && stop<=first_issue,"candidate split clocks")
        new(start,stop,asof,first_issue)
    end
end
split_object(s)=Dict("start_inclusive"=>F.clock(s.start),"stop_exclusive"=>F.clock(s.stop),
    "target_asof"=>F.clock(s.asof),"first_evaluation_issue"=>F.clock(s.first_issue))
function recipe_for(id)
    need(id isa AbstractString && any(x->x.id==id,S.POINT_GRID),"candidate outside frozen16grid")
    M.recipe_for_candidate(id)
end
function case_metadata(value,index;actual=false)
    need(value isa Q.VerifiedCases && value.actual_source_authority_granted===actual && value.actual_fit_performed===false,"candidate case evidence mode")
    request=decode(Vector{UInt8}(codeunits(value.request_json)))
    need(digest(codeunits(value.request_json))==value.request_sha256 && F.isint(index) && 1<=index<=length(request["tables"]),"candidate case/index binding")
    need(request["layout"]=="ground-only" && request["role"] in ("primary","independent") &&
        request["evidence_mode"]==(actual ? "actual_archival_cache" : "generated"),"candidate ground11 case required")
    request
end
include("candidate_descriptor_metadata.jl")
function descriptor(value,index,id,split;operation,point=nothing,actual=false,admission=nothing,selection_sha256=nothing)
    req=case_metadata(value,index;actual);recipe=recipe_for(id)
    facts=Dict{String,Any}("operation"=>operation,
        "case_request_sha256"=>value.request_sha256,"case_manifest_sha256"=>value.manifest_sha256,
        "case_complete_sha256"=>value.complete_sha256,"table_index"=>Int(index),"station"=>req["tables"][index]["station"],
        "role"=>req["role"],"candidate_id"=>String(id),"recipe_sha256"=>recipe.sha256,
        "grid_sha256"=>M.grid_identity(:R2),"split"=>split_object(split),"runtime"=>string(VERSION),
        "policy_sha256"=>policy_identity(),"job_sha256"=>job_identity(),
        "point_source_sha256"=>F.source_identity(),"actual_source_access"=>actual,
        "case_read_admission_sha256"=>admission,"selection_sha256"=>selection_sha256)
    if point!==nothing
        p=F._check_capsule(point)
        need(p.actual===actual && point.expected.recipe_sha256==recipe.sha256 &&
            p.context["case_request_sha256"]==value.request_sha256 && p.context["table_index"]==index,"candidate point binding")
        if actual && operation=="fixed_candidate_2017_validation_only"
            need(p.context["start_inclusive"]==F.clock(DateTime(2017)) &&
                p.context["stop_exclusive"]==F.clock(DateTime(2017,7,1)) && p.context["rows"]>=1000,
                "actual candidate must retain locked Jan-Jun training and minimum rows")
        end
        facts["point"]=point_descriptor(point)
    else
        facts["point"]=nothing
    end
    descriptor_from_metadata(facts)
end
function policy_identity()
    p=SolarSINDy.CompletedMinuteReplay.ReplayPolicy()
    # Canonical policy semantics are pinned by the checked loaded replay source.
    digest(canonical(Dict("schema"=>"candidate_default_completed_policy31_v1","phase_period_minutes"=>31,
        "label_delay_ms"=>1845000,"replay_source_sha256"=>digest(read(joinpath(pkgdir(SolarSINDy),"src","completed_minute_replay.jl"))));limit=4096))
end
work_budget()=Dict("rows_max"=>MAX_ROWS,"features"=>11,"target_queries_max"=>MAX_ROWS,"native_fits"=>0,
    "stage_seconds"=>7200,"available_physical_bytes_min"=>8589934592,"free_disk_bytes_min"=>17179869184,
    "one_BLAS_thread"=>true,"concurrency"=>1)
point_descriptor(point)=begin
    checked=F._check_capsule(point)
    Dict("context_sha256"=>point.expected.context_sha256,"recipe_sha256"=>point.expected.recipe_sha256,
        "model_sha256"=>digest(checked.model_bytes),"receipt_sha256"=>digest(checked.receipt_bytes),
        "source_sha256"=>point.expected.source_sha256)
end
function feature_audit(value,index,split)
    counts=Dict(k=>0 for k in ("scheduled","outside_issue","not_ready_stop","not_ready_asof","feature_rejected","eligible_features"))
    for i in 1:length(value.features[index])
        f=V.feature_row(value.features[index],i);counts["scheduled"]+=1
        key=!(split.start<=f.issue<split.stop) ? "outside_issue" : !(f.label_ready<split.stop) ? "not_ready_stop" :
            !(f.label_ready<=split.asof) ? "not_ready_asof" : !f.accepted ? "feature_rejected" : "eligible_features"
        counts[key]+=1
    end
    counts
end
function finish_audit(counts,n)
    need(F.isint(n) && 2<=n<=MAX_ROWS && n<=counts["eligible_features"]<=MAX_ROWS,"candidate row/query bound")
    out=copy(counts);out["label_rejected"]=counts["eligible_features"]-n;out["accepted"]=n;out
end
function cohort_sha(rows,issues,ready)
    io=IOBuffer();write(io,"CANDIDATE_VALIDATION_COHORT_V1\n",htol(UInt64(length(rows))))
    for i in eachindex(rows)
        write(io,htol(UInt64(rows[i])),htol(reinterpret(UInt64,Int64(Dates.value(issues[i])))),
            htol(reinterpret(UInt64,Int64(Dates.value(ready[i])))))
    end
    digest(take!(io))
end
function _input_numeric(value,index,split)
    arrays=Q.split_arrays(value,index;start=split.start,stop=split.stop,asof=split.asof)
    input,_=F._data_sha(arrays);counts=finish_audit(feature_audit(value,index,split),length(arrays.rows))
    ready=[V.feature_row(value.features[index],row).label_ready for row in arrays.rows]
    (;arrays,input,counts,cohort=cohort_sha(arrays.rows,arrays.issues,ready))
end
function _prepare_input(value,index,id,split,directory;authorization=nothing,actual=false,admission=nothing,
                        operation="fixed_candidate_2017_fit_input_only",selection_sha256=nothing,attempt_bytes=UInt8[],closecheck=()->nothing)
    _assert_fresh_destination(directory)
    d=descriptor(value,index,id,split;operation,actual,admission,selection_sha256)
    p=_input_numeric(value,index,split);req=case_metadata(value,index;actual)
    actual && need(length(p.arrays.rows)>=1000,"actual candidate input minimum1000 rows")
    summary=Dict("schema"=>"candidate_fit_input_summary_v1","descriptor_sha256"=>digest(canonical(d)),
        "input_sha256"=>p.input,"rows"=>length(p.arrays.rows),"first_row"=>first(p.arrays.rows),"last_row"=>last(p.arrays.rows),
        "cohort_sha256"=>p.cohort,"counts"=>p.counts,"table"=>req["tables"][index],"point_source_sha256"=>F.source_identity(),
        "native_fits"=>0,"actual_source_access"=>actual)
    _publish_input(d,summary,authorization,directory;attempt_bytes,closecheck)
end
function prepare_candidate_fit_input_generated(value::Q.VerifiedCases,index,id,split::CandidateSplit,directory)
    _assert_fresh_destination(directory)
    d=descriptor(value,index,id,split;operation="fixed_candidate_2017_fit_input_only")
    attempt=_spend_generated(directory,d,:input)
    _prepare_input(value,index,id,split,directory;attempt_bytes=attempt.bytes,closecheck=()->_close_generated(attempt))
end
function prepare_candidate_fit_input_archival(value::B.ReverifiedCases,index,id,split::CandidateSplit,directory;authority=nothing)
    auth,d,attempt=_admit_and_spend(value,index,id,split,directory,authority,:input)
    _prepare_input(value.cases,index,id,split,directory;authorization=auth,actual=true,admission=d["case_read_admission_sha256"],attempt_bytes=attempt.bytes,
        closecheck=()->_close_admission(value,index,id,split,directory,authority,:input,d,attempt))
end
function expected_candidate_fit(input)
    _check_input(input);d=input.descriptor;s=input.summary;actual=d["actual_source_access"]
    context=Dict{String,Any}("schema"=>(actual ? "source_bound_archival_fit_context_v1" : "source_bound_generated_fit_context_v1"),
        "evidence_mode"=>(actual ? "actual_archival_cache" : "generated"),
        "case_request_sha256"=>d["case_request_sha256"],"case_manifest_sha256"=>d["case_manifest_sha256"],
        "case_complete_sha256"=>d["case_complete_sha256"],"table_index"=>d["table_index"],"table"=>s["table"],
        "role"=>d["role"],"layout"=>"ground-only","start_inclusive"=>d["split"]["start_inclusive"],
        "stop_exclusive"=>d["split"]["stop_exclusive"],"target_asof"=>d["split"]["target_asof"],
        "label_ready_before"=>d["split"]["stop_exclusive"],"rows"=>s["rows"],"first_row"=>s["first_row"],"last_row"=>s["last_row"],
        "input_sha256"=>s["input_sha256"],"target_transform"=>"log1p_rate_reference_1_nt_per_min",
        "recipe_sha256"=>d["recipe_sha256"],"source_sha256"=>s["point_source_sha256"],"library_sha256"=>F.LIBRARY_SHA,"runtime"=>d["runtime"])
    actual && (context["case_read_admission_sha256"]=d["case_read_admission_sha256"])
    F.ExpectedFit(digest(canonical(context)),d["recipe_sha256"],s["point_source_sha256"],F.LIBRARY_SHA)
end
function _match_input(value,index,id,split,input;actual=false,admission=nothing)
    _check_input(input);d=descriptor(value,index,id,split;operation=input.descriptor["operation"],actual,admission,
        selection_sha256=input.descriptor["selection_sha256"])
    need(canonical(d)==canonical(input.descriptor),"candidate fit input descriptor differs");expected_candidate_fit(input)
end
function fit_candidate_generated(value::Q.VerifiedCases,index,id,split::CandidateSplit,input;directory)
    _assert_fresh_destination(directory)
    expected=_match_input(value,index,id,split,input)
    result=F.fit_generated(value,index,recipe_for(id),expected;start=split.start,stop=split.stop,asof=split.asof)
    publication=F.publish_fit(result.capsule,directory)
    point=F.read_published(directory;expected,model_sha256=publication.model_sha256,receipt_sha256=publication.receipt_sha256,complete_sha256=publication.complete_sha256)
    FitPublication(point,String(directory),publication)
end
function fit_candidate_archival(value::B.ReverifiedCases,index,id,split::CandidateSplit,input;directory,authority=nothing)
    _assert_fresh_destination(directory)
    _locked_split(split,:input);admission=digest(B.admission_bytes(value))
    need(input.descriptor["selection_sha256"]===nothing,"selected archival refit requires its distinct entry/permit")
    expected=_match_input(value.cases,index,id,split,input;actual=true,admission)
    need(authority isa F.FitAuthority,"candidate requires independently supplied original F fit grant")
    # Original F admission is metadata-only. Spend this wrapper's own producer
    # attempt before entering F's first of three target preparations.
    F._admit_archival(value,index,recipe_for(id),expected,split.start,split.stop,split.asof,authority)
    grant=A._pinned(authority.grant_path,authority.grant_sha256;canonical=true,limit=131072)
    _whole_qualification(authority,grant)
    execution=realpath(dirname(authority.grant_path))
    need(Sys.free_memory()>=8589934592 && Base.Filesystem.diskstat(execution).available>=17179869184,
        "candidate fit current native memory/disk admission")
    need(!ispath(joinpath(execution,"POINT_FIT_ATTEMPT.json")) && !islink(joinpath(execution,"POINT_FIT_ATTEMPT.json")),"original point fit permit already spent")
    attempt=joinpath(execution,"CANDIDATE_POINT_FIT_ATTEMPT.json")
    bytes=canonical(Dict("schema"=>"candidate_point_fit_attempt_v1","grant_sha256"=>authority.grant_sha256,
        "input_descriptor_sha256"=>input.pins.descriptor_sha256,"context_sha256"=>expected.context_sha256,
        "candidate_id"=>String(id),"started_utc"=>F.clock(now(UTC)));limit=16384)
    R._exclusive_write(attempt,bytes);started=time_ns()
    result=F.fit_archival(value,index,recipe_for(id),expected;start=split.start,stop=split.stop,asof=split.asof,authority)
    need((time_ns()-started)/1e9<=7200 && R.bounded_read(attempt;limit=16384)==bytes,"candidate producer attempt/time changed")
    publication=F.publish_fit(result.capsule,directory)
    point=F.read_published(directory;expected,model_sha256=publication.model_sha256,receipt_sha256=publication.receipt_sha256,complete_sha256=publication.complete_sha256)
    FitPublication(point,String(directory),publication)
end
function log_mse(targets,predictions)
    need(targets isa AbstractVector{Float64} && predictions isa AbstractVector{Float64} &&
        2<=length(targets)==length(predictions)<=MAX_ROWS,"validation metric geometry")
    need(all(x->isfinite(x)&&x>=0,targets) && all(isfinite,predictions),"validation metric domain")
    residuals=log1p.(targets).-predictions
    need(all(isfinite,residuals) && all(x->isfinite(abs2(x)),residuals),"validation residual overflow")
    loss=mean(abs2,residuals);need(isfinite(loss) && loss>=0,"validation mean overflow");loss
end
function _validate(value,index,id,split,point,directory;authorization=nothing,actual=false,admission=nothing,attempt_bytes=UInt8[],closecheck=()->nothing)
    _assert_fresh_destination(directory)
    d=descriptor(value,index,id,split;operation="fixed_candidate_2017_validation_only",point,actual,admission)
    p=K._prepare(value,index,point,split.start,split.stop,split.asof,split.first_issue;admission)
    pc=decode(Vector{UInt8}(codeunits(p.context)));rows=K._decode_records(p.records,pc["rows"])
    actual && need(pc["rows"]>=1000,"actual candidate validation minimum1000 rows")
    counts=finish_audit(feature_audit(value,index,split),pc["rows"])
    metric=Dict("schema"=>"candidate_raw_log_mse_v1","definition"=>"Statistics.mean(abs2,log1p(raw_target)-raw_log_prediction)",
        "loss_bits"=>C.bits(log_mse(rows.targets,rows.predictions)),"rows"=>pc["rows"],"clipped"=>false,"scale_normalized"=>false)
    context=Dict("schema"=>"candidate_validation_context_v1","descriptor"=>d,"descriptor_sha256"=>digest(canonical(d)),
        "calibration_numeric_context"=>pc,"cohort_sha256"=>cohort_sha(rows.rows,rows.issues,rows.ready),"counts"=>counts,
        "records_sha256"=>digest(p.records),"ranks_sha256"=>digest(p.ranks),"metric_sha256"=>digest(canonical(metric;limit=16384)))
    _publish_validation(context,p.records,p.ranks,metric,authorization,directory;attempt_bytes,closecheck)
end
function validate_candidate_generated(value::Q.VerifiedCases,index,id,split::CandidateSplit,point::F.BoundFit,directory)
    _assert_fresh_destination(directory)
    d=descriptor(value,index,id,split;operation="fixed_candidate_2017_validation_only",point)
    attempt=_spend_generated(directory,d,:validation)
    _validate(value,index,id,split,point,directory;attempt_bytes=attempt.bytes,closecheck=()->_close_generated(attempt))
end
function validate_candidate_archival(value::B.ReverifiedCases,index,id,split::CandidateSplit,point::F.BoundFit,directory;
                                      authority=nothing,producer=nothing,native=nothing)
    auth,d,attempt=_admit_and_spend(value,index,id,split,directory,authority,:validation;point,producer,native)
    _validate(value.cases,index,id,split,point,directory;authorization=auth,actual=true,admission=d["case_read_admission_sha256"],attempt_bytes=attempt.bytes,
        closecheck=()->_close_admission(value,index,id,split,directory,authority,:validation,d,attempt;point,producer,native))
end
include("candidate_validation_authority.jl")
include("candidate_validation_codec.jl")
include("candidate_validation_independent.jl")
end
