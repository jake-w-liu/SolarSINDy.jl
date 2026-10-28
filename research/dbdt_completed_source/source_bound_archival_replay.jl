"""Fixed-model archival comparisons on the common completed-source event engine."""
module SourceBoundArchivalReplay
using Dates,SHA,SolarSINDy
import ..SourceBoundSharedReplay, ..SourceBoundArchivalStatic
const S=SourceBoundSharedReplay
const T=SourceBoundArchivalStatic
const K=S.K
const F=SolarSINDy.ArchiveSourceBoundPointFit
const B=SolarSINDy.ArchiveCatalogCaseReader
const J=SolarSINDy.ArchiveCatalogSourceJob
const A=SolarSINDy.ArchiveCatalogCaseAuthority
const D=SolarSINDy.CompletedSourceDBDT
const E=D.CompletedLabelEvents
const R=D.CompletedRefitArtifacts
need(ok,message)=ok || throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(bytes))
const PREFIX="research/dbdt_completed_source/"
const JOB_FILES=Tuple(sort!(unique([T.JOB_FILES...,
    "source_bound_archival_replay.jl","source_bound_archival_replay_trace.jl","source_bound_shared_replay.jl",
    "source_bound_evaluation_binding.jl","publisher_up_projection.jl","publisher_up_values.jl","retained_publisher_up.jl"])))
const JOB_PINS=Tuple(digest(read(joinpath(@__DIR__,name))) for name in JOB_FILES)
export ReplayAuthority,prepare_request,replay_archival,result_digest

"""Pins for one fixed existing-case replay, with no fitting or discovery permission."""
struct ReplayAuthority
    grant_path::String
    grant_sha256::String
    source_proof_path::String
    source_proof_sha256::String
    qualification_path::String
    qualification_sha256::String
end
function job_identity()
    root=realpath(pkgdir(SolarSINDy))
    for (name,pin) in zip(JOB_FILES,JOB_PINS)
        path=joinpath(@__DIR__,name)
        need(J._same(realpath(path),joinpath(root,split(PREFIX*name,'/')...)) && digest(read(path))==pin,
            "archival replay job outside unchanged loaded package")
    end
    T.job_identity();S._evaluation_job_identity()
    digest(R.canonical_bytes(Dict("schema"=>"archival_shared_replay_job_closure_v1",
        "files"=>[Dict("path"=>PREFIX*name,"sha256"=>pin) for (name,pin) in zip(JOB_FILES,JOB_PINS)]);limit=32768))
end
function _specifications(subscribers,alpha)
    need((subscribers isa AbstractVector || subscribers isa Tuple) && 2<=length(subscribers)<=16,
        "archival replay requires bounded fixed subscriber specifications")
    subscribers isa AbstractVector && Base.require_one_based_indexing(subscribers)
    specs=Tuple(subscribers)
    need(all(s->s isa E.SubscriberSpec && s.nominal==alpha,specs) && allunique(s.name for s in specs),
        "archival replay subscriber identity or nominal differs")
    need(any(s->s.kind===:phase && s.mandatory,specs) && any(s->s.kind===:fifo && s.mandatory,specs),
        "archival replay requires mandatory phase and FIFO comparisons")
    encoded=[Dict("name"=>String(s.name),"kind"=>String(s.kind),"nominal_bits"=>F.bits(s.nominal),"gamma_bits"=>F.bits(s.gamma),
        "phase_period"=>String(s.phase_period),"rolling_capacity"=>s.rolling_capacity,"mandatory"=>s.mandatory) for s in specs]
    (;specs,encoded)
end
function _read_admission(value::B.ReverifiedCases)
    cases=value.cases
    need(cases.actual_source_authority_granted===true && cases.actual_fit_performed===false,
        "archival replay requires reverified actual cases without fit permission")
    admission=R.decode_canonical(B.admission_bytes(value);limit=131072)
    J._object(admission,Set(("schema","evidence_mode","request_sha256","artifact_manifest_sha256","artifact_complete_sha256",
        "producer_head","producer_grant_sha256","producer_source_proof_sha256","producer_qualification_sha256",
        "reader_head","reader_grant_sha256","reader_source_proof_sha256","reader_qualification_sha256",
        "all_artifact_bytes_recomputed","new_case_generation_authorized","actual_model_fit_authorized","provider_receipts_available","actual_fits")))
    need(admission["schema"]=="archive_catalog_case_read_admission_v1" && admission["evidence_mode"]=="actual_archival_cache" &&
        admission["request_sha256"]==cases.request_sha256 && admission["artifact_manifest_sha256"]==cases.manifest_sha256 &&
        admission["artifact_complete_sha256"]==cases.complete_sha256 && admission["all_artifact_bytes_recomputed"]===true &&
        admission["new_case_generation_authorized"]===false && admission["actual_model_fit_authorized"]===false &&
        admission["provider_receipts_available"]===false && F.isint(admission["actual_fits"]) && admission["actual_fits"]==0,
        "archival replay case admission differs")
    for key in ("producer_grant_sha256","producer_source_proof_sha256","producer_qualification_sha256",
                "reader_grant_sha256","reader_source_proof_sha256","reader_qualification_sha256");J._sha(admission[key]);end
    for key in ("producer_head","reader_head")
        need(admission[key] isa String && occursin(r"\A[0-9a-f]{40}\z",admission[key]),"archival replay case source head")
    end
    admission
end
function _metadata(fitting::B.ReverifiedCases,fitting_index,evaluation::B.ReverifiedCases,index,point,calibration,expected,static;
                   start,stop,asof,subscribers,nominal,grace_ms,warmup)
    need(warmup isa Bool,"archival replay warmup selector must be Boolean")
    need(all(t->t isa DateTime && D.whole(t),(start,stop,asof)) && start<stop<=asof,"archival replay whole-minute interval")
    grace=E.checked_grace(grace_ms);alpha=Float64(D.nominal_value(nominal));need(0<alpha<1,"archival replay nominal domain")
    specifications=_specifications(subscribers,alpha)
    fitting_admission=_read_admission(fitting);evaluation_admission=_read_admission(evaluation)
    need(static isa NamedTuple && keys(static)==(:capsule,:expected),"archival replay needs static capsule and external pins")
    need(static.capsule isa T.ArchivalStaticCapsule && static.expected isa T.ExpectedStatic,"archival replay needs archival static data")
    static_checked=T.check_calibration(static.capsule,point,calibration,expected;expected_static=static.expected)
    need(static_checked.numeric.alpha==alpha,"archival replay static/adaptive nominal differs")
    static_request=T._metadata(fitting,fitting_index,point,calibration,expected;nominal=alpha)
    need(static_checked.context["request_sha256"]==digest(static_request.bytes),"archival replay static training request differs")
    cases=evaluation.cases;need(F.isint(index) && 1<=index<=length(cases.features),"archival replay evaluation table index")
    request=R.decode_canonical(Vector{UInt8}(codeunits(cases.request_json));limit=131072)
    need(digest(codeunits(cases.request_json))==cases.request_sha256 && request["schema"]=="archive_catalog_case_request_v1" &&
        request["evidence_mode"]=="actual_archival_cache" && request["runtime"]==string(VERSION),"archival replay evaluation request differs")
    table=request["tables"][index];features=cases.features[index];pc=static_request.pc;kc=static_request.kc
    need(pc.context["role"]==request["role"] && pc.context["layout"]==request["layout"] &&
        pc.context["table"]["station"]==table["station"]==features.station,"archival replay role/layout/station differs")
    need(F._fit_clock(kc.context["stop_exclusive"])<=start && kc.first_issue<=start &&
        J._clock(table["start_inclusive"])<=start<stop<=J._clock(table["stop_exclusive"]) &&
        asof<=cases.labels[index].asof,"archival replay overlaps calibration or exceeds sealed evaluation table")
    mode=request["layout"]=="ground-only" ? :ground11 : :mixed8
    need(features.mode===mode && D.policy_equal(features.policy,D.CMR.ReplayPolicy()),"archival replay feature layout or policy differs")
    context=Dict("schema"=>"archival_shared_replay_request_v1","evidence_mode"=>"actual_archival_cache","runtime"=>string(VERSION),
        "job_sha256"=>job_identity(),"fitting_case_read_admission_sha256"=>digest(B.admission_bytes(fitting)),
        "evaluation_case_read_admission_sha256"=>digest(B.admission_bytes(evaluation)),"static_fit_request_sha256"=>digest(static_request.bytes),
        "receipt_source_id"=>"archival-catalog/"*J._sha(request["source"]["manifest_sha256"])*"/ground-"*features.station,
        "point"=>Dict("context_sha256"=>point.expected.context_sha256,"recipe_sha256"=>point.expected.recipe_sha256,
            "source_sha256"=>point.expected.source_sha256,"model_sha256"=>digest(pc.model_bytes),"receipt_sha256"=>digest(pc.receipt_bytes)),
        "calibration"=>Dict("context_sha256"=>expected.context_sha256,"input_sha256"=>expected.input_sha256,"receipt_sha256"=>digest(kc.receipt_bytes)),
        "static"=>Dict("context_sha256"=>static.expected.context_sha256,"numeric_sha256"=>static.expected.numeric_sha256,
            "receipt_sha256"=>digest(static_checked.receipt_bytes)),
        "evaluation_case"=>Dict("request_sha256"=>cases.request_sha256,"manifest_sha256"=>cases.manifest_sha256,
            "complete_sha256"=>cases.complete_sha256,"table_index"=>Int(index),"table"=>table,"feature_payload_sha256"=>features.payload_sha256,"source"=>request["source"]),
        "start_inclusive"=>F.clock(start),"stop_exclusive"=>F.clock(stop),"target_asof"=>F.clock(asof),
        "nominal_bits"=>F.bits(alpha),"subscribers"=>specifications.encoded,"grace_ms"=>grace,"publisher_warmup"=>warmup,
        "static_methods"=>collect(String.(S.B.METHODS)),"availability_kind"=>"assumed_completed_source_intervals",
        "measured_receipts_available"=>false,"scientific_coverage_qualified"=>false)
    bytes=R.canonical_bytes(context;limit=131072)
    (;context,bytes,fitting_admission,evaluation_admission,static_request,static_checked,request,table,pc,kc,
      alpha,specs=specifications.specs,grace)
end
"""Prepare only fixed metadata; no evaluation feature/target arrays or replay is consumed."""
function prepare_request(fitting::B.ReverifiedCases,fitting_index,evaluation::B.ReverifiedCases,index,point,calibration,expected,static;
                         start,stop,asof,subscribers,nominal,grace_ms=0,warmup=true)
    prepared=_metadata(fitting,fitting_index,evaluation,index,point,calibration,expected,static;
        start,stop,asof,subscribers,nominal,grace_ms,warmup)
    (;request_bytes=copy(prepared.bytes),request_sha256=digest(prepared.bytes))
end
function _admit(fitting,fitting_index,point,calibration,expected,prepared,authority,static_producer,calibration_producer,point_producer)
    need(authority isa ReplayAuthority,"archival replay requires distinct replay-only authority")
    need(static_producer isa T.StaticFitAuthority,"archival replay needs original static producer evidence")
    grant=A._pinned(authority.grant_path,authority.grant_sha256;canonical=true,limit=131072)
    J._object(grant,Set(("schema","operation","request_sha256","job_sha256","source_proof_sha256","qualification_sha256","head",
        "actual_source_access_authorized","replay_authorized","actual_model_fit_authorized","actual_uncertainty_fit_authorized",
        "new_case_generation_authorized","dataset_discovery","provider_receipts_available","expected_replays","execution_directory")))
    need(grant["schema"]=="archival_shared_replay_grant_v1" && grant["operation"]=="fixed_models_completed_source_shared31_replay_only" &&
        grant["request_sha256"]==digest(prepared.bytes) && grant["job_sha256"]==job_identity() &&
        grant["source_proof_sha256"]==J._sha(authority.source_proof_sha256)==prepared.fitting_admission["reader_source_proof_sha256"]==prepared.evaluation_admission["reader_source_proof_sha256"] &&
        grant["qualification_sha256"]==J._sha(authority.qualification_sha256)==prepared.fitting_admission["reader_qualification_sha256"]==prepared.evaluation_admission["reader_qualification_sha256"] &&
        grant["actual_source_access_authorized"]===true && grant["replay_authorized"]===true && grant["actual_model_fit_authorized"]===false &&
        grant["actual_uncertainty_fit_authorized"]===false && grant["new_case_generation_authorized"]===false &&
        grant["dataset_discovery"]===false && grant["provider_receipts_available"]===false &&
        F.isint(grant["expected_replays"]) && grant["expected_replays"]==1,"archival replay grant differs")
    execution=J._text(grant["execution_directory"];limit=4096);need(isabspath(execution),"archival replay execution path must be absolute")
    J._ordinary(execution;directory=true)
    need(J._same(realpath(execution),realpath(dirname(authority.grant_path))),"archival replay permit outside execution directory")
    root_parts=splitpath(realpath(pkgdir(SolarSINDy)));execution_parts=splitpath(realpath(execution))
    need(!(length(execution_parts)>=length(root_parts) && all(J._same(a,b) for (a,b) in zip(root_parts,execution_parts))),
        "archival replay execution would modify qualified source")
    proof=A._pinned(authority.source_proof_path,authority.source_proof_sha256;limit=1048576)
    qualification=A._pinned(authority.qualification_path,authority.qualification_sha256;limit=1048576)
    A._qualification(proof,qualification,authority,grant)
    need(proof["head"]==prepared.fitting_admission["reader_head"]==prepared.evaluation_admission["reader_head"] &&
        all(get(proof["source_sha256"],PREFIX*name,nothing)==pin for (name,pin) in zip(JOB_FILES,JOB_PINS)),"archival replay job absent from qualification")
    original=T._admit(fitting,fitting_index,point,calibration,expected,prepared.static_request,static_producer,calibration_producer,point_producer)
    static_receipt=R.decode_canonical(prepared.static_checked.receipt_bytes;limit=65536)["authorization"]
    need(original==Dict(key=>value for (key,value) in static_receipt if key!="attempt_sha256"),"archival replay static producer admission differs")
    T._retained_attempt(realpath(dirname(static_producer.grant_path)),"STATIC_FIT_ATTEMPT.json",static_receipt["attempt_sha256"])
    Dict("schema"=>"archival_shared_replay_admission_v1","head"=>proof["head"],"grant_sha256"=>authority.grant_sha256,
        "source_proof_sha256"=>authority.source_proof_sha256,"qualification_sha256"=>authority.qualification_sha256,
        "request_sha256"=>digest(prepared.bytes),"static_grant_sha256"=>static_receipt["grant_sha256"],
        "static_attempt_sha256"=>static_receipt["attempt_sha256"],"replay_authorized"=>true,
        "actual_model_fit_authorized"=>false,"actual_uncertainty_fit_authorized"=>false,"provider_receipts_available"=>false)
end
include("source_bound_archival_replay_trace.jl")
"""Replay admitted existing cases once with all fixed comparisons and no new fitting."""
function replay_archival(fitting::B.ReverifiedCases,fitting_index,evaluation::B.ReverifiedCases,index,point,calibration,expected,static;
        start,stop,asof,subscribers,nominal,grace_ms=0,warmup=true,authority=nothing,
        static_producer=nothing,calibration_producer=nothing,point_producer=nothing)
    need(authority isa ReplayAuthority,"archival replay requires distinct replay-only authority")
    prepared=_metadata(fitting,fitting_index,evaluation,index,point,calibration,expected,static;
        start,stop,asof,subscribers,nominal,grace_ms,warmup)
    auth=_admit(fitting,fitting_index,point,calibration,expected,prepared,authority,static_producer,calibration_producer,point_producer)
    attempt=R.canonical_bytes(Dict("schema"=>"archival_shared_replay_attempt_v1","grant_sha256"=>authority.grant_sha256,
        "request_sha256"=>digest(prepared.bytes),"started_utc"=>F.clock(now(UTC)));limit=16384)
    path=joinpath(realpath(dirname(authority.grant_path)),"REPLAY_ATTEMPT.json");R._exclusive_write(path,attempt)
    request_pin=digest(prepared.bytes)
    input=S._common_inputs(evaluation.cases,index,point,calibration,expected,start,stop,asof,prepared.pc,prepared.kc,
        prepared.request,prepared.table,request_pin)
    audit=(;scheduled=length(input.prepared),features_accepted=length(input.accepted),features_rejected=length(input.prepared)-length(input.accepted),
        rejection_reasons=copy(input.reasons),source_bound_provenance_verified=true,scientific_coverage_qualified=false,
        measured_receipts_available=false,actual_source_authority_granted=false,actual_source_access=true,actual_fits=0,
        mandatory_static_inventory_complete=true,explicit_evaluation_binding_matched=true)
    result=if isempty(input.accepted)
        (;status=:no_feature_eligible,audit,shared=nothing,publisher_upper=nothing,publisher_absolute=nothing,static_bounds=nothing,
          study_pass=false,evaluation_binding_sha256=request_pin)
    else
        fixed=T.predict_upper(static.capsule,input.matrix,input.centers[input.accepted],input.scales[input.accepted],input.peaks;
            point,calibration,expected,expected_static=static.expected,start)
        S._execute_common(input;start,stop,asof,specs=prepared.specs,alpha=prepared.alpha,grace_ms=prepared.grace,warmup,fixed,audit,
            protocol_id="archival-source-bound-shared31-v1/"*request_pin,source_id=prepared.context["receipt_source_id"])
    end
    feature_rows=[(;row_id=input.row_ids[i],issue=row.issue,accepted=row.accepted,reason=row.reason,
        center=input.centers[i],scale=input.scales[i]) for (i,row) in enumerate(input.prepared)]
    output_pin=result_digest(result,feature_rows;request_sha256=request_pin)
    fresh=_metadata(fitting,fitting_index,evaluation,index,point,calibration,expected,static;
        start,stop,asof,subscribers,nominal,grace_ms,warmup)
    need(fresh.bytes==prepared.bytes && _admit(fitting,fitting_index,point,calibration,expected,fresh,authority,
        static_producer,calibration_producer,point_producer)==auth,"archival replay source or producer admission changed")
    T._retained_attempt(dirname(path),basename(path),digest(attempt));auth["attempt_sha256"]=digest(attempt)
    need(result_digest(result,feature_rows;request_sha256=request_pin)==output_pin,"archival replay output changed during recheck")
    receipt=R.canonical_bytes(Dict("schema"=>"archival_shared_replay_receipt_v1","complete"=>true,"request_sha256"=>request_pin,
        "result_sha256"=>output_pin,"scheduled"=>audit.scheduled,"feature_eligible"=>audit.features_accepted,
        "feature_rejected"=>audit.features_rejected,"source_bound_provenance_verified"=>true,"actual_source_access"=>true,
        "actual_model_fits"=>0,"actual_uncertainty_parameter_fits"=>0,"provider_receipts_available"=>false,
        "software_qualification"=>false,"scientific_coverage_qualified"=>false,"independent_confirmation_qualified"=>false,
        "authorization"=>auth);limit=65536)
    R._exclusive_write(joinpath(dirname(path),"REPLAY_RECEIPT.json"),receipt)
    receipt_pin=digest(receipt)
    (;result,feature_rows,request=String(copy(prepared.bytes)),request_sha256=request_pin,result_sha256=output_pin,
      receipt=String(copy(receipt)),receipt_sha256=receipt_pin)
end
end
