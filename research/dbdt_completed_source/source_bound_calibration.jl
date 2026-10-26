"""Source-bound signed calibration scores for immutable point-model capsules.

This offline job retains chronological scores and their stable rank projection.
It uses the existing completed-source calibration and replay definitions.
"""
module SourceBoundCalibration
using Dates,SHA,SolarSINDy
const F=SolarSINDy.ArchiveSourceBoundPointFit
const B=SolarSINDy.ArchiveCatalogCaseReader
const Q=SolarSINDy.ArchiveCatalogCaseJob
const J=SolarSINDy.ArchiveCatalogSourceJob
const A=SolarSINDy.ArchiveCatalogCaseAuthority
const FS=SolarSINDy.ArchiveCatalogSnapshotFilesystem
const D=SolarSINDy.CompletedSourceDBDT
const R=D.CompletedRefitArtifacts
const C=SolarSINDy.ArchiveCatalogCases
export ExpectedCalibration,CalibrationAuthority,CalibrationCapsule,prepare_expected,
    calibrate_generated,calibrate_archival,publish_calibration,read_calibration,completed_calibration
need(ok,message)=ok || throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(bytes))
const JOB_NAME="research/dbdt_completed_source/source_bound_calibration.jl"
const JOB_PIN=digest(read(@__FILE__))
const CODEC_NAME="research/dbdt_completed_source/source_bound_calibration_codec.jl"
const CODEC_PATH=joinpath(@__DIR__,"source_bound_calibration_codec.jl")
const CODEC_PIN=digest(read(CODEC_PATH))
const ROW_BYTES=56
const ROWS_PER_SHARD=8192
const MAX_ROWS=600000
function job_identity()
    need(digest(read(@__FILE__))==JOB_PIN && digest(read(CODEC_PATH))==CODEC_PIN,"loaded calibration job changed")
    need(J._same(realpath(@__FILE__),joinpath(realpath(pkgdir(SolarSINDy)),split(JOB_NAME,'/')...)),"calibration job outside loaded package")
    digest(R.canonical_bytes(Dict("schema"=>"source_bound_calibration_job_closure_v1","files"=>[
        Dict("path"=>JOB_NAME,"sha256"=>JOB_PIN),Dict("path"=>CODEC_NAME,"sha256"=>CODEC_PIN)]);limit=16384))
end
"""External pins for a complete calibration context and its full case inputs."""
struct ExpectedCalibration
    context_sha256::String
    input_sha256::String
    function ExpectedCalibration(context,input)
        new(J._sha(context),J._sha(input))
    end
end
"""A separately pinned score-only permit and current whole-package qualification."""
struct CalibrationAuthority
    grant_path::String
    grant_sha256::String
    source_proof_path::String
    source_proof_sha256::String
    qualification_path::String
    qualification_sha256::String
end
struct _Token end
"""Immutable data-only chronological records, stable ranks and provenance."""
struct CalibrationCapsule
    context::String
    records::String
    ranks::String
    receipt::String
    function CalibrationCapsule(context,records,ranks,receipt,::_Token)
        new(String(context),String(copy(records)),String(copy(ranks)),String(receipt))
    end
end
function _point_pins(point)
    checked=F._check_capsule(point)
    (;checked,model=digest(checked.model_bytes),receipt=digest(checked.receipt_bytes))
end
function _prepare(value::Q.VerifiedCases,index,point,start,stop,asof,first_issue;admission=nothing)
    actual=admission!==nothing
    need(value.actual_source_authority_granted===actual && value.actual_fit_performed===false,"calibration case authority differs")
    pins=_point_pins(point);pc=pins.checked.context
    need(pins.checked.actual===actual,"calibration model evidence mode differs")
    need(all(t->t isa DateTime,(start,stop,asof,first_issue)) && start<stop<=asof && stop<=first_issue,"calibration split clocks")
    actual && need(DateTime(2017)<=start<stop<=DateTime(2018),"archival calibration requires strict 2017 domain")
    need(pc["case_request_sha256"]==value.request_sha256 && pc["case_manifest_sha256"]==value.manifest_sha256 &&
        pc["case_complete_sha256"]==value.complete_sha256 && pc["table_index"]==index,"calibration model/case identity differs")
    need(F._fit_clock(pc["stop_exclusive"])<=start,"point training overlaps calibration")
    arrays=Q.split_arrays(value,index;start,stop,asof)
    input,transformed=F._data_sha(arrays)
    predictions=F.predict_log(point,arrays.features)
    need(length(predictions)==length(transformed) && all(isfinite,predictions),"calibration point predictions")
    ready=[C.feature_row(value.features[index],row).label_ready for row in arrays.rows]
    need(issorted(ready)&&all(t->t<stop<=first_issue,ready),"calibration strict label readiness")
    scores=(transformed.-predictions)./arrays.scales
    need(all(isfinite,scores),"calibration finite signed scores")
    io=IOBuffer()
    for i in eachindex(scores)
        write(io,htol(UInt64(arrays.rows[i])),htol(reinterpret(UInt64,Int64(Dates.value(arrays.issues[i])))),
            htol(reinterpret(UInt64,Int64(Dates.value(ready[i])))))
        for v in (arrays.targets[i],Float64(predictions[i]),arrays.scales[i],scores[i]);write(io,htol(reinterpret(UInt64,v)));end
    end
    records=take!(io);order=sortperm(scores;alg=MergeSort)
    ranks=Vector{UInt8}(reinterpret(UInt8,htol.(UInt32.(order))))
    context=Dict("schema"=>"source_bound_calibration_context_v1","evidence_mode"=>(actual ? "actual_archival_cache" : "generated"),
        "case_request_sha256"=>value.request_sha256,"case_manifest_sha256"=>value.manifest_sha256,"case_complete_sha256"=>value.complete_sha256,
        "table_index"=>Int(index),"role"=>pc["role"],"layout"=>pc["layout"],"table"=>pc["table"],
        "point_context_sha256"=>point.expected.context_sha256,"point_recipe_sha256"=>point.expected.recipe_sha256,
        "point_model_sha256"=>pins.model,"point_receipt_sha256"=>pins.receipt,"point_source_sha256"=>point.expected.source_sha256,
        "job_sha256"=>job_identity(),"runtime"=>string(VERSION),"start_inclusive"=>F.clock(start),"stop_exclusive"=>F.clock(stop),
        "target_asof"=>F.clock(asof),"first_evaluation_issue"=>F.clock(first_issue),"point_training_stop"=>pc["stop_exclusive"],
        "rows"=>length(scores),"input_sha256"=>input,"records_sha256"=>digest(records),"ranks_sha256"=>digest(ranks),
        "score_definition"=>"(log1p(target)-point_log_prediction)/case_scale",
        "rank_definition"=>"stable_MergeSort_isless_Float64_chronological_indices",
        "availability_kind"=>"assumed_completed_source_intervals","measured_receipts_available"=>false)
    actual && (context["case_read_admission_sha256"]=admission)
    bytes=R.canonical_bytes(context;limit=131072)
    (;context=String(copy(bytes)),context_sha256=digest(bytes),input,records,ranks)
end
"""Plan a generated calibration without granting any archival source access."""
function prepare_expected(value::Q.VerifiedCases,index,point;start,stop,asof,first_issue)
    p=_prepare(value,index,point,start,stop,asof,first_issue)
    ExpectedCalibration(p.context_sha256,p.input)
end
function _finish(prepared,expected,authorization=nothing)
    need(prepared.context_sha256==expected.context_sha256 && prepared.input==expected.input_sha256,"external calibration binding differs")
    actual=authorization!==nothing
    receipt=Dict("schema"=>"source_bound_calibration_receipt_v1","complete"=>true,"context_sha256"=>prepared.context_sha256,
        "input_sha256"=>prepared.input,"records_sha256"=>digest(prepared.records),"ranks_sha256"=>digest(prepared.ranks),
        "actual_source_access"=>actual,"actual_model_fit_performed"=>false,"provider_receipts_available"=>false,
        "software_qualification"=>false)
    actual && (receipt["authorization"]=authorization)
    CalibrationCapsule(prepared.context,prepared.records,prepared.ranks,String(R.canonical_bytes(receipt;limit=65536)),_Token())
end
"""Compute generated scores with strict training/calibration/evaluation separation."""
function calibrate_generated(value::Q.VerifiedCases,index,point,expected::ExpectedCalibration;start,stop,asof,first_issue)
    _finish(_prepare(value,index,point,start,stop,asof,first_issue),expected)
end
function _producer_evidence(point,producer)
    need(producer isa F.FitAuthority,"actual calibration needs the original point-fit producer evidence")
    pins=_point_pins(point);need(pins.checked.actual,"actual calibration requires an archival point fit")
    auth=pins.checked.receipt["authorization"]
    need(auth["grant_sha256"]==J._sha(producer.grant_sha256) && auth["source_proof_sha256"]==J._sha(producer.source_proof_sha256) &&
        auth["qualification_sha256"]==J._sha(producer.qualification_sha256),"point-fit producer pins differ")
    grant=A._pinned(producer.grant_path,producer.grant_sha256;canonical=true,limit=131072)
    J._object(grant,F._FIT_GRANT_FIELDS)
    proof=A._pinned(producer.source_proof_path,producer.source_proof_sha256;limit=1048576)
    qualification=A._pinned(producer.qualification_path,producer.qualification_sha256;limit=1048576)
    A._qualification_evidence(proof,qualification,producer,grant)
    need(grant["schema"]=="archive_source_bound_point_fit_grant_v1" && grant["operation"]=="fixed_recipe_2017_point_fit_only" &&
        grant["actual_source_access_authorized"]===true && grant["actual_model_fit_authorized"]===true &&
        grant["dataset_discovery"]===false && grant["provider_receipts_available"]===false && grant["new_case_generation_authorized"]===false &&
        F.isint(grant["expected_model_fits"]) && grant["expected_model_fits"]==1 && grant["head"]==auth["head"] &&
        grant["context_sha256"]==point.expected.context_sha256 && grant["recipe_sha256"]==point.expected.recipe_sha256 &&
        grant["input_sha256"]==pins.checked.context["input_sha256"] && grant["source_sha256"]==point.expected.source_sha256 &&
        grant["library_sha256"]==point.expected.library_sha256,"original point fit grant differs")
    # Original qualification is retained provenance. The current whole-source
    # qualification is checked separately; numeric source closure must agree.
    entries=[Dict("path"=>"SolarSINDy/"*name,"sha256"=>pin) for (name,pin) in sort!(collect(proof["source_sha256"]);by=first)
        if startswith(name,"src/") && endswith(name,".jl")]
    need(digest(R.canonical_bytes(Dict("schema"=>"source_bound_fit_source_closure_v1","files"=>entries);limit=262144))==point.expected.source_sha256,
        "original fit qualification numeric source differs")
    auth
end
function _admit(value,index,point,expected,start,stop,asof,first_issue,authority,producer)
    need(authority isa CalibrationAuthority,"actual calibration requires distinct score-only authority")
    admission=F._archival_request(value,index,start,stop,asof)
    need(stop<=first_issue,"calibration stop follows evaluation")
    grant=A._pinned(authority.grant_path,authority.grant_sha256;canonical=true,limit=131072)
    fields=Set(("schema","operation","case_read_admission_sha256","case_request_sha256","case_manifest_sha256","case_complete_sha256",
        "point_context_sha256","point_recipe_sha256","point_model_sha256","point_receipt_sha256","context_sha256","input_sha256",
        "job_sha256","source_proof_sha256","qualification_sha256","head","table_index","start_inclusive","stop_exclusive","target_asof",
        "first_evaluation_issue","actual_source_access_authorized","actual_model_fit_authorized","new_case_generation_authorized",
        "calibration_authorized","dataset_discovery","provider_receipts_available","execution_directory"))
    J._object(grant,fields);pins=_point_pins(point);admission_pin=digest(B.admission_bytes(value))
    need(grant["schema"]=="archive_source_bound_calibration_grant_v1" && grant["operation"]=="fixed_model_2017_scores_only" &&
        grant["case_read_admission_sha256"]==admission_pin && grant["case_request_sha256"]==value.cases.request_sha256 &&
        grant["case_manifest_sha256"]==value.cases.manifest_sha256 && grant["case_complete_sha256"]==value.cases.complete_sha256 &&
        grant["point_context_sha256"]==point.expected.context_sha256 && grant["point_recipe_sha256"]==point.expected.recipe_sha256 &&
        grant["point_model_sha256"]==pins.model && grant["point_receipt_sha256"]==pins.receipt &&
        grant["context_sha256"]==expected.context_sha256 && grant["input_sha256"]==expected.input_sha256 &&
        grant["job_sha256"]==job_identity() && grant["source_proof_sha256"]==J._sha(authority.source_proof_sha256)==admission["reader_source_proof_sha256"] &&
        grant["qualification_sha256"]==J._sha(authority.qualification_sha256)==admission["reader_qualification_sha256"] &&
        F.isint(grant["table_index"]) && grant["table_index"]==index && grant["start_inclusive"]==F.clock(start) && grant["stop_exclusive"]==F.clock(stop) &&
        grant["target_asof"]==F.clock(asof) && grant["first_evaluation_issue"]==F.clock(first_issue) &&
        grant["actual_source_access_authorized"]===true && grant["actual_model_fit_authorized"]===false &&
        grant["new_case_generation_authorized"]===false && grant["calibration_authorized"]===true &&
        grant["dataset_discovery"]===false && grant["provider_receipts_available"]===false,"archival calibration grant differs")
    execution=J._text(grant["execution_directory"];limit=4096)
    need(isabspath(execution),"calibration execution path must be absolute");J._ordinary(execution;directory=true)
    need(J._same(realpath(execution),realpath(dirname(authority.grant_path))),"calibration permit outside execution directory")
    proof=A._pinned(authority.source_proof_path,authority.source_proof_sha256;limit=1048576)
    qualification=A._pinned(authority.qualification_path,authority.qualification_sha256;limit=1048576)
    A._qualification(proof,qualification,authority,grant)
    need(proof["head"]==admission["reader_head"] && get(proof["source_sha256"],JOB_NAME,nothing)==JOB_PIN &&
        get(proof["source_sha256"],CODEC_NAME,nothing)==CODEC_PIN,"calibration job absent from qualification")
    producer_receipt=_producer_evidence(point,producer)
    Dict("schema"=>"archive_calibration_admission_v1","head"=>proof["head"],"grant_sha256"=>authority.grant_sha256,
        "source_proof_sha256"=>authority.source_proof_sha256,"qualification_sha256"=>authority.qualification_sha256,
        "case_read_admission_sha256"=>admission_pin,"producer_grant_sha256"=>producer_receipt["grant_sha256"],
        "producer_source_proof_sha256"=>producer_receipt["source_proof_sha256"],"producer_qualification_sha256"=>producer_receipt["qualification_sha256"],
        "calibration_authorized"=>true,"actual_model_fit_authorized"=>false,"provider_receipts_available"=>false)
end
"""Compute archival scores only after distinct current and producer qualifications.

The fixed permit is consumed atomically; it authorizes no new fit or selection.
"""
function calibrate_archival(value::B.ReverifiedCases,index,point,expected::ExpectedCalibration;
                            start,stop,asof,first_issue,authority=nothing,producer=nothing)
    auth=_admit(value,index,point,expected,start,stop,asof,first_issue,authority,producer)
    attempt=R.canonical_bytes(Dict("schema"=>"archive_calibration_attempt_v1","grant_sha256"=>authority.grant_sha256,
        "context_sha256"=>expected.context_sha256,"input_sha256"=>expected.input_sha256,"started_utc"=>F.clock(now(UTC)));limit=16384)
    path=joinpath(realpath(dirname(authority.grant_path)),"CALIBRATION_ATTEMPT.json");R._exclusive_write(path,attempt)
    prepared=_prepare(value.cases,index,point,start,stop,asof,first_issue;admission=digest(B.admission_bytes(value)))
    need(_admit(value,index,point,expected,start,stop,asof,first_issue,authority,producer)==auth,"calibration admission changed")
    need(R.bounded_read(path;limit=16384)==attempt,"calibration attempt changed")
    auth["attempt_sha256"]=digest(attempt)
    _finish(prepared,expected,auth)
end
include("source_bound_calibration_codec.jl")
end
