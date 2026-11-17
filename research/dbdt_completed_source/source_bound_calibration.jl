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
const FS=SolarSINDy.ArchiveCatalogSnapshotFilesystem
const D=SolarSINDy.CompletedSourceDBDT
const R=D.CompletedRefitArtifacts
const C=SolarSINDy.ArchiveCatalogCases
export ExpectedCalibration,CalibrationCapsule,prepare_expected,
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
"""Derive calibration input/context pins from reverified archival cases."""
function prepare_expected(value::B.ReverifiedCases,index,point;start,stop,asof,first_issue)
    F._archival_request(value,index,start,stop,asof)
    p=_prepare(value.cases,index,point,start,stop,asof,first_issue;admission=digest(B.admission_bytes(value)))
    ExpectedCalibration(p.context_sha256,p.input)
end
function execution_observations(started,started_utc,available)
    Dict("started_utc"=>started_utc,"elapsed_ns"=>Int(time_ns()-started),
        "available_physical_bytes_at_start"=>Int(available),"peak_rss_bytes"=>Int(Sys.maxrss()),
        "julia_threads"=>Threads.nthreads(),"blas_threads"=>F.BLAS.get_num_threads())
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
"""Compute archival scores from the bound point model and reverified case inputs."""
function calibrate_archival(value::B.ReverifiedCases,index,point,expected::ExpectedCalibration;
                            start,stop,asof,first_issue,authority=nothing,producer=nothing)
    need(authority===nothing && producer===nothing,"calibration uses case/model bindings, not execution permits")
    F._archival_request(value,index,start,stop,asof)
    admission=digest(B.admission_bytes(value));started=time_ns();started_utc=F.clock(now(UTC));available=Sys.free_memory()
    prepared=_prepare(value.cases,index,point,start,stop,asof,first_issue;admission)
    need(prepared.context_sha256==expected.context_sha256 && prepared.input==expected.input_sha256,"external calibration binding differs")
    pins=_point_pins(point)
    auth=Dict("schema"=>"archive_calibration_binding_v2","case_read_admission_sha256"=>admission,
        "point_context_sha256"=>point.expected.context_sha256,"point_model_sha256"=>pins.model,
        "point_receipt_sha256"=>pins.receipt,"context_sha256"=>expected.context_sha256,
        "input_sha256"=>expected.input_sha256,"job_sha256"=>job_identity())
    fresh=_prepare(value.cases,index,point,start,stop,asof,first_issue;admission)
    F._archival_request(value,index,start,stop,asof)
    need(digest(B.admission_bytes(value))==admission && fresh==prepared,"calibration inputs changed during computation")
    auth["runtime_observations"]=execution_observations(started,started_utc,available)
    _finish(prepared,expected,auth)
end
include("source_bound_calibration_codec.jl")
end
