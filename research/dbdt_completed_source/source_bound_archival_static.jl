"""Separately admitted archival calibration of the four fixed static bounds."""
module SourceBoundArchivalStatic
using Dates,SHA,SolarSINDy
import ..SourceBoundSharedReplay
const S=SourceBoundSharedReplay.B
const K=SourceBoundSharedReplay.K
const F=SolarSINDy.ArchiveSourceBoundPointFit
const B=SolarSINDy.ArchiveCatalogCaseReader
const J=SolarSINDy.ArchiveCatalogSourceJob
const A=SolarSINDy.ArchiveCatalogCaseAuthority
const FS=SolarSINDy.ArchiveCatalogSnapshotFilesystem
const D=SolarSINDy.CompletedSourceDBDT
const R=D.CompletedRefitArtifacts
const ExpectedStatic=S.ExpectedStatic
need(ok,message)=ok || throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(bytes))
const CONVENTION="locked_heteroGaussian_raw_normalized_fixedCalibrationEdge_split_v1"
const JOB_FILES=("source_bound_archival_static.jl","source_bound_static_artifact_codec.jl",
    "source_bound_static_bounds.jl","source_bound_calibration.jl","source_bound_calibration_codec.jl")
const JOB_PINS=Tuple(digest(read(joinpath(@__DIR__,name))) for name in JOB_FILES)
const PREFIX="research/dbdt_completed_source/"
export StaticFitAuthority,ArchivalStaticCapsule,ExpectedStatic,prepare_request,calibrate_archival,
    check_calibration,predict_upper,publish_static,read_static

"""A fixed request permit and current full-source software qualification."""
struct StaticFitAuthority
    grant_path::String
    grant_sha256::String
    source_proof_path::String
    source_proof_sha256::String
    qualification_path::String
    qualification_sha256::String
end
struct _Token end
owned(value)=String(copy(value isa AbstractString ? codeunits(value) : value))
"""Owned data only; construction itself grants no source or fit authority."""
struct ArchivalStaticCapsule
    context::String
    numeric::String
    receipt::String
    function ArchivalStaticCapsule(context,numeric,receipt,::_Token)
        new(owned(context),owned(numeric),owned(receipt))
    end
end
function job_identity()
    root=realpath(pkgdir(SolarSINDy))
    for (name,pin) in zip(JOB_FILES,JOB_PINS)
        path=joinpath(@__DIR__,name)
        need(J._same(realpath(path),joinpath(root,split(PREFIX*name,'/')...)) && digest(read(path))==pin,
            "archival static job outside unchanged loaded package")
    end
    S.job_identity();K.job_identity()
    digest(R.canonical_bytes(Dict("schema"=>"archival_static_job_closure_v1",
        "files"=>[Dict("path"=>PREFIX*name,"sha256"=>pin) for (name,pin) in zip(JOB_FILES,JOB_PINS)]);limit=16384))
end
function _request(point,calibration,expected,pc,kc,admission_pin;nominal)
    need(pc.actual && kc.actual,"archival static requires actual point and calibration capsules")
    c=kc.context;p=pc.context
    train_start=F._fit_clock(p["start_inclusive"]);train_stop=F._fit_clock(p["stop_exclusive"])
    start=F._fit_clock(c["start_inclusive"]);stop=F._fit_clock(c["stop_exclusive"])
    need(DateTime(2017)<=train_start<train_stop<=start<stop<=DateTime(2018),"archival static strict 2017 splits")
    need(c["case_read_admission_sha256"]==J._sha(admission_pin),"archival static read admission differs")
    alpha=Float64(D.nominal_value(nominal));need(isfinite(alpha) && 0<alpha<1,"archival static nominal domain")
    request=Dict("schema"=>"archival_static_fit_request_v1","evidence_mode"=>"actual_archival_cache",
        "runtime"=>string(VERSION),"job_sha256"=>job_identity(),"case_read_admission_sha256"=>admission_pin,
        "point_context_sha256"=>point.expected.context_sha256,"point_model_sha256"=>c["point_model_sha256"],
        "point_receipt_sha256"=>c["point_receipt_sha256"],"point_source_sha256"=>point.expected.source_sha256,
        "calibration_context_sha256"=>expected.context_sha256,"calibration_input_sha256"=>expected.input_sha256,
        "calibration_receipt_sha256"=>digest(kc.receipt_bytes),"training_input_sha256"=>p["input_sha256"],
        "training_rows"=>p["rows"],"calibration_rows"=>c["rows"],"splits"=>Dict(
            "training_start_inclusive"=>p["start_inclusive"],"training_stop_exclusive"=>p["stop_exclusive"],
            "calibration_start_inclusive"=>c["start_inclusive"],"calibration_stop_exclusive"=>c["stop_exclusive"],
            "calibration_target_asof"=>c["target_asof"],"first_evaluation_issue"=>c["first_evaluation_issue"]),"nominal_bits"=>F.bits(alpha),
        "methods"=>collect(String.(S.METHODS)),"convention"=>CONVENTION,
        "availability_kind"=>"assumed_completed_source_intervals","measured_receipts_available"=>false)
    for key in ("case_request_sha256","case_manifest_sha256","case_complete_sha256","table_index","role","layout","table")
        request[key]=c[key]
    end
    (;request,bytes=R.canonical_bytes(request;limit=32768),pc,kc,alpha)
end
function _metadata(value::B.ReverifiedCases,index,point,calibration,expected;nominal)
    pc=F._check_capsule(point);kc=K._checked(calibration,point,expected);c=kc.context
    start=F._fit_clock(c["start_inclusive"]);stop=F._fit_clock(c["stop_exclusive"])
    admission=F._archival_request(value,index,start,stop,F._fit_clock(c["target_asof"]))
    need(c["table_index"]==index && c["case_request_sha256"]==value.cases.request_sha256 &&
        c["case_manifest_sha256"]==value.cases.manifest_sha256 && c["case_complete_sha256"]==value.cases.complete_sha256,
        "archival static cases differ from calibration")
    prepared=_request(point,calibration,expected,pc,kc,digest(B.admission_bytes(value));nominal)
    (;prepared...,admission)
end
"""Prepare metadata only; no source feature/label arrays or fitted outputs are read."""
function prepare_request(value::B.ReverifiedCases,index,point,calibration,expected;nominal=.1)
    prepared=_metadata(value,index,point,calibration,expected;nominal)
    (;request_bytes=copy(prepared.bytes),request_sha256=digest(prepared.bytes))
end
function _retained_attempt(directory,name,pin)
    path=joinpath(directory,name);J._ordinary(path)
    before=J._identity(path);body=R.bounded_read(path;limit=16384)
    need(digest(body)==J._sha(pin) && J._identity(path)==before,"archival producer attempt changed")
    nothing
end
function _admit(value,index,point,calibration,expected,prepared,authority,calibration_producer,point_producer)
    need(authority isa StaticFitAuthority,"archival static requires distinct uncertainty-fit authority")
    grant=A._pinned(authority.grant_path,authority.grant_sha256;canonical=true,limit=131072)
    J._object(grant,Set(("schema","operation","request_sha256","job_sha256","source_proof_sha256","qualification_sha256","head",
        "actual_source_access_authorized","actual_point_model_fit_authorized","actual_uncertainty_fit_authorized",
        "expected_uncertainty_fits","new_case_generation_authorized","dataset_discovery","provider_receipts_available","execution_directory")))
    need(grant["schema"]=="archival_static_fit_grant_v1" && grant["operation"]=="fixed_point_static_uncertainty_2017_only" &&
        grant["request_sha256"]==digest(prepared.bytes) && grant["job_sha256"]==job_identity() &&
        grant["source_proof_sha256"]==J._sha(authority.source_proof_sha256)==prepared.admission["reader_source_proof_sha256"] &&
        grant["qualification_sha256"]==J._sha(authority.qualification_sha256)==prepared.admission["reader_qualification_sha256"] &&
        grant["actual_source_access_authorized"]===true && grant["actual_point_model_fit_authorized"]===false &&
        grant["actual_uncertainty_fit_authorized"]===true && F.isint(grant["expected_uncertainty_fits"]) && grant["expected_uncertainty_fits"]==1 &&
        grant["new_case_generation_authorized"]===false && grant["dataset_discovery"]===false && grant["provider_receipts_available"]===false,
        "archival static grant differs")
    execution=J._text(grant["execution_directory"];limit=4096)
    need(isabspath(execution),"static execution directory must be absolute");J._ordinary(execution;directory=true)
    need(J._same(realpath(execution),realpath(dirname(authority.grant_path))),"static grant outside execution directory")
    root_parts=splitpath(realpath(pkgdir(SolarSINDy)));execution_parts=splitpath(realpath(execution))
    need(!(length(execution_parts)>=length(root_parts) && all(J._same(a,b) for (a,b) in zip(root_parts,execution_parts))),
        "static execution would modify qualified source")
    proof=A._pinned(authority.source_proof_path,authority.source_proof_sha256;limit=1048576)
    qualification=A._pinned(authority.qualification_path,authority.qualification_sha256;limit=1048576)
    A._qualification(proof,qualification,authority,grant)
    need(proof["head"]==prepared.admission["reader_head"] &&
        all(get(proof["source_sha256"],PREFIX*name,nothing)==pin for (name,pin) in zip(JOB_FILES,JOB_PINS)),
        "static job absent from qualification")
    c=prepared.kc.context
    producer_auth=K._admit(value,index,point,expected,F._fit_clock(c["start_inclusive"]),F._fit_clock(c["stop_exclusive"]),
        F._fit_clock(c["target_asof"]),F._fit_clock(c["first_evaluation_issue"]),calibration_producer,point_producer)
    retained=prepared.kc.receipt["authorization"]
    need(producer_auth==Dict(key=>value for (key,value) in retained if key!="attempt_sha256"),"calibration producer admission differs")
    _retained_attempt(realpath(dirname(calibration_producer.grant_path)),"CALIBRATION_ATTEMPT.json",retained["attempt_sha256"])
    Dict("schema"=>"archival_static_fit_admission_v1","head"=>proof["head"],"grant_sha256"=>authority.grant_sha256,
        "source_proof_sha256"=>authority.source_proof_sha256,"qualification_sha256"=>authority.qualification_sha256,
        "request_sha256"=>digest(prepared.bytes),"case_read_admission_sha256"=>prepared.request["case_read_admission_sha256"],
        "calibration_grant_sha256"=>retained["grant_sha256"],"calibration_source_proof_sha256"=>retained["source_proof_sha256"],
        "calibration_qualification_sha256"=>retained["qualification_sha256"],"calibration_attempt_sha256"=>retained["attempt_sha256"],
        "uncertainty_fit_authorized"=>true,"actual_point_model_fit_authorized"=>false,"provider_receipts_available"=>false)
end
"""Fit uncertainty parameters once after checking distinct current and producer evidence."""
function calibrate_archival(value::B.ReverifiedCases,index,point,calibration,expected;
                            nominal=.1,authority=nothing,calibration_producer=nothing,point_producer=nothing)
    need(authority isa StaticFitAuthority,"archival static requires distinct uncertainty-fit authority")
    prepared=_metadata(value,index,point,calibration,expected;nominal)
    auth=_admit(value,index,point,calibration,expected,prepared,authority,calibration_producer,point_producer)
    attempt=R.canonical_bytes(Dict("schema"=>"archival_static_fit_attempt_v1","grant_sha256"=>authority.grant_sha256,
        "request_sha256"=>digest(prepared.bytes),"started_utc"=>F.clock(now(UTC)));limit=16384)
    path=joinpath(realpath(dirname(authority.grant_path)),"STATIC_FIT_ATTEMPT.json");R._exclusive_write(path,attempt)
    values=S._calibration_values(value.cases,index,point,prepared.pc,prepared.kc;nominal=prepared.alpha)
    need(values.training_input_sha256==prepared.request["training_input_sha256"] &&
        values.training_rows==prepared.request["training_rows"] && values.calibration_rows==prepared.request["calibration_rows"],
        "static fitted cohort differs from fixed request")
    fresh=_metadata(value,index,point,calibration,expected;nominal=prepared.alpha)
    need(fresh.bytes==prepared.bytes && _admit(value,index,point,calibration,expected,fresh,authority,calibration_producer,point_producer)==auth,
        "static source or producer admission changed")
    _retained_attempt(dirname(path),basename(path),digest(attempt));auth["attempt_sha256"]=digest(attempt)
    context=R.canonical_bytes(Dict("schema"=>"archival_static_calibration_v1","request"=>prepared.request,
        "request_sha256"=>digest(prepared.bytes),"numeric_sha256"=>digest(values.bytes),
        "regime_bin_counts"=>values.numeric.counts,"empty_regime_bins"=>findall(==(0),values.numeric.counts));limit=32768)
    expected_static=ExpectedStatic(digest(context),digest(values.bytes))
    receipt=R.canonical_bytes(Dict("schema"=>"archival_static_calibration_receipt_v1","complete"=>true,
        "context_sha256"=>expected_static.context_sha256,"numeric_sha256"=>expected_static.numeric_sha256,
        "actual_source_access"=>true,"actual_point_model_fits"=>0,"uncertainty_parameter_fits"=>1,
        "provider_receipts_available"=>false,"software_qualification"=>false,"scientific_coverage_qualified"=>false,
        "authorization"=>auth);limit=65536)
    capsule=ArchivalStaticCapsule(context,values.bytes,receipt,_Token())
    check_calibration(capsule,point,calibration,expected;expected_static)
    (;capsule,expected=expected_static)
end
function check_calibration(value::ArchivalStaticCapsule,point,calibration,expected;expected_static::ExpectedStatic)
    pc=F._check_capsule(point);kc=K._checked(calibration,point,expected)
    cb=Vector{UInt8}(codeunits(value.context));nb=Vector{UInt8}(codeunits(value.numeric));rb=Vector{UInt8}(codeunits(value.receipt))
    need(digest(cb)==expected_static.context_sha256 && digest(nb)==expected_static.numeric_sha256,"archival static external pins differ")
    c=R.decode_canonical(cb;limit=32768);r=R.decode_canonical(rb;limit=65536)
    J._object(c,Set(("schema","request","request_sha256","numeric_sha256","regime_bin_counts","empty_regime_bins")))
    numeric=S._decode(nb)
    prepared=_request(point,calibration,expected,pc,kc,kc.context["case_read_admission_sha256"];nominal=numeric.alpha)
    need(c["schema"]=="archival_static_calibration_v1" && c["request"]==prepared.request &&
        c["request_sha256"]==digest(prepared.bytes) && c["numeric_sha256"]==expected_static.numeric_sha256,
        "archival static fixed request differs")
    need(length(numeric.means)==(kc.context["layout"]=="ground-only" ? 11 : 8),"archival static layout width differs")
    counts=c["regime_bin_counts"];rows=kc.context["rows"]
    need(counts isa Vector && length(counts)==4 && all(x->F.isint(x)&&0<=x<=rows,counts) &&
        sum(counts)==rows && c["empty_regime_bins"]==findall(==(0),counts),"archival static regime counts differ")
    J._object(r,Set(("schema","complete","context_sha256","numeric_sha256","actual_source_access","actual_point_model_fits",
        "uncertainty_parameter_fits","provider_receipts_available","software_qualification","scientific_coverage_qualified","authorization")))
    need(r["schema"]=="archival_static_calibration_receipt_v1" && r["complete"]===true && r["context_sha256"]==expected_static.context_sha256 &&
        r["numeric_sha256"]==expected_static.numeric_sha256 && r["actual_source_access"]===true &&
        F.isint(r["actual_point_model_fits"]) && r["actual_point_model_fits"]==0 && F.isint(r["uncertainty_parameter_fits"]) && r["uncertainty_parameter_fits"]==1 &&
        r["provider_receipts_available"]===false && r["software_qualification"]===false && r["scientific_coverage_qualified"]===false,
        "archival static receipt differs")
    auth=r["authorization"];ca=kc.receipt["authorization"]
    J._object(auth,Set(("schema","head","grant_sha256","source_proof_sha256","qualification_sha256","request_sha256","case_read_admission_sha256",
        "calibration_grant_sha256","calibration_source_proof_sha256","calibration_qualification_sha256","calibration_attempt_sha256",
        "uncertainty_fit_authorized","actual_point_model_fit_authorized","provider_receipts_available","attempt_sha256")))
    need(auth["schema"]=="archival_static_fit_admission_v1" && auth["request_sha256"]==c["request_sha256"] &&
        auth["case_read_admission_sha256"]==prepared.request["case_read_admission_sha256"] && auth["head"]==ca["head"] &&
        auth["uncertainty_fit_authorized"]===true && auth["actual_point_model_fit_authorized"]===false && auth["provider_receipts_available"]===false,
        "archival static authorization differs")
    for key in ("grant_sha256","source_proof_sha256","qualification_sha256","attempt_sha256")
        J._sha(auth[key]);need(auth["calibration_"*key]==ca[key],"archival static calibration producer differs")
    end
    (;context=c,numeric,context_bytes=cb,numeric_bytes=nb,receipt_bytes=rb,
      context_sha256=expected_static.context_sha256,numeric_sha256=expected_static.numeric_sha256)
end
"""Predict from frozen data only; this operation grants no new source or fit access."""
function predict_upper(value::ArchivalStaticCapsule,X,centers,scales,peaks;point,calibration,expected,expected_static,start)
    v=check_calibration(value,point,calibration,expected;expected_static);request=v.context["request"]
    need(start isa DateTime && D.whole(start) && F._fit_clock(request["splits"]["calibration_stop_exclusive"])<=start &&
        F._fit_clock(request["splits"]["first_evaluation_issue"])<=start,"archival static prediction precedes evaluation")
    result=S._predict(v.numeric,X,centers,scales,peaks)
    (;result...,methods=S.METHODS,context_sha256=v.context_sha256,numeric_sha256=v.numeric_sha256,
        actual_source_authority_granted=false,actual_point_model_fits=0,scientific_coverage_qualified=false)
end
include("source_bound_static_artifact_codec.jl")
"""Publish checked archival numeric data and write completion last."""
function publish_static(value::ArchivalStaticCapsule,point,calibration,expected,directory;expected_static)
    v=check_calibration(value,point,calibration,expected;expected_static)
    prepared=_prepare_publication(directory,v.context_bytes,v.numeric_bytes,v.receipt_bytes,expected_static)
    check_calibration(value,point,calibration,expected;expected_static)
    _complete_publication(prepared)
end
"""Read bounded numeric data using outside pins, without source authority or fitting."""
function read_static(directory,point,calibration,expected;expected_static,manifest_sha256,receipt_sha256,complete_sha256)
    v=_read_publication(directory,expected_static;manifest_sha256,receipt_sha256,complete_sha256)
    capsule=ArchivalStaticCapsule(v.context,v.numeric,v.receipt,_Token())
    check_calibration(capsule,point,calibration,expected;expected_static);capsule
end
end
