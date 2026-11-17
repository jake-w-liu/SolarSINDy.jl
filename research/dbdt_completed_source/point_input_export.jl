"""Fixed-split data-only export for independent point-fit input planning.

This module grants no fit, discovery, selection or scientific authority.
"""
module PointInputExport
using Dates,SHA,SolarSINDy
const F=SolarSINDy.ArchiveSourceBoundPointFit
const Q=SolarSINDy.ArchiveCatalogCaseJob
const B=SolarSINDy.ArchiveCatalogCaseReader
const J=SolarSINDy.ArchiveCatalogSourceJob
const R=SolarSINDy.CompletedSourceDBDT.CompletedRefitArtifacts
const ENTRY_PIN=bytes2hex(sha256(read(@__FILE__)))
const CHUNK_BYTES=8_388_608
const MAX_BYTES=134_217_728
export planning_request,export_generated,export_archival
need(ok,message)=ok || throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(bytes))
function _request(value::Q.VerifiedCases,index,start,stop,asof;admission=nothing)
    actual=admission!==nothing
    need(value.actual_source_authority_granted===actual && value.actual_fit_performed===false,"planning case authority differs")
    need(F.isint(index)&&1<=index<=length(value.features),"planning table index")
    need(all(t->t isa DateTime && SolarSINDy.CompletedSourceDBDT.whole(t),(start,stop,asof)) && start<stop<=asof,"planning split clocks")
    need(digest(read(@__FILE__))==ENTRY_PIN,"planning entry source changed")
    request=R.decode_canonical(Vector{UInt8}(codeunits(value.request_json));limit=131072)
    need(digest(codeunits(value.request_json))==value.request_sha256 && request["runtime"]==string(VERSION),"planning source request differs")
    need(request["evidence_mode"]==(actual ? "actual_archival_cache" : "generated"),"planning evidence mode differs")
    table=request["tables"][index]
    need(J._clock(table["start_inclusive"])<=start<stop<=J._clock(table["stop_exclusive"]) && asof<=J._clock(table["label_asof"]),"planning split exceeds case table")
    result=Dict("schema"=>"source_bound_point_input_planning_request_v1","evidence_mode"=>request["evidence_mode"],
        "case_request_sha256"=>value.request_sha256,"case_manifest_sha256"=>value.manifest_sha256,"case_complete_sha256"=>value.complete_sha256,
        "table_index"=>Int(index),"table"=>table,"runtime"=>string(VERSION),"role"=>request["role"],"layout"=>request["layout"],
        "start_inclusive"=>F.clock(start),"stop_exclusive"=>F.clock(stop),"target_asof"=>F.clock(asof),
        "entry_sha256"=>ENTRY_PIN,"numeric_source_sha256"=>F.source_identity(),"library_sha256"=>F.LIBRARY_SHA,
        "feature_payload_sha256"=>value.features[index].payload_sha256,"label_payload_sha256"=>value.labels[index].payload_sha256,
        "actual_model_fit_authorized"=>false,"provider_receipts_available"=>false)
    actual && (result["case_read_admission_sha256"]=digest(B.admission_bytes(admission)))
    result
end
"""Plan only metadata of a generated input export; this reads no new labels."""
planning_request(value::Q.VerifiedCases,index;start,stop,asof)=_request(value,index,start,stop,asof)
"""Plan metadata of a reverified2017input export without granting access or fitting."""
function planning_request(value::B.ReverifiedCases,index;start,stop,asof)
    F._archival_request(value,index,start,stop,asof)
    _request(value.cases,index,start,stop,asof;admission=value)
end
function _stream(arrays)
    expected,y=F._data_sha(arrays);n,p=size(arrays.features)
    bytes=length(codeunits("SOURCE_BOUND_POINT_INPUT_V1\n"))+16+8n*(p+6)
    need(bytes<=MAX_BYTES,"planning input exceeds stream bound")
    io=IOBuffer(sizehint=bytes)
    write(io,"SOURCE_BOUND_POINT_INPUT_V1\n",htol.(UInt64[n,p]),htol.(UInt64.(arrays.rows)),
        htol.(reinterpret(UInt64,Int64[Dates.value(t) for t in arrays.issues])))
    for column in axes(arrays.features,2)
        values=collect(view(arrays.features,:,column));write(io,htol.(reinterpret(UInt64,values)))
    end
    for values in (arrays.targets,y,arrays.raw_now,arrays.scales);write(io,htol.(reinterpret(UInt64,values)));end
    body=take!(io)
    need(length(body)==bytes && digest(body)==expected,"export stream differs from native fit input hash")
    body,expected
end
function _export(value,index,destination,start,stop,asof,request,auth)
    need(isabspath(destination) && normpath(destination)==destination,"planning destination must be canonical absolute path")
    J._ordinary(dirname(destination);directory=true)
    need(!ispath(destination)&&!islink(destination),"planning export destination exists")
    actual=auth!==nothing
    started=time_ns();started_utc=F.clock(now(UTC));available=Sys.free_memory()
    need(value.actual_source_authority_granted===actual,"planning output evidence differs")
    arrays=Q.split_arrays(value,index;start,stop,asof);body,input=_stream(arrays)
    context=Dict("schema"=>"source_bound_point_input_export_context_v1","request"=>request,
        "rows"=>length(arrays.rows),"columns"=>size(arrays.features,2),"first_row"=>first(arrays.rows),"last_row"=>last(arrays.rows),
        "input_sha256"=>input,"input_bytes"=>length(body),"strict_label_ready_before"=>F.clock(stop),
        "target_transform"=>"log1p_rate_reference_1_nt_per_min","availability_kind"=>"assumed_completed_source_intervals",
        "measured_receipts_available"=>false)
    context_bytes=R.canonical_bytes(context;limit=262144)
    receipt=Dict("schema"=>"source_bound_point_input_export_receipt_v1","complete"=>true,"context_sha256"=>digest(context_bytes),
        "input_sha256"=>input,"actual_source_access"=>actual,"actual_model_fit_performed"=>false,
        "fit_authority_granted"=>false,"provider_receipts_available"=>false,"software_qualification"=>false)
    if actual
        binding=copy(auth)
        binding["runtime_observations"]=Dict("started_utc"=>started_utc,"elapsed_ns"=>Int(time_ns()-started),
            "available_physical_bytes_at_start"=>Int(available),"peak_rss_bytes"=>Int(Sys.maxrss()),
            "julia_threads"=>Threads.nthreads(),"blas_threads"=>F.BLAS.get_num_threads())
        receipt["authorization"]=binding
    end
    receipt_bytes=R.canonical_bytes(receipt;limit=65536)
    mkdir(destination);J._ordinary(destination;directory=true);members=Any[]
    for (name,bytes) in (("context.json",context_bytes),("receipt.json",receipt_bytes))
        R._exclusive_write(joinpath(destination,name),bytes)
        push!(members,Dict("name"=>name,"bytes"=>length(bytes),"sha256"=>digest(bytes)))
    end
    chunks=0
    for start_byte in 1:CHUNK_BYTES:length(body)
        chunks+=1;bytes=body[start_byte:min(start_byte+CHUNK_BYTES-1,length(body))]
        name="input-"*lpad(chunks,4,'0')*".bin"
        R._exclusive_write(joinpath(destination,name),bytes)
        push!(members,Dict("name"=>name,"bytes"=>length(bytes),"sha256"=>digest(bytes)))
    end
    manifest=R.canonical_bytes(Dict("schema"=>"source_bound_point_input_export_manifest_v1","context_sha256"=>digest(context_bytes),
        "input_sha256"=>input,"input_bytes"=>length(body),"chunk_bytes"=>CHUNK_BYTES,"chunks"=>chunks,"members"=>members);limit=65536)
    R._exclusive_write(joinpath(destination,"manifest.json"),manifest)
    need(digest(read(@__FILE__))==ENTRY_PIN && F.source_identity()==request["numeric_source_sha256"],"planning code changed during export")
    marker=R.canonical_bytes(Dict("schema"=>"source_bound_point_input_export_complete_v1","manifest_sha256"=>digest(manifest),
        "context_sha256"=>digest(context_bytes),"receipt_sha256"=>digest(receipt_bytes),"input_sha256"=>input,
        "members"=>length(members)+2,"actual_fits"=>0);limit=4096)
    result=(;context_sha256=digest(context_bytes),manifest_sha256=digest(manifest),complete_sha256=digest(marker),receipt_sha256=digest(receipt_bytes),
      input_sha256=input,rows=length(arrays.rows),columns=size(arrays.features,2),chunks,actual_source_access=actual,actual_fits=0)
    (;result,marker)
end
function _complete(destination,prepared)
    R._exclusive_write(joinpath(destination,"COMPLETE.json"),prepared.marker)
    prepared.result
end
"""Export generated-only inputs with no actual permissions or model fitting."""
function export_generated(value::Q.VerifiedCases,index,destination;start,stop,asof)
    request=planning_request(value,index;start,stop,asof)
    _complete(destination,_export(value,index,destination,start,stop,asof,request,nothing))
end
"""Export one strict 2017 input stream from reverified archival cases."""
function export_archival(value::B.ReverifiedCases,index,destination;start,stop,asof,authority=nothing)
    need(authority===nothing,"input export uses case/input bindings, not execution permits")
    F._archival_request(value,index,start,stop,asof)
    request=planning_request(value,index;start,stop,asof)
    auth=Dict("schema"=>"archive_point_input_binding_v2","request_sha256"=>digest(R.canonical_bytes(request;limit=131072)),
        "case_read_admission_sha256"=>request["case_read_admission_sha256"],"entry_sha256"=>ENTRY_PIN)
    prepared=_export(value.cases,index,destination,start,stop,asof,request,auth)
    F._archival_request(value,index,start,stop,asof)
    need(planning_request(value,index;start,stop,asof)==request,"planning inputs changed during export")
    _complete(destination,prepared)

end
end
