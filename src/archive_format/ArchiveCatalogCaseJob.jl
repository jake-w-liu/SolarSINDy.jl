"""Sealed source-bound artifact publication and independent verified reading.

Archival features require a separate qualified feature-only authority. Model
fitting is outside this API.
Every reader recomputes the complete feature and label bytes from the separately
pinned source snapshot; no unchecked binary object reconstruction is exposed.
"""
module ArchiveCatalogCaseJob
using Dates,SHA
import ..ArchiveCatalogSourceJob, ..CompletedSourceDBDT, ..ArchiveCatalogCases
import ..ArchiveCatalogSnapshotFilesystem
import ..ArchiveCatalogCaseAuthority
const F=ArchiveCatalogSnapshotFilesystem
const J=ArchiveCatalogSourceJob
const R=CompletedSourceDBDT.CompletedRefitArtifacts
const C=ArchiveCatalogCases
const V=F.V
"""Caller pins for a feature-only grant, locked source agreement and qualified worker."""
const Authority=ArchiveCatalogCaseAuthority.Authority
export Authority, VerifiedCases, publish, read_verified, split_arrays
const ROWS_PER_SHARD=8192
const MAX_TABLES=16
const MAX_TOTAL_ROWS=1_320_000
struct _Checked end
"""Immutable verified case schedules and labels with no model-fit authority."""
struct VerifiedCases
    request_json::String
    source::NamedTuple
    features::Tuple
    labels::Tuple
    request_sha256::String
    manifest_sha256::String
    complete_sha256::String
    actual_source_authority_granted::Bool
    actual_fit_performed::Bool
    function VerifiedCases(::_Checked,args...);new(args...);end
end
need(ok,message)=ok ? nothing : throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(IOBuffer(bytes)))
function _clock(value)
    J._clock(value)
end
function _request(path,pin)
    J._ordinary(path);before=J._identity(path)
    bytes=R.bounded_read(path;limit=131072)
    need(J._identity(path)==before && digest(bytes)==J._sha(pin),"case job request pin")
    value=R.decode_canonical(bytes;limit=131072)
    J._object(value,Set(("schema","evidence_mode","layout","role","runtime","source","tables")))
    generated=value["schema"]=="generated_catalog_case_request_v1" && value["evidence_mode"]=="generated"
    archival=value["schema"]=="archive_catalog_case_request_v1" && value["evidence_mode"]=="actual_archival_cache"
    need(generated || archival,"generated case job authority required")
    need(value["runtime"]==string(VERSION) && value["runtime"] in ("1.12.6","1.13.1"),"case job native runtime")
    need(value["layout"] in ("ground-only","mixed") && value["role"] in ("primary","independent"),"case job context")
    J._object(value["source"],Set(("directory","manifest_sha256","complete_sha256")))
    J._text(value["source"]["directory"]);J._sha(value["source"]["manifest_sha256"]);J._sha(value["source"]["complete_sha256"])
    tables=value["tables"];need(tables isa AbstractVector && 1<=length(tables)<=MAX_TABLES,"case job table count")
    names=Set{String}();total=0
    for table in tables
        J._object(table,Set(("id","station","start_inclusive","stop_exclusive","label_asof")))
        name=V._name(table["id"]);need(!(lowercase(name) in names),"case job table identity");push!(names,lowercase(name))
        need(table["station"] in ("CMO","FRD"),"case job station")
        start=_clock(table["start_inclusive"]);stop=_clock(table["stop_exclusive"]);asof=_clock(table["label_asof"])
        rows=Dates.value(stop-start)÷60000
        need(start<stop && 0<rows<=600000 && stop<=asof,"case job issue domain")
        need(total<=MAX_TOTAL_ROWS-rows,"case job cumulative schedule bound");total+=rows
    end
    value,bytes,before
end
function _source(request,binding)
    source=request["source"]
    archival=request["evidence_mode"]=="actual_archival_cache"
    need(archival ? binding isa AbstractDict && binding["operation"]=="sealed_features_labels_only" : binding===nothing,
         "case source authority context differs")
    F._load_pinned(source["directory"];manifest_sha256=source["manifest_sha256"],complete_sha256=source["complete_sha256"],
                  role=request["role"],layout=request["layout"],runtime=request["runtime"],evidence_mode=request["evidence_mode"])
end
function _features(request,source)
    output=Any[]
    for table in request["tables"]
        ground=getproperty(source.catalogs,Symbol("ground-"*table["station"]))
        mode=request["layout"]=="ground-only" ? :ground11 : :mixed8
        wind=mode===:mixed8 ? getproperty(source.catalogs,Symbol("wind-"*table["station"])) : nothing
        issues=collect(_clock(table["start_inclusive"]):Minute(1):(_clock(table["stop_exclusive"])-Minute(1)))
        push!(output,C.seal_catalog_features(ground,issues;mode,wind))
    end
    output
end
function _labels(request,source,features)
    [C.audit_catalog_labels(features[index],getproperty(source.catalogs,Symbol("ground-"*table["station"]));asof=_clock(table["label_asof"]))
     for (index,table) in enumerate(request["tables"])]
end
function _metadata(table,features,labels,feature_shards,label_shards)
    Dict("declaration"=>table,"scheduled"=>length(features),"mode"=>String(features.mode),
         "feature_payload_sha256"=>features.payload_sha256,"feature_schedule_sha256"=>features.schedule_sha256,
         "label_payload_sha256"=>labels.payload_sha256,"label_audit_sha256"=>labels.audit_sha256,
         "counts"=>Dict(string(key)=>value for (key,value) in pairs(C.case_counts(features,labels))),
         "feature_shards"=>feature_shards,"label_shards"=>label_shards)
end
"""Publish feature tables first, then separately matured labels.

The completion marker is the last exclusive write. Failures retain the partial
directory. Actual sources need a feature-only grant; source-only grants and
unqualified code cannot authorize this operation.
"""
function publish(request_path,destination;request_sha256,authority=nothing)
    request,request_bytes,request_identity=_request(request_path,request_sha256)
    binding=ArchiveCatalogCaseAuthority.authorize(request,request_sha256,authority)
    need(isabspath(destination) && normpath(destination)==destination,"case job destination path")
    J._ordinary(dirname(destination);directory=true)
    need(!ispath(destination) && !islink(destination),"case job destination exists")
    source=_source(request,binding)
    features=_features(request,source) # Entire feature schedule sealed before any target query.
    mkdir(destination);members=Any[];total=0
    function put(name,bytes)
        need(length(members)<1022 && ncodeunits(bytes)<=8_388_608 && total<=1_073_741_824-ncodeunits(bytes),"case job artifact bound")
        R._exclusive_write(joinpath(destination,name),Vector{UInt8}(codeunits(bytes)))
        push!(members,Dict("name"=>name,"bytes"=>ncodeunits(bytes),"sha256"=>digest(bytes)));total+=ncodeunits(bytes)
        name
    end
    function shards(index,kind,payload,width,rows)
        names=String[]
        for first in 1:ROWS_PER_SHARD:rows
            last=min(first+ROWS_PER_SHARD-1,rows)
            name="table-"*lpad(string(index),2,'0')*"-"*kind*"-"*lpad(string(length(names)+1),4,'0')*".bin"
            push!(names,put(name,String(Vector{UInt8}(codeunits(payload)[(first-1)*width+1:last*width]))))
        end
        names
    end
    fs=[shards(index,"features",value.payload,160,length(value)) for (index,value) in enumerate(features)]
    labels=_labels(request,source,features)
    ls=[shards(index,"labels",value.payload,64,length(value)) for (index,value) in enumerate(labels)]
    need(J._identity(request_path)==request_identity && R.bounded_read(request_path;limit=131072)==request_bytes,"case job request changed")
    archival=binding!==nothing
    metadata=Dict("schema"=>(archival ? "archive_catalog_case_artifact_v1" : "generated_catalog_case_artifact_v1"),"request_sha256"=>request_sha256,"source"=>request["source"],
        "runtime"=>request["runtime"],"role"=>request["role"],"layout"=>request["layout"],
        "tables"=>[_metadata(table,features[index],labels[index],fs[index],ls[index]) for (index,table) in enumerate(request["tables"])],
        "members"=>members,"actual_source_authority_granted"=>archival,"provider_receipts_available"=>false,"actual_fits"=>0)
    archival && (metadata["authority"]=binding)
    bytes=R.canonical_bytes(metadata;limit=1_048_576)
    need(total<=1_073_741_824-length(bytes)-4096,"case job artifact metadata bound")
    R._exclusive_write(joinpath(destination,"manifest.json"),bytes)
    marker=Dict("schema"=>(archival ? "archive_catalog_case_complete_v1" : "generated_catalog_case_complete_v1"),"manifest_sha256"=>digest(bytes),"members"=>length(members)+2,
                "snapshot_bytes_without_marker"=>total+length(bytes),"actual_fits"=>0)
    R._exclusive_write(joinpath(destination,"COMPLETE.json"),R.canonical_bytes(marker;limit=4096))
    metadata
end
"""Verify all persisted bytes by recomputation from the separate pinned source.

Rehashed malformed clocks, reason/status bytes, padding, features, labels and
support hashes cannot pass merely by making member hashes self-consistent.
"""
function read_verified(request_path,directory;request_sha256,manifest_sha256,complete_sha256,authority=nothing)
    request,request_bytes,request_identity=_request(request_path,request_sha256)
    binding=ArchiveCatalogCaseAuthority.authorize(request,request_sha256,authority)
    _read_verified_bound(request_path,directory,request_sha256,manifest_sha256,complete_sha256,
        request,request_bytes,request_identity,binding)
end
function _read_verified_bound(request_path,directory,request_sha256,manifest_sha256,complete_sha256,
                              request,request_bytes,request_identity,binding)
    archival=binding!==nothing
    J._ordinary(directory;directory=true)
    names=readdir(directory);need(2<=length(names)<=1024,"case reader filesystem member bound")
    text,mi=F._read(joinpath(directory,"manifest.json"),1_048_576)
    marker_text,ci=F._read(joinpath(directory,"COMPLETE.json"),4096)
    need(digest(text)==J._sha(manifest_sha256) && digest(marker_text)==J._sha(complete_sha256),"case reader metadata pin")
    metadata=R.decode_canonical(Vector{UInt8}(codeunits(text));limit=1_048_576)
    marker=R.decode_canonical(Vector{UInt8}(codeunits(marker_text));limit=4096)
    fields=Set(("schema","request_sha256","source","runtime","role","layout","tables","members","actual_source_authority_granted","provider_receipts_available","actual_fits"))
    archival && push!(fields,"authority")
    J._object(metadata,fields)
    J._object(marker,Set(("schema","manifest_sha256","members","snapshot_bytes_without_marker","actual_fits")))
    need(metadata["schema"]==(archival ? "archive_catalog_case_artifact_v1" : "generated_catalog_case_artifact_v1") &&
         marker["schema"]==(archival ? "archive_catalog_case_complete_v1" : "generated_catalog_case_complete_v1"),"case reader schema")
    archival && need(metadata["authority"]==binding,"case reader authority binding differs")
    need(metadata["request_sha256"]==request_sha256 && metadata["source"]==request["source"] &&
         all(metadata[key]==request[key] for key in ("runtime","role","layout")),"case reader request binding")
    need(metadata["actual_source_authority_granted"]===archival && metadata["provider_receipts_available"]===false &&
         V._count(metadata["actual_fits"],0)==V._count(marker["actual_fits"],0)==0,"case reader evidence flags")
    need(marker["manifest_sha256"]==manifest_sha256,"case reader completion binding")
    entries=metadata["members"]
    need(entries isa AbstractVector && 1<=length(entries)<=1022,"case reader descriptor bound")
    allowed=String["manifest.json","COMPLETE.json"];seen=Set(lowercase.(allowed));total=ncodeunits(text)
    for entry in entries
        V._object(entry,("name","bytes","sha256"));name=V._name(entry["name"])
        need(!(lowercase(name) in seen),"case reader duplicate member");push!(seen,lowercase(name));push!(allowed,name)
        count=V._count(entry["bytes"],8_388_608);V._sha(entry["sha256"])
        need(total<=1_073_741_824-count,"case reader byte bound");total+=count
    end
    need(Set(allowed)==Set(names) && V._count(marker["members"],1024)==length(names) &&
         V._count(marker["snapshot_bytes_without_marker"],1_073_741_824)==total,"case reader fileset/count differs")
    tables=metadata["tables"];need(tables isa AbstractVector && length(tables)==length(request["tables"]),"case reader table count")
    source=_source(request,binding);features=_features(request,source);labels=_labels(request,source,features)
    used=String[];identities=Dict{String,Any}()
    for (index,table) in enumerate(tables)
        J._object(table,Set(("declaration","scheduled","mode","feature_payload_sha256","feature_schedule_sha256","label_payload_sha256","label_audit_sha256","counts","feature_shards","label_shards")))
        f=features[index];l=labels[index]
        for (kind,payload,width,field) in (("features",f.payload,160,"feature_shards"),("labels",l.payload,64,"label_shards"))
            shards=table[field];need(shards isa AbstractVector && length(shards)==cld(length(f),ROWS_PER_SHARD),"case reader shard count")
            for (ordinal,name) in enumerate(shards)
                V._name(name)
                need(name=="table-"*lpad(string(index),2,'0')*"-"*kind*"-"*lpad(string(ordinal),4,'0')*".bin","case reader shard identity")
                descriptor=filter(entry->entry["name"]==name,entries);need(length(descriptor)==1,"case reader shard absent")
                first=(ordinal-1)*ROWS_PER_SHARD+1;last=min(ordinal*ROWS_PER_SHARD,length(f))
                bytes,identity=F._read(joinpath(directory,name),8_388_608)
                need(ncodeunits(bytes)==(last-first+1)*width==only(descriptor)["bytes"] && digest(bytes)==only(descriptor)["sha256"],"case reader shard bytes/hash")
                need(codeunits(bytes)==codeunits(payload)[(first-1)*width+1:last*width],"case reader recomputed payload differs")
                push!(used,name);identities[name]=identity
            end
        end
        expected=_metadata(request["tables"][index],f,l,table["feature_shards"],table["label_shards"])
        # Explicit count checks reject Bool values despite Bool == Int equality.
        V._count(table["scheduled"],600000;minimum=1)
        J._object(table["counts"],Set(keys(expected["counts"])))
        for count in values(table["counts"]);V._count(count,600000);end
        need(table==expected,"case reader recomputed metadata differs")
    end
    need(Set(used)==Set(entry["name"] for entry in entries) && length(used)==length(entries),"case reader unused member")
    need(readdir(directory)==names && J._identity(joinpath(directory,"manifest.json"))==mi && J._identity(joinpath(directory,"COMPLETE.json"))==ci &&
         all(J._identity(joinpath(directory,name))==identity for (name,identity) in identities),"case reader artifact changed")
    need(J._identity(request_path)==request_identity && R.bounded_read(request_path;limit=131072)==request_bytes,"case reader request changed")
    VerifiedCases(_Checked(),String(copy(request_bytes)),source,Tuple(features),Tuple(labels),request_sha256,manifest_sha256,complete_sha256,archival,false)
end
"""Project a strict issue-domain split without opening out-of-domain targets.

Both issue membership and strict label readiness are checked before each target
query. The full-year maturity cutoff does not authorize cross-split targets.
"""
function _split_arrays(value::VerifiedCases,index,start,stop,asof,target)
    request=R.decode_canonical(Vector{UInt8}(codeunits(value.request_json));limit=131072)
    need(value.actual_fit_performed===false &&
         value.actual_source_authority_granted===(request["evidence_mode"]=="actual_archival_cache"),"case split context")
    need(index isa Integer && !(index isa Bool) && 1<=index<=length(value.features),"case split table index")
    need(start isa DateTime && stop isa DateTime && asof isa DateTime && start<stop<=asof,"case split clocks")
    f=value.features[index];labels=value.labels[index];C._matching(f,labels)
    need(C.feature_row(f,1).issue<=start<stop<=C.feature_row(f,length(f)).issue+Minute(1) && asof<=labels.asof,"case split exceeds sealed artifact domain")
    rows=Int[];issues=DateTime[];features=Vector{Float64}[];targets=Float64[];now=Float64[];scales=Float64[]
    for row in 1:length(f)
        feature=C.feature_row(f,row)
        start<=feature.issue<stop && feature.label_ready<stop && feature.label_ready<=asof && feature.accepted || continue
        label=target(labels,row)
        need(label.issue==feature.issue && label.label_ready==feature.label_ready,"case split label binding")
        label.accepted || continue
        push!(rows,row);push!(issues,feature.issue);push!(features,collect(feature.features));push!(targets,label.target)
        push!(now,feature.raw_now);push!(scales,feature.scale)
    end
    width=f.mode===:ground11 ? 11 : 8;matrix=Matrix{Float64}(undef,length(rows),width)
    for row in eachindex(rows),column in 1:width;matrix[row,column]=features[row][column];end
    (;rows,issues,features=matrix,targets,raw_now=now,scales,label_ready_before=stop,
      availability_kind=:assumed_completed_source_intervals,measured_receipts_available=false,
      actual_source_authority_granted=value.actual_source_authority_granted,actual_fit_performed=false)
end
"""Return owned split arrays after issue-domain and strict label-readiness checks; no fit occurs."""
split_arrays(value::VerifiedCases,index;start,stop,asof)=_split_arrays(value,index,start,stop,asof,C.label_row)
end
