"""Pure reconstruction of caller-supplied source-snapshot bytes.

No file, network, feature or model access occurs here. All input buffers are
owned before validation. A returned catalog grants no actual-source authority.
"""
module ArchiveCatalogSnapshotValues
using Dates,SHA
import ..ArchiveCatalogSourceJob, ..CompletedSourceDBDT, ..ArchiveFormatParsers, ..ArchiveFormatAudit, ..ArchiveSourceCatalog, ..ArchiveCatalogAudit
const J=ArchiveCatalogSourceJob
const R=CompletedSourceDBDT.CompletedRefitArtifacts
const P=ArchiveFormatParsers
const A=ArchiveFormatAudit
const C=ArchiveSourceCatalog
const I=ArchiveCatalogAudit
const MEMBER_LIMIT=8_388_608
const TOTAL_LIMIT=1_073_741_824
need(ok,message)=ok ? nothing : throw(ArgumentError(message))
_hash(value)=bytes2hex(sha256(IOBuffer(value)))
function _sha(value)
    need(value isa String && occursin(r"^[0-9a-f]{64}$",value),"snapshot digest syntax")
    value
end
function _count(value,maximum;minimum=0)
    need(value isa Integer && !(value isa Bool) && minimum<=value<=maximum,"snapshot integer bound")
    Int(value)
end
function _object(value,fields)
    need(value isa AbstractDict && Set(keys(value))==Set(fields),"snapshot object fields")
    value
end
function _name(value)
    need(value isa String && 1<=ncodeunits(value)<=128 && occursin(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",value) && !occursin("..",value),"snapshot member name")
    value
end
function _owned(value,limit)
    need(value isa String || value isa Vector{UInt8},"ordinary snapshot byte buffer required")
    bytes=value isa String ? ncodeunits(value) : length(value)
    need(bytes<=limit,"snapshot buffer size")
    value isa String ? String(copy(codeunits(value))) : String(copy(value))
end

"""Validate and reconstruct explicitly supplied snapshot members without any I/O.

Expected metadata hashes come from the caller's separate evidence binding.
Recorded cache paths are passive declaration fields and are never opened.
All original text, raw ledgers and selected witnesses must agree numerically.
"""
function reconstruct(manifest_bytes,complete_bytes,provided;manifest_sha256,complete_sha256,
                     expected_layout,expected_role,expected_runtime,expected_evidence_mode)
    need(expected_layout in ("ground-only","mixed") && expected_role in ("primary","independent") &&
         expected_runtime in ("1.12.6","1.13.1") && expected_evidence_mode in ("generated","actual_archival_cache"),"snapshot context differs")
    text=_owned(manifest_bytes,1_048_576);marker_text=_owned(complete_bytes,4096)
    need(_hash(text)==_sha(manifest_sha256) && _hash(marker_text)==_sha(complete_sha256),"snapshot metadata pin differs")
    m=R.decode_canonical(Vector{UInt8}(codeunits(text));limit=1_048_576)
    marker=R.decode_canonical(Vector{UInt8}(codeunits(marker_text));limit=4096)
    _object(m,("schema","request_sha256","evidence_mode","layout","role","runtime","bodies","catalogs","members","read_trace",
               "provider_receipts_available","independent_source_qualification","actual_fits"))
    _object(marker,("schema","manifest_sha256","members","snapshot_bytes_without_marker","independent_source_qualification","actual_fits"))
    need(m["schema"]=="archive_catalog_snapshot_v1" && marker["schema"]=="archive_catalog_snapshot_complete_v1","snapshot schema")
    need(m["layout"]==expected_layout && m["role"]==expected_role && m["runtime"]==expected_runtime && m["evidence_mode"]==expected_evidence_mode,"snapshot context binding")
    need(m["provider_receipts_available"]===false && m["independent_source_qualification"]===false && marker["independent_source_qualification"]===false,"snapshot evidence flags")
    need(_count(m["actual_fits"],0)==_count(marker["actual_fits"],0)==0 && marker["manifest_sha256"]==manifest_sha256,"snapshot completion binding")
    _sha(m["request_sha256"])
    descriptors=m["members"]
    need(descriptors isa AbstractVector && 1<=length(descriptors)<=1022 && provided isa Dict{String,String} && length(provided)==length(descriptors),"snapshot members shape")
    expected_names=String[];seen=Set{String}();total=ncodeunits(text)+ncodeunits(marker_text)
    # Own all supplied strings before checking their hashes or parsing bodies.
    members=Dict{String,String}()
    for descriptor in descriptors
        _object(descriptor,("name","bytes","sha256"))
        name=_name(descriptor["name"]);key=lowercase(name)
        need(!(key in seen) && haskey(provided,name),"snapshot duplicate or absent member")
        push!(seen,key);push!(expected_names,name)
        bytes=_count(descriptor["bytes"],MEMBER_LIMIT)
        need(total<=TOTAL_LIMIT-bytes,"snapshot cumulative byte limit")
        owned=_owned(provided[name],MEMBER_LIMIT)
        need(ncodeunits(owned)==bytes && _hash(owned)==_sha(descriptor["sha256"]),"snapshot member bytes differ")
        total+=bytes;members[name]=owned
    end
    need(Set(keys(provided))==Set(expected_names),"snapshot member fileset differs")
    need(_count(marker["members"],1024)==length(members)+2 &&
         _count(marker["snapshot_bytes_without_marker"],TOTAL_LIMIT)==total-ncodeunits(marker_text),"snapshot total bytes/count differs")
    bodies=m["bodies"];need(bodies isa AbstractVector && 1<=length(bodies)<=49,"snapshot body count")
    need(m["read_trace"] isa AbstractVector && length(m["read_trace"])==length(bodies),"snapshot source trace count")
    for entry in m["read_trace"]
        _object(entry,("logical_id","bytes","sha256"))
        J._text(entry["logical_id"];limit=128);_count(entry["bytes"],MEMBER_LIMIT;minimum=1);_sha(entry["sha256"])
    end
    required=expected_layout=="ground-only" ? Set(("ground-CMO","ground-FRD")) : Set(("ground-CMO","ground-FRD","wind-CMO","wind-FRD"))
    need(m["catalogs"] isa AbstractDict && Set(keys(m["catalogs"]))==required,"snapshot declared catalog set")
    for expected in values(m["catalogs"])
        _object(expected,("kind","station","product","body_ids","shards","selected_sha256","raw_catalog_sha256","raw_rows","rows","duplicates","complete_vectors","declared_minutes","missing_minutes","uncovered_minutes"))
        for key in ("raw_rows","rows","duplicates","complete_vectors")
            _count(expected[key],1_600_000)
        end
        for key in ("declared_minutes","missing_minutes","uncovered_minutes")
            _count(expected[key],typemax(Int))
        end
        _sha(expected["selected_sha256"]);_sha(expected["raw_catalog_sha256"])
        need(expected["body_ids"] isa AbstractVector && 1<=length(expected["body_ids"])<=32,"snapshot catalog body identities")
        for id in expected["body_ids"];_sha(id);end
        need(expected["shards"] isa AbstractVector && length(expected["shards"])==cld(expected["rows"],8192),"snapshot shard geometry")
        for name in expected["shards"];_name(name);end
    end
    groups=Dict{String,Vector{Any}}();trace=Any[];ids=Set{String}();used=Set{String}()
    primary=expected_role=="primary"
    for (index,body_info) in enumerate(bodies)
        _object(body_info,("logical_id","original","raw","body_ids","declaration"))
        declaration=_object(body_info["declaration"],J.INPUT_KEYS)
        id=J._text(declaration["logical_id"];limit=128)
        J._text(declaration["path"]);J._text(declaration["role"];limit=128);J._text(declaration["provenance"])
        need(body_info["logical_id"]==id && !(id in ids),"snapshot logical identity differs");push!(ids,id)
        original=_name(body_info["original"]);raw_name=_name(body_info["raw"])
        need(original=="body-"*lpad(string(index),3,'0')*".txt" && raw_name=="raw-"*lpad(string(index),3,'0')*".bin","snapshot source member order")
        need(haskey(members,original)&&haskey(members,raw_name),"snapshot body member absent")
        push!(used,original);push!(used,raw_name)
        source=members[original]
        need(ncodeunits(source)==_count(declaration["bytes"],MEMBER_LIMIT;minimum=1) && _hash(source)==_sha(declaration["sha256"]),"snapshot original body binding")
        kind=declaration["kind"];labels=declaration["stream_labels"]
        need(kind in ("ground","wind") && (expected_layout!="ground-only"||kind=="ground"),"snapshot source layout")
        need(labels isa AbstractVector && labels in (["CMO"],["FRD"],["CMO","FRD"]),"snapshot source stream labels")
        if kind=="ground"
            need(declaration["station"] in ("CMO","FRD") && labels==[declaration["station"]] && declaration["product"]=="quasi-definitive" && declaration["stop_inclusive"]===true,"snapshot ground declaration")
        else
            need(declaration["station"]===nothing && declaration["product"]=="OMNI_HRO_1MIN" && declaration["stop_inclusive"]===false,"snapshot wind declaration")
        end
        expected_ids=Dict{String,Any}()
        for station in labels
            kw=(;station,product=declaration["product"],start=J._clock(declaration["start_inclusive"]),stop=J._clock(declaration["stop"]))
            parsed=primary ? (kind=="ground" ? P.parse_ground(source;kw...) : P.parse_wind(source;kw...)) :
                             (kind=="ground" ? A.audit_ground(source;kw...) : A.audit_wind(source;kw...))
            raw=primary ? parsed.raw_storage : parsed.raw_records
            need(raw==members[raw_name],"snapshot reparsed raw ledger differs")
            stream=kind*"-"*station
            group=get!(groups,stream,Any[]);need(length(group)<32,"snapshot catalog body limit");push!(group,parsed)
            expected_ids[stream]=primary ? C.body_identity(parsed) : I.audit_identity(parsed)
        end
        need(body_info["body_ids"]==expected_ids,"snapshot body identities differ")
        push!(trace,Dict("logical_id"=>id,"bytes"=>ncodeunits(source),"sha256"=>_hash(source)))
    end
    need(Set(keys(groups))==Set(keys(m["catalogs"]))==required && m["read_trace"]==trace,"snapshot stream set or source trace differs")
    output=Dict{String,Any}()
    for stream in sort!(collect(required))
        expected=m["catalogs"][stream]
        _object(expected,("kind","station","product","body_ids","shards","selected_sha256","raw_catalog_sha256","raw_rows","rows","duplicates","complete_vectors","declared_minutes","missing_minutes","uncovered_minutes"))
        catalog=primary ? C.build_catalog(groups[stream]) : I.audit_catalog(groups[stream])
        shards=expected["shards"];need(shards isa AbstractVector && length(shards)==cld(catalog.rows,8192),"snapshot selected shard count")
        io=IOBuffer(sizehint=80*catalog.rows)
        for (index,name) in enumerate(shards)
            _name(name);need(name=="catalog-"*stream*"-"*lpad(string(index),4,'0')*".bin" && haskey(members,name),"snapshot selected member identity")
            need(ncodeunits(members[name])==80*min(8192,catalog.rows-8192*(index-1)),"snapshot selected shard width")
            push!(used,name);write(io,codeunits(members[name]))
        end
        need(String(take!(io))==catalog.selected_storage,"snapshot reparsed selected witnesses differ")
        need(J._catalog_metadata(catalog,shards)==expected,"snapshot catalog metadata differs")
        output[stream]=catalog
    end
    need(used==Set(expected_names),"snapshot unused member present")
    names=sort!(collect(keys(output)))
    (;catalogs=NamedTuple{Tuple(Symbol.(names))}(Tuple(output[name] for name in names)),
      manifest_sha256,complete_sha256,evidence_mode=expected_evidence_mode,role=expected_role,
      runtime=expected_runtime,layout=expected_layout,validated_member_names=Tuple(expected_names),
      actual_source_authority_granted=false,provider_receipts_available=false,
      independent_source_qualification=false,actual_fits=0)
end
end
