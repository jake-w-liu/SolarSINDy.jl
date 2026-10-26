"""Read explicitly pinned source snapshots through ordinary bounded files.

The generated entry point rejects archival inputs. The internal reader provides
no authority; archival jobs must admit their separate feature grant first.
Recorded cache paths are passive metadata and are never opened.
"""
module ArchiveCatalogSnapshotFilesystem
using SHA
import ..ArchiveCatalogSourceJob, ..CompletedSourceDBDT
import ..ArchiveCatalogSnapshotValues
const J=ArchiveCatalogSourceJob
const R=CompletedSourceDBDT.CompletedRefitArtifacts
const V=ArchiveCatalogSnapshotValues
need(ok,message)=ok ? nothing : throw(ArgumentError(message))
digest(bytes)=bytes2hex(sha256(IOBuffer(bytes)))
function _read(path,limit)
    J._ordinary(path);identity=J._identity(path)
    bytes=R.bounded_read(path;limit)
    need(J._identity(path)==identity,"snapshot file changed during read")
    String(bytes),identity
end
"""Load one snapshot with external manifest/completion pins and explicit context.

The source path must be absolute with ordinary parent components. Exact file
membership, bounded sizes and native file identities are checked before and
after pure reconstruction. Partial or linked snapshots are rejected.
"""
function _load_pinned(source;manifest_sha256,complete_sha256,role,layout,runtime=string(VERSION),evidence_mode)
    J._ordinary(source;directory=true)
    need(runtime==string(VERSION) && runtime in ("1.12.6","1.13.1"),"snapshot native runtime differs")
    need(role in ("primary","independent") && layout in ("ground-only","mixed"),"snapshot context differs")
    need(evidence_mode in ("generated","actual_archival_cache"),"snapshot evidence mode differs")
    names=readdir(source)
    need(2<=length(names)<=1024 && "manifest.json" in names && "COMPLETE.json" in names,"snapshot filesystem members")
    text,manifest_identity=_read(joinpath(source,"manifest.json"),1_048_576)
    marker,marker_identity=_read(joinpath(source,"COMPLETE.json"),4096)
    need(digest(text)==V._sha(manifest_sha256) && digest(marker)==V._sha(complete_sha256),"snapshot filesystem metadata pin")
    metadata=R.decode_canonical(Vector{UInt8}(codeunits(text));limit=1_048_576)
    V._object(metadata,("schema","request_sha256","evidence_mode","layout","role","runtime","bodies","catalogs","members","read_trace","provider_receipts_available","independent_source_qualification","actual_fits"))
    need(metadata["schema"]=="archive_catalog_snapshot_v1" && metadata["evidence_mode"]==evidence_mode,"generated snapshot authority required before member access")
    need(metadata["role"]==role && metadata["layout"]==layout && metadata["runtime"]==runtime,"snapshot filesystem context")
    entries=metadata["members"]
    need(entries isa AbstractVector && 1<=length(entries)<=1022,"snapshot filesystem descriptor bound")
    allowed=String["manifest.json","COMPLETE.json"];seen=Set(lowercase.(allowed));total=ncodeunits(text)+ncodeunits(marker)
    for entry in entries
        V._object(entry,("name","bytes","sha256"));name=V._name(entry["name"])
        need(!(lowercase(name) in seen),"snapshot filesystem duplicate member")
        push!(seen,lowercase(name));push!(allowed,name)
        count=V._count(entry["bytes"],V.MEMBER_LIMIT);V._sha(entry["sha256"])
        need(total<=V.TOTAL_LIMIT-count,"snapshot filesystem byte bound");total+=count
    end
    need(Set(names)==Set(allowed),"snapshot filesystem fileset differs")
    # Check every declared path/size before reading the first member body.
    identities=Dict{String,Any}()
    for entry in entries
        path=joinpath(source,entry["name"]);J._ordinary(path)
        identity=J._identity(path)
        need(identity[3]==entry["bytes"],"snapshot filesystem member size")
        identities[entry["name"]]=identity
    end
    members=Dict{String,String}()
    for entry in entries
        name=entry["name"];bytes,identity=_read(joinpath(source,name),V.MEMBER_LIMIT)
        need(identity==identities[name] && ncodeunits(bytes)==entry["bytes"] && digest(bytes)==entry["sha256"],"snapshot filesystem member changed")
        members[name]=bytes
    end
    value=V.reconstruct(text,marker,members;manifest_sha256,complete_sha256,expected_role=role,expected_layout=layout,expected_runtime=runtime,expected_evidence_mode=evidence_mode)
    need(readdir(source)==names && J._identity(joinpath(source,"manifest.json"))==manifest_identity &&
         J._identity(joinpath(source,"COMPLETE.json"))==marker_identity &&
         all(J._identity(joinpath(source,name))==identity for (name,identity) in identities),"snapshot filesystem changed during reconstruction")
    value
end
"""Read a generated snapshot; actual archival metadata is rejected before body reads."""
load_generated(source;kwargs...)=_load_pinned(source;kwargs...,evidence_mode="generated")
end
