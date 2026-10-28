const STATIC_PAYLOADS=("context.json","numeric.bin","receipt.json")
const STATIC_LIMITS=(32768,4096,65536)

"""Write opaque bounded data files and prepare a completion marker without source authority."""
function _prepare_publication(directory,context::Vector{UInt8},numeric::Vector{UInt8},receipt::Vector{UInt8},expected::ExpectedStatic)
    bodies=(copy(context),copy(numeric),copy(receipt))
    need(all(0<length(body)<=limit for (body,limit) in zip(bodies,STATIC_LIMITS)),"static publication payload bound")
    need(digest(bodies[1])==expected.context_sha256 && digest(bodies[2])==expected.numeric_sha256,"static publication external input pins")
    need(directory isa AbstractString && !isempty(directory),"static publication directory")
    path=abspath(directory);J._ordinary(dirname(path);directory=true)
    need(!ispath(path) && !islink(path),"static publication must be fresh")
    entries=[Dict("name"=>name,"bytes"=>length(body),"sha256"=>digest(body)) for (name,body) in zip(STATIC_PAYLOADS,bodies)]
    manifest=R.canonical_bytes(Dict("schema"=>"source_bound_static_artifact_manifest_v1","context_sha256"=>expected.context_sha256,
        "numeric_sha256"=>expected.numeric_sha256,"receipt_sha256"=>digest(bodies[3]),"members"=>entries);limit=16384)
    manifest_pin=digest(manifest);receipt_pin=digest(bodies[3])
    marker=R.canonical_bytes(Dict("schema"=>"source_bound_static_artifact_complete_v1","manifest_sha256"=>manifest_pin,
        "context_sha256"=>expected.context_sha256,"numeric_sha256"=>expected.numeric_sha256,
        "receipt_sha256"=>receipt_pin,"members"=>5,"data_only"=>true);limit=4096)
    marker_pin=digest(marker)
    mkdir(path);J._ordinary(path;directory=true)
    for (name,body) in zip(STATIC_PAYLOADS,bodies);R._exclusive_write(joinpath(path,name),body);end
    R._exclusive_write(joinpath(path,"manifest.json"),manifest)
    (;directory=path,expected,manifest_sha256=manifest_pin,receipt_sha256=receipt_pin,complete_sha256=marker_pin,
      marker=String(marker),members=Tuple((;name=entry["name"],bytes=entry["bytes"],sha256=entry["sha256"]) for entry in entries))
end

"""Verify every prepared file before creating the completion marker exclusively."""
function _complete_publication(prepared)
    path=prepared.directory;J._ordinary(path;directory=true)
    need(Set(readdir(path))==Set((STATIC_PAYLOADS...,"manifest.json")),"static prepared fileset changed")
    identities=Dict{String,Any}()
    for (entry,limit) in zip(prepared.members,STATIC_LIMITS)
        text,id=FS._read(joinpath(path,entry.name),limit);bytes=codeunits(text);identities[entry.name]=id
        need(length(bytes)==entry.bytes && digest(bytes)==entry.sha256,"static prepared payload changed")
    end
    text,id=FS._read(joinpath(path,"manifest.json"),16384);identities["manifest.json"]=id
    need(digest(codeunits(text))==prepared.manifest_sha256 && digest(codeunits(prepared.marker))==prepared.complete_sha256,
        "static prepared metadata changed")
    need(Set(readdir(path))==Set((STATIC_PAYLOADS...,"manifest.json")) &&
        all(J._identity(joinpath(path,name))==id for (name,id) in identities),"static prepared publication changed")
    R._exclusive_write(joinpath(path,"COMPLETE.json"),Vector{UInt8}(codeunits(prepared.marker)))
    _read_publication(path,prepared.expected;manifest_sha256=prepared.manifest_sha256,
        receipt_sha256=prepared.receipt_sha256,complete_sha256=prepared.complete_sha256)
    (;manifest_sha256=prepared.manifest_sha256,receipt_sha256=prepared.receipt_sha256,complete_sha256=prepared.complete_sha256)
end

"""Read a five-file data-only frame; this structural check grants no source authority."""
function _read_publication(directory,expected::ExpectedStatic;manifest_sha256,receipt_sha256,complete_sha256)
    J._ordinary(directory;directory=true);names=readdir(directory)
    need(length(names)==5 && Set(names)==Set((STATIC_PAYLOADS...,"manifest.json","COMPLETE.json")),"static publication fileset")
    identities=Dict{String,Any}()
    function member(name,limit)
        text,id=FS._read(joinpath(directory,name),limit);identities[name]=id
        Vector{UInt8}(codeunits(text))
    end
    mb=member("manifest.json",16384);cb=member("COMPLETE.json",4096)
    need(digest(mb)==J._sha(manifest_sha256) && digest(cb)==J._sha(complete_sha256),"static external publication pins")
    manifest=R.decode_canonical(mb;limit=16384);marker=R.decode_canonical(cb;limit=4096)
    J._object(manifest,Set(("schema","context_sha256","numeric_sha256","receipt_sha256","members")))
    J._object(marker,Set(("schema","manifest_sha256","context_sha256","numeric_sha256","receipt_sha256","members","data_only")))
    need(manifest["schema"]=="source_bound_static_artifact_manifest_v1" && marker["schema"]=="source_bound_static_artifact_complete_v1" &&
        manifest["context_sha256"]==marker["context_sha256"]==expected.context_sha256 &&
        manifest["numeric_sha256"]==marker["numeric_sha256"]==expected.numeric_sha256 &&
        manifest["receipt_sha256"]==marker["receipt_sha256"]==J._sha(receipt_sha256) &&
        marker["manifest_sha256"]==manifest_sha256 && F.isint(marker["members"]) && marker["members"]==5 && marker["data_only"]===true,
        "static publication metadata binding")
    entries=manifest["members"];need(entries isa AbstractVector && length(entries)==3,"static payload descriptor count")
    bodies=Vector{Vector{UInt8}}()
    for (index,entry) in enumerate(entries)
        J._object(entry,Set(("name","bytes","sha256")))
        name=STATIC_PAYLOADS[index];limit=STATIC_LIMITS[index]
        need(entry["name"]==name && F.isint(entry["bytes"]) && 0<entry["bytes"]<=limit,"static payload descriptor")
        body=member(name,limit)
        need(length(body)==entry["bytes"] && digest(body)==J._sha(entry["sha256"]),"static payload content binding")
        push!(bodies,body)
    end
    need(digest(bodies[1])==expected.context_sha256 && digest(bodies[2])==expected.numeric_sha256 && digest(bodies[3])==receipt_sha256,
        "static payload external binding")
    need(readdir(directory)==names && all(J._identity(joinpath(directory,name))==id for (name,id) in identities),"static publication changed during reading")
    (;context=bodies[1],numeric=bodies[2],receipt=bodies[3],codec_only=true)
end
