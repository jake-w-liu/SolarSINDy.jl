"""
    ArchiveCatalogSourceJob

Bounded, offline reconstruction of explicitly pinned archival bodies. A native
job publishes original bytes, raw row ledgers and selected catalog witnesses.
Generated jobs grant no archival authority. Actual cache access additionally
requires caller-pinned authorization, admission plan and full software evidence.
A single-role snapshot does not establish independent source qualification,
historical provider availability, model skill or a forecasting result.
"""
module ArchiveCatalogSourceJob
using Dates, SHA, JSON3
using ..ArchiveFormatParsers
using ..ArchiveFormatAudit
using ..ArchiveSourceCatalog
using ..ArchiveCatalogAudit
using ..CompletedSourceDBDT: CompletedRefitArtifacts
const R = CompletedRefitArtifacts
const MEMBER_BYTES = 8_388_608
const SNAPSHOT_BYTES = 1_073_741_824
const MAX_MEMBERS = 1024
const SHARD_ROWS = 8192
const REQUEST_KEYS = Set(("schema","evidence_mode","layout","role","runtime","data_root","inputs"))
const INPUT_KEYS = Set(("logical_id","path","kind","station","stream_labels","product",
    "start_inclusive","stop","stop_inclusive","bytes","sha256","role","provenance"))
const UTC_FORMAT = dateformat"yyyy-mm-ddTHH:MM:SS.sss"
need(ok, message) = ok ? nothing : throw(ArgumentError(message))
digest(bytes) = bytes2hex(sha256(bytes))

"""
    Authority(grant_path, grant_sha256, plan_path, plan_sha256,
              source_proof_path, source_proof_sha256, qualification_path, qualification_sha256)

Explicit caller pins for actual-cache reconstruction. Recorded paths in these
documents are never executed or used to locate other evidence files.
"""
struct Authority
    grant_path::String
    grant_sha256::String
    plan_path::String
    plan_sha256::String
    source_proof_path::String
    source_proof_sha256::String
    qualification_path::String
    qualification_sha256::String
end

function _object(value, keys)
    need(value isa AbstractDict && Set(Base.keys(value)) == keys, "object fields differ")
    value
end
function _text(value; limit=4096)
    need(value isa AbstractString && 0<ncodeunits(value)<=limit && isvalid(value) &&
        all(c->!iscntrl(c), value), "invalid bounded text")
    String(value)
end
function _sha(value)
    text = _text(value; limit=64)
    need(occursin(r"^[0-9a-f]{64}$", text), "invalid SHA256")
    text
end
function _ordinary(path; directory=false)
    value = _text(path)
    need(isabspath(value) && normpath(value)==value, "path must be normalized and absolute")
    cursor = value
    while true
        need(!islink(cursor), "link path component is forbidden")
        parent = dirname(cursor)
        parent==cursor && break
        cursor = parent
    end
    need(directory ? isdir(value) : isfile(value), "ordinary path does not exist")
    value
end
_same(a,b) = Sys.iswindows() ? lowercase(a)==lowercase(b) : a==b
function _inside(path, root)
    relative = relpath(path, root)
    need(!isabspath(relative) && relative!=".." && !startswith(relative,".."*string(Base.Filesystem.path_separator)),
        "input path escapes data root")
    nothing
end
function _clock(value)
    text = _text(value; limit=24)
    need(occursin(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z$",text),
        "UTC declaration syntax differs")
    stamp = try DateTime(text[1:end-1],UTC_FORMAT) catch; throw(ArgumentError("invalid UTC declaration")); end
    need(floor(stamp,Minute)==stamp, "declaration is off minute")
    stamp
end
_identity(path) = let s=stat(path); (s.device,s.inode,s.size,s.mtime,s.ctime,s.mode); end
function _pinned(path, pin; canonical=false, limit=MEMBER_BYTES)
    _ordinary(path)
    before = _identity(path)
    bytes = R.bounded_read(path;limit)
    need(_identity(path)==before && digest(bytes)==_sha(pin), "pinned evidence bytes changed")
    canonical ? R.decode_canonical(bytes;limit) : JSON3.read(bytes,Any)
end

function _validate(request, request_path)
    _object(request,REQUEST_KEYS)
    need(request["schema"]=="archive_catalog_audit_request_v1", "request schema differs")
    need(request["evidence_mode"] in ("generated","actual_archival_cache"), "evidence mode differs")
    need(request["layout"] in ("ground-only","mixed") && request["role"] in ("primary","independent"),
        "layout or native role differs")
    need(request["runtime"]==string(VERSION) && request["runtime"] in ("1.12.6","1.13.1"),
        "native Julia runtime differs")
    root = _ordinary(request["data_root"];directory=true)
    if request["evidence_mode"]=="generated"
        need(_same(root,joinpath(dirname(request_path),"inputs")), "generated data root differs")
    end
    inputs = request["inputs"]
    need(inputs isa AbstractVector && 1<=length(inputs)<=49, "input count differs")
    ids = Set{String}(); paths = Set{String}(); streams = Dict{String,Int}()
    for input in inputs
        _object(input,INPUT_KEYS)
        id = _text(input["logical_id"];limit=128)
        need(occursin(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",id) && !(id in ids), "duplicate or invalid logical ID")
        push!(ids,id)
        kind = input["kind"]
        need(kind in ("ground","wind"), "input kind differs")
        need(request["layout"]!="ground-only" || kind=="ground", "ground-only request contains wind")
        labels = input["stream_labels"]
        need(labels isa AbstractVector && labels in (["CMO"],["FRD"],["CMO","FRD"]), "stream labels differ")
        if kind=="ground"
            need(input["station"] in ("CMO","FRD") && labels==[input["station"]] &&
                input["product"]=="quasi-definitive" && input["stop_inclusive"]===true,
                "ground declaration differs")
        else
            need(input["station"]===nothing && input["product"]=="OMNI_HRO_1MIN" &&
                input["stop_inclusive"]===false, "wind declaration differs")
        end
        start, stop = _clock(input["start_inclusive"]), _clock(input["stop"])
        span = Int128(Dates.value(stop))-Int128(Dates.value(start))
        need(0<span<=32*86_400_000, "declaration exceeds the bounded minute interval")
        count = input["bytes"]
        need(count isa Integer && !(count isa Bool) && 1<=count<=MEMBER_BYTES, "body size differs")
        _sha(input["sha256"]); _text(input["role"];limit=128); _text(input["provenance"])
        path = _ordinary(input["path"]); _inside(path,root)
        need(_same(dirname(path),root), "inputs must be direct data-root members")
        key = Sys.iswindows() ? lowercase(path) : path
        need(!(key in paths), "duplicate input path"); push!(paths,key)
        if request["evidence_mode"]=="generated"
            need(startswith(basename(path),"generated-") && input["role"]=="generated_fixture" &&
                input["provenance"]=="generated_fixture", "generated descriptor differs")
        end
        for label in labels
            stream = kind*"-"*label
            streams[stream] = get(streams,stream,0)+1
            need(streams[stream]<=32, "per-catalog body count exceeded")
        end
    end
    expected = request["layout"]=="ground-only" ? Set(("ground-CMO","ground-FRD")) :
        Set(("ground-CMO","ground-FRD","wind-CMO","wind-FRD"))
    need(Set(keys(streams))==expected, "required stream set differs")
    inputs
end

function _source_inventory(proof)
    need(proof isa AbstractDict && get(proof,"complete",false)===true &&
        get(proof,"all_commit_filters_match",false)===true, "source proof is incomplete")
    inventory = proof["source_sha256"]
    need(inventory isa AbstractDict && 1<=length(inventory)<=4096, "source inventory differs")
    root = realpath(pkgdir(parentmodule(@__MODULE__)))
    worker = "src/archive_format/ArchiveCatalogSourceJob.jl"
    need(haskey(inventory,worker) && _same(realpath(@__FILE__),joinpath(root,split(worker,'/')...)),
        "worker is absent from qualified source inventory")
    for (name,pin) in inventory
        need(name isa AbstractString && !isabspath(name) && !occursin('\\',name) &&
            all(s->s!="." && s!=".." && !isempty(s),split(name,'/')), "invalid source inventory name")
        path = joinpath(root,split(name,'/')...); _ordinary(path)
        need(open(io->bytes2hex(sha256(io)),path)==_sha(pin), "loaded package differs from qualified source proof")
    end
    nothing
end
function _authorize(request, request_digest, authority)
    if request["evidence_mode"]=="generated"
        need(authority===nothing, "generated jobs cannot consume actual-source authority")
        return nothing
    end
    need(authority isa Authority, "actual-cache job requires caller-pinned authority")
    grant = _pinned(authority.grant_path,authority.grant_sha256;canonical=true,limit=131072)
    _object(grant,Set(("schema","request_sha256","plan_sha256","source_proof_sha256",
        "qualification_sha256","head","actual_source_access_authorized","dataset_discovery",
        "provider_receipts_available","operation")))
    need(grant["schema"]=="archive_catalog_actual_grant_v1" &&
        grant["request_sha256"]==request_digest && grant["plan_sha256"]==authority.plan_sha256 &&
        grant["source_proof_sha256"]==authority.source_proof_sha256 &&
        grant["qualification_sha256"]==authority.qualification_sha256 &&
        grant["actual_source_access_authorized"]===true && grant["dataset_discovery"]===false &&
        grant["provider_receipts_available"]===false && grant["operation"]=="source_reconstruction_only",
        "actual-source grant differs")
    plan = _pinned(authority.plan_path,authority.plan_sha256)
    need(get(plan,"complete",false)===true && plan["schema"]=="actual_archive_catalog_admission_precode_v1",
        "admission plan is incomplete")
    subset = filter(x->request["layout"]=="mixed" || x["kind"]=="ground",plan["admitted_on_execution"])
    need(request["inputs"]==subset, "actual inputs differ from frozen admission plan subset")
    proof = _pinned(authority.source_proof_path,authority.source_proof_sha256)
    qualification = _pinned(authority.qualification_path,authority.qualification_sha256)
    need(get(qualification,"complete",false)===true && qualification["head"]==proof["head"]==grant["head"] &&
        qualification["source_proof"]["sha256"]==authority.source_proof_sha256 &&
        qualification["software_qualification_only"]===true &&
        qualification["unfiltered_package_and_full_experiments_executed"]===true &&
        qualification["strict_docs_both_complete"]===true &&
        qualification["all_eight_latest_ci_jobs_passed"]===true &&
        qualification["all_eight_raw_logs_retained"]===true && qualification["harness_counts"]["FAIL"]==0,
        "full software qualification differs")
    jobs = qualification["ci_jobs"]
    expected = Set(["Julia $v / $s-latest" for v in ("1.12.6","1.13.1") for s in ("ubuntu","macos","windows")])
    union!(expected,Set(("Documentation build","Julia 1.12 deployment images")))
    need(length(jobs)==8 && length(Set(j["job_id"] for j in jobs))==8 &&
        Set(j["name"] for j in jobs)==expected, "eight native CI evidence identities differ")
    for job in jobs
        for key in ("log","receipt","original_log","original_receipt"); _sha(job[key]["sha256"]); end
    end
    _source_inventory(proof)
    proof
end

function _catalog_metadata(catalog, shards)
    value = Dict{String,Any}("kind"=>String(catalog.kind),"station"=>catalog.station,
        "product"=>catalog.product,"body_ids"=>collect(catalog.body_ids),"shards"=>shards)
    for key in (:selected_sha256,:raw_catalog_sha256,:raw_rows,:rows,:duplicates,:complete_vectors,
                :declared_minutes,:missing_minutes,:uncovered_minutes)
        value[string(key)] = getproperty(catalog,key)
    end
    value
end

"""
    reconstruct(request_path, destination; authority=nothing)

Reconstruct one primary or independent snapshot using an explicit canonical
request. Publish exclusive members and a completion marker last. Failures leave
their partial directory intact. The result contains no model fits or skill claim.
"""
function reconstruct(request_path::AbstractString, destination::AbstractString; authority=nothing)
    request_path = _ordinary(request_path)
    request_bytes = R.bounded_read(request_path;limit=131072)
    request = R.decode_canonical(request_bytes;limit=131072)
    inputs = _validate(request,request_path)
    proof = _authorize(request,digest(request_bytes),authority)
    need(isabspath(destination) && normpath(destination)==destination &&
        !(basename(destination) in ("",".","..")), "destination path differs")
    _ordinary(dirname(destination);directory=true)
    need(!ispath(destination) && !islink(destination), "destination already exists")
    # All declarations and authorization are checked before dataset bytes are read.
    before = [_identity(input["path"]) for input in inputs]
    need(all(i->before[i][3]==inputs[i]["bytes"],eachindex(inputs)), "source body size differs")
    mkdir(destination)
    members = Any[]; total = 0
    function publish(name, bytes)
        need(bytes isa AbstractVector{UInt8} && length(bytes)<=MEMBER_BYTES, "member bound exceeded")
        need(length(members)<MAX_MEMBERS-2 && total<=SNAPSHOT_BYTES-length(bytes), "snapshot bound exceeded")
        R._exclusive_write(joinpath(destination,name),bytes)
        total += length(bytes)
        push!(members,Dict("name"=>name,"bytes"=>length(bytes),"sha256"=>digest(bytes)))
        name
    end
    groups = Dict{String,Vector{Any}}(); bodies = Any[]; trace = Any[]
    primary = request["role"]=="primary"
    for (index,input) in enumerate(inputs)
        path = input["path"]; bytes = R.bounded_read(path)
        need(_identity(path)==before[index] && length(bytes)==input["bytes"] && digest(bytes)==input["sha256"],
            "source body identity or hash differs")
        text = String(copy(bytes)); need(isvalid(text), "source body is not UTF-8")
        raw_name = nothing; raw_digest = nothing; identities = Dict{String,Any}()
        original_name = publish("body-"*lpad(string(index),3,'0')*".txt",bytes)
        for station in input["stream_labels"]
            kwargs = (station=station,product=input["product"],start=_clock(input["start_inclusive"]),stop=_clock(input["stop"]))
            body = if primary
                input["kind"]=="ground" ? ArchiveFormatParsers.parse_ground(text;kwargs...) : ArchiveFormatParsers.parse_wind(text;kwargs...)
            else
                input["kind"]=="ground" ? ArchiveFormatAudit.audit_ground(text;kwargs...) : ArchiveFormatAudit.audit_wind(text;kwargs...)
            end
            raw = primary ? body.raw_storage : body.raw_records
            rawbytes = Vector{UInt8}(codeunits(raw)); current_digest = digest(rawbytes)
            if raw_name===nothing
                raw_name = publish("raw-"*lpad(string(index),3,'0')*".bin",rawbytes); raw_digest=current_digest
            else
                need(current_digest==raw_digest, "shared wind raw ledgers differ by stream label")
            end
            stream = input["kind"]*"-"*station
            push!(get!(groups,stream,Any[]),body)
            identities[stream] = primary ? ArchiveSourceCatalog.body_identity(body) : ArchiveCatalogAudit.audit_identity(body)
        end
        push!(bodies,Dict("logical_id"=>input["logical_id"],"original"=>original_name,
            "raw"=>raw_name,"body_ids"=>identities,"declaration"=>input))
        push!(trace,Dict("logical_id"=>input["logical_id"],"bytes"=>length(bytes),"sha256"=>digest(bytes)))
    end
    catalogs = Dict{String,Any}()
    for stream in sort(collect(keys(groups)))
        catalog = primary ? ArchiveSourceCatalog.build_catalog(groups[stream]) : ArchiveCatalogAudit.audit_catalog(groups[stream])
        shards = String[]
        storage = codeunits(catalog.selected_storage)
        for start in 1:SHARD_ROWS:catalog.rows
            stop = min(start+SHARD_ROWS-1,catalog.rows)
            name = "catalog-"*stream*"-"*lpad(string(length(shards)+1),4,'0')*".bin"
            push!(shards,publish(name,Vector{UInt8}(storage[(start-1)*80+1:stop*80])))
        end
        catalogs[stream] = _catalog_metadata(catalog,shards)
    end
    need(all(i->_identity(inputs[i]["path"])==before[i],eachindex(inputs)), "source identity changed during reconstruction")
    need(R.bounded_read(request_path;limit=131072)==request_bytes, "request changed during reconstruction")
    proof===nothing || _source_inventory(proof)
    manifest = Dict("schema"=>"archive_catalog_snapshot_v1","request_sha256"=>digest(request_bytes),
        "evidence_mode"=>request["evidence_mode"],"layout"=>request["layout"],"role"=>request["role"],
        "runtime"=>string(VERSION),"bodies"=>bodies,"catalogs"=>catalogs,"members"=>members,"read_trace"=>trace,
        "provider_receipts_available"=>false,"independent_source_qualification"=>false,"actual_fits"=>0)
    manifest_bytes = R.canonical_bytes(manifest;limit=1_048_576)
    need(total<=SNAPSHOT_BYTES-length(manifest_bytes)-4096, "snapshot metadata bound exceeded")
    R._exclusive_write(joinpath(destination,"manifest.json"),manifest_bytes)
    marker = Dict("schema"=>"archive_catalog_snapshot_complete_v1","manifest_sha256"=>digest(manifest_bytes),
        "members"=>length(members)+2,"snapshot_bytes_without_marker"=>total+length(manifest_bytes),
        "independent_source_qualification"=>false,"actual_fits"=>0)
    R._exclusive_write(joinpath(destination,"COMPLETE.json"),R.canonical_bytes(marker))
    manifest
end
end
