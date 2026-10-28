const _KIND_CONTEXT = Dict("schedule"=>"schedule_kind_context","fit_input"=>"fit_input_kind_context",
    "refit_point"=>"point_kind_context","calibration"=>"cal_kind_context",
    "native_job"=>"native_job_kind_context","source_audit"=>"source_audit_kind_context")

function _pins(pins)
    require(pins isa AbstractDict,"caller pins must be a digest-to-path map")
    require(length(pins)<=MAX_GRAPH_FILES,"caller resolution map exceeds bound")
    result=Dict{String,String}()
    for (key,value) in pins
        pin=sha_string(key);path=bounded_string(value;nonempty=true)
        require(isabspath(path),"caller pin path must be absolute")
        require(!haskey(result,pin),"duplicate caller pin")
        result[pin]=normpath(path)
    end
    result
end
function _resolve(pins,pin;directory=nothing)
    text=sha_string(pin)
    require(haskey(pins,text),"required caller pin is absent")
    path=pins[text]
    require(!islink(path),"caller pin cannot resolve a symlink")
    require(directory===nothing ? (isfile(path)||isdir(path)) : (directory ? isdir(path) : isfile(path)),"caller pin has wrong file kind")
    path
end
_same_path(a,b) = Sys.iswindows() ? lowercase(realpath(a))==lowercase(realpath(b)) : realpath(a)==realpath(b)

function _chunk_catalog(names,prefix;optional=false,single=nothing)
    parts=filter(n->startswith(n,prefix*"-"),names)
    single!==nothing && filter!(n->n!=single,parts)
    if single!==nothing && single in names
        require(isempty(parts),"single and chunked representations cannot coexist")
        return [single]
    end
    sort!(parts)
    require(optional || !isempty(parts),"required row payload is absent")
    expected=[_row_name(i,prefix) for i in 1:length(parts)]
    require(parts==expected,"chunk names must be consecutive")
    parts
end
function _catalog(kind,names)
    require(kind in ARTIFACT_KINDS,"unsupported artifact kind")
    if kind=="schedule"
        return vcat(["context.json"],_chunk_catalog(names,"rows"),["native-audit.json","independent-schedule-check.json"])
    elseif kind=="fit_input"
        return vcat(["context.json"],_chunk_catalog(names,"rows";optional=true),_chunk_catalog(names,"excluded";optional=true),["native-input-audit.json","independent-input-check.json"])
    elseif kind=="refit_point"
        return ["context.json","recipe.json","preprocessing.json","model.json","fit-input-reference.json","fit-receipt.json","native-fit-receipt.json","independent-fit-check.json"]
    elseif kind=="calibration"
        return vcat(["context.json","point-reference.json","input-reference.json"],_chunk_catalog(names,"rows"),_chunk_catalog(names,"sorted"),["native-cal-receipt.json","independent-cal-check.json"])
    elseif kind=="source_audit"
        return vcat(["context.json","source-permissions.json"],_chunk_catalog(names,"source-audit";single="source-audit-data.json"),["native-audit.json","independent-parser-check.json"])
    else
        return vcat(["context.json","job.json"],_chunk_catalog(names,"payload";single="payload.json"),["internal-receipt.json","native-receipt.json","native.log"])
    end
end
function _member_limit(name)
    name=="native.log" && return MAX_MEMBER_BYTES
    (startswith(name,"rows-") || startswith(name,"excluded-") || startswith(name,"sorted-") ||
     startswith(name,"payload") || startswith(name,"source-audit-") || name=="model.json") && return MAX_MEMBER_BYTES
    MAX_METADATA_BYTES
end

function _context_shape(context,protocol::RefitProtocol,kind::String)
    _schema(context,"context");spec=_spec(protocol)
    require(context["protocol_sha256"]==protocol.sha256,"context protocol pin differs")
    for key in ("protocol_id","station","ground_product","wind_product_or_null","units","layout","feature_names","cohort_kind","policy","availability_kind","measured_receipts_available")
        require(context[key]==spec[key],"context identity differs: "*key)
    end
    _gitsha(context["package_commit"]);_gitsha(context["package_tree"])
    sha_string(context["julia_project_sha256"]);sha_string(context["julia_manifest_sha256"]);sha_string(context["runtime_id"])
    codes=context["code_hashes"]
    require(codes isa AbstractDict && 0<length(codes)<=MAX_FIELDS,"invalid loaded code identity map")
    for (name,pin) in codes;bounded_string(name;nonempty=true);sha_string(pin);end
    _schema(context["kind_context"],_KIND_CONTEXT[kind])
    source=_nullable_sha(context["source_manifest_sha256_or_null"])
    schedule=_nullable_sha(context["schedule_manifest_sha256_or_null"])
    if kind=="source_audit"
        require(source===nothing && schedule===nothing,"source audit cannot pin itself/future schedule")
    elseif kind=="schedule"
        require(source!==nothing && schedule===nothing,"schedule bootstrap pins differ")
    elseif kind=="native_job"
        op=context["kind_context"]["operation"]
        require(op in OPERATIONS,"unsupported native operation")
        if op in ("source_audit","source_independent_audit")
            require(source===nothing && schedule===nothing,"source bootstrap context must not form a cycle")
        elseif op=="feature_schedule"
            require(source!==nothing && schedule===nothing,"schedule bootstrap context pins differ")
        else
            require(source!==nothing && schedule!==nothing,"native numerical/input descendants require committed source/schedule pins")
        end
        input=_nullable_sha(context["kind_context"]["input_manifest_sha256_or_null"])
        require((op=="source_audit")===(input===nothing),"native input null is allowed only for source producer")
    else
        require(source!==nothing && schedule!==nothing,"descendant requires committed source/schedule pins")
    end
    context
end

mutable struct _Graph
    stack::Set{String}
    cache::Dict{String,_Bundle}
    files::Int
    bytes::Int64
end
_Graph() = _Graph(Set{String}(),Dict{String,_Bundle}(),0,Int64(0))

function _graph_edges(bundle::_Bundle)
    context=decode_canonical(bundle.context);kind=bundle.kind;k=context["kind_context"]
    edges=String[]
    for key in ("source_manifest_sha256_or_null","schedule_manifest_sha256_or_null")
        context[key]!==nothing && push!(edges,sha_string(context[key]))
    end
    if kind=="source_audit"
        append!(edges,[sha_string(k["primary_native_job_manifest_sha256"]),sha_string(k["independent_native_job_manifest_sha256"])])
    elseif kind=="native_job"
        k["input_manifest_sha256_or_null"]!==nothing && push!(edges,sha_string(k["input_manifest_sha256_or_null"]))
        request=_member_json(bundle,"job.json");_schema(request,"job_request")
        for item in _array(request["dependency_pins"])
            _schema(item,"dependency_pin_entry");identifier(item["role"]);push!(edges,sha_string(item["manifest_sha256"]))
        end
    elseif kind=="refit_point"
        push!(edges,sha_string(k["input_manifest_sha256"]))
    elseif kind=="calibration"
        append!(edges,[sha_string(k["point_manifest_sha256"]),sha_string(k["input_manifest_sha256"])])
    end
    for (name,bytes) in bundle.members
        if name in ("native-audit.json","native-input-audit.json","native-fit-receipt.json","native-cal-receipt.json")
            value=decode_canonical(bytes);_schema(value,"native_receipt_reference")
            push!(edges,sha_string(value["native_job_manifest_sha256"]))
        elseif name in ("independent-parser-check.json","independent-schedule-check.json","independent-input-check.json","independent-fit-check.json","independent-cal-check.json")
            value=decode_canonical(bytes);_schema(value,"independent_check_reference")
            push!(edges,sha_string(value["native_job_manifest_sha256"]))
        end
    end
    unique(edges)
end

function _read_bundle(directory::AbstractString,protocol::RefitProtocol,pins;
                      expected_kind=nothing,expected_manifest=nothing,graph=_Graph(),depth=1)
    require(depth<=MAX_DEPTH,"artifact dependency graph depth exceeded")
    require(isabspath(directory) && isdir(directory) && !islink(directory),"artifact must be an ordinary absolute directory")
    names=readdir(directory)
    require(length(names)<=MAX_FILES && length(unique(lowercase.(names)))==length(names),"artifact member/case collision bound exceeded")
    require("manifest.json" in names && "COMMIT.json" in names,"uncommitted artifact")
    for name in names
        require(occursin(r"^[A-Za-z0-9_.-]+$",name),"unsupported artifact member name")
        path=joinpath(directory,name)
        require(isfile(path) && !islink(path),"artifact members must be ordinary files")
    end
    commitbytes=bounded_read(joinpath(directory,"COMMIT.json");limit=MAX_METADATA_BYTES)
    commit=decode_canonical(commitbytes;limit=MAX_METADATA_BYTES);_schema(commit,"commit")
    require(commit["schema"]=="dbdt_completed_artifact_commit_v1" && commit["protocol_sha256"]==protocol.sha256,"artifact commit identity differs")
    manifestpin=sha_string(commit["manifest_sha256"]);commitpin=digest(commitbytes)
    expected_manifest!==nothing && require(manifestpin==expected_manifest,"artifact manifest differs from requested pin")
    require(_same_path(_resolve(pins,manifestpin;directory=true),directory) && _same_path(_resolve(pins,commitpin;directory=true),directory),"manifest/COMMIT are not caller pinned to this directory")
    require(!(manifestpin in graph.stack),"artifact dependency cycle")
    if haskey(graph.cache,manifestpin)
        cached=graph.cache[manifestpin]
        require(cached.manifest_sha256==manifestpin && cached.commit_sha256==commitpin &&
            cached.protocol.sha256==protocol.sha256 && cached.kind==commit["kind"] &&
            _same_path(cached.directory,directory),"cached artifact identity differs")
        require(expected_kind===nothing || cached.kind==expected_kind,"cached artifact kind differs")
        return cached
    end
    push!(graph.stack,manifestpin)
    manifestbytes=bounded_read(joinpath(directory,"manifest.json");limit=MAX_METADATA_BYTES)
    require(digest(manifestbytes)==manifestpin,"manifest bytes differ")
    manifest=decode_canonical(manifestbytes;limit=MAX_METADATA_BYTES);_schema(manifest,"manifest")
    kind=bounded_string(manifest["kind"])
    require(kind in ARTIFACT_KINDS && (expected_kind===nothing || expected_kind==kind),"artifact kind differs")
    require(manifest["schema"]=="dbdt_completed_"*kind*"_v1" && manifest["codec"]==CODEC && manifest["evidence_mode"]=="synthetic_only" && manifest["protocol_sha256"]==protocol.sha256 && commit["kind"]==kind,"artifact header differs")
    rowcount=checked_integer(manifest["row_count"],0,MAX_ROWS)
    entries=_array(manifest["members"];maximum=MAX_FILES-2)
    require(checked_integer(manifest["member_count"],0,MAX_FILES-2)==length(entries),"member count differs")
    headernames=String[];members=Pair{String,String}[]
    total=Int64(length(commitbytes)+length(manifestbytes))
    for entry in entries
        _schema(entry,"member");name=bounded_string(entry["name"];nonempty=true)
        require(name in names && !(name in ("manifest.json","COMMIT.json")),"invalid member declaration")
        count=checked_integer(entry["bytes"],0,_member_limit(name));pin=sha_string(entry["sha256"])
        data=bounded_read(joinpath(directory,name);limit=_member_limit(name))
        require(length(data)==count && digest(data)==pin,"artifact member bytes differ")
        name!="native.log" && decode_canonical(data;limit=_member_limit(name))
        total=Base.Checked.checked_add(total,length(data));require(total<=MAX_BUNDLE_BYTES,"bundle total byte bound exceeded")
        push!(headernames,name);push!(members,name=>String(data))
    end
    require(length(unique(headernames))==length(headernames) && headernames==_catalog(kind,headernames),"member grammar/order differs")
    require(Set(names)==Set(vcat(headernames,["manifest.json","COMMIT.json"])),"extra/missing artifact member")
    contextbytes=last(only(filter(p->first(p)=="context.json",members)))
    require(digest(contextbytes)==manifest["context_sha256"],"context hash differs")
    _context_shape(decode_canonical(contextbytes),protocol,kind)
    graph.files+=length(names);graph.bytes=Base.Checked.checked_add(graph.bytes,total)
    require(graph.files<=MAX_GRAPH_FILES && graph.bytes<=MAX_GRAPH_BYTES,"referenced artifact graph bound exceeded")
    bundle=_Bundle(_VERIFIED,realpath(directory),manifestpin,commitpin,kind,protocol,contextbytes,Tuple(members),rowcount)
    for edge in _graph_edges(bundle)
        _read_bundle(_resolve(pins,edge;directory=true),protocol,pins;expected_manifest=edge,graph,depth=depth+1)
    end
    pop!(graph.stack,manifestpin)
    # Recheck exact bytes/member set after all dependency I/O, under the stated
    # caller-owned quiescent-path contract. No hostile filesystem CAS is claimed.
    require(Set(readdir(directory))==Set(names),"artifact member set changed during load")
    require(digest(bounded_read(joinpath(directory,"COMMIT.json");limit=MAX_METADATA_BYTES))==commitpin && digest(bounded_read(joinpath(directory,"manifest.json");limit=MAX_METADATA_BYTES))==manifestpin,"artifact header changed during load")
    for (name,bytes) in members
        require(bounded_read(joinpath(directory,name);limit=_member_limit(name))==Vector{UInt8}(codeunits(bytes)),"artifact member changed during load")
    end
    graph.cache[manifestpin]=bundle
    bundle
end

function _exclusive_write(path,data)
    require(data isa AbstractVector{UInt8} && length(data)<=MAX_MEMBER_BYTES,"invalid exclusive member bytes")
    Base.require_one_based_indexing(data)
    flags=Base.JL_O_WRONLY | Base.JL_O_CREAT | Base.JL_O_EXCL
    file=Base.Filesystem.open(path,flags,0o644)
    try
        write(file,data)
    finally
        close(file)
    end
    require(bounded_read(path)==data,"exclusive write readback differs")
    nothing
end

function _publish_bundle(directory,protocol::RefitProtocol,kind::String,context,members,rows;
                         pins,failpoint=:none)
    require(kind in ARTIFACT_KINDS,"unknown publication kind")
    rowcount=checked_integer(rows,0,MAX_ROWS);_context_shape(context,protocol,kind)
    require(failpoint in (:none,:beforemkdir,:aftereachmember,:beforemanifest,:aftermanifest_beforeCOMMIT,:afterCOMMIT_beforefinalcheck),"unsupported publication test failpoint")
    path=abspath(directory);parent=dirname(path)
    require(isdir(parent) && !islink(parent) && basename(path) ∉ (".","..",""),"publication parent must be an existing ordinary directory")
    require(!ispath(path) && !islink(path),"publication destination already exists")
    data=Dict{String,String}("context.json"=>_jsontext(context;limit=MAX_METADATA_BYTES))
    for (name,value) in members
        require(!haskey(data,name),"duplicate publication member")
        text=value isa AbstractString ? String(value) : String(Vector{UInt8}(value))
        require(ncodeunits(text)<=_member_limit(name),"member byte bound exceeded")
        name!="native.log" && decode_canonical(text;limit=_member_limit(name))
        data[name]=text
    end
    names=_catalog(kind,collect(keys(data)))
    require(Set(names)==Set(keys(data)) && length(names)+2<=MAX_FILES,"publication member set differs")
    entries=Any[Dict{String,Any}("name"=>name,"bytes"=>ncodeunits(data[name]),"sha256"=>digest(data[name])) for name in names]
    require(sum(ncodeunits(v) for v in values(data))<=MAX_BUNDLE_BYTES-2MAX_METADATA_BYTES,"publication total bound exceeded")
    manifest=Dict{String,Any}("schema"=>"dbdt_completed_"*kind*"_v1","codec"=>CODEC,"kind"=>kind,
        "evidence_mode"=>"synthetic_only","protocol_sha256"=>protocol.sha256,"context_sha256"=>digest(data["context.json"]),
        "row_count"=>rowcount,"member_count"=>length(entries),"members"=>entries)
    manifestbytes=canonical_bytes(manifest;limit=MAX_METADATA_BYTES);manifestpin=digest(manifestbytes)
    commit=Dict{String,Any}("schema"=>"dbdt_completed_artifact_commit_v1","manifest_sha256"=>manifestpin,"protocol_sha256"=>protocol.sha256,"kind"=>kind)
    commitbytes=canonical_bytes(commit;limit=MAX_METADATA_BYTES);commitpin=digest(commitbytes)
    localpins=_pins(pins);localpins[manifestpin]=path;localpins[commitpin]=path
    failpoint===:beforemkdir && fail("injected failure before publication mkdir")
    mkdir(path)
    marker=false
    try
        for name in names
            _exclusive_write(joinpath(path,name),Vector{UInt8}(codeunits(data[name])))
            failpoint===:aftereachmember && fail("injected failure after member")
        end
        failpoint===:beforemanifest && fail("injected failure before manifest")
        _exclusive_write(joinpath(path,"manifest.json"),manifestbytes)
        failpoint===:aftermanifest_beforeCOMMIT && fail("injected failure before COMMIT")
        _exclusive_write(joinpath(path,"COMMIT.json"),commitbytes);marker=true
        failpoint===:afterCOMMIT_beforefinalcheck && fail("injected failure after COMMIT")
        bundle=_read_bundle(path,protocol,localpins;expected_kind=kind,expected_manifest=manifestpin)
        return bundle
    catch primary_error
        cleanup_error=nothing
        try
            source=joinpath(path,"COMMIT.json");target=joinpath(path,"INCOMPLETE-COMMIT.json")
            # Invalidate only this attempt's exact ordinary completion. A
            # replacement/unknown marker belongs to somebody else and is kept.
            if marker && isfile(source) && !islink(source) && bounded_read(source;limit=MAX_METADATA_BYTES)==commitbytes
                require(dirname(abspath(source))==abspath(path) && dirname(abspath(target))==abspath(path),"quarantine containment differs")
                require(!ispath(target) && !islink(target),"incomplete marker already exists")
                Base.Filesystem.rename(source,target)
            end
        catch error
            cleanup_error=error
        end
        cleanup_error!==nothing && throw(CompositeException(Any[primary_error,cleanup_error]))
        rethrow()
    end
end
