module DstIntervalArtifacts

using Dates, SHA, TOML
using ..DstIntervalAdapter
using ..DstIntervalStream
using ..DstIntervalTraining
const A=DstIntervalAdapter
const S=DstIntervalStream
const T=DstIntervalTraining
const R=A.Replay
const LOADED_SOURCE_SHA256=bytes2hex(sha256(read(@__FILE__)))
T.verify_source_identity()

export save_bundle, load_bundle

function verify_source_identity()
    dependencies=T.verify_source_identity()
    bytes2hex(sha256(read(@__FILE__)))==LOADED_SOURCE_SHA256 ||
        throw(ArgumentError("loadedsource/disk mismatch: DstIntervalArtifacts.jl"))
    (dependencies...,"DstIntervalArtifacts.jl"=>LOADED_SOURCE_SHA256)
end

const MODEL_FIELDS=(:schema,:evidence_mode,:contract_sha256,:gains,:setting_index,:taus,:sigma,:eta,
    :delay_hours,:boundary,:source_sha256,:source_hashes,:fit_count,:informative_count,:storm_count,
    :fit_data_sha256,:fit_mask_sha256,:max_target_complete,:max_label_receipt,:max_fit_issue,:solver,:model_id)
const SCHEMA="candidate_a_synthetic_bundle_v1"
const MAX_MEMBER_BYTES=1_000_000
_hash(x)=bytes2hex(sha256(x))
_require(condition,message)=condition || throw(ArgumentError(message))

# A deliberately small, typed TOML codec. No eval, Julia Serialization, or
# caller-selected types are loaded. Field order is explicit and hash-stable.
function _encode(value)
    if value isa NamedTuple
        Dict("type"=>"namedtuple","names"=>collect(String.(keys(value))),"items"=>[_encode(x) for x in values(value)])
    elseif value isa Tuple
        Dict("type"=>"tuple","items"=>[_encode(x) for x in value])
    elseif value isa Pair
        Dict("type"=>"pair","first"=>_encode(first(value)),"second"=>_encode(last(value)))
    elseif value isa DateTime
        Dict("type"=>"datetime","milliseconds"=>string(Dates.value(value)))
    elseif value isa Bool
        Dict("type"=>"bool","value"=>value)
    elseif value isa Int
        Dict("type"=>"int","value"=>string(value))
    elseif value isa Float64
        _require(isfinite(value),"bundle contains nonfinite numeric value")
        Dict("type"=>"float64","bits"=>string(reinterpret(UInt64,value),base=16,pad=16))
    elseif value isa String
        Dict("type"=>"string","value"=>value)
    elseif value isa Symbol
        Dict("type"=>"symbol","value"=>String(value))
    elseif value===nothing
        Dict("type"=>"nothing")
    else
        throw(ArgumentError("unsupported bundle type $(typeof(value))"))
    end
end

function _decode(value;depth=0)
    _require(depth<=32 && value isa AbstractDict && haskey(value,"type"),"invalid or excessively nested typed value")
    kind=value["type"]
    expected=kind=="namedtuple" ? ["type","names","items"] : kind=="tuple" ? ["type","items"] :
        kind=="pair" ? ["type","first","second"] : kind=="datetime" ? ["type","milliseconds"] :
        kind=="float64" ? ["type","bits"] : kind=="nothing" ? ["type"] :
        kind in ("bool","int","string","symbol") ? ["type","value"] : String[]
    _require(!isempty(expected) && Set(keys(value))==Set(expected),"unknown typed value or fields")
    descend(x)=_decode(x;depth=depth+1)
    if kind=="namedtuple"
        names=value["names"];items=value["items"]
        _require(names isa AbstractVector && items isa AbstractVector && length(names)==length(items)<=128,
            "invalid namedtuple shape")
        _require(all(x->x isa String && occursin(r"\A[A-Za-z_][A-Za-z_0-9]*\z",x),names) && allunique(names),
            "invalid namedtuple field names")
        return NamedTuple{Tuple(Symbol.(names))}(Tuple(descend(x) for x in items))
    elseif kind=="tuple"
        _require(value["items"] isa AbstractVector && length(value["items"])<=128,"invalid tuple shape")
        return Tuple(descend(x) for x in value["items"])
    elseif kind=="pair"
        return descend(value["first"])=>descend(value["second"])
    elseif kind=="datetime"
        return DateTime(Dates.UTInstant{Millisecond}(Millisecond(parse(Int64,value["milliseconds"]))))
    elseif kind=="float64"
        bits=value["bits"]
        _require(bits isa String && occursin(r"\A[0-9a-f]{16}\z",bits),"invalid Float64 bits")
        result=reinterpret(Float64,parse(UInt64,bits;base=16))
        _require(isfinite(result),"nonfinite decoded value")
        return result
    elseif kind=="int"
        return parse(Int,value["value"])
    elseif kind=="bool"
        _require(value["value"] isa Bool,"invalid Bool value")
        return value["value"]
    elseif kind=="string"
        _require(value["value"] isa String,"invalid String value")
        return value["value"]
    elseif kind=="symbol"
        _require(value["value"] isa String && occursin(r"\A[A-Za-z_][A-Za-z_0-9]*\z",value["value"]),"invalid Symbol")
        return Symbol(value["value"])
    end
    nothing
end

function _validate_model(model)
    _require(model isa NamedTuple && keys(model)==MODEL_FIELDS,"model schema fields differ")
    _require(model.schema=="candidate_a_intervalmean_research_v1" &&
        model.evidence_mode=="synthetic_or_unqualified_research" && model.contract_sha256==T.CONTRACT_SHA256,
        "model schema or contract differs")
    _require(model.setting_index isa Int && 1<=model.setting_index<=18,"invalid model setting index")
    setting=A.settings()[model.setting_index]
    _require(model.taus isa NTuple{2,Float64} && model.sigma isa Float64 && model.eta isa Float64 &&
        (model.taus,model.sigma,model.eta)==(setting.taus,setting.sigma,setting.eta),"model configuration differs")
    _require(model.delay_hours isa Int && model.delay_hours in (0,1),"invalid model latency")
    _require(model.gains isa NTuple{2,Float64} && all(isfinite,model.gains) && all(>=(0),model.gains),"invalid model gains")
    _require(model.boundary isa DateTime && floor(model.boundary,Hour)==model.boundary,"invalid training boundary")
    _require(model.source_hashes==T.verify_source_identity(),"model loaded source identity differs")
    _require(all(T._valid_hash,(model.source_sha256,model.fit_data_sha256,model.fit_mask_sha256,model.model_id)),"invalid model digest")
    _require(model.fit_count isa Int && model.informative_count isa Int && model.storm_count isa Int &&
        100<=model.informative_count<=model.fit_count<=typemax(Int)÷2 &&
        0<=model.storm_count<=2model.fit_count,"invalid model counts")
    _require(model.max_fit_issue isa DateTime && model.max_target_complete isa DateTime && model.max_label_receipt isa DateTime &&
        A.fit_eligible(model.max_fit_issue,model.max_target_complete,model.max_label_receipt,model.boundary) &&
        model.max_target_complete==A._shift(model.max_fit_issue,6) &&
        model.max_label_receipt==A._shift(model.max_target_complete,model.delay_hours),
        "model training clocks violate embargo")
    _require(model.solver isa NamedTuple && model.solver.gains==model.gains,"model solver gains differ")
    base=NamedTuple{MODEL_FIELDS[1:end-1]}(Tuple(model)[1:end-1])
    _require(T._hash(base)==model.model_id,"model identity differs from contents")
    model
end

function _checkpoint_data(cp)
    _require(cp isa S.StreamCheckpoint,"unsupported checkpoint")
    pending=cp.pending===nothing ? nothing :
        (index=cp.pending.index,start=cp.pending.start,stop=cp.pending.stop,q=cp.pending.q,pressure=cp.pending.pressure)
    (last_index=cp.last_index,last_issue=cp.last_issue,source_sha256=cp.source_sha256,prefix_sha256=cp.prefix_sha256,
        delay_hours=cp.delay_hours,taus=cp.taus,sigma=cp.sigma,
        sealed=(boundary=cp.sealed.boundary,mean=cp.sealed.mean,covariance=cp.sealed.covariance),
        basis=cp.basis,pending=pending,active=cp.active,consecutive_valid=cp.consecutive_valid,reset_tag=cp.reset_tag)
end

function _restore_checkpoint(data,model)
    expected=(:last_index,:last_issue,:source_sha256,:prefix_sha256,:delay_hours,:taus,:sigma,:sealed,:basis,:pending,
        :active,:consecutive_valid,:reset_tag)
    _require(data isa NamedTuple && keys(data)==expected,"checkpoint schema differs")
    _require(data.last_index isa Int && data.last_index>=0 && data.last_issue isa DateTime &&
        floor(data.last_issue,Hour)==data.last_issue,"invalid checkpoint cursor")
    _require((data.source_sha256,data.delay_hours,data.taus,data.sigma)==
        (model.source_sha256,model.delay_hours,model.taus,model.sigma),"checkpoint model source/config differs")
    _require(T._valid_hash(data.prefix_sha256),"invalid checkpoint prefix digest")
    _require(data.basis isa NTuple{2,NTuple{2,Float64}} && all(isfinite,Iterators.flatten(data.basis)),"invalid checkpoint basis")
    _require(data.sealed isa NamedTuple && keys(data.sealed)==(:boundary,:mean,:covariance),"invalid sealed checkpoint fields")
    _require(data.sealed.mean isa NTuple{2,Float64} && data.sealed.covariance isa NTuple{2,NTuple{2,Float64}} &&
        data.sealed.boundary isa DateTime && floor(data.sealed.boundary,Hour)==data.sealed.boundary,
        "sealed checkpoint dimensions or clock differ")
    _require(data.active isa Bool && data.consecutive_valid isa Int && data.reset_tag isa DateTime &&
        floor(data.reset_tag,Hour)==data.reset_tag,"invalid checkpoint status")
    sealed=R.Checkpoint(data.sealed.boundary,collect(data.sealed.mean),R._matrix(data.sealed.covariance))
    pending=if data.pending===nothing
        nothing
    else
        p=data.pending
        _require(p isa NamedTuple && keys(p)==(:index,:start,:stop,:q,:pressure),"pending checkpoint fields differ")
        _require(p.index==data.last_index && p.start==A._shift(data.last_issue,-1) && p.stop==data.last_issue &&
            p.q isa Float64 && isfinite(p.q) && p.q>=0 && p.pressure isa Float64 && isfinite(p.pressure),"invalid pending driver")
        S.PendingDriver(p.index,p.start,p.stop,p.q,p.pressure)
    end
    if data.active
        _require(1<=data.consecutive_valid<=data.last_index &&
            data.reset_tag==A._shift(data.last_issue,-data.consecutive_valid),"checkpoint active counter differs")
        _require(data.delay_hours==0 ? (pending===nothing && sealed.boundary==data.last_issue) :
            (pending!==nothing && sealed.boundary==pending.start),"checkpoint pending/sealed clocks differ")
    else
        _require(data.consecutive_valid==0 && pending===nothing && sealed.boundary==data.last_issue,
            "inactive checkpoint tail differs")
    end
    S.StreamCheckpoint(data.last_index,data.last_issue,data.source_sha256,data.prefix_sha256,data.delay_hours,
        data.taus,data.sigma,sealed,data.basis,pending,data.active,data.consecutive_valid,data.reset_tag)
end

function _toml_bytes(value)
    buffer=IOBuffer();TOML.print(buffer,value;sorted=true);take!(buffer)
end

# Validate the same typed codec accepted by load_bundle before publishing.
# Encoding and TOML materialization happen first; this is not an RSS/time bound.
function _typed_payload_bytes(value)
    encoded=_encode(value)
    _decode(encoded)
    bytes=_toml_bytes(Dict("value"=>encoded))
    _require(length(bytes)<=MAX_MEMBER_BYTES,"encoded bundle member exceeds supported byte bound")
    bytes
end

function _write_new(path,bytes)
    file=Base.Filesystem.open(path,Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL,0o600)
    try
        write(file,bytes)==length(bytes) || error("short artifact write")
    finally
        close(file)
    end
    _require(read(path)==bytes,"artifact readback differs")
end

function _directory(path)
    absolute=abspath(path)
    _require(isdir(dirname(absolute)) && !islink(dirname(absolute)),"artifact parent must be an existing ordinary directory")
    absolute
end

function _quarantine_completion(path,expected)
    if expected!==nothing && isfile(path) && read(path)==expected
        # A known failed publication keeps its original bytes under an explicit
        # incomplete name. Both paths are fixed members of this owned newdir.
        target=path*".incomplete"
        _require(dirname(abspath(path))==dirname(abspath(target)),"invalid quarantine path")
        mv(path,target;force=false)
    end
end

"""Publish a new research-only bundle. mkdir and exclusive member creation do
not replace existing paths. A failure retains an incomplete directory; only a
last-written manifest is a completion marker. Caller owns source/output paths
throughout I/O; this does not claim hostile filesystem mutation resistance.
"""
function save_bundle(directory,model,checkpoint;provenance)
    hashes=verify_source_identity();_validate_model(model)
    cpdata=_checkpoint_data(checkpoint);_restore_checkpoint(cpdata,model)
    _require(provenance isa NamedTuple,"explicit immutable research provenance required")
    payloads=("model.toml"=>_typed_payload_bytes(model),
        "checkpoint.toml"=>_typed_payload_bytes(cpdata),
        "provenance.toml"=>_typed_payload_bytes(provenance))
    destination=_directory(directory);mkdir(destination)
    manifestpath=joinpath(destination,"manifest.toml");completion_bytes=nothing
    try
        for (name,bytes) in payloads
            _write_new(joinpath(destination,name),bytes)
        end
        verify_source_identity()==hashes || throw(ArgumentError("sources changed during bundle publication"))
        manifest=Dict("schema"=>SCHEMA,"evidence_mode"=>"synthetic_or_unqualified_research",
            "contract_sha256"=>T.CONTRACT_SHA256,"model_id"=>model.model_id,"source_sha256"=>model.source_sha256,
            "setting_index"=>model.setting_index,"delay_hours"=>model.delay_hours,"clock_contract"=>"completed_interval_end_h4_h6_L0_L1_v1",
            "source_hashes"=>Dict(hashes),
            "members"=>[Dict("name"=>name,"sha256"=>_hash(bytes),"bytes"=>length(bytes)) for (name,bytes) in payloads])
        completion_bytes=_toml_bytes(manifest);_write_new(manifestpath,completion_bytes)
        verify_source_identity()==hashes || throw(ArgumentError("sources changed at bundle completion"))
        (directory=destination,manifest_sha256=_hash(completion_bytes),model_id=model.model_id)
    catch
        _quarantine_completion(manifestpath,completion_bytes)
        rethrow()
    end
end

"""Load only a caller-pinned completed research bundle. Digest checks bind all
members; exact loaded source identities and typed schema are independently
checked. Resume additionally checks its actual consumed input prefix.
"""
function load_bundle(directory;expected_manifest_sha256,expected_sources)
    hashes=verify_source_identity()
    _require(T._valid_hash(expected_manifest_sha256) && Dict(expected_sources)==Dict(hashes),"expected source/manifest pin differs")
    path=abspath(directory)
    _require(isdir(path) && !islink(path) && Set(readdir(path))==Set(["model.toml","checkpoint.toml","provenance.toml","manifest.toml"]),
        "bundle directory members differ")
    for name in readdir(path)
        member=joinpath(path,name)
        _require(isfile(member) && !islink(member) && filesize(member)<=MAX_MEMBER_BYTES,"bundle member is not a bounded ordinary file")
    end
    manifestbytes=read(joinpath(path,"manifest.toml"))
    _require(_hash(manifestbytes)==expected_manifest_sha256,"manifest digest differs")
    manifest=TOML.parse(String(copy(manifestbytes)))
    fields=Set(["schema","evidence_mode","contract_sha256","model_id","source_sha256","setting_index","delay_hours","clock_contract","source_hashes","members"])
    _require(Set(keys(manifest))==fields && manifest["schema"]==SCHEMA &&
        manifest["evidence_mode"]=="synthetic_or_unqualified_research" && manifest["contract_sha256"]==T.CONTRACT_SHA256 &&
        manifest["clock_contract"]=="completed_interval_end_h4_h6_L0_L1_v1" && manifest["source_hashes"]==Dict(hashes),"manifest schema/source/clock differs")
    members=manifest["members"]
    _require(length(members)==3 && [m["name"] for m in members]==["model.toml","checkpoint.toml","provenance.toml"],"manifest member allowlist differs")
    decoded=Dict{String,Any}()
    for item in members
        _require(Set(keys(item))==Set(["name","sha256","bytes"]) && T._valid_hash(item["sha256"]),"invalid member receipt")
        bytes=read(joinpath(path,item["name"]))
        _require(length(bytes)==item["bytes"] && _hash(bytes)==item["sha256"],"member bytes/digest differ")
        document=TOML.parse(String(copy(bytes)))
        _require(Set(keys(document))==Set(["value"]),"unknown member fields")
        decoded[item["name"]]=_decode(document["value"])
    end
    model=_validate_model(decoded["model.toml"])
    _require((manifest["model_id"],manifest["source_sha256"],manifest["setting_index"],manifest["delay_hours"])==
        (model.model_id,model.source_sha256,model.setting_index,model.delay_hours),"manifest/model identity differs")
    checkpoint=_restore_checkpoint(decoded["checkpoint.toml"],model)
    _require(decoded["provenance.toml"] isa NamedTuple,"invalid provenance schema")
    verify_source_identity()==hashes || throw(ArgumentError("sources changed during bundle load"))
    (model=model,checkpoint=checkpoint,provenance=decoded["provenance.toml"],manifest_sha256=expected_manifest_sha256,source_hashes=hashes)
end

end # module
