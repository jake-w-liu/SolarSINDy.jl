struct _Verified end
const _VERIFIED = _Verified()

struct _Bundle
    directory::String
    manifest_sha256::String
    commit_sha256::String
    kind::String
    protocol::RefitProtocol
    context::String
    members::Tuple{Vararg{Pair{String,String}}}
    rows::Int64
    _Bundle(::_Verified,args...) = new(args...)
end

struct AuditedSources
    _bundle::_Bundle
    source_bound_provenance_verified::Bool
    AuditedSources(::_Verified,b::_Bundle) = new(b,false)
end
struct ScheduleHandle
    _bundle::_Bundle
    source_bound_provenance_verified::Bool
    ScheduleHandle(::_Verified,b::_Bundle) = new(b,false)
end
struct FitInput
    _bundle::_Bundle
    source_bound_provenance_verified::Bool
    FitInput(::_Verified,b::_Bundle) = new(b,false)
end
struct PointHandle
    _bundle::_Bundle
    source_bound_provenance_verified::Bool
    PointHandle(::_Verified,b::_Bundle) = new(b,false)
end
struct CalHandle
    _bundle::_Bundle
    source_bound_provenance_verified::Bool
    CalHandle(::_Verified,b::_Bundle) = new(b,false)
end

struct _Draft
    kind::String
    protocol::RefitProtocol
    scientific_context::String
    metadata::String
    chunks::Tuple{Vararg{String}}
    members::Tuple{Vararg{Pair{String,String}}}
    rows::Int64
    _Draft(::_Verified,args...) = new(args...)
end

_bundle(value::Union{AuditedSources,ScheduleHandle,FitInput,PointHandle,CalHandle}) = value._bundle
_pin(value) = _bundle(value).manifest_sha256
_context(value) = decode_canonical(_bundle(value).context;limit=MAX_METADATA_BYTES)
function _member(bundle::_Bundle,name::String)
    for pair in bundle.members
        first(pair)==name && return Vector{UInt8}(codeunits(last(pair)))
    end
    fail("required verified member is absent")
end
_member(value,name::String) = _member(_bundle(value),name)
_member_json(value,name::String) = decode_canonical(_member(value,name))

function _pack_rows(rows)
    result=String[]
    for first in 1:CHUNK_ROWS:length(rows)
        last=min(length(rows),first+CHUNK_ROWS-1)
        push!(result,_jsontext(rows[first:last]))
    end
    Tuple(result)
end
function _unpack_rows(chunks)
    rows=Any[]
    for (index,chunk) in enumerate(chunks)
        part=_array(decode_canonical(chunk);nonempty=true)
        require(index==length(chunks) || length(part)==CHUNK_ROWS,"nonfinal row chunk has wrong length")
        require(length(rows)+length(part)<=MAX_ROWS,"row count bound exceeded")
        append!(rows,part)
    end
    rows
end
_row_name(index,prefix="rows") = prefix*"-"*lpad(string(index),6,'0')*".json"

function _frame_digest(domain::String,chunks)
    state=SHA.SHA2_256_CTX()
    SHA.update!(state,Vector{UInt8}(codeunits(domain)))
    SHA.update!(state,UInt8[0])
    header=IOBuffer();write(header,hton(UInt64(length(chunks))));SHA.update!(state,take!(header))
    for chunk in chunks
        bytes=chunk isa AbstractString ? Vector{UInt8}(codeunits(chunk)) : Vector{UInt8}(chunk)
        header=IOBuffer();write(header,hton(UInt64(length(bytes))));SHA.update!(state,take!(header));SHA.update!(state,bytes)
    end
    bytes2hex(SHA.digest!(state))
end
function _ordered_digest(ids)
    state=SHA.SHA2_256_CTX();SHA.update!(state,Vector{UInt8}(codeunits("DBDTORDER1")));SHA.update!(state,UInt8[0])
    header=IOBuffer();write(header,hton(UInt64(length(ids))));SHA.update!(state,take!(header))
    for id in ids
        bytes=Vector{UInt8}(codeunits(identifier(id)))
        header=IOBuffer();write(header,hton(UInt16(length(bytes))));SHA.update!(state,take!(header));SHA.update!(state,bytes)
    end
    bytes2hex(SHA.digest!(state))
end

function _row_id(protocol::RefitProtocol,issue::DateTime)
    s=_spec(protocol)
    value=Dict{String,Any}(k=>s[k] for k in ("station","layout","cohort_kind","policy"))
    value["protocol_sha256"]=protocol.sha256;value["issue"]=time_text(issue)
    digest(canonical_bytes(value))
end

function _data_rows(value,prefix="rows")
    b=_bundle(value)
    names=[first(pair) for pair in b.members if startswith(first(pair),prefix*"-")]
    sort!(names)
    require(names==[_row_name(i,prefix) for i in 1:length(names)],"row chunk catalog differs")
    chunks=Tuple(String(_member(b,name)) for name in names)
    _unpack_rows(chunks)
end
