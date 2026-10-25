"Bounded canonical data-only JSON. Preflight precedes the JSON backend."
const _MAX_VALUES=600_000
const _MAX_CONTAINERS=4096
const _MAX_DEPTH=12
mutable struct _JSONBudget
    values::Int
    containers::Int
end
function _json_value!(budget,depth,container=false)
    _need(depth<=_MAX_DEPTH,"JSON nesting exceeds limit")
    budget.values+=1;_need(budget.values<=_MAX_VALUES,"JSON value budget exceeded")
    if container;budget.containers+=1;_need(budget.containers<=_MAX_CONTAINERS,"JSON container budget exceeded");end
end
function _json_write(io,value,budget,depth)
    container=value isa AbstractDict||value isa AbstractVector||value isa Tuple
    _json_value!(budget,depth,container)
    if value isa Bool
        write(io,value ? "true" : "false")
    elseif value isa Integer
        _need(typemin(Int64)<=value<=typemax(Int64),"JSON integer bound")
        print(io,value)
    elseif value isa AbstractString
        _need(ncodeunits(value)<=4096&&all(c->0x20<=c<=0x7e&&c!=0x22&&c!=0x5c,codeunits(value)),"plain bounded ASCII string required")
        write(io,'"');write(io,value);write(io,'"')
    elseif value isa AbstractDict
        _need(length(value)<=32&&all(k->k isa String&&ncodeunits(k)<=128,keys(value)),"bounded plain JSON keys required")
        write(io,'{')
        for (i,key) in enumerate(sort!(collect(keys(value))))
            i>1&&write(io,',');_json_write(io,key,budget,depth+1);write(io,':');_json_write(io,value[key],budget,depth+1)
        end
        write(io,'}')
    elseif value isa AbstractVector||value isa Tuple
        _need(length(value)<=MAX_ROWS,"JSON array bound")
        write(io,'[')
        for (i,item) in enumerate(value);i>1&&write(io,',');_json_write(io,item,budget,depth+1);end
        write(io,']')
    else
        throw(ArgumentError("unsupported JSON value type"))
    end
    _need(position(io)<MAX_BYTES,"canonical JSON byte bound")
end
function _canonical(value)
    io=IOBuffer();_json_write(io,value,_JSONBudget(0,0),1);write(io,'\n');take!(io)
end
mutable struct _JSONCursor
    bytes::Vector{UInt8}
    at::Int
    budget::_JSONBudget
end
function _space!(cursor)
    bytes=cursor.bytes
    while cursor.at<=length(bytes)&&bytes[cursor.at] in (0x20,0x09,0x0a,0x0d);cursor.at+=1;end
end
function _consume!(cursor,byte)
    _need(cursor.at<=length(cursor.bytes)&&cursor.bytes[cursor.at]==byte,"JSON delimiter differs");cursor.at+=1
end
function _plain_string!(cursor;key=false)
    _consume!(cursor,0x22);start=cursor.at
    while cursor.at<=length(cursor.bytes)&&cursor.bytes[cursor.at]!=0x22
        byte=cursor.bytes[cursor.at]
        _need(0x20<=byte<=0x7e&&byte!=0x5c,"JSON escaped/non-ASCII string rejected")
        cursor.at+=1;_need(cursor.at-start<=(key ? 128 : 4096),"JSON string bound")
    end
    stop=cursor.at-1;_consume!(cursor,0x22)
    key ? String(cursor.bytes[start:stop]) : nothing
end
function _preflight_value!(cursor,depth)
    _space!(cursor);_need(cursor.at<=length(cursor.bytes),"truncated JSON value")
    byte=cursor.bytes[cursor.at];container=byte in (0x7b,0x5b)
    _json_value!(cursor.budget,depth,container)
    if byte==0x7b
        cursor.at+=1;_space!(cursor);keys=Set{String}()
        if cursor.at<=length(cursor.bytes)&&cursor.bytes[cursor.at]==0x7d;cursor.at+=1;return;end
        while true
            key=_plain_string!(cursor;key=true)
            _need(length(keys)<32&&!(key in keys),"duplicate or excess JSON keys");push!(keys,key)
            _space!(cursor);_consume!(cursor,0x3a);_preflight_value!(cursor,depth+1);_space!(cursor)
            _need(cursor.at<=length(cursor.bytes),"truncated JSON object")
            if cursor.bytes[cursor.at]==0x7d;cursor.at+=1;break;end
            _consume!(cursor,0x2c);_space!(cursor)
        end
    elseif byte==0x5b
        cursor.at+=1;_space!(cursor);count=0
        if cursor.at<=length(cursor.bytes)&&cursor.bytes[cursor.at]==0x5d;cursor.at+=1;return;end
        while true
            count+=1;_need(count<=MAX_ROWS,"JSON array exceeds limit")
            _preflight_value!(cursor,depth+1);_space!(cursor)
            _need(cursor.at<=length(cursor.bytes),"truncated JSON array")
            if cursor.bytes[cursor.at]==0x5d;cursor.at+=1;break;end
            _consume!(cursor,0x2c)
        end
    elseif byte==0x22
        _plain_string!(cursor)
    elseif byte==0x74||byte==0x66
        literal=byte==0x74 ? codeunits("true") : codeunits("false")
        for expected in literal;_consume!(cursor,expected);end
    else
        start=cursor.at;byte==0x2d&&(cursor.at+=1)
        _need(cursor.at<=length(cursor.bytes)&&0x30<=cursor.bytes[cursor.at]<=0x39,"integer JSON token required")
        if cursor.bytes[cursor.at]==0x30
            cursor.at+=1
        else
            while cursor.at<=length(cursor.bytes)&&0x30<=cursor.bytes[cursor.at]<=0x39
                cursor.at+=1;_need(cursor.at-start<=20,"JSON integer token bound")
            end
        end
        _need(cursor.at>length(cursor.bytes)||!(cursor.bytes[cursor.at] in (0x2e,0x65,0x45)),"noninteger JSON token rejected before backend")
        value=tryparse(Int64,String(cursor.bytes[start:cursor.at-1]));_need(value!==nothing,"JSON integer range")
    end
end
function _preflight(bytes::Vector{UInt8})
    _need(1<=length(bytes)<=MAX_BYTES,"bounded JSON bytes required")
    cursor=_JSONCursor(bytes,1,_JSONBudget(0,0));_preflight_value!(cursor,1);_space!(cursor)
    _need(cursor.at==length(bytes)+1,"trailing JSON content");nothing
end
function _decode_bytes(bytes::Vector{UInt8};backend=JSON3.read)
    _preflight(bytes)
    value=try backend(String(copy(bytes)),Any) catch error
        error isa InterruptException&&rethrow();throw(ArgumentError("JSON backend rejected data"))
    end
    _need(_canonical(value)==bytes,"noncanonical JSON encoding")
    value
end
function _ordinary(path;directory=false)
    absolute=abspath(path);current=absolute
    while true
        _need(!islink(current),"symlink artifact path rejected")
        if Sys.iswindows()
            attributes=ccall((:GetFileAttributesW,"kernel32"),stdcall,UInt32,(Cwstring,),current)
            _need(attributes!=typemax(UInt32)&&(attributes&UInt32(0x400))==0,"reparse or unavailable artifact path rejected")
        end
        parent=dirname(current);parent==current&&break;current=parent
    end
    _need(directory ? isdir(absolute) : isfile(absolute),"ordinary artifact path required")
    absolute
end
function _read_json(path,expected_sha256)
    _need(_hex(expected_sha256),"external expected content SHA required")
    absolute=_ordinary(path);size=filesize(absolute);_need(1<=size<=MAX_BYTES,"artifact size bound before read")
    bytes=open(absolute,"r") do io
        value=read(io,Int(size));_need(length(value)==size&&eof(io),"artifact changed while reading");value
    end
    _need(filesize(absolute)==size&&bytes2hex(sha256(bytes))==expected_sha256,"artifact content pin differs")
    _decode_bytes(bytes)
end
function _write_json(path,value)
    absolute=abspath(path);_ordinary(dirname(absolute);directory=true)
    _need(!ispath(absolute)&&!islink(absolute),"new artifact destination required")
    bytes=_canonical(value)
    io=Base.Filesystem.open(absolute,Base.Filesystem.JL_O_WRONLY|Base.Filesystem.JL_O_CREAT|Base.Filesystem.JL_O_EXCL,0o600)
    try write(io,bytes) finally close(io) end
    (;path=absolute,sha256=bytes2hex(sha256(bytes)),bytes=length(bytes))
end
function _exact_keys(value,names)
    _need(value isa AbstractDict&&all(k->k isa String,keys(value))&&Set(keys(value))==Set(names),"closed data field set differs")
end
const _ENCODER_KEYS=("schema","feature_count","feature_schema_sha256","feature_contract_sha256","training_rows",
    "finite_counts","imputation_f64_bits","center_f64_bits","scale_f64_bits","training_design_sha256")
function _encoder_object(encoder::Encoder342)
    _check_encoder(encoder)
    Dict{String,Any}("schema"=>"matched342_encoder_v1","feature_count"=>WIDTH,
        "feature_schema_sha256"=>FEATURE_SCHEMA_SHA256,"feature_contract_sha256"=>FEATURE_CONTRACT_SHA256,
        "training_rows"=>collect(encoder.training_rows),"finite_counts"=>collect(encoder.finite_counts),
        "imputation_f64_bits"=>_bits.(collect(encoder.imputation)),"center_f64_bits"=>_bits.(collect(encoder.center)),
        "scale_f64_bits"=>_bits.(collect(encoder.scale)),"training_design_sha256"=>encoder.training_design_sha256)
end
"SHA256 of the exact canonical ten-field encoder member, including ordered training-row identity."
preprocessing_sha256(encoder::Encoder342)=bytes2hex(sha256(_canonical(_encoder_object(encoder))))
"Exclusively create a bounded canonical data-only encoder member; return path, byte SHA and size."
function save_encoder(path,encoder::Encoder342)
    source_identity();result=_write_json(path,_encoder_object(encoder));source_identity();result
end
"Load immutable encoder data under external content/design SHA pins, closed schema and pre-parser bounds."
function load_encoder(path;expected_sha256,expected_training_design_sha256)
    source_identity();_need(_hex(expected_training_design_sha256),"external expected training design SHA required")
    object=_read_json(path,expected_sha256);_exact_keys(object,_ENCODER_KEYS)
    _need(object["schema"]=="matched342_encoder_v1"&&_int(object["feature_count"],WIDTH,WIDTH)==WIDTH&&
        object["feature_schema_sha256"]==FEATURE_SCHEMA_SHA256&&object["feature_contract_sha256"]==FEATURE_CONTRACT_SHA256&&
        object["training_design_sha256"]==expected_training_design_sha256,"encoder context differs")
    rows=object["training_rows"];counts=object["finite_counts"]
    _need(rows isa AbstractVector&&1<=length(rows)<=MAX_ROWS&&counts isa AbstractVector&&length(counts)==BASE_WIDTH,"encoder arrays differ")
    checked_rows=Int[_int(i,1,MAX_ROWS) for i in rows]
    _need(issorted(checked_rows)&&allunique(checked_rows),"encoder ordered rows differ")
    parameters=Tuple{Vararg{Float64}}[]
    for (key,n) in (("imputation_f64_bits",BASE_WIDTH),("center_f64_bits",WIDTH),("scale_f64_bits",WIDTH))
        values=object[key];_need(values isa AbstractVector&&length(values)==n,"encoder parameter width differs")
        push!(parameters,Tuple(_float.(values)))
    end
    encoder=Encoder342(_training_rows(checked_rows),Tuple(_int(n,0,length(rows)) for n in counts),parameters...,
        String(expected_training_design_sha256))
    _check_encoder(encoder);_need(preprocessing_sha256(encoder)==expected_sha256,"decoded encoder differs");source_identity();encoder
end
